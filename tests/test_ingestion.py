import io
import json
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

import openpyxl
import pytest

from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product
from agrodash.ingestion.models import (
    IngestionRun,
    IngestionStatus,
    IngestionTrigger,
    RejectionReason,
    SourceLayout,
)
from agrodash.ingestion.service import MAX_FILE_BYTES, IngestionError, ingest_file
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType

FIXTURES = Path(__file__).parent / "fixtures"


def load(session, name: str, content: bytes | None = None) -> IngestionRun:
    content = (FIXTURES / name).read_bytes() if content is None else content
    return ingest_file(session, content, name, IngestionTrigger.FILE)


def reasons(run: IngestionRun) -> Counter:
    return Counter(row.reason for row in run.quarantined_rows)


def test_wholesale_csv_loads_prices_and_quarantines_bad_rows(session):
    run = load(session, "mayorista.csv")

    assert (run.status, run.layout) == (IngestionStatus.SUCCEEDED, SourceLayout.WHOLESALE)
    assert (run.rows_read, run.prices_loaded, run.rows_rejected, run.values_missing) == (
        11,
        4,
        6,
        1,
    )
    assert reasons(run) == {
        RejectionReason.UNIT_MISMATCH: 1,
        RejectionReason.UNKNOWN_UNIT: 1,
        RejectionReason.INVALID_DATE: 1,
        RejectionReason.INVALID_PRICE: 1,
        RejectionReason.DUPLICATE_PRICE: 1,
        RejectionReason.UNEXPECTED_LAYOUT: 1,
    }
    rice = session.query(Product).filter_by(name="Arroz Selecto/ primera").one()
    first_week = session.query(Price).filter_by(product=rice, observed_on=date(2017, 8, 1)).one()
    assert (first_week.amount, first_week.price_per_base_unit) == (
        Decimal("2350.00"),
        Decimal("18.8000"),
    )
    assert first_week.market.name == "Mercado Nuevo"
    assert (first_week.price_type, first_week.source, first_week.status) == (
        PriceType.WHOLESALE,
        PriceSource.OFFICIAL,
        PriceStatus.VALIDATED,
    )
    assert first_week.ingestion_run_id == run.id
    assert session.query(Price).filter_by(product=rice).count() == 2


def test_quarantined_rows_keep_their_cells_and_line_number(session):
    run = load(session, "mayorista.csv")

    unknown = next(row for row in run.quarantined_rows if row.reason == "unknown_unit")
    assert unknown.row_number == 6
    assert json.loads(unknown.raw_values)[:4] == [
        "Ajo",
        "Legumbres-Hortalizas",
        "Mayorista",
        "Saco",
    ]
    assert "Saco" in unknown.detail


def test_retail_sheet_loads_one_price_per_market(session):
    run = load(session, "minorista.csv")

    assert (run.layout, run.prices_loaded, run.rows_rejected, run.values_missing) == (
        SourceLayout.RETAIL,
        13,
        1,
        14,
    )
    assert reasons(run) == {RejectionReason.INVALID_PRICE: 1}
    markets = {market.name: market.kind for market in session.query(Market)}
    assert len(markets) == 7
    assert markets["Supermercado"] is MarketKind.SUPERMARKET
    assert markets["Mercado Conaprope"] is MarketKind.RETAIL
    supermarket = session.query(Price).join(Market).filter(Market.name == "Supermercado").all()
    assert {(price.observed_on, price.amount) for price in supermarket} == {
        (date(2017, 8, 1), Decimal("24.50")),
        (date(2026, 9, 29), Decimal("76.48")),
    }


def test_wholesale_and_retail_names_share_the_product_catalog(session):
    load(session, "mayorista.csv")
    load(session, "minorista.csv")

    rice = session.query(Product).filter_by(name="Arroz Selecto/ primera").one()
    types = Counter(price.price_type for price in session.query(Price).filter_by(product=rice))
    assert types == {PriceType.WHOLESALE: 2, PriceType.RETAIL: 4}


def test_each_price_type_keeps_its_own_base_unit(session):
    load(session, "mayorista.csv")
    load(session, "minorista.csv")

    plantain = session.query(Product).filter_by(name="Plátano (Barahonero), primera").one()
    prices = session.query(Price).filter_by(product=plantain)
    bases = {price.price_type: price.unit.base_unit for price in prices}
    assert bases == {PriceType.WHOLESALE: BaseUnit.UNIT, PriceType.RETAIL: BaseUnit.POUND}


def test_loading_the_same_file_twice_changes_nothing(session):
    first = load(session, "mayorista.csv")
    second = load(session, "mayorista.csv")

    assert second.id == first.id
    assert session.query(IngestionRun).count() == 1
    assert session.query(Price).count() == 4


def test_the_same_data_in_another_format_only_adds_new_prices(session):
    load(session, "minorista.csv")

    run = load(session, "minorista.xls")

    assert (run.status, run.prices_loaded, run.rows_rejected) == (IngestionStatus.SUCCEEDED, 0, 1)
    assert session.query(Price).count() == 13


def test_xlsx_workbooks_are_read(session):
    workbook = openpyxl.Workbook()
    workbook.active.append(
        [
            "Arroz Selecto/ primera",
            "Cereal",
            "Mayorista",
            "Saco/125 lb",
            2350,
            "Primera",
            "Agosto",
            2017,
        ]
    )
    content = io.BytesIO()
    workbook.save(content)

    run = load(session, "mayorista.xlsx", content.getvalue())

    assert (run.layout, run.prices_loaded) == (SourceLayout.WHOLESALE, 1)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("precios.pdf", b"%PDF-1.4", "Unsupported file type '.pdf'"),
        ("precios.xlsx", b"not a workbook", "Could not read precios.xlsx"),
        ("vacio.csv", b"", "precios.csv has no rows"),
    ],
)
def test_unreadable_files_fail_with_a_recorded_run(session, name, content, message):
    message = message.replace("precios.csv", name)

    with pytest.raises(IngestionError, match=message):
        load(session, name, content)

    run = session.query(IngestionRun).one()
    assert (run.status, run.file_name, run.prices_loaded) == (IngestionStatus.FAILED, name, 0)
    assert message in run.error
    assert run.finished_at is not None
    assert session.query(Price).count() == 0


def test_a_failed_file_can_be_loaded_again_after_fixing_the_cause(session, monkeypatch):
    content = (FIXTURES / "mayorista.csv").read_bytes()
    monkeypatch.setattr("agrodash.ingestion.service.MAX_FILE_BYTES", 10)
    with pytest.raises(IngestionError, match="larger than"):
        load(session, "mayorista.csv", content)
    monkeypatch.setattr("agrodash.ingestion.service.MAX_FILE_BYTES", MAX_FILE_BYTES)

    run = load(session, "mayorista.csv", content)

    assert run.status is IngestionStatus.SUCCEEDED
    assert [r.status for r in session.query(IngestionRun).order_by(IngestionRun.id)] == [
        IngestionStatus.FAILED,
        IngestionStatus.SUCCEEDED,
    ]
