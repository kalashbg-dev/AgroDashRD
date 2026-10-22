import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from agrodash import cli
from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, ProductAlias, Unit
from agrodash.ingestion import quality
from agrodash.ingestion.models import IngestionTrigger
from agrodash.ingestion.quality import build_report
from agrodash.ingestion.service import ingest_file
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType

FIXTURES = Path(__file__).parent / "fixtures"


def add_series(session, name, weeks_by_month, price_type=PriceType.RETAIL):
    """Prices on the given week numbers (1-5) of each (year, month)."""
    product = Product(name=name, category="Frutas", base_unit=BaseUnit.POUND)
    market = session.query(Market).first() or Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
    unit = session.query(Unit).first() or Unit(
        code="1-lb", label="1 lb", base_unit=BaseUnit.POUND, factor=Decimal(1)
    )
    for (year, month), weeks in weeks_by_month.items():
        for week in weeks:
            session.add(
                Price(
                    product=product,
                    market=market,
                    unit=unit,
                    observed_on=date(year, month, min(1 + 7 * (week - 1), 28)),
                    price_type=price_type,
                    source=PriceSource.OFFICIAL,
                    status=PriceStatus.VALIDATED,
                    amount=Decimal("10"),
                    price_per_base_unit=Decimal("10"),
                )
            )
    session.commit()
    return product


def test_report_counts_rejections_by_reason_and_unreviewed_aliases(session):
    ingest_file(session, (FIXTURES / "mayorista.csv").read_bytes(), "m.csv", IngestionTrigger.FILE)
    session.query(ProductAlias).filter_by(raw_name="arroz selecto primera").update(
        {"reviewed": True}
    )
    session.commit()

    report = build_report(session)

    assert report.rejections == {
        "duplicate_price": 1,
        "invalid_date": 1,
        "invalid_price": 1,
        "unexpected_layout": 1,
        "unit_mismatch": 1,
        "unknown_unit": 1,
    }
    assert report.official_prices == {PriceType.WHOLESALE: 5}
    assert (report.products, report.aliases) == (4, 4)
    assert sorted(report.unreviewed_aliases) == [
        ("cilantro long standing primera", "Cilantro (Long Standing), primera"),
        ("leche liquida", "Leche (Líquida)"),
        ("platano barahonero primera", "Plátano (Barahonero), primera"),
    ]


def test_series_coverage_counts_weeks_one_to_four_between_first_and_last_price(session):
    add_series(session, "Mango", {(2026, 7): [1, 2, 3, 4, 5], (2026, 8): [1, 3], (2026, 9): [2]})

    series = build_report(session).series[0]

    assert (series.product, series.market, series.price_type, series.base_unit) == (
        "Mango",
        "Mercado Nuevo",
        "retail",
        "lb",
    )
    assert (series.first_on, series.last_on, series.weeks) == (
        date(2026, 7, 1),
        date(2026, 9, 8),
        8,
    )
    # 7 of the 12 slots for weeks 1-4 in July-September have a price.
    assert series.coverage == pytest.approx(7 / 12)


def test_only_long_dense_and_current_series_are_ready_to_forecast(session, monkeypatch):
    monkeypatch.setattr(quality, "MIN_WEEKS", 8)
    full = {(2026, month): [1, 2, 3, 4] for month in (7, 8, 9)}
    add_series(session, "Ready", full)
    add_series(session, "Short", {(2026, 9): [1, 2, 3, 4]})
    add_series(session, "Sparse", {(2026, month): [1, 2] for month in (5, 6, 7, 8, 9)})
    add_series(session, "Stale", {(2025, month): [1, 2, 3, 4] for month in (1, 2, 3, 4)})

    report = build_report(session)

    ready = {series.product: series.ready for series in report.series}
    assert ready == {"Ready": True, "Short": False, "Sparse": False, "Stale": False}
    assert report.latest_observation == date(2026, 9, 22)


def test_quality_report_command_prints_the_summary_and_exports_series(
    cli_database, tmp_path, capsys
):
    cli.main(["ingest", str(FIXTURES / "mayorista.csv"), str(FIXTURES / "minorista.csv")])
    capsys.readouterr()
    export = tmp_path / "series.csv"

    assert cli.main(["quality-report", "--series-csv", str(export)]) == 0

    out = capsys.readouterr().out
    assert "Official prices: 18 (5 wholesale, 13 retail)" in out
    assert "Quarantine: 7 rejections" in out
    assert "  unit_mismatch: 1" in out
    assert "Catalog: 5 products, 5 name aliases (5 unreviewed), " in out
    assert "Series: 17, ready to forecast: 0" in out
    rows = list(csv.DictReader(export.open(encoding="utf-8")))
    assert len(rows) == 17
    assert set(rows[0]) == {
        "product_id",
        "market_id",
        "product",
        "market",
        "price_type",
        "base_unit",
        "first_on",
        "last_on",
        "weeks",
        "coverage",
        "ready",
    }


def test_renamed_products_are_suggested_for_review(session):
    add_series(
        session,
        "Cilantro/ (Long Standing)/ grande/ primera",
        {(2018, 1): [1, 2, 3, 4], (2018, 2): [1, 2]},
    )
    add_series(session, "Cilantro (Long Standing)/ primera", {(2018, 2): [3, 4], (2018, 3): [1, 2]})
    add_series(session, "Cilantro (Long Standing)/ segunda", {(2019, 1): [1]})
    add_series(session, "Mango (Keitt)", {(2018, 2): [3, 4]})

    report = build_report(session)

    assert report.rename_candidates == [
        ("Cilantro/ (Long Standing)/ grande/ primera", "Cilantro (Long Standing)/ primera", 7)
    ]


def test_rejections_count_only_the_latest_load_of_each_report_type(session):
    for name in ("minorista.csv", "minorista.xls", "mayorista.csv"):
        ingest_file(session, (FIXTURES / name).read_bytes(), name, IngestionTrigger.FILE)

    rejections = build_report(session).rejections

    assert rejections["invalid_price"] == 2
    assert sum(rejections.values()) == 7


def test_staleness_is_measured_within_each_price_type(session, monkeypatch):
    monkeypatch.setattr(quality, "MIN_WEEKS", 8)
    add_series(session, "Retail", {(2026, m): [1, 2, 3, 4] for m in (7, 8, 9)})
    add_series(
        session, "Wholesale", {(2026, m): [1, 2, 3, 4] for m in (3, 4, 5)}, PriceType.WHOLESALE
    )

    ready = {series.product: series.ready for series in build_report(session).series}

    assert ready == {"Retail": True, "Wholesale": True}


def test_quality_report_lists_unreviewed_names_and_reports_bad_export_paths(
    cli_database, tmp_path, capsys
):
    cli.main(["ingest", str(FIXTURES / "mayorista.csv")])
    capsys.readouterr()

    exit_code = cli.main(["quality-report", "--series-csv", str(tmp_path / "missing" / "s.csv")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Unreviewed names (4):" in captured.out
    assert "  cilantro long standing primera -> Cilantro (Long Standing), primera" in captured.out
    assert "error: cannot write" in captured.err


def test_a_series_that_changes_unit_is_reported_once_per_unit(session):
    weeks = ["Primera", "Segunda", "Tercera", "Cuarta"]
    rows = [f"Pepino;Legumbres-Hortalizas;Mayorista;Saco/90 lb;600;{w};Enero;2024" for w in weeks]
    rows += [f"Pepino;Legumbres-Hortalizas;Mayorista;Ciento;900;{w};Febrero;2024" for w in weeks]
    ingest_file(session, "\r\n".join(rows).encode(), "m.csv", IngestionTrigger.FILE)

    series = [(s.product, s.base_unit, s.weeks) for s in build_report(session).series]

    assert series == [("Pepino", "lb", 4), ("Pepino", "unit", 4)]
