from datetime import date
from decimal import Decimal

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, ProductAlias, Unit
from agrodash.infra.db import MIGRATIONS_LOCATION, Base, alembic_config, create_db_engine
from agrodash.infra.settings import get_settings
from agrodash.ingestion.models import (
    IngestionRun,
    IngestionStatus,
    IngestionTrigger,
    QuarantinedRow,
    RejectionReason,
    SourceLayout,
)
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType


@pytest.fixture
def catalog(session):
    product = Product(name="Arroz Selecto", category="Cereales", base_unit=BaseUnit.POUND)
    market = Market(name="Mercado Nuevo", kind=MarketKind.WHOLESALE)
    unit = Unit(code="saco_125_lb", label="Saco/125 lb", base_unit=BaseUnit.POUND, factor=125)
    session.add_all([product, market, unit])
    session.commit()
    return product, market, unit


def make_price(catalog, **overrides) -> Price:
    product, market, unit = catalog
    values = {
        "product": product,
        "market": market,
        "unit": unit,
        "observed_on": date(2017, 8, 1),
        "price_type": PriceType.WHOLESALE,
        "source": PriceSource.OFFICIAL,
        "status": PriceStatus.VALIDATED,
        "amount": Decimal("2350.00"),
        "price_per_base_unit": Decimal("18.8000"),
    }
    values.update(overrides)
    return Price(**values)


def test_migrations_match_models(engine):
    with engine.connect() as connection:
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)

    assert diff == []


def test_downgrade_removes_every_table(engine, database_url):
    command.downgrade(alembic_config(database_url), "base")

    assert inspect(engine).get_table_names() == ["alembic_version"]


def test_official_price_is_unique_per_product_market_week_and_type(session, catalog):
    session.add(make_price(catalog))
    session.commit()

    session.add(make_price(catalog, amount=Decimal("2400.00")))
    with pytest.raises(IntegrityError):
        session.commit()


def test_field_reports_for_the_same_week_can_coexist(session, catalog):
    field_report = {"source": PriceSource.FIELD_REPORT, "status": PriceStatus.PENDING}
    session.add_all([make_price(catalog, **field_report), make_price(catalog, **field_report)])
    session.commit()

    assert session.query(Price).count() == 2


def test_wholesale_and_retail_prices_coexist_for_the_same_week(session, catalog):
    session.add_all([make_price(catalog), make_price(catalog, price_type=PriceType.RETAIL)])
    session.commit()

    assert session.query(Price).count() == 2


def test_price_amounts_keep_decimal_precision(session, catalog):
    session.add(make_price(catalog))
    session.commit()
    session.expire_all()

    price = session.query(Price).one()
    assert price.amount == Decimal("2350.00")
    assert price.price_per_base_unit == Decimal("18.8000")
    assert price.observed_on == date(2017, 8, 1)


def test_price_requires_existing_product(session, catalog):
    session.add(make_price(catalog))
    session.commit()

    with pytest.raises(IntegrityError):
        session.execute(text("UPDATE prices SET product_id = 999"))
        session.commit()


def test_rejects_unknown_price_status(session, catalog):
    session.add(make_price(catalog))
    session.commit()

    with pytest.raises(IntegrityError):
        session.execute(text("UPDATE prices SET status = 'approved-by-typo'"))
        session.commit()


def test_rejects_non_positive_amounts(session, catalog):
    session.add(make_price(catalog, amount=Decimal("0")))
    with pytest.raises(IntegrityError):
        session.commit()


def test_product_alias_maps_each_raw_name_once(session, catalog):
    product, _, _ = catalog
    session.add(ProductAlias(raw_name="arroz selecto/ primera", product=product))
    session.commit()
    assert session.query(ProductAlias).one().reviewed is False

    session.add(ProductAlias(raw_name="arroz selecto/ primera", product=product))
    with pytest.raises(IntegrityError):
        session.commit()


def test_alembic_config_keeps_url_encoded_passwords():
    url = "postgresql+psycopg://user:p%40ss@db:5432/agrodash"

    assert alembic_config(url).get_main_option("sqlalchemy.url") == url


def test_cli_migrations_read_the_database_url_from_settings(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("AGRODASH_DATABASE_URL", url)
    get_settings.cache_clear()
    config = Config()
    config.set_main_option("script_location", MIGRATIONS_LOCATION)

    try:
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()

    assert "prices" in inspect(create_db_engine(url)).get_table_names()


def make_run(**overrides) -> IngestionRun:
    values = {
        "file_name": "mayorista.csv",
        "file_sha256": "a" * 64,
        "layout": SourceLayout.WHOLESALE,
        "trigger": IngestionTrigger.FILE,
        "status": IngestionStatus.SUCCEEDED,
    }
    values.update(overrides)
    return IngestionRun(**values)


def test_a_file_is_ingested_successfully_only_once(session):
    session.add(make_run())
    session.commit()

    session.add(make_run())
    with pytest.raises(IntegrityError):
        session.commit()


def test_failed_runs_of_the_same_file_can_be_retried(session):
    session.add_all([make_run(status=IngestionStatus.FAILED), make_run()])
    session.commit()

    assert session.query(IngestionRun).count() == 2


def test_quarantined_rows_and_prices_point_to_their_run(session, catalog):
    run = make_run()
    rejected = QuarantinedRow(
        run=run, row_number=7, reason=RejectionReason.UNKNOWN_UNIT, raw_values='["x"]'
    )
    session.add_all([rejected, make_price(catalog, ingestion_run=run)])
    session.commit()

    assert session.query(Price).one().ingestion_run_id == run.id
    assert session.query(Price).one().is_outlier is False
    assert run.quarantined_rows == [rejected]
