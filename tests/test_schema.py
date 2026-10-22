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
from agrodash.forecasting.models import Forecast, ForecastEvaluation, ForecastRun, ForecastStatus
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


def make_forecast_run(**overrides) -> ForecastRun:
    values = {"model_version": "f4.1", "horizon": 4, "status": ForecastStatus.SUCCEEDED}
    values.update(overrides)
    return ForecastRun(**values)


def test_forecasts_and_evaluations_belong_to_their_run(session, catalog):
    product, market, _ = catalog
    run = make_forecast_run()
    series = {
        "product": product,
        "market": market,
        "price_type": PriceType.RETAIL,
        "base_unit": BaseUnit.POUND,
    }
    session.add_all(
        [
            ForecastEvaluation(run=run, model="AutoETS", mase=0.8, selected=True, **series),
            Forecast(
                run=run,
                model="AutoETS",
                step=1,
                target_on=date(2026, 10, 1),
                value=Decimal("30.5"),
                lo_80=Decimal("28"),
                hi_80=Decimal("33"),
                lo_95=Decimal("27"),
                hi_95=Decimal("34"),
                **series,
            ),
        ]
    )
    session.commit()

    assert [forecast.value for forecast in run.forecasts] == [Decimal("30.5000")]
    assert run.evaluations[0].selected is True


def test_a_series_has_one_forecast_per_step_in_a_run(session, catalog):
    product, market, _ = catalog
    run = make_forecast_run()
    forecast = {
        "run": run,
        "product": product,
        "market": market,
        "price_type": PriceType.RETAIL,
        "base_unit": BaseUnit.POUND,
        "model": "AutoETS",
        "step": 1,
        "target_on": date(2026, 10, 1),
        "value": Decimal("30"),
    }
    session.add_all([Forecast(**forecast), Forecast(**forecast)])

    with pytest.raises(IntegrityError):
        session.commit()


def test_dropping_market_location_works_on_a_database_with_prices(tmp_path):
    url = f"sqlite:///{tmp_path / 'populated.db'}"
    command.upgrade(alembic_config(url), "2c4cc5bb4efe")
    engine = create_db_engine(url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO markets (id, name, kind, is_active) VALUES (1, 'Mercado Nuevo', 'retail', 1);"
            )
        )
        connection.execute(
            text(
                "INSERT INTO products (id, name, category, base_unit, is_active)"
                " VALUES (1, 'Papa', 'Raices', 'lb', 1);"
            )
        )
        connection.execute(
            text(
                "INSERT INTO units (id, code, label, base_unit, factor) VALUES (1, '1-lb', '1 lb', 'lb', 1);"
            )
        )
        connection.execute(
            text(
                "INSERT INTO prices (product_id, market_id, unit_id, observed_on, price_type, source,"
                " status, amount, price_per_base_unit, is_outlier)"
                " VALUES (1, 1, 1, '2026-09-01', 'retail', 'official', 'validated', 10, 10, 0);"
            )
        )

    command.upgrade(alembic_config(url), "head")

    columns = {column["name"] for column in inspect(engine).get_columns("markets")}
    assert "location" not in columns
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM prices")).scalar() == 1
    engine.dispose()
