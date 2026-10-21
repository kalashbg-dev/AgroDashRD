from datetime import date
from decimal import Decimal

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, ProductAlias, Unit
from agrodash.infra.db import Base, alembic_config
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
