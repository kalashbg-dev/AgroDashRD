from datetime import date
from decimal import Decimal

import pytest

from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, Unit
from agrodash.forecasting.series import (
    SeriesKey,
    load_training_data,
    next_periods,
    period_date,
    period_of,
)
from agrodash.ingestion.normalize import parse_week_date
from agrodash.ingestion.quality import series_coverage
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType

WEEKS = ["Primera", "Segunda", "Tercera", "Cuarta", "Quinta"]


@pytest.fixture
def catalog(session):
    product = Product(name="Arroz Selecto", category="Cereales", base_unit=BaseUnit.POUND)
    other = Product(name="Papa", category="Raices-Tuberculos", base_unit=BaseUnit.POUND)
    market = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
    unit = Unit(code="1-lb", label="1 lb", base_unit=BaseUnit.POUND, factor=Decimal(1))
    session.add_all([product, other, market, unit])
    session.commit()
    return product, other, market, unit


def add_price(session, catalog, product, week, month, year, amount, **extra):
    _, _, market, unit = catalog
    values = {
        "product": product,
        "market": market,
        "unit": unit,
        "observed_on": parse_week_date(week, month, year),
        "price_type": PriceType.RETAIL,
        "source": PriceSource.OFFICIAL,
        "status": PriceStatus.VALIDATED,
        "amount": Decimal(amount),
        "price_per_base_unit": Decimal(amount),
    }
    values.update(extra)
    session.add(Price(**values))


def key(catalog, product) -> SeriesKey:
    return SeriesKey(product.id, catalog[2].id, PriceType.RETAIL, BaseUnit.POUND)


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ((2023, 2, 5), date(2023, 2, 28)),
        ((2024, 2, 5), date(2024, 2, 29)),
        ((2023, 8, 4), date(2023, 8, 22)),
    ],
)
def test_periods_and_stored_dates_convert_both_ways(period, expected):
    assert period_date(period) == expected
    assert period_of(expected) == period


def test_future_periods_follow_weeks_one_to_four_and_roll_into_the_next_month():
    # Fifth weeks follow no calendar rule, so forecasts never assume one.
    assert next_periods((2026, 11, 3), 4) == [
        (2026, 11, 4),
        (2026, 12, 1),
        (2026, 12, 2),
        (2026, 12, 3),
    ]
    assert next_periods((2026, 12, 5), 2) == [(2027, 1, 1), (2027, 1, 2)]


def test_series_are_indexed_by_report_period_with_gaps_interpolated(session, catalog):
    rice, potato, *_ = catalog
    # Rice skips "Tercera"; the potato series publishes it, so it is a report period.
    for week, amount in (("Primera", "10"), ("Segunda", "12"), ("Cuarta", "16"), ("Quinta", "18")):
        add_price(session, catalog, rice, week, "Agosto", 2023, amount)
    for week in WEEKS:
        add_price(session, catalog, potato, week, "Agosto", 2023, "20")
    session.commit()

    data = load_training_data(session, [key(catalog, rice)])

    assert data.calendar == [(2023, 8, n) for n in range(1, 6)]
    rows = data.frame.to_dict("records")
    uid = key(catalog, rice).uid
    assert [(row["unique_id"], row["ds"], row["y"]) for row in rows] == [
        (uid, 0, 10.0),
        (uid, 1, 12.0),
        (uid, 2, 14.0),
        (uid, 3, 16.0),
        (uid, 4, 18.0),
    ]
    assert data.imputed == {uid: 1}
    assert data.keys[uid] == key(catalog, rice)


def test_field_reports_and_other_units_stay_out_of_training(session, catalog):
    rice, *_ = catalog
    add_price(session, catalog, rice, "Primera", "Agosto", 2023, "10")
    add_price(
        session,
        catalog,
        rice,
        "Primera",
        "Agosto",
        2023,
        "99",
        source=PriceSource.FIELD_REPORT,
        status=PriceStatus.VALIDATED,
    )
    session.commit()

    data = load_training_data(session, [key(catalog, rice)])

    assert data.frame["y"].tolist() == [10.0]


def test_quality_series_carry_the_ids_forecasting_needs(session, catalog):
    rice, _, market, _ = catalog
    add_price(session, catalog, rice, "Primera", "Agosto", 2023, "10")
    session.commit()

    (series,) = series_coverage(session)

    assert (series.product_id, series.market_id) == (rice.id, market.id)
    assert SeriesKey.from_coverage(series) == key(catalog, rice)


def test_months_the_reports_skipped_still_count_their_four_weeks(session, catalog):
    rice, *_ = catalog
    # Like August 2019: only the first week was published, then September.
    add_price(session, catalog, rice, "Primera", "Agosto", 2019, "10")
    add_price(session, catalog, rice, "Primera", "Septiembre", 2019, "14")
    session.commit()

    data = load_training_data(session, [key(catalog, rice)])

    assert data.calendar[:5] == [
        (2019, 8, 1),
        (2019, 8, 2),
        (2019, 8, 3),
        (2019, 8, 4),
        (2019, 9, 1),
    ]
    assert data.frame["y"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert data.imputed == {key(catalog, rice).uid: 3}
