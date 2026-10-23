from datetime import date, timedelta
from decimal import Decimal

import pytest

from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, Unit
from agrodash.forecasting.models import Forecast, ForecastRun, ForecastStatus
from agrodash.forecasting.queries import latest_forecasts
from agrodash.ingestion import quality
from agrodash.ingestion.normalize import week_slot
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType
from agrodash.prices.queries import (
    latest_prices,
    price_changes,
    price_history,
    products_history,
    products_with_prices,
)

RETAIL = PriceType.RETAIL


@pytest.fixture
def catalog(session):
    rice = Product(name="Arroz Selecto", category="Cereales", base_unit=BaseUnit.POUND)
    retired = Product(
        name="Arroz Viejo", category="Cereales", base_unit=BaseUnit.POUND, is_active=False
    )
    mango = Product(name="Mango", category="Frutas", base_unit=BaseUnit.UNIT)
    nuevo = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
    mina = Market(name="Mercado Los Mina", kind=MarketKind.RETAIL)
    pound = Unit(code="1-lb", label="1 lb", base_unit=BaseUnit.POUND, factor=Decimal(1))
    session.add_all([rice, retired, mango, nuevo, mina, pound])
    session.commit()
    return {
        "rice": rice,
        "retired": retired,
        "mango": mango,
        "nuevo": nuevo,
        "mina": mina,
        "pound": pound,
    }


def add(session, catalog, product, market, day, value, **extra):
    values = {
        "product": catalog[product],
        "market": catalog[market],
        "unit": catalog["pound"],
        "observed_on": day,
        "price_type": RETAIL,
        "source": PriceSource.OFFICIAL,
        "status": PriceStatus.VALIDATED,
        "amount": Decimal(value),
        "price_per_base_unit": Decimal(value),
    }
    values.update(extra)
    session.add(Price(**values))


def test_only_active_products_with_validated_prices_are_offered(session, catalog):
    add(session, catalog, "rice", "nuevo", date(2026, 9, 1), "30")
    add(session, catalog, "retired", "nuevo", date(2026, 9, 1), "30")
    add(session, catalog, "mango", "nuevo", date(2026, 9, 1), "20", status=PriceStatus.PENDING)
    session.commit()

    assert [(p.name, p.category) for p in products_with_prices(session, RETAIL)] == [
        ("Arroz Selecto", "Cereales")
    ]
    assert products_with_prices(session, PriceType.WHOLESALE) == []


def test_latest_prices_per_market_with_the_change_over_four_weeks(session, catalog):
    for day, value in (
        (date(2026, 8, 1), "25"),
        (date(2026, 8, 29), "30"),
        (date(2026, 9, 1), "33"),
    ):
        add(session, catalog, "rice", "nuevo", day, value)
    add(session, catalog, "rice", "mina", date(2026, 8, 22), "28")
    add(
        session,
        catalog,
        "rice",
        "nuevo",
        date(2026, 9, 8),
        "99",
        source=PriceSource.FIELD_REPORT,
        status=PriceStatus.PENDING,
    )
    session.commit()

    rows = latest_prices(session, catalog["rice"].id, RETAIL)

    assert [(r.market, r.value, r.observed_on, r.change) for r in rows] == [
        ("Mercado Nuevo", Decimal("33.0000"), date(2026, 9, 1), pytest.approx(0.32)),
        ("Mercado Los Mina", Decimal("28.0000"), date(2026, 8, 22), None),
    ]
    assert rows[0].base_unit == "lb"


def test_price_history_returns_the_chosen_markets_since_a_date(session, catalog):
    for week in range(1, 5):
        add(session, catalog, "rice", "nuevo", week_slot(2026, 9, week), str(30 + week))
        add(session, catalog, "rice", "mina", week_slot(2026, 9, week), str(20 + week))
    session.commit()

    history = price_history(
        session, catalog["rice"].id, RETAIL, "lb", [catalog["nuevo"].id], since=date(2026, 9, 8)
    )

    assert [(h.market, h.observed_on, h.value) for h in history] == [
        ("Mercado Nuevo", date(2026, 9, 8), 32.0),
        ("Mercado Nuevo", date(2026, 9, 15), 33.0),
        ("Mercado Nuevo", date(2026, 9, 22), 34.0),
    ]


def test_forecasts_come_from_the_latest_succeeded_run(session, catalog):
    series = {
        "product": catalog["rice"],
        "market": catalog["nuevo"],
        "price_type": RETAIL,
        "base_unit": BaseUnit.POUND,
    }
    old = ForecastRun(model_version="v", horizon=4, status=ForecastStatus.SUCCEEDED)
    new = ForecastRun(model_version="v", horizon=4, status=ForecastStatus.SUCCEEDED)
    failed = ForecastRun(model_version="v", horizon=4, status=ForecastStatus.FAILED)
    session.add_all([old, new, failed])
    for run, value in ((old, "10"), (new, "33"), (failed, "99")):
        session.add(
            Forecast(
                run=run,
                model="Naive",
                step=1,
                target_on=date(2026, 10, 1),
                value=Decimal(value),
                lo_80=Decimal("30"),
                hi_80=Decimal("35"),
                lo_95=Decimal("29"),
                hi_95=Decimal("37"),
                **series,
            )
        )
    session.commit()

    forecasts = latest_forecasts(session, catalog["rice"].id, RETAIL)

    assert [(f.market, f.step, f.value, f.lo_80, f.hi_95) for f in forecasts] == [
        ("Mercado Nuevo", 1, Decimal("33.0000"), Decimal("30.0000"), Decimal("37.0000"))
    ]


def test_series_say_why_they_are_not_ready(session, catalog, monkeypatch):
    monkeypatch.setattr(quality, "MIN_WEEKS", 8)
    add(session, catalog, "rice", "nuevo", date(2026, 9, 1), "30")
    for weeks_ago in range(0, 40, 4):
        add(session, catalog, "mango", "mina", date(2026, 9, 22) - timedelta(weeks=weeks_ago), "20")
    session.commit()

    reasons = {(s.product, s.market): s.not_ready for s in quality.series_coverage(session)}

    assert reasons[("Arroz Selecto", "Mercado Nuevo")] == ("short", "sparse")
    assert reasons[("Mango", "Mercado Los Mina")] == ("sparse",)


def test_price_changes_use_the_four_week_rule_and_skip_stale_and_retired_products(session, catalog):
    day = date(2026, 9, 22)
    for market, before, now in (("nuevo", 40, 50), ("mina", 40, 60)):
        add(session, catalog, "rice", market, day - timedelta(weeks=4), before)
        add(session, catalog, "rice", market, day, now)
    add(session, catalog, "mango", "nuevo", day - timedelta(weeks=5), 20)
    add(session, catalog, "mango", "nuevo", day, 15)
    # Retired, and a price too old to be current: neither is listed.
    add(session, catalog, "retired", "nuevo", day - timedelta(weeks=4), 10)
    add(session, catalog, "retired", "nuevo", day, 30)
    session.commit()

    changes = price_changes(session, RETAIL)

    assert [(c.product, c.value, round(c.change, 3)) for c in changes] == [
        ("Arroz Selecto", 55.0, 0.375),
        ("Mango", 15.0, -0.25),
    ]


def test_products_history_averages_markets_per_week(session, catalog):
    day = date(2026, 9, 22)
    add(session, catalog, "rice", "nuevo", day, 40)
    add(session, catalog, "rice", "mina", day, 50)
    add(session, catalog, "mango", "nuevo", day, 12)
    add(session, catalog, "mango", "nuevo", day - timedelta(weeks=60), 9)
    session.commit()

    points = products_history(
        session,
        [catalog["rice"].id, catalog["mango"].id],
        RETAIL,
        "lb",
        day - timedelta(weeks=8),
    )

    assert [(p.market, p.observed_on, p.value) for p in points] == [
        ("Arroz Selecto", day, 45.0),
        ("Mango", day, 12.0),
    ]
