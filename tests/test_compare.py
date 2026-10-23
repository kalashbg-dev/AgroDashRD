from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from agrodash.prices.compare import (
    HARVEST_UNITS,
    harvest_values,
    market_ranking,
    price_spread,
)
from agrodash.prices.queries import LatestPrice


def latest(market, value, day=date(2026, 9, 22), base_unit="lb"):
    return LatestPrice(1, market, base_unit, Decimal(value), day, None)


def test_markets_are_ranked_cheapest_first_against_the_average_of_current_prices():
    ranking = market_ranking(
        [
            latest("Mercado Nuevo", 40),
            latest("Supermercado", 60),
            latest("Mercadom", 50),
            # Not reported for months: neither ranked nor in the average.
            latest("Mercado Viejo", 5, day=date(2026, 3, 1)),
            latest("Otra unidad", 3, base_unit="unit"),
        ],
        "lb",
    )

    assert [(row.market, row.value, round(row.difference, 2)) for row in ranking] == [
        ("Mercado Nuevo", 40.0, -0.2),
        ("Mercadom", 50.0, 0.0),
        ("Supermercado", 60.0, 0.2),
    ]


def test_the_spread_is_retail_over_wholesale_per_month():
    wholesale = pd.DataFrame(
        [(1, 1, 2026, 1, 20.0), (1, 1, 2026, 2, 25.0)],
        columns=["product_id", "market_id", "year", "month", "price"],
    )
    retail = pd.DataFrame(
        [(1, 2, 2026, 1, 28.0), (1, 3, 2026, 1, 32.0), (1, 2, 2026, 3, 40.0)],
        columns=["product_id", "market_id", "year", "month", "price"],
    )

    spread = price_spread(wholesale, retail)

    assert spread["month"].tolist() == [date(2026, 1, 1)]
    assert spread[["wholesale", "retail"]].values.tolist() == [[20.0, 30.0]]
    assert spread["spread"].tolist() == [pytest.approx(0.5)]


def test_a_harvest_is_valued_in_each_market_from_its_price_per_base_unit():
    ranking = market_ranking([latest("Mercado Nuevo", 40), latest("Mercadom", 50)], "lb")

    values = harvest_values(ranking, quantity=3, factor=HARVEST_UNITS["lb"]["quintales (100 lb)"])

    assert values == [("Mercado Nuevo", 12000.0), ("Mercadom", 15000.0)]
