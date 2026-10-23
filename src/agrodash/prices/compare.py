"""Comparisons across markets and price types: where to sell and what a harvest is worth."""

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from agrodash.prices.queries import LatestPrice

# A market whose last price is older than this is no longer comparable.
CURRENT_WITHIN = timedelta(weeks=8)
# How a harvest is counted, per base unit, as the factor to that base unit.
HARVEST_UNITS = {
    "lb": {"libras": 1.0, "quintales (100 lb)": 100.0, "kilogramos": 2.20462},
    "unit": {"unidades": 1.0, "cientos": 100.0, "docenas": 12.0},
    "l": {"litros": 1.0, "galones": 3.78541},
}


@dataclass(frozen=True)
class RankedMarket:
    market: str
    value: float
    observed_on: date
    # Relative to the average of the ranked markets: -0.2 is 20% cheaper.
    difference: float


def market_ranking(latest: list[LatestPrice], base_unit: str) -> list[RankedMarket]:
    """Markets with a current price in `base_unit`, cheapest first."""
    priced = [row for row in latest if row.base_unit == base_unit]
    if not priced:
        return []
    newest = max(row.observed_on for row in priced)
    current = [row for row in priced if row.observed_on >= newest - CURRENT_WITHIN]
    average = sum(float(row.value) for row in current) / len(current)
    return sorted(
        (
            RankedMarket(
                row.market, float(row.value), row.observed_on, float(row.value) / average - 1
            )
            for row in current
        ),
        key=lambda row: row.value,
    )


def price_spread(wholesale: pd.DataFrame, retail: pd.DataFrame) -> pd.DataFrame:
    """Monthly wholesale and retail price (mean over markets) and retail / wholesale - 1.

    Takes monthly_prices frames of one product; months missing either type are left out.
    """
    sides = [
        frame.groupby(["year", "month"])["price"].mean().rename(name)
        for name, frame in (("wholesale", wholesale), ("retail", retail))
    ]
    both = pd.concat(sides, axis=1).dropna().reset_index()
    return pd.DataFrame(
        {
            "month": [
                date(int(y), int(m), 1) for y, m in zip(both["year"], both["month"], strict=True)
            ],
            "wholesale": both["wholesale"],
            "retail": both["retail"],
            "spread": both["retail"] / both["wholesale"] - 1,
        }
    )


def harvest_values(
    ranking: list[RankedMarket], quantity: float, factor: float
) -> list[tuple[str, float]]:
    """What `quantity` units (of `factor` base units each) fetch in each ranked market."""
    return [(row.market, quantity * factor * row.value) for row in ranking]
