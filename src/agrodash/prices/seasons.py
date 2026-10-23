"""Seasonal price patterns: the months a product is usually dear or cheap.

Each month is indexed against its own year and market (100 = that year's average),
so a dearer market or a year of inflation does not bend the pattern; the profile is
the median index of each month across years.
"""

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import extract, func, select
from sqlalchemy.orm import Session

from agrodash.catalog.models import Product, Unit
from agrodash.prices.models import Price, PriceType
from agrodash.prices.queries import published

HIGH = 110
LOW = 90
# A year's average needs most of its months, or one season would weigh too much.
FULL_YEAR_MONTHS = 9
# Fewer years than this make a month's index anecdotal; pages label it.
MIN_YEARS = 3


@dataclass(frozen=True)
class MonthProfile:
    month: int
    index: float
    years: int
    level: str  # "Alto", "Bajo" or "Normal"


def monthly_prices(session: Session, product_ids: list[int], price_type: PriceType) -> pd.DataFrame:
    """Mean published price per product, market and month, in each product's base unit."""
    year = extract("year", Price.observed_on)
    month = extract("month", Price.observed_on)
    rows = session.execute(
        select(
            Price.product_id,
            Price.market_id,
            year,
            month,
            func.avg(Price.price_per_base_unit),
        )
        .join(Product, Price.product_id == Product.id)
        .join(Unit, Price.unit_id == Unit.id)
        .where(
            Price.product_id.in_(product_ids),
            Price.price_type == price_type,
            Unit.base_unit == Product.base_unit,
            *published(),
        )
        .group_by(Price.product_id, Price.market_id, year, month)
        .order_by(Price.product_id, Price.market_id, year, month)
    ).all()
    frame = pd.DataFrame(rows, columns=["product_id", "market_id", "year", "month", "price"])
    return frame.astype({"year": int, "month": int, "price": float})


def season_index(monthly: pd.DataFrame) -> pd.DataFrame:
    """Index per product, year and month: the median over markets of month / year mean."""
    by_year = monthly.groupby(["product_id", "market_id", "year"])["price"]
    full = monthly[by_year.transform("size") >= FULL_YEAR_MONTHS]
    full = full.assign(
        index=full["price"]
        / full.groupby(["product_id", "market_id", "year"])["price"].transform("mean")
        * 100
    )
    return full.groupby(["product_id", "year", "month"], as_index=False)["index"].median()


def season_profile(index: pd.DataFrame) -> list[MonthProfile]:
    """The usual index of each month (median across years), for one product's index."""
    months = index.groupby("month")["index"].agg(["median", "size"])
    return [
        MonthProfile(int(month), float(row["median"]), int(row["size"]), _level(row["median"]))
        for month, row in months.iterrows()
    ]


def _level(index: float) -> str:
    if index >= HIGH:
        return "Alto"
    if index <= LOW:
        return "Bajo"
    return "Normal"
