from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, Unit
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType
from agrodash.prices.seasons import monthly_prices, season_index, season_profile


def monthly(rows):
    return pd.DataFrame(rows, columns=["product_id", "market_id", "year", "month", "price"])


def year_of(product_id, market_id, year, base, peaks):
    """Twelve monthly prices at `base`, except the months in `peaks` (month -> price)."""
    return [(product_id, market_id, year, month, peaks.get(month, base)) for month in range(1, 13)]


def test_the_index_is_each_month_against_its_own_year_and_market():
    rows = year_of(1, 1, 2020, 100, {3: 130, 9: 70}) + year_of(1, 2, 2020, 200, {3: 260, 9: 140})

    index = season_index(monthly(rows))

    march = index[(index.year == 2020) & (index.month == 3)]
    assert march["index"].round(1).tolist() == [130.0]


def test_years_with_too_few_months_do_not_count():
    rows = year_of(1, 1, 2020, 100, {}) + [(1, 1, 2021, month, 150) for month in range(1, 7)]

    index = season_index(monthly(rows))

    assert sorted(index.year.unique()) == [2020]


def test_the_profile_marks_high_and_low_months_and_counts_years():
    rows = [
        row
        for year in (2019, 2020, 2021)
        for row in year_of(1, 1, year, 100, {3: 130, 4: 112 if year < 2021 else 100, 9: 70})
    ]

    profile = {month.month: month for month in season_profile(season_index(monthly(rows)))}

    assert (profile[3].level, profile[4].level, profile[9].level, profile[6].level) == (
        "Alto",
        "Alto",
        "Bajo",
        "Normal",
    )
    assert profile[3].years == 3
    assert profile[3].index == pytest.approx(130 / 101 * 100, rel=0.01)


def test_monthly_prices_average_the_published_weeks_in_the_products_base_unit(session):
    product = Product(name="Ajo/ primera", category="Condimentos", base_unit=BaseUnit.POUND)
    market = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
    pound = Unit(code="1-lb", label="1 lb", base_unit=BaseUnit.POUND, factor=Decimal(1))
    each = Unit(code="1-unit", label="1 unit", base_unit=BaseUnit.UNIT, factor=Decimal(1))
    session.add_all([product, market, pound, each])
    for day, unit, amount, status in (
        (date(2020, 3, 1), pound, 100, PriceStatus.VALIDATED),
        (date(2020, 3, 8), pound, 120, PriceStatus.VALIDATED),
        (date(2020, 3, 15), pound, 500, PriceStatus.PENDING),
        (date(2020, 3, 22), each, 9, PriceStatus.VALIDATED),
    ):
        session.add(
            Price(
                product=product,
                market=market,
                unit=unit,
                observed_on=day,
                price_type=PriceType.RETAIL,
                source=PriceSource.OFFICIAL,
                status=status,
                amount=Decimal(amount),
                price_per_base_unit=Decimal(amount),
            )
        )
    session.commit()

    frame = monthly_prices(session, [product.id], PriceType.RETAIL)

    assert frame[["year", "month", "price"]].values.tolist() == [[2020, 3, 110.0]]
