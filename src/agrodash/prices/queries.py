"""Read-only queries behind the public price pages."""

import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agrodash.catalog.models import Market, Product, Unit
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType
from agrodash.prices.reports import series_base_unit

CHANGE_WINDOW = timedelta(weeks=4)
# A product whose latest price is older than this is not in the current reports.
CURRENT_WITHIN = timedelta(weeks=8)


@dataclass(frozen=True)
class ProductChoice:
    id: int
    name: str
    category: str
    # Latest published price of this product and type.
    latest_on: date


@dataclass(frozen=True)
class LatestPrice:
    market_id: int
    market: str
    base_unit: str
    value: Decimal
    observed_on: date
    # Relative change against the price four weeks earlier; None without one.
    change: float | None


@dataclass(frozen=True)
class HistoryPoint:
    market: str
    observed_on: date
    value: float


def published():
    """Official validated prices: what the public pages show."""
    return (Price.source == PriceSource.OFFICIAL, Price.status == PriceStatus.VALIDATED)


def products_with_prices(session: Session, price_type: PriceType) -> list[ProductChoice]:
    """Active products with published prices of `price_type`, with their latest date."""
    rows = session.execute(
        select(Product.id, Product.name, Product.category, func.max(Price.observed_on))
        .join(Price, Price.product_id == Product.id)
        .where(Product.is_active, Price.price_type == price_type, *published())
        .group_by(Product.id, Product.name, Product.category)
        .order_by(Product.category, Product.name)
    )
    return [ProductChoice(*row) for row in rows]


def latest_prices(session: Session, product_id: int, price_type: PriceType) -> list[LatestPrice]:
    """The latest price of the product in every market, newest first."""
    rows = session.execute(
        select(Market.id, Market.name, Unit.base_unit, Price.observed_on, Price.price_per_base_unit)
        .join(Market, Price.market_id == Market.id)
        .join(Unit, Price.unit_id == Unit.id)
        .where(Price.product_id == product_id, Price.price_type == price_type, *published())
        .order_by(Market.id, Unit.base_unit, Price.observed_on)
    ).all()
    series: dict[tuple, list] = {}
    for market_id, market, base_unit, day, value in rows:
        series.setdefault((market_id, market, str(base_unit)), []).append((day, value))
    latest = []
    for (market_id, market, base_unit), points in series.items():
        day, value = points[-1]
        latest.append(LatestPrice(market_id, market, base_unit, value, day, _change(points)))
    return sorted(latest, key=lambda price: (-price.observed_on.toordinal(), price.market))


def _change(points: list[tuple[date, Decimal]]) -> float | None:
    """Last price against the last one at least CHANGE_WINDOW earlier; None without one."""
    day, value = points[-1]
    earlier = [v for d, v in points if d <= day - CHANGE_WINDOW]
    return float(value / earlier[-1] - 1) if earlier else None


@dataclass(frozen=True)
class PriceChange:
    product_id: int
    product: str
    category: str
    base_unit: str
    # Mean of the markets' latest prices.
    value: float
    # Median over markets of the change against four weeks earlier.
    change: float


def price_changes(session: Session, price_type: PriceType) -> list[PriceChange]:
    """Current products by their four-week change, largest rise first."""
    newest = session.scalar(
        select(func.max(Price.observed_on)).where(Price.price_type == price_type, *published())
    )
    if newest is None:
        return []
    rows = session.execute(
        select(
            Product.id,
            Product.name,
            Product.category,
            Price.market_id,
            Unit.base_unit,
            Price.observed_on,
            Price.price_per_base_unit,
        )
        .join(Product, Price.product_id == Product.id)
        .join(Unit, Price.unit_id == Unit.id)
        .where(
            Product.is_active,
            Price.price_type == price_type,
            Price.observed_on >= newest - CURRENT_WITHIN - CHANGE_WINDOW - timedelta(weeks=4),
            *published(),
        )
        .order_by(Price.observed_on)
    ).all()
    series: dict[tuple, list] = {}
    for product_id, name, category, market_id, base_unit, day, value in rows:
        series.setdefault((product_id, name, category, str(base_unit), market_id), []).append(
            (day, value)
        )
    by_product: dict[tuple, list] = {}
    for (*product, _), points in series.items():
        change = _change(points)
        if points[-1][0] >= newest - CURRENT_WITHIN and change is not None:
            by_product.setdefault(tuple(product), []).append((float(points[-1][1]), change))
    # A product sold in two base units keeps the one more markets report.
    chosen: dict[int, tuple] = {}
    for product, markets in by_product.items():
        if product[0] not in chosen or len(markets) > len(by_product[chosen[product[0]]]):
            chosen[product[0]] = product
    changes = []
    for product in chosen.values():
        markets = by_product[product]
        changes.append(
            PriceChange(
                *product,
                value=sum(value for value, _ in markets) / len(markets),
                change=float(statistics.median(change for _, change in markets)),
            )
        )
    return sorted(changes, key=lambda change: -change.change)


def products_history(
    session: Session,
    product_ids: list[int],
    price_type: PriceType,
    base_unit: str,
    since: date,
) -> list[HistoryPoint]:
    """Weekly price of several products, averaged over markets; `market` holds the product."""
    rows = session.execute(
        select(Product.name, Price.observed_on, func.avg(Price.price_per_base_unit))
        .join(Product, Price.product_id == Product.id)
        .join(Unit, Price.unit_id == Unit.id)
        .where(
            Price.product_id.in_(product_ids),
            Price.price_type == price_type,
            Unit.base_unit == base_unit,
            Price.observed_on >= since,
            *published(),
        )
        .group_by(Product.name, Price.observed_on)
        .order_by(Product.name, Price.observed_on)
    )
    return [HistoryPoint(name, day, float(value)) for name, day, value in rows]


def price_history(
    session: Session,
    product_id: int,
    price_type: PriceType,
    base_unit: str,
    market_ids: list[int],
    since: date,
) -> list[HistoryPoint]:
    rows = session.execute(
        select(Market.name, Price.observed_on, Price.price_per_base_unit)
        .join(Market, Price.market_id == Market.id)
        .join(Unit, Price.unit_id == Unit.id)
        .where(
            Price.product_id == product_id,
            Price.price_type == price_type,
            Unit.base_unit == base_unit,
            Price.market_id.in_(market_ids),
            Price.observed_on >= since,
            *published(),
        )
        .order_by(Market.name, Price.observed_on)
    )
    return [HistoryPoint(market, day, float(value)) for market, day, value in rows]


def latest_report_date(session: Session) -> date | None:
    return session.scalar(select(func.max(Price.observed_on)).where(*published()))


@dataclass(frozen=True)
class Choice:
    id: int
    label: str


@dataclass(frozen=True)
class ReportRow:
    product: str
    market: str
    observed_on: date
    amount: Decimal
    unit: str
    status: str
    is_outlier: bool


def units_for_report(session: Session, product_id: int, price_type: PriceType) -> list[Choice]:
    """Units a report may use: those in the base unit of the product's official series."""
    product = session.get(Product, product_id)
    base_unit = series_base_unit(session, product, price_type)
    rows = session.execute(
        select(Unit.id, Unit.label).where(Unit.base_unit == base_unit).order_by(Unit.factor)
    )
    return [Choice(*row) for row in rows]


def active_markets(session: Session) -> list[Choice]:
    rows = session.execute(
        select(Market.id, Market.name).where(Market.is_active).order_by(Market.name)
    )
    return [Choice(*row) for row in rows]


def reports_by(session: Session, reporter_id: int, limit: int = 10) -> list[ReportRow]:
    """A reporter's latest field reports, newest first."""
    rows = session.execute(
        select(
            Product.name,
            Market.name,
            Price.observed_on,
            Price.amount,
            Unit.label,
            Price.status,
            Price.is_outlier,
        )
        .join(Product, Price.product_id == Product.id)
        .join(Market, Price.market_id == Market.id)
        .join(Unit, Price.unit_id == Unit.id)
        .where(Price.source == PriceSource.FIELD_REPORT, Price.reporter_id == reporter_id)
        .order_by(Price.created_at.desc(), Price.id.desc())
        .limit(limit)
    )
    return [ReportRow(*row[:5], str(row[5]), row[6]) for row in rows]
