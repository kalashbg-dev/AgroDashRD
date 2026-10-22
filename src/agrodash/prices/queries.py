"""Read-only queries behind the public price pages."""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agrodash.catalog.models import Market, Product, Unit
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType

CHANGE_WINDOW = timedelta(weeks=4)


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


def _published():
    """Official validated prices: what the public pages show."""
    return (Price.source == PriceSource.OFFICIAL, Price.status == PriceStatus.VALIDATED)


def products_with_prices(session: Session, price_type: PriceType) -> list[ProductChoice]:
    """Active products with published prices of `price_type`, with their latest date."""
    rows = session.execute(
        select(Product.id, Product.name, Product.category, func.max(Price.observed_on))
        .join(Price, Price.product_id == Product.id)
        .where(Product.is_active, Price.price_type == price_type, *_published())
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
        .where(Price.product_id == product_id, Price.price_type == price_type, *_published())
        .order_by(Market.id, Unit.base_unit, Price.observed_on)
    ).all()
    series: dict[tuple, list] = {}
    for market_id, market, base_unit, day, value in rows:
        series.setdefault((market_id, market, str(base_unit)), []).append((day, value))
    latest = []
    for (market_id, market, base_unit), points in series.items():
        day, value = points[-1]
        earlier = [v for d, v in points if d <= day - CHANGE_WINDOW]
        change = float(value / earlier[-1] - 1) if earlier else None
        latest.append(LatestPrice(market_id, market, base_unit, value, day, change))
    return sorted(latest, key=lambda price: (-price.observed_on.toordinal(), price.market))


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
            *_published(),
        )
        .order_by(Market.name, Price.observed_on)
    )
    return [HistoryPoint(market, day, float(value)) for market, day, value in rows]


def latest_report_date(session: Session) -> date | None:
    return session.scalar(select(func.max(Price.observed_on)).where(*_published()))
