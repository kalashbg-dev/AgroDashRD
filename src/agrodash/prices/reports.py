"""Field price reports: submitted by technicians, flagged against the official series,
and validated or rejected by an admin before they count."""

from datetime import date, timedelta
from decimal import Decimal
from statistics import median

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from agrodash.accounts.models import User
from agrodash.accounts.service import can_manage, can_report_prices
from agrodash.catalog.models import BaseUnit, Market, Product, Unit
from agrodash.prices.models import (
    Price,
    PriceSource,
    PriceStatus,
    PriceType,
    per_base_unit,
    round_amount,
)

# A report more than 50% away from the recent official median is flagged for review.
OUTLIER_DEVIATION = Decimal("0.5")
REFERENCE_WINDOW = timedelta(weeks=8)


class ReportError(ValueError):
    """A report or review that cannot be accepted, safe to show to the user."""


def report_price(
    session: Session,
    reporter: User,
    *,
    product_id: int,
    market_id: int,
    unit_id: int,
    amount: Decimal,
    observed_on: date,
    price_type: PriceType,
) -> Price:
    if not (reporter.is_active and can_report_prices(reporter.role)):
        raise ReportError("Your account cannot report prices")
    product = session.get(Product, product_id)
    market = session.get(Market, market_id)
    unit = session.get(Unit, unit_id)
    if product is None or market is None or unit is None:
        raise ReportError("Unknown product, market or unit")
    amount = round_amount(amount)
    if amount <= 0:
        raise ReportError("Price must be greater than zero")
    if observed_on > date.today():
        raise ReportError("The observation date cannot be in the future")
    series_unit = _series_base_unit(session, product, price_type)
    if unit.base_unit != series_unit:
        raise ReportError(
            f"{product.name} {price_type} prices are per {series_unit}; "
            f"choose a unit in {series_unit}"
        )

    converted = per_base_unit(amount, unit.factor)
    reference = _official_median(session, product, market, price_type, observed_on)
    price = Price(
        product=product,
        market=market,
        unit=unit,
        reporter_id=reporter.id,
        observed_on=observed_on,
        price_type=price_type,
        source=PriceSource.FIELD_REPORT,
        status=PriceStatus.PENDING,
        amount=amount,
        price_per_base_unit=converted,
        is_outlier=reference is not None
        and abs(converted - reference) > reference * OUTLIER_DEVIATION,
    )
    session.add(price)
    session.commit()
    return price


def review_report(session: Session, reviewer: User, price_id: int, *, approve: bool) -> Price:
    if not (reviewer.is_active and can_manage(reviewer.role)):
        raise ReportError("Your account cannot review reports")
    # Conditional update: if another admin reviewed it meanwhile, nothing matches.
    reviewed = session.execute(
        update(Price)
        .where(
            Price.id == price_id,
            Price.source == PriceSource.FIELD_REPORT,
            Price.status == PriceStatus.PENDING,
        )
        .values(status=PriceStatus.VALIDATED if approve else PriceStatus.REJECTED)
        .execution_options(synchronize_session=False)
    )
    if reviewed.rowcount != 1:
        session.rollback()
        raise ReportError("This price is not a pending field report")
    session.commit()
    return session.get(Price, price_id)


def pending_reports(session: Session) -> list[Price]:
    """Reports waiting for review, outliers first, then oldest first."""
    return list(
        session.scalars(
            select(Price)
            .where(Price.source == PriceSource.FIELD_REPORT, Price.status == PriceStatus.PENDING)
            .order_by(Price.is_outlier.desc(), Price.created_at, Price.id)
        )
    )


def _series_base_unit(session: Session, product: Product, price_type: PriceType) -> BaseUnit:
    """Base unit the official series uses, or the product's when there is no series yet."""
    most_used = session.scalar(
        select(Unit.base_unit)
        .join(Price, Price.unit_id == Unit.id)
        .where(
            Price.product_id == product.id,
            Price.price_type == price_type,
            Price.source == PriceSource.OFFICIAL,
        )
        .group_by(Unit.base_unit)
        .order_by(func.count().desc(), Unit.base_unit)
        .limit(1)
    )
    return most_used or product.base_unit


def _official_median(
    session: Session, product: Product, market: Market, price_type: PriceType, on: date
) -> Decimal | None:
    """Median official price per base unit in the weeks before `on`: at this market when
    it has prices, otherwise across all markets."""
    query = select(Price.price_per_base_unit).where(
        Price.product_id == product.id,
        Price.price_type == price_type,
        Price.source == PriceSource.OFFICIAL,
        Price.status == PriceStatus.VALIDATED,
        Price.observed_on.between(on - REFERENCE_WINDOW, on),
    )
    values = list(session.scalars(query.where(Price.market_id == market.id))) or list(
        session.scalars(query)
    )
    return median(values) if values else None
