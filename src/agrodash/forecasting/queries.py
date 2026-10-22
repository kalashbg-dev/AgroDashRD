"""Read-only queries behind the public forecast page."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from agrodash.catalog.models import Market
from agrodash.forecasting.models import Forecast, ForecastRun, ForecastStatus
from agrodash.prices.models import PriceType


@dataclass(frozen=True)
class ForecastPoint:
    market_id: int
    market: str
    base_unit: str
    step: int
    target_on: date
    value: Decimal
    lo_80: Decimal | None
    hi_80: Decimal | None
    lo_95: Decimal | None
    hi_95: Decimal | None


def latest_run(session: Session) -> ForecastRun | None:
    return session.scalar(
        select(ForecastRun)
        .where(ForecastRun.status == ForecastStatus.SUCCEEDED)
        .order_by(ForecastRun.id.desc())
        .limit(1)
    )


def latest_forecasts(
    session: Session, product_id: int, price_type: PriceType
) -> list[ForecastPoint]:
    """Forecasts of the product from the latest succeeded run, by market and step."""
    run = latest_run(session)
    if run is None:
        return []
    rows = session.execute(
        select(
            Market.id,
            Market.name,
            Forecast.base_unit,
            Forecast.step,
            Forecast.target_on,
            Forecast.value,
            Forecast.lo_80,
            Forecast.hi_80,
            Forecast.lo_95,
            Forecast.hi_95,
        )
        .join(Market, Forecast.market_id == Market.id)
        .where(
            Forecast.run_id == run.id,
            Forecast.product_id == product_id,
            Forecast.price_type == price_type,
        )
        .order_by(Market.name, Forecast.step)
    )
    return [
        ForecastPoint(market_id, market, str(base_unit), *rest)
        for market_id, market, base_unit, *rest in rows
    ]
