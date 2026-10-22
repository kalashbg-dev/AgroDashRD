from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agrodash.catalog.models import BaseUnit, Market, Product
from agrodash.infra.db import Base, enum_column
from agrodash.prices.models import PriceType


class ForecastStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ForecastRun(Base):
    """One run of the forecasting job. The latest succeeded run is the one shown."""

    __tablename__ = "forecast_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[ForecastStatus] = mapped_column(
        enum_column(ForecastStatus, "forecast_status"), default=ForecastStatus.RUNNING
    )
    # Code and library versions that produced the run, to compare runs over time.
    model_version: Mapped[str] = mapped_column(String(120))
    horizon: Mapped[int]
    series_count: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(String(500))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    evaluations: Mapped[list["ForecastEvaluation"]] = relationship(back_populates="run")
    forecasts: Mapped[list["Forecast"]] = relationship(back_populates="run")


class _SeriesColumns:
    """The series a row belongs to: product, market, price type and base unit."""

    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"))
    price_type: Mapped[PriceType] = mapped_column(enum_column(PriceType, "price_type"))
    base_unit: Mapped[BaseUnit] = mapped_column(enum_column(BaseUnit, "base_unit"))


class ForecastEvaluation(_SeriesColumns, Base):
    """Backtest error of one model on one series; the lowest MASE is selected."""

    __tablename__ = "forecast_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("forecast_runs.id"), index=True)
    model: Mapped[str] = mapped_column(String(40))
    # Mean absolute error scaled by the series' one-step naive error: below 1 beats it.
    mase: Mapped[float]
    selected: Mapped[bool] = mapped_column(default=False)

    run: Mapped[ForecastRun] = relationship(back_populates="evaluations")
    product: Mapped[Product] = relationship()
    market: Mapped[Market] = relationship()


class Forecast(_SeriesColumns, Base):
    """Predicted price per base unit for one future report period of a series."""

    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint("run_id", "product_id", "market_id", "price_type", "base_unit", "step"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("forecast_runs.id"), index=True)
    model: Mapped[str] = mapped_column(String(40))
    # 1 is the next report period; target_on is that period's stored-date slot.
    step: Mapped[int]
    target_on: Mapped[date]
    value: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    lo_80: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    hi_80: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    lo_95: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    hi_95: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))

    run: Mapped[ForecastRun] = relationship(back_populates="forecasts")
    product: Mapped[Product] = relationship()
    market: Mapped[Market] = relationship()
