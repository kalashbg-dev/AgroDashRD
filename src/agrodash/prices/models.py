from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Numeric, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agrodash.accounts.models import User
from agrodash.catalog.models import Market, Product, Unit
from agrodash.infra.db import Base, enum_column


class PriceType(StrEnum):
    WHOLESALE = "wholesale"
    RETAIL = "retail"


class PriceSource(StrEnum):
    OFFICIAL = "official"
    FIELD_REPORT = "field_report"


class PriceStatus(StrEnum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"


class Price(Base):
    """One observed price for a product in a market during a week.

    `amount` is the price of the reported selling unit; `price_per_base_unit` is the
    same price converted to the product's base unit so series stay comparable.
    """

    __tablename__ = "prices"
    __table_args__ = (
        CheckConstraint("amount > 0", name="amount_positive"),
        CheckConstraint("price_per_base_unit > 0", name="price_per_base_unit_positive"),
        # Official sources publish one price per week; field reports may repeat.
        Index(
            "uq_prices_official_observation",
            "product_id",
            "market_id",
            "observed_on",
            "price_type",
            unique=True,
            sqlite_where=text("source = 'official'"),
            postgresql_where=text("source = 'official'"),
        ),
        Index("ix_prices_series", "product_id", "market_id", "price_type", "observed_on"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"))
    unit_id: Mapped[int] = mapped_column(ForeignKey("units.id"))
    # Who submitted a field report; official prices have no reporter.
    reporter_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    observed_on: Mapped[date]
    price_type: Mapped[PriceType] = mapped_column(enum_column(PriceType, "price_type"))
    source: Mapped[PriceSource] = mapped_column(enum_column(PriceSource, "price_source"))
    status: Mapped[PriceStatus] = mapped_column(
        enum_column(PriceStatus, "price_status"), default=PriceStatus.PENDING
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    price_per_base_unit: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    product: Mapped[Product] = relationship()
    market: Mapped[Market] = relationship()
    unit: Mapped[Unit] = relationship()
    reporter: Mapped[User | None] = relationship()
