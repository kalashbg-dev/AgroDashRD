from decimal import Decimal
from enum import StrEnum

from sqlalchemy import CheckConstraint, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agrodash.infra.db import Base, enum_column


class BaseUnit(StrEnum):
    """Unit every price is converted to before comparing or forecasting."""

    POUND = "lb"
    UNIT = "unit"
    LITER = "l"


class MarketKind(StrEnum):
    WHOLESALE = "wholesale"
    RETAIL = "retail"
    SUPERMARKET = "supermarket"


class Unit(Base):
    """A selling unit as reported (e.g. "Saco/125 lb") and its size in base units."""

    __tablename__ = "units"
    __table_args__ = (CheckConstraint("factor > 0", name="factor_positive"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(40), unique=True)
    label: Mapped[str] = mapped_column(String(80))
    base_unit: Mapped[BaseUnit] = mapped_column(enum_column(BaseUnit, "base_unit"))
    factor: Mapped[Decimal] = mapped_column(Numeric(12, 4))


class UnitAlias(Base):
    """Raw unit text found in a source file, mapped to its canonical unit."""

    __tablename__ = "unit_aliases"

    id: Mapped[int] = mapped_column(primary_key=True)
    raw_text: Mapped[str] = mapped_column(String(120), unique=True)
    unit_id: Mapped[int] = mapped_column(ForeignKey("units.id"))

    unit: Mapped[Unit] = relationship()


class Product(Base):
    """A product as sold, one per grade and size (e.g. "Arroz Selecto/ primera").

    `base_unit` is the unit of the series that first introduced the product. Each price
    type keeps its own base unit (melons are priced per pound wholesale and per piece
    retail); ingestion only enforces that one series never mixes base units.
    """

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    category: Mapped[str] = mapped_column(String(60), index=True)
    base_unit: Mapped[BaseUnit] = mapped_column(enum_column(BaseUnit, "base_unit"))
    is_active: Mapped[bool] = mapped_column(default=True)


class ProductAlias(Base):
    """Raw product name found in a source file, mapped to its canonical product.

    Aliases drafted automatically start unreviewed until an admin confirms them.
    """

    __tablename__ = "product_aliases"

    id: Mapped[int] = mapped_column(primary_key=True)
    raw_name: Mapped[str] = mapped_column(String(200), unique=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    reviewed: Mapped[bool] = mapped_column(default=False)

    product: Mapped[Product] = relationship()


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    kind: Mapped[MarketKind] = mapped_column(enum_column(MarketKind, "market_kind"))
    location: Mapped[str | None] = mapped_column(String(120))
    is_active: Mapped[bool] = mapped_column(default=True)
