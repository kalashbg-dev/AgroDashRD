"""create catalog and prices

Revision ID: 8d730078212e
Revises:
Create Date: 2026-10-21 15:18:14.699000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8d730078212e"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "markets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "wholesale",
                "retail",
                "supermarket",
                name="market_kind",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("location", sa.String(length=120), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "kind IN ('wholesale', 'retail', 'supermarket')", name=op.f("ck_markets_market_kind")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_markets")),
        sa.UniqueConstraint("name", name=op.f("uq_markets_name")),
    )
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("category", sa.String(length=60), nullable=False),
        sa.Column(
            "base_unit",
            sa.Enum(
                "lb",
                "unit",
                "l",
                name="base_unit",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.CheckConstraint("base_unit IN ('lb', 'unit', 'l')", name=op.f("ck_products_base_unit")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_products")),
        sa.UniqueConstraint("name", name=op.f("uq_products_name")),
    )
    with op.batch_alter_table("products", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_products_category"), ["category"], unique=False)

    op.create_table(
        "units",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=40), nullable=False),
        sa.Column("label", sa.String(length=80), nullable=False),
        sa.Column(
            "base_unit",
            sa.Enum(
                "lb",
                "unit",
                "l",
                name="base_unit",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("factor", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.CheckConstraint("base_unit IN ('lb', 'unit', 'l')", name=op.f("ck_units_base_unit")),
        sa.CheckConstraint("factor > 0", name=op.f("ck_units_factor_positive")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_units")),
        sa.UniqueConstraint("code", name=op.f("uq_units_code")),
    )
    op.create_table(
        "prices",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("market_id", sa.Integer(), nullable=False),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("observed_on", sa.Date(), nullable=False),
        sa.Column(
            "price_type",
            sa.Enum(
                "wholesale",
                "retail",
                name="price_type",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column(
            "source",
            sa.Enum(
                "official",
                "field_report",
                name="price_source",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "validated",
                "rejected",
                name="price_status",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("amount", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("price_per_base_unit", sa.Numeric(precision=14, scale=4), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "price_type IN ('wholesale', 'retail')", name=op.f("ck_prices_price_type")
        ),
        sa.CheckConstraint(
            "source IN ('official', 'field_report')", name=op.f("ck_prices_price_source")
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'validated', 'rejected')", name=op.f("ck_prices_price_status")
        ),
        sa.CheckConstraint("amount > 0", name=op.f("ck_prices_amount_positive")),
        sa.CheckConstraint(
            "price_per_base_unit > 0", name=op.f("ck_prices_price_per_base_unit_positive")
        ),
        sa.ForeignKeyConstraint(
            ["market_id"], ["markets.id"], name=op.f("fk_prices_market_id_markets")
        ),
        sa.ForeignKeyConstraint(
            ["product_id"], ["products.id"], name=op.f("fk_prices_product_id_products")
        ),
        sa.ForeignKeyConstraint(["unit_id"], ["units.id"], name=op.f("fk_prices_unit_id_units")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_prices")),
    )
    with op.batch_alter_table("prices", schema=None) as batch_op:
        batch_op.create_index(
            "ix_prices_series",
            ["product_id", "market_id", "price_type", "observed_on"],
            unique=False,
        )
        batch_op.create_index(
            "uq_prices_official_observation",
            ["product_id", "market_id", "observed_on", "price_type"],
            unique=True,
            sqlite_where=sa.text("source = 'official'"),
            postgresql_where=sa.text("source = 'official'"),
        )

    op.create_table(
        "product_aliases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("raw_name", sa.String(length=200), nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("reviewed", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["product_id"], ["products.id"], name=op.f("fk_product_aliases_product_id_products")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_product_aliases")),
        sa.UniqueConstraint("raw_name", name=op.f("uq_product_aliases_raw_name")),
    )
    with op.batch_alter_table("product_aliases", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_product_aliases_product_id"), ["product_id"], unique=False
        )

    op.create_table(
        "unit_aliases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.String(length=120), nullable=False),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["unit_id"], ["units.id"], name=op.f("fk_unit_aliases_unit_id_units")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_unit_aliases")),
        sa.UniqueConstraint("raw_text", name=op.f("uq_unit_aliases_raw_text")),
    )


def downgrade() -> None:
    op.drop_table("unit_aliases")
    with op.batch_alter_table("product_aliases", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_product_aliases_product_id"))

    op.drop_table("product_aliases")
    with op.batch_alter_table("prices", schema=None) as batch_op:
        batch_op.drop_index(
            "uq_prices_official_observation",
            sqlite_where=sa.text("source = 'official'"),
            postgresql_where=sa.text("source = 'official'"),
        )
        batch_op.drop_index("ix_prices_series")

    op.drop_table("prices")
    op.drop_table("units")
    with op.batch_alter_table("products", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_products_category"))

    op.drop_table("products")
    op.drop_table("markets")
