"""add forecast runs evaluations and forecasts

Revision ID: 2c4cc5bb4efe
Revises: 807c815fc538
Create Date: 2026-10-22 13:44:01.555235
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "2c4cc5bb4efe"
down_revision: str | None = "807c815fc538"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "forecast_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "running",
                "succeeded",
                "failed",
                name="forecast_status",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("series_count", sa.Integer(), nullable=False),
        sa.Column("error", sa.String(length=500), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')",
            name=op.f("ck_forecast_runs_forecast_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_forecast_runs")),
    )
    op.create_table(
        "forecast_evaluations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=40), nullable=False),
        sa.Column("mase", sa.Double(), nullable=False),
        sa.Column("selected", sa.Boolean(), nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("market_id", sa.Integer(), nullable=False),
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
        sa.CheckConstraint(
            "base_unit IN ('lb', 'unit', 'l')", name=op.f("ck_forecast_evaluations_base_unit")
        ),
        sa.CheckConstraint(
            "price_type IN ('wholesale', 'retail')", name=op.f("ck_forecast_evaluations_price_type")
        ),
        sa.ForeignKeyConstraint(
            ["market_id"], ["markets.id"], name=op.f("fk_forecast_evaluations_market_id_markets")
        ),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["products.id"],
            name=op.f("fk_forecast_evaluations_product_id_products"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["forecast_runs.id"],
            name=op.f("fk_forecast_evaluations_run_id_forecast_runs"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_forecast_evaluations")),
    )
    with op.batch_alter_table("forecast_evaluations", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_forecast_evaluations_run_id"), ["run_id"], unique=False
        )

    op.create_table(
        "forecasts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=40), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("target_on", sa.Date(), nullable=False),
        sa.Column("value", sa.Numeric(precision=14, scale=4), nullable=False),
        sa.Column("lo_80", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("hi_80", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("lo_95", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("hi_95", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("market_id", sa.Integer(), nullable=False),
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
        sa.CheckConstraint("base_unit IN ('lb', 'unit', 'l')", name=op.f("ck_forecasts_base_unit")),
        sa.CheckConstraint(
            "price_type IN ('wholesale', 'retail')", name=op.f("ck_forecasts_price_type")
        ),
        sa.ForeignKeyConstraint(
            ["market_id"], ["markets.id"], name=op.f("fk_forecasts_market_id_markets")
        ),
        sa.ForeignKeyConstraint(
            ["product_id"], ["products.id"], name=op.f("fk_forecasts_product_id_products")
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["forecast_runs.id"], name=op.f("fk_forecasts_run_id_forecast_runs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_forecasts")),
        sa.UniqueConstraint(
            "run_id",
            "product_id",
            "market_id",
            "price_type",
            "base_unit",
            "step",
            name=op.f("uq_forecasts_run_id"),
        ),
    )
    with op.batch_alter_table("forecasts", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_forecasts_run_id"), ["run_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("forecasts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_forecasts_run_id"))

    op.drop_table("forecasts")
    with op.batch_alter_table("forecast_evaluations", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_forecast_evaluations_run_id"))

    op.drop_table("forecast_evaluations")
    op.drop_table("forecast_runs")
