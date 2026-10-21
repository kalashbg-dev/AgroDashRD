"""add ingestion runs and quarantine

Revision ID: cb3364211b13
Revises: adf580b781b9
Create Date: 2026-10-21 19:56:52.570885
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "cb3364211b13"
down_revision: str | None = "adf580b781b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("file_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "layout",
            sa.Enum(
                "wholesale",
                "retail",
                name="source_layout",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column(
            "trigger",
            sa.Enum(
                "drive",
                "upload",
                "file",
                name="ingestion_trigger",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "running",
                "succeeded",
                "failed",
                name="ingestion_status",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("started_by_id", sa.Integer(), nullable=True),
        sa.Column("rows_read", sa.Integer(), nullable=False),
        sa.Column("prices_loaded", sa.Integer(), nullable=False),
        sa.Column("rows_rejected", sa.Integer(), nullable=False),
        sa.Column("values_missing", sa.Integer(), nullable=False),
        sa.Column("error", sa.String(length=500), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "layout IN ('wholesale', 'retail')", name=op.f("ck_ingestion_runs_source_layout")
        ),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')",
            name=op.f("ck_ingestion_runs_ingestion_status"),
        ),
        sa.CheckConstraint(
            "\"trigger\" IN ('drive', 'upload', 'file')",
            name=op.f("ck_ingestion_runs_ingestion_trigger"),
        ),
        sa.ForeignKeyConstraint(
            ["started_by_id"], ["users.id"], name=op.f("fk_ingestion_runs_started_by_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_runs")),
    )
    with op.batch_alter_table("ingestion_runs", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_ingestion_runs_file_sha256"), ["file_sha256"], unique=False
        )
        batch_op.create_index(
            "uq_ingestion_runs_succeeded_file",
            ["file_sha256"],
            unique=True,
            sqlite_where=sa.text("status = 'succeeded'"),
            postgresql_where=sa.text("status = 'succeeded'"),
        )

    op.create_table(
        "quarantined_rows",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("row_number", sa.Integer(), nullable=False),
        sa.Column(
            "reason",
            sa.Enum(
                "invalid_date",
                "invalid_price",
                "unknown_unit",
                "unit_mismatch",
                "unexpected_layout",
                "duplicate_price",
                name="rejection_reason",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("detail", sa.String(length=200), nullable=True),
        sa.Column("raw_values", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "reason IN ('invalid_date', 'invalid_price', 'unknown_unit', 'unit_mismatch', "
            "'unexpected_layout', 'duplicate_price')",
            name=op.f("ck_quarantined_rows_rejection_reason"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.id"],
            name=op.f("fk_quarantined_rows_run_id_ingestion_runs"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_quarantined_rows")),
    )
    with op.batch_alter_table("quarantined_rows", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_quarantined_rows_run_id"), ["run_id"], unique=False)

    with op.batch_alter_table("prices", schema=None) as batch_op:
        batch_op.add_column(sa.Column("ingestion_run_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("is_outlier", sa.Boolean(), server_default=sa.false(), nullable=False)
        )
        batch_op.create_index(
            batch_op.f("ix_prices_ingestion_run_id"), ["ingestion_run_id"], unique=False
        )
        batch_op.create_foreign_key(
            batch_op.f("fk_prices_ingestion_run_id_ingestion_runs"),
            "ingestion_runs",
            ["ingestion_run_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("prices", schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f("fk_prices_ingestion_run_id_ingestion_runs"), type_="foreignkey"
        )
        batch_op.drop_index(batch_op.f("ix_prices_ingestion_run_id"))
        batch_op.drop_column("is_outlier")
        batch_op.drop_column("ingestion_run_id")

    with op.batch_alter_table("quarantined_rows", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_quarantined_rows_run_id"))

    op.drop_table("quarantined_rows")
    with op.batch_alter_table("ingestion_runs", schema=None) as batch_op:
        batch_op.drop_index(
            "uq_ingestion_runs_succeeded_file",
            sqlite_where=sa.text("status = 'succeeded'"),
            postgresql_where=sa.text("status = 'succeeded'"),
        )
        batch_op.drop_index(batch_op.f("ix_ingestion_runs_file_sha256"))

    op.drop_table("ingestion_runs")
