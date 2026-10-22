"""add users and price reporter

Revision ID: adf580b781b9
Revises: 8d730078212e
Create Date: 2026-10-21 15:26:16.337357
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "adf580b781b9"
down_revision: str | None = "8d730078212e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=120), nullable=True),
        sa.Column(
            "role",
            sa.Enum(
                "AGRICULTOR",
                "TECNICO",
                "ADMIN",
                "SUPERADMIN",
                name="role",
                native_enum=False,
                create_constraint=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("failed_login_attempts", sa.Integer(), nullable=False),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('AGRICULTOR', 'TECNICO', 'ADMIN', 'SUPERADMIN')", name=op.f("ck_users_role")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )
    with op.batch_alter_table("prices", schema=None) as batch_op:
        batch_op.add_column(sa.Column("reporter_id", sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f("ix_prices_reporter_id"), ["reporter_id"], unique=False)
        batch_op.create_foreign_key(
            batch_op.f("fk_prices_reporter_id_users"), "users", ["reporter_id"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("prices", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("fk_prices_reporter_id_users"), type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_prices_reporter_id"))
        batch_op.drop_column("reporter_id")

    op.drop_table("users")
