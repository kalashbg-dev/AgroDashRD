"""drop unused market location

Revision ID: 9b40c1b7dccc
Revises: 2c4cc5bb4efe
Create Date: 2026-10-22 14:08:42.652666
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9b40c1b7dccc"
down_revision: str | None = "2c4cc5bb4efe"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Native DROP COLUMN (SQLite 3.35+): rebuilding the table instead fails once prices
    # reference markets, because the app enforces foreign keys on SQLite.
    with op.batch_alter_table("markets", schema=None, recreate="never") as batch_op:
        batch_op.drop_column("location")


def downgrade() -> None:
    with op.batch_alter_table("markets", schema=None, recreate="never") as batch_op:
        batch_op.add_column(sa.Column("location", sa.VARCHAR(length=120), nullable=True))
