"""mark repeated prices

Official weeks that repeat the previous week's prices (audit G3) are stored with a
flag so forecasts can skip them while the history still shows them.

Revision ID: 3f7a1c9d2e4b
Revises: 5e1f0c2a9b7d
Create Date: 2026-10-28 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3f7a1c9d2e4b"
down_revision: str | None = "5e1f0c2a9b7d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Native ADD/DROP COLUMN: rebuilding prices fails on SQLite once rows reference it.
    with op.batch_alter_table("prices", schema=None, recreate="never") as batch_op:
        batch_op.add_column(
            sa.Column("repeated", sa.Boolean(), server_default=sa.false(), nullable=False)
        )


def downgrade() -> None:
    with op.batch_alter_table("prices", schema=None, recreate="never") as batch_op:
        batch_op.drop_column("repeated")
