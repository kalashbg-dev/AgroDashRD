"""add rows corrected to ingestion runs

Revision ID: 807c815fc538
Revises: cb3364211b13
Create Date: 2026-10-22 00:56:46.596147
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "807c815fc538"
down_revision: str | None = "cb3364211b13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("ingestion_runs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("rows_corrected", sa.Integer(), server_default=sa.text("0"), nullable=False)
        )


def downgrade() -> None:
    with op.batch_alter_table("ingestion_runs", schema=None) as batch_op:
        batch_op.drop_column("rows_corrected")
