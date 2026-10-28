"""add product name properties

The base name, variety, color, size and grade a product's name states, so pages can
group and filter by them. The next load fills them for every product.

Revision ID: 5b9e7d3c1f2a
Revises: 8c2d4e6f1a3b
Create Date: 2026-10-28 17:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5b9e7d3c1f2a"
down_revision: str | None = "8c2d4e6f1a3b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

COLUMNS = (("base_name", 60), ("variety", 80), ("color", 20), ("size", 20), ("grade", 20))


def upgrade() -> None:
    # Native ADD/DROP COLUMN: rebuilding products fails on SQLite once prices reference it.
    with op.batch_alter_table("products", schema=None, recreate="never") as batch_op:
        for name, length in COLUMNS:
            batch_op.add_column(sa.Column(name, sa.String(length=length), nullable=True))
    op.create_index("ix_products_base_name", "products", ["base_name"])


def downgrade() -> None:
    op.drop_index("ix_products_base_name", table_name="products")
    with op.batch_alter_table("products", schema=None, recreate="never") as batch_op:
        for name, _ in reversed(COLUMNS):
            batch_op.drop_column(name)
