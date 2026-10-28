"""move wholesale prices to their reference market

Wholesale prices in the Dominican Republic come from Mercadom and Mercado Nuevo, the two
markets that trade wholesale; the report gives one price for both. They were stored under
Mercado Nuevo, which also holds its own retail prices. They move to a market of their own,
and Mercado Nuevo keeps only its retail prices.

Revision ID: 5e1f0c2a9b7d
Revises: eb10aa80820f
Create Date: 2026-10-23 22:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5e1f0c2a9b7d"
down_revision: str | None = "eb10aa80820f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_MARKET = "Mercado Nuevo"
REFERENCE_MARKET = "Referencia mayorista (Mercadom y Mercado Nuevo)"
# Every table whose rows belong to a (product, market, price type) series.
SERIES_TABLES = ("prices", "forecasts", "forecast_evaluations")


def upgrade() -> None:
    connection = op.get_bind()
    old = _market_id(connection, OLD_MARKET)
    if old is None:
        return
    reference = _market_id(connection, REFERENCE_MARKET)
    if reference is None:
        connection.execute(
            sa.text(
                "INSERT INTO markets (name, kind, is_active) VALUES (:name, 'wholesale', :active)"
            ),
            {"name": REFERENCE_MARKET, "active": True},
        )
        reference = _market_id(connection, REFERENCE_MARKET)
    _move_wholesale(connection, old, reference)
    connection.execute(sa.text("UPDATE markets SET kind = 'retail' WHERE id = :id"), {"id": old})


def downgrade() -> None:
    connection = op.get_bind()
    reference = _market_id(connection, REFERENCE_MARKET)
    if reference is None:
        return
    old = _market_id(connection, OLD_MARKET)
    if old is None:
        connection.execute(
            sa.text("UPDATE markets SET name = :name WHERE id = :id"),
            {"name": OLD_MARKET, "id": reference},
        )
        return
    _move_wholesale(connection, reference, old)
    connection.execute(sa.text("DELETE FROM markets WHERE id = :id"), {"id": reference})
    connection.execute(sa.text("UPDATE markets SET kind = 'wholesale' WHERE id = :id"), {"id": old})


def _market_id(connection, name: str) -> int | None:
    return connection.execute(
        sa.text("SELECT id FROM markets WHERE name = :name"), {"name": name}
    ).scalar()


def _move_wholesale(connection, source: int, target: int) -> None:
    for name in SERIES_TABLES:
        table = sa.table(name, sa.column("market_id"), sa.column("price_type"))
        connection.execute(
            sa.update(table)
            .where(table.c.market_id == source, table.c.price_type == "wholesale")
            .values(market_id=target)
        )
