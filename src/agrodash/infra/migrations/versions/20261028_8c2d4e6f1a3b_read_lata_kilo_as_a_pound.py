"""read Lata/1Kilo as a pound

Retail Ají Gustoso and Ají Cachucha were sold by "Lata/1Kilo" until julio 2022 and by
the pound after. Read as a kilo, those prices sat below wholesale and the label change
kept the same prices: the can is a pound. The label's alias moves to the pound unit and
the stored official retail prices of both products are converted again; the amounts as
published do not change.

Revision ID: 8c2d4e6f1a3b
Revises: 3f7a1c9d2e4b
Create Date: 2026-10-28 15:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8c2d4e6f1a3b"
down_revision: str | None = "3f7a1c9d2e4b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LABEL = "Lata/1Kilo"
KILO, POUND = "2.2046-lb", "1-lb"
# The last week sold by the can; from agosto 2022 the report already says "Lib".
LAST_CAN_WEEK = "2022-07-31"


def upgrade() -> None:
    # Per pound = the stored per-kilo-label value x 2.2046, the amount as published.
    _move(KILO, POUND, up=2.2046, down=1)


def downgrade() -> None:
    _move(POUND, KILO, up=1, down=2.2046)


def _move(old_code: str, new_code: str, *, up: float, down: float) -> None:
    connection = op.get_bind()
    units = dict(
        connection.execute(
            sa.text("SELECT code, id FROM units WHERE code IN (:old, :new)"),
            {"old": old_code, "new": new_code},
        ).all()
    )
    if len(units) < 2:
        return  # nothing was ever loaded in one of the two units
    connection.execute(
        sa.text("UPDATE unit_aliases SET unit_id = :new WHERE raw_text = :label"),
        {"new": units[new_code], "label": LABEL},
    )
    connection.execute(
        sa.text(
            "UPDATE prices SET unit_id = :new, "
            "price_per_base_unit = ROUND(price_per_base_unit * :up / :down, 4) "
            "WHERE unit_id = :old AND price_type = 'retail' AND source = 'official' "
            "AND observed_on <= :last AND product_id IN (SELECT id FROM products"
            " WHERE lower(name) LIKE '%gustoso%' OR lower(name) LIKE '%cachucha%')"
        ),
        {
            "new": units[new_code],
            "old": units[old_code],
            "last": LAST_CAN_WEEK,
            "up": up,
            "down": down,
        },
    )
