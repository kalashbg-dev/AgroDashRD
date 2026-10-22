"""remove the farmer role

Revision ID: eb10aa80820f
Revises: 9b40c1b7dccc
Create Date: 2026-10-22 16:50:51.577211
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "eb10aa80820f"
down_revision: str | None = "9b40c1b7dccc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


ROLES_BEFORE = "role IN ('AGRICULTOR', 'TECNICO', 'ADMIN', 'SUPERADMIN')"
ROLES_AFTER = "role IN ('TECNICO', 'ADMIN', 'SUPERADMIN')"


def upgrade() -> None:
    # Never change or delete accounts silently: stop and say what to fix first.
    farmers = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM users WHERE role = 'AGRICULTOR'"))
        .scalar()
    )
    if farmers:
        raise RuntimeError(
            f"{farmers} AGRICULTOR account(s) still exist; give them another role or delete "
            "them, then upgrade again"
        )
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_constraint(op.f("ck_users_role"), type_="check")
        batch_op.create_check_constraint(op.f("ck_users_role"), ROLES_AFTER)


def downgrade() -> None:
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_constraint(op.f("ck_users_role"), type_="check")
        batch_op.create_check_constraint(op.f("ck_users_role"), ROLES_BEFORE)
