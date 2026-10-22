from datetime import UTC, datetime, timedelta

import pytest

from agrodash.accounts.models import Role, User
from agrodash.accounts.service import (
    AccountError,
    create_account,
    list_users,
    manageable_roles,
    set_active,
    unlock,
)

PASSWORD = "correct horse battery"


@pytest.fixture
def people(session):
    accounts = {}
    for role in Role:
        user = User(email=f"{role.lower()}@example.com", password_hash="x", role=role)
        session.add(user)
        accounts[role] = user
    session.commit()
    return accounts


@pytest.mark.parametrize(
    ("actor", "roles"),
    [
        (Role.TECNICO, []),
        (Role.ADMIN, [Role.TECNICO]),
        (Role.SUPERADMIN, [Role.TECNICO, Role.ADMIN, Role.SUPERADMIN]),
    ],
)
def test_who_can_manage_which_roles(actor, roles):
    assert manageable_roles(actor) == roles


def test_admins_create_technicians_but_not_admins(session, people):
    created = create_account(
        session, people[Role.ADMIN], "nuevo@example.com", PASSWORD, Role.TECNICO, "Nuevo"
    )

    assert (created.role, created.full_name) == (Role.TECNICO, "Nuevo")
    with pytest.raises(AccountError, match="cannot create ADMIN accounts"):
        create_account(session, people[Role.ADMIN], "otro@example.com", PASSWORD, Role.ADMIN)


def test_deactivating_and_reactivating_an_account(session, people):
    set_active(session, people[Role.ADMIN], people[Role.TECNICO].id, active=False)
    assert session.get(User, people[Role.TECNICO].id).is_active is False

    set_active(session, people[Role.ADMIN], people[Role.TECNICO].id, active=True)
    assert session.get(User, people[Role.TECNICO].id).is_active is True


@pytest.mark.parametrize(
    ("actor", "target", "message"),
    [
        (Role.ADMIN, Role.ADMIN, "cannot manage ADMIN accounts"),
        (Role.ADMIN, Role.SUPERADMIN, "cannot manage SUPERADMIN accounts"),
        (Role.TECNICO, Role.TECNICO, "Your account cannot manage accounts"),
    ],
)
def test_nobody_manages_accounts_above_their_reach(session, people, actor, target, message):
    with pytest.raises(AccountError, match=message):
        set_active(session, people[actor], people[target].id, active=False)
    assert session.get(User, people[target].id).is_active is True


def test_nobody_deactivates_themselves(session, people):
    with pytest.raises(AccountError, match="cannot deactivate your own account"):
        set_active(session, people[Role.SUPERADMIN], people[Role.SUPERADMIN].id, active=False)


def test_unlocking_clears_the_lockout(session, people):
    technician = people[Role.TECNICO]
    technician.locked_until = datetime.now(UTC) + timedelta(minutes=10)
    technician.failed_login_attempts = 3
    session.commit()

    unlock(session, people[Role.ADMIN], technician.id)

    assert (technician.locked_until, technician.failed_login_attempts) == (None, 0)


def test_users_are_listed_by_role_then_email(session, people):
    assert [user.role for user in list_users(session)] == [
        Role.SUPERADMIN,
        Role.ADMIN,
        Role.TECNICO,
    ]


@pytest.mark.parametrize(
    "action",
    [
        lambda s, p: create_account(s, p[Role.ADMIN], "nuevo@example.com", PASSWORD, Role.TECNICO),
        lambda s, p: set_active(s, p[Role.ADMIN], p[Role.TECNICO].id, active=False),
        lambda s, p: unlock(s, p[Role.ADMIN], p[Role.TECNICO].id),
    ],
)
def test_a_deactivated_admin_can_no_longer_manage_accounts(session, people, action):
    people[Role.ADMIN].is_active = False
    session.commit()

    with pytest.raises(AccountError, match="Your account cannot manage accounts"):
        action(session, people)
    assert session.get(User, people[Role.TECNICO].id).is_active is True


def test_a_deactivated_superadmin_cannot_reactivate_itself(session, people):
    superadmin = people[Role.SUPERADMIN]
    superadmin.is_active = False
    session.commit()

    with pytest.raises(AccountError, match="Your account cannot manage accounts"):
        set_active(session, superadmin, superadmin.id, active=True)
