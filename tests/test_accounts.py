import io
from datetime import UTC, datetime, timedelta

import pytest

from agrodash import cli
from agrodash.accounts.models import Role, User
from agrodash.accounts.service import (
    LOCKOUT_DURATION,
    MAX_FAILED_LOGINS,
    AccountError,
    AuthenticationError,
    authenticate,
    can_manage,
    can_report_prices,
    create_user,
)
from agrodash.infra.db import create_db_engine, upgrade_database
from agrodash.infra.settings import get_settings

PASSWORD = "correct horse battery"
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


@pytest.fixture
def user(session):
    return create_user(session, "Tecnico@Example.com ", PASSWORD, Role.TECNICO)


def test_create_user_normalizes_email_and_hashes_password(session, user):
    assert user.email == "tecnico@example.com"
    assert user.password_hash.startswith("$argon2id$")
    assert PASSWORD not in user.password_hash
    assert session.get(User, user.id).role is Role.TECNICO


def test_create_user_rejects_duplicate_email_regardless_of_case(session, user):
    with pytest.raises(AccountError, match="already registered"):
        create_user(session, "TECNICO@example.com", PASSWORD, Role.AGRICULTOR)


def test_create_user_rejects_short_passwords(session):
    with pytest.raises(AccountError, match="at least 12 characters"):
        create_user(session, "a@example.com", "short-pass", Role.ADMIN)


def test_create_user_rejects_invalid_email(session):
    with pytest.raises(AccountError, match="valid email"):
        create_user(session, "not-an-email", PASSWORD, Role.ADMIN)


def test_authenticate_returns_user_and_records_login(session, user):
    logged_in = authenticate(session, "TECNICO@example.com", PASSWORD, now=NOW)

    assert logged_in.id == user.id
    assert logged_in.failed_login_attempts == 0
    assert logged_in.last_login_at.replace(tzinfo=UTC) == NOW


@pytest.mark.parametrize(
    ("email", "password"),
    [("tecnico@example.com", "wrong password!!"), ("nobody@example.com", PASSWORD)],
)
def test_wrong_credentials_get_the_same_generic_error(session, user, email, password):
    with pytest.raises(AuthenticationError, match="^Invalid email or password$"):
        authenticate(session, email, password, now=NOW)


def test_failed_login_is_counted(session, user):
    with pytest.raises(AuthenticationError):
        authenticate(session, user.email, "wrong password!!", now=NOW)

    assert session.get(User, user.id).failed_login_attempts == 1


def test_successful_login_resets_failed_attempts(session, user):
    with pytest.raises(AuthenticationError):
        authenticate(session, user.email, "wrong password!!", now=NOW)

    authenticate(session, user.email, PASSWORD, now=NOW)

    assert session.get(User, user.id).failed_login_attempts == 0


def test_account_locks_after_repeated_failures_until_lockout_expires(session, user):
    for _ in range(MAX_FAILED_LOGINS):
        with pytest.raises(AuthenticationError):
            authenticate(session, user.email, "wrong password!!", now=NOW)

    with pytest.raises(AuthenticationError, match="^Invalid email or password$"):
        authenticate(session, user.email, PASSWORD, now=NOW + LOCKOUT_DURATION / 2)

    later = NOW + LOCKOUT_DURATION + timedelta(seconds=1)
    assert authenticate(session, user.email, PASSWORD, now=later).id == user.id


def test_inactive_user_cannot_log_in(session, user):
    user.is_active = False
    session.commit()

    with pytest.raises(AuthenticationError, match="^Invalid email or password$"):
        authenticate(session, user.email, PASSWORD, now=NOW)


@pytest.mark.parametrize(
    ("role", "reports", "manages"),
    [
        (Role.AGRICULTOR, False, False),
        (Role.TECNICO, True, False),
        (Role.ADMIN, True, True),
        (Role.SUPERADMIN, True, True),
    ],
)
def test_role_permissions(role, reports, manages):
    assert can_report_prices(role) is reports
    assert can_manage(role) is manages


@pytest.fixture
def cli_database(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    upgrade_database(url)
    monkeypatch.setenv("AGRODASH_DATABASE_URL", url)
    get_settings.cache_clear()
    yield create_db_engine(url)
    get_settings.cache_clear()


def test_create_admin_command_reads_password_from_stdin(cli_database, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(PASSWORD + "\n"))

    exit_code = cli.main(["create-admin", "--email", "Owner@Example.com", "--superadmin"])

    assert exit_code == 0
    assert "Created SUPERADMIN owner@example.com" in capsys.readouterr().out
    from sqlalchemy.orm import Session

    with Session(cli_database) as session:
        assert authenticate(session, "owner@example.com", PASSWORD, now=NOW).role is Role.SUPERADMIN


def test_create_admin_command_reports_errors(cli_database, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))

    exit_code = cli.main(["create-admin", "--email", "admin@example.com"])

    assert exit_code == 1
    assert "at least 12 characters" in capsys.readouterr().err
