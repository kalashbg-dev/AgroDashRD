import re
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from agrodash.accounts.models import Role, User

MIN_PASSWORD_LENGTH = 12
MAX_FAILED_LOGINS = 5
LOCKOUT_DURATION = timedelta(minutes=15)

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_hasher = PasswordHasher()


class AccountError(ValueError):
    """Invalid account data, safe to show to the person creating the account."""


class AuthenticationError(Exception):
    """Login refused. The message never reveals whether the email exists."""

    def __init__(self) -> None:
        super().__init__("Invalid email or password")


def can_report_prices(role: Role) -> bool:
    return role in {Role.TECNICO, Role.ADMIN, Role.SUPERADMIN}


def can_manage(role: Role) -> bool:
    return role in {Role.ADMIN, Role.SUPERADMIN}


def normalize_email(email: str) -> str:
    return email.strip().lower()


def create_user(
    session: Session, email: str, password: str, role: Role, full_name: str | None = None
) -> User:
    email = normalize_email(email)
    if not _EMAIL_PATTERN.match(email):
        raise AccountError("Enter a valid email address")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AccountError(f"Password must have at least {MIN_PASSWORD_LENGTH} characters")
    if _find_by_email(session, email):
        raise AccountError(f"{email} is already registered")

    user = User(email=email, password_hash=_hasher.hash(password), role=role, full_name=full_name)
    session.add(user)
    session.commit()
    return user


def authenticate(session: Session, email: str, password: str, now: datetime | None = None) -> User:
    now = now or datetime.now(UTC)
    user = _find_by_email(session, normalize_email(email))
    if user is None or not user.is_active or _is_locked(user, now):
        # Spend the same hashing time as a real check so timing does not reveal accounts.
        _verify(_dummy_hash(), password)
        raise AuthenticationError

    if not _verify(user.password_hash, password):
        _record_failed_login(session, user, now)
        raise AuthenticationError

    if _hasher.check_needs_rehash(user.password_hash):
        user.password_hash = _hasher.hash(password)
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = now
    session.commit()
    return user


def _find_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(User.email == email))


def _record_failed_login(session: Session, user: User, now: datetime) -> None:
    # Increment in SQL so concurrent failed logins cannot overwrite each other's count.
    attempts = session.scalar(
        update(User)
        .where(User.id == user.id)
        .values(failed_login_attempts=User.failed_login_attempts + 1)
        .returning(User.failed_login_attempts)
    )
    if attempts >= MAX_FAILED_LOGINS:
        session.execute(
            update(User)
            .where(User.id == user.id)
            .values(locked_until=now + LOCKOUT_DURATION, failed_login_attempts=0)
        )
    session.commit()


def _is_locked(user: User, now: datetime) -> bool:
    if user.locked_until is None:
        return False
    # SQLite returns naive datetimes; values are always stored in UTC.
    locked_until = user.locked_until.replace(tzinfo=user.locked_until.tzinfo or UTC)
    return now < locked_until


def _verify(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


@lru_cache
def _dummy_hash() -> str:
    return _hasher.hash("timing-equalizer")
