import re
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
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
    try:
        session.commit()
    except IntegrityError as error:
        # Another request registered the same email between the check and the insert.
        session.rollback()
        raise AccountError(f"{email} is already registered") from error
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


# Who an account may create, deactivate or unlock. Nobody manages accounts above them.
_MANAGEABLE = {
    Role.TECNICO: [],
    Role.ADMIN: [Role.TECNICO],
    Role.SUPERADMIN: [Role.TECNICO, Role.ADMIN, Role.SUPERADMIN],
}
_RANK = {Role.SUPERADMIN: 0, Role.ADMIN: 1, Role.TECNICO: 2}


def manageable_roles(role: Role) -> list[Role]:
    return list(_MANAGEABLE[role])


def list_users(session: Session) -> list[User]:
    """Every account, highest role first, then by email."""
    return sorted(session.scalars(select(User)), key=lambda user: (_RANK[user.role], user.email))


def create_account(
    session: Session,
    actor: User,
    email: str,
    password: str,
    role: Role,
    full_name: str | None = None,
) -> User:
    _check_manager(actor)
    if role not in manageable_roles(actor.role):
        raise AccountError(f"You cannot create {role} accounts")
    return create_user(session, email, password, role, full_name)


def set_active(session: Session, actor: User, user_id: int, *, active: bool) -> User:
    user = _managed(session, actor, user_id)
    if user.id == actor.id and not active:
        raise AccountError("You cannot deactivate your own account")
    user.is_active = active
    session.commit()
    return user


def unlock(session: Session, actor: User, user_id: int) -> User:
    user = _managed(session, actor, user_id)
    user.locked_until = None
    user.failed_login_attempts = 0
    session.commit()
    return user


def _check_manager(actor: User) -> None:
    # A deactivated admin may still have a page open; refuse it here, not only on pages.
    if not (actor.is_active and can_manage(actor.role)):
        raise AccountError("Your account cannot manage accounts")


def _managed(session: Session, actor: User, user_id: int) -> User:
    _check_manager(actor)
    user = session.get(User, user_id)
    if user is None:
        raise AccountError("Unknown account")
    if user.role not in manageable_roles(actor.role):
        raise AccountError(f"You cannot manage {user.role} accounts")
    return user
