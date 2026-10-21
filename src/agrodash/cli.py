import argparse
import getpass
import sys

from sqlalchemy.orm import Session

from agrodash.accounts.models import Role
from agrodash.accounts.service import AccountError, create_user
from agrodash.infra.db import create_db_engine
from agrodash.infra.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agrodash", description="AgroDashRD management tasks")
    commands = parser.add_subparsers(dest="command", required=True)

    create_admin = commands.add_parser(
        "create-admin",
        help="create an ADMIN account (password read from stdin when not interactive)",
    )
    create_admin.add_argument("--email", required=True)
    create_admin.add_argument("--full-name")
    create_admin.add_argument(
        "--superadmin", action="store_true", help="create a SUPERADMIN instead of an ADMIN"
    )

    args = parser.parse_args(argv)
    if args.command == "create-admin":
        return _create_admin(args)
    return 2


def _create_admin(args: argparse.Namespace) -> int:
    role = Role.SUPERADMIN if args.superadmin else Role.ADMIN
    try:
        password = _read_password()
        engine = create_db_engine(get_settings().database_url)
        with Session(engine) as session:
            user = create_user(session, args.email, password, role, args.full_name)
            created = f"Created {user.role} {user.email}"
        engine.dispose()
    except AccountError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(created)
    return 0


def _read_password() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.readline().rstrip("\n")
    password = getpass.getpass("Password: ")
    if getpass.getpass("Repeat password: ") != password:
        raise AccountError("Passwords do not match")
    return password


if __name__ == "__main__":
    sys.exit(main())
