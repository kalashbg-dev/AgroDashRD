import argparse
import getpass
import sys
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from agrodash.accounts.models import Role
from agrodash.accounts.service import AccountError, create_user
from agrodash.infra.db import create_db_engine
from agrodash.infra.settings import get_settings
from agrodash.ingestion.drive import DriveError, download
from agrodash.ingestion.models import IngestionTrigger
from agrodash.ingestion.service import IngestionError, ingest_file, loaded_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agrodash", description="AgroDashRD management tasks")
    commands = parser.add_subparsers(dest="command", required=True)

    create_admin_help = "create an ADMIN account (password read from stdin when not interactive)"
    create_admin = commands.add_parser(
        "create-admin", help=create_admin_help, description=_sentence(create_admin_help)
    )
    create_admin.add_argument("--email", required=True)
    create_admin.add_argument("--full-name")
    create_admin.add_argument(
        "--superadmin", action="store_true", help="create a SUPERADMIN instead of an ADMIN"
    )

    ingest_help = "load official price reports (.csv, .xlsx, .xls) from files or Google Drive"
    ingest = commands.add_parser("ingest", help=ingest_help, description=_sentence(ingest_help))
    ingest.add_argument("files", nargs="*", help="report files to load")
    ingest.add_argument(
        "--drive", action="store_true", help="download and load AGRODASH_DRIVE_FILE_IDS"
    )

    args = parser.parse_args(argv)
    if args.command == "ingest" and not (args.files or args.drive):
        ingest.error("give files to load or --drive")
    try:
        if args.command == "create-admin":
            return _create_admin(args)
        return _ingest(args)
    except ValidationError as error:
        # Only the messages: pydantic's full report would echo the database URL and its password.
        messages = "; ".join(detail["msg"] for detail in error.errors())
        print(f"error: invalid configuration: {messages}", file=sys.stderr)
        return 1
    except SQLAlchemyError:
        print(
            "error: database is unreachable or not migrated (run 'alembic upgrade head')",
            file=sys.stderr,
        )
        return 1


def _sentence(text: str) -> str:
    # str.capitalize() would also lowercase "ADMIN" and "Google Drive".
    return text[0].upper() + text[1:]


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
        return sys.stdin.readline().rstrip("\r\n")
    password = getpass.getpass("Password: ")
    if getpass.getpass("Repeat password: ") != password:
        raise AccountError("Passwords do not match")
    return password


def _ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    ok = True
    with Session(engine) as session:
        for path in map(Path, args.files):
            try:
                content = path.read_bytes()
            except FileNotFoundError:
                print(f"error: {path.name}: file not found", file=sys.stderr)
                ok = False
                continue
            except OSError as error:
                print(f"error: {path.name}: cannot read file ({error.strerror})", file=sys.stderr)
                ok = False
                continue
            ok &= _ingest_one(session, path.name, content, IngestionTrigger.FILE)
        if args.drive and not settings.drive_file_id_list:
            print("error: AGRODASH_DRIVE_FILE_IDS is empty", file=sys.stderr)
            ok = False
        if args.drive:
            for file_id in settings.drive_file_id_list:
                try:
                    file_name, content = download(file_id)
                except DriveError as error:
                    print(f"error: {error}", file=sys.stderr)
                    ok = False
                    continue
                ok &= _ingest_one(session, file_name, content, IngestionTrigger.DRIVE)
    engine.dispose()
    return 0 if ok else 1


def _ingest_one(session: Session, file_name: str, content: bytes, trigger) -> bool:
    if run := loaded_run(session, content):
        print(f"{file_name}: already loaded (run {run.id})")
        return True
    try:
        run = ingest_file(session, content, file_name, trigger)
    except IngestionError as error:
        print(f"error: {file_name}: {error}", file=sys.stderr)
        return False
    print(
        f"{file_name}: loaded {run.prices_loaded} prices, quarantined {run.rows_rejected} rows, "
        f"{run.values_missing} missing values (run {run.id})"
    )
    return True


if __name__ == "__main__":
    sys.exit(main())
