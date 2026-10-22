from enum import StrEnum

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, Enum, MetaData, create_engine, event
from sqlalchemy.orm import DeclarativeBase

# Deterministic constraint names keep Alembic migrations portable across SQLite and PostgreSQL.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

MIGRATIONS_LOCATION = "agrodash.infra:migrations"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def enum_column(enum_class: type[StrEnum], name: str) -> Enum:
    """Portable enum stored as text with a CHECK constraint on its values."""
    return Enum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
        values_callable=lambda members: [member.value for member in members],
    )


def create_db_engine(url: str) -> Engine:
    engine = create_engine(url)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def _enable_sqlite_foreign_keys(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def alembic_config(url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", MIGRATIONS_LOCATION)
    # ConfigParser treats "%" as interpolation; URL-encoded passwords need it escaped.
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def upgrade_database(url: str) -> None:
    command.upgrade(alembic_config(url), "head")
