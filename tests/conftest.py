import os

import pytest
from sqlalchemy import make_url, text
from sqlalchemy.orm import Session

from agrodash.infra.db import create_db_engine, upgrade_database
from agrodash.infra.settings import get_settings

# CI sets this to a disposable PostgreSQL database; locally the PostgreSQL runs are skipped.
POSTGRES_URL = os.getenv("AGRODASH_TEST_POSTGRES_URL")


@pytest.fixture(params=["sqlite", "postgresql"])
def database_url(request, tmp_path):
    if request.param == "sqlite":
        return f"sqlite:///{tmp_path / 'test.db'}"
    if not POSTGRES_URL:
        pytest.skip("AGRODASH_TEST_POSTGRES_URL is not set")
    # Each test wipes the schema, so only accept databases that are clearly disposable.
    if not (make_url(POSTGRES_URL).database or "").endswith("_test"):
        pytest.fail("AGRODASH_TEST_POSTGRES_URL must name a database ending in '_test'")
    engine = create_db_engine(POSTGRES_URL)
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    return POSTGRES_URL


@pytest.fixture
def engine(database_url):
    upgrade_database(database_url)
    engine = create_db_engine(database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


@pytest.fixture
def cli_database(tmp_path, monkeypatch):
    """A migrated SQLite database that the `agrodash` CLI reads from settings."""
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    upgrade_database(url)
    monkeypatch.setenv("AGRODASH_DATABASE_URL", url)
    get_settings.cache_clear()
    engine = create_db_engine(url)
    yield engine
    engine.dispose()
    get_settings.cache_clear()
