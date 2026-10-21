import os

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from agrodash.infra.db import create_db_engine, upgrade_database

# CI sets this to a disposable PostgreSQL database; locally the PostgreSQL runs are skipped.
POSTGRES_URL = os.getenv("AGRODASH_TEST_POSTGRES_URL")


@pytest.fixture(params=["sqlite", "postgresql"])
def database_url(request, tmp_path):
    if request.param == "sqlite":
        return f"sqlite:///{tmp_path / 'test.db'}"
    if not POSTGRES_URL:
        pytest.skip("AGRODASH_TEST_POSTGRES_URL is not set")
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
