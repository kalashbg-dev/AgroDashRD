import pytest
from sqlalchemy import create_engine, inspect

from agrodash import cli
from agrodash.infra.settings import get_settings


@pytest.fixture(autouse=True)
def fresh_settings():
    # Settings are cached per process: clear them around every test, even a failing one.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_migrate_creates_the_schema_and_can_run_again(tmp_path, monkeypatch, capsys):
    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    monkeypatch.setenv("AGRODASH_DATABASE_URL", url)

    assert cli.main(["migrate"]) == 0
    assert cli.main(["migrate"]) == 0

    tables = set(inspect(create_engine(url)).get_table_names())
    assert {"products", "prices", "users", "forecasts"} <= tables
    assert capsys.readouterr().out.count("Database schema is up to date") == 2


def test_migration_errors_are_one_line_not_a_traceback(monkeypatch, capsys):
    from alembic.util import CommandError

    def fail(url):
        raise CommandError("Can't locate revision identified by 'abc123'")

    monkeypatch.setattr(cli, "upgrade_database", fail)

    assert cli.main(["migrate"]) == 1
    assert capsys.readouterr().err == (
        "error: cannot migrate the database: Can't locate revision identified by 'abc123'\n"
    )


def test_an_unmigrated_database_points_to_agrodash_migrate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGRODASH_DATABASE_URL", f"sqlite:///{tmp_path / 'empty.db'}")

    assert cli.main(["forecast", "--skip-benchmark"]) == 1
    assert "run 'agrodash migrate'" in capsys.readouterr().err
