from sqlalchemy import create_engine, inspect

from agrodash import cli
from agrodash.infra.settings import get_settings


def test_migrate_creates_the_schema_and_can_run_again(tmp_path, monkeypatch, capsys):
    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    monkeypatch.setenv("AGRODASH_DATABASE_URL", url)
    get_settings.cache_clear()

    assert cli.main(["migrate"]) == 0
    assert cli.main(["migrate"]) == 0

    tables = set(inspect(create_engine(url)).get_table_names())
    assert {"products", "prices", "users", "forecasts"} <= tables
    assert capsys.readouterr().out.count("Database schema is up to date") == 2
    get_settings.cache_clear()
