import pytest
from pydantic import ValidationError

from agrodash.infra.settings import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("AGRODASH_ENV", "AGRODASH_DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)


def load() -> Settings:
    return Settings(_env_file=None)


def test_development_defaults_to_local_sqlite():
    settings = load()

    assert settings.env == "development"
    assert settings.database_url.startswith("sqlite")


def test_reads_values_from_environment(monkeypatch):
    monkeypatch.setenv("AGRODASH_ENV", "production")
    monkeypatch.setenv("AGRODASH_DATABASE_URL", "postgresql+psycopg://u:p@db:5432/agrodash")

    settings = load()

    assert settings.env == "production"
    assert settings.database_url == "postgresql+psycopg://u:p@db:5432/agrodash"


def test_production_refuses_to_start_without_a_real_database(monkeypatch):
    monkeypatch.setenv("AGRODASH_ENV", "production")

    with pytest.raises(ValidationError, match="AGRODASH_DATABASE_URL must point to PostgreSQL"):
        load()


def test_rejects_unknown_environment(monkeypatch):
    monkeypatch.setenv("AGRODASH_ENV", "staging-typo")

    with pytest.raises(ValidationError):
        load()
