from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The official wholesale (.xlsx) and retail (.xls) reports.
OFFICIAL_DRIVE_FILE_IDS = "17K3F_hI0OV_eWwNFo0WESPsxrnhuQKKK,1lUxB0FdL7PEs4c-jgQ1J8RS0jDhWqaum"


class Settings(BaseSettings):
    """Runtime configuration, read from AGRODASH_* environment variables or `.env`."""

    model_config = SettingsConfigDict(env_prefix="AGRODASH_", env_file=".env", extra="ignore")

    env: Literal["development", "production"] = "development"
    database_url: str = "sqlite:///agrodash.db"
    # Public Google Drive files loaded by `agrodash ingest --drive`, comma-separated:
    # the official wholesale (.xlsx) and retail (.xls) price reports.
    drive_file_ids: str = OFFICIAL_DRIVE_FILE_IDS

    @field_validator("drive_file_ids")
    @classmethod
    def _default_when_blank(cls, value: str) -> str:
        # docker-compose passes an empty value when .env does not set the variable.
        return value if value.strip() else OFFICIAL_DRIVE_FILE_IDS

    @property
    def drive_file_id_list(self) -> list[str]:
        return [file_id.strip() for file_id in self.drive_file_ids.split(",") if file_id.strip()]

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, value: str) -> str:
        value = value.strip()
        # Hosts often inject "postgres://" (rejected by SQLAlchemy 2) or a driverless
        # "postgresql://" (which would load psycopg2); pin the installed psycopg 3 driver.
        for prefix in ("postgres://", "postgresql://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value.removeprefix(prefix)
        return value

    @model_validator(mode="after")
    def _require_postgresql_in_production(self) -> "Settings":
        scheme = self.database_url.partition("://")[0].lower()
        if self.env == "production" and scheme.split("+")[0] != "postgresql":
            raise ValueError(
                "AGRODASH_DATABASE_URL must point to PostgreSQL when AGRODASH_ENV=production"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
