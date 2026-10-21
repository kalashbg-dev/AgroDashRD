from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from AGRODASH_* environment variables or `.env`."""

    model_config = SettingsConfigDict(env_prefix="AGRODASH_", env_file=".env", extra="ignore")

    env: Literal["development", "production"] = "development"
    database_url: str = "sqlite:///agrodash.db"

    @field_validator("database_url")
    @classmethod
    def _strip_database_url(cls, value: str) -> str:
        return value.strip()

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
