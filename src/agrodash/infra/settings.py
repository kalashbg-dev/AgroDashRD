from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from AGRODASH_* environment variables or `.env`."""

    model_config = SettingsConfigDict(env_prefix="AGRODASH_", env_file=".env", extra="ignore")

    env: Literal["development", "production"] = "development"
    database_url: str = "sqlite:///agrodash.db"

    @model_validator(mode="after")
    def _require_real_database_in_production(self) -> "Settings":
        if self.env == "production" and self.database_url.startswith("sqlite"):
            raise ValueError(
                "AGRODASH_DATABASE_URL must point to PostgreSQL when AGRODASH_ENV=production"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
