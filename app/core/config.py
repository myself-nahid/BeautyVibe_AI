"""
app/core/config.py
──────────────────
Centralised, type-safe settings powered by pydantic-settings.
All environment variables are validated at startup; missing required
values raise an immediate, descriptive error instead of a silent None.
"""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_ENV: str = "development"
    APP_SECRET_KEY: str = "change-me-in-production"
    APP_DEBUG: bool = True

    DATABASE_URL: str = "sqlite:///./glowflow.db"

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"

    MAX_IMAGE_SIZE_MB: int = 5
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]

    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def openai_key_must_not_be_empty(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to your .env file."
            )
        return v

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def max_image_bytes(self) -> int:
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.DATABASE_URL


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    return Settings()