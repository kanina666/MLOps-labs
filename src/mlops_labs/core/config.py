from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/mlops_labs"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800

    app_version_override: str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @property
    def app_version(self) -> str:
        if self.app_version_override:
            return self.app_version_override
        try:
            return version("mlops-labs")
        except PackageNotFoundError:
            return "0.0.0-dev"


@cache
def get_settings() -> Settings:
    return Settings()
