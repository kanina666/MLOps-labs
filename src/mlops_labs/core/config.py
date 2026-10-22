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

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_uri: str = "models:/m5_sales_model@champion"

    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    rabbitmq_queue: str = "mlops.tasks"
    rabbitmq_dlq_queue: str = "mlops.tasks.dlq"
    rabbitmq_exchange: str = "mlops.direct"
    rabbitmq_dlx_exchange: str = "mlops.dlx"
    rabbitmq_routing_key: str = "task.predict"

    rabbitmq_prefetch_count: int = 1
    rabbitmq_max_queue_length: int = 10

    # increase in tests to simulate longer inference time and test retry mechanism
    inference_delay_ms: float = 0.0
    max_retries: int = 3
    retry_after_seconds: int = 5

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
