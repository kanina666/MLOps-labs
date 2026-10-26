from functools import cache
from importlib.metadata import PackageNotFoundError, version
from math import ceil
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

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

    # Zero disables the limit for the unbounded benchmark
    rabbitmq_prefetch_count: int = Field(default=1, ge=0)
    # Buffer: 2 workers * ceil(1s retry pause / 0.2s inference) = 10
    # Recheck changed settings with benchmark_backpressure --capacity-study
    rabbitmq_max_queue_length: int = Field(default=10, ge=0)
    worker_concurrency: int = Field(default=1, ge=1)
    worker_health_port: int = Field(default=8000, ge=1, le=65535)

    inference_delay_ms: float = Field(default=200.0, ge=0)
    max_retries: int = Field(default=3, ge=1)
    retry_delay_seconds: float = Field(default=0.2, gt=0)
    publish_timeout_seconds: float = Field(default=0.5, gt=0)
    source_retry_delay_seconds: float = Field(default=0.05, gt=0)
    source_retry_max_delay_seconds: float = Field(default=1.0, gt=0)
    retry_after_seconds: int | None = Field(default=None, ge=1)

    @property
    def effective_retry_after_seconds(self) -> int:
        if self.retry_after_seconds is not None:
            return self.retry_after_seconds
        return max(1, ceil(self.inference_delay_ms / 1000 / self.worker_concurrency))

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
