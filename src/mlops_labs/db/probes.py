import logging
import time

from sqlalchemy.ext.asyncio import AsyncEngine

from mlops_labs.db.base import ping_postgres
from mlops_labs.domain.system import DependencyHealth

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class PostgresHealthCheck:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    async def check(self) -> DependencyHealth:
        start = time.perf_counter()
        try:
            version = await ping_postgres(self.engine)
        except Exception as exc:
            logger.warning(
                "dependency.unavailable",
                extra={"dependency": "postgres", "error_type": type(exc).__name__},
            )
            return DependencyHealth(
                name="postgres",
                healthy=False,
                latency_ms=_elapsed_ms(start),
                error="postgres is unavailable",
            )
        return DependencyHealth(
            name="postgres", healthy=True, version=version, latency_ms=_elapsed_ms(start)
        )
