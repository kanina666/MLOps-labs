import asyncio
import logging
import time

from mlops_labs.db.base import ping_postgres
from mlops_labs.domain.system import DependencyHealth, SystemHealth

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


async def _check_postgres() -> DependencyHealth:
    start = time.perf_counter()
    try:
        version = await ping_postgres()
    except Exception:
        return DependencyHealth(
            name="postgres",
            healthy=False,
            latency_ms=_elapsed_ms(start),
            error="postgres is unavailable",
        )
    return DependencyHealth(
        name="postgres", healthy=True, version=version, latency_ms=_elapsed_ms(start)
    )


async def check_system() -> SystemHealth:
    results = await asyncio.gather(
        _check_postgres(),
    )

    return SystemHealth(healthy=all(item.healthy for item in results), checks=tuple(results))
