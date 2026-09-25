import asyncio
import logging

from mlops_labs.domain.system import DependencyHealth, HealthCheck, SystemHealth

logger = logging.getLogger(__name__)


class SystemHealthCheck:
    def __init__(self, checks: list[HealthCheck]) -> None:
        self._checks = checks

    async def check(self) -> SystemHealth:
        if not self._checks:
            return SystemHealth(healthy=True, checks=())

        tasks: list[asyncio.Task[DependencyHealth]] = []
        async with asyncio.TaskGroup() as tg:
            for check in self._checks:
                tasks.append(tg.create_task(self._safe_check(check)))

        results = tuple(task.result() for task in tasks)
        is_healthy = all(item.healthy for item in results)
        return SystemHealth(healthy=is_healthy, checks=results)

    @staticmethod
    async def _safe_check(check: HealthCheck) -> DependencyHealth:
        """Isolate probe execution so a failing probe does not abort the TaskGroup."""
        try:
            return await check.check()
        except Exception as exc:
            name = getattr(check, "name", check.__class__.__name__)
            logger.error(
                "dependency.check_failed",
                extra={"dependency": name, "error_type": type(exc).__name__},
            )
            return DependencyHealth(
                name=name,
                healthy=False,
                latency_ms=0.0,
                error="dependency check failed",
            )
