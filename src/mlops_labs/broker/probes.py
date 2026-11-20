import logging
import time

import aio_pika

from mlops_labs.domain.system import DependencyHealth

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class RabbitMQHealthCheck:
    def __init__(self, url: str, timeout: float = 2.0) -> None:
        self.url = url
        self.timeout = timeout

    async def check(self) -> DependencyHealth:
        start = time.perf_counter()
        try:
            connection = await aio_pika.connect(self.url, timeout=self.timeout)
            async with connection:
                server_props = getattr(connection, "server_properties", {})
                raw_ver = server_props.get("version") if isinstance(server_props, dict) else None
                version = str(raw_ver) if raw_ver is not None else None
                return DependencyHealth(
                    name="rabbitmq",
                    healthy=True,
                    version=version,
                    latency_ms=_elapsed_ms(start),
                )
        except Exception as exc:
            logger.warning(
                "dependency.unavailable",
                extra={"dependency": "rabbitmq", "error_type": type(exc).__name__},
            )
            return DependencyHealth(
                name="rabbitmq",
                healthy=False,
                latency_ms=_elapsed_ms(start),
                error="rabbitmq is unavailable",
            )
