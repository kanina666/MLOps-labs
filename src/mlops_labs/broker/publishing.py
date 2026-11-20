import asyncio
from typing import Any

from faststream.rabbit import RabbitBroker

from mlops_labs.core.config import Settings


async def publish_task(
    broker: RabbitBroker,
    settings: Settings,
    task_id: str,
    payload: dict[str, Any],
) -> None:
    async with asyncio.timeout(settings.publish_timeout_seconds):
        await broker.publish(
            {"task_id": task_id, "payload": payload},
            queue=settings.rabbitmq_queue,
            persist=True,
            mandatory=True,
            timeout=settings.publish_timeout_seconds,
        )
