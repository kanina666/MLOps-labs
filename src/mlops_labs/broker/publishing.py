import asyncio
from time import time_ns
from typing import Any

from aio_pika.exceptions import DeliveryError
from faststream.rabbit import RabbitBroker
from opentelemetry import propagate, trace
from pamqp.commands import Basic

from mlops_labs.core.config import Settings
from mlops_labs.core.telemetry import batch_id_context, error_on_span, tracer


def publish_failure(exc: BaseException) -> str:
    if isinstance(exc, DeliveryError) and isinstance(exc.frame, Basic.Nack):
        return "nack"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "broker_error"


async def publish_task(
    broker: RabbitBroker,
    settings: Settings,
    task_id: str,
    payload: dict[str, Any],
) -> None:
    with tracer.start_as_current_span(
        "publish",
        kind=trace.SpanKind.PRODUCER,
        attributes={
            "messaging.system": "rabbitmq",
            "messaging.destination.name": settings.rabbitmq_queue,
        },
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        headers = {"published_at_unix_ns": str(time_ns())}
        if batch_id := batch_id_context.get():
            headers["batch_id"] = batch_id
        propagate.inject(headers)
        try:
            async with asyncio.timeout(settings.publish_timeout_seconds):
                await broker.publish(
                    {"task_id": task_id, "payload": payload},
                    queue=settings.rabbitmq_queue,
                    correlation_id=task_id,
                    headers={**headers},
                    persist=True,
                    mandatory=True,
                    timeout=settings.publish_timeout_seconds,
                )
        except Exception as exc:
            span.set_attribute("messaging.publish.result", publish_failure(exc))
            error_on_span(span, exc)
            raise
