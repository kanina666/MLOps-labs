from faststream.rabbit import ExchangeType, RabbitBroker, RabbitExchange, RabbitQueue
from faststream.rabbit.schemas.queue import ClassicQueueArgs

from mlops_labs.core.config import Settings


def build_tasks_exchange(settings: Settings) -> RabbitExchange:
    return RabbitExchange(
        name=settings.rabbitmq_exchange,
        type=ExchangeType.DIRECT,
        durable=True,
    )


def build_dlx_exchange(settings: Settings) -> RabbitExchange:
    return RabbitExchange(
        name=settings.rabbitmq_dlx_exchange,
        type=ExchangeType.DIRECT,
        durable=True,
    )


def build_tasks_queue(settings: Settings) -> RabbitQueue:
    arguments: ClassicQueueArgs = {
        "x-overflow": "reject-publish",
        "x-dead-letter-exchange": settings.rabbitmq_dlx_exchange,
        "x-dead-letter-routing-key": settings.rabbitmq_dlq_queue,
    }
    if settings.rabbitmq_max_queue_length:
        arguments["x-max-length"] = settings.rabbitmq_max_queue_length
    return RabbitQueue(
        name=settings.rabbitmq_queue,
        durable=True,
        arguments=arguments,
    )


def build_dlq_queue(settings: Settings) -> RabbitQueue:
    return RabbitQueue(
        name=settings.rabbitmq_dlq_queue,
        durable=True,
    )


async def declare_topology(broker: RabbitBroker, settings: Settings) -> None:
    """Declare exchanges and queues (including DLX and DLQ) on the broker."""
    dlx = await broker.declare_exchange(build_dlx_exchange(settings))
    dlq = await broker.declare_queue(build_dlq_queue(settings))
    if hasattr(dlq, "bind"):
        await dlq.bind(dlx, routing_key=settings.rabbitmq_dlq_queue)

    await broker.declare_queue(build_tasks_queue(settings))
