import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import time_ns
from typing import Annotated, Any

from faststream import AckPolicy, Context, ContextRepo, Depends, FastStream
from faststream.asgi import AsgiFastStream
from faststream.exceptions import NackMessage, RejectMessage
from faststream.rabbit import Channel, RabbitBroker, RabbitMessage, RabbitRouter
from opentelemetry import propagate, trace
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from mlops_labs.broker.topology import build_tasks_queue, declare_topology
from mlops_labs.core.config import Settings, get_settings
from mlops_labs.core.logging import configure_logging
from mlops_labs.core.telemetry import (
    error_on_span,
    start_telemetry,
    stop_telemetry,
    task_context,
    task_duration,
    tracer,
    worker_inflight,
    worker_redeliveries,
    worker_tasks,
)
from mlops_labs.db.base import create_db_engine, create_session_factory, init_db
from mlops_labs.db.task_repository import TaskRepository
from mlops_labs.domain.tasks import TaskPayload
from mlops_labs.services.model import ModelPredictor, ModelUnavailableError, load_model

logger = logging.getLogger(__name__)
TASK_ID = re.compile(r"[a-zA-Z0-9_-]{1,36}")


class WorkerTaskPayloadSchema(BaseModel):
    """Schema for task payload inside the queue message."""

    recent_sales: list[float]
    day_of_week: int = Field(..., ge=0, le=6)
    is_snap: int = Field(default=0, ge=0, le=1)


class TaskEnvelope(BaseModel):
    """Message contract for tasks consumed from RabbitMQ."""

    task_id: str = Field(..., min_length=1)
    payload: WorkerTaskPayloadSchema


async def get_task_repository(
    session_factory: Annotated[async_sessionmaker[AsyncSession], Context("session_factory")],
) -> AsyncGenerator[TaskRepository, None]:
    """Provide a scoped TaskRepository for a single message execution."""
    async with session_factory() as session:
        yield TaskRepository(session)


async def handle_task_message(
    body: dict[str, Any],
    repo: TaskRepository,
    predictor: ModelPredictor,
    worker_settings: Settings,
    raw_message: RabbitMessage | None = None,
) -> None:
    """Core message handler orchestrating validation, idempotency, inference, and DLQ routing."""
    headers = dict(raw_message.headers) if raw_message else {}
    correlation_id = raw_message.correlation_id if raw_message else None
    candidate = body.get("task_id") if isinstance(body, dict) else None
    body_task_id = candidate if isinstance(candidate, str) and TASK_ID.fullmatch(candidate) else ""
    task_id = body_task_id or (
        correlation_id
        if isinstance(correlation_id, str) and TASK_ID.fullmatch(correlation_id)
        else None
    )
    batch_id = headers.get("batch_id")
    redelivered = bool(raw_message and raw_message.raw_message.redelivered)
    published_at = str(headers.get("published_at_unix_ns", ""))
    published_ns = (
        int(published_at)
        if published_at.isascii() and published_at.isdigit() and len(published_at) <= 20
        else None
    )

    with (
        task_context(task_id, batch_id if isinstance(batch_id, str) else None),
        tracer.start_as_current_span(
            "process",
            context=propagate.extract(headers),
            kind=trace.SpanKind.CONSUMER,
            attributes={
                "messaging.system": "rabbitmq",
                "messaging.destination.name": worker_settings.rabbitmq_queue,
                "messaging.rabbitmq.redelivered": redelivered,
            },
            record_exception=False,
            set_status_on_exception=False,
        ) as span,
    ):
        if task_id:
            span.set_attribute("task.id", task_id)
        if redelivered:
            worker_redeliveries.add(1)
        if published_ns is not None and not redelivered:
            queue_wait = max(0.0, (time_ns() - published_ns) / 1_000_000_000)
            span.set_attribute("messaging.rabbitmq.queue_wait_seconds", queue_wait)
            logger.info("task.received", extra={"queue_wait_seconds": queue_wait})
        else:
            logger.info("task.received")

        worker_inflight.add(1)
        try:
            await process_payload(body, body_task_id, repo, predictor, worker_settings, span)
        except NackMessage:
            worker_tasks.add(1, {"outcome": "error"})
            raise
        except RejectMessage:
            worker_tasks.add(1, {"outcome": "dlq"})
            span.add_event(
                "task.dead_letter",
                {"messaging.destination.name": worker_settings.rabbitmq_dlq_queue},
            )
            logger.error("task.dead_letter", extra={"attempt": repo.processing_attempts or 1})
            raise
        except Exception as exc:
            error_on_span(span, exc)
            worker_tasks.add(1, {"outcome": "dlq"})
            span.add_event(
                "task.dead_letter",
                {"messaging.destination.name": worker_settings.rabbitmq_dlq_queue},
            )
            logger.exception("task.dead_letter", extra={"attempt": repo.processing_attempts or 1})
            raise
        finally:
            worker_inflight.add(-1)
        if repo.processing_attempts:
            worker_tasks.add(1, {"outcome": "ok"})
            if published_ns is not None:
                task_duration.record(max(0.0, (time_ns() - published_ns) / 1_000_000_000))


async def process_payload(
    body: Any,
    raw_task_id: str,
    repo: TaskRepository,
    predictor: ModelPredictor,
    worker_settings: Settings,
    span: trace.Span,
) -> None:
    raw_payload = body.get("payload") if isinstance(body, dict) else None
    if not raw_task_id or not isinstance(raw_payload, dict):
        span.set_attribute("messaging.delivery.attempt", 1)
        error_on_span(span, ValueError("Invalid message envelope"))
        logger.error("worker.invalid_message", extra={"error_type": "InvalidMessage"})
        raise RejectMessage()

    # Count attempts in DB: redelivered is only a boolean flag
    if not await repo.claim_for_processing(task_id=raw_task_id, payload=raw_payload):
        logger.info("task.skipped")
        return

    attempt = repo.processing_attempts
    span.set_attribute("messaging.delivery.attempt", attempt)
    try:
        envelope = TaskEnvelope.model_validate({"task_id": raw_task_id, "payload": raw_payload})
        payload = TaskPayload(
            recent_sales=envelope.payload.recent_sales,
            day_of_week=envelope.payload.day_of_week,
            is_snap=envelope.payload.is_snap,
        )
        result = await predictor.predict(payload)
    except (ValidationError, ValueError, ModelUnavailableError) as exc:
        await repo.mark_failed(task_id=raw_task_id, error=str(exc))
        error_on_span(span, exc)
        logger.exception("task.failed", extra={"attempt": attempt})
        raise RejectMessage() from exc
    except Exception as exc:
        error_on_span(span, exc)
        if attempt < worker_settings.max_retries:
            await repo.mark_retry(task_id=raw_task_id, error=str(exc))
            span.add_event(
                "task.retry", {"retry.delay_seconds": worker_settings.retry_delay_seconds}
            )
            logger.exception("task.retry", extra={"attempt": attempt})
            await asyncio.sleep(worker_settings.retry_delay_seconds)
            raise NackMessage(requeue=True) from exc

        await repo.mark_failed(task_id=raw_task_id, error=str(exc))
        logger.exception("task.failed_dlq", extra={"attempt": attempt})
        raise RejectMessage() from exc

    # Save the result before returning and letting FastStream send ACK
    with tracer.start_as_current_span(
        "db.write_result",
        attributes={"db.system.name": "postgresql", "db.operation.name": "UPDATE"},
        record_exception=False,
        set_status_on_exception=False,
    ) as write_span:
        try:
            await repo.mark_completed(
                task_id=raw_task_id,
                prediction=result.prediction,
                model_version=result.model_version,
                run_id=result.run_id,
            )
        except Exception as exc:
            error_on_span(write_span, exc)
            raise
        logger.info("task.completed", extra={"attempt": attempt})


def build_worker_router(settings: Settings) -> RabbitRouter:
    router = RabbitRouter()

    @router.subscriber(
        queue=build_tasks_queue(settings),
        ack_policy=AckPolicy.REJECT_ON_ERROR,
    )
    async def process_task(
        body: Any,
        raw_message: RabbitMessage,
        repo: Annotated[TaskRepository, Depends(get_task_repository)],
        predictor: Annotated[ModelPredictor, Context("predictor")],
        worker_settings: Annotated[Settings, Context("settings")],
    ) -> None:
        await handle_task_message(
            body=body,
            repo=repo,
            predictor=predictor,
            worker_settings=worker_settings,
            raw_message=raw_message,
        )

    return router


def create_worker_app(settings: Settings | None = None) -> FastStream:
    resolved_settings = settings or get_settings()

    channel = Channel(prefetch_count=resolved_settings.rabbitmq_prefetch_count)
    broker = RabbitBroker(
        url=resolved_settings.rabbitmq_url,
        default_channel=channel,
        logger=logging.getLogger("mlops_labs.faststream.rabbit"),
        log_level=logging.DEBUG,
        # FastStream 0.7.7 skips draining with None; infinity waits for active handlers
        graceful_timeout=float("inf"),
    )
    broker.include_router(build_worker_router(resolved_settings))

    @asynccontextmanager
    async def lifespan(context: ContextRepo) -> AsyncGenerator[None, None]:
        configure_logging(resolved_settings.log_level, "worker")
        providers = start_telemetry("worker", resolved_settings)
        worker_inflight.add(0)
        worker_redeliveries.add(0)
        for outcome in ("ok", "error", "dlq"):
            worker_tasks.add(0, {"outcome": outcome})

        engine = create_db_engine(resolved_settings)
        predictor: ModelPredictor | None = None
        try:
            await init_db(engine)
            session_factory = create_session_factory(engine)
            loaded_model = await asyncio.to_thread(load_model, settings=resolved_settings)
            if loaded_model is None:
                raise ModelUnavailableError("Worker cannot start without its registered model")
            predictor = ModelPredictor(
                loaded_model=loaded_model,
                max_workers=resolved_settings.worker_concurrency,
                inference_delay_ms=resolved_settings.inference_delay_ms,
            )

            context.set_global("settings", resolved_settings)
            context.set_global("session_factory", session_factory)
            context.set_global("predictor", predictor)

            # FastStream has not connected the broker yet when lifespan starts
            await broker.connect()
            await declare_topology(broker, resolved_settings)
            logger.info("worker.started")
            yield
        except Exception:
            logger.exception("worker.failed")
            raise
        finally:
            await broker.stop()
            if predictor is not None:
                await asyncio.to_thread(predictor.shutdown)
            await engine.dispose()
            context.reset_global("predictor")
            context.reset_global("session_factory")
            context.reset_global("settings")
            await stop_telemetry(providers)
            logger.info("worker.stopped")

    worker_app = FastStream(
        broker, lifespan=lifespan, logger=logging.getLogger("mlops_labs.faststream.app")
    )

    @worker_app.on_shutdown
    async def log_shutdown_start() -> None:
        logger.info("worker.shutdown_started")

    return worker_app


app = create_worker_app()


async def worker_health(scope: Scope, receive: Receive, send: Send) -> None:
    await JSONResponse({"status": "ok"})(scope, receive, send)


def create_worker_asgi_app(settings: Settings | None = None) -> AsgiFastStream:
    return create_worker_app(settings).as_asgi(asgi_routes=[("/healthz", worker_health)])


asgi_app = app.as_asgi(asgi_routes=[("/healthz", worker_health)])

if __name__ == "__main__":
    import uvicorn

    configure_logging(get_settings().log_level, "worker")
    uvicorn.run(
        asgi_app,
        host="0.0.0.0",  # noqa: S104 - container health endpoint
        port=get_settings().worker_health_port,
        timeout_graceful_shutdown=None,
        log_config=None,
        access_log=False,
    )
