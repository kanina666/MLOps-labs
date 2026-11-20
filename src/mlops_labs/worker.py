import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from faststream import AckPolicy, Context, ContextRepo, Depends, FastStream
from faststream.asgi import AsgiFastStream
from faststream.exceptions import NackMessage, RejectMessage
from faststream.rabbit import Channel, RabbitBroker, RabbitMessage, RabbitRouter
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from mlops_labs.broker.topology import build_tasks_queue, declare_topology
from mlops_labs.core.config import Settings, get_settings
from mlops_labs.core.logging import configure_logging
from mlops_labs.db.base import create_db_engine, create_session_factory, init_db
from mlops_labs.db.task_repository import TaskRepository
from mlops_labs.domain.tasks import TaskPayload
from mlops_labs.services.model import ModelPredictor, ModelUnavailableError, load_model

logger = logging.getLogger(__name__)


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
    if not isinstance(body, dict):
        logger.error("worker.invalid_message", extra={"error_type": "InvalidMessage"})
        raise RejectMessage()

    raw_task_id = str(body.get("task_id", "")).strip()
    raw_payload = body.get("payload")
    if not raw_task_id or not isinstance(raw_payload, dict):
        logger.error("worker.invalid_message", extra={"error_type": "InvalidMessage"})
        raise RejectMessage()

    # Count attempts in DB: redelivered is only a boolean flag
    if not await repo.claim_for_processing(task_id=raw_task_id, payload=raw_payload):
        logger.info("task.skipped", extra={"task_id": raw_task_id})
        return

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
        logger.warning(
            "task.failed",
            extra={"task_id": raw_task_id, "error_type": type(exc).__name__},
        )
        raise RejectMessage() from exc
    except Exception as exc:
        attempt = repo.processing_attempts
        if attempt < worker_settings.max_retries:
            await repo.mark_retry(task_id=raw_task_id, error=str(exc))
            logger.warning(
                "task.retry",
                extra={"task_id": raw_task_id, "error_type": type(exc).__name__},
            )
            await asyncio.sleep(worker_settings.retry_delay_seconds)
            raise NackMessage(requeue=True) from exc

        await repo.mark_failed(task_id=raw_task_id, error=str(exc))
        logger.error(
            "task.failed_dlq",
            extra={"task_id": raw_task_id, "error_type": type(exc).__name__},
        )
        raise RejectMessage() from exc

    # Save the result before returning and letting FastStream send ACK
    await repo.mark_completed(
        task_id=raw_task_id,
        prediction=result.prediction,
        model_version=result.model_version,
        run_id=result.run_id,
    )
    logger.info("task.completed", extra={"task_id": raw_task_id})


def build_worker_router(settings: Settings) -> RabbitRouter:
    router = RabbitRouter()

    @router.subscriber(
        queue=build_tasks_queue(settings),
        ack_policy=AckPolicy.REJECT_ON_ERROR,
    )
    async def process_task(
        body: dict[str, Any],
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
        # FastStream 0.7.7 skips draining with None; infinity waits for active handlers
        graceful_timeout=float("inf"),
    )
    broker.include_router(build_worker_router(resolved_settings))

    @asynccontextmanager
    async def lifespan(context: ContextRepo) -> AsyncGenerator[None, None]:
        configure_logging(resolved_settings.log_level)

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
        finally:
            await broker.stop()
            if predictor is not None:
                await asyncio.to_thread(predictor.shutdown)
            await engine.dispose()
            context.reset_global("predictor")
            context.reset_global("session_factory")
            context.reset_global("settings")
            logger.info("worker.stopped")

    return FastStream(broker, lifespan=lifespan)


app = create_worker_app()


async def worker_health(scope: Scope, receive: Receive, send: Send) -> None:
    await JSONResponse({"status": "ok"})(scope, receive, send)


def create_worker_asgi_app(settings: Settings | None = None) -> AsgiFastStream:
    return create_worker_app(settings).as_asgi(asgi_routes=[("/healthz", worker_health)])


asgi_app = app.as_asgi(asgi_routes=[("/healthz", worker_health)])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        asgi_app,
        host="0.0.0.0",  # noqa: S104 - container health endpoint
        port=get_settings().worker_health_port,
        timeout_graceful_shutdown=None,
    )
