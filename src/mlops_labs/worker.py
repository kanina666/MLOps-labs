import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from faststream import FastStream
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from mlops_labs.core.config import Settings, get_settings
from mlops_labs.core.logging import configure_logging
from mlops_labs.db.base import create_db_engine, create_session_factory
from mlops_labs.db.task_repository import TaskRepository
from mlops_labs.domain.model import LoadedModel
from mlops_labs.features import make_features
from mlops_labs.services.model import load_model

logger = logging.getLogger(__name__)

settings: Settings = get_settings()
channel = Channel(prefetch_count=settings.rabbitmq_prefetch_count)
broker = RabbitBroker(url=settings.rabbitmq_url, default_channel=channel)

engine: AsyncEngine | None = None
session_factory: async_sessionmaker[AsyncSession] | None = None
loaded_model: LoadedModel | None = None


@asynccontextmanager
async def lifespan(app: Any = None) -> AsyncGenerator[None, None]:
    global engine, session_factory, loaded_model
    configure_logging(settings.log_level)

    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)
    loaded_model = load_model(settings=settings)

    logger.info("worker.started")
    try:
        yield
    finally:
        if engine is not None:
            await engine.dispose()
        logger.info("worker.stopped")


@broker.subscriber(
    queue=RabbitQueue(name=settings.rabbitmq_queue, durable=True),
    channel=channel,
)
async def process_task(message: dict[str, Any]) -> None:
    if session_factory is None:
        raise RuntimeError("Worker session factory is not initialized")

    task_id = str(message.get("task_id", "")).strip()
    payload = message.get("payload")
    if not task_id or not isinstance(payload, dict):
        logger.error("worker.invalid_message", extra={"error_type": "InvalidMessage"})
        return

    async with session_factory() as session:
        repo = TaskRepository(session)

        # idempotency claim
        if not await repo.claim_for_processing(task_id=task_id, payload=payload):
            logger.info("task.skipped", extra={"task_id": task_id})
            return

        if loaded_model is None or loaded_model.model is None:
            await repo.mark_failed(task_id=task_id, error="Model is not loaded or unavailable")
            logger.error(
                "task.failed",
                extra={"task_id": task_id, "error_type": "ModelUnavailable"},
            )
            return

        # extract features & predict
        try:
            recent_sales = payload.get("recent_sales")
            day_of_week = payload.get("day_of_week")
            if recent_sales is None or day_of_week is None:
                raise ValueError("Missing recent_sales or day_of_week")

            features_df = pd.DataFrame(
                [make_features(recent_sales, int(day_of_week), int(payload.get("is_snap", 0)))]
            )

            if settings.inference_delay_ms > 0:
                await asyncio.sleep(settings.inference_delay_ms / 1000.0)

            prediction = await asyncio.to_thread(loaded_model.model.predict, features_df)
            pred_value = round(float(prediction[0]), 4)

            await repo.mark_completed(
                task_id=task_id,
                prediction=pred_value,
                model_version=loaded_model.model_version,
                run_id=loaded_model.run_id,
            )
            logger.info("task.completed", extra={"task_id": task_id})
        except Exception as exc:
            await repo.mark_failed(task_id=task_id, error=str(exc))
            logger.warning(
                "task.failed",
                extra={
                    "task_id": task_id,
                    "error_type": type(exc).__name__,
                },
            )


app = FastStream(broker, lifespan=lifespan)

if __name__ == "__main__":
    asyncio.run(app.run())
