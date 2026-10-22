import logging
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from faststream.rabbit import TestRabbitBroker
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from mlops_labs import worker
from mlops_labs.core.config import Settings
from mlops_labs.db.base import create_session_factory, init_db
from mlops_labs.db.task_repository import TaskRepository
from mlops_labs.domain.model import LoadedModel
from mlops_labs.domain.tasks import TaskStatus


@pytest.fixture(autouse=True)
def _mock_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "configure_logging", lambda _lvl: None)


@pytest.fixture
async def test_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    await init_db(engine)
    session_factory = create_session_factory(engine)
    yield engine, session_factory
    await engine.dispose()


@pytest.fixture
def mock_loaded_model() -> LoadedModel:
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([42.123456])
    return LoadedModel(
        model=mock_model,
        model_version="v1.0",
        run_id="run-test-001",
        model_uri="models:/m5_sales_model/1",
    )


@pytest.fixture
def configure_worker(test_db, mock_loaded_model, monkeypatch: pytest.MonkeyPatch):
    engine, session_factory = test_db
    monkeypatch.setattr(worker, "engine", engine)
    monkeypatch.setattr(worker, "session_factory", session_factory)
    monkeypatch.setattr(worker, "loaded_model", mock_loaded_model)
    monkeypatch.setattr(
        worker, "settings", Settings(rabbitmq_queue="mlops.tasks", inference_delay_ms=0.0)
    )
    return session_factory


async def test_worker_processes_task_successfully(configure_worker) -> None:
    session_factory = configure_worker
    message = {
        "task_id": "test-task-1",
        "payload": {
            "recent_sales": [10.0] * 28,
            "day_of_week": 3,
            "is_snap": 1,
        },
    }

    async with TestRabbitBroker(worker.broker) as br:
        await br.publish(message, queue="mlops.tasks")

    async with session_factory() as session:
        repo = TaskRepository(session)
        task = await repo.get_by_id("test-task-1")
        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.prediction == 42.1235
        assert task.model_version == "v1.0"
        assert task.run_id == "run-test-001"
        assert task.error is None


async def test_worker_inbox_idempotency_duplicate_skipped(configure_worker) -> None:
    session_factory = configure_worker

    # Pre-create and complete task
    async with session_factory() as session:
        repo = TaskRepository(session)
        await repo.create_task(
            task_id="idempotent-task",
            payload={"recent_sales": [1.0] * 28, "day_of_week": 4},
        )
        await repo.mark_completed(
            task_id="idempotent-task",
            prediction=999.0,
            model_version="old",
            run_id="old-run",
        )

    # Duplicate delivery
    async with TestRabbitBroker(worker.broker) as br:
        await br.publish(
            {
                "task_id": "idempotent-task",
                "payload": {"recent_sales": [1.0] * 28, "day_of_week": 4},
            },
            queue="mlops.tasks",
        )

    async with session_factory() as session:
        repo = TaskRepository(session)
        task = await repo.get_by_id("idempotent-task")
        assert task is not None
        assert task.prediction == 999.0
        assert task.model_version == "old"


async def test_worker_missing_task_id_ignored(
    configure_worker, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="mlops_labs.worker")

    async with TestRabbitBroker(worker.broker) as br:
        await br.publish({"payload": {}}, queue="mlops.tasks")

    assert any(r.getMessage() == "worker.invalid_message" for r in caplog.records)


async def test_worker_empty_payload_marks_failed(configure_worker) -> None:
    session_factory = configure_worker

    async with TestRabbitBroker(worker.broker) as br:
        await br.publish({"task_id": "empty-payload-task", "payload": {}}, queue="mlops.tasks")

    async with session_factory() as session:
        repo = TaskRepository(session)
        task = await repo.get_by_id("empty-payload-task")
        assert task is not None
        assert task.status == TaskStatus.FAILED
        assert "Missing recent_sales" in str(task.error)


async def test_worker_invalid_message_logged(
    configure_worker, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="mlops_labs.worker")

    async with TestRabbitBroker(worker.broker) as br:
        await br.publish({"task_id": ""}, queue="mlops.tasks")

    assert any(r.getMessage() == "worker.invalid_message" for r in caplog.records)


async def test_worker_feature_extraction_failure_marks_failed(configure_worker) -> None:
    session_factory = configure_worker

    message = {
        "task_id": "short-sales-task",
        "payload": {"recent_sales": [1.0] * 10, "day_of_week": 3},
    }

    async with TestRabbitBroker(worker.broker) as br:
        await br.publish(message, queue="mlops.tasks")

    async with session_factory() as session:
        repo = TaskRepository(session)
        task = await repo.get_by_id("short-sales-task")
        assert task is not None
        assert task.status == TaskStatus.FAILED
        assert "At least 28 days of sales history are required" in (task.error or "")


async def test_worker_model_unavailable_marks_failed(
    configure_worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_factory = configure_worker
    monkeypatch.setattr(worker, "loaded_model", None)

    message = {
        "task_id": "no-model-task",
        "payload": {"recent_sales": [1.0] * 28, "day_of_week": 2},
    }

    async with TestRabbitBroker(worker.broker) as br:
        await br.publish(message, queue="mlops.tasks")

    async with session_factory() as session:
        repo = TaskRepository(session)
        task = await repo.get_by_id("no-model-task")
        assert task is not None
        assert task.status == TaskStatus.FAILED
        assert task.error == "Model is not loaded or unavailable"


async def test_worker_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_load_model = MagicMock(
        return_value=LoadedModel(
            model=MagicMock(),
            model_version="champion-1",
            run_id="run-champ",
            model_uri="models:/m5/1",
        )
    )
    monkeypatch.setattr(worker, "load_model", mock_load_model)

    engine_mock = MagicMock()
    engine_mock.dispose = AsyncMock()
    monkeypatch.setattr(worker, "create_db_engine", lambda _s: engine_mock)

    monkeypatch.setattr(worker, "engine", None)
    monkeypatch.setattr(worker, "session_factory", None)
    monkeypatch.setattr(worker, "loaded_model", None)

    async with worker.lifespan():
        assert worker.loaded_model is not None
        assert worker.session_factory is not None
        mock_load_model.assert_called_once()

    engine_mock.dispose.assert_awaited_once()


async def test_direct_process_task_success(configure_worker) -> None:
    session_factory = configure_worker
    msg = {
        "task_id": "direct-task-1",
        "payload": {"recent_sales": [10.0] * 28, "day_of_week": 2, "is_snap": 0},
    }
    await worker.process_task(message=msg)

    async with session_factory() as s:
        repo = TaskRepository(s)
        task = await repo.get_by_id("direct-task-1")
        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.prediction == 42.1235
