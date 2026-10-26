import logging
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from faststream import TestApp
from faststream.exceptions import NackMessage, RejectMessage
from faststream.rabbit import TestRabbitBroker
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from mlops_labs import worker
from mlops_labs.core.config import Settings
from mlops_labs.db.base import create_session_factory, init_db
from mlops_labs.db.task_repository import TaskRepository
from mlops_labs.domain.model import LoadedModel
from mlops_labs.domain.tasks import TaskStatus
from mlops_labs.services.model import ModelPredictor, ModelUnavailableError


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
def worker_env(test_db, mock_loaded_model, monkeypatch: pytest.MonkeyPatch):
    engine, session_factory = test_db
    settings = Settings(
        rabbitmq_queue="mlops.tasks",
        inference_delay_ms=0.0,
        max_retries=3,
    )
    monkeypatch.setattr(worker, "create_db_engine", lambda _s: engine)
    monkeypatch.setattr(worker, "load_model", lambda **_kw: mock_loaded_model)
    app = worker.create_worker_app(settings)
    return app, session_factory, settings


async def test_worker_processes_task_successfully(worker_env) -> None:
    app, session_factory, _ = worker_env
    message = {
        "task_id": "test-task-1",
        "payload": {
            "recent_sales": [10.0] * 28,
            "day_of_week": 3,
            "is_snap": 1,
        },
    }

    async with TestRabbitBroker(app.broker) as br, TestApp(app):
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


async def test_worker_inbox_idempotency_duplicate_skipped(worker_env, mock_loaded_model) -> None:
    app, session_factory, _ = worker_env

    async with TestRabbitBroker(app.broker) as br, TestApp(app):
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

        mock_loaded_model.model.predict.assert_not_called()


async def test_worker_missing_task_id_ignored(worker_env, caplog: pytest.LogCaptureFixture) -> None:
    app, _, _ = worker_env
    caplog.set_level(logging.ERROR, logger="mlops_labs.worker")

    async with TestRabbitBroker(app.broker) as br, TestApp(app):
        await br.publish({"payload": {}}, queue="mlops.tasks")

    assert any(r.getMessage() == "worker.invalid_message" for r in caplog.records)


async def test_worker_empty_payload_marks_failed(worker_env) -> None:
    app, session_factory, _ = worker_env

    async with TestRabbitBroker(app.broker) as br, TestApp(app):
        await br.publish({"task_id": "empty-payload-task", "payload": {}}, queue="mlops.tasks")

        async with session_factory() as session:
            repo = TaskRepository(session)
            task = await repo.get_by_id("empty-payload-task")
            assert task is not None
            assert task.status == TaskStatus.FAILED
            assert "recent_sales" in str(task.error)


async def test_worker_invalid_message_logged(worker_env, caplog: pytest.LogCaptureFixture) -> None:
    app, _, _ = worker_env
    caplog.set_level(logging.ERROR, logger="mlops_labs.worker")

    async with TestRabbitBroker(app.broker) as br, TestApp(app):
        await br.publish({"task_id": ""}, queue="mlops.tasks")

    assert any(r.getMessage() == "worker.invalid_message" for r in caplog.records)


async def test_worker_feature_extraction_failure_marks_failed(worker_env) -> None:
    app, session_factory, _ = worker_env

    message = {
        "task_id": "short-sales-task",
        "payload": {"recent_sales": [1.0] * 10, "day_of_week": 3},
    }

    async with TestRabbitBroker(app.broker) as br, TestApp(app):
        await br.publish(message, queue="mlops.tasks")

        async with session_factory() as session:
            repo = TaskRepository(session)
            task = await repo.get_by_id("short-sales-task")
            assert task is not None
            assert task.status == TaskStatus.FAILED
            assert "At least 28 days of sales history are required" in (task.error or "")


async def test_worker_model_unavailable_fails_startup(
    test_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = test_db
    settings = Settings(rabbitmq_queue="mlops.tasks", inference_delay_ms=0.0)
    monkeypatch.setattr(worker, "create_db_engine", lambda _s: engine)
    monkeypatch.setattr(worker, "load_model", lambda **_kw: None)
    app = worker.create_worker_app(settings)

    with pytest.raises(ModelUnavailableError, match="cannot start"):
        async with TestRabbitBroker(app.broker), TestApp(app):
            pytest.fail("Worker must not consume tasks without a model")


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
    monkeypatch.setattr(worker, "init_db", AsyncMock())

    app = worker.create_worker_app(Settings())
    async with TestRabbitBroker(app.broker), TestApp(app):
        assert app.context.get("predictor") is not None
        assert app.context.get("session_factory") is not None
        mock_load_model.assert_called_once()

    engine_mock.dispose.assert_awaited_once()


async def test_handle_task_message_retry_counter_survives_new_repository_sessions(test_db) -> None:
    _, session_factory = test_db
    settings = Settings(max_retries=3, retry_delay_seconds=0.001)

    failing_model = MagicMock()
    failing_model.predict.side_effect = RuntimeError("Temporary GPU/IO glitch")
    loaded = LoadedModel(
        model=failing_model,
        model_version="v1",
        run_id="r1",
        model_uri="models:/m5/1",
    )
    predictor = ModelPredictor(loaded_model=loaded)
    msg = {
        "task_id": "transient-task-1",
        "payload": {"recent_sales": [10.0] * 28, "day_of_week": 2, "is_snap": 0},
    }

    try:
        for attempt in range(1, 4):
            async with session_factory() as session:
                repo = TaskRepository(session)
                expected = NackMessage if attempt < 3 else RejectMessage
                with pytest.raises(expected):
                    await worker.handle_task_message(
                        body=msg,
                        repo=repo,
                        predictor=predictor,
                        worker_settings=settings,
                    )
                task = await repo.get_by_id("transient-task-1")
                assert task is not None
                assert task.attempts == attempt
                assert task.status == (TaskStatus.PENDING if attempt < 3 else TaskStatus.FAILED)

        async with session_factory() as session:
            repo = TaskRepository(session)
            task = await repo.get_by_id("transient-task-1")
            assert task is not None
            assert task.status == TaskStatus.FAILED
            assert "Temporary GPU/IO glitch" in (task.error or "")
        assert failing_model.predict.call_count == 3
    finally:
        predictor.shutdown()
