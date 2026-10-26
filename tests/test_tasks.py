import asyncio
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from mlops_labs.api.dependencies import get_broker, get_session_factory
from mlops_labs.core.config import Settings, get_settings
from mlops_labs.db.base import create_session_factory, init_db
from mlops_labs.db.task_repository import TaskRepository
from mlops_labs.domain.tasks import TaskStatus
from mlops_labs.main import app


@pytest.fixture
async def task_test_db() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    await init_db(engine)
    session_factory = create_session_factory(engine)
    yield session_factory
    await engine.dispose()


@pytest.fixture
def mock_broker() -> AsyncMock:
    broker = AsyncMock()
    broker.publish = AsyncMock(return_value=None)
    return broker


@pytest.fixture(autouse=True)
def _setup_app_dependencies(
    task_test_db: async_sessionmaker[AsyncSession],
    mock_broker: AsyncMock,
) -> Generator[None, None, None]:
    app.dependency_overrides[get_session_factory] = lambda: task_test_db
    app.dependency_overrides[get_broker] = lambda: mock_broker
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    yield
    app.dependency_overrides.clear()


async def test_submit_task_success(
    async_client: AsyncClient,
    task_test_db: async_sessionmaker[AsyncSession],
    mock_broker: AsyncMock,
) -> None:
    payload = {
        "recent_sales": [10.0] * 28,
        "day_of_week": 3,
        "is_snap": 0,
    }
    response = await async_client.post("/api/v1/tasks", json=payload)

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "PENDING"
    task_id = str(data["task_id"])

    # Verify task was published to broker
    mock_broker.publish.assert_awaited_once()
    publish_args, publish_kwargs = mock_broker.publish.call_args
    assert publish_args[0]["task_id"] == task_id
    assert publish_args[0]["payload"]["recent_sales"] == [10.0] * 28
    assert publish_kwargs["persist"] is True
    assert publish_kwargs["mandatory"] is True
    assert publish_kwargs["timeout"] == Settings(_env_file=None).publish_timeout_seconds

    # Verify task exists in DB with PENDING status
    async with task_test_db() as session:
        repo = TaskRepository(session)
        task = await repo.get_by_id(task_id)
        assert task is not None
        assert task.status == TaskStatus.PENDING


async def test_submit_task_load_shedding_queue_full(
    async_client: AsyncClient,
    task_test_db: async_sessionmaker[AsyncSession],
    mock_broker: AsyncMock,
) -> None:
    # Simulate queue overflow / reject-publish error
    mock_broker.publish.side_effect = RuntimeError("Queue overflow: reject-publish")

    payload = {
        "recent_sales": [10.0] * 28,
        "day_of_week": 3,
        "is_snap": 0,
    }
    response = await async_client.post("/api/v1/tasks", json=payload)

    assert response.status_code == 503
    assert response.headers.get("retry-after") == str(
        Settings(_env_file=None).effective_retry_after_seconds
    )
    assert "Service overloaded" in response.json()["detail"]

    # Crucial: task must NOT remain in DB in PENDING state!
    async with task_test_db() as session:
        repo = TaskRepository(session)
        refused_task_id = mock_broker.publish.call_args.args[0]["task_id"]
        assert await repo.get_by_id(refused_task_id) is None


async def test_submit_task_publish_timeout_has_no_pending_row(
    async_client: AsyncClient,
    task_test_db: async_sessionmaker[AsyncSession],
    mock_broker: AsyncMock,
) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, publish_timeout_seconds=0.01, retry_after_seconds=3
    )

    async def stalled_publish(*_args, **_kwargs) -> None:
        await asyncio.sleep(60)

    mock_broker.publish.side_effect = stalled_publish
    response = await async_client.post(
        "/api/v1/tasks",
        json={"recent_sales": [10.0] * 28, "day_of_week": 3, "is_snap": 0},
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "3"
    task_id = mock_broker.publish.call_args.args[0]["task_id"]
    async with task_test_db() as session:
        assert await TaskRepository(session).get_by_id(task_id) is None


async def test_submit_task_broker_unavailable(
    async_client: AsyncClient,
) -> None:
    app.dependency_overrides.pop(get_broker, None)
    app.state.broker = None

    payload = {
        "recent_sales": [10.0] * 28,
        "day_of_week": 3,
        "is_snap": 0,
    }
    response = await async_client.post("/api/v1/tasks", json=payload)

    assert response.status_code == 503
    assert response.headers.get("retry-after") == str(
        Settings(_env_file=None).effective_retry_after_seconds
    )
    assert response.json()["detail"] == "Broker is not initialized or unavailable"


async def test_submit_task_validation_error_short_history(
    async_client: AsyncClient,
) -> None:
    payload = {
        "recent_sales": [10.0] * 10,
        "day_of_week": 3,
        "is_snap": 0,
    }
    response = await async_client.post("/api/v1/tasks", json=payload)
    assert response.status_code == 422


async def test_submit_task_validation_error_invalid_day(
    async_client: AsyncClient,
) -> None:
    payload = {
        "recent_sales": [10.0] * 28,
        "day_of_week": 7,
        "is_snap": 0,
    }
    response = await async_client.post("/api/v1/tasks", json=payload)
    assert response.status_code == 422


async def test_get_task_success(
    async_client: AsyncClient,
    task_test_db: async_sessionmaker[AsyncSession],
) -> None:
    # Seed a task in DB
    async with task_test_db() as session:
        repo = TaskRepository(session)
        created = await repo.create_task(
            payload={"recent_sales": [1.0] * 28, "day_of_week": 2, "is_snap": 0},
            task_id="01923456-789a-7def-8123-456789abcdef",
        )
        await repo.mark_completed(
            task_id=created.task_id,
            prediction=42.1234,
            model_version="v2",
            run_id="run-42",
        )

    response = await async_client.get(f"/api/v1/tasks/{created.task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == created.task_id
    assert data["status"] == "COMPLETED"
    assert data["prediction"] == 42.1234
    assert data["model_version"] == "v2"
    assert data["run_id"] == "run-42"
    assert data["error"] is None


async def test_get_task_not_found(
    async_client: AsyncClient,
) -> None:
    response = await async_client.get("/api/v1/tasks/01923456-789a-7def-8123-000000000000")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
