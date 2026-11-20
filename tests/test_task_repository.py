import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from mlops_labs.db.base import create_session_factory, init_db
from mlops_labs.db.task_repository import TaskRepository
from mlops_labs.domain.tasks import TaskStatus


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Provide an isolated in-memory SQLite database session for repository tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    await init_db(engine)
    session_factory = create_session_factory(engine)

    async with session_factory() as s:
        yield s

    await engine.dispose()


async def test_create_task_generates_uuid7_and_sets_pending(session: AsyncSession) -> None:
    repo = TaskRepository(session)
    payload = {"recent_sales": [1.0] * 28, "day_of_week": 2, "is_snap": 0}

    task = await repo.create_task(payload=payload)

    assert task.task_id is not None
    # Validate that generated ID is RFC 9562 UUIDv7
    parsed_uuid = UUID(task.task_id)
    assert parsed_uuid.version == 7
    assert task.status == TaskStatus.PENDING
    assert task.payload == payload
    assert task.prediction is None
    assert task.error is None
    assert task.created_at is not None
    assert task.updated_at is not None


async def test_get_by_id_returns_task_or_none(session: AsyncSession) -> None:
    repo = TaskRepository(session)
    payload = {"recent_sales": [2.0] * 28, "day_of_week": 4}

    created = await repo.create_task(payload=payload)
    retrieved = await repo.get_by_id(created.task_id)

    assert retrieved is not None
    assert retrieved.task_id == created.task_id
    assert retrieved.status == TaskStatus.PENDING
    assert retrieved.payload == payload

    non_existent = await repo.get_by_id("non-existent-id")
    assert non_existent is None


async def test_claim_for_processing_inbox_idempotency(session: AsyncSession) -> None:
    repo = TaskRepository(session)
    payload = {"recent_sales": [3.0] * 28, "day_of_week": 1}

    task = await repo.create_task(payload=payload)

    # 1. First claim: transitions from PENDING -> PROCESSING
    claimed = await repo.claim_for_processing(task.task_id)
    assert claimed is True

    current = await repo.get_by_id(task.task_id)
    assert current is not None
    assert current.status == TaskStatus.PROCESSING

    # 2. Worker completes the inference
    completed = await repo.mark_completed(
        task_id=task.task_id,
        prediction=105.5,
        model_version="1",
        run_id="run-test-123",
    )
    assert completed is not None
    assert completed.status == TaskStatus.COMPLETED

    # 3. Duplicate delivery (Inbox check): redelivered message must be skipped
    claimed_again = await repo.claim_for_processing(task.task_id)
    assert claimed_again is False


async def test_claim_unregistered_task_creates_processing_record(session: AsyncSession) -> None:
    repo = TaskRepository(session)
    payload = {"recent_sales": [4.0] * 28, "day_of_week": 5}
    external_task_id = "external-task-001"

    # Simulates a task submitted directly via broker
    claimed = await repo.claim_for_processing(task_id=external_task_id, payload=payload)
    assert claimed is True

    record = await repo.get_by_id(external_task_id)
    assert record is not None
    assert record.status == TaskStatus.PROCESSING
    assert record.payload == payload


async def test_mark_completed_and_failed(session: AsyncSession) -> None:
    repo = TaskRepository(session)

    # Completed scenario
    task_ok = await repo.create_task(payload={"test": 1})
    completed = await repo.mark_completed(
        task_id=task_ok.task_id,
        prediction=42.0,
        model_version="v2",
        run_id="run-456",
    )
    assert completed is not None
    assert completed.status == TaskStatus.COMPLETED
    assert completed.prediction == 42.0
    assert completed.model_version == "v2"
    assert completed.run_id == "run-456"
    assert completed.error is None

    # Failed scenario
    task_err = await repo.create_task(payload={"test": 2})
    failed = await repo.mark_failed(task_id=task_err.task_id, error="Model timeout")
    assert failed is not None
    assert failed.status == TaskStatus.FAILED
    assert failed.error == "Model timeout"
    assert failed.prediction is None


async def test_uuid7_time_monotonicity(session: AsyncSession) -> None:
    repo = TaskRepository(session)

    task1 = await repo.create_task(payload={"idx": 1})
    await asyncio.sleep(0.005)
    task2 = await repo.create_task(payload={"idx": 2})

    # Time-ordered: later task ID is lexicographically greater
    assert task1.task_id < task2.task_id
