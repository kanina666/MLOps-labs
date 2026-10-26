from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from mlops_labs.core.uuid7 import generate_uuid7_str
from mlops_labs.db.models import TaskRecord
from mlops_labs.domain.tasks import Task, TaskStatus


class TaskRepository:
    """Repository managing Task lifecycle and persistence in PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._processing_attempts = 0

    @property
    def processing_attempts(self) -> int:
        return self._processing_attempts

    async def create_task(
        self,
        payload: dict[str, Any],
        task_id: str | None = None,
    ) -> Task:
        """Insert without overwriting an existing result"""
        assigned_id = task_id or generate_uuid7_str()
        existing = await self._session.get(TaskRecord, assigned_id, populate_existing=True)
        if existing is not None:
            return existing.to_domain()
        dialect = self._session.get_bind().dialect.name
        insert = postgres_insert if dialect == "postgresql" else sqlite_insert
        statement = (
            insert(TaskRecord)
            .values(task_id=assigned_id, status=TaskStatus.PENDING, payload=payload)
            .on_conflict_do_nothing(index_elements=[TaskRecord.task_id])
        )
        await self._session.execute(statement)
        # Make PENDING visible to API inserts before acquiring the inference lock
        await self._session.commit()
        record = await self._session.get(TaskRecord, assigned_id, populate_existing=True)
        if record is None:
            raise RuntimeError("Task insert did not produce a record")
        return record.to_domain()

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task record from database"""
        record = await self._session.get(TaskRecord, task_id)
        if record is not None:
            await self._session.delete(record)
            await self._session.commit()
            return True
        return False

    async def get_by_id(self, task_id: str) -> Task | None:
        """Retrieve task by its UUIDv7 ID."""
        stmt = select(TaskRecord).where(TaskRecord.task_id == task_id)
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def claim_for_processing(
        self,
        task_id: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """
        Idempotently claim a task for worker processing (Inbox pattern).

        Returns:
            True if the task was successfully claimed for inference.
            False if the task is already COMPLETED (idempotent skip) or cannot be claimed.
        """
        if payload is not None:
            await self.create_task(payload=payload, task_id=task_id)
        statement = (
            select(TaskRecord)
            .where(TaskRecord.task_id == task_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        record = (await self._session.execute(statement)).scalar_one_or_none()
        if record is None or record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
            await self._session.rollback()
            return False

        record.status = TaskStatus.PROCESSING
        record.attempts += 1
        self._processing_attempts = record.attempts
        # Hold the row lock until the result or retry commits to prevent duplicate work
        # Defer UPDATE so API inserts can still read the committed PENDING row
        return True

    async def mark_retry(self, task_id: str, error: str) -> Task | None:
        """Save the attempt before requeue"""
        record = await self._session.get(TaskRecord, task_id)
        if record is None:
            return None
        if record.status != TaskStatus.COMPLETED:
            record.status = TaskStatus.PENDING
            record.error = error
        await self._session.commit()
        await self._session.refresh(record)
        return record.to_domain()

    async def mark_completed(
        self,
        task_id: str,
        prediction: float,
        model_version: str | None = None,
        run_id: str | None = None,
    ) -> Task | None:
        """Record inference result and transition status to COMPLETED."""
        record = await self._session.get(TaskRecord, task_id)
        if record is None:
            return None
        if record.status == TaskStatus.COMPLETED:
            await self._session.commit()
            return record.to_domain()

        record.status = TaskStatus.COMPLETED
        record.prediction = prediction
        record.model_version = model_version
        record.run_id = run_id
        record.error = None
        await self._session.commit()
        await self._session.refresh(record)
        return record.to_domain()

    async def mark_failed(self, task_id: str, error: str) -> Task | None:
        """Record error reason and transition status to FAILED."""
        record = await self._session.get(TaskRecord, task_id)
        if record is None:
            return None
        if record.status == TaskStatus.COMPLETED:
            await self._session.commit()
            return record.to_domain()

        record.status = TaskStatus.FAILED
        record.error = error
        await self._session.commit()
        await self._session.refresh(record)
        return record.to_domain()
