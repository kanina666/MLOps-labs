from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mlops_labs.core.uuid7 import generate_uuid7_str
from mlops_labs.db.models import TaskRecord
from mlops_labs.domain.tasks import Task, TaskStatus


class TaskRepository:
    """Repository managing Task lifecycle and persistence in PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_task(
        self,
        payload: dict[str, Any],
        task_id: str | None = None,
    ) -> Task:
        """Create a new task with PENDING status."""
        assigned_id = task_id or generate_uuid7_str()
        record = TaskRecord(
            task_id=assigned_id,
            status=TaskStatus.PENDING,
            payload=payload,
        )
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record.to_domain()

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
        record = await self._session.get(TaskRecord, task_id)
        if record is None:
            if payload is not None:
                record = TaskRecord(
                    task_id=task_id,
                    status=TaskStatus.PROCESSING,
                    payload=payload,
                )
                self._session.add(record)
                await self._session.commit()
                return True
            return False

        if record.status == TaskStatus.COMPLETED:
            # Task has already been processed and finalized; skip duplicate work.
            return False

        record.status = TaskStatus.PROCESSING
        await self._session.commit()
        return True

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

        record.status = TaskStatus.FAILED
        record.error = error
        await self._session.commit()
        await self._session.refresh(record)
        return record.to_domain()
