from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class TaskPayload:
    recent_sales: list[float]
    day_of_week: int
    is_snap: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recent_sales": self.recent_sales,
            "day_of_week": self.day_of_week,
            "is_snap": self.is_snap,
        }


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    status: TaskStatus
    payload: dict[str, Any]
    prediction: float | None = None
    model_version: str | None = None
    run_id: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
