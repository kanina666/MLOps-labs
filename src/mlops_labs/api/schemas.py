from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthzResponse(BaseModel):
    status: Literal["ok"]


class VersionResponse(BaseModel):
    version: str


class ComponentHealth(BaseModel):
    status: Literal["ok", "error"]
    response_time_ms: float
    version: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, ComponentHealth]


class ProcessRequest(BaseModel):
    recent_sales: list[float] = Field(
        ...,
        min_length=28,
        description="Sales history for at least the last 28 days",
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of the week (0 = Monday, 6 = Sunday)",
    )
    is_snap: int = Field(
        default=0,
        ge=0,
        le=1,
        description="SNAP event flag (0 or 1)",
    )


class ProcessResponse(BaseModel):
    prediction: float = Field(..., description="Predicted sales units")
    model_version: str | None = Field(default=None, description="Loaded Model Registry version")
    run_id: str | None = Field(default=None, description="MLflow Run that produced the model")


class TaskSubmitResponse(BaseModel):
    task_id: str = Field(..., description="UUIDv7 identifier of the created task")
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] = Field(
        ..., description="Current status of the task"
    )


class TaskResponse(BaseModel):
    task_id: str = Field(..., description="UUIDv7 identifier of the task")
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] = Field(
        ..., description="Current lifecycle status of the task"
    )
    prediction: float | None = Field(default=None, description="Inference prediction result")
    model_version: str | None = Field(default=None, description="Loaded Model Registry version")
    run_id: str | None = Field(default=None, description="MLflow Run ID that produced the model")
    error: str | None = Field(default=None, description="Sanitized error description if failed")
    created_at: datetime | None = Field(default=None, description="Task creation timestamp")
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")
