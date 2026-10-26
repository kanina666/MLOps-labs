from mlops_labs.api.schemas import (
    ComponentHealth,
    HealthResponse,
    TaskResponse,
    TaskSubmitResponse,
)
from mlops_labs.domain.system import SystemHealth
from mlops_labs.domain.tasks import Task


def to_health_response_dto(system_health: SystemHealth) -> HealthResponse:
    return HealthResponse(
        status="ok" if system_health.healthy else "degraded",
        checks={
            item.name: ComponentHealth(
                status="ok" if item.healthy else "error",
                response_time_ms=item.latency_ms,
                version=item.version,
                error=item.error,
            )
            for item in system_health.checks
        },
    )


def to_task_response_dto(task: Task) -> TaskResponse:
    return TaskResponse(
        task_id=task.task_id,
        status=task.status.value,
        prediction=task.prediction,
        model_version=task.model_version,
        run_id=task.run_id,
        error=task.error,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def to_task_submit_response_dto(task: Task) -> TaskSubmitResponse:
    return TaskSubmitResponse(
        task_id=task.task_id,
        status=task.status.value,
    )
