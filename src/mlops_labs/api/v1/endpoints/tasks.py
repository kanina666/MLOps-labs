import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from faststream.rabbit import RabbitBroker

from mlops_labs.api.dependencies import get_broker, get_task_repository
from mlops_labs.api.schemas import ProcessRequest, TaskResponse, TaskSubmitResponse
from mlops_labs.api.v1.mappers import to_task_response_dto, to_task_submit_response_dto
from mlops_labs.core.config import Settings, get_settings
from mlops_labs.core.uuid7 import generate_uuid7_str
from mlops_labs.db.task_repository import TaskRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_task(
    payload: ProcessRequest,
    repo: Annotated[TaskRepository, Depends(get_task_repository)],
    broker: Annotated[RabbitBroker, Depends(get_broker)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TaskSubmitResponse:
    task_id = generate_uuid7_str()
    payload_dict = payload.model_dump()

    task = await repo.create_task(payload=payload_dict, task_id=task_id)

    try:
        await broker.publish(
            {"task_id": task_id, "payload": payload_dict},
            queue=settings.rabbitmq_queue,
        )
    except Exception as exc:
        await repo.delete_task(task_id)
        logger.warning(
            "task.shed",
            extra={
                "task_id": task_id,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service overloaded or queue is full",
            headers={"Retry-After": str(settings.retry_after_seconds)},
        ) from exc

    return to_task_submit_response_dto(task)


@router.get("/{task_id}", response_model=TaskResponse, status_code=status.HTTP_200_OK)
async def get_task(
    task_id: str,
    repo: Annotated[TaskRepository, Depends(get_task_repository)],
) -> TaskResponse:
    task = await repo.get_by_id(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    return to_task_response_dto(task)
