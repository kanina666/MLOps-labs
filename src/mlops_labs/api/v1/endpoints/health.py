from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from mlops_labs.api.dependencies import get_system_health_check
from mlops_labs.api.schemas import HealthResponse, VersionResponse
from mlops_labs.api.v1.mappers import to_health_response_dto
from mlops_labs.core.config import get_settings
from mlops_labs.services.health import SystemHealthCheck

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    response: Response,
    service: Annotated[SystemHealthCheck, Depends(get_system_health_check)],
) -> HealthResponse:
    system_health = await service.check()

    if not system_health.healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return to_health_response_dto(system_health)


@router.get("/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    return VersionResponse(version=get_settings().app_version)
