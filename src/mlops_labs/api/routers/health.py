from fastapi import APIRouter

from mlops_labs.core.config import get_settings
from mlops_labs.schemas import HealthzResponse, VersionResponse

healthz_router = APIRouter(tags=["infra"])
api_router = APIRouter(prefix="/api/v1", tags=["api"])


@healthz_router.get("/healthz", response_model=HealthzResponse)
async def healthz() -> HealthzResponse:
    return HealthzResponse(status="ok")


@api_router.get("/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    return VersionResponse(version=get_settings().app_version)
