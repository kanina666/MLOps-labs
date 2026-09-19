import logging
import time

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from mlops_labs.core.config import get_settings
from mlops_labs.db.base import engine
from mlops_labs.schemas import ComponentHealth, HealthResponse, HealthzResponse, VersionResponse

logger = logging.getLogger(__name__)

healthz_router = APIRouter(tags=["infra"])
api_router = APIRouter(prefix="/api/v1", tags=["api"])


@healthz_router.get("/healthz", response_model=HealthzResponse)
async def healthz() -> HealthzResponse:
    return HealthzResponse(status="ok")


@api_router.get("/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    return VersionResponse(version=get_settings().app_version)


async def _check_postgres() -> ComponentHealth:
    start = time.perf_counter()
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SHOW server_version"))
            pg_version = result.scalar_one()
    except Exception:
        logger.exception("postgres health check failed")
        return ComponentHealth(
            status="error",
            response_time_ms=_elapsed_ms(start),
            error="postgres is unavailable",
        )
    return ComponentHealth(status="ok", response_time_ms=_elapsed_ms(start), version=pg_version)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


@api_router.get("/health", response_model=HealthResponse)
async def health_check(response: Response) -> HealthResponse:
    checks = {"postgres": await _check_postgres()}
    degraded = any(c.status != "ok" for c in checks.values())
    if degraded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="degraded" if degraded else "ok", checks=checks)
