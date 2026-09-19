from fastapi import APIRouter

from mlops_labs.schemas import HealthzResponse

healthz_router = APIRouter(tags=["infra"])


@healthz_router.get("/healthz", response_model=HealthzResponse)
async def healthz() -> HealthzResponse:
    return HealthzResponse(status="ok")
