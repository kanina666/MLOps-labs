from fastapi import APIRouter

from mlops_labs.schemas import HealthzResponse

router = APIRouter(tags=["infra"])


@router.get("/healthz", response_model=HealthzResponse)
async def healthz() -> HealthzResponse:
    return HealthzResponse(status="ok")
