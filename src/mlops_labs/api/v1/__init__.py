from fastapi import APIRouter

from mlops_labs.api.v1.endpoints.health import router as health_router
from mlops_labs.api.v1.endpoints.tasks import router as tasks_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(tasks_router)
