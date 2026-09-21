from fastapi import APIRouter

from mlops_labs.api.v1 import router as v1_router

router = APIRouter(prefix="/api", tags=["api"])
router.include_router(v1_router, prefix="/v1", tags=["v1"])
