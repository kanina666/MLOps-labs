from fastapi import FastAPI

from mlops_labs.api import router as api_router
from mlops_labs.api.healthz import router as healthz_router

app = FastAPI(title="MLOps Labs")
app.include_router(healthz_router)
app.include_router(api_router)
