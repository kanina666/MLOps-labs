from fastapi import FastAPI

from mlops_labs.api.routers.health import healthz_router

app = FastAPI(title="MLOps Labs")
app.include_router(healthz_router)
