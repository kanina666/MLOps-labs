from fastapi import FastAPI

from mlops_labs.api.routers.health import api_router, healthz_router

app = FastAPI(title="MLOps Labs")
app.include_router(healthz_router)
app.include_router(api_router)
