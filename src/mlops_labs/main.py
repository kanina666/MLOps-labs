from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mlops_labs.api import router as api_router
from mlops_labs.api.healthz import router as healthz_router
from mlops_labs.core.config import get_settings
from mlops_labs.db.base import create_db_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    engine = create_db_engine(settings)
    app.state.settings = settings
    app.state.engine = engine

    try:
        yield
    finally:
        await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="MLOps Labs", lifespan=lifespan)
    app.include_router(healthz_router)
    app.include_router(api_router)
    return app


app = create_app()
