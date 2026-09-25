import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request, Response

from mlops_labs.api import router as api_router
from mlops_labs.api.healthz import router as healthz_router
from mlops_labs.core.config import get_settings
from mlops_labs.core.logging import configure_logging
from mlops_labs.db.base import create_db_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_db_engine(settings)
    app.state.settings = settings
    app.state.engine = engine

    try:
        logger.info("app.started", extra={"version": settings.app_version})
        yield
    finally:
        await engine.dispose()
        logger.info("app.stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="MLOps Labs", lifespan=lifespan)

    @app.middleware("http")
    async def log_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            level = logging.INFO
            if status_code >= 500:
                level = logging.ERROR
            elif status_code >= 400:
                level = logging.WARNING
            elif request.url.path == "/healthz":
                level = logging.DEBUG
            logger.log(
                level,
                "http.request",
                extra={
                    "method": request.method,
                    "route": getattr(request.scope.get("route"), "path", "<unmatched>"),
                    "status_code": status_code,
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                },
            )

    app.include_router(healthz_router)
    app.include_router(api_router)
    return app


app = create_app()
