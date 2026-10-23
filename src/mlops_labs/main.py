import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request, Response
from faststream.rabbit import RabbitBroker

from mlops_labs.api import router as api_router
from mlops_labs.api.healthz import router as healthz_router
from mlops_labs.broker.probes import RabbitMQHealthCheck
from mlops_labs.broker.topology import declare_topology
from mlops_labs.core.config import get_settings
from mlops_labs.core.logging import configure_logging
from mlops_labs.db.base import create_db_engine, create_session_factory, init_db
from mlops_labs.db.probes import PostgresHealthCheck

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_db_engine(settings)
    if hasattr(engine, "begin"):
        await init_db(engine)
    session_factory = create_session_factory(engine)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory

    postgres_check = PostgresHealthCheck(engine=engine)
    rabbitmq_check = RabbitMQHealthCheck(url=settings.rabbitmq_url)
    app.state.health_checks = [postgres_check, rabbitmq_check]

    broker = RabbitBroker(url=settings.rabbitmq_url)
    app.state.broker = None
    try:
        await broker.connect()
        await declare_topology(broker, settings)
        app.state.broker = broker
    except Exception as exc:
        logger.warning(
            "broker.connection_failed",
            extra={"error_type": type(exc).__name__},
        )

    try:
        logger.info("app.started", extra={"version": settings.app_version})
        yield
    finally:
        await broker.stop()
        await engine.dispose()
        logger.info("app.stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="MLOps Labs", lifespan=lifespan)
    app.state.health_checks = []
    app.state.broker = None

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
