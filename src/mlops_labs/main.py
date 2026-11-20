import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.routing import iter_route_contexts
from faststream.rabbit import Channel, RabbitBroker
from opentelemetry import propagate, trace

from mlops_labs.api import router as api_router
from mlops_labs.api.healthz import router as healthz_router
from mlops_labs.broker.probes import RabbitMQHealthCheck
from mlops_labs.broker.topology import declare_topology
from mlops_labs.core.config import get_settings
from mlops_labs.core.logging import configure_logging
from mlops_labs.core.telemetry import (
    api_duration,
    api_requests,
    error_on_span,
    start_telemetry,
    stop_telemetry,
    task_context,
    tracer,
)
from mlops_labs.db.base import create_db_engine, create_session_factory, init_db
from mlops_labs.db.probes import PostgresHealthCheck

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.log_level, "api")
    providers = start_telemetry("api", settings)
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

    broker = RabbitBroker(
        url=settings.rabbitmq_url,
        default_channel=Channel(on_return_raises=True),
        logger=logging.getLogger("mlops_labs.faststream.rabbit"),
    )
    app.state.broker = None

    async def connect_broker() -> bool:
        try:
            await broker.connect()
            await declare_topology(broker, settings)
        except Exception as exc:
            logger.warning("broker.connection_failed", extra={"error_type": type(exc).__name__})
            return False
        app.state.broker = broker
        return True

    async def retry_broker() -> None:
        while True:
            await asyncio.sleep(1)
            if await connect_broker():
                logger.info("broker.connection_restored")
                return

    reconnect = None if await connect_broker() else asyncio.create_task(retry_broker())

    try:
        logger.info("app.started", extra={"version": settings.app_version})
        yield
    finally:
        logger.info("app.shutdown_started")
        if reconnect is not None:
            reconnect.cancel()
            await asyncio.gather(reconnect, return_exceptions=True)
        await broker.stop()
        await engine.dispose()
        await stop_telemetry(providers)
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
        method = (
            request.method
            if request.method
            in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"}
            else "_OTHER"
        )
        with tracer.start_as_current_span(
            method,
            context=propagate.extract(request.headers),
            kind=trace.SpanKind.SERVER,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                response = await call_next(request)
                status_code = response.status_code
                if status_code >= 500:
                    span.set_status(trace.StatusCode.ERROR)
                return response
            except Exception as exc:
                error_on_span(span, exc)
                raise
            finally:
                matched_route = request.scope.get("route")
                route = (
                    next(
                        (
                            ctx.path
                            for ctx in iter_route_contexts(app.routes)
                            if ctx.original_route is matched_route
                        ),
                        None,
                    )
                    or "<unmatched>"
                )
                duration = perf_counter() - started
                attributes: dict[str, str | int] = {
                    "method": method,
                    "route": route,
                    "status_code": status_code,
                }
                span.update_name(f"{method} {route}")
                span.set_attributes(
                    {
                        "http.request.method": method,
                        "http.route": route,
                        "http.response.status_code": status_code,
                    }
                )
                api_requests.add(1, attributes)
                api_duration.record(duration, attributes)
                level = logging.INFO
                if status_code >= 500:
                    level = logging.ERROR
                elif status_code >= 400:
                    level = logging.WARNING
                elif route in {"/healthz", "/api/v1/health"}:
                    level = logging.DEBUG
                with task_context(getattr(request.state, "task_id", None)):
                    logger.log(
                        level,
                        "http.request",
                        extra={**attributes, "duration_ms": round(duration * 1000, 3)},
                    )

    app.include_router(healthz_router)
    app.include_router(api_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    configure_logging(get_settings().log_level, "api")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None, access_log=False)  # noqa: S104
