from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from faststream.rabbit import RabbitBroker
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from mlops_labs.db.base import create_session_factory
from mlops_labs.db.task_repository import TaskRepository
from mlops_labs.domain.system import HealthCheck
from mlops_labs.services.health import SystemHealthCheck


def get_db_engine(request: Request) -> AsyncEngine:
    engine = getattr(request.app.state, "engine", None)
    if not isinstance(engine, AsyncEngine):
        raise RuntimeError("AsyncEngine is not initialized in app.state")
    return engine


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        engine = get_db_engine(request)
        factory = create_session_factory(engine)
        request.app.state.session_factory = factory
    return factory


async def get_task_repository(
    factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
) -> AsyncGenerator[TaskRepository, None]:
    async with factory() as session:
        yield TaskRepository(session)


def get_broker(request: Request) -> RabbitBroker:
    broker = getattr(request.app.state, "broker", None)
    if not isinstance(broker, RabbitBroker):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Broker is not initialized or unavailable",
            headers={"Retry-After": "5"},
        )
    return broker


def get_health_checks(request: Request) -> list[HealthCheck]:
    checks = getattr(request.app.state, "health_checks", None)
    if checks is None:
        return []
    return list(checks)


def get_system_health_check(
    checks: Annotated[list[HealthCheck], Depends(get_health_checks)],
) -> SystemHealthCheck:
    return SystemHealthCheck(checks=checks)
