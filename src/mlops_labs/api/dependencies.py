from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncEngine

from mlops_labs.db.probes import PostgresHealthCheck
from mlops_labs.domain.system import HealthCheck
from mlops_labs.services.health import SystemHealthCheck


def get_db_engine(request: Request) -> AsyncEngine:
    engine = getattr(request.app.state, "engine", None)
    if not isinstance(engine, AsyncEngine):
        raise RuntimeError("AsyncEngine is not initialized in app.state")
    return engine


def get_health_checks(engine: Annotated[AsyncEngine, Depends(get_db_engine)]) -> list[HealthCheck]:
    return [PostgresHealthCheck(engine=engine)]


def get_system_health_check(
    checks: Annotated[list[HealthCheck], Depends(get_health_checks)],
) -> SystemHealthCheck:
    return SystemHealthCheck(checks=checks)
