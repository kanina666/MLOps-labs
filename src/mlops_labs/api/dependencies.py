from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncEngine

from mlops_labs.api.schemas import ProcessRequest
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


def get_model(request: Request) -> Any:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable",
        )
    return model


def extract_features(payload: ProcessRequest) -> pd.DataFrame:
    sales = payload.recent_sales
    return pd.DataFrame(
        [
            {
                "lag_1": float(sales[-1]),
                "lag_7": float(sales[-7]),
                "lag_14": float(sales[-14]),
                "lag_28": float(sales[-28]),
                "rolling_mean_7": float(sum(sales[-7:]) / 7.0),
                "day_of_week": int(payload.day_of_week),
                "is_snap": int(payload.is_snap),
            }
        ]
    )
