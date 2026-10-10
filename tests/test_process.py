from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from httpx import AsyncClient

from mlops_labs.api.dependencies import extract_features, get_model
from mlops_labs.api.schemas import ProcessRequest
from mlops_labs.main import app


def test_extract_features_calculation() -> None:
    history = [float(i) for i in range(1, 29)]
    payload = ProcessRequest(recent_sales=history, day_of_week=5, is_snap=1)
    features_df = extract_features(payload)

    assert isinstance(features_df, pd.DataFrame)
    assert features_df.loc[0, "lag_1"] == 28.0
    assert features_df.loc[0, "lag_7"] == 22.0
    assert features_df.loc[0, "lag_14"] == 15.0
    assert features_df.loc[0, "lag_28"] == 1.0
    assert features_df.loc[0, "rolling_mean_7"] == pytest.approx(sum(history[-7:]) / 7.0)
    assert features_df.loc[0, "day_of_week"] == 5
    assert features_df.loc[0, "is_snap"] == 1


async def test_process_success(async_client: AsyncClient) -> None:
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([123.4567])
    app.dependency_overrides[get_model] = lambda: mock_model

    payload = {
        "recent_sales": [10.0] * 28,
        "day_of_week": 3,
        "is_snap": 0,
    }
    response = await async_client.post("/api/v1/process", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": 123.4567}
    mock_model.predict.assert_called_once()


async def test_process_model_unavailable(async_client: AsyncClient) -> None:
    app.dependency_overrides.clear()
    app.state.model = None

    payload = {
        "recent_sales": [10.0] * 28,
        "day_of_week": 3,
        "is_snap": 0,
    }
    response = await async_client.post("/api/v1/process", json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded or unavailable"


async def test_process_validation_error_short_history(async_client: AsyncClient) -> None:
    app.dependency_overrides[get_model] = lambda: MagicMock()
    try:
        payload = {
            "recent_sales": [10.0] * 10,
            "day_of_week": 3,
            "is_snap": 0,
        }
        response = await async_client.post("/api/v1/process", json=payload)
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


async def test_process_validation_error_invalid_day(async_client: AsyncClient) -> None:
    app.dependency_overrides[get_model] = lambda: MagicMock()
    try:
        payload = {
            "recent_sales": [10.0] * 28,
            "day_of_week": 7,
            "is_snap": 0,
        }
        response = await async_client.post("/api/v1/process", json=payload)
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()
