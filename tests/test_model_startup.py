import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from mlops_labs import main
from mlops_labs.core.config import Settings
from mlops_labs.core.logging import JsonFormatter
from mlops_labs.db.base import init_db
from mlops_labs.domain.model import LoadedModel
from mlops_labs.services.model import load_model


async def test_alias_is_resolved_once_and_loaded_version_is_reported(monkeypatch, caplog) -> None:
    settings = Settings(model_uri="models:/m5_sales_model@champion")
    monkeypatch.setattr("mlops_labs.services.model.get_settings", lambda: settings)

    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7", run_id="run-7")
    monkeypatch.setattr("mlops_labs.services.model.MlflowClient", lambda: client)

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([123.0])
    mlflow_load = MagicMock(return_value=mock_model)
    monkeypatch.setattr("mlops_labs.services.model.mlflow.pyfunc.load_model", mlflow_load)

    caplog.set_level(logging.INFO, logger="mlops_labs.services.model")

    loaded = load_model(settings=settings)

    assert loaded is not None
    assert isinstance(loaded, LoadedModel)
    assert loaded.model_version == "7"
    assert loaded.run_id == "run-7"
    assert loaded.model_uri == "models:/m5_sales_model/7"
    assert loaded.model == mock_model

    # Moving alias after loading does not affect loaded model
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="8", run_id="run-8")
    assert loaded.model_version == "7"

    client.get_model_version_by_alias.assert_called_once_with("m5_sales_model", "champion")
    mlflow_load.assert_called_once_with("models:/m5_sales_model/7")

    record = next(record for record in caplog.records if record.message == "model.loaded")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["model_version"] == "7"
    assert payload["run_id"] == "run-7"
    assert payload["model_uri"] == "models:/m5_sales_model/7"


def test_load_model_by_name_and_custom_alias(monkeypatch) -> None:
    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="2", run_id="run-2")
    monkeypatch.setattr("mlops_labs.services.model.MlflowClient", lambda: client)

    mock_model = MagicMock()
    mlflow_load = MagicMock(return_value=mock_model)
    monkeypatch.setattr("mlops_labs.services.model.mlflow.pyfunc.load_model", mlflow_load)

    loaded = load_model(model_name="retail_model", alias="candidate")

    assert loaded is not None
    assert loaded.model_version == "2"
    assert loaded.run_id == "run-2"
    assert loaded.model_uri == "models:/retail_model/2"
    client.get_model_version_by_alias.assert_called_once_with("retail_model", "candidate")
    mlflow_load.assert_called_once_with("models:/retail_model/2")


def test_load_model_direct_uri_without_alias(monkeypatch) -> None:
    mock_model = MagicMock()
    mlflow_load = MagicMock(return_value=mock_model)
    monkeypatch.setattr("mlops_labs.services.model.mlflow.pyfunc.load_model", mlflow_load)

    loaded = load_model(model_name="runs:/xyz123/model")

    assert loaded is not None
    assert loaded.model_version is None
    assert loaded.run_id is None
    assert loaded.model_uri == "runs:/xyz123/model"
    mlflow_load.assert_called_once_with("runs:/xyz123/model")


async def test_missing_champion_keeps_healthz_available(monkeypatch, caplog) -> None:
    settings = Settings(model_uri="models:/m5_sales_model@champion")
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = RuntimeError("Alias is not assigned")
    monkeypatch.setattr("mlops_labs.services.model.MlflowClient", lambda: client)

    caplog.set_level(logging.WARNING, logger="mlops_labs.services.model")
    loaded = load_model(settings=settings)
    assert loaded is None

    record = next(record for record in caplog.records if record.message == "model.load_failed")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["error_type"] == "RuntimeError"
    assert payload["model_uri"] == "models:/m5_sales_model@champion"

    # Main app lifespan runs without model and keeps healthz functional
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    monkeypatch.setattr(main, "create_db_engine", lambda _settings: engine)
    monkeypatch.setattr(main, "configure_logging", lambda _level: None)
    app = main.create_app()

    async with main.lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            assert (await http.get("/healthz")).status_code == 200
            assert (await http.get("/training")).status_code == 404
            response = await http.post(
                "/api/v1/tasks", json={"recent_sales": [10.0] * 28, "day_of_week": 3}
            )
            assert response.status_code == 503
        assert getattr(app.state, "model", None) is None
        assert getattr(app.state, "model_info", None) is None
