import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from httpx import ASGITransport, AsyncClient

from mlops_labs import main
from mlops_labs.core.config import Settings
from mlops_labs.core.logging import JsonFormatter


async def test_alias_is_resolved_once_and_loaded_version_is_reported(monkeypatch, caplog) -> None:
    settings = Settings(model_uri="models:/m5_sales_model@champion")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "configure_logging", lambda _level: None)
    engine = SimpleNamespace(dispose=AsyncMock())
    monkeypatch.setattr(main, "create_db_engine", lambda _settings: engine)
    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7", run_id="run-7")
    monkeypatch.setattr(main, "MlflowClient", lambda: client)
    model = MagicMock()
    model.predict.return_value = np.array([123.0])
    load_model = MagicMock(return_value=model)
    monkeypatch.setattr(main.mlflow.pyfunc, "load_model", load_model)
    app = main.create_app()
    caplog.set_level(logging.INFO, logger="mlops_labs.main")

    async with main.lifespan(app):
        # Moving the alias after startup must not change the model serving requests.
        client.get_model_version_by_alias.return_value = SimpleNamespace(
            version="8", run_id="run-8"
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            response = await http.post(
                "/api/v1/process",
                json={"recent_sales": [10.0] * 28, "day_of_week": 3},
            )
            assert response.status_code == 200
            assert response.json() == {
                "prediction": 123.0,
                "model_version": "7",
                "run_id": "run-7",
            }

    client.get_model_version_by_alias.assert_called_once_with("m5_sales_model", "champion")
    load_model.assert_called_once_with("models:/m5_sales_model/7")
    assert model.predict.call_count == 1
    assert app.state.model_info["model_version"] == "7"
    record = next(record for record in caplog.records if record.message == "model.loaded")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["model_version"] == "7"
    assert payload["run_id"] == "run-7"
    engine.dispose.assert_awaited_once()


async def test_missing_champion_keeps_healthz_available(monkeypatch, caplog) -> None:
    monkeypatch.setattr(main, "get_settings", lambda: Settings())
    monkeypatch.setattr(main, "configure_logging", lambda _level: None)
    monkeypatch.setattr(
        main, "create_db_engine", lambda _settings: SimpleNamespace(dispose=AsyncMock())
    )
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = RuntimeError("Alias is not assigned")
    monkeypatch.setattr(main, "MlflowClient", lambda: client)
    app = main.create_app()

    async with main.lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            assert (await http.get("/healthz")).status_code == 200
            assert (await http.get("/training")).status_code == 404
            response = await http.post(
                "/api/v1/process", json={"recent_sales": [10.0] * 28, "day_of_week": 3}
            )
            assert response.status_code == 503
        assert app.state.model_info is None
    record = next(record for record in caplog.records if record.message == "model.load_failed")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["error_type"] == "RuntimeError"
    assert payload["model_uri"] == "models:/m5_sales_model@champion"
