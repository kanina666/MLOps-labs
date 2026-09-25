import json
import logging
import subprocess
import sys
from datetime import datetime

import pytest
from fastapi import Response
from httpx import ASGITransport, AsyncClient

from mlops_labs.core.logging import JsonFormatter
from mlops_labs.main import create_app


@pytest.mark.parametrize(
    ("path", "status", "level", "calls"),
    [
        ("/items/private-value", 200, "INFO", 1),
        ("/items/private-value", 503, "ERROR", 1),
        ("/items/private-value", 500, "ERROR", 1),
        ("/missing/private-value", 404, "WARNING", 0),
        ("/healthz", 200, "DEBUG", 0),
    ],
)
async def test_request_logged_once(
    caplog: pytest.LogCaptureFixture, path: str, status: int, level: str, calls: int
) -> None:
    app = create_app()
    invoked = 0

    @app.get("/items/{item_id}")
    async def endpoint() -> Response:
        nonlocal invoked
        invoked += 1
        if status == 500:
            raise RuntimeError("private-error")
        return Response(status_code=status)

    caplog.set_level(logging.DEBUG, logger="mlops_labs.main")
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.request(
            "GET",
            path + "?password=private-query",
            headers={"Authorization": "Bearer private-token"},
            content=b"private-body",
        )

    assert response.status_code == status
    assert invoked == calls
    records = [r for r in caplog.records if r.getMessage() == "http.request"]
    assert len(records) == 1
    serialized = JsonFormatter().format(records[0])
    payload = json.loads(serialized)
    assert payload["status_code"] == status
    assert payload["level"] == level
    assert payload["method"] == "GET"
    expected_route = "/items/{item_id}" if calls else path
    assert payload["route"] == ("<unmatched>" if status == 404 else expected_route)
    assert payload["duration_ms"] >= 0
    assert datetime.fromisoformat(payload["timestamp"]).utcoffset().total_seconds() == 0
    assert "private-" not in serialized


def test_exception_keeps_location_without_secret_message(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("uvicorn.error")
    try:
        raise RuntimeError("postgresql://user:private-password@db:5432")
    except RuntimeError:
        logger.exception("Exception in ASGI application")

    serialized = JsonFormatter().format(caplog.records[-1])
    payload = json.loads(serialized)
    assert payload["error_type"] == "RuntimeError"
    assert any("test_logging.py:" in frame for frame in payload["traceback"])
    assert "private-password" not in serialized


def test_reconfiguration_and_log_level() -> None:
    # A separate process keeps the real logging setup isolated from pytest's log capture.
    code = """
import logging
from mlops_labs.core.logging import configure_logging

configure_logging("INFO")
configure_logging("INFO")
logging.getLogger("mlops_labs.main").info("app.event")
logging.getLogger("uvicorn.error").info("server.event")
logging.getLogger("uvicorn.access").info("duplicate.access")
configure_logging("WARNING")
logging.getLogger("mlops_labs.main").info("hidden.event")
logging.getLogger("mlops_labs.main").warning("warning.event")
"""
    result = subprocess.run(  # noqa: S603 -- current Python with a fixed test program
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=10
    )
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["event"] for r in records] == ["app.event", "server.event", "warning.event"]
    assert result.stderr == ""
