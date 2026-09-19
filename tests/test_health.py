from collections.abc import Iterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from mlops_labs.api.routers import health
from mlops_labs.core.config import Settings, get_settings
from mlops_labs.schemas import ComponentHealth


def _fake_engine(*, pg_version: str = "16.1", error: Exception | None = None) -> MagicMock:
    @asynccontextmanager
    async def connect():
        if error is not None:
            raise error
        conn = AsyncMock()
        conn.execute.return_value = MagicMock(scalar_one=MagicMock(return_value=pg_version))
        yield conn

    engine = MagicMock()
    engine.connect = connect
    return engine


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_healthz_ok(async_client: AsyncClient) -> None:
    response = await async_client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_version_from_package_metadata(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/version")
    assert response.status_code == 200
    assert response.json()["version"] not in ("", "0.0.0-dev")


async def test_version_override_from_env(
    async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_VERSION_OVERRIDE", "1.2.3")
    response = await async_client.get("/api/v1/version")
    assert response.json() == {"version": "1.2.3"}


def test_version_fallback_when_package_not_installed() -> None:
    from importlib.metadata import PackageNotFoundError

    with patch("mlops_labs.core.config.version", side_effect=PackageNotFoundError):
        assert Settings().app_version == "0.0.0-dev"


async def test_check_postgres_ok() -> None:
    with patch.object(health, "engine", _fake_engine(pg_version="16.1")):
        result = await health._check_postgres()
    assert result.status == "ok"
    assert result.version == "16.1"
    assert result.error is None
    assert result.response_time_ms >= 0


async def test_check_postgres_error_hides_details() -> None:
    secret_error = ConnectionRefusedError("postgresql://user:password@db:5432 refused")
    with patch.object(health, "engine", _fake_engine(error=secret_error)):
        result = await health._check_postgres()
    assert result.status == "error"
    assert result.version is None
    assert result.error == "postgres is unavailable"
    assert "password" not in (result.error or "")


async def test_health_ok_when_db_available(async_client: AsyncClient) -> None:
    fake = ComponentHealth(status="ok", response_time_ms=1.0, version="16.1")
    with patch.object(health, "_check_postgres", return_value=fake):
        response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["postgres"]["version"] == "16.1"


async def test_health_503_when_db_unavailable(async_client: AsyncClient) -> None:
    with patch.object(health, "engine", _fake_engine(error=ConnectionRefusedError("boom"))):
        response = await async_client.get("/api/v1/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["postgres"]["status"] == "error"


async def test_unknown_route_returns_404(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/nope")
    assert response.status_code == 404
