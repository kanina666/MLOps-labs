from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from mlops_labs.api.dependencies import get_health_checks
from mlops_labs.core.config import Settings, get_settings
from mlops_labs.db.probes import PostgresHealthCheck
from mlops_labs.domain.system import DependencyHealth, HealthCheck
from mlops_labs.main import app
from mlops_labs.services.health import SystemHealthCheck


def _fake_engine(*, pg_version: str = "16.1", error: Exception | None = None) -> MagicMock:
    @asynccontextmanager
    async def connect() -> AsyncGenerator[AsyncMock, None]:
        if error is not None:
            raise error
        conn = AsyncMock()
        conn.execute.return_value = MagicMock(scalar_one=MagicMock(return_value=pg_version))
        yield conn

    engine = MagicMock()
    engine.connect = connect
    return engine


@pytest.fixture(autouse=True)
def _clear_settings_and_overrides() -> Iterator[None]:
    get_settings.cache_clear()
    app.dependency_overrides.clear()
    yield
    get_settings.cache_clear()
    app.dependency_overrides.clear()


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


async def test_unknown_route_returns_404(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/lolz")
    assert response.status_code == 404


async def test_postgres_health_check_ok() -> None:
    engine = _fake_engine(pg_version="16.1")
    probe = PostgresHealthCheck(engine=engine)

    assert isinstance(probe, HealthCheck)
    result = await probe.check()

    assert result.healthy is True
    assert result.version == "16.1"
    assert result.error is None
    assert result.latency_ms >= 0


async def test_postgres_health_check_error_hides_details(caplog: pytest.LogCaptureFixture) -> None:
    secret_error = ConnectionRefusedError("postgresql://user:secret_password@db:5432 refused")
    engine = _fake_engine(error=secret_error)
    probe = PostgresHealthCheck(engine=engine)

    assert isinstance(probe, HealthCheck)
    result = await probe.check()

    assert result.healthy is False
    assert result.version is None
    assert result.error == "postgres is unavailable"
    assert "secret_password" not in (result.error or "")
    assert "secret_password" not in caplog.text
    records = [r for r in caplog.records if r.getMessage() == "dependency.unavailable"]
    assert len(records) == 1
    assert records[0].error_type == "ConnectionRefusedError"


async def test_system_health_check_aggregates_all_healthy() -> None:
    class FakeProbe:
        def __init__(self, name: str) -> None:
            self.name = name

        async def check(self) -> DependencyHealth:
            return DependencyHealth(name=self.name, healthy=True, latency_ms=1.5)

    orchestrator = SystemHealthCheck(checks=[FakeProbe("db"), FakeProbe("cache")])
    system_health = await orchestrator.check()

    assert system_health.healthy is True
    assert len(system_health.checks) == 2
    assert {c.name for c in system_health.checks} == {"db", "cache"}


async def test_system_health_check_marks_degraded_on_any_failure() -> None:
    class FailingProbe:
        async def check(self) -> DependencyHealth:
            return DependencyHealth(
                name="failing_dep",
                healthy=False,
                latency_ms=2.0,
                error="dependency down",
            )

    class HealthyProbe:
        async def check(self) -> DependencyHealth:
            return DependencyHealth(name="healthy_dep", healthy=True, latency_ms=1.0)

    orchestrator = SystemHealthCheck(checks=[FailingProbe(), HealthyProbe()])
    system_health = await orchestrator.check()

    assert system_health.healthy is False
    assert len(system_health.checks) == 2


async def test_system_health_check_empty_checks() -> None:
    orchestrator = SystemHealthCheck(checks=[])
    system_health = await orchestrator.check()

    assert system_health.healthy is True
    assert system_health.checks == ()


async def test_system_health_check_safe_check_isolates_unhandled_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BuggyProbe:
        name = "buggy"

        async def check(self) -> DependencyHealth:
            raise RuntimeError("unexpected probe failure")

    class WorkingProbe:
        async def check(self) -> DependencyHealth:
            return DependencyHealth(name="working", healthy=True, latency_ms=1.0)

    orchestrator = SystemHealthCheck(checks=[BuggyProbe(), WorkingProbe()])
    system_health = await orchestrator.check()

    assert system_health.healthy is False
    checks_by_name = {c.name: c for c in system_health.checks}
    assert checks_by_name["buggy"].healthy is False
    assert checks_by_name["buggy"].error == "dependency check failed"
    assert "unexpected probe failure" not in caplog.text
    records = [r for r in caplog.records if r.getMessage() == "dependency.check_failed"]
    assert len(records) == 1
    assert records[0].error_type == "RuntimeError"
    assert checks_by_name["working"].healthy is True


async def test_health_ok_when_db_available(async_client: AsyncClient) -> None:
    engine = _fake_engine(pg_version="16.1")
    app.dependency_overrides[get_health_checks] = lambda: [PostgresHealthCheck(engine=engine)]

    response = await async_client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["postgres"]["status"] == "ok"
    assert body["checks"]["postgres"]["version"] == "16.1"


async def test_health_503_when_db_unavailable(async_client: AsyncClient) -> None:
    engine = _fake_engine(error=ConnectionRefusedError("connection refused"))
    app.dependency_overrides[get_health_checks] = lambda: [PostgresHealthCheck(engine=engine)]

    response = await async_client.get("/api/v1/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["postgres"]["status"] == "error"
    assert body["checks"]["postgres"]["error"] == "postgres is unavailable"


async def test_health_extensibility_multiple_probes(async_client: AsyncClient) -> None:
    class FakeRedisProbe:
        async def check(self) -> DependencyHealth:
            return DependencyHealth(
                name="redis",
                healthy=True,
                latency_ms=0.5,
                version="7.2.0",
            )

    class FakePgProbe:
        async def check(self) -> DependencyHealth:
            return DependencyHealth(
                name="postgres",
                healthy=True,
                latency_ms=1.2,
                version="16.1",
            )

    app.dependency_overrides[get_health_checks] = lambda: [FakePgProbe(), FakeRedisProbe()]

    response = await async_client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "postgres" in body["checks"]
    assert "redis" in body["checks"]
    assert body["checks"]["redis"]["version"] == "7.2.0"


async def test_rabbitmq_health_check_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlops_labs.broker.probes import RabbitMQHealthCheck

    mock_conn = MagicMock()
    mock_conn.server_properties = {"version": "3.12.0"}
    mock_conn.__aenter__.return_value = mock_conn
    mock_conn.__aexit__.return_value = None

    monkeypatch.setattr("aio_pika.connect", AsyncMock(return_value=mock_conn))

    probe = RabbitMQHealthCheck(url="amqp://guest:guest@localhost:5672/")
    assert isinstance(probe, HealthCheck)
    result = await probe.check()

    assert result.healthy is True
    assert result.version == "3.12.0"
    assert result.error is None
    assert result.latency_ms >= 0


async def test_rabbitmq_health_check_error_masks_credentials(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from mlops_labs.broker.probes import RabbitMQHealthCheck

    monkeypatch.setattr(
        "aio_pika.connect",
        AsyncMock(
            side_effect=ConnectionRefusedError(
                "amqp://guest:secret_password@localhost:5672/ failed"
            )
        ),
    )

    probe = RabbitMQHealthCheck(url="amqp://guest:secret_password@localhost:5672/")
    assert isinstance(probe, HealthCheck)
    result = await probe.check()

    assert result.healthy is False
    assert result.version is None
    assert result.error == "rabbitmq is unavailable"
    assert "secret_password" not in (result.error or "")
    assert "secret_password" not in caplog.text
    records = [r for r in caplog.records if r.getMessage() == "dependency.unavailable"]
    assert len(records) == 1
    assert getattr(records[0], "error_type", None) == "ConnectionRefusedError"


async def test_main_lifespan_initializes_health_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from mlops_labs import main
    from mlops_labs.broker.probes import RabbitMQHealthCheck
    from mlops_labs.db.probes import PostgresHealthCheck

    engine = SimpleNamespace(dispose=AsyncMock())
    monkeypatch.setattr(main, "create_db_engine", lambda _s: engine)
    monkeypatch.setattr(main, "configure_logging", lambda _lvl: None)

    app_instance = main.create_app()
    assert app_instance.state.health_checks == []

    async with main.lifespan(app_instance):
        checks = app_instance.state.health_checks
        assert len(checks) == 2
        assert any(isinstance(c, PostgresHealthCheck) for c in checks)
        assert any(isinstance(c, RabbitMQHealthCheck) for c in checks)

        # Verify dependency reads from app.state
        req = MagicMock()
        req.app.state.health_checks = checks
        assert get_health_checks(req) == checks

    engine.dispose.assert_awaited_once()
