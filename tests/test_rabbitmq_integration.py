"""Isolated Docker tests enabled by RUN_RABBITMQ_INTEGRATION=1"""

import asyncio
import json
import os
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import aio_pika
import httpx
import numpy as np
import pytest
from faststream import TestApp
from faststream.rabbit import Channel, RabbitBroker
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mlops_labs import worker
from mlops_labs.api.dependencies import get_broker, get_session_factory
from mlops_labs.broker.topology import declare_topology
from mlops_labs.core.config import Settings, get_settings
from mlops_labs.db.base import create_session_factory, init_db
from mlops_labs.db.models import TaskRecord
from mlops_labs.db.task_repository import TaskRepository
from mlops_labs.domain.model import LoadedModel
from mlops_labs.domain.tasks import TaskStatus
from mlops_labs.main import create_app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_RABBITMQ_INTEGRATION") != "1",
        reason="Set RUN_RABBITMQ_INTEGRATION=1 to start isolated Docker containers",
    ),
]

PAYLOAD = {"recent_sales": [10.0] * 28, "day_of_week": 3, "is_snap": 0}


@pytest.fixture(scope="module")
def rabbit_container() -> Iterator[Any]:
    from testcontainers.core.container import DockerContainer

    with (
        DockerContainer("rabbitmq:3.13-management-alpine")
        .with_env("RABBITMQ_DEFAULT_USER", "lab")
        .with_env("RABBITMQ_DEFAULT_PASS", "lab")
        .with_exposed_ports(5672, 15672)
    ) as container:
        url = (
            f"http://{container.get_container_host_ip()}:"
            f"{container.get_exposed_port(15672)}/api/overview"
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                if httpx.get(url, auth=("lab", "lab"), timeout=1).is_success:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            pytest.fail("Isolated RabbitMQ did not become ready")
        yield container


@pytest.fixture(scope="module")
def postgres_container() -> Iterator[Any]:
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    with (
        DockerContainer("postgres:16-alpine")
        .with_env("POSTGRES_USER", "lab")
        .with_env("POSTGRES_PASSWORD", "lab")
        .with_env("POSTGRES_DB", "lab")
        .with_exposed_ports(5432)
    ) as container:
        wait_for_logs(container, "database system is ready to accept connections")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            check = container.get_wrapped_container().exec_run(
                ["pg_isready", "-h", "127.0.0.1", "-U", "lab", "-d", "lab"]
            )
            if check.exit_code == 0:
                break
            time.sleep(0.2)
        else:
            pytest.fail("Isolated PostgreSQL did not become ready")
        yield container


def broker_settings(container: Any, **overrides: Any) -> Settings:
    suffix = uuid4().hex
    values = {
        "rabbitmq_url": (
            f"amqp://lab:lab@{container.get_container_host_ip()}:"
            f"{container.get_exposed_port(5672)}/"
        ),
        "rabbitmq_queue": f"integration.{suffix}",
        "rabbitmq_dlq_queue": f"integration.{suffix}.dlq",
        "rabbitmq_dlx_exchange": f"integration.{suffix}.dlx",
        "rabbitmq_exchange": f"integration.{suffix}.exchange",
        "rabbitmq_max_queue_length": 1,
        "rabbitmq_prefetch_count": 1,
        "inference_delay_ms": 0,
        "max_retries": 3,
        "retry_delay_seconds": 0.01,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
async def isolated_db(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}")
    await init_db(engine)
    try:
        yield engine
    finally:
        await engine.dispose()


async def wait_for_task(engine: AsyncEngine, task_id: str, status: TaskStatus) -> Any:
    sessions = create_session_factory(engine)
    deadline = asyncio.get_running_loop().time() + 15
    while asyncio.get_running_loop().time() < deadline:
        async with sessions() as session:
            task = await TaskRepository(session).get_by_id(task_id)
        if task is not None and task.status == status:
            return task
        await asyncio.sleep(0.02)
    pytest.fail(f"Task {task_id} did not reach {status}")


async def test_real_overflow_returns_503_and_leaves_no_pending(
    rabbit_container: Any, isolated_db: AsyncEngine
) -> None:
    settings = broker_settings(rabbit_container)
    broker = RabbitBroker(settings.rabbitmq_url, default_channel=Channel(on_return_raises=True))
    await broker.connect()
    await declare_topology(broker, settings)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_broker] = lambda: broker
    app.dependency_overrides[get_session_factory] = lambda: create_session_factory(isolated_db)
    try:
        # No consumer, so the first message fills the queue
        await broker.publish({"seed": True}, queue=settings.rabbitmq_queue, persist=True)
        with pytest.raises(aio_pika.exceptions.DeliveryError):
            await broker.publish({"overflow": True}, queue=settings.rabbitmq_queue, persist=True)
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            started = asyncio.get_running_loop().time()
            response = await client.post("/api/v1/tasks", json=PAYLOAD)
            assert asyncio.get_running_loop().time() - started < 1
        assert response.status_code == 503
        assert response.headers["retry-after"] == str(settings.effective_retry_after_seconds)
        async with create_session_factory(isolated_db)() as session:
            assert await session.scalar(select(func.count()).select_from(TaskRecord)) == 0
        connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        async with connection:
            queue = await (await connection.channel()).get_queue(settings.rabbitmq_queue)
            seed = await queue.get(timeout=5)
            assert seed is not None
            await seed.ack()
            await queue.declare()
            assert queue.declaration_result.message_count == 0
            async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
                accepted = await client.post("/api/v1/tasks", json=PAYLOAD)
            assert accepted.status_code == 202
            message = await queue.get(timeout=5)
            assert message is not None
            assert message.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
            assert json.loads(message.body)["task_id"] == accepted.json()["task_id"]
            await message.ack()
            await queue.delete(if_unused=True, if_empty=True)
            # A missing queue must also return 503 without a new DB row
            async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
                unroutable = await client.post("/api/v1/tasks", json=PAYLOAD)
            assert unroutable.status_code == 503
            async with create_session_factory(isolated_db)() as session:
                task_ids = (await session.scalars(select(TaskRecord.task_id))).all()
                assert task_ids == [accepted.json()["task_id"]]
    finally:
        await broker.stop()


async def test_real_redelivery_stops_at_limit_and_routes_to_dlq(
    rabbit_container: Any, isolated_db: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = broker_settings(rabbit_container)
    model = MagicMock()
    model.predict.side_effect = RuntimeError("repeatable temporary failure")
    loaded = LoadedModel(model=model, model_version="1", run_id="integration", model_uri="test")
    monkeypatch.setattr(worker, "create_db_engine", lambda _: isolated_db)
    monkeypatch.setattr(worker, "load_model", lambda **_: loaded)
    app = worker.create_worker_app(settings)
    task_id = str(uuid4())
    async with TestApp(app):
        await app.broker.publish(
            {"task_id": task_id, "payload": PAYLOAD},
            queue=settings.rabbitmq_queue,
            persist=True,
        )
        task = await wait_for_task(isolated_db, task_id, TaskStatus.FAILED)
        assert task.attempts == 3
        connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        async with connection:
            channel = await connection.channel()
            queue = await channel.get_queue(settings.rabbitmq_dlq_queue)
            message = await queue.get(timeout=5)
            assert message is not None
            assert json.loads(message.body)["task_id"] == task_id
            await message.ack()
        await asyncio.sleep(0.1)
        assert model.predict.call_count == 3


async def test_worker_health_stays_responsive_during_synchronous_inference(
    rabbit_container: Any, isolated_db: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = broker_settings(rabbit_container)
    prediction_started = threading.Event()

    def slow_predict(_: Any) -> np.ndarray[Any, Any]:
        prediction_started.set()
        time.sleep(0.6)
        return np.array([42.0])

    model = MagicMock()
    model.predict.side_effect = slow_predict
    loaded = LoadedModel(model=model, model_version="1", run_id="integration", model_uri="test")
    monkeypatch.setattr(worker, "create_db_engine", lambda _: isolated_db)
    monkeypatch.setattr(worker, "load_model", lambda **_: loaded)
    app = worker.create_worker_app(settings)
    task_id = str(uuid4())
    async with TestApp(app):
        await app.broker.publish(
            {"task_id": task_id, "payload": PAYLOAD}, queue=settings.rabbitmq_queue
        )
        for _ in range(100):
            if prediction_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert prediction_started.is_set()
        # Health and consumption share the event loop
        asgi_app = app.as_asgi(asgi_routes=[("/healthz", worker.worker_health)])
        async with AsyncClient(transport=ASGITransport(asgi_app), base_url="http://test") as client:
            started = asyncio.get_running_loop().time()
            for _ in range(5):
                assert (await client.get("/healthz")).status_code == 200
                await asyncio.sleep(0.03)
            assert asyncio.get_running_loop().time() - started < 0.5
        await wait_for_task(isolated_db, task_id, TaskStatus.COMPLETED)


async def test_postgres_duplicate_claim_waits_and_preserves_first_result(
    postgres_container: Any,
) -> None:
    url = (
        f"postgresql+asyncpg://lab:lab@{postgres_container.get_container_host_ip()}:"
        f"{postgres_container.get_exposed_port(5432)}/lab"
    )
    engine = create_async_engine(url)
    await init_db(engine)
    sessions = create_session_factory(engine)
    task_id = str(uuid4())
    try:
        async with sessions() as first_session, sessions() as second_session:
            first = TaskRepository(first_session)
            second = TaskRepository(second_session)
            await first.create_task(task_id=task_id, payload=PAYLOAD)
            assert await first.claim_for_processing(task_id=task_id, payload=PAYLOAD)
            duplicate = asyncio.create_task(
                second.claim_for_processing(task_id=task_id, payload=PAYLOAD)
            )
            await asyncio.sleep(0.1)
            assert not duplicate.done()
            await first.mark_completed(task_id=task_id, prediction=42, model_version="1")
            assert await asyncio.wait_for(duplicate, timeout=5) is False
            task = await second.get_by_id(task_id)
            assert task is not None
            assert task.attempts == 1
            assert task.prediction == 42
    finally:
        await engine.dispose()


@pytest.mark.skipif(
    os.environ.get("RUN_WORKER_DOCKER_TESTS") != "1",
    reason="Set RUN_WORKER_DOCKER_TESTS=1 after building MLOPS_TEST_IMAGE",
)
async def test_linux_sigterm_drains_inflight_before_ack(
    rabbit_container: Any, postgres_container: Any, tmp_path: Path
) -> None:
    from testcontainers.core.container import DockerContainer

    image = os.environ.get("MLOPS_TEST_IMAGE", "mlops-labs:backpressure-lab")
    # Match the host UID to access the Linux bind mount
    container_user = {"user": f"{os.getuid()}:{os.getgid()}"} if os.name == "posix" else {}
    # Create artifact URIs at the worker's /fixture path
    bootstrap = "\n".join(
        [
            "import mlflow, mlflow.sklearn, numpy as np",
            "from sklearn.dummy import DummyRegressor",
            "mlflow.set_tracking_uri('sqlite:////fixture/mlflow.db')",
            "mlflow.set_registry_uri('sqlite:////fixture/mlflow.db')",
            "mlflow.create_experiment('shutdown-integration',artifact_location='/fixture/artifacts')",
            "mlflow.set_experiment('shutdown-integration')",
            "model=DummyRegressor().fit(np.ones((3,7)),np.array([40.,42.,44.]))",
            "with mlflow.start_run() as run:",
            "    mlflow.sklearn.log_model(model,'model',pip_requirements=[])",
            "    version=mlflow.register_model(f'runs:/{run.info.run_id}/model','shutdown-model')",
            "mlflow.MlflowClient().set_registered_model_alias('shutdown-model','champion',version.version)",
        ]
    )
    initializer = (
        DockerContainer(image)
        .with_volume_mapping(str(tmp_path), "/fixture", "rw")
        .with_kwargs(working_dir="/fixture", **container_user)
        .with_command(["python", "-c", bootstrap])
    )
    initializer.start()
    try:
        status = await asyncio.to_thread(initializer.get_wrapped_container().wait, timeout=120)
        assert status["StatusCode"] == 0, initializer.get_wrapped_container().logs().decode()
    finally:
        initializer.stop()

    settings = broker_settings(rabbit_container, inference_delay_ms=2000)
    db_host = postgres_container.get_container_host_ip()
    db_port = postgres_container.get_exposed_port(5432)
    host_database_url = f"postgresql+asyncpg://lab:lab@{db_host}:{db_port}/lab"
    worker_container = (
        DockerContainer(image)
        .with_volume_mapping(str(tmp_path), "/fixture", "rw")
        .with_kwargs(
            extra_hosts={"host.docker.internal": "host-gateway"},
            working_dir="/fixture",
            **container_user,
        )
        .with_env("MLFLOW_TRACKING_URI", "sqlite:////fixture/mlflow.db")
        .with_env("MLFLOW_REGISTRY_URI", "sqlite:////fixture/mlflow.db")
        .with_env("MODEL_URI", "models:/shutdown-model@champion")
        .with_env(
            "DATABASE_URL", f"postgresql+asyncpg://lab:lab@host.docker.internal:{db_port}/lab"
        )
        .with_env(
            "RABBITMQ_URL",
            f"amqp://lab:lab@host.docker.internal:{rabbit_container.get_exposed_port(5672)}/",
        )
        .with_env("RABBITMQ_QUEUE", settings.rabbitmq_queue)
        .with_env("RABBITMQ_DLQ_QUEUE", settings.rabbitmq_dlq_queue)
        .with_env("RABBITMQ_DLX_EXCHANGE", settings.rabbitmq_dlx_exchange)
        .with_env("INFERENCE_DELAY_MS", "2000")
        .with_env("WORKER_HEALTH_PORT", "8000")
        .with_exposed_ports(8000)
        .with_command(["python", "-m", "mlops_labs.worker"])
    )
    worker_container.start()
    engine = create_async_engine(host_database_url)
    broker = RabbitBroker(settings.rabbitmq_url)
    task_id = str(uuid4())
    try:
        health_url = (
            f"http://{worker_container.get_container_host_ip()}:"
            f"{worker_container.get_exposed_port(8000)}/healthz"
        )
        async with httpx.AsyncClient() as client:
            deadline = asyncio.get_running_loop().time() + 90
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if (await client.get(health_url, timeout=1)).is_success:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
            else:
                pytest.fail(worker_container.get_wrapped_container().logs().decode())
        await broker.connect()
        await broker.publish(
            {"task_id": task_id, "payload": PAYLOAD}, queue=settings.rabbitmq_queue, persist=True
        )
        # PENDING becomes visible after claim; PROCESSING remains uncommitted
        await wait_for_task(engine, task_id, TaskStatus.PENDING)
        # Send a real Linux SIGTERM
        await asyncio.to_thread(worker_container.get_wrapped_container().stop, timeout=15)
        task = await wait_for_task(engine, task_id, TaskStatus.COMPLETED)
        assert task.prediction == 42
        assert task.model_version == "1"
        assert task.attempts == 1
        connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        async with connection:
            queue = await (await connection.channel()).get_queue(settings.rabbitmq_queue)
            await queue.declare()
            assert queue.declaration_result.message_count == 0
        assert worker_container.get_wrapped_container().wait()["StatusCode"] == 0
    finally:
        await broker.stop()
        await engine.dispose()
        worker_container.stop()
