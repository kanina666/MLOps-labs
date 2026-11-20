import json
import logging
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock

from aio_pika.exceptions import DeliveryError
from httpx import ASGITransport, AsyncClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pamqp.commands import Basic

from mlops_labs import main, source, worker
from mlops_labs.api.dependencies import get_broker, get_task_repository
from mlops_labs.broker import publishing
from mlops_labs.core import telemetry
from mlops_labs.core.config import Settings
from mlops_labs.core.logging import JsonFormatter
from mlops_labs.db.task_repository import TaskRepository
from mlops_labs.domain.model import LoadedModel
from mlops_labs.domain.tasks import Task, TaskStatus
from mlops_labs.services import model

PAYLOAD = {"recent_sales": [10.0] * 28, "day_of_week": 3, "is_snap": 0}


async def test_api_to_inference_keeps_trace_and_logging_context(monkeypatch) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    for module in (main, publishing, worker, model):
        monkeypatch.setattr(module, "tracer", tracer)

    api_repo = AsyncMock(spec=TaskRepository)
    api_repo.create_task.side_effect = lambda *, task_id, payload: Task(
        task_id, TaskStatus.PENDING, payload
    )
    broker = AsyncMock()
    app = main.create_app()
    app.dependency_overrides[get_task_repository] = lambda: api_repo
    app.dependency_overrides[get_broker] = lambda: broker

    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/api/v1/tasks", json=PAYLOAD)
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    sent = broker.publish.await_args
    assert sent is not None
    assert sent.kwargs["correlation_id"] == task_id
    message = MagicMock(
        correlation_id=task_id,
        headers=sent.kwargs["headers"],
        raw_message=MagicMock(redelivered=False),
    )

    logs = []

    def predict(_features):
        record = logging.LogRecord("model", logging.INFO, __file__, 0, "model.test", (), None)
        logs.append(json.loads(JsonFormatter().format(record)))
        return [42.0]

    repo = AsyncMock(spec=TaskRepository)
    repo.processing_attempts = 1
    repo.claim_for_processing.return_value = True
    loaded = LoadedModel(MagicMock(predict=predict), "5", "run", "uri", "champion")
    predictor = model.ModelPredictor(loaded)
    try:
        await worker.handle_task_message(sent.args[0], repo, predictor, Settings(), message)
        spans = {span.name: span for span in exporter.get_finished_spans()}
    finally:
        predictor.shutdown()
        provider.shutdown()

    assert set(spans) == {
        "POST /api/v1/tasks",
        "publish",
        "process",
        "inference",
        "db.write_result",
    }
    assert len({span.context.trace_id for span in spans.values()}) == 1
    assert spans["publish"].parent == spans["POST /api/v1/tasks"].context
    assert spans["process"].parent.span_id == spans["publish"].context.span_id
    assert spans["inference"].parent == spans["process"].context
    assert spans["db.write_result"].parent == spans["process"].context
    inference = spans["inference"]
    assert inference.attributes["model.version"] == "5"
    assert inference.attributes["model.alias"] == "champion"
    assert logs[0]["task_id"] == task_id
    assert logs[0]["trace_id"] == f"{inference.context.trace_id:032x}"
    assert logs[0]["span_id"] == f"{inference.context.span_id:016x}"
    assert telemetry.task_id_context.get() is None
    repo.mark_completed.assert_awaited_once()


def test_error_telemetry_omits_exception_values() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with provider.get_tracer(__name__).start_as_current_span("check") as span:
            try:
                raise RuntimeError("postgresql://user:secret-marker@host/db")
            except RuntimeError as exc:
                telemetry.error_on_span(span, exc)
        finished = exporter.get_finished_spans()[0]
    finally:
        provider.shutdown()
    assert finished.status.status_code == StatusCode.ERROR
    assert finished.events[0].name == "exception"
    assert "secret-marker" not in str(finished.events[0].attributes)
    assert "test_telemetry.py" in finished.events[0].attributes["exception.stacktrace"]


async def test_source_reports_broker_nack_and_batch_without_payload(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(source, "configure_logging", lambda *_: None)
    monkeypatch.setattr(source, "start_telemetry", lambda *_: None)
    monkeypatch.setattr(source, "declare_topology", AsyncMock())
    monkeypatch.setattr(
        source, "tracer", MagicMock(start_as_current_span=lambda *a, **k: nullcontext(MagicMock()))
    )
    broker = AsyncMock()
    broker.publish.side_effect = [DeliveryError(None, Basic.Nack()), None]
    monkeypatch.setattr(source, "RabbitBroker", lambda *a, **k: broker)
    logger = logging.getLogger(source.__name__)
    captured = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(json.loads(JsonFormatter().format(record)))

    handler = Capture()
    logger.addHandler(handler)
    caplog.set_level(logging.INFO, logger=source.__name__)
    batch = tmp_path / "test.jsonl"
    batch.write_text("\n" + json.dumps(PAYLOAD) + "\n", encoding="utf-8")
    try:
        stats = await source.run_source(batch, Settings(source_retry_delay_seconds=0.001))
    finally:
        logger.removeHandler(handler)

    assert stats.published_total == 1
    assert stats.retries_total == 1
    entries = {entry["event"]: entry for entry in captured}
    assert entries["source.backoff"]["reason"] == "nack"
    assert entries["source.backoff"]["task_id"] == entries["source.published"]["task_id"]
    assert entries["source.complete"]["batch_id"] == entries["source.published"]["batch_id"]
    assert "recent_sales" not in str(captured)
