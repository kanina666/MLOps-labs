import asyncio
import os
import socket
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from typing import Any
from urllib.parse import urlsplit

import psutil  # type: ignore[import-untyped]
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExportResult,
    MetricsData,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import OTELResourceDetector, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult

from mlops_labs.core.config import Settings

tracer = trace.get_tracer("mlops_labs")
meter = metrics.get_meter("mlops_labs")
task_id_context: ContextVar[str | None] = ContextVar("task_id", default=None)
batch_id_context: ContextVar[str | None] = ContextVar("batch_id", default=None)

api_requests = meter.create_counter("mlops_api_requests")
api_duration = meter.create_histogram("mlops_api_request_duration", unit="s")
source_published = meter.create_counter("mlops_source_published")
source_nacks = meter.create_counter("mlops_source_nacks")
source_retries = meter.create_counter("mlops_source_retries")
source_wait = meter.create_counter("mlops_source_wait", unit="s")
worker_tasks = meter.create_counter("mlops_worker_tasks")
worker_inflight = meter.create_up_down_counter("mlops_worker_inflight")
worker_redeliveries = meter.create_counter("mlops_worker_redeliveries")
inference_duration = meter.create_histogram("mlops_inference_duration", unit="s")
task_duration = meter.create_histogram("mlops_task_duration", unit="s")


@contextmanager
def task_context(task_id: str | None, batch_id: str | None = None) -> Iterator[None]:
    task_token = task_id_context.set(task_id)
    batch_token = batch_id_context.set(batch_id)
    try:
        yield
    finally:
        batch_id_context.reset(batch_token)
        task_id_context.reset(task_token)


def error_on_span(span: trace.Span, error: BaseException) -> None:
    if span.is_recording():
        error_type = type(error).__name__
        span.set_status(trace.StatusCode.ERROR, error_type)
        span.record_exception(
            error,
            attributes={
                "exception.message": error_type,
                "exception.stacktrace": "\n".join(
                    f"{frame.filename}:{frame.lineno} in {frame.name}"
                    for frame in traceback.extract_tb(error.__traceback__)
                ),
            },
        )


def endpoint_available(endpoint: str) -> bool:
    try:
        address = urlsplit(endpoint)
        if address.hostname is None:
            return False
        return bool(socket.getaddrinfo(address.hostname, address.port, type=socket.SOCK_STREAM))
    except (OSError, ValueError):
        return False


class TraceExporter(OTLPSpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not endpoint_available(self._endpoint):
            return SpanExportResult.FAILURE
        return super().export(spans)


class MetricExporter(OTLPMetricExporter):
    def export(
        self, metrics_data: MetricsData, timeout_millis: float | None = 10_000, **kwargs: Any
    ) -> MetricExportResult:
        if not endpoint_available(self._endpoint):
            return MetricExportResult.FAILURE
        return super().export(metrics_data, timeout_millis=timeout_millis, **kwargs)


@cache
def configure_telemetry(
    service: str, app_version: str, inference_delay_ms: float
) -> tuple[TracerProvider, MeterProvider] | None:
    if os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        return None
    if not any(
        os.getenv(name)
        for name in (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        )
    ):
        return None

    resource = Resource.create(
        {
            "service.name": service,
            "service.instance.id": socket.gethostname(),
            "service.version": app_version,
            "deployment.environment.name": "dev",
        }
    ).merge(OTELResourceDetector().detect())

    traces = TracerProvider(resource=resource, shutdown_on_exit=False)
    if os.getenv("OTEL_TRACES_EXPORTER", "otlp") != "none":
        traces.add_span_processor(BatchSpanProcessor(TraceExporter()))
    trace.set_tracer_provider(traces)

    base = max(0.01, inference_delay_ms / 1000)
    views = [
        View(instrument_name=name, aggregation=ExplicitBucketHistogramAggregation(boundaries))
        for name, boundaries in (
            ("mlops_api_request_duration", [0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2]),
            (
                "mlops_inference_duration",
                [base * factor for factor in (0.5, 0.9, 1, 1.1, 1.25, 1.5, 2, 3, 5)],
            ),
            ("mlops_task_duration", [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 10, 20, 60]),
        )
    ]
    readers = []
    if os.getenv("OTEL_METRICS_EXPORTER", "otlp") != "none":
        readers.append(PeriodicExportingMetricReader(MetricExporter()))
    metric_provider = MeterProvider(
        resource=resource, metric_readers=readers, views=views, shutdown_on_exit=False
    )
    metrics.set_meter_provider(metric_provider)
    process = psutil.Process()

    def resident_memory(_: metrics.CallbackOptions) -> Iterator[metrics.Observation]:
        yield metrics.Observation(process.memory_info().rss, {"role": service})

    meter.create_observable_gauge("mlops_process_resident_memory", [resident_memory], unit="By")
    return traces, metric_provider


def start_telemetry(
    service: str, settings: Settings
) -> tuple[TracerProvider, MeterProvider] | None:
    return configure_telemetry(service, settings.app_version, settings.inference_delay_ms)


async def stop_telemetry(providers: tuple[TracerProvider, MeterProvider] | None) -> None:
    if providers is not None:
        traces, metric_provider = providers
        await asyncio.gather(
            asyncio.to_thread(traces.shutdown),
            asyncio.to_thread(metric_provider.shutdown, timeout_millis=1500),
        )
