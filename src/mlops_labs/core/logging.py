import json
import logging
import logging.config
import os
import re
import traceback
from datetime import UTC, datetime

from opentelemetry.trace import get_current_span

from mlops_labs.core.telemetry import batch_id_context, task_id_context

EVENT_NAME = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+")


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str = "mlops-labs") -> None:
        super().__init__()
        self.service = os.getenv("OTEL_SERVICE_NAME", service)
        self.environment = os.getenv("LOG_ENV", "dev")

    def format(self, record: logging.LogRecord) -> str:
        message = record.msg if isinstance(record.msg, str) and not record.args else ""
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": message if EVENT_NAME.fullmatch(message) else "library.log",
            "service": self.service,
            "env": self.environment,
        }
        for field in (
            "method",
            "route",
            "status_code",
            "duration_ms",
            "dependency",
            "error_type",
            "error_code",
            "version",
            "model_uri",
            "model_version",
            "model_alias",
            "run_id",
            "attempt",
            "delay_seconds",
            "reason",
            "queue_wait_seconds",
            "published_total",
            "retries_total",
            "elapsed_seconds",
            "source_rate",
            "window_source_rate",
            "line_number",
        ):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if task_id := task_id_context.get():
            payload["task_id"] = task_id
        if batch_id := batch_id_context.get():
            payload["batch_id"] = batch_id
        span = get_current_span().get_span_context()
        if span.is_valid:
            payload["trace_id"] = f"{span.trace_id:032x}"
            payload["span_id"] = f"{span.span_id:016x}"
        if record.exc_info and record.exc_info[1] is not None:
            payload["error_type"] = type(record.exc_info[1]).__name__
            payload["traceback"] = [
                f"{frame.filename}:{frame.lineno} in {frame.name}"
                for frame in traceback.extract_tb(record.exc_info[2])
            ]
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str, service: str = "mlops-labs") -> None:
    """Configure once at startup; reconfiguration replaces handlers instead of adding them."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": JsonFormatter, "service": service}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "mlflow": {"handlers": [], "propagate": True, "level": "NOTSET"},
                "uvicorn": {"handlers": [], "propagate": True, "level": "NOTSET"},
                "uvicorn.access": {"handlers": [], "propagate": False},
            },
            "root": {"handlers": ["console"], "level": level},
        }
    )
    logging.captureWarnings(True)
