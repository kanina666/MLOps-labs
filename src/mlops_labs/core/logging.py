import json
import logging
import logging.config
import traceback
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for field in (
            "method",
            "route",
            "status_code",
            "duration_ms",
            "dependency",
            "error_type",
            "version",
        ):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info and record.exc_info[1] is not None:
            payload["error_type"] = type(record.exc_info[1]).__name__
            payload["traceback"] = [
                f"{frame.filename}:{frame.lineno} in {frame.name}"
                for frame in traceback.extract_tb(record.exc_info[2])
            ]
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> None:
    """Configure once at startup; reconfiguration replaces handlers instead of adding them."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": JsonFormatter}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "json",
                }
            },
            "root": {"handlers": ["console"], "level": level},
            "loggers": {
                "mlops_labs": {"handlers": [], "level": "NOTSET", "propagate": True},
                "uvicorn": {"handlers": [], "level": "NOTSET", "propagate": True},
                "uvicorn.error": {"handlers": [], "level": "NOTSET", "propagate": True},
                "uvicorn.access": {"handlers": [], "propagate": False},
            },
        }
    )
