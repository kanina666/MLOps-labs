"""Stream JSONL tasks with backpressure from RabbitMQ"""

import argparse
import asyncio
import hashlib
import json
import logging
import signal
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO
from uuid import NAMESPACE_URL, uuid5

from aio_pika.exceptions import AMQPError
from faststream.rabbit import Channel, RabbitBroker

from mlops_labs.api.schemas import ProcessRequest
from mlops_labs.broker.publishing import publish_failure, publish_task
from mlops_labs.broker.topology import declare_topology
from mlops_labs.core.config import Settings, get_settings
from mlops_labs.core.logging import configure_logging
from mlops_labs.core.telemetry import (
    error_on_span,
    source_nacks,
    source_published,
    source_retries,
    source_wait,
    start_telemetry,
    stop_telemetry,
    task_context,
    tracer,
)

logger = logging.getLogger(__name__)


@dataclass
class SourceStats:
    published_total: int = 0
    retries_total: int = 0


def read_task_messages(
    path: Path, batch_id: str | None = None
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Read one record at a time with stable IDs for batch restarts"""
    if batch_id is None:
        with path.open("rb") as stream:
            batch_id = hashlib.file_digest(stream, "sha256").hexdigest()

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = ProcessRequest.model_validate_json(line).model_dump()
            except ValueError:
                logger.exception("source.invalid_line", extra={"line_number": line_number})
                raise ValueError(f"Invalid task at line {line_number}") from None
            task_id = str(uuid5(NAMESPACE_URL, f"mlops-batch:{batch_id}:{line_number}"))
            yield task_id, payload


class StatsReporter:
    def __init__(self, stream: TextIO | None, interval: float = 1.0) -> None:
        self.stream = stream
        self.interval = interval
        self.started = perf_counter()
        self.last_report = self.started
        self.last_published = 0

    def __call__(self, stats: SourceStats, *, final: bool = False) -> None:
        now = perf_counter()
        elapsed = now - self.started
        window = now - self.last_report
        if not final and window < self.interval:
            return
        report = {
            "event": "complete" if final else "progress",
            "elapsed_seconds": round(elapsed, 6),
            "published_total": stats.published_total,
            "retries_total": stats.retries_total,
            "source_rate": stats.published_total / elapsed if elapsed else 0.0,
            "window_source_rate": (
                (stats.published_total - self.last_published) / window if window else 0.0
            ),
        }
        if self.stream is not None:
            self.stream.write(json.dumps(report) + "\n")
            self.stream.flush()
        logger.info(
            "source.complete" if final else "source.progress",
            extra={key: value for key, value in report.items() if key != "event"},
        )
        self.last_report = now
        self.last_published = stats.published_total


async def publish_batch(
    messages: Iterable[tuple[str, dict[str, Any]]],
    broker: RabbitBroker,
    settings: Settings,
    *,
    reporter: StatsReporter | None = None,
    events_stream: TextIO | None = None,
    batch_id: str | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SourceStats:
    """Retry the current task without accumulating a backlog in memory"""
    stats = SourceStats()
    for task_id, payload in messages:
        with (
            task_context(task_id, batch_id),
            tracer.start_as_current_span(
                "source.task",
                attributes={"task.id": task_id},
                record_exception=False,
                set_status_on_exception=False,
            ),
        ):
            delay = min(
                settings.source_retry_delay_seconds, settings.source_retry_max_delay_seconds
            )
            while True:
                try:
                    await publish_task(broker, settings, task_id, payload)
                except (AMQPError, OSError, TimeoutError) as exc:
                    stats.retries_total += 1
                    source_retries.add(1)
                    reason = publish_failure(exc)
                    if reason == "nack":
                        source_nacks.add(1)
                    with tracer.start_as_current_span(
                        "source.backoff",
                        attributes={
                            "retry.reason": reason,
                            "retry.delay_seconds": delay,
                        },
                    ) as span:
                        error_on_span(span, exc)
                        logger.warning(
                            "source.backoff", extra={"reason": reason, "delay_seconds": delay}
                        )
                        if reporter is not None:
                            reporter(stats)
                        started = perf_counter()
                        try:
                            await sleep(delay)
                        finally:
                            source_wait.add(perf_counter() - started)
                    delay = min(delay * 2, settings.source_retry_max_delay_seconds)
                else:
                    stats.published_total += 1
                    source_published.add(1)
                    logger.info("source.published")
                    if events_stream is not None:
                        events_stream.write(
                            json.dumps(
                                {
                                    "event": "published",
                                    "task_id": task_id,
                                    "published_at": datetime.now(UTC).isoformat(),
                                }
                            )
                            + "\n"
                        )
                        events_stream.flush()
                    if reporter is not None:
                        reporter(stats)
                    break
    with task_context(None, batch_id):
        if reporter is not None:
            reporter(stats, final=True)
        return stats


async def run_source(
    path: Path,
    settings: Settings,
    *,
    batch_id: str | None = None,
    stats_stream: TextIO | None = None,
    events_stream: TextIO | None = None,
) -> SourceStats:
    configure_logging(settings.log_level, "source")
    providers = start_telemetry("source", settings)
    broker = RabbitBroker(
        settings.rabbitmq_url,
        default_channel=Channel(on_return_raises=True),
        logger=logging.getLogger("mlops_labs.faststream.rabbit"),
    )
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    if current is not None and not hasattr(signal, "CTRL_BREAK_EVENT"):
        loop.add_signal_handler(signal.SIGTERM, current.cancel)
    try:
        if batch_id is None:
            with path.open("rb") as stream:
                batch_id = hashlib.file_digest(stream, "sha256").hexdigest()
        with task_context(None, batch_id):
            logger.info("source.started")
        for counter in (source_published, source_nacks, source_retries, source_wait):
            counter.add(0)
        if providers is not None:
            await asyncio.to_thread(providers[1].force_flush, timeout_millis=1000)
        await broker.connect()
        await declare_topology(broker, settings)
        return await publish_batch(
            read_task_messages(path, batch_id),
            broker,
            settings,
            reporter=StatsReporter(stats_stream),
            events_stream=events_stream,
            batch_id=batch_id,
        )
    except Exception:
        logger.exception("source.failed")
        raise
    finally:
        logger.info("source.shutdown_started")
        await broker.stop()
        await stop_telemetry(providers)
        logger.info("source.stopped")
        if current is not None and not hasattr(signal, "CTRL_BREAK_EVENT"):
            loop.remove_signal_handler(signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL ProcessRequest payloads")
    parser.add_argument("--batch-id", help="Reuse this ID only for the same immutable input batch")
    parser.add_argument("--stats-file", type=Path, help="Write streaming JSONL counters here")
    parser.add_argument("--events-file", type=Path, help="Optional per-task publication trace")
    args = parser.parse_args()
    settings = get_settings()
    with ExitStack() as stack:
        stats_stream = None
        events_stream = None
        if args.stats_file is not None:
            args.stats_file.parent.mkdir(parents=True, exist_ok=True)
            stats_stream = stack.enter_context(args.stats_file.open("w", encoding="utf-8"))
        if args.events_file is not None:
            args.events_file.parent.mkdir(parents=True, exist_ok=True)
            events_stream = stack.enter_context(args.events_file.open("w", encoding="utf-8"))
        try:
            asyncio.run(
                run_source(
                    args.input,
                    settings,
                    batch_id=args.batch_id,
                    stats_stream=stats_stream,
                    events_stream=events_stream,
                )
            )
        except asyncio.CancelledError:
            return
        except Exception:
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
