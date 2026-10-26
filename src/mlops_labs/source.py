"""Stream JSONL tasks with backpressure from RabbitMQ"""

import argparse
import asyncio
import hashlib
import json
import sys
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
from mlops_labs.broker.publishing import publish_task
from mlops_labs.broker.topology import declare_topology
from mlops_labs.core.config import Settings, get_settings


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
            except ValueError as exc:
                raise ValueError(f"Invalid task at {path}:{line_number}: {exc}") from exc
            task_id = str(uuid5(NAMESPACE_URL, f"mlops-batch:{batch_id}:{line_number}"))
            yield task_id, payload


class StatsReporter:
    def __init__(self, stream: TextIO, interval: float = 1.0) -> None:
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
        self.stream.write(json.dumps(report) + "\n")
        self.stream.flush()
        self.last_report = now
        self.last_published = stats.published_total


async def publish_batch(
    messages: Iterable[tuple[str, dict[str, Any]]],
    broker: RabbitBroker,
    settings: Settings,
    *,
    reporter: StatsReporter | None = None,
    events_stream: TextIO | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SourceStats:
    """Retry the current task without accumulating a backlog in memory"""
    stats = SourceStats()
    for task_id, payload in messages:
        delay = min(settings.source_retry_delay_seconds, settings.source_retry_max_delay_seconds)
        while True:
            try:
                await publish_task(broker, settings, task_id, payload)
            except (AMQPError, OSError, TimeoutError):
                stats.retries_total += 1
                if reporter is not None:
                    reporter(stats)
                await sleep(delay)
                delay = min(delay * 2, settings.source_retry_max_delay_seconds)
            else:
                stats.published_total += 1
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
    if reporter is not None:
        reporter(stats, final=True)
    return stats


async def run_source(
    path: Path,
    settings: Settings,
    *,
    batch_id: str | None = None,
    stats_stream: TextIO = sys.stdout,
    events_stream: TextIO | None = None,
) -> SourceStats:
    broker = RabbitBroker(
        settings.rabbitmq_url,
        default_channel=Channel(on_return_raises=True),
    )
    try:
        await broker.connect()
        await declare_topology(broker, settings)
        return await publish_batch(
            read_task_messages(path, batch_id),
            broker,
            settings,
            reporter=StatsReporter(stats_stream),
            events_stream=events_stream,
        )
    finally:
        await broker.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL ProcessRequest payloads")
    parser.add_argument("--batch-id", help="Reuse this ID only for the same immutable input batch")
    parser.add_argument("--stats-file", type=Path, help="Write streaming JSONL counters here")
    parser.add_argument("--events-file", type=Path, help="Optional per-task publication trace")
    args = parser.parse_args()
    settings = get_settings()
    with ExitStack() as stack:
        stats_stream = sys.stdout
        events_stream = None
        if args.stats_file is not None:
            args.stats_file.parent.mkdir(parents=True, exist_ok=True)
            stats_stream = stack.enter_context(args.stats_file.open("w", encoding="utf-8"))
        if args.events_file is not None:
            args.events_file.parent.mkdir(parents=True, exist_ok=True)
            events_stream = stack.enter_context(args.events_file.open("w", encoding="utf-8"))
        asyncio.run(
            run_source(
                args.input,
                settings,
                batch_id=args.batch_id,
                stats_stream=stats_stream,
                events_stream=events_stream,
            )
        )


if __name__ == "__main__":
    main()
