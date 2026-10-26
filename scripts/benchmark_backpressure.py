"""Measure backpressure with isolated services and a synthetic model

Run with the dev/web environment and Docker available:
    python scripts/benchmark_backpressure.py --output reports/backpressure
    python scripts/benchmark_backpressure.py --capacity-study --repeats 2
"""

import argparse
import asyncio
import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, suppress
from datetime import UTC, datetime
from math import ceil, isfinite
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx
import mlflow.sklearn
import numpy as np
import psutil
from sklearn.dummy import DummyRegressor
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

import mlflow
from mlops_labs.core.config import get_settings
from mlops_labs.db.base import create_session_factory, init_db
from mlops_labs.db.models import TaskRecord
from mlops_labs.domain.tasks import TaskStatus
from mlops_labs.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("unbounded_1_worker", 0, 0, 1),
    ("bounded_1_worker", 10, 1, 1),
    ("bounded_2_workers", 10, 1, 2),
)


def queue_capacity_basis(
    delay_ms: float, retry_seconds: float, workers: int = 2, concurrency: int = 1
) -> int:
    """Size the buffer for one maximum source retry pause"""
    if not isfinite(delay_ms) or delay_ms <= 0:
        raise ValueError("Inference delay must be positive and finite")
    if not isfinite(retry_seconds) or retry_seconds <= 0:
        raise ValueError("Retry sleep must be positive and finite")
    for slots in (workers, concurrency):
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
            raise ValueError("Workers and concurrency must be positive integers")
    # Round per slot for simultaneous worker completions
    return workers * concurrency * ceil(retry_seconds / (delay_ms / 1000))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # The last live-log line may be incomplete
            continue
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def process_rss(process: subprocess.Popen[Any]) -> int:
    try:
        parent = psutil.Process(process.pid)
        processes = [parent, *parent.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    total = 0
    for child in processes:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            total += int(child.memory_info().rss)
    return total


def start_process(command: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen[Any]:
    with log_path.open("wb") as log:
        return subprocess.Popen(  # noqa: S603 - fixed Python executable and argument list
            [sys.executable, *command],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


async def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        # Windows venv launchers can own a child Python process
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in reversed(children):
            with suppress(psutil.NoSuchProcess):
                child.terminate()
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, timeout=5)
        _, alive = await asyncio.to_thread(psutil.wait_procs, children, timeout=10)
        for child in alive:
            with suppress(psutil.NoSuchProcess):
                child.kill()


def register_fixture(directory: Path) -> tuple[str, str]:
    tracking_uri = f"sqlite:///{(directory / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.create_experiment(
        "backpressure-fixture", artifact_location=(directory / "artifacts").as_uri()
    )
    mlflow.set_experiment("backpressure-fixture")
    model = DummyRegressor(strategy="mean").fit(
        np.ones((3, len(FEATURE_NAMES))), np.array([40.0, 42.0, 44.0])
    )
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model", pip_requirements=[])
        version = mlflow.register_model(
            f"runs:/{run.info.run_id}/model", "backpressure-fixture-model"
        )
    client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    client.set_registered_model_alias("backpressure-fixture-model", "champion", version.version)
    # Release cached engines so Windows can delete the temporary SQLite file
    for store in (client._tracking_client.store, client._get_registry_client().store):
        if store.db_uri == tracking_uri:
            store.engine.dispose()
    return tracking_uri, "models:/backpressure-fixture-model@champion"


async def wait_http(
    client: httpx.AsyncClient,
    url: str,
    process: subprocess.Popen[Any] | None = None,
    log: Path | None = None,
    auth: tuple[str, str] | None = None,
) -> None:
    deadline = asyncio.get_running_loop().time() + 90
    while asyncio.get_running_loop().time() < deadline:
        if process is not None and process.poll() is not None:
            detail = (
                await asyncio.to_thread(log.read_text, encoding="utf-8")
                if log
                else "Process exited"
            )
            raise RuntimeError(detail)
        try:
            if (await client.get(url, auth=auth, timeout=1)).is_success:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.2)
    raise TimeoutError(f"Service did not become ready: {url}")


async def run_case(
    case: tuple[str, int, int, int],
    args: argparse.Namespace,
    rabbit: Any,
    database_url: str,
    tracking_uri: str,
    model_uri: str,
    input_path: Path,
    client: httpx.AsyncClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    name, max_length, prefetch, worker_count = case
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    queue = f"benchmark.{uuid4().hex}"
    rabbit_host = rabbit.get_container_host_ip()
    management = f"http://{rabbit_host}:{rabbit.get_exposed_port(15672)}"
    env = os.environ.copy()
    env.update(
        {
            "DATABASE_URL": database_url,
            "MLFLOW_TRACKING_URI": tracking_uri,
            "MLFLOW_REGISTRY_URI": tracking_uri,
            "MODEL_URI": model_uri,
            "RABBITMQ_URL": f"amqp://lab:lab@{rabbit_host}:{rabbit.get_exposed_port(5672)}/",
            "RABBITMQ_QUEUE": queue,
            "RABBITMQ_DLQ_QUEUE": f"{queue}.dlq",
            "RABBITMQ_DLX_EXCHANGE": f"{queue}.dlx",
            "RABBITMQ_EXCHANGE": f"{queue}.exchange",
            "RABBITMQ_MAX_QUEUE_LENGTH": str(max_length),
            "RABBITMQ_PREFETCH_COUNT": str(prefetch),
            "WORKER_CONCURRENCY": "1",
            "INFERENCE_DELAY_MS": str(args.delay_ms),
            "LOG_LEVEL": "WARNING",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if args.capacity_study:
        # Stress each capacity with the maximum retry pause
        env["SOURCE_RETRY_DELAY_SECONDS"] = str(args.capacity_retry_seconds)
        env["SOURCE_RETRY_MAX_DELAY_SECONDS"] = str(args.capacity_retry_seconds)
    engine = create_async_engine(database_url)
    await init_db(engine)
    async with engine.begin() as connection:
        await connection.execute(delete(TaskRecord))
    sessions = create_session_factory(engine)
    workers: list[subprocess.Popen[Any]] = []
    source: subprocess.Popen[Any] | None = None
    samples: list[dict[str, Any]] = []
    baseline_rss = 0
    try:
        for index in range(worker_count):
            port = free_port()
            worker_env = {**env, "WORKER_HEALTH_PORT": str(port)}
            log = output / f"worker_{index + 1}.log"
            process = start_process(["-m", "mlops_labs.worker"], worker_env, log)
            workers.append(process)
            await wait_http(client, f"http://127.0.0.1:{port}/healthz", process, log)
        baseline_rss = sum(process_rss(process) for process in workers)
        started = time.monotonic()
        source = start_process(
            [
                "-m",
                "mlops_labs.source",
                "--input",
                str(input_path),
                "--batch-id",
                queue,
                "--stats-file",
                str(output / "source_stats.jsonl"),
                "--events-file",
                str(output / "source_events.jsonl"),
            ],
            env,
            output / "source.log",
        )
        deadline = started + args.timeout
        previous_accepted = 0
        previous_sample = started
        next_progress_seconds = 10.0
        while time.monotonic() < deadline:
            elapsed = time.monotonic() - started
            for index, process in enumerate(workers):
                if process.poll() is not None:
                    raise RuntimeError((output / f"worker_{index + 1}.log").read_text("utf-8"))
            queue_response = await client.get(
                f"{management}/api/queues/%2F/{quote(queue, safe='')}", auth=("lab", "lab")
            )
            queue_response.raise_for_status()
            queue_data = queue_response.json()
            events = read_jsonl(output / "source_events.jsonl")
            stats = read_jsonl(output / "source_stats.jsonl")
            accepted = len(events)
            now = time.monotonic()
            worker_rss = sum(process_rss(process) for process in workers)
            async with sessions() as session:
                counts = dict(
                    (
                        await session.execute(
                            select(TaskRecord.status, func.count()).group_by(TaskRecord.status)
                        )
                    ).all()
                )
                completed = counts.get(TaskStatus.COMPLETED, 0)
                failed = counts.get(TaskStatus.FAILED, 0)
            if failed:
                raise RuntimeError(f"{name}: {failed} tasks failed; inspect worker logs")
            samples.append(
                {
                    "case": name,
                    "elapsed_seconds": round(elapsed, 4),
                    "queue_ready": queue_data.get("messages_ready", 0),
                    "queue_unacked": queue_data.get("messages_unacknowledged", 0),
                    "worker_rss_bytes": worker_rss,
                    "worker_rss_growth_bytes": max(0, worker_rss - baseline_rss),
                    "source_rss_bytes": process_rss(source),
                    "accepted_total": accepted,
                    "source_tasks_per_second": round(
                        (accepted - previous_accepted) / max(now - previous_sample, 0.000001), 4
                    ),
                    "source_retries": stats[-1]["retries_total"] if stats else 0,
                    "completed_total": completed,
                }
            )
            previous_accepted, previous_sample = accepted, now
            if args.capacity_study and elapsed >= next_progress_seconds:
                print(
                    f"{name}: {elapsed:.0f}s, published={accepted}/{args.tasks}, "
                    f"completed={completed}/{args.tasks}, "
                    f"ready={queue_data.get('messages_ready', 0)}, "
                    f"unacked={queue_data.get('messages_unacknowledged', 0)}",
                    flush=True,
                )
                next_progress_seconds = elapsed + 10
            if source.poll() is not None:
                if source.returncode:
                    raise RuntimeError((output / "source.log").read_text("utf-8"))
                if completed == args.tasks:
                    break
            await asyncio.sleep(args.sample_interval)
        else:
            raise TimeoutError(f"{name} did not complete within {args.timeout}s")

        published = {
            event["task_id"]: datetime.fromisoformat(event["published_at"])
            for event in read_jsonl(output / "source_events.jsonl")
        }
        async with sessions() as session:
            records = (
                await session.execute(
                    select(
                        TaskRecord.task_id,
                        TaskRecord.status,
                        TaskRecord.attempts,
                        TaskRecord.updated_at,
                        TaskRecord.model_version,
                    )
                )
            ).all()
            if len(records) != args.tasks or {row.task_id for row in records} != set(published):
                raise RuntimeError(f"{name}: accepted tasks and completed records do not match")
            if any(row.status.value != "COMPLETED" or row.attempts != 1 for row in records):
                raise RuntimeError(f"{name}: expected exactly one successful execution per task")
            tasks = []
            for record in records:
                finished = record.updated_at.replace(tzinfo=UTC)
                tasks.append(
                    {
                        "case": name,
                        "task_id": record.task_id,
                        "accepted_at": published[record.task_id].isoformat(),
                        "completed_at": finished.isoformat(),
                        "accepted_to_complete_seconds": round(
                            max(0, (finished - published[record.task_id]).total_seconds()), 6
                        ),
                        "attempts": record.attempts,
                        "model_version": record.model_version,
                    }
                )
        latencies = [task["accepted_to_complete_seconds"] for task in tasks]
        stats = read_jsonl(output / "source_stats.jsonl")
        processing_window = (
            max(datetime.fromisoformat(task["completed_at"]) for task in tasks)
            - min(published.values())
        ).total_seconds()
        peak_ready = max(sample["queue_ready"] for sample in samples)
        peak_unacked = max(sample["queue_unacked"] for sample in samples)
        if max_length and peak_ready > max_length:
            raise RuntimeError(f"{name}: observed queue capacity violation")
        if prefetch and peak_unacked > prefetch * worker_count:
            raise RuntimeError(f"{name}: observed prefetch capacity violation")
        steady_window = None
        interior = [
            sample
            for sample in samples
            if max(30, max_length * 3) <= sample["accepted_total"] <= args.tasks - max_length * 2
        ]
        if max_length and len(interior) >= 2:
            first, last = interior[0], interior[-1]
            steady_seconds = last["elapsed_seconds"] - first["elapsed_seconds"]
            if steady_seconds > 0:
                steady_window = {
                    "start_seconds": first["elapsed_seconds"],
                    "end_seconds": last["elapsed_seconds"],
                    "duration_seconds": round(steady_seconds, 4),
                    "samples": len(interior),
                    "start_accepted": first["accepted_total"],
                    "end_accepted": last["accepted_total"],
                    "start_completed": first["completed_total"],
                    "end_completed": last["completed_total"],
                    "source_tasks_per_second": round(
                        (last["accepted_total"] - first["accepted_total"]) / steady_seconds, 4
                    ),
                    "completion_tasks_per_second": round(
                        (last["completed_total"] - first["completed_total"]) / steady_seconds, 4
                    ),
                }
        summary = {
            "case": name,
            "queue_limit": max_length,
            "prefetch": prefetch,
            "workers": worker_count,
            "accepted": len(published),
            "completed": len(tasks),
            "duration_seconds": round(time.monotonic() - started, 3),
            "processing_window_seconds": round(processing_window, 4),
            "completion_throughput_tasks_per_second": round(len(tasks) / processing_window, 4),
            "source_duration_seconds": stats[-1]["elapsed_seconds"],
            "source_accepted_tasks_per_second": stats[-1]["source_rate"],
            "steady_window": steady_window,
            "peak_queue_ready": peak_ready,
            "peak_queue_unacked": peak_unacked,
            "peak_worker_rss_bytes": max(sample["worker_rss_bytes"] for sample in samples),
            "peak_worker_rss_growth_bytes": max(
                sample["worker_rss_growth_bytes"] for sample in samples
            ),
            "peak_source_rss_bytes": max(sample["source_rss_bytes"] for sample in samples),
            "source_retries": stats[-1]["retries_total"],
            "latency_median_seconds": round(float(np.median(latencies)), 4),
            "latency_p95_seconds": round(float(np.percentile(latencies, 95)), 4),
        }
        write_csv(output / "timeseries.csv", samples)
        write_csv(output / "tasks.csv", tasks)
        return samples, tasks, summary
    finally:
        if source is not None:
            await stop_process(source)
        for process in workers:
            await stop_process(process)
        await engine.dispose()


def plot_results(output: Path, samples: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    for name, *_ in CASES:
        selected = [sample for sample in samples if sample["case"] == name]
        seconds = [sample["elapsed_seconds"] for sample in selected]
        line = axes[0, 0].plot(
            seconds, [sample["queue_ready"] for sample in selected], label=f"{name}: ready"
        )[0]
        axes[0, 0].plot(
            seconds,
            [sample["queue_unacked"] for sample in selected],
            linestyle="--",
            color=line.get_color(),
            label=f"{name}: unacked",
        )
        axes[0, 1].plot(
            seconds,
            [sample["worker_rss_growth_bytes"] / 1024**2 for sample in selected],
            label=name,
        )
        axes[0, 2].plot(
            seconds,
            [sample["source_rss_bytes"] / 1024**2 for sample in selected],
            label=name,
        )
        axes[1, 0].plot(
            seconds, [sample["source_tasks_per_second"] for sample in selected], label=name
        )
        case_tasks = sorted(
            (task for task in tasks if task["case"] == name), key=lambda task: task["accepted_at"]
        )
        origin = datetime.fromisoformat(case_tasks[0]["accepted_at"])
        axes[1, 1].scatter(
            [
                (datetime.fromisoformat(task["accepted_at"]) - origin).total_seconds()
                for task in case_tasks
            ],
            [task["accepted_to_complete_seconds"] for task in case_tasks],
            s=12,
            alpha=0.65,
            label=name,
        )
        axes[1, 2].plot(seconds, [sample["completed_total"] for sample in selected], label=name)
    axes[0, 0].set(title="Broker backlog (ready / unacked)", ylabel="Messages")
    axes[0, 1].set(title="Worker RSS growth after startup", ylabel="MiB")
    axes[0, 2].set(title="Source process RSS (0 after exit)", ylabel="MiB")
    axes[1, 0].set(title="Confirmed source publication rate", ylabel="Tasks / second")
    axes[1, 1].set(title="Confirmed publication to committed result", ylabel="Seconds")
    axes[1, 2].set(title="Committed results over time", ylabel="Completed tasks")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("Seconds")
        axis.legend(fontsize=7)
    axes[1, 1].set_xlabel("Publication time since first accepted task, seconds")
    figure.tight_layout()
    figure.savefig(output / "backpressure.png", dpi=150)
    plt.close(figure)


def capacity_results(summaries: list[dict[str, Any]], capacity: int) -> list[dict[str, Any]]:
    result = []
    for limit in sorted({summary["queue_limit"] for summary in summaries}):
        runs = [summary for summary in summaries if summary["queue_limit"] == limit]
        result.append(
            {
                "queue_limit": limit,
                "formula_recommendation": limit == capacity,
                "repeats": len(runs),
                "accepted": sum(run["accepted"] for run in runs),
                "completed": sum(run["completed"] for run in runs),
                "median_completion_throughput_tasks_per_second": round(
                    float(
                        np.median([run["completion_throughput_tasks_per_second"] for run in runs])
                    ),
                    4,
                ),
                "median_latency_p95_seconds": round(
                    float(np.median([run["latency_p95_seconds"] for run in runs])), 4
                ),
                "worst_latency_p95_seconds": max(run["latency_p95_seconds"] for run in runs),
                "peak_queue_ready": max(run["peak_queue_ready"] for run in runs),
                "source_retries": sum(run["source_retries"] for run in runs),
                "throughput_by_repeat": [
                    run["completion_throughput_tasks_per_second"] for run in runs
                ],
                "latency_p95_by_repeat": [run["latency_p95_seconds"] for run in runs],
            }
        )
    return result


def plot_capacity_results(output: Path, results: list[dict[str, Any]], capacity: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    limits = [row["queue_limit"] for row in results]
    axes[0].plot(
        limits,
        [row["median_completion_throughput_tasks_per_second"] for row in results],
        "o-",
        label="Median across repeats",
    )
    axes[1].plot(
        limits,
        [row["median_latency_p95_seconds"] for row in results],
        "o-",
        label="Median p95 across repeats",
    )
    for row in results:
        axes[0].scatter(
            [row["queue_limit"]] * row["repeats"],
            row["throughput_by_repeat"],
            color="gray",
            alpha=0.6,
            s=15,
        )
        axes[1].scatter(
            [row["queue_limit"]] * row["repeats"],
            row["latency_p95_by_repeat"],
            color="gray",
            alpha=0.6,
            s=15,
        )
    axes[0].set(title="Two workers: measured throughput", ylabel="Completed tasks / second")
    axes[1].set(title="Confirmed publication to committed result", ylabel="p95 latency, seconds")
    for axis in axes:
        axis.axvline(capacity, linestyle="--", color="tab:red", label=f"Formula Q={capacity}")
        axis.set_xlabel("Queue capacity (ready messages)")
        axis.set_xticks(limits)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Capacity study: fixed maximum source retry sleep; synthetic inference")
    figure.tight_layout()
    figure.savefig(output / "capacity.png", dpi=150)
    plt.close(figure)


async def benchmark(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(CASES)
    if args.capacity_study:
        args.capacity_retry_seconds = get_settings().source_retry_max_delay_seconds
        capacity = queue_capacity_basis(args.delay_ms, args.capacity_retry_seconds)
        limits = sorted({1, max(1, capacity // 2), capacity, capacity * 2})
        cases = [
            (f"capacity_{limit}_repeat_{repeat}", limit, 1, 2)
            for limit in limits
            for repeat in range(1, args.repeats + 1)
        ]
        print(
            f"Capacity basis: 2 workers * 1 slot * ceil({args.capacity_retry_seconds}s "
            f"/ ({args.delay_ms}ms / 1000)) = {capacity}; candidates={limits}, "
            f"repeats={args.repeats}. Source retry sleep is fixed at its configured maximum.",
            flush=True,
        )
    with (
        tempfile.TemporaryDirectory(prefix="mlops-backpressure-") as temporary,
        ExitStack() as stack,
    ):
        directory = Path(temporary)
        input_path = directory / "tasks.jsonl"
        with input_path.open("w", encoding="utf-8") as stream:
            for index in range(args.tasks):
                stream.write(
                    json.dumps(
                        {
                            "recent_sales": [4000.0 + index % 7] * args.history_days,
                            "day_of_week": index % 7,
                            "is_snap": index % 2,
                        }
                    )
                    + "\n"
                )
        tracking_uri, model_uri = register_fixture(directory)
        rabbit = stack.enter_context(
            DockerContainer("rabbitmq:3.13-management-alpine")
            .with_env("RABBITMQ_DEFAULT_USER", "lab")
            .with_env("RABBITMQ_DEFAULT_PASS", "lab")
            .with_env(
                "RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS", "-rabbit collect_statistics_interval 1000"
            )
            .with_exposed_ports(5672, 15672)
        )
        postgres = stack.enter_context(
            DockerContainer("postgres:16-alpine")
            .with_env("POSTGRES_USER", "lab")
            .with_env("POSTGRES_PASSWORD", "lab")
            .with_env("POSTGRES_DB", "lab")
            .with_exposed_ports(5432)
        )
        wait_for_logs(postgres, "database system is ready to accept connections")
        pg_deadline = time.monotonic() + 30
        while time.monotonic() < pg_deadline:
            check = await asyncio.to_thread(
                postgres.get_wrapped_container().exec_run,
                ["pg_isready", "-h", "127.0.0.1", "-U", "lab", "-d", "lab"],
            )
            if check.exit_code == 0:
                break
            await asyncio.sleep(0.2)
        else:
            raise TimeoutError("Isolated PostgreSQL did not become ready")
        database_url = (
            f"postgresql+asyncpg://lab:lab@{postgres.get_container_host_ip()}:"
            f"{postgres.get_exposed_port(5432)}/lab"
        )
        samples: list[dict[str, Any]] = []
        tasks: list[dict[str, Any]] = []
        summaries = []
        async with httpx.AsyncClient(timeout=10) as client:
            await wait_http(
                client,
                f"http://{rabbit.get_container_host_ip()}:{rabbit.get_exposed_port(15672)}/api/overview",
                auth=("lab", "lab"),
            )
            for index, case in enumerate(cases, start=1):
                print(f"Running {index}/{len(cases)}: {case[0]}...", flush=True)
                case_samples, case_tasks, summary = await run_case(
                    case, args, rabbit, database_url, tracking_uri, model_uri, input_path, client
                )
                samples.extend(case_samples)
                tasks.extend(case_tasks)
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
        prefix = "capacity_" if args.capacity_study else ""
        write_csv(args.output / f"{prefix}timeseries.csv", samples)
        write_csv(args.output / f"{prefix}tasks.csv", tasks)
        report = {
            "measured_at": datetime.now(UTC).isoformat(),
            "python_version": sys.version,
            "tasks_per_case": args.tasks,
            "history_days_per_task": args.history_days,
            "inference_delay_ms": args.delay_ms,
            "sample_interval_seconds": args.sample_interval,
            "model": "Synthetic DummyRegressor fixture registered with champion alias",
            "latency_definition": "Confirmed publication to committed COMPLETED result; excludes waiting before source publication",
            "memory_definition": "Sum of worker process-tree RSS (including Windows venv redirector children); growth relative to post-startup RSS",
            "throughput_definition": "Completed task count divided by time from first confirmed publication to last committed result",
            "steady_window_definition": "Bounded cases only: sampled accepted count between max(30,3*queue_limit) and tasks-2*queue_limit; source and completion rates use the same first/last samples. Null when no adequate interior window exists.",
            "limitations": [
                "Synthetic benchmark measures infrastructure, not forecasting quality",
                "Management counters are sampled and may miss short peaks",
                "Repeated local runs are illustrative; remeasure for production sizing"
                if args.capacity_study
                else "Each setting is run once; repeat for stable production sizing",
                "Workers are host Python processes; Linux SIGTERM is checked separately by Docker integration test",
            ],
            "cases": summaries,
        }
        if args.capacity_study:
            results = capacity_results(summaries, capacity)
            report["selection_basis"] = {
                "principle": "Choose the minimum ready-message buffer covering the maximum intentional source retry sleep at the configured aggregate service-rate upper bound",
                "formula": "workers * concurrency_per_worker * ceil(max_retry_sleep_seconds / (inference_delay_ms / 1000))",
                "rounding": "Round separately per execution slot to cover simultaneous worker completions",
                "workers": 2,
                "worker_count_basis": "The assignment compares one worker with horizontal scaling to two workers",
                "concurrency_per_worker": 1,
                "prefetch_per_worker": 1,
                "queue_limit_definition": "Ready messages only; in-flight messages are limited separately by prefetch",
                "max_retry_sleep_seconds": args.capacity_retry_seconds,
                "minimum_service_seconds": args.delay_ms / 1000,
                "configured_aggregate_rate_upper_bound_tasks_per_second": 2
                / (args.delay_ms / 1000),
                "recommended_queue_limit": capacity,
                "minimum_ready_capacity_for_retry_sleep": capacity,
                "candidates": limits,
                "repeats_per_capacity": args.repeats,
                "source_retry_mode": "Both initial and maximum retry sleeps fixed to the configured maximum for this study only",
                "selection_is_measurement_optimization": False,
            }
            report["capacity_results"] = results
            report["limitations"].extend(
                [
                    "The sizing principle covers intentional retry sleep; publish confirmation, network delay, outages, and retries can extend the complete reaction interval",
                    "This is a conservative lab sizing rule, not a globally optimal capacity or a latency SLA",
                    "Rate upper bound uses the artificial minimum inference duration; repeat sizing when delay, concurrency, worker count, or source retry settings change",
                    "Normal adaptive backoff is evaluated by the ordinary benchmark; this study deliberately fixes the longest configured sleep",
                    "A larger capacity may improve observed throughput while increasing latency; measurements are reported without overriding the stated sizing principle",
                ]
            )
            for row in results:
                print(
                    f"Q={row['queue_limit']}: median throughput="
                    f"{row['median_completion_throughput_tasks_per_second']}/s, "
                    f"median p95={row['median_latency_p95_seconds']}s, "
                    f"completed={row['completed']}/{row['accepted']}",
                    flush=True,
                )
        filename = "capacity_study.json" if args.capacity_study else "summary.json"
        (args.output / filename).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.capacity_study:
            plot_capacity_results(args.output, results, capacity)
        else:
            plot_results(args.output, samples, tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "backpressure")
    parser.add_argument("--tasks", type=int, default=120)
    parser.add_argument("--history-days", type=int, default=4096)
    parser.add_argument("--delay-ms", type=float, default=200)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--capacity-study",
        action="store_true",
        help="Compare ready-message limits derived from the source retry sleep; leaves ordinary reports intact",
    )
    parser.add_argument("--repeats", type=int, default=2, help="Runs per capacity in study mode")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if (
        args.tasks < 10
        or args.history_days < 28
        or not isfinite(args.delay_ms)
        or args.delay_ms <= 0
    ):
        parser.error("Use at least 10 tasks, 28 history days, and a positive inference delay")
    if args.sample_interval <= 0 or args.timeout <= 0:
        parser.error("Sampling interval and timeout must be positive")
    if args.repeats < 1:
        parser.error("Repeats must be at least one")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
