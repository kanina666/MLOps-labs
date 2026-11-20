"""Prepare synthetic data and reproduce the observability scenarios on the Compose stack."""

import argparse
import asyncio
import io
import json
import logging
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import mlflow.pyfunc
import pandas as pd
from faststream.rabbit import Channel, RabbitBroker
from mlflow.exceptions import MlflowException

from mlflow import MlflowClient
from mlops_labs.broker.publishing import publish_task
from mlops_labs.broker.topology import declare_topology
from mlops_labs.core.config import get_settings
from mlops_labs.core.logging import configure_logging
from mlops_labs.core.telemetry import start_telemetry, stop_telemetry, task_context, tracer

logger = logging.getLogger(__name__)
MODEL_NAME = "mlops_observability_demo"


class DemoModel(mlflow.pyfunc.PythonModel):
    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> list[float]:
        if (model_input["lag_1"] == -9999).any():
            raise RuntimeError("Demo inference failure")
        return [42.0] * len(model_input)


def payload(history_length: int, *, poison: bool = False) -> dict[str, Any]:
    history = [10.0] * history_length
    if poison:
        history[-1] = -9999.0
    return {"recent_sales": history, "day_of_week": 3, "is_snap": 0}


def register_model(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException as exc:
        if (
            exc.error_code != "RESOURCE_DOES_NOT_EXIST"
            and exc.error_code != "INVALID_PARAMETER_VALUE"
        ):
            raise
    else:
        logger.info("demo.model_exists", extra={"model_version": version.version})
        return

    mlflow.set_experiment("observability-demo")
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=DemoModel(), pip_requirements=[])
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
    logger.info("demo.model_registered", extra={"model_version": version.version})


def write_batch(path: Path, tasks: int, history_length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload(history_length), separators=(",", ":")) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for _ in range(tasks):
            stream.write(line)


async def publish_poison(*, malformed: bool) -> None:
    settings = get_settings()
    providers = start_telemetry("demo", settings)
    broker = RabbitBroker(settings.rabbitmq_url, default_channel=Channel(on_return_raises=True))
    try:
        await broker.connect()
        await declare_topology(broker, settings)
        task_id = str(uuid4())
        body = payload(28, poison=True)
        if malformed:
            body["day_of_week"] = 100
        with task_context(task_id), tracer.start_as_current_span("demo.poison"):
            await publish_task(broker, settings, task_id, body)
            logger.info("demo.poison_published")
    finally:
        await broker.stop()
        await stop_telemetry(providers)


async def load_api(url: str, requests: int, concurrency: int) -> None:
    submitted = payload(28)
    async with httpx.AsyncClient(timeout=4) as client:

        async def send_requests(slot: int) -> None:
            for _ in range(slot, requests, concurrency):
                response = await client.post(f"{url.rstrip('/')}/api/v1/tasks", json=submitted)
                logger.info("demo.api_response", extra={"status_code": response.status_code})

        await asyncio.gather(*(send_requests(slot) for slot in range(concurrency)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    init = actions.add_parser("init")
    init.add_argument("--tracking-uri", default="http://localhost:5000")
    batch = actions.add_parser("batch")
    batch.add_argument("--path", type=Path, default=Path("data/batch.jsonl"))
    batch.add_argument("--tasks", type=int, default=6000)
    batch.add_argument("--history-length", type=int, default=4096)
    poison = actions.add_parser("poison")
    poison.add_argument("--malformed", action="store_true")
    load = actions.add_parser("api")
    load.add_argument("--url", default="http://localhost:8000")
    load.add_argument("--requests", type=int, default=200)
    load.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    configure_logging(get_settings().log_level, "demo")
    if args.action == "init":
        with redirect_stdout(io.StringIO()):
            register_model(args.tracking_uri)
    elif args.action == "batch":
        if args.tasks < 1 or args.history_length < 28:
            parser.error("Use at least one task and at least 28 values of history")
        write_batch(args.path, args.tasks, args.history_length)
    elif args.action == "poison":
        asyncio.run(publish_poison(malformed=args.malformed))
    else:
        if args.requests < 1 or args.concurrency < 1:
            parser.error("Requests and concurrency must be positive")
        asyncio.run(load_api(args.url, args.requests, args.concurrency))


if __name__ == "__main__":
    main()
