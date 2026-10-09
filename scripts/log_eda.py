import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "M5_EDA_HW_point_2"


def validate_tracking_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "Use the repository's HTTP(S) MLflow server, not a local file/SQLite store."
        )
    return uri.rstrip("/")


def validate_snapshot(artifact_dir: Path) -> dict:
    """Reject partial reruns and changed artifacts before creating a remote run."""
    manifest_path = artifact_dir / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "No completed EDA snapshot. Run all cells in notebooks/M5_EDA_MLOps.ipynb and save it."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    required = {
        "summary_metrics.json",
        "analysis_config.json",
        "environment.json",
        "data_manifest.json",
        "tables/backtest_metrics.csv",
    }
    if not required.issubset(files) or not any(name.startswith("figures/") for name in files):
        raise ValueError("Incomplete EDA manifest. Run the entire notebook again.")
    for name, expected in files.items():
        path = (artifact_dir / name).resolve()
        if not path.is_relative_to(artifact_dir.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid EDA artifact: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(
                f"EDA artifact changed after completion: {name}. Rerun the whole notebook."
            )
    return manifest


def log_eda(
    root: Path = ROOT, tracking_uri: str | None = None, reports: tuple[Path, ...] = ()
) -> dict:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    artifact_dir = root / "artifacts" / "eda"
    snapshot = validate_snapshot(artifact_dir)
    uri = validate_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    for report in reports:
        if not report.is_file():
            raise FileNotFoundError(report)
    try:
        client_version = importlib.metadata.version("mlflow-skinny")
        from mlflow.entities import Dataset, DatasetInput, InputTag
        from mlflow.tracking import MlflowClient

        import mlflow
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError("Install EDA dependencies: uv sync --locked --group eda") from exc

    import requests

    try:
        response = requests.get(f"{uri}/health", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "MLflow is unavailable. From repository run: docker compose up -d --build mlflow"
        ) from exc

    config = json.loads((artifact_dir / "analysis_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    with (artifact_dir / "tables" / "backtest_metrics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads((artifact_dir / "data_manifest.json").read_text(encoding="utf-8"))
    dataset_inputs = []
    for record in manifest["files"]:
        dataset = Dataset(
            name=f"m5_{Path(record['filename']).stem}",
            digest=record["sha256"][:32],
            source_type="http",
            source=json.dumps({"url": record["source_url"]}),
        )
        dataset_inputs.append(
            DatasetInput(dataset, tags=[InputTag("mlflow.data.context", "source")])
        )
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = mlflow.set_experiment(EXPERIMENT)
    if not experiment.artifact_location.startswith("mlflow-artifacts:"):
        raise ValueError(
            "This experiment does not use the repository's artifact proxy. Check server configuration."
        )
    with mlflow.start_run(run_name="EDA_full_panel") as parent:
        parent_id = parent.info.run_id
        client.log_inputs(parent_id, datasets=dataset_inputs)
        mlflow.set_tags(
            {
                "task": "Homework point 2: EDA",
                "dataset": "M5 Forecasting Accuracy",
                "scope": "all 30490 bottom-level series",
                "storage": "repository_compose",
                "source_type": "pinned public mirror",
            }
        )
        mlflow.log_params({**config, "mlflow_client_version": client_version})
        mlflow.log_metrics(metrics)
        for name in snapshot["files"]:
            artifact_path = "eda/" + Path(name).parent.as_posix()
            mlflow.log_artifact(str(artifact_dir / name), artifact_path=artifact_path.rstrip("/."))
        mlflow.log_artifact(str(artifact_dir / "analysis_manifest.json"), artifact_path="eda")
        child_ids = []
        for model in dict.fromkeys(row["model"] for row in rows):
            model_rows = [row for row in rows if row["model"] == model]
            with mlflow.start_run(run_name=f"EDA_check_{model}", nested=True) as child:
                child_ids.append(child.info.run_id)
                client.log_inputs(child.info.run_id, datasets=dataset_inputs)
                mlflow.log_params({"rule": model, "horizon": config["horizon_days"]})
                for row in model_rows:
                    mlflow.log_metrics(
                        {
                            key: float(value)
                            for key, value in row.items()
                            if key not in {"fold", "model"}
                        },
                        step=int(row["fold"]),
                    )
                mlflow.log_metric(
                    "mean_WAPE", sum(float(row["WAPE"]) for row in model_rows) / len(model_rows)
                )
        for report in reports:
            mlflow.log_artifact(str(report), artifact_path="report")
            if report.suffix.lower() == ".ipynb":
                from nbconvert import HTMLExporter

                notebook = json.loads(report.read_text(encoding="utf-8"))
                if not any(cell.get("outputs") for cell in notebook["cells"]):
                    print(
                        f"{report.name}: no saved outputs. Run and save the notebook in Jupyter "
                        "to include results and charts in the HTML preview."
                    )
                exporter = HTMLExporter(template_name="classic", embed_images=True)
                html, _ = exporter.from_filename(str(report))
                html_name = report.with_suffix(".html").name
                mlflow.log_text(html, f"report/{html_name}")
        for folder in ["figures", "tables"]:
            expected_count = sum(Path(name).parts[0] == folder for name in snapshot["files"])
            if len(client.list_artifacts(parent_id, f"eda/{folder}")) != expected_count:
                raise RuntimeError(f"MLflow artifact listing does not match {folder}.")
        info = {
            "experiment_name": EXPERIMENT,
            "experiment_id": experiment.experiment_id,
            "parent_run_id": parent_id,
            "child_run_ids": child_ids,
            "tracking_uri": uri,
            "artifact_uri": mlflow.get_artifact_uri(),
            "run_url": f"{uri}/#/experiments/{experiment.experiment_id}/runs/{parent_id}",
        }
        mlflow.log_dict(info, "run_info.json")
    (root / "artifacts" / "mlflow_run.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print(f"MLflow run [FINISHED]: {info['run_url']}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        help="Report file; notebooks are also exported as HTML.",
    )
    args = parser.parse_args()
    reports = args.report or [ROOT / "notebooks" / "M5_EDA_MLOps.ipynb"]
    log_eda(tracking_uri=args.tracking_uri, reports=tuple(reports))


main()
