import csv
import json
import os
from pathlib import Path

from mlflow.entities import Dataset, DatasetInput, InputTag
from mlflow.tracking import MlflowClient
from nbconvert import HTMLExporter

import mlflow

ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "artifacts" / "eda"
NOTEBOOK = ROOT / "notebooks" / "M5_EDA_MLOps.ipynb"
EXPERIMENT = "m5-eda"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def log_eda() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    manifest = json.loads((EDA_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    datasets = [
        DatasetInput(
            Dataset(
                name=f"m5_{Path(f['filename']).stem}",
                digest=f["sha256"][:32],
                source_type="http",
                source=f["source_url"],
            ),
            tags=[InputTag("mlflow.data.context", "raw_data")],
        )
        for f in manifest["files"]
    ]

    with mlflow.start_run(run_name="EDA_full_panel") as run:
        client.log_inputs(run.info.run_id, datasets=datasets)
        mlflow.set_tags({"stage": "EDA", "dataset": "M5 Forecasting", "scope": "30490 series"})

        if (EDA_DIR / "analysis_config.json").exists():
            config = json.loads((EDA_DIR / "analysis_config.json").read_text(encoding="utf-8"))
            mlflow.log_params(config)

        if (EDA_DIR / "summary_metrics.json").exists():
            metrics = json.loads((EDA_DIR / "summary_metrics.json").read_text(encoding="utf-8"))
            mlflow.log_metrics(metrics)

        if (EDA_DIR / "figures").exists():
            mlflow.log_artifacts(str(EDA_DIR / "figures"), artifact_path="figures")
        if (EDA_DIR / "tables").exists():
            mlflow.log_artifacts(str(EDA_DIR / "tables"), artifact_path="tables")

        if NOTEBOOK.exists():
            mlflow.log_artifact(str(NOTEBOOK), artifact_path="report")
            exporter = HTMLExporter(template_name="classic", embed_images=True)
            html, _ = exporter.from_filename(str(NOTEBOOK))
            mlflow.log_text(html, "report/M5_EDA_MLOps.html")

        print(f"[OK] EDA_full_panel logged: {run.info.run_id}")

    summary_file = EDA_DIR / "tables" / "backtest_summary.csv"
    metrics_file = EDA_DIR / "tables" / "backtest_metrics.csv"

    if summary_file.exists() and metrics_file.exists():
        with summary_file.open(encoding="utf-8-sig") as sf:
            summaries = {row["model"]: row for row in csv.DictReader(sf)}

        with metrics_file.open(encoding="utf-8-sig") as mf:
            fold_metrics = list(csv.DictReader(mf))

        for model_name, summary in summaries.items():
            with mlflow.start_run(run_name=f"baseline_{model_name}"):
                mlflow.set_tags({"stage": "baseline", "model_family": "heuristic_rule"})
                mlflow.log_params({"rule": model_name, "horizon_days": 28, "n_folds": 3})

                mlflow.log_metrics(
                    {
                        "WAPE_mean": float(summary["WAPE_mean"]),
                        "MAE_mean": float(summary["MAE"]),
                        "RMSSE_mean": float(summary["mean_RMSSE"]),
                        "weighted_RMSSE": float(summary["revenue_weighted_RMSSE_bottom"]),
                        "bias": float(summary["bias"]),
                        "aggregate_WAPE": float(summary["aggregate_WAPE"]),
                    }
                )

                for f_row in fold_metrics:
                    if f_row["model"] == model_name:
                        step = int(f_row["fold"])
                        mlflow.log_metrics(
                            {
                                "WAPE": float(f_row["WAPE"]),
                                "MAE": float(f_row["MAE"]),
                                "RMSSE": float(f_row["mean_RMSSE"]),
                            },
                            step=step,
                        )

                print(f"[OK] baseline_{model_name} logged")

    print(f"\nAll runs successfully logged. MLflow UI: {TRACKING_URI}")


if __name__ == "__main__":
    log_eda()
