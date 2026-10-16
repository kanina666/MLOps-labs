import hashlib
import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import mlflow
from mlflow import MlflowClient
from training import train as pipeline
from training.data import prepare_data
from training.log_train import ModelSpec


def write_snapshot(path: Path, days: int = 45) -> None:
    path.mkdir(parents=True, exist_ok=True)
    # Reverse columns and calendar rows to exercise alignment by numeric day ID.
    sales = {"store_id": ["CA_1", "CA_2"]}
    sales.update({f"d_{day}": [day, 1000] for day in range(days, 0, -1)})
    pd.DataFrame(sales).to_csv(path / "sales_train_validation.csv", index=False)
    calendar = pd.DataFrame(
        {
            "d": [f"d_{day}" for day in range(1, days + 1)],
            "date": pd.date_range("2020-01-01", periods=days),
            "snap_CA": np.arange(days) % 2,
        }
    )
    calendar.iloc[::-1].to_csv(path / "calendar.csv", index=False)


def test_added_sales_days_are_used_and_features_follow_numeric_order(tmp_path: Path) -> None:
    write_snapshot(tmp_path, days=1914)
    data = prepare_data(tmp_path, validation_days=3)

    assert data.y_validation.tolist() == [1912, 1913, 1914]
    assert data.x_validation.iloc[-1]["lag_1"] == 1913
    assert data.x_validation.iloc[-1]["lag_28"] == 1886
    assert data.x_validation.iloc[-1]["rolling_mean_7"] == 1910
    assert data.snapshot["train"]["last_day"] == "d_1911"
    assert data.snapshot["validation"]["last_day"] == "d_1914"
    assert len(data.x_train) == 1914 - 28 - 3


def test_changed_files_use_actual_digest_and_ignore_stale_manifest(tmp_path: Path) -> None:
    write_snapshot(tmp_path)
    sales_path = tmp_path / "sales_train_validation.csv"
    original_hash = hashlib.sha256(sales_path.read_bytes()).hexdigest()
    manifest = {
        "files": [
            {
                "filename": sales_path.name,
                "sha256": original_hash,
                "source_url": "https://example.org/original.csv",
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    original = prepare_data(tmp_path, validation_days=3)
    assert original.inputs[0].dataset.source_type == "http"

    sales = pd.read_csv(sales_path)
    sales.loc[sales["store_id"] == "CA_1", "d_45"] = 99
    sales.to_csv(sales_path, index=False)
    changed = prepare_data(tmp_path, validation_days=3)
    digest = hashlib.sha256(sales_path.read_bytes()).hexdigest()

    assert changed.inputs[0].dataset.digest == digest[:32]
    assert changed.inputs[0].dataset.digest != original.inputs[0].dataset.digest
    assert changed.inputs[0].dataset.source_type == "local"
    assert json.loads(changed.inputs[0].dataset.source) == {"uri": str(sales_path.resolve())}
    assert changed.snapshot["files"][0]["sha256"] == digest


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("no_store", "no rows for store CA_1"),
        ("gap", "consecutive sequence"),
        ("negative", "finite nonnegative"),
        ("missing_value", "finite nonnegative"),
        ("missing_calendar", "missing sales days"),
        ("short", "Not enough sales days"),
    ],
)
def test_invalid_inputs_fail_before_training(tmp_path: Path, problem: str, message: str) -> None:
    write_snapshot(tmp_path, days=30 if problem == "short" else 45)
    path = tmp_path / "sales_train_validation.csv"
    sales = pd.read_csv(path)
    if problem == "no_store":
        sales["store_id"] = "CA_2"
    elif problem == "gap":
        sales = sales.drop(columns="d_20")
    elif problem == "negative":
        sales.loc[0, "d_20"] = -1
    elif problem == "missing_value":
        sales.loc[0, "d_20"] = np.nan
    elif problem == "missing_calendar":
        calendar_path = tmp_path / "calendar.csv"
        calendar = pd.read_csv(calendar_path)
        calendar.loc[calendar["d"] != "d_20"].to_csv(calendar_path, index=False)
    sales.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        prepare_data(tmp_path, validation_days=3)


def test_local_pipeline_logs_lineage_initializes_and_preserves_champion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    write_snapshot(data_dir)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(
        pipeline,
        "model_specs",
        lambda: [ModelSpec("tiny_ridge", Ridge(), {})],
    )
    try:
        first = pipeline.train(
            data_dir,
            tracking_uri,
            model_name="test_sales",
            validation_days=3,
        )
        client = MlflowClient()
        assert client.get_registered_model("test_sales").aliases == {"champion": "1"}
        assert first["champion_version"] == "1"
        second = pipeline.train(
            data_dir,
            tracking_uri,
            model_name="test_sales",
            validation_days=3,
        )
        assert first["versions"] == {"tiny_ridge": "1"}
        assert second["versions"] == {"tiny_ridge": "2"}
        assert second["champion_version"] == "1"
        assert str(client.get_model_version_by_alias("test_sales", "champion").version) == "1"
        parent = client.get_run(second["run_id"])
        candidate = client.get_run(second["best_run_id"])
        assert parent.info.status == "FINISHED"
        assert candidate.data.tags["mlflow.parentRunId"] == parent.info.run_id
        assert set(candidate.data.metrics) == {"WAPE", "MAE", "RMSE", "R2"}
        assert len(candidate.inputs.dataset_inputs) == 2
        artifacts = {artifact.path for artifact in client.list_artifacts(candidate.info.run_id)}
        assert {"model", "plots", "data_snapshot.json"}.issubset(artifacts)
        summary_path = client.download_artifacts(parent.info.run_id, "training_result.json")
        assert json.loads(Path(summary_path).read_text(encoding="utf-8")) == second
        loaded = mlflow.sklearn.load_model("models:/test_sales@champion")
        assert len(loaded.predict(prepare_data(data_dir, 3).x_validation)) == 3
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous_uri)


def test_failed_pipeline_marks_parent_failed(tmp_path: Path) -> None:
    previous_uri = mlflow.get_tracking_uri()
    try:
        with pytest.raises(FileNotFoundError, match="Missing M5 input"):
            pipeline.train(
                tmp_path / "missing",
                (tmp_path / "mlruns").as_uri(),
            )
        client = MlflowClient()
        experiment = client.get_experiment_by_name("m5-forecasting")
        assert experiment is not None
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        assert runs[0].info.status == "FAILED"
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous_uri)
