import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlflow.entities import Dataset, DatasetInput, InputTag
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
EXPERIMENT = "m5-forecasting"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def load_raw_series() -> tuple[np.ndarray, pd.DataFrame]:
    sales_path = DATA_RAW / "sales_train_validation.csv"
    calendar_path = DATA_RAW / "calendar.csv"

    if not sales_path.exists() or not calendar_path.exists():
        raise FileNotFoundError(
            "Raw M5 dataset files not found. Run 'uv run python scripts/download_m5.py' first."
        )

    calendar = pd.read_csv(calendar_path).iloc[:1913]
    day_cols = [f"d_{i}" for i in range(1, 1914)]

    chunks = [
        chunk[chunk["store_id"] == "CA_1"][day_cols]
        for chunk in pd.read_csv(sales_path, chunksize=5000, usecols=["store_id"] + day_cols)
    ]
    ca1_sales = pd.concat(chunks).sum(axis=0).to_numpy(dtype=np.float32)
    return ca1_sales, calendar


def build_dataset(
    sales: np.ndarray, calendar: pd.DataFrame, history_len: int = 28
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    targets = []
    for t in range(history_len, len(sales)):
        hist = sales[t - history_len : t]
        rows.append(
            {
                "lag_1": float(hist[-1]),
                "lag_7": float(hist[-7]),
                "lag_14": float(hist[-14]),
                "lag_28": float(hist[-28]),
                "rolling_mean_7": float(np.mean(hist[-7:])),
                "day_of_week": int(calendar.iloc[t]["wday"]) % 7,
                "is_snap": int(calendar.iloc[t]["snap_CA"]),
            }
        )
        targets.append(sales[t])
    return pd.DataFrame(rows), np.array(targets, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"WAPE": wape, "MAE": mae, "RMSE": rmse, "R2": r2}


def save_prediction_plot(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label="Actual Sales", color="black", linewidth=1.5)
    ax.plot(y_pred, label="Predicted", color="tab:blue", linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Day in Test Horizon (28 days)")
    ax.set_ylabel("Total Units Sold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def train() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    print("[1/5] Loading data for store CA_1 and building features...")
    sales, calendar = load_raw_series()
    df_all, y_all = build_dataset(sales, calendar)

    test_horizon = 28
    df_train = df_all.iloc[:-test_horizon].reset_index(drop=True)
    y_train = y_all[:-test_horizon]
    df_test = df_all.iloc[-test_horizon:].reset_index(drop=True)
    y_test = y_all[-test_horizon:]

    manifest_path = ROOT / "data" / "manifest.json"
    if not manifest_path.exists():
        manifest_path = ROOT / "artifacts" / "eda" / "data_manifest.json"

    datasets = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = [
            DatasetInput(
                Dataset(
                    name=f"m5_{Path(f['filename']).stem}",
                    digest=f["sha256"][:32],
                    source_type="http",
                    source=f["source_url"],
                ),
                tags=[InputTag("mlflow.data.context", "training")],
            )
            for f in manifest["files"]
        ]

    runs_info = {}

    # Run 1: Baseline Ridge on basic lags
    base_features = ["lag_1", "lag_7", "day_of_week", "is_snap"]
    print("[2/5] Training Run 1: baseline_ridge...")
    model_ridge = Ridge(alpha=1.0)
    with mlflow.start_run(run_name="baseline_ridge") as run1:
        if datasets:
            client.log_inputs(run1.info.run_id, datasets=datasets)
        model_ridge.fit(df_train[base_features], y_train)
        pred_ridge = model_ridge.predict(df_test[base_features])
        metrics_ridge = compute_metrics(y_test, pred_ridge)

        sig_ridge = infer_signature(df_train[base_features].head(2), y_train[:2])
        mlflow.set_tags({"model_family": "linear", "stage": "baseline"})
        mlflow.log_params({"model_type": "Ridge", "alpha": 1.0, "features": str(base_features)})
        mlflow.log_metrics(metrics_ridge)
        mlflow.sklearn.log_model(model_ridge, artifact_path="model", signature=sig_ridge)
        runs_info["baseline_ridge"] = (run1.info.run_id, metrics_ridge["WAPE"])

    # Run 2: HistGradientBoosting default on all 7 features
    sig_full = infer_signature(df_train.head(2), y_train[:2])
    print("[3/5] Training Run 2: hgb_default...")
    model_hgb_def = HistGradientBoostingRegressor(random_state=42)
    with mlflow.start_run(run_name="hgb_default") as run2:
        if datasets:
            client.log_inputs(run2.info.run_id, datasets=datasets)
        model_hgb_def.fit(df_train, y_train)
        pred_hgb_def = model_hgb_def.predict(df_test)
        metrics_hgb_def = compute_metrics(y_test, pred_hgb_def)

        mlflow.set_tags({"model_family": "gradient_boosting", "stage": "model_v1"})
        mlflow.log_params(
            {
                "model_type": "HistGradientBoostingRegressor",
                "features_count": len(df_train.columns),
                "learning_rate": 0.1,
                "max_iter": 100,
            }
        )
        mlflow.log_metrics(metrics_hgb_def)
        mlflow.sklearn.log_model(model_hgb_def, artifact_path="model", signature=sig_full)

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_file = Path(tmp_dir) / "forecast_vs_actual.png"
            save_prediction_plot(y_test, pred_hgb_def, "HGB Default vs Actual", plot_file)
            mlflow.log_artifact(str(plot_file), artifact_path="plots")

        runs_info["hgb_default"] = (run2.info.run_id, metrics_hgb_def["WAPE"])

    # Run 3: HistGradientBoosting tuned on all 7 features
    print("[4/5] Training Run 3: hgb_tuned...")
    model_hgb_tuned = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=150,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=42,
    )
    with mlflow.start_run(run_name="hgb_tuned") as run3:
        if datasets:
            client.log_inputs(run3.info.run_id, datasets=datasets)
        model_hgb_tuned.fit(df_train, y_train)
        pred_hgb_tuned = model_hgb_tuned.predict(df_test)
        metrics_hgb_tuned = compute_metrics(y_test, pred_hgb_tuned)

        mlflow.set_tags({"model_family": "gradient_boosting", "stage": "model_v2"})
        mlflow.log_params(
            {
                "model_type": "HistGradientBoostingRegressor",
                "features_count": len(df_train.columns),
                "learning_rate": 0.03,
                "max_iter": 150,
                "min_samples_leaf": 15,
                "l2_regularization": 1.0,
            }
        )
        mlflow.log_metrics(metrics_hgb_tuned)
        mlflow.sklearn.log_model(model_hgb_tuned, artifact_path="model", signature=sig_full)

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_file = Path(tmp_dir) / "forecast_vs_actual.png"
            save_prediction_plot(y_test, pred_hgb_tuned, "HGB Tuned vs Actual", plot_file)
            mlflow.log_artifact(str(plot_file), artifact_path="plots")

        runs_info["hgb_tuned"] = (run3.info.run_id, metrics_hgb_tuned["WAPE"])

    # Register in Model Registry
    print("[5/5] Registering models in Model Registry...")
    model_name = "m5_sales_model"

    v1_run_id = runs_info["hgb_default"][0]
    reg_v1 = mlflow.register_model(f"runs:/{v1_run_id}/model", model_name)
    client.update_model_version(
        name=model_name,
        version=reg_v1.version,
        description="HistGradientBoosting (default) trained on M5 CA_1 daily sales with 7 lag features.",
    )
    client.set_model_version_tag(model_name, reg_v1.version, "dataset", "m5_ca1_daily_sales")

    v2_run_id = runs_info["hgb_tuned"][0]
    reg_v2 = mlflow.register_model(f"runs:/{v2_run_id}/model", model_name)
    client.update_model_version(
        name=model_name,
        version=reg_v2.version,
        description="HistGradientBoosting (tuned) trained on M5 CA_1 daily sales with 7 lag features and SNAP calendar events.",
    )
    client.set_model_version_tag(model_name, reg_v2.version, "dataset", "m5_ca1_daily_sales")

    client.set_registered_model_alias(model_name, "champion", reg_v2.version)

    print(f"\nModel '{model_name}' successfully registered:")
    print(
        f"  - Version {reg_v1.version} from hgb_default (WAPE: {runs_info['hgb_default'][1]:.4f})"
    )
    print(f"  - Version {reg_v2.version} from hgb_tuned (WAPE: {runs_info['hgb_tuned'][1]:.4f})")
    print(f"  - Alias @champion -> Version {reg_v2.version}")
    print(f"\nAll runs completed. Open MLflow UI: {TRACKING_URI}/#/experiments")


if __name__ == "__main__":
    train()
