"""Fit each candidate through the same MLflow logging path."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
from mlflow import MlflowClient
from training.data import TrainingData


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    tags: dict[str, str]

    @property
    def params(self) -> dict[str, Any]:
        estimator = self.estimator
        preprocessing = {}
        if isinstance(estimator, Pipeline):
            preprocessing["preprocessing"] = ", ".join(
                type(step).__name__ for _, step in estimator.steps[:-1]
            )
            estimator = estimator.steps[-1][1]
        return {
            **estimator.get_params(deep=False),
            "model_type": type(estimator).__name__,
            **preprocessing,
        }


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "baseline_ridge",
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            {"model_family": "linear", "stage": "baseline"},
        ),
        ModelSpec(
            "hgb_default",
            HistGradientBoostingRegressor(random_state=42),
            {"model_family": "gradient_boosting", "stage": "model_v1"},
        ),
        ModelSpec(
            "hgb_tuned",
            HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=150,
                min_samples_leaf=15,
                l2_regularization=1.0,
                random_state=42,
            ),
            {"model_family": "gradient_boosting", "stage": "model_v2"},
        ),
        ModelSpec(
            "random_forest",
            RandomForestRegressor(n_estimators=100, random_state=42),
            {"model_family": "random_forest", "stage": "candidate"},
        ),
    ]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "WAPE": float(np.abs(y_true - y_pred).sum() / y_true.sum()),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_prediction_plot(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path
) -> None:
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(y_true, label="Actual Sales", color="black", linewidth=1.5)
    ax.plot(y_pred, label="Predicted", color="tab:blue", linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Day in validation window (one-step predictions)")
    ax.set_ylabel("Total Units Sold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)


def train_and_log(spec: ModelSpec, data: TrainingData) -> dict[str, Any]:
    print(f"Training {spec.name}...", flush=True)
    with mlflow.start_run(run_name=spec.name, nested=True) as run:
        MlflowClient().log_inputs(run.info.run_id, datasets=data.inputs)
        mlflow.set_tags({"store_id": "CA_1", "validation_mode": "rolling_one_step", **spec.tags})
        mlflow.log_params({"features_count": len(data.x_train.columns), **spec.params})
        mlflow.log_dict(data.snapshot, "data_snapshot.json")
        spec.estimator.fit(data.x_train, data.y_train)
        predictions = spec.estimator.predict(data.x_validation)
        metrics = compute_metrics(data.y_validation, predictions)
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError(f"{spec.name} produced nonfinite validation metrics")
        mlflow.log_metrics(metrics)
        example = data.x_train.head(2)
        mlflow.sklearn.log_model(
            spec.estimator,
            artifact_path="model",
            signature=infer_signature(example, spec.estimator.predict(example)),
            input_example=example,
            pip_requirements=[
                f"scikit-learn=={sklearn.__version__}",
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            plot = Path(temp_dir) / "forecast_vs_actual.png"
            save_prediction_plot(data.y_validation, predictions, spec.name, plot)
            mlflow.log_artifact(str(plot), artifact_path="plots")
        return {"model": spec.name, "run_id": run.info.run_id, "metrics": metrics}
