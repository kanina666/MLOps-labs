import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import mlflow.pyfunc
import pandas as pd

from mlflow import MlflowClient
from mlops_labs.core.config import Settings, get_settings
from mlops_labs.domain.model import LoadedModel, PredictionResult
from mlops_labs.domain.tasks import TaskPayload
from mlops_labs.features import make_features

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when the ML model is not loaded or unavailable for inference."""


class ModelPredictor:
    """Encapsulates feature engineering and CPU-bound inference in a dedicated thread pool."""

    def __init__(
        self,
        loaded_model: LoadedModel | None,
        max_workers: int = 1,
        inference_delay_ms: float = 0.0,
    ) -> None:
        self._loaded_model = loaded_model
        self._inference_delay_ms = inference_delay_ms
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="inference",
        )

    @property
    def loaded_model(self) -> LoadedModel | None:
        return self._loaded_model

    async def predict(self, payload: TaskPayload) -> PredictionResult:
        if self._loaded_model is None:
            raise ModelUnavailableError("Model is not loaded or unavailable")

        features_row = make_features(
            history=payload.recent_sales,
            day_of_week=payload.day_of_week,
            is_snap=payload.is_snap,
        )
        features_df = pd.DataFrame([features_row])

        if self._inference_delay_ms > 0:
            await asyncio.sleep(self._inference_delay_ms / 1000.0)

        loop = asyncio.get_running_loop()
        raw_pred = await loop.run_in_executor(
            self._executor,
            self._loaded_model.model.predict,
            features_df,
        )
        return PredictionResult(
            prediction=round(float(raw_pred[0]), 4),
            model_version=self._loaded_model.model_version,
            run_id=self._loaded_model.run_id,
        )

    def shutdown(self) -> None:
        """Wait for in-flight inference tasks to finish and release thread pool resources."""
        self._executor.shutdown(wait=True)


def load_model(
    model_name: str | None = None,
    alias: str = "champion",
    settings: Settings | None = None,
) -> LoadedModel | None:
    """Load model from MLflow using alias or URI, returning LoadedModel or None on error."""
    if settings is None:
        settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    if model_name is not None:
        if model_name.startswith(("models:/", "runs:/", "file://", "/")):
            target_uri = model_name
        elif alias:
            target_uri = f"models:/{model_name}@{alias}"
        else:
            target_uri = f"models:/{model_name}"
    else:
        target_uri = settings.model_uri

    try:
        resolved_uri = target_uri
        model_version: str | None = None
        run_id: str | None = None

        if resolved_uri.startswith("models:/") and "@" in resolved_uri:
            parsed_name, parsed_alias = resolved_uri.removeprefix("models:/").rsplit("@", 1)
            client = MlflowClient()
            version = client.get_model_version_by_alias(parsed_name, parsed_alias)
            model_version = version.version
            run_id = version.run_id
            resolved_uri = f"models:/{parsed_name}/{model_version}"

        loaded_pyfunc = mlflow.pyfunc.load_model(resolved_uri)
        loaded = LoadedModel(
            model=loaded_pyfunc,
            model_version=model_version,
            run_id=run_id,
            model_uri=resolved_uri,
        )
        logger.info(
            "model.loaded",
            extra={
                "model_uri": resolved_uri,
                "model_version": model_version,
                "run_id": run_id,
            },
        )
        return loaded
    except Exception as exc:
        logger.warning(
            "model.load_failed",
            extra={
                "error_type": type(exc).__name__,
                "error_code": getattr(exc, "error_code", None),
                "model_uri": target_uri,
            },
        )
        return None
