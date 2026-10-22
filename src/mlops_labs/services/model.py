import logging

import mlflow.pyfunc

from mlflow import MlflowClient
from mlops_labs.core.config import Settings, get_settings
from mlops_labs.domain.model import LoadedModel

logger = logging.getLogger(__name__)


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
