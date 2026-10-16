"""Register candidates and initialize champion without replacing an existing alias"""

from typing import Any

import mlflow
from mlflow import MlflowClient


def register_models(results: list[dict[str, Any]], model_name: str) -> dict[str, str]:
    client = MlflowClient()
    versions = {}
    for result in results:
        registered = mlflow.register_model(f"runs:/{result['run_id']}/model", model_name)
        version = str(registered.version)
        client.update_model_version(
            name=model_name,
            version=version,
            description=f"{result['model']}: next-day total sales for M5 store CA_1.",
        )
        client.set_model_version_tag(model_name, version, "dataset", "m5_ca1_daily_sales")
        versions[result["model"]] = version
        print(
            f"Version {registered.version}: {result['model']} "
            f"(WAPE={result['metrics']['WAPE']:.4f})",
            flush=True,
        )
    return versions


def initialize_champion(model_name: str, candidate_version: str) -> str:
    """Use the selected candidate only when the registry has no champion yet."""
    client = MlflowClient()
    current_version = client.get_registered_model(model_name).aliases.get("champion")
    if current_version is not None:
        return str(current_version)
    client.set_registered_model_alias(model_name, "champion", candidate_version)
    return candidate_version
