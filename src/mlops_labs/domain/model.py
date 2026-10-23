from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: Any
    model_version: str | None
    run_id: str | None
    model_uri: str


@dataclass(frozen=True, slots=True)
class PredictionResult:
    prediction: float
    model_version: str | None
    run_id: str | None
