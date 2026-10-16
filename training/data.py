"""Load an M5 snapshot and record the files actually used for training."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlflow.entities import Dataset, DatasetInput, InputTag

from mlops_labs.features import FEATURE_NAMES, HISTORY_DAYS, make_features

ROOT = Path(__file__).resolve().parents[1]
STORE_ID = "CA_1"
INPUT_FILES = ("sales_train_validation.csv", "calendar.csv")


@dataclass
class TrainingData:
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_validation: pd.DataFrame
    y_validation: np.ndarray
    inputs: list[DatasetInput]
    snapshot: dict[str, Any]


def default_data_dir() -> Path:
    path = ROOT / "data" / "raw"
    return path if path.exists() else ROOT.parent / "data" / "raw"


def _file_metadata(data_dir: Path) -> list[dict[str, Any]]:
    # An old manifest may describe the original download, not this snapshot.
    manifests = []
    for path in dict.fromkeys((data_dir / "manifest.json", ROOT / "data" / "manifest.json")):
        if path.is_file():
            manifests.extend(json.loads(path.read_text(encoding="utf-8")).get("files", []))

    files = []
    for filename in INPUT_FILES:
        path = (data_dir / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing M5 input file: {path}")
        with path.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        match = next(
            (
                item
                for item in manifests
                if item.get("filename") == filename
                and item.get("sha256") == sha256
                and str(item.get("source_url", "")).startswith(("https://", "http://"))
            ),
            None,
        )
        files.append(
            {
                "filename": filename,
                "path": str(path),
                "sha256": sha256,
                "digest": sha256[:32],
                "source_type": "http" if match else "local",
                "source": {"url": match["source_url"]} if match else {"uri": str(path)},
            }
        )
    return files


def load_raw_series(data_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    sales_path = data_dir / INPUT_FILES[0]
    columns = pd.read_csv(sales_path, nrows=0).columns
    day_cols = sorted(
        (column for column in columns if re.fullmatch(r"d_[1-9]\d*", column)),
        key=lambda column: int(column[2:]),
    )
    if not day_cols or day_cols != [f"d_{day}" for day in range(1, len(day_cols) + 1)]:
        raise ValueError("Sales day columns must form a consecutive sequence d_1, d_2, ...")
    if "store_id" not in columns:
        raise ValueError("Sales file must contain store_id")

    total = np.zeros(len(day_cols), dtype=np.float64)
    store_rows = 0
    for chunk in pd.read_csv(sales_path, chunksize=5000, usecols=["store_id", *day_cols]):
        selected = chunk.loc[chunk["store_id"] == STORE_ID, day_cols]
        values = selected.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                "CA_1 sales must be finite nonnegative numbers, without missing values"
            )
        store_rows += len(selected)
        total += values.sum(axis=0)
    if not store_rows:
        raise ValueError("Sales file contains no rows for store CA_1")
    if not np.isfinite(total).all():
        raise ValueError("Aggregated sales must be finite")

    calendar = pd.read_csv(data_dir / INPUT_FILES[1])
    if not {"d", "date", "snap_CA"}.issubset(calendar.columns):
        raise ValueError("Calendar must contain d, date and snap_CA")
    if calendar["d"].duplicated().any():
        raise ValueError("Calendar contains duplicate days")
    if not set(day_cols).issubset(calendar["d"]):
        raise ValueError("Calendar is missing sales days")
    calendar = calendar.set_index("d").loc[day_cols].reset_index()
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    if (
        calendar["date"].isna().any()
        or not calendar["date"].diff().iloc[1:].eq(pd.Timedelta(days=1)).all()
    ):
        raise ValueError("Calendar dates must form a consecutive daily sequence")
    if not calendar["snap_CA"].isin([0, 1]).all():
        raise ValueError("Calendar snap_CA must contain only 0 or 1")
    return total, calendar


def build_dataset(sales: np.ndarray, calendar: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    weekdays = calendar["date"].dt.dayofweek.to_numpy()
    rows = [
        make_features(
            sales[day - HISTORY_DAYS : day].tolist(),
            int(weekdays[day]),
            int(calendar.iloc[day]["snap_CA"]),
        )
        for day in range(HISTORY_DAYS, len(sales))
    ]
    return pd.DataFrame(rows, columns=FEATURE_NAMES), sales[HISTORY_DAYS:]


def prepare_data(data_dir: Path, validation_days: int) -> TrainingData:
    if validation_days < 2:
        raise ValueError("validation_days must be at least 2 to calculate R2")
    files = _file_metadata(data_dir)
    sales, calendar = load_raw_series(data_dir)
    if len(sales) < HISTORY_DAYS + validation_days + 2:
        raise ValueError("Not enough sales days for history, training and validation")
    features, targets = build_dataset(sales, calendar)
    if targets[-validation_days:].sum() == 0:
        raise ValueError("Validation sales sum must be positive to calculate WAPE")

    def period(start: int, stop: int) -> dict[str, Any]:
        return {
            "first_day": calendar.iloc[start]["d"],
            "last_day": calendar.iloc[stop - 1]["d"],
            "start_date": calendar.iloc[start]["date"].date().isoformat(),
            "end_date": calendar.iloc[stop - 1]["date"].date().isoformat(),
            "rows": stop - start,
        }

    inputs = [
        DatasetInput(
            Dataset(
                name=f"m5_{Path(item['filename']).stem}",
                digest=item["digest"],
                source_type=item["source_type"],
                source=json.dumps(item["source"]),
            ),
            tags=[InputTag("mlflow.data.context", "source")],
        )
        for item in files
    ]
    snapshot = {
        "store_id": STORE_ID,
        "files": files,
        "history_days": HISTORY_DAYS,
        "features": list(FEATURE_NAMES),
        "validation_mode": "rolling_one_step",
        "train": period(HISTORY_DAYS, len(sales) - validation_days),
        "validation": period(len(sales) - validation_days, len(sales)),
    }
    return TrainingData(
        x_train=features.iloc[:-validation_days].reset_index(drop=True),
        y_train=targets[:-validation_days],
        x_validation=features.iloc[-validation_days:].reset_index(drop=True),
        y_validation=targets[-validation_days:],
        inputs=inputs,
        snapshot=snapshot,
    )
