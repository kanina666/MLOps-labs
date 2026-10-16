from collections.abc import Sequence

HISTORY_DAYS = 28
FEATURE_NAMES = (
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "day_of_week",
    "is_snap",
)


def make_features(
    history: Sequence[float], day_of_week: int, is_snap: int
) -> dict[str, float | int]:
    """Build next-day features from observed sales before the target day."""
    if len(history) < HISTORY_DAYS:
        raise ValueError(f"At least {HISTORY_DAYS} days of sales history are required")
    return {
        "lag_1": float(history[-1]),
        "lag_7": float(history[-7]),
        "lag_14": float(history[-14]),
        "lag_28": float(history[-28]),
        "rolling_mean_7": float(sum(history[-7:]) / 7),
        "day_of_week": day_of_week,
        "is_snap": is_snap,
    }
