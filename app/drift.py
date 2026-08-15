"""Cheap input-drift check: compare live feature stats to the training baseline.

If live inputs wander far from what the model was trained on, predictions
are extrapolations you should not trust. This flags that early.
"""
import numpy as np

# Per-feature training means (recorded when the model was trained).
BASELINE_MEAN = np.array([5.84, 3.05, 3.76, 1.20])
# Allowed absolute drift before we flag a feature.
DRIFT_TOLERANCE = np.array([1.5, 1.0, 2.0, 1.0])


def drift_flags(rows: list[list[float]]) -> list[str]:
    """Return the names of features whose live mean drifted past tolerance."""
    matrix = np.array(rows, dtype=float)
    live_mean = matrix.mean(axis=0)
    delta = np.abs(live_mean - BASELINE_MEAN)
    from app.predictor import FEATURE_ORDER
    return [
        FEATURE_ORDER[i]
        for i in range(len(FEATURE_ORDER))
        if delta[i] > DRIFT_TOLERANCE[i]
    ]
