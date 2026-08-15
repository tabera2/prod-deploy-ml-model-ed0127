"""Load the ACTIVE model version once and expose clean predict seams."""
import numpy as np

from app.registry import active_version, load_model

_VERSION = active_version()       # which version we resolved at startup
_MODEL = load_model(_VERSION)     # the artifact for that version, loaded once

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

FEATURE_ORDER = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def model_version() -> str:
    return _VERSION


def _decode(label_idx: int, proba) -> dict:
    return {
        "label": CLASS_NAMES[label_idx],
        "confidence": float(proba[label_idx]),
    }


def predict_one(features: list[float]) -> dict:
    row = np.array(features, dtype=float).reshape(1, -1)
    label_idx = int(_MODEL.predict(row)[0])
    proba = _MODEL.predict_proba(row)[0]
    return _decode(label_idx, proba)


def predict_batch(rows: list[list[float]]) -> list[dict]:
    matrix = np.array(rows, dtype=float)
    label_idxs = _MODEL.predict(matrix)
    probas = _MODEL.predict_proba(matrix)
    return [_decode(int(i), p) for i, p in zip(label_idxs, probas)]
