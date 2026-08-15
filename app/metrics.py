"""Prometheus metrics for the model service: latency + prediction counts."""
import time

from prometheus_client import Counter, Histogram

# Latency of a prediction call, labeled by endpoint and model version.
predict_latency = Histogram(
    "predict_seconds", "Prediction latency", ["endpoint", "version"]
)
# How many predictions per class — a cheap output-drift signal.
predictions_total = Counter(
    "predictions_total", "Predictions made", ["label", "version"]
)


class timed:
    """Context manager: record one prediction's latency."""

    def __init__(self, endpoint: str, version: str):
        self.endpoint, self.version = endpoint, version

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.start
        predict_latency.labels(self.endpoint, self.version).observe(elapsed)
