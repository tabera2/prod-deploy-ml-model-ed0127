"""FastAPI service: validated, versioned, instrumented prediction."""
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.predictor import predict_one, predict_batch, model_version
from app.schemas import PredictRequest, PredictResponse, BatchRequest
from app.metrics import timed, predictions_total
from app.drift import drift_flags

app = FastAPI(title="Iris Model Service")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict:
    with timed("predict", model_version()):
        result = predict_one(req.features)
    predictions_total.labels(result["label"], model_version()).inc()
    return result


@app.post("/predict/batch")
def predict_many(req: BatchRequest) -> dict:
    rows = [item.features for item in req.items]
    with timed("predict_batch", model_version()):
        results = predict_batch(rows)
    return {
        "predictions": results,
        "drift": drift_flags(rows),     # warn if this batch looks off
        "version": model_version(),
    }


@app.get("/healthz")
def healthz() -> dict:
    # Liveness: is the PROCESS up? Cheap, no dependencies.
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    # Readiness: is the MODEL loaded and servable?
    return {"status": "ready", "version": model_version()}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
