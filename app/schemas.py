"""Strict request/response schemas. Bad input is rejected at the edge."""
from pydantic import BaseModel, Field, field_validator

N_FEATURES = 4  # the model was trained on exactly four features
MAX_BATCH = 500  # cap a batch so one request can't exhaust memory


class PredictRequest(BaseModel):
    # Exactly four numbers, in the trained feature order.
    features: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)

    @field_validator("features")
    @classmethod
    def all_finite(cls, v: list[float]) -> list[float]:
        # NaN / inf would corrupt the prediction — reject them here.
        for x in v:
            if x != x or x in (float("inf"), float("-inf")):
                raise ValueError("features must be finite numbers")
        return v


class PredictResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class BatchRequest(BaseModel):
    # A bounded list of single-prediction requests, reusing its validation.
    items: list[PredictRequest] = Field(min_length=1, max_length=MAX_BATCH)
