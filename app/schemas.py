from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeaturesRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Single-row feature mapping.")


class RecordsRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, description="Dataframe-like time series records.")


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str | None
    created_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    model_uri: str | None

