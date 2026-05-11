from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import pandas as pd

from app.model_loader import ModelBundle
from src.io.s3 import S3ObjectNotFoundError, is_s3_uri, read_csv, read_parquet, write_csv, write_parquet


LOGGER = logging.getLogger(__name__)
DEFAULT_PREDICTIONS_URI = "artifacts/predictions"
DEFAULT_MODEL_STAGE_OR_ALIAS = "Production"


def _prediction_logs_base_uri() -> str:
    return os.getenv("PREDICTIONS_URI") or DEFAULT_PREDICTIONS_URI


def _daily_log_uri(base_uri: str, created_at: datetime, suffix: str) -> str:
    filename = f"predictions_{created_at.date().isoformat()}.{suffix}"
    if is_s3_uri(base_uri):
        return f"{base_uri.rstrip('/')}/{filename}"
    return str(Path(base_uri) / filename)


def _parse_model_name(model_uri: str) -> str | None:
    if model_uri.startswith("models:/"):
        remainder = model_uri.removeprefix("models:/").lstrip("/")
        if "@" in remainder:
            return remainder.rsplit("@", 1)[0] or None
        if "/" in remainder:
            return remainder.rsplit("/", 1)[0] or None
        return remainder or None

    if model_uri.startswith("runs:/"):
        return None

    parsed = urlparse(model_uri)
    if parsed.scheme == "s3":
        return Path(parsed.path).stem or None
    return Path(model_uri).stem or None


def _model_name(bundle: ModelBundle) -> str | None:
    for key in ("model_name", "registered_model_name", "registered_model"):
        value = bundle.metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return (
        os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
        or os.getenv("MLFLOW_MODEL_NAME")
        or _parse_model_name(bundle.model_uri)
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def build_prediction_log_record(
    *,
    bundle: ModelBundle,
    features: dict[str, Any],
    prediction: Any,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    timestamp = created_at or datetime.now(timezone.utc)
    return {
        "prediction_id": str(uuid4()),
        "created_at": timestamp.isoformat(),
        "model_name": _model_name(bundle),
        "model_version": bundle.model_version,
        "model_stage_or_alias": os.getenv("MODEL_STAGE_OR_ALIAS", DEFAULT_MODEL_STAGE_OR_ALIAS),
        "features": _json_safe(features),
        "prediction": _json_safe(prediction),
        "request_source": "api",
        "actual": None,
    }


def _append_dataframe(existing: pd.DataFrame | None, record: dict[str, Any]) -> pd.DataFrame:
    row = pd.DataFrame([record])
    if existing is None or existing.empty:
        return row
    return pd.concat([existing, row], ignore_index=True)


def _read_existing_parquet(uri: str) -> pd.DataFrame | None:
    try:
        return read_parquet(uri)
    except (FileNotFoundError, S3ObjectNotFoundError):
        return None


def _read_existing_csv(uri: str) -> pd.DataFrame | None:
    try:
        return read_csv(uri)
    except (FileNotFoundError, S3ObjectNotFoundError):
        return None


def write_prediction_log(record: dict[str, Any]) -> str:
    created_at = datetime.fromisoformat(str(record["created_at"]))
    base_uri = _prediction_logs_base_uri()
    parquet_uri = _daily_log_uri(base_uri, created_at, "parquet")

    try:
        df = _append_dataframe(_read_existing_parquet(parquet_uri), record)
        write_parquet(df, parquet_uri)
        return parquet_uri
    except Exception as parquet_error:
        csv_uri = _daily_log_uri(base_uri, created_at, "csv")
        LOGGER.warning(
            "Could not write prediction parquet log to %s, falling back to CSV %s: %s",
            parquet_uri,
            csv_uri,
            parquet_error,
        )
        df = _append_dataframe(_read_existing_csv(csv_uri), record)
        write_csv(df, csv_uri)
        return csv_uri


def log_prediction(
    *,
    bundle: ModelBundle,
    features: dict[str, Any],
    prediction: Any,
) -> str:
    record = build_prediction_log_record(bundle=bundle, features=features, prediction=prediction)
    return write_prediction_log(record)
