from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.model_loader import ModelBundle


def _prediction_to_float(prediction: Any) -> float:
    if isinstance(prediction, pd.DataFrame):
        values = prediction.to_numpy().reshape(-1)
    elif isinstance(prediction, pd.Series):
        values = prediction.to_numpy().reshape(-1)
    else:
        values = np.asarray(prediction).reshape(-1)
    if values.size == 0:
        raise ValueError("Model returned an empty prediction.")
    if pd.isna(values[0]):
        raise ValueError(f"Model returned an empty/NA prediction: {values[0]!r}")
    return float(values[0])


def _expected_feature_columns(bundle: ModelBundle) -> list[str] | None:
    model_columns = getattr(bundle.model, "feature_names_in_", None)
    if model_columns is not None:
        return [str(column) for column in model_columns]

    metadata_columns = bundle.metadata.get("feature_columns")
    if isinstance(metadata_columns, list) and metadata_columns:
        return [str(column) for column in metadata_columns]

    return None


def _align_features(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    expected_columns = _expected_feature_columns(bundle)
    if not expected_columns:
        return frame

    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required features: {missing_columns}")

    return frame[expected_columns]


def _mlflow_signature_types(bundle: ModelBundle) -> dict[str, str]:
    signature = getattr(getattr(bundle.model, "metadata", None), "signature", None)
    inputs = getattr(signature, "inputs", None)
    if inputs is None:
        return {}

    types: dict[str, str] = {}
    for column in inputs:
        name = getattr(column, "name", None)
        dtype = getattr(column, "type", None)
        if name and dtype is not None:
            types[str(name)] = str(getattr(dtype, "name", dtype)).lower()
    return types


def _coerce_signature_types(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    signature_types = _mlflow_signature_types(bundle)
    if not signature_types:
        return frame

    coerced = frame.copy()
    for column, dtype in signature_types.items():
        if column not in coerced.columns:
            continue
        if dtype == "integer":
            coerced[column] = pd.to_numeric(coerced[column], errors="raise").astype(np.int32)
        elif dtype == "long":
            coerced[column] = pd.to_numeric(coerced[column], errors="raise").astype(np.int64)
        elif dtype in {"double", "float"}:
            coerced[column] = pd.to_numeric(coerced[column], errors="raise").astype(float)
    return coerced


def _required_pipeline_input_columns(bundle: ModelBundle) -> list[str]:
    pipeline = bundle.feature_pipeline
    required_columns = [
        getattr(pipeline, "date_col", None),
        getattr(pipeline, "target_col", None),
    ]
    macro_columns = getattr(pipeline, "macro_columns_", []) or []
    required_columns.extend(macro_columns)
    return [str(column) for column in required_columns if column]


def _prepare_source_frame(records: list[dict[str, Any]], bundle: ModelBundle) -> pd.DataFrame:
    source_frame = pd.DataFrame(records).replace({pd.NA: np.nan})
    required_columns = _required_pipeline_input_columns(bundle)
    missing_columns = [column for column in required_columns if column not in source_frame.columns]
    if missing_columns:
        raise ValueError(
            "Input records are missing columns required by the fitted feature pipeline: "
            f"{missing_columns}"
        )

    date_col = getattr(bundle.feature_pipeline, "date_col", None)
    numeric_columns = [column for column in required_columns if column != date_col]
    for column in numeric_columns:
        source_frame[column] = pd.to_numeric(source_frame[column], errors="coerce")

    missing_value_columns = source_frame.columns[source_frame.isna().any()].tolist()
    if missing_value_columns:
        raise ValueError(
            "Input records contain missing or non-numeric values before feature pipeline transform. "
            f"Problem columns: {missing_value_columns}"
        )

    return source_frame


def _prepare_model_frame(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    prepared = _align_features(frame, bundle)
    prepared = prepared.replace({pd.NA: np.nan})

    missing_columns = prepared.columns[prepared.isna().any()].tolist()
    if missing_columns:
        raise ValueError(
            "Feature frame contains missing values. "
            f"Missing columns: {missing_columns}. "
            "For /predict, provide enough history to calculate all lag and rolling features "
            "(at least 31 records for the current lag/rolling configuration)."
        )

    numeric_frame = prepared.apply(pd.to_numeric, errors="ignore")
    numeric_columns = numeric_frame.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        infinite_columns = numeric_columns[
            np.isinf(numeric_frame[numeric_columns].to_numpy(dtype=float)).any(axis=0)
        ].tolist()
        if infinite_columns:
            raise ValueError(f"Feature frame contains infinite values: {infinite_columns}")

    return _coerce_signature_types(numeric_frame, bundle)


def _response(prediction: float, bundle: ModelBundle) -> dict[str, Any]:
    return {
        "prediction": prediction,
        "model_version": bundle.model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _feature_row(frame: pd.DataFrame) -> dict[str, Any]:
    return frame.iloc[0].to_dict()


def predict_from_features(features: dict[str, Any], bundle: ModelBundle) -> dict[str, Any]:
    if not features:
        raise ValueError("features must not be empty.")

    frame = pd.DataFrame([features])
    frame = _prepare_model_frame(frame, bundle)
    prediction = bundle.model.predict(frame)
    response = _response(_prediction_to_float(prediction), bundle)
    response["_features"] = _feature_row(frame)
    return response


def predict_from_records(records: list[dict[str, Any]], bundle: ModelBundle) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty.")

    source_frame = _prepare_source_frame(records, bundle)
    feature_frame = bundle.feature_pipeline.transform(source_frame)
    if feature_frame.empty:
        raise ValueError("Feature pipeline returned an empty dataframe.")

    latest_features = feature_frame.tail(1).reset_index(drop=True)
    latest_features = _prepare_model_frame(latest_features, bundle)
    prediction = bundle.model.predict(latest_features)
    response = _response(_prediction_to_float(prediction), bundle)
    response["_features"] = _feature_row(latest_features)
    return response
