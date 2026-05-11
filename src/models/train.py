from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.evaluate import regression_metrics
from src.models.training import fit_model


@dataclass(frozen=True)
class SupervisedDataset:
    X: pd.DataFrame
    y: pd.Series
    dates: pd.Series | None
    feature_columns: list[str]
    prepared_frame: pd.DataFrame


@dataclass(frozen=True)
class TemporalSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    train_dates: pd.Series | None
    X_valid: pd.DataFrame
    y_valid: pd.Series
    valid_dates: pd.Series | None
    X_test: pd.DataFrame
    y_test: pd.Series
    test_dates: pd.Series | None


def prepare_supervised_dataset(
    df: pd.DataFrame,
    *,
    target_col: str,
    date_col: str,
) -> SupervisedDataset:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    prepared = df.copy()
    if date_col in prepared.columns:
        prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce")
        invalid_dates = int(prepared[date_col].isna().sum())
        if invalid_dates:
            raise ValueError(f"Date column '{date_col}' contains {invalid_dates} invalid values.")
        prepared = prepared.sort_values(date_col).reset_index(drop=True)

    prepared = prepared.dropna(subset=[target_col]).reset_index(drop=True)
    feature_columns = [
        column
        for column in prepared.select_dtypes(include=[np.number]).columns
        if column != target_col
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found for model training.")

    dates = prepared[date_col].copy() if date_col in prepared.columns else None
    return SupervisedDataset(
        X=prepared[feature_columns].copy(),
        y=prepared[target_col].astype(float).copy(),
        dates=dates,
        feature_columns=[str(column) for column in feature_columns],
        prepared_frame=prepared,
    )


def split_train_valid_test(
    dataset: SupervisedDataset,
    *,
    validation_size: float,
    test_size: float | None = None,
) -> TemporalSplit:
    if test_size is None:
        test_size = validation_size
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1.")
    if validation_size + test_size >= 0.8:
        raise ValueError("validation_size + test_size is too large for a stable train split.")
    if len(dataset.X) < 5:
        raise ValueError("Need at least 5 rows for train/validation/test split.")

    n_rows = len(dataset.X)
    test_rows = max(1, int(n_rows * test_size))
    valid_rows = max(1, int(n_rows * validation_size))
    train_rows = n_rows - valid_rows - test_rows
    if train_rows < 1:
        raise ValueError("Not enough rows after train/validation/test split.")

    train_slice = slice(0, train_rows)
    valid_slice = slice(train_rows, train_rows + valid_rows)
    test_slice = slice(train_rows + valid_rows, n_rows)
    dates = dataset.dates

    return TemporalSplit(
        X_train=dataset.X.iloc[train_slice].copy(),
        y_train=dataset.y.iloc[train_slice].copy(),
        train_dates=dates.iloc[train_slice].copy() if dates is not None else None,
        X_valid=dataset.X.iloc[valid_slice].copy(),
        y_valid=dataset.y.iloc[valid_slice].copy(),
        valid_dates=dates.iloc[valid_slice].copy() if dates is not None else None,
        X_test=dataset.X.iloc[test_slice].copy(),
        y_test=dataset.y.iloc[test_slice].copy(),
        test_dates=dates.iloc[test_slice].copy() if dates is not None else None,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> tuple[Any, str]:
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    return fit_model(
        X_train=X_train,
        y_train=y_train,
        model_kind=str(training_config.get("model_kind", "flaml")),
        fallback_model=str(training_config.get("fallback_model", "gradient_boosting")),
        time_budget=int(training_config.get("time_budget", 600)),
        metric=str(training_config.get("metric", model_config.get("metric", "mae"))),
        prediction_horizon=int(config.get("prediction_horizon", 1)),
        random_state=int(training_config.get("random_state", 42)),
    )


def evaluate_split(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    return regression_metrics(y, model.predict(X))


def feature_importance_frame(model: Any, feature_columns: list[str]) -> pd.DataFrame | None:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": np.asarray(importances, dtype=float),
        }
    ).sort_values("importance", ascending=False)
