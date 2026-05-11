from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.evaluation import regression_metrics


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainResult:
    model: Any
    feature_columns: list[str]
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    estimator_name: str
    train_rows: int
    validation_rows: int


def prepare_supervised_frame(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series | None]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    prepared = df.copy()
    date_values = None
    if date_col in prepared.columns:
        prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce")
        if prepared[date_col].isna().any():
            raise ValueError(f"Date column '{date_col}' contains invalid datetimes.")
        prepared = prepared.sort_values(date_col).reset_index(drop=True)
        date_values = prepared[date_col].copy()

    prepared = prepared.dropna(subset=[target_col]).reset_index(drop=True)
    feature_columns = [
        col for col in prepared.select_dtypes(include="number").columns
        if col != target_col
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found for training.")

    X = prepared[feature_columns].copy()
    y = prepared[target_col].astype(float).copy()
    return X, y, feature_columns, date_values


def time_train_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if len(X) < 2:
        raise ValueError("Need at least 2 rows to create a train/validation split.")

    split_idx = max(1, int(len(X) * (1 - validation_size)))
    if split_idx >= len(X):
        split_idx = len(X) - 1
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def build_fallback_model(model_name: str, random_state: int) -> Pipeline:
    if model_name == "random_forest":
        estimator = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingRegressor(random_state=random_state)
    else:
        raise ValueError(f"Unsupported fallback model: {model_name}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_kind: str,
    fallback_model: str,
    time_budget: int,
    metric: str,
    prediction_horizon: int,
    random_state: int,
) -> tuple[Any, str]:
    if model_kind == "flaml":
        import_errors: list[str] = []
        for module_name in ("TS_model.automl_tuning", "DS_research.automl_tuning"):
            try:
                module = __import__(module_name, fromlist=["ModelSelector"])
                selector = module.ModelSelector(time_budget=time_budget, metric=metric)
                model = selector.find_best_model(X_train, y_train, period=prediction_horizon)
                estimator_name = getattr(selector, "best_model_name", None) or "automl"
                return model, f"{module_name}:{estimator_name}"
            except Exception as error:
                import_errors.append(f"{module_name}: {error}")
                LOGGER.warning("AutoML training via %s failed: %s", module_name, error)
        try:
            raise RuntimeError("AutoML training failed. " + " | ".join(import_errors))
        except RuntimeError as error:
            if fallback_model == "none":
                raise
            LOGGER.warning("Falling back to %s: %s", fallback_model, error)

    model = build_fallback_model(model_name=fallback_model, random_state=random_state)
    model.fit(X_train, y_train)
    return model, fallback_model


def train_time_series_model(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> TrainResult:
    target_col = config["target_col"]
    date_col = config["date_col"]
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})

    X, y, feature_columns, _ = prepare_supervised_frame(df=df, target_col=target_col, date_col=date_col)
    X_train, X_valid, y_train, y_valid = time_train_validation_split(
        X=X,
        y=y,
        validation_size=float(
            training_config.get("validation_size", config.get("model", {}).get("validation_size", 0.2))
        ),
    )
    model, estimator_name = fit_model(
        X_train=X_train,
        y_train=y_train,
        model_kind=str(training_config.get("model_kind", "flaml")),
        fallback_model=str(training_config.get("fallback_model", "gradient_boosting")),
        time_budget=int(training_config.get("time_budget", 600)),
        metric=str(training_config.get("metric", "mae")),
        prediction_horizon=int(config.get("prediction_horizon", 1)),
        random_state=int(training_config.get("random_state", 42)),
    )

    key_rate = float(evaluation_config.get("key_rate", 0.21))
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    return TrainResult(
        model=model,
        feature_columns=feature_columns,
        train_metrics={f"train_{key}": value for key, value in regression_metrics(y_train, train_pred, key_rate).items()},
        validation_metrics={
            f"validation_{key}": value
            for key, value in regression_metrics(y_valid, valid_pred, key_rate).items()
        },
        estimator_name=estimator_name,
        train_rows=len(X_train),
        validation_rows=len(X_valid),
    )
