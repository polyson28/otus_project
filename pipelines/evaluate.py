from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.io.s3 import read_parquet, write_json
from src.models.mlflow_utils import (
    get_model_alias,
    get_registered_model_name,
    setup_mlflow,
)


LOGGER = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MLflow model on a features dataset.")
    parser.add_argument("--features-uri", required=True, help="Input features parquet URI/path.")
    parser.add_argument("--model-uri", default=None, help="MLflow model URI. Defaults to configured champion alias.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--output-uri", default=None, help="Optional metrics JSON URI/path.")
    return parser.parse_args()


def default_model_uri(config: dict[str, Any]) -> str:
    return f"models:/{get_registered_model_name(config)}@{get_model_alias(config)}"


def prepare_supervised_data(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
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
        col for col in prepared.select_dtypes(include=[np.number]).columns
        if col != target_col
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found for evaluation.")

    return prepared[feature_columns].copy(), prepared[target_col].astype(float).copy()


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    non_zero_mask = y_true_array != 0
    if non_zero_mask.any():
        metrics["mape"] = float(
            np.mean(np.abs((y_true_array[non_zero_mask] - y_pred_array[non_zero_mask]) / y_true_array[non_zero_mask]))
        )
    if len(y_true) >= 2:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    return metrics


def load_mlflow_model(model_uri: str) -> Any:
    import mlflow.pyfunc
    import mlflow.sklearn

    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as pyfunc_error:
        LOGGER.warning("Could not load model via mlflow.pyfunc: %s", pyfunc_error)
        return mlflow.sklearn.load_model(model_uri)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import mlflow

    config = load_config(args.config)
    setup_mlflow(config)

    model_uri = args.model_uri or default_model_uri(config)
    model = load_mlflow_model(model_uri)
    features_df = read_parquet(args.features_uri)
    X, y = prepare_supervised_data(
        features_df,
        target_col=config["target_col"],
        date_col=config["date_col"],
    )

    predictions = model.predict(X)
    metrics = calculate_metrics(y, predictions)

    with mlflow.start_run(run_name="evaluation"):
        mlflow.set_tags(
            {
                "run_type": "evaluation",
                "model_uri": model_uri,
                "features_uri": args.features_uri,
                "project": config.get("project_name", "ts_project"),
            }
        )
        mlflow.log_metrics(metrics)

    if args.output_uri:
        write_json(
            {
                "model_uri": model_uri,
                "features_uri": args.features_uri,
                "metrics": metrics,
            },
            args.output_uri,
        )

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")


if __name__ == "__main__":
    main()
