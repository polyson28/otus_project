from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.io.s3 import (
    S3Error,
    download_file,
    is_s3_uri,
    joblib_load_from_uri,
    read_json,
    read_parquet,
)
from src.models.mlflow_utils import (
    find_model_version_by_run_id,
    get_current_alias_version,
    get_model_alias,
    get_model_version_metric,
    get_registered_model_name,
    infer_and_log_model,
    log_config_params,
    log_dataset_info,
    set_model_alias,
    setup_mlflow,
    should_promote_model,
)
from src.models.training import fit_model


LOGGER = logging.getLogger("train")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a time series model and register it in MLflow.")
    parser.add_argument("--features-uri", required=True, help="Input features parquet URI/path.")
    parser.add_argument("--feature-pipeline-uri", required=True, help="Feature pipeline joblib URI/path.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--promote-if-better", type=parse_bool, default=True, help="Promote model when metric improves.")
    parser.add_argument("--run-name", default=None, help="Optional MLflow run name.")
    return parser.parse_args()


def companion_uri(uri: str, filename: str) -> str:
    if "/" not in uri:
        return filename
    return uri.rsplit("/", 1)[0] + "/" + filename


def read_optional_json(uri: str) -> dict[str, Any] | None:
    try:
        payload = read_json(uri)
    except (FileNotFoundError, S3Error, json.JSONDecodeError) as error:
        LOGGER.info("Optional JSON artifact is unavailable at %s: %s", uri, error)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("Optional JSON artifact at %s is not an object; skipping.", uri)
        return None
    return payload


def materialize_uri(uri: str, local_path: Path) -> Path:
    if is_s3_uri(uri):
        download_file(uri, local_path)
    else:
        shutil.copyfile(Path(uri), local_path)
    return local_path


def prepare_supervised_data(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
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
        raise ValueError("No numeric feature columns found for model training.")

    return (
        prepared[feature_columns].copy(),
        prepared[target_col].astype(float).copy(),
        prepared.copy(),
        feature_columns,
    )


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    prepared_df: pd.DataFrame,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if len(X) < 2:
        raise ValueError("Need at least 2 rows for train/validation split.")

    split_idx = max(1, int(len(X) * (1 - validation_size)))
    if split_idx >= len(X):
        split_idx = len(X) - 1
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
        prepared_df.iloc[split_idx:].copy(),
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> tuple[Any, str]:
    training_config = config.get("training", {})
    return fit_model(
        X_train=X_train,
        y_train=y_train,
        model_kind=str(training_config.get("model_kind", "flaml")),
        fallback_model=str(training_config.get("fallback_model", "random_forest")),
        time_budget=int(training_config.get("time_budget", 600)),
        metric=str(training_config.get("metric", config.get("model", {}).get("metric", "mae"))),
        prediction_horizon=int(config.get("prediction_horizon", 1)),
        random_state=int(training_config.get("random_state", 42)),
    )


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    non_zero_mask = np.asarray(y_true) != 0
    if non_zero_mask.any():
        y_true_array = np.asarray(y_true, dtype=float)
        y_pred_array = np.asarray(y_pred, dtype=float)
        metrics["mape"] = float(
            np.mean(np.abs((y_true_array[non_zero_mask] - y_pred_array[non_zero_mask]) / y_true_array[non_zero_mask]))
        )
    if len(y_true) >= 2:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    return metrics


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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import mlflow

    config = load_config(args.config)
    setup_mlflow(config)

    target_col = config["target_col"]
    date_col = config["date_col"]
    validation_size = float(config.get("model", {}).get("validation_size", 0.2))
    metric_name = str(config.get("model", {}).get("metric", "mae"))
    min_improvement = float(config.get("model", {}).get("min_improvement", 0.0))
    registered_model_name = get_registered_model_name(config)
    champion_alias = get_model_alias(config)

    features_df = read_parquet(args.features_uri)
    feature_pipeline = joblib_load_from_uri(args.feature_pipeline_uri)
    feature_schema_uri = companion_uri(args.feature_pipeline_uri, "feature_schema.json")
    feature_schema = read_optional_json(feature_schema_uri)

    X, y, prepared_df, feature_columns = prepare_supervised_data(
        features_df,
        target_col=target_col,
        date_col=date_col,
    )
    X_train, X_val, y_train, y_val, validation_frame = time_series_split(
        X=X,
        y=y,
        prepared_df=prepared_df,
        validation_size=validation_size,
    )
    model, estimator_name = train_model(X_train, y_train, config)
    validation_predictions = model.predict(X_val)
    metrics = calculate_metrics(y_val, validation_predictions)

    promoted_to_champion = False
    new_model_version: str | None = None

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        local_feature_pipeline = materialize_uri(args.feature_pipeline_uri, tmp_dir / "feature_pipeline.pkl")
        local_feature_schema = tmp_dir / "feature_schema.json"
        if feature_schema is not None:
            local_feature_schema.write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8")

        local_metrics = tmp_dir / "metrics.json"
        local_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        model_metadata = {
            "registered_model_name": registered_model_name,
            "estimator_name": estimator_name,
            "target_col": target_col,
            "date_col": date_col,
            "feature_columns": feature_columns,
            "features_uri": args.features_uri,
            "feature_pipeline_uri": args.feature_pipeline_uri,
            "feature_schema_uri": feature_schema_uri if feature_schema is not None else None,
            "validation_size": validation_size,
        }
        local_metadata = tmp_dir / "model_metadata.json"
        local_metadata.write_text(json.dumps(model_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        validation_output = X_val.copy()
        if date_col in validation_frame.columns:
            validation_output[date_col] = validation_frame[date_col].values
        validation_output[target_col] = y_val.values
        validation_output["prediction"] = validation_predictions
        local_predictions = tmp_dir / "validation_predictions.parquet"
        try:
            validation_output.to_parquet(local_predictions, index=False)
        except Exception as error:
            LOGGER.warning("Could not write parquet predictions, falling back to CSV: %s", error)
            local_predictions = tmp_dir / "validation_predictions.csv"
            validation_output.to_csv(local_predictions, index=False)

        local_importance = tmp_dir / "feature_importance.csv"
        importances = feature_importance_frame(model, feature_columns)
        if importances is not None:
            importances.to_csv(local_importance, index=False)

        with mlflow.start_run(run_name=args.run_name) as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            log_config_params(config)
            log_dataset_info(features_df, args.features_uri, prefix="train")
            mlflow.log_params(
                {
                    "estimator_name": estimator_name,
                    "feature_count": len(feature_columns),
                    "train_rows": len(X_train),
                    "validation_rows": len(X_val),
                    "feature_pipeline_type": type(feature_pipeline).__name__,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(local_feature_pipeline), artifact_path="artifacts")
            if feature_schema is not None:
                mlflow.log_artifact(str(local_feature_schema), artifact_path="artifacts")
            mlflow.log_artifact(str(local_metrics), artifact_path="artifacts")
            mlflow.log_artifact(str(local_metadata), artifact_path="artifacts")
            mlflow.log_artifact(str(local_predictions), artifact_path="artifacts")
            if importances is not None:
                mlflow.log_artifact(str(local_importance), artifact_path="artifacts")

            infer_and_log_model(
                model=model,
                X_train=X_train,
                registered_model_name=registered_model_name,
                artifact_path="model",
                input_example=X_train.head(5),
            )
            new_model_version = find_model_version_by_run_id(registered_model_name, run_id)

            if args.promote_if_better:
                current_version = get_current_alias_version(registered_model_name, champion_alias)
                old_metric = (
                    get_model_version_metric(registered_model_name, current_version, metric_name)
                    if current_version is not None
                    else None
                )
                if should_promote_model(
                    new_metric=metrics[metric_name],
                    old_metric=old_metric,
                    metric_name=metric_name,
                    min_improvement=min_improvement,
                ):
                    set_model_alias(registered_model_name, new_model_version, champion_alias)
                    promoted_to_champion = True

    print(f"run_id: {run_id}")
    print(f"experiment_id: {experiment_id}")
    print(f"registered_model_name: {registered_model_name}")
    print(f"new_model_version: {new_model_version}")
    print(f"promoted_to_champion: {str(promoted_to_champion).lower()}")


if __name__ == "__main__":
    main()
