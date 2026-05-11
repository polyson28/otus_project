from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.io.s3 import (
    download_file,
    is_s3_uri,
    joblib_load_from_uri,
    S3ObjectNotFoundError,
    latest_object_uri,
    read_csv,
    read_excel,
    read_json,
    read_parquet,
    write_json,
)
from src.models.evaluate import backtest_time_windows, date_range_payload, prefixed_metrics
from src.models.mlflow_registry import (
    get_current_production_version,
    get_model_version_metric,
    log_model_candidate,
    promote_to_production,
    should_promote,
)
from src.models.train import (
    evaluate_split,
    feature_importance_frame,
    prepare_supervised_dataset,
    split_train_valid_test,
    train_model,
)
from src.models.mlflow_utils import setup_mlflow


LOGGER = logging.getLogger("train")
DEFAULT_REPORTS_DIR = "artifacts/training"


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
    parser = argparse.ArgumentParser(description="Train and register a time series model in MLflow.")
    parser.add_argument("--features-uri", required=True, help="Input features dataset URI/path.")
    parser.add_argument("--feature-pipeline-uri", required=True, help="Feature pipeline joblib URI/path.")
    parser.add_argument("--experiment-name", required=True, help="MLflow experiment name.")
    parser.add_argument("--registered-model-name", required=True, help="MLflow registered model name.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Optional MLflow run name.")
    parser.add_argument("--promote-if-better", type=parse_bool, default=True, help="Promote when validation MAE improves.")
    return parser.parse_args()


def read_features(uri: str) -> pd.DataFrame:
    lower_uri = uri.lower()
    if lower_uri.endswith(".parquet"):
        return read_parquet(uri)
    if lower_uri.endswith(".csv"):
        return read_csv(uri)
    if lower_uri.endswith((".xlsx", ".xls")):
        return read_excel(uri)
    raise ValueError(f"Unsupported features dataset format: {uri}")


def companion_uri(uri: str, filename: str) -> str:
    return uri.rsplit("/", 1)[0] + "/" + filename if "/" in uri else filename


def read_optional_json(uri: str) -> dict[str, Any] | None:
    try:
        payload = read_json(uri)
    except Exception as error:
        LOGGER.info("Optional JSON is unavailable at %s: %s", uri, error)
        return None
    return payload if isinstance(payload, dict) else None


def load_feature_schema(feature_pipeline_uri: str, config: dict[str, Any]) -> dict[str, Any] | None:
    schema_uri = os.getenv("FEATURE_SCHEMA_URI") or companion_uri(feature_pipeline_uri, "feature_schema.json")
    schema = read_optional_json(schema_uri)
    if schema is not None:
        return schema

    prefix = s3_features_prefix(config)
    if not prefix:
        return None
    try:
        latest_schema_uri = latest_object_uri(
            f"{prefix.rstrip('/')}/train",
            suffixes=(".json",),
            preferred_filename="feature_schema.json",
        )
    except Exception as error:
        LOGGER.info("Could not find latest feature schema in S3: %s", error)
        return None
    return read_optional_json(latest_schema_uri)


def s3_features_prefix(config: dict[str, Any]) -> str | None:
    env_uri = os.getenv("FEATURES_S3_PREFIX") or os.getenv("FEATURES_PREFIX_URI")
    if env_uri:
        return env_uri
    s3_config = config.get("s3", {})
    bucket = os.getenv("YANDEX_S3_BUCKET") or s3_config.get("bucket")
    prefix = str(s3_config.get("features_prefix", "features/")).strip("/")
    if not bucket:
        return None
    return f"s3://{bucket}/{prefix}"


def resolve_input_uri(
    uri: str,
    *,
    config: dict[str, Any],
    env_name: str,
    suffixes: tuple[str, ...],
) -> str:
    env_uri = os.getenv(env_name)
    if env_uri:
        LOGGER.info("Using %s from environment: %s", env_name, env_uri)
        return env_uri
    if is_s3_uri(uri) or Path(uri).exists():
        return uri

    prefix = s3_features_prefix(config)
    if not prefix:
        return uri

    preferred_prefixes = [
        f"{prefix.rstrip('/')}/train",
        prefix,
    ]
    errors: list[str] = []
    for candidate_prefix in preferred_prefixes:
        try:
            latest_uri = latest_object_uri(
                candidate_prefix,
                suffixes=suffixes,
                preferred_filename=Path(uri).name,
            )
            break
        except S3ObjectNotFoundError as error:
            errors.append(str(error))
    else:
        raise S3ObjectNotFoundError("Could not find training input in S3. " + " | ".join(errors))

    LOGGER.warning("Local input %s not found. Using latest S3 artifact: %s", uri, latest_uri)
    return latest_uri


def ensure_supervised_features(
    features_df: pd.DataFrame,
    *,
    feature_pipeline: Any,
    feature_pipeline_uri: str,
    config: dict[str, Any],
    target_col: str,
    date_col: str,
) -> pd.DataFrame:
    if target_col in features_df.columns:
        return features_df

    schema = load_feature_schema(feature_pipeline_uri, config)
    source_data_uri = schema.get("source_data_uri") if schema else None
    if not source_data_uri:
        raise ValueError(
            f"Missing target column: {target_col}. Feature schema has no source_data_uri to rebuild supervised data."
        )

    LOGGER.warning(
        "Features do not contain target column %s. Rebuilding supervised training frame from %s.",
        target_col,
        source_data_uri,
    )
    source_df = read_features(str(source_data_uri))
    if target_col not in source_df.columns:
        raise ValueError(f"Source data {source_data_uri} does not contain target column: {target_col}")
    if date_col not in source_df.columns:
        raise ValueError(f"Source data {source_data_uri} does not contain date column: {date_col}")

    rebuilt = feature_pipeline.transform(source_df)
    aligned_source = source_df.loc[rebuilt.index, [date_col, target_col]].reset_index(drop=True)
    rebuilt = rebuilt.reset_index(drop=True)
    rebuilt[date_col] = aligned_source[date_col].values
    rebuilt[target_col] = aligned_source[target_col].values
    feature_columns = getattr(feature_pipeline, "feature_columns", [])
    if feature_columns:
        rebuilt = rebuilt.dropna(subset=list(feature_columns))
    rebuilt = rebuilt.dropna(subset=[target_col]).reset_index(drop=True)
    return rebuilt


def materialize_uri(uri: str, local_path: Path) -> Path:
    if is_s3_uri(uri):
        download_file(uri, local_path)
    else:
        shutil.copyfile(Path(uri), local_path)
    return local_path


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as error:
        LOGGER.info("Git commit is unavailable: %s", error)
        return None
    commit = result.stdout.strip()
    return commit or None


def set_experiment(experiment_name: str, config: dict[str, Any]) -> None:
    import mlflow

    setup_mlflow(config)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_root = (
            os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")
            or os.getenv("MLFLOW_ARTIFACT_ROOT")
            or config.get("mlflow", {}).get("artifact_root")
        )
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_root)
    mlflow.set_experiment(experiment_name)


def log_params_tags(
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    estimator_name: str,
    feature_columns: list[str],
    split,
    feature_pipeline: Any,
) -> None:
    import mlflow

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    commit = git_commit()
    params = {
        "target_col": config.get("target_col"),
        "date_col": config.get("date_col"),
        "prediction_horizon": config.get("prediction_horizon"),
        "model_kind": training_config.get("model_kind"),
        "fallback_model": training_config.get("fallback_model"),
        "time_budget": training_config.get("time_budget"),
        "metric": model_config.get("metric", "mae"),
        "validation_size": model_config.get("validation_size", 0.2),
        "feature_count": len(feature_columns),
        "train_rows": len(split.X_train),
        "validation_rows": len(split.X_valid),
        "test_rows": len(split.X_test),
        "estimator_name": estimator_name,
        "feature_pipeline_type": type(feature_pipeline).__name__,
    }
    mlflow.log_params({key: str(value) for key, value in params.items() if value is not None})
    mlflow.set_tags(
        {
            "model_type": estimator_name,
            "dataset_uri": getattr(args, "resolved_features_uri", args.features_uri),
            "feature_pipeline_uri": getattr(args, "resolved_feature_pipeline_uri", args.feature_pipeline_uri),
            "created_by": "train_pipeline",
            **({"git_commit": commit} if commit else {}),
        }
    )


def predictions_output_frame(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series | None,
    predictions,
    *,
    target_col: str,
    date_col: str,
) -> pd.DataFrame:
    output = X.copy()
    if dates is not None:
        output[date_col] = dates.values
    output[target_col] = y.values
    output["prediction"] = predictions
    return output


def write_dataframe_artifact(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> Path:
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as error:
        LOGGER.warning("Could not write parquet artifact %s, falling back to CSV: %s", parquet_path, error)
        df.to_csv(csv_path, index=False)
        return csv_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import mlflow

    config = load_config(args.config)
    set_experiment(args.experiment_name, config)

    target_col = str(config["target_col"])
    date_col = str(config["date_col"])
    model_config = config.get("model", {})
    validation_size = float(model_config.get("validation_size", 0.2))
    test_size = float(model_config.get("test_size", validation_size))
    metric_name = str(model_config.get("metric", "mae"))
    min_improvement = float(model_config.get("min_improvement", 0.0))
    production_alias = str(config.get("mlflow", {}).get("production_alias", "Production"))

    features_uri = resolve_input_uri(
        args.features_uri,
        config=config,
        env_name="FEATURES_URI",
        suffixes=(".parquet", ".csv"),
    )
    feature_pipeline_uri = resolve_input_uri(
        args.feature_pipeline_uri,
        config=config,
        env_name="FEATURE_PIPELINE_URI",
        suffixes=(".pkl", ".joblib"),
    )
    args.resolved_features_uri = features_uri
    args.resolved_feature_pipeline_uri = feature_pipeline_uri

    features_df = read_features(features_uri)
    feature_pipeline = joblib_load_from_uri(feature_pipeline_uri)
    features_df = ensure_supervised_features(
        features_df,
        feature_pipeline=feature_pipeline,
        feature_pipeline_uri=feature_pipeline_uri,
        config=config,
        target_col=target_col,
        date_col=date_col,
    )
    dataset = prepare_supervised_dataset(features_df, target_col=target_col, date_col=date_col)
    split = split_train_valid_test(dataset, validation_size=validation_size, test_size=test_size)
    model, estimator_name = train_model(split.X_train, split.y_train, config)

    train_metrics = evaluate_split(model, split.X_train, split.y_train)
    validation_predictions = model.predict(split.X_valid)
    validation_metrics = evaluate_split(model, split.X_valid, split.y_valid)
    test_predictions = model.predict(split.X_test)
    test_metrics = evaluate_split(model, split.X_test, split.y_test)
    backtest = backtest_time_windows(
        model,
        pd.concat([split.X_valid, split.X_test], ignore_index=True),
        pd.concat([split.y_valid, split.y_test], ignore_index=True),
        pd.concat([split.valid_dates, split.test_dates], ignore_index=True)
        if split.valid_dates is not None and split.test_dates is not None
        else None,
        n_windows=int(config.get("evaluation", {}).get("backtest_windows", 3)),
        min_window_size=int(config.get("evaluation", {}).get("backtest_min_window_size", 10)),
    )

    metrics_to_log = {
        **prefixed_metrics(train_metrics, "train"),
        **prefixed_metrics(validation_metrics, "validation"),
        **prefixed_metrics(test_metrics, "test"),
        metric_name: float(validation_metrics[metric_name]),
    }

    promoted_to_production = False
    promotion_reason = "promotion_disabled" if not args.promote_if_better else "not_evaluated"
    production_version = None
    production_metric = None
    registry_result = None
    run_id = None
    experiment_id = None

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        local_feature_pipeline = materialize_uri(feature_pipeline_uri, tmp_dir / "feature_pipeline.pkl")
        local_config = materialize_uri(args.config, tmp_dir / "config.yaml")

        local_metadata = tmp_dir / "model_metadata.json"
        local_metadata.write_text(
            json.dumps(
                {
                    "registered_model_name": args.registered_model_name,
                    "estimator_name": estimator_name,
                    "target_col": target_col,
                    "date_col": date_col,
                    "feature_columns": dataset.feature_columns,
                    "features_uri": features_uri,
                    "feature_pipeline_uri": feature_pipeline_uri,
                    "validation_size": validation_size,
                    "test_size": test_size,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        local_validation_predictions = write_dataframe_artifact(
            predictions_output_frame(
                split.X_valid,
                split.y_valid,
                split.valid_dates,
                validation_predictions,
                target_col=target_col,
                date_col=date_col,
            ),
            tmp_dir / "validation_predictions.parquet",
            tmp_dir / "validation_predictions.csv",
        )
        local_test_predictions = write_dataframe_artifact(
            predictions_output_frame(
                split.X_test,
                split.y_test,
                split.test_dates,
                test_predictions,
                target_col=target_col,
                date_col=date_col,
            ),
            tmp_dir / "test_predictions.parquet",
            tmp_dir / "test_predictions.csv",
        )

        local_backtest = tmp_dir / "backtest.json"
        local_backtest.write_text(json.dumps(backtest, ensure_ascii=False, indent=2), encoding="utf-8")

        local_importance = tmp_dir / "feature_importance.csv"
        importances = feature_importance_frame(model, dataset.feature_columns)
        if importances is not None:
            importances.to_csv(local_importance, index=False)

        with mlflow.start_run(run_name=args.run_name) as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            log_params_tags(
                config=config,
                args=args,
                estimator_name=estimator_name,
                feature_columns=dataset.feature_columns,
                split=split,
                feature_pipeline=feature_pipeline,
            )
            mlflow.log_metrics(metrics_to_log)
            mlflow.log_dict(
                {
                    "train": date_range_payload(split.train_dates),
                    "validation": date_range_payload(split.valid_dates),
                    "test": date_range_payload(split.test_dates),
                },
                "artifacts/date_ranges.json",
            )
            mlflow.log_dict({"feature_columns": dataset.feature_columns}, "artifacts/feature_columns.json")
            mlflow.log_artifact(str(local_feature_pipeline), artifact_path="artifacts")
            mlflow.log_artifact(str(local_config), artifact_path="artifacts")
            mlflow.log_artifact(str(local_metadata), artifact_path="artifacts")
            mlflow.log_artifact(str(local_validation_predictions), artifact_path="artifacts")
            mlflow.log_artifact(str(local_test_predictions), artifact_path="artifacts")
            mlflow.log_artifact(str(local_backtest), artifact_path="artifacts")
            if importances is not None:
                mlflow.log_artifact(str(local_importance), artifact_path="artifacts")

            registry_result = log_model_candidate(
                model=model,
                X_train=split.X_train,
                registered_model_name=args.registered_model_name,
                artifact_path="model",
                input_example=split.X_train.head(5),
            )

            mlflow.set_tag("registered_model_name", args.registered_model_name)
            mlflow.set_tag("registry_registered", str(registry_result.registered).lower())
            if registry_result.model_version:
                mlflow.set_tag("model_version", registry_result.model_version)

            if args.promote_if_better and registry_result.registered and registry_result.model_version:
                production_version = get_current_production_version(
                    args.registered_model_name,
                    alias=production_alias,
                )
                production_metric = (
                    get_model_version_metric(args.registered_model_name, production_version, metric_name)
                    if production_version
                    else None
                )
                if should_promote(
                    new_metric=validation_metrics[metric_name],
                    production_metric=production_metric,
                    metric_name=metric_name,
                    min_improvement=min_improvement,
                ):
                    promote_to_production(
                        registered_model_name=args.registered_model_name,
                        new_version=registry_result.model_version,
                        previous_version=production_version,
                        alias=production_alias,
                    )
                    promoted_to_production = True
                    promotion_reason = "new_model_better"
                    mlflow.set_tag("candidate_status", "promoted")
                else:
                    promotion_reason = "new_model_not_better"
                    mlflow.set_tag("candidate_status", "candidate")
            elif args.promote_if_better and not registry_result.registered:
                promotion_reason = "registry_unavailable"
                mlflow.set_tag("candidate_status", "candidate_registry_unavailable")
            else:
                mlflow.set_tag("candidate_status", "candidate")

    report = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_name": args.experiment_name,
        "registered_model_name": args.registered_model_name,
        "features_uri": features_uri,
        "feature_pipeline_uri": feature_pipeline_uri,
        "model_uri": registry_result.model_uri if registry_result else None,
        "model_version": registry_result.model_version if registry_result else None,
        "registry_registered": registry_result.registered if registry_result else False,
        "registry_error": registry_result.error if registry_result else None,
        "estimator_name": estimator_name,
        "feature_columns": dataset.feature_columns,
        "metrics": metrics_to_log,
        "backtest": backtest,
        "date_ranges": {
            "train": date_range_payload(split.train_dates),
            "validation": date_range_payload(split.valid_dates),
            "test": date_range_payload(split.test_dates),
        },
        "promotion": {
            "promote_if_better": args.promote_if_better,
            "promoted_to_production": promoted_to_production,
            "reason": promotion_reason,
            "production_alias": production_alias,
            "previous_production_version": production_version,
            "previous_production_metric": production_metric,
        },
    }
    report_uri = str(Path(DEFAULT_REPORTS_DIR) / f"train_report_{run_id}.json")
    write_json(report, report_uri)

    print(f"run_id: {run_id}")
    print(f"experiment_id: {experiment_id}")
    print(f"registered_model_name: {args.registered_model_name}")
    print(f"model_version: {report['model_version']}")
    print(f"registry_registered: {str(report['registry_registered']).lower()}")
    print(f"promoted_to_production: {str(promoted_to_production).lower()}")
    print(f"train_report_uri: {report_uri}")


if __name__ == "__main__":
    main()
