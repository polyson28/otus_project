from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any, Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)


def _env_or_config(env_name: str, value: Any, default: Any = None) -> Any:
    env_value = os.getenv(env_name)
    if env_value not in (None, ""):
        return env_value
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        nested_env_value = os.getenv(value[2:-1])
        if nested_env_value not in (None, ""):
            return nested_env_value
        return default
    if value not in (None, ""):
        return value
    return default


def setup_mlflow(config: dict) -> None:
    import mlflow

    mlflow_config = config.get("mlflow", {})
    tracking_uri = _env_or_config("MLFLOW_TRACKING_URI", mlflow_config.get("tracking_uri"))
    experiment_name = get_experiment_name(config)
    artifact_root = _env_or_config(
        "MLFLOW_DEFAULT_ARTIFACT_ROOT",
        mlflow_config.get("artifact_root"),
        _env_or_config("MLFLOW_ARTIFACT_ROOT", mlflow_config.get("artifact_root")),
    )

    s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    if s3_endpoint_url:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint_url
        LOGGER.info("Using MLflow S3 endpoint: %s", s3_endpoint_url)

    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))
    if artifact_root and mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=str(artifact_root))
    mlflow.set_experiment(experiment_name)

    LOGGER.info(
        "MLflow configured: tracking_uri=%s, experiment=%s, artifact_root=%s",
        mlflow.get_tracking_uri(),
        experiment_name,
        artifact_root,
    )


def get_experiment_name(config: dict) -> str:
    mlflow_config = config.get("mlflow", {})
    return str(
        _env_or_config(
            "MLFLOW_EXPERIMENT_NAME",
            mlflow_config.get("experiment_name"),
            config.get("project_name", "Default"),
        )
    )


def get_registered_model_name(config: dict) -> str:
    mlflow_config = config.get("mlflow", {})
    return str(
        _env_or_config(
            "MLFLOW_REGISTERED_MODEL_NAME",
            mlflow_config.get("registered_model_name"),
            _env_or_config("MLFLOW_MODEL_NAME", mlflow_config.get("model_name"), "ts-project-forecast-model"),
        )
    )


def get_model_alias(config: dict) -> str:
    mlflow_config = config.get("mlflow", {})
    return str(
        _env_or_config(
            "MLFLOW_MODEL_ALIAS",
            mlflow_config.get("model_alias"),
            mlflow_config.get("champion_alias", "champion"),
        )
    )


def _get_git_commit() -> Optional[str]:
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


def log_config_params(config: dict) -> None:
    import mlflow

    feature_config = config.get("feature_engineering", {})
    model_config = config.get("model", {})
    params = {
        "target_col": config.get("target_col"),
        "date_col": config.get("date_col"),
        "prediction_horizon": config.get("prediction_horizon"),
        "lags": feature_config.get("lags"),
        "rolling_windows": feature_config.get("rolling_windows"),
        "validation_size": model_config.get("validation_size"),
        "model_metric": model_config.get("metric"),
    }
    git_commit = _get_git_commit()
    if git_commit:
        params["git_commit"] = git_commit

    safe_params = {key: str(value) for key, value in params.items() if value is not None}
    mlflow.log_params(safe_params)
    LOGGER.info("Logged MLflow config params: %s", sorted(safe_params))


def log_dataset_info(df: pd.DataFrame, dataset_uri: str, prefix: str = "train") -> None:
    import mlflow

    metrics = {
        f"{prefix}_rows": int(len(df)),
        f"{prefix}_columns_count": int(len(df.columns)),
    }
    params: dict[str, str] = {
        f"{prefix}_dataset_uri": dataset_uri,
    }

    date_col = "date" if "date" in df.columns else None
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not dates.empty:
            params[f"{prefix}_min_date"] = dates.min().isoformat()
            params[f"{prefix}_max_date"] = dates.max().isoformat()

    mlflow.log_metrics(metrics)
    mlflow.log_params(params)
    LOGGER.info(
        "Logged dataset info: prefix=%s, rows=%s, columns=%s, uri=%s",
        prefix,
        metrics[f"{prefix}_rows"],
        metrics[f"{prefix}_columns_count"],
        dataset_uri,
    )


def infer_and_log_model(
    model: Any,
    X_train: pd.DataFrame,
    registered_model_name: str,
    artifact_path: str = "model",
    input_example: Optional[pd.DataFrame] = None,
) -> str:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    example = input_example if input_example is not None else X_train.head(5)
    signature = None
    try:
        predictions = model.predict(example)
        signature = infer_signature(example, predictions)
        LOGGER.info("Inferred MLflow model signature.")
    except Exception as error:
        LOGGER.warning("Could not infer MLflow model signature: %s", error)

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
        input_example=example,
        signature=signature,
    )
    model_uri = getattr(model_info, "model_uri", None) or f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
    LOGGER.info(
        "Logged model to MLflow: registered_model_name=%s, model_uri=%s",
        registered_model_name,
        model_uri,
    )
    return str(model_uri)


def find_model_version_by_run_id(
    registered_model_name: str,
    run_id: str,
    timeout_seconds: int = 60,
) -> str:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        versions = client.search_model_versions(f"name = '{registered_model_name}'")
        for version in versions:
            if version.run_id == run_id:
                LOGGER.info(
                    "Found registered model version: model=%s, version=%s, run_id=%s",
                    registered_model_name,
                    version.version,
                    run_id,
                )
                return str(version.version)
        time.sleep(1)
    raise RuntimeError(
        f"Could not find registered model version for model={registered_model_name}, run_id={run_id}."
    )


def set_model_alias(
    registered_model_name: str,
    version: str | int,
    alias: str = "champion",
) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    version_str = str(version)
    try:
        client.set_registered_model_alias(registered_model_name, alias, version_str)
        LOGGER.info(
            "Set MLflow model alias: model=%s, version=%s, alias=%s",
            registered_model_name,
            version_str,
            alias,
        )
    except AttributeError:
        client.set_model_version_tag(registered_model_name, version_str, "env", "production")
        client.set_model_version_tag(registered_model_name, version_str, alias, "true")
        LOGGER.warning(
            "MLflow alias API is unavailable. Set fallback tags on model=%s, version=%s.",
            registered_model_name,
            version_str,
        )


def get_current_alias_version(
    registered_model_name: str,
    alias: str = "champion",
) -> Optional[str]:
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_model_name, alias)
        LOGGER.info(
            "Current MLflow alias version: model=%s, alias=%s, version=%s",
            registered_model_name,
            alias,
            version.version,
        )
        return str(version.version)
    except (AttributeError, MlflowException) as error:
        LOGGER.info(
            "No MLflow model alias found: model=%s, alias=%s, error=%s",
            registered_model_name,
            alias,
            error,
        )
        return None


def get_model_version_metric(
    registered_model_name: str,
    version: str | int,
    metric_name: str,
) -> Optional[float]:
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    version_str = str(version)
    try:
        model_version = client.get_model_version(registered_model_name, version_str)
        run = client.get_run(model_version.run_id)
    except MlflowException as error:
        LOGGER.info(
            "Could not load metric for model=%s, version=%s: %s",
            registered_model_name,
            version_str,
            error,
        )
        return None

    metric = run.data.metrics.get(metric_name)
    LOGGER.info(
        "Loaded MLflow metric: model=%s, version=%s, metric=%s, value=%s",
        registered_model_name,
        version_str,
        metric_name,
        metric,
    )
    return float(metric) if metric is not None else None


def should_promote_model(
    new_metric: float,
    old_metric: Optional[float],
    metric_name: str = "mae",
    min_improvement: float = 0.0,
) -> bool:
    lower_is_better = metric_name.lower() in {"mae", "rmse", "mape"}
    if old_metric is None:
        LOGGER.info("Promoting model because no previous %s metric exists.", metric_name)
        return True
    if not lower_is_better:
        LOGGER.warning("Metric %s is not configured as lower-is-better. Promotion denied.", metric_name)
        return False

    should_promote = new_metric <= old_metric - min_improvement
    LOGGER.info(
        "Promotion decision for %s: new=%s, old=%s, min_improvement=%s, promote=%s",
        metric_name,
        new_metric,
        old_metric,
        min_improvement,
        should_promote,
    )
    return should_promote
