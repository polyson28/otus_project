from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistryResult:
    model_uri: str
    model_version: str | None
    registered: bool
    error: str | None = None


def log_model_candidate(
    *,
    model: Any,
    X_train,
    registered_model_name: str,
    artifact_path: str = "model",
    input_example=None,
) -> RegistryResult:
    import mlflow
    import mlflow.sklearn
    from mlflow.exceptions import MlflowException
    from mlflow.models import infer_signature

    example = input_example if input_example is not None else X_train.head(5)
    signature = None
    try:
        signature = infer_signature(example, model.predict(example))
    except Exception as error:
        LOGGER.warning("Could not infer MLflow signature: %s", error)

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        input_example=example,
        signature=signature,
    )
    model_uri = str(getattr(model_info, "model_uri", f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"))

    try:
        registered = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        version = str(getattr(registered, "version", "")) or find_model_version_by_run_id(
            registered_model_name,
            mlflow.active_run().info.run_id,
        )
        return RegistryResult(model_uri=model_uri, model_version=version, registered=True)
    except MlflowException as error:
        LOGGER.warning(
            "MLflow Model Registry is unavailable for '%s'. Logging model artifact without registration. "
            "Use a database-backed MLflow backend store for registry support. Error: %s",
            registered_model_name,
            error,
        )
        mlflow.set_tag("registry_status", "candidate_not_registered")
        mlflow.set_tag("registry_error", str(error)[:500])
        return RegistryResult(model_uri=model_uri, model_version=None, registered=False, error=str(error))


def find_model_version_by_run_id(
    registered_model_name: str,
    run_id: str,
    timeout_seconds: int = 60,
) -> str | None:
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            versions = client.search_model_versions(f"name = '{registered_model_name}'")
        except MlflowException as error:
            LOGGER.warning("Could not search MLflow model versions: %s", error)
            return None
        for version in versions:
            if version.run_id == run_id:
                return str(version.version)
        time.sleep(1)
    return None


def get_current_production_version(
    registered_model_name: str,
    *,
    alias: str = "Production",
) -> str | None:
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_model_name, alias)
        return str(version.version)
    except (AttributeError, MlflowException):
        pass

    try:
        versions = client.get_latest_versions(registered_model_name, stages=["Production"])
    except (AttributeError, MlflowException) as error:
        LOGGER.info("No Production model found for %s: %s", registered_model_name, error)
        return None
    if not versions:
        return None
    return str(versions[0].version)


def get_model_version_metric(
    registered_model_name: str,
    version: str | int,
    metric_name: str,
) -> float | None:
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        model_version = client.get_model_version(registered_model_name, str(version))
        run = client.get_run(model_version.run_id)
    except MlflowException as error:
        LOGGER.info("Could not load metric for model=%s version=%s: %s", registered_model_name, version, error)
        return None
    metric = run.data.metrics.get(metric_name)
    return float(metric) if metric is not None else None


def should_promote(
    *,
    new_metric: float,
    production_metric: float | None,
    metric_name: str = "mae",
    min_improvement: float = 0.0,
) -> bool:
    lower_is_better = metric_name.lower() in {"mae", "rmse", "mape"}
    if not lower_is_better:
        LOGGER.warning("Promotion metric %s is not configured as lower-is-better.", metric_name)
        return False
    if production_metric is None:
        return True
    return new_metric <= production_metric - min_improvement


def promote_to_production(
    *,
    registered_model_name: str,
    new_version: str | int,
    previous_version: str | int | None,
    alias: str = "Production",
) -> None:
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    version_str = str(new_version)
    previous_version_str = str(previous_version) if previous_version is not None else None

    try:
        client.set_registered_model_alias(registered_model_name, alias, version_str)
        client.set_model_version_tag(registered_model_name, version_str, "env", "production")
        if previous_version_str:
            client.set_model_version_tag(registered_model_name, previous_version_str, "env", "previous")
            client.set_model_version_tag(registered_model_name, previous_version_str, "previous", "true")
        return
    except (AttributeError, MlflowException) as alias_error:
        LOGGER.warning("Could not set MLflow alias %s: %s. Trying stage transition.", alias, alias_error)

    try:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version_str,
            stage="Production",
            archive_existing_versions=True,
        )
        if previous_version_str:
            client.set_model_version_tag(registered_model_name, previous_version_str, "env", "previous")
    except MlflowException as error:
        raise RuntimeError(f"Could not promote MLflow model version to Production: {error}") from error
