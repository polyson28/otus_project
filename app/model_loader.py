from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
import tempfile
from typing import Any

from dotenv import load_dotenv
import joblib

from src.io.s3 import joblib_load_from_uri, read_json


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    feature_pipeline: Any
    metadata: dict[str, Any]
    model_uri: str
    feature_pipeline_uri: str
    metadata_uri: str
    model_version: str | None
    loaded_at: str


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else None


def _is_mlflow_uri(uri: str) -> bool:
    return uri.startswith("models:/") or uri.startswith("runs:/")


def _model_source() -> str:
    return (_optional_env("MODEL_SOURCE") or "auto").strip().lower()


def _default_mlflow_model_uri() -> str | None:
    model_name = _mlflow_model_name_candidates()[0] if _mlflow_model_name_candidates() else None
    alias = _optional_env("MODEL_ALIAS") or _optional_env("MLFLOW_MODEL_ALIAS") or "latest"
    if not model_name:
        return None
    return f"models:/{model_name}@{alias}"


def _unique(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in (None, ""):
            continue
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _config_registered_model_name() -> str | None:
    try:
        from src.config import load_config

        config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
        config = load_config(config_path)
    except Exception as error:
        LOGGER.info("Could not load config registered model name: %s", error)
        return None
    value = config.get("mlflow", {}).get("registered_model_name")
    return str(value) if value not in (None, "") else None


def _mlflow_model_name_candidates() -> list[str]:
    return _unique(
        [
            _optional_env("MLFLOW_REGISTERED_MODEL_NAME"),
            _optional_env("MLFLOW_MODEL_NAME"),
            _optional_env("REGISTERED_MODEL_NAME"),
            _config_registered_model_name(),
            "ts-project-forecast-model",
        ]
    )


def _resolve_latest_model_uri(model_name: str) -> str:
    _setup_mlflow_from_env()
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name = '{model_name}'")
    except MlflowException as error:
        raise RuntimeError(f"Could not search MLflow model versions for {model_name}: {error}") from error
    if not versions:
        raise RuntimeError(f"Registered Model with name={model_name} has no versions")

    latest = max(versions, key=lambda version: int(version.version))
    return f"models:/{model_name}/{latest.version}"


def _mlflow_model_uri_candidates() -> list[str]:
    explicit_uri = _optional_env("MODEL_URI") or _optional_env("MODEL_REGISTRY_URI")
    model_version = _optional_env("MODEL_VERSION")
    alias = _optional_env("MODEL_ALIAS") or _optional_env("MLFLOW_MODEL_ALIAS") or "latest"
    alias_is_latest = alias.strip().lower() in {"latest", "latest_version", "none"}

    candidates: list[str] = []
    for model_name in _mlflow_model_name_candidates():
        if model_version:
            candidates.append(f"models:/{model_name}/{model_version}")
        elif alias_is_latest:
            try:
                candidates.append(_resolve_latest_model_uri(model_name))
            except Exception as error:
                LOGGER.warning("Could not resolve latest MLflow version for model=%s: %s", model_name, error)
        else:
            candidates.append(f"models:/{model_name}@{alias}")
            try:
                candidates.append(_resolve_latest_model_uri(model_name))
            except Exception as error:
                LOGGER.warning(
                    "Could not add latest fallback for MLflow model=%s after alias=%s: %s",
                    model_name,
                    alias,
                    error,
                )

    if explicit_uri and _is_mlflow_uri(explicit_uri):
        candidates.append(explicit_uri)
    return _unique(candidates)


def _load_first_available_mlflow_model() -> ModelBundle:
    errors: list[str] = []
    for model_uri in _mlflow_model_uri_candidates():
        try:
            return _load_from_mlflow(model_uri)
        except Exception as error:
            errors.append(f"{model_uri}: {error}")
            LOGGER.warning("Could not load MLflow model candidate %s: %s", model_uri, error)
    raise RuntimeError("Could not load any MLflow model candidate. " + " | ".join(errors))


def _extract_model_version(metadata: dict[str, Any]) -> str | None:
    for key in ("model_version", "version", "new_model_version", "run_id"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _setup_mlflow_from_env() -> None:
    from src.config import load_config
    from src.models.mlflow_utils import setup_mlflow

    config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    setup_mlflow(load_config(config_path))


def _load_mlflow_model(model_uri: str) -> Any:
    import mlflow.pyfunc
    import mlflow.sklearn

    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as pyfunc_error:
        LOGGER.warning("Could not load MLflow model via pyfunc, trying sklearn: %s", pyfunc_error)
        return mlflow.sklearn.load_model(model_uri)


def _parse_models_uri(model_uri: str) -> tuple[str, str | None, str | None]:
    if not model_uri.startswith("models:/"):
        return "", None, None

    remainder = model_uri.removeprefix("models:/").lstrip("/")
    if "@" in remainder:
        model_name, alias = remainder.rsplit("@", 1)
        return model_name, None, alias

    parts = remainder.rsplit("/", 1)
    if len(parts) == 2:
        model_name, version = parts
        return model_name, version, None

    return remainder, None, None


def _resolve_mlflow_run(model_uri: str) -> tuple[str | None, str | None]:
    if model_uri.startswith("runs:/"):
        remainder = model_uri.removeprefix("runs:/").lstrip("/")
        run_id = remainder.split("/", 1)[0]
        return run_id, None

    model_name, version, alias = _parse_models_uri(model_uri)
    if not model_name:
        return None, None

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    if alias:
        model_version = client.get_model_version_by_alias(model_name, alias)
    elif version:
        model_version = client.get_model_version(model_name, version)
    else:
        return None, None

    return model_version.run_id, str(model_version.version)


def _download_mlflow_artifact(run_id: str, artifact_path: str) -> str:
    from mlflow.tracking import MlflowClient

    local_dir = tempfile.mkdtemp(prefix="ts_project_mlflow_artifacts_")
    return MlflowClient().download_artifacts(run_id, artifact_path, local_dir)


def _load_mlflow_companion_artifacts(
    run_id: str,
    feature_pipeline_uri: str | None,
    metadata_uri: str | None,
) -> tuple[Any, dict[str, Any], str, str]:
    if feature_pipeline_uri:
        LOGGER.info("Loading feature pipeline from %s", feature_pipeline_uri)
        feature_pipeline = joblib_load_from_uri(feature_pipeline_uri)
        resolved_feature_pipeline_uri = feature_pipeline_uri
    else:
        artifact_path = "artifacts/feature_pipeline.pkl"
        LOGGER.info("Loading feature pipeline from MLflow artifact: run_id=%s path=%s", run_id, artifact_path)
        local_feature_pipeline = _download_mlflow_artifact(run_id, artifact_path)
        feature_pipeline = joblib.load(local_feature_pipeline)
        resolved_feature_pipeline_uri = f"mlflow-artifact://{run_id}/{artifact_path}"

    if metadata_uri:
        LOGGER.info("Loading model metadata from %s", metadata_uri)
        metadata = read_json(metadata_uri)
        resolved_metadata_uri = metadata_uri
    else:
        artifact_path = "artifacts/model_metadata.json"
        LOGGER.info("Loading model metadata from MLflow artifact: run_id=%s path=%s", run_id, artifact_path)
        try:
            local_metadata = _download_mlflow_artifact(run_id, artifact_path)
            import json

            with open(local_metadata, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            resolved_metadata_uri = f"mlflow-artifact://{run_id}/{artifact_path}"
        except Exception as error:
            LOGGER.warning("Could not load model metadata artifact, using minimal metadata: %s", error)
            metadata = {"run_id": run_id}
            resolved_metadata_uri = f"missing://{run_id}/{artifact_path}"

    if not isinstance(metadata, dict):
        raise RuntimeError(f"Model metadata must be a JSON object: {resolved_metadata_uri}")

    return feature_pipeline, metadata, resolved_feature_pipeline_uri, resolved_metadata_uri


def _load_from_mlflow(model_uri: str) -> ModelBundle:
    _setup_mlflow_from_env()

    LOGGER.info("Loading model from MLflow URI %s", model_uri)
    model = _load_mlflow_model(model_uri)
    run_id, registry_version = _resolve_mlflow_run(model_uri)
    if not run_id:
        raise RuntimeError(f"Could not resolve MLflow run for model URI: {model_uri}")

    feature_pipeline, metadata, feature_pipeline_uri, metadata_uri = _load_mlflow_companion_artifacts(
        run_id=run_id,
        feature_pipeline_uri=_optional_env("FEATURE_PIPELINE_URI"),
        metadata_uri=_optional_env("MODEL_METADATA_URI"),
    )

    if registry_version and "model_version" not in metadata:
        metadata["model_version"] = registry_version
    if "run_id" not in metadata:
        metadata["run_id"] = run_id

    model_version = _extract_model_version(metadata)
    LOGGER.info("Model artifacts loaded: model_uri=%s model_version=%s", model_uri, model_version)

    return ModelBundle(
        model=model,
        feature_pipeline=feature_pipeline,
        metadata=metadata,
        model_uri=model_uri,
        feature_pipeline_uri=feature_pipeline_uri,
        metadata_uri=metadata_uri,
        model_version=model_version,
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )


def _load_from_joblib_uris(model_uri: str) -> ModelBundle:
    feature_pipeline_uri = _required_env("FEATURE_PIPELINE_URI")
    metadata_uri = _required_env("MODEL_METADATA_URI")

    LOGGER.info("Loading model from %s", model_uri)
    model = joblib_load_from_uri(model_uri)

    LOGGER.info("Loading feature pipeline from %s", feature_pipeline_uri)
    feature_pipeline = joblib_load_from_uri(feature_pipeline_uri)

    LOGGER.info("Loading model metadata from %s", metadata_uri)
    metadata = read_json(metadata_uri)
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Model metadata must be a JSON object: {metadata_uri}")

    model_version = _extract_model_version(metadata)
    LOGGER.info("Model artifacts loaded: model_uri=%s model_version=%s", model_uri, model_version)

    return ModelBundle(
        model=model,
        feature_pipeline=feature_pipeline,
        metadata=metadata,
        model_uri=model_uri,
        feature_pipeline_uri=feature_pipeline_uri,
        metadata_uri=metadata_uri,
        model_version=model_version,
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )


def load_model_bundle() -> ModelBundle:
    load_dotenv()

    source = _model_source()
    if source not in {"auto", "mlflow", "local"}:
        raise RuntimeError("MODEL_SOURCE must be one of: auto, mlflow, local")

    if source == "mlflow":
        return _load_first_available_mlflow_model()
    elif source == "local":
        model_uri = _optional_env("MODEL_URI")
    else:
        model_uri = _optional_env("MODEL_URI") or _optional_env("MODEL_REGISTRY_URI") or _default_mlflow_model_uri()

    if not model_uri:
        raise RuntimeError(
            "Missing model URI. Set MODEL_URI, MODEL_REGISTRY_URI, or "
            "MODEL_SOURCE=mlflow with REGISTERED_MODEL_NAME and MODEL_ALIAS."
        )

    if _is_mlflow_uri(model_uri):
        return _load_from_mlflow(model_uri)

    return _load_from_joblib_uris(model_uri)
