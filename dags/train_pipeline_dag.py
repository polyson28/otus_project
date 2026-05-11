from __future__ import annotations

from datetime import datetime
import os

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


PROJECT_DIR = "/opt/airflow/project"
CONFIG_PATH = "configs/config.yaml"


def setting(name: str, default: str) -> str:
    return setting_any((name,), default)


def setting_any(names: tuple[str, ...], default: str) -> str:
    for name in names:
        env_value = os.getenv(name)
        if env_value not in (None, ""):
            return env_value
    for name in names:
        variable_value = Variable.get(name, default_var=None)
        if variable_value not in (None, ""):
            return variable_value
    return default


RAW_DATA_URI = setting("RAW_DATA_URI", "Data/initial_data.xlsx")
INTERIM_DATA_URI = setting("INTERIM_DATA_URI", "artifacts/interim/data.parquet")
FEATURES_URI = setting("FEATURES_URI", "artifacts/features/features.parquet")
FEATURE_PIPELINE_URI = setting("FEATURE_PIPELINE_URI", "artifacts/features/feature_pipeline.pkl")
FEATURE_SCHEMA_URI = setting("FEATURE_SCHEMA_URI", "artifacts/features/feature_schema.json")
MLFLOW_EXPERIMENT_NAME = setting("MLFLOW_EXPERIMENT_NAME", "ts-project")
REGISTERED_MODEL_NAME = setting_any(
    ("MLFLOW_REGISTERED_MODEL_NAME", "REGISTERED_MODEL_NAME", "MLFLOW_MODEL_NAME"),
    "ts-project-forecast-model",
)
MLFLOW_TRACKING_URI = setting("MLFLOW_TRACKING_URI", "http://mlflow:5001")
API_RELOAD_URL = setting("API_RELOAD_URL", "http://api:8000/reload_model")


COMMON_ENV = {
    "PYTHONPATH": PROJECT_DIR,
    "RAW_DATA_URI": RAW_DATA_URI,
    "INTERIM_DATA_URI": INTERIM_DATA_URI,
    "FEATURES_URI": FEATURES_URI,
    "FEATURE_PIPELINE_URI": FEATURE_PIPELINE_URI,
    "FEATURE_SCHEMA_URI": FEATURE_SCHEMA_URI,
    "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
    "MLFLOW_REGISTERED_MODEL_NAME": REGISTERED_MODEL_NAME,
    "REGISTERED_MODEL_NAME": REGISTERED_MODEL_NAME,
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "API_RELOAD_URL": API_RELOAD_URL,
}


def pipeline_command(command: str) -> str:
    return f"set -euo pipefail\ncd {PROJECT_DIR}\n{command}"


def make_pipeline_task(task_id: str, command: str):
    command = pipeline_command(command)
    return BashOperator(task_id=task_id, bash_command=command, env=COMMON_ENV, append_env=True)


def reload_api_model() -> None:
    import urllib.request

    request = urllib.request.Request(API_RELOAD_URL, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        print(response.read().decode("utf-8"))


with DAG(
    dag_id="ts_train_pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "ts_project", "retries": 0},
    tags=["ts_project", "training"],
) as dag:
    preprocess_data = make_pipeline_task(
        "preprocess_data",
        'python pipelines/preprocess.py --input-uri "$RAW_DATA_URI" --output-uri "$INTERIM_DATA_URI" --config configs/config.yaml',
    )

    build_features = make_pipeline_task(
        "build_features",
        (
            'python pipelines/build_features.py --input-uri "$INTERIM_DATA_URI" '
            '--output-uri "$FEATURES_URI" --feature-pipeline-uri "$FEATURE_PIPELINE_URI" '
            '--feature-schema-uri "$FEATURE_SCHEMA_URI" --config configs/config.yaml --mode train'
        ),
    )

    train_model = make_pipeline_task(
        "train_model",
        (
            'python pipelines/train.py --features-uri "$FEATURES_URI" '
            '--feature-pipeline-uri "$FEATURE_PIPELINE_URI" --experiment-name "$MLFLOW_EXPERIMENT_NAME" '
            '--registered-model-name "$REGISTERED_MODEL_NAME" --config configs/config.yaml '
            "--promote-if-better true"
        ),
    )

    reload_api_model = PythonOperator(
        task_id="reload_api_model",
        python_callable=reload_api_model,
    )

    preprocess_data >> build_features >> train_model >> reload_api_model
