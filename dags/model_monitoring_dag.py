from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


PROJECT_DIR = "/opt/airflow/project"


def setting(name: str, default: str) -> str:
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return env_value
    return Variable.get(name, default_var=default)


INTERIM_DATA_URI = setting("INTERIM_DATA_URI", "artifacts/interim/data.parquet")
REFERENCE_DATA_URI = setting("REFERENCE_DATA_URI", "artifacts/interim/reference_data.parquet")
PREDICTIONS_URI = setting("PREDICTIONS_URI", "artifacts/predictions")
ACTUALS_URI = setting("ACTUALS_URI", "")
DATA_MONITORING_REPORT_URI = setting(
    "DATA_MONITORING_REPORT_URI",
    "artifacts/monitoring/data_report_latest.json",
)
PREDICTION_MONITORING_REPORT_URI = setting(
    "PREDICTION_MONITORING_REPORT_URI",
    "artifacts/monitoring/prediction_report_latest.json",
)
RETRAIN_DECISION_URI = setting(
    "RETRAIN_DECISION_URI",
    "artifacts/monitoring/retrain_decision_latest.json",
)
MLFLOW_TRACKING_URI = setting("MLFLOW_TRACKING_URI", "http://mlflow:5001")


COMMON_ENV = {
    "PYTHONPATH": PROJECT_DIR,
    "INTERIM_DATA_URI": INTERIM_DATA_URI,
    "REFERENCE_DATA_URI": REFERENCE_DATA_URI,
    "PREDICTIONS_URI": PREDICTIONS_URI,
    "ACTUALS_URI": ACTUALS_URI,
    "DATA_MONITORING_REPORT_URI": DATA_MONITORING_REPORT_URI,
    "PREDICTION_MONITORING_REPORT_URI": PREDICTION_MONITORING_REPORT_URI,
    "RETRAIN_DECISION_URI": RETRAIN_DECISION_URI,
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
}


def pipeline_command(command: str) -> str:
    return f"set -euo pipefail\ncd {PROJECT_DIR}\n{command}"


def make_pipeline_task(task_id: str, command: str):
    command = pipeline_command(command)
    return BashOperator(task_id=task_id, bash_command=command, env=COMMON_ENV, append_env=True)


def _local_project_path(path_or_uri: str) -> Path:
    path = Path(path_or_uri)
    if path.is_absolute():
        return path
    return Path(PROJECT_DIR) / path


def _read_json(path_or_uri: str) -> dict:
    if path_or_uri.startswith("s3://"):
        from src.io.s3 import read_json

        payload = read_json(path_or_uri)
    else:
        payload = json.loads(_local_project_path(path_or_uri).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path_or_uri}")
    return payload


def branch_on_decision() -> str:
    payload = _read_json(RETRAIN_DECISION_URI)
    decision = str(payload.get("decision", "")).lower()
    need_retrain = bool(payload.get("need_retrain"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if decision == "blocked":
        return "blocked_retrain"
    if decision == "retrain" or need_retrain:
        return "trigger_train_pipeline"
    return "skip_retrain"


with DAG(
    dag_id="ts_model_monitoring",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={"owner": "ts_project", "retries": 0},
    tags=["ts_project", "monitoring"],
) as dag:
    monitor_data = make_pipeline_task(
        "monitor_data",
        (
            'python pipelines/monitor_data.py --data-uri "$INTERIM_DATA_URI" '
            '--reference-data-uri "$REFERENCE_DATA_URI" '
            '--output-uri "$DATA_MONITORING_REPORT_URI" --config configs/config.yaml'
        ),
    )

    monitor_predictions = make_pipeline_task(
        "monitor_predictions",
        (
            'python pipelines/monitor_predictions.py --predictions-uri "$PREDICTIONS_URI" '
            '--actuals-uri "$ACTUALS_URI" --output-uri "$PREDICTION_MONITORING_REPORT_URI" '
            "--config configs/config.yaml"
        ),
    )

    decide_retrain = make_pipeline_task(
        "decide_retrain",
        (
            'python pipelines/decide_retrain.py --data-report-uri "$DATA_MONITORING_REPORT_URI" '
            '--prediction-report-uri "$PREDICTION_MONITORING_REPORT_URI" '
            '--output-uri "$RETRAIN_DECISION_URI" --config configs/config.yaml'
        ),
    )

    branch_on_retrain_decision = BranchPythonOperator(
        task_id="branch_on_retrain_decision",
        python_callable=branch_on_decision,
    )

    trigger_train_pipeline = TriggerDagRunOperator(
        task_id="trigger_train_pipeline",
        trigger_dag_id="ts_train_pipeline",
        wait_for_completion=True,
        poke_interval=30,
        conf={"triggered_by": "ts_model_monitoring", "decision_uri": RETRAIN_DECISION_URI},
    )
    blocked_retrain = EmptyOperator(task_id="blocked_retrain")
    skip_retrain = EmptyOperator(task_id="skip_retrain")

    [monitor_data, monitor_predictions] >> decide_retrain >> branch_on_retrain_decision
    branch_on_retrain_decision >> trigger_train_pipeline
    branch_on_retrain_decision >> blocked_retrain
    branch_on_retrain_decision >> skip_retrain
