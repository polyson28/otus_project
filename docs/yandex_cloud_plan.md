# Yandex Cloud deployment plan

This document describes the target production layout for the TS forecasting project in Yandex Cloud.
Do not store real access keys, passwords, service-account keys, or OAuth tokens in this repository.

## Required services

- **Yandex Object Storage**
  - Stores raw data, interim datasets, features, prediction logs, monitoring reports, training reports, and MLflow artifacts.
  - Recommended bucket layout:
    - `s3://<bucket>/raw/`
    - `s3://<bucket>/interim/`
    - `s3://<bucket>/features/`
    - `s3://<bucket>/predictions/`
    - `s3://<bucket>/monitoring/`
    - `s3://<bucket>/training/`
    - `s3://<bucket>/mlflow-artifacts/`
    - `s3://<bucket>/airflow/dags/`

- **Yandex Managed Service for PostgreSQL**
  - MLflow backend store for experiments and Model Registry.
  - Airflow metadata DB is managed by Yandex Managed Service for Apache Airflow when using the managed service.

- **MLflow Tracking Server**
  - Can run on a VM or another container runtime.
  - Uses Managed PostgreSQL as backend store.
  - Uses Object Storage as artifact root.

- **Yandex Container Registry**
  - Stores immutable application images:
    - `cr.yandex/<registry_id>/ts-project-api:<tag>`
    - `cr.yandex/<registry_id>/ts-project-airflow:<tag>`
    - `cr.yandex/<registry_id>/ts-project-pipeline:<tag>`

- **Yandex Managed Service for Apache Airflow**
  - Recommended orchestrator for `ts_train_pipeline` and `ts_model_monitoring`.
  - DAGs are copied or synced to the Managed Airflow DAGs storage.
  - Airflow Variables and Connections are configured in the Airflow UI or through secret management/automation.

## Container images

Build and push images with a CI/CD job or locally:

```bash
yc container registry configure-docker

docker build -f Dockerfile.api -t cr.yandex/<registry_id>/ts-project-api:<tag> .
docker build -f Dockerfile.airflow -t cr.yandex/<registry_id>/ts-project-airflow:<tag> .
docker build -f Dockerfile.pipeline -t cr.yandex/<registry_id>/ts-project-pipeline:<tag> .

docker push cr.yandex/<registry_id>/ts-project-api:<tag>
docker push cr.yandex/<registry_id>/ts-project-airflow:<tag>
docker push cr.yandex/<registry_id>/ts-project-pipeline:<tag>
```

Use a unique `<tag>` per release, for example a git SHA.

## Environment variables

### Common

```text
YANDEX_S3_ENDPOINT_URL=https://storage.yandexcloud.net
YANDEX_S3_REGION=ru-central1
YANDEX_S3_BUCKET=<bucket>
AWS_DEFAULT_REGION=ru-central1
CONFIG_PATH=configs/config.yaml
LOG_LEVEL=INFO
```

### Object Storage URIs

```text
RAW_DATA_URI=s3://<bucket>/raw/
INTERIM_DATA_URI=s3://<bucket>/interim/data.parquet
REFERENCE_DATA_URI=s3://<bucket>/interim/reference_data.parquet
FEATURES_URI=s3://<bucket>/features/features.parquet
FEATURE_PIPELINE_URI=s3://<bucket>/features/feature_pipeline.pkl
FEATURE_SCHEMA_URI=s3://<bucket>/features/feature_schema.json
PREDICTIONS_URI=s3://<bucket>/predictions
REFERENCE_PREDICTIONS_URI=s3://<bucket>/predictions/reference_predictions.parquet
ACTUALS_URI=s3://<bucket>/actuals/actuals.parquet
MONITORING_REPORTS_URI=s3://<bucket>/monitoring
TRAIN_SUMMARY_URI=s3://<bucket>/training/train_summary_latest.json
EVALUATION_URI=s3://<bucket>/training/evaluation_latest.json
```

### MLflow

```text
MLFLOW_TRACKING_URI=http://<mlflow-service-host>:5001
MLFLOW_EXPERIMENT_NAME=ts-project
MLFLOW_REGISTERED_MODEL_NAME=ts-project-forecast-model
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://<bucket>/mlflow-artifacts
MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
MLFLOW_BACKEND_STORE_URI=postgresql://<user>:<password>@<postgres-host>:6432/<db>
```

### FastAPI

```text
MODEL_SOURCE=mlflow
REGISTERED_MODEL_NAME=ts_forecasting_model
MODEL_ALIAS=Production
MLFLOW_TRACKING_URI=http://<mlflow-service-host>:5001
API_RELOAD_URL=http://<api-service-name>:8000/reload_model
```

### Airflow Variables

Set these in Yandex Managed Airflow UI or via automation:

```text
RAW_DATA_URI
INTERIM_DATA_URI
REFERENCE_DATA_URI
FEATURES_URI
FEATURE_PIPELINE_URI
FEATURE_SCHEMA_URI
PREDICTIONS_URI
REFERENCE_PREDICTIONS_URI
ACTUALS_URI
MONITORING_REPORTS_URI
MLFLOW_TRACKING_URI
REGISTERED_MODEL_NAME
API_RELOAD_URL
TRAIN_RUN_NAME
TRAIN_SUMMARY_URI
EVALUATION_URI
```

## Secrets

Store these in Yandex Lockbox, Managed Airflow secrets, or CI/CD secret storage:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `MLFLOW_BACKEND_STORE_URI` or separate PostgreSQL user/password/host/db values
- Container Registry deploy credentials if not using a service account binding
- Any service account JSON/key material, if used

Recommended access model:

- API service account: read MLflow artifacts, write prediction logs.
- Airflow service account: read/write all project Object Storage prefixes, trigger API reload, access MLflow.
- MLflow service account: read/write `mlflow-artifacts/`.

## Production flow

1. Raw source data lands in `s3://<bucket>/raw/`.
2. Managed Airflow runs `ts_model_monitoring` daily:
   - reads current data and prediction logs;
   - writes reports to `s3://<bucket>/monitoring/`;
   - calls `decide_retrain.py`;
   - triggers `ts_train_pipeline` only when decision is `retrain`.
3. Managed Airflow runs `ts_train_pipeline` monthly or on demand:
   - preprocesses raw data to `interim/`;
   - builds features and feature pipeline in `features/`;
   - trains a candidate model;
   - logs metrics/artifacts to MLflow;
   - registers candidate in MLflow Model Registry;
   - promotes to `Production` if better;
   - calls `POST /reload_model` on the API if promotion happened.
4. FastAPI runs on a VM or another container runtime:
   - loads `models:/<registered_model>@Production` from MLflow;
   - serves predictions;
   - writes prediction logs to `s3://<bucket>/predictions/`.
5. MLflow Tracking Server stores metadata in Managed PostgreSQL and artifacts in Object Storage.

## Notes

- Managed Airflow should not depend on local `docker-compose` volumes. Use Object Storage paths and Airflow Variables instead.
- Keep DAG files in `dags/` and sync them to Managed Airflow DAGs storage.
- Prefer immutable image tags. Avoid deploying `latest` to production.
- Use network/security groups so Airflow can reach MLflow and API reload endpoint, while public access is limited to required endpoints.
