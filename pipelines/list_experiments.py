from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.models.mlflow_utils import get_experiment_name, get_registered_model_name, setup_mlflow


LOGGER = logging.getLogger("list_experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List MLflow experiments, recent runs, and model versions.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent runs to show.")
    return parser.parse_args()


def format_timestamp(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return ""
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def get_run_name(run: Any) -> str:
    return run.data.tags.get("mlflow.runName", "")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from mlflow.tracking import MlflowClient

    config = load_config(args.config)
    setup_mlflow(config)
    client = MlflowClient()

    experiment_name = get_experiment_name(config)
    current_experiment = client.get_experiment_by_name(experiment_name)

    print("Experiments:")
    for experiment in client.search_experiments():
        print(
            f"- experiment_id={experiment.experiment_id} "
            f"name={experiment.name} "
            f"lifecycle_stage={experiment.lifecycle_stage}"
        )

    print()
    print(f"Recent runs for experiment: {experiment_name}")
    if current_experiment is None:
        print("- no experiment found")
    else:
        runs = client.search_runs(
            experiment_ids=[current_experiment.experiment_id],
            max_results=args.limit,
            order_by=["attributes.start_time DESC"],
        )
        if not runs:
            print("- no runs found")
        for run in runs:
            print(
                f"- run_id={run.info.run_id} "
                f"run_name={get_run_name(run)} "
                f"status={run.info.status} "
                f"start_time={format_timestamp(run.info.start_time)} "
                f"metrics.mae={run.data.metrics.get('mae', '')} "
                f"metrics.rmse={run.data.metrics.get('rmse', '')} "
                f"params.model_type={run.data.params.get('model_type', run.data.params.get('estimator_name', ''))}"
            )

    print()
    print("Registered model versions:")
    model_name = get_registered_model_name(config)
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        print(f"- no versions found for model={model_name}")
    for version in versions:
        aliases = getattr(version, "aliases", None) or []
        print(
            f"- model_name={version.name} "
            f"version={version.version} "
            f"aliases={','.join(aliases)} "
            f"run_id={version.run_id} "
            f"creation_timestamp={format_timestamp(version.creation_timestamp)}"
        )


if __name__ == "__main__":
    main()
