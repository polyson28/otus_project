from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.models.mlflow_utils import set_model_alias, setup_mlflow


LOGGER = logging.getLogger("promote_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually promote an MLflow model version to an alias.")
    parser.add_argument("--model-name", required=True, help="MLflow registered model name.")
    parser.add_argument("--version", required=True, help="MLflow registered model version.")
    parser.add_argument("--alias", default="champion", help="Alias to assign. Defaults to champion.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    config = load_config(args.config)
    setup_mlflow(config)

    client = MlflowClient()
    try:
        client.get_registered_model(args.model_name)
        client.get_model_version(args.model_name, str(args.version))
    except MlflowException as error:
        raise SystemExit(
            f"Registered model/version not found: model={args.model_name}, version={args.version}. {error}"
        ) from error

    set_model_alias(args.model_name, str(args.version), args.alias)
    promoted_at = datetime.now(timezone.utc).isoformat()
    client.set_model_version_tag(args.model_name, str(args.version), "env", "production")
    client.set_model_version_tag(args.model_name, str(args.version), "promoted_at", promoted_at)
    client.set_model_version_tag(args.model_name, str(args.version), "promoted_by", "manual_cli")

    print(f"model_name: {args.model_name}")
    print(f"version: {args.version}")
    print(f"alias: {args.alias}")


if __name__ == "__main__":
    main()
