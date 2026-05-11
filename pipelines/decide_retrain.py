from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.io.s3 import read_json, write_json
from src.monitoring.retrain_decision import decide_retrain


LOGGER = logging.getLogger("decide_retrain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide whether production model retraining should run.")
    parser.add_argument("--data-report-uri", default=os.getenv("DATA_REPORT_URI"))
    parser.add_argument("--prediction-report-uri", default=os.getenv("PREDICTION_REPORT_URI"))
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--config", default=os.getenv("CONFIG_PATH", "configs/config.yaml"))
    return parser.parse_args()


def _read_optional_report(uri: str | None) -> dict[str, Any] | None:
    if not uri:
        return None
    report = read_json(uri)
    if not isinstance(report, dict):
        raise ValueError(f"Monitoring report must be a JSON object: {uri}")
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    data_report = _read_optional_report(args.data_report_uri)
    prediction_report = _read_optional_report(args.prediction_report_uri)
    decision = decide_retrain(data_report, prediction_report, config)

    write_json(decision, args.output_uri)
    LOGGER.info("Retrain decision written to %s", args.output_uri)
    print(args.output_uri)
    print(f"decision: {decision['decision']}")
    print(f"need_retrain: {decision['need_retrain']}")


if __name__ == "__main__":
    main()
