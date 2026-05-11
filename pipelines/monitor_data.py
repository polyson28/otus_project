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
from src.monitoring.drift import data_drift_report
from src.monitoring.quality import data_quality_report
from src.monitoring.report import make_report, read_table, report_uri, worst_status, write_report


LOGGER = logging.getLogger("monitor_data")
DEFAULT_REPORTS_URI = "artifacts/monitoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch data quality and drift monitoring.")
    parser.add_argument("--data-uri", default=os.getenv("DATA_URI"), help="Fresh production data URI/path.")
    parser.add_argument(
        "--reference-data-uri",
        default=os.getenv("REFERENCE_DATA_URI"),
        help="Reference dataset URI/path.",
    )
    parser.add_argument("--config", default=os.getenv("CONFIG_PATH", "configs/config.yaml"))
    parser.add_argument(
        "--output-uri",
        default=os.getenv("DATA_MONITORING_REPORT_URI"),
        help="Exact output JSON URI/path. Overrides --reports-uri when provided.",
    )
    parser.add_argument(
        "--reports-uri",
        default=os.getenv("MONITORING_REPORTS_URI", DEFAULT_REPORTS_URI),
        help="Directory/base URI for dated JSON monitoring reports.",
    )
    parser.add_argument("--report-date", default=os.getenv("MONITORING_DATE"))
    return parser.parse_args()


def _required(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required argument or env variable: {name}")
    return value


def _thresholds(config: dict[str, Any]) -> dict[str, Any]:
    monitoring = config.get("monitoring", {})
    return {
        "min_rows_for_check": int(monitoring.get("min_rows_for_check", 30)),
        "psi_threshold_warning": float(monitoring.get("psi_threshold_warning", 0.1)),
        "psi_threshold_critical": float(monitoring.get("psi_threshold_critical", 0.25)),
        "mae_degradation_ratio": float(monitoring.get("mae_degradation_ratio", 1.2)),
        "anomaly_retrain_min_consecutive_days": int(
            monitoring.get("anomaly_retrain_min_consecutive_days", 2)
        ),
    }


def build_report(
    current_uri: str,
    reference_uri: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = _thresholds(config)
    date_col = str(config.get("date_col", "date"))

    try:
        current = read_table(current_uri)
    except Exception as error:
        LOGGER.warning("Could not read current data from %s: %s", current_uri, error)
        return make_report(
            status="critical",
            need_retrain=False,
            reasons=["data_quality:missing_current_data", "alert:data_quality_critical"],
            metrics={
                "data_uri": current_uri,
                "reference_data_uri": reference_uri,
                "quality": {"error": str(error)},
                "drift": {"skipped": True, "reason": "missing_current_data"},
            },
            thresholds=thresholds,
        )

    reference_missing_error: str | None = None
    try:
        reference = read_table(reference_uri)
    except Exception as error:
        reference_missing_error = str(error)
        LOGGER.warning(
            "Could not read reference data from %s. Using current data as a temporary "
            "reference so monitoring can produce a warning report: %s",
            reference_uri,
            error,
        )
        reference = current.copy()

    quality_metrics, quality_reasons, quality_status = data_quality_report(
        current,
        reference,
        date_col=date_col,
        min_rows=thresholds["min_rows_for_check"],
    )
    drift_metrics, drift_reasons, drift_status = data_drift_report(
        reference,
        current,
        psi_warning=thresholds["psi_threshold_warning"],
        psi_critical=thresholds["psi_threshold_critical"],
        exclude_columns={date_col},
    )

    status = worst_status(quality_status, drift_status)
    reasons = quality_reasons + drift_reasons
    data_quality_is_critical = quality_status == "critical"
    need_retrain = bool(drift_status == "critical" and not data_quality_is_critical)

    if reference_missing_error is not None:
        status = worst_status(status, "warning")
        reasons.append("data_drift:missing_reference_data")
        drift_metrics["reference_fallback"] = {
            "used_current_as_reference": True,
            "error": reference_missing_error,
        }
        need_retrain = False

    if data_quality_is_critical:
        reasons.append("alert:data_quality_critical")

    return make_report(
        status=status,
        need_retrain=need_retrain,
        reasons=reasons,
        metrics={
            "data_uri": current_uri,
            "reference_data_uri": reference_uri,
            "quality": quality_metrics,
            "drift": drift_metrics,
        },
        thresholds=thresholds,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    data_uri = _required(args.data_uri, "DATA_URI")
    reference_data_uri = _required(args.reference_data_uri, "REFERENCE_DATA_URI")

    output_uri = args.output_uri or report_uri(args.reports_uri, "data_report", args.report_date)
    report = build_report(data_uri, reference_data_uri, config)
    write_report(report, output_uri)

    LOGGER.info("Data monitoring report written to %s", output_uri)
    print(output_uri)
    print(f"status: {report['status']}")
    print(f"need_retrain: {report['need_retrain']}")


if __name__ == "__main__":
    main()
