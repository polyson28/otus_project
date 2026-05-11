from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.io.s3 import S3ObjectNotFoundError, is_s3_uri, read_csv, read_parquet
from src.monitoring.anomaly import (
    change_point_report,
    consecutive_anomaly_days,
    isolation_forest_anomaly_report,
)
from src.monitoring.drift import population_stability_index
from src.monitoring.quality import prediction_summary, regression_quality
from src.monitoring.report import make_report, read_table, report_uri, worst_status, write_report


LOGGER = logging.getLogger("monitor_predictions")
DEFAULT_PREDICTIONS_URI = "artifacts/predictions"
DEFAULT_REPORTS_URI = "artifacts/monitoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch prediction drift and quality monitoring.")
    parser.add_argument(
        "--predictions-uri",
        default=os.getenv("PREDICTIONS_URI", DEFAULT_PREDICTIONS_URI),
        help="Prediction log file URI/path or base directory URI/path.",
    )
    parser.add_argument(
        "--reference-predictions-uri",
        default=os.getenv("REFERENCE_PREDICTIONS_URI"),
        help="Reference predictions URI/path.",
    )
    parser.add_argument(
        "--actuals-uri",
        default=os.getenv("ACTUALS_URI"),
        help="Optional actual values URI/path.",
    )
    parser.add_argument("--config", default=os.getenv("CONFIG_PATH", "configs/config.yaml"))
    parser.add_argument(
        "--output-uri",
        default=os.getenv("PREDICTION_MONITORING_REPORT_URI"),
        help="Exact output JSON URI/path. Overrides --reports-uri when provided.",
    )
    parser.add_argument(
        "--reports-uri",
        default=os.getenv("MONITORING_REPORTS_URI", DEFAULT_REPORTS_URI),
        help="Directory/base URI for JSON monitoring reports.",
    )
    parser.add_argument("--report-date", default=os.getenv("MONITORING_DATE"))
    parser.add_argument("--baseline-mae", type=float, default=_float_env("BASELINE_MAE"))
    return parser.parse_args()


def _float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    return float(value)


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
        "outlier_share_warning": float(monitoring.get("outlier_share_warning", 0.1)),
    }


def _daily_prediction_log_uri(base_uri: str, report_date: str | None, suffix: str) -> str:
    day = report_date or datetime.now(timezone.utc).date().isoformat()
    filename = f"predictions_{day}.{suffix}"
    if is_s3_uri(base_uri):
        return f"{base_uri.rstrip('/')}/{filename}"
    return str(Path(base_uri) / filename)


def _read_prediction_logs(uri_or_base: str, report_date: str | None) -> tuple[pd.DataFrame, str]:
    lower_uri = uri_or_base.lower()
    if lower_uri.endswith(".parquet"):
        return read_parquet(uri_or_base), uri_or_base
    if lower_uri.endswith(".csv"):
        return read_csv(uri_or_base), uri_or_base

    parquet_uri = _daily_prediction_log_uri(uri_or_base, report_date, "parquet")
    try:
        return read_parquet(parquet_uri), parquet_uri
    except (FileNotFoundError, S3ObjectNotFoundError, ImportError, ValueError) as error:
        LOGGER.warning("Could not read parquet prediction logs from %s: %s", parquet_uri, error)
        csv_uri = _daily_prediction_log_uri(uri_or_base, report_date, "csv")
        return read_csv(csv_uri), csv_uri


def _parse_features(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _add_monitoring_date(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    prepared = frame.copy()
    if date_col in prepared.columns:
        prepared["_monitoring_date"] = pd.to_datetime(prepared[date_col], errors="coerce").dt.date
        return prepared

    if "features" in prepared.columns:
        prepared["_monitoring_date"] = prepared["features"].map(
            lambda value: _parse_features(value).get(date_col)
        )
        prepared["_monitoring_date"] = pd.to_datetime(prepared["_monitoring_date"], errors="coerce").dt.date
        if prepared["_monitoring_date"].notna().any():
            return prepared

    prepared["_monitoring_date"] = pd.to_datetime(prepared.get("created_at"), errors="coerce").dt.date
    return prepared


def _attach_actuals(
    predictions: pd.DataFrame,
    actuals_uri: str | None,
    *,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    prepared = predictions.copy()
    if "actual" in prepared.columns and prepared["actual"].notna().any():
        return prepared
    if "actual" in prepared.columns:
        prepared = prepared.drop(columns=["actual"])
    if not actuals_uri:
        return prepared

    actuals = read_table(actuals_uri)
    actual_column = "actual" if "actual" in actuals.columns else target_col
    if actual_column not in actuals.columns:
        raise ValueError(f"Actuals dataset must contain '{target_col}' or 'actual' column.")

    if "prediction_id" in prepared.columns and "prediction_id" in actuals.columns:
        joined = prepared.merge(
            actuals[["prediction_id", actual_column]].rename(columns={actual_column: "actual"}),
            on="prediction_id",
            how="left",
        )
        return joined

    prepared = _add_monitoring_date(prepared, date_col)
    if date_col not in actuals.columns:
        raise ValueError("Actuals dataset must contain prediction_id or configured date column.")
    actuals = actuals.copy()
    actuals["_monitoring_date"] = pd.to_datetime(actuals[date_col], errors="coerce").dt.date
    return prepared.merge(
        actuals[["_monitoring_date", actual_column]].rename(columns={actual_column: "actual"}),
        on="_monitoring_date",
        how="left",
    )


def _baseline_mae(reference: pd.DataFrame, cli_value: float | None) -> float | None:
    if cli_value is not None:
        return cli_value
    if {"prediction", "actual"}.issubset(reference.columns) and reference["actual"].notna().any():
        metrics = regression_quality(reference["actual"], reference["prediction"])
        return metrics.get("mae")
    return None


def build_report(
    predictions_uri_or_base: str,
    reference_predictions_uri: str | None,
    actuals_uri: str | None,
    config: dict[str, Any],
    *,
    report_date: str | None,
    baseline_mae: float | None,
) -> dict[str, Any]:
    thresholds = _thresholds(config)
    date_col = str(config.get("date_col", "date"))
    target_col = str(config.get("target_col", "actual"))

    try:
        current, resolved_predictions_uri = _read_prediction_logs(predictions_uri_or_base, report_date)
    except Exception as error:
        LOGGER.warning("Could not read prediction logs from %s: %s", predictions_uri_or_base, error)
        return make_report(
            status="warning",
            need_retrain=False,
            reasons=["prediction_logs:missing"],
            metrics={
                "predictions_uri": predictions_uri_or_base,
                "reference_predictions_uri": reference_predictions_uri,
                "actuals_uri": actuals_uri,
                "prediction_summary": prediction_summary(pd.Series(dtype=float)),
                "prediction_logs_error": str(error),
                "prediction_drift": {
                    "skipped": True,
                    "reason": "missing_prediction_logs",
                },
            },
            thresholds=thresholds,
        )

    actuals_error: str | None = None
    try:
        current = _attach_actuals(current, actuals_uri, date_col=date_col, target_col=target_col)
    except Exception as error:
        actuals_error = str(error)
        LOGGER.warning("Could not attach actual values from %s: %s", actuals_uri, error)
    current = _add_monitoring_date(current, date_col)

    if "prediction" not in current.columns:
        raise ValueError("Prediction logs must contain 'prediction' column.")

    reasons: list[str] = []
    status = "ok"
    need_retrain = False
    metrics: dict[str, Any] = {
        "predictions_uri": resolved_predictions_uri,
        "reference_predictions_uri": reference_predictions_uri,
        "actuals_uri": actuals_uri,
        "prediction_summary": prediction_summary(current["prediction"]),
    }
    if actuals_error is not None:
        status = "warning"
        reasons.append("quality:missing_actuals")
        metrics["actuals_error"] = actuals_error

    if metrics["prediction_summary"]["count"] < thresholds["min_rows_for_check"]:
        status = "warning"
        reasons.append("prediction_quality:row_count<min_rows")

    if metrics["prediction_summary"].get("outlier_share") is not None and (
        metrics["prediction_summary"]["outlier_share"] >= thresholds["outlier_share_warning"]
    ):
        status = "warning"
        reasons.append("prediction_drift:outlier_share>threshold")

    reference = None
    if reference_predictions_uri:
        try:
            reference = read_table(reference_predictions_uri)
        except Exception as error:
            LOGGER.warning(
                "Could not read reference predictions from %s: %s",
                reference_predictions_uri,
                error,
            )
            status = worst_status(status, "warning")
            reasons.append("prediction_drift:missing_reference_predictions")
            metrics["reference_predictions_error"] = str(error)

    if reference is not None:
        if "prediction" not in reference.columns:
            raise ValueError("Reference predictions must contain 'prediction' column.")
        metrics["reference_prediction_summary"] = prediction_summary(reference["prediction"])
        psi = population_stability_index(reference["prediction"], current["prediction"])
        metrics["prediction_drift"] = {"psi": psi}
        if psi is not None and psi >= thresholds["psi_threshold_critical"]:
            status = "critical"
            need_retrain = True
            reasons.append("prediction_drift:psi>critical")
        elif psi is not None and psi >= thresholds["psi_threshold_warning"] and status != "critical":
            status = "warning"
            reasons.append("prediction_drift:psi>threshold")

        anomaly = isolation_forest_anomaly_report(reference["prediction"], current["prediction"])
        metrics["anomaly"] = anomaly
        if anomaly.get("anomaly_count", 0) > 0:
            status = worst_status(status, "warning")
            reasons.append("anomaly:outlier_detected")

        combined_values = pd.concat([reference["prediction"], current["prediction"]], ignore_index=True)
        change_point = change_point_report(combined_values, current_window=len(current))
        metrics["change_point"] = change_point
        if change_point.get("change_point_detected") or change_point.get("red_breakpoint_detected"):
            status = worst_status(status, "warning")
            reasons.append("anomaly:change_point_detected")

        if anomaly.get("available") and not anomaly.get("skipped"):
            current_with_flags = current.copy()
            flags = anomaly.get("anomaly_flags", [])
            current_with_flags["_anomaly_flag"] = [False] * len(current_with_flags)
            current_with_flags.loc[current_with_flags.index[: len(flags)], "_anomaly_flag"] = flags
            consecutive_days = consecutive_anomaly_days(
                current_with_flags,
                date_col="_monitoring_date",
                anomaly_flag_col="_anomaly_flag",
            )
            metrics["anomaly"]["consecutive_days"] = consecutive_days
            if consecutive_days >= thresholds["anomaly_retrain_min_consecutive_days"]:
                need_retrain = True
                status = worst_status(status, "critical")
                reasons.append("anomaly:consecutive_days>=threshold")
    else:
        status = worst_status(status, "warning")
        reasons.append("prediction_drift:missing_reference_predictions")

    if "actual" in current.columns and current["actual"].notna().any():
        quality_metrics = regression_quality(current["actual"], current["prediction"])
        metrics["quality"] = quality_metrics
        baseline = _baseline_mae(reference, baseline_mae) if reference is not None else baseline_mae
        metrics["quality"]["baseline_mae"] = baseline
        if baseline and quality_metrics.get("mae") is not None:
            ratio = quality_metrics["mae"] / baseline
            metrics["quality"]["mae_degradation_ratio"] = ratio
            if ratio >= thresholds["mae_degradation_ratio"]:
                status = worst_status(status, "warning")
                need_retrain = True
                reasons.append("quality:mae_degradation")

    return make_report(
        status=status,
        need_retrain=need_retrain,
        reasons=reasons,
        metrics=metrics,
        thresholds=thresholds,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    report = build_report(
        args.predictions_uri,
        args.reference_predictions_uri,
        args.actuals_uri,
        config,
        report_date=args.report_date,
        baseline_mae=args.baseline_mae,
    )
    output_uri = args.output_uri or report_uri(args.reports_uri, "prediction_report", args.report_date)
    write_report(report, output_uri)

    LOGGER.info("Prediction monitoring report written to %s", output_uri)
    print(output_uri)
    print(f"status: {report['status']}")
    print(f"need_retrain: {report['need_retrain']}")


if __name__ == "__main__":
    main()
