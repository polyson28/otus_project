from __future__ import annotations

from typing import Any

from src.monitoring.report import utc_now_iso


def _monitoring_thresholds(config: dict[str, Any]) -> dict[str, Any]:
    monitoring = config.get("monitoring", {})
    return {
        "min_rows_for_check": int(monitoring.get("min_rows_for_check", 30)),
        "psi_threshold_critical": float(monitoring.get("psi_threshold_critical", 0.25)),
        "mae_degradation_ratio": float(monitoring.get("mae_degradation_ratio", 1.2)),
        "anomaly_retrain_min_consecutive_days": int(
            monitoring.get("anomaly_retrain_min_consecutive_days", 2)
        ),
    }


def _reasons(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return []
    reasons = report.get("reasons", [])
    return [str(reason) for reason in reasons if reason not in (None, "")]


def _metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    metrics = report.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _quality_metric(metrics: dict[str, Any], name: str) -> Any:
    quality = metrics.get("quality", {})
    return quality.get(name) if isinstance(quality, dict) else None


def _report_used(name: str, report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    return {
        "name": name,
        "created_at": report.get("created_at"),
        "status": report.get("status"),
        "need_retrain": bool(report.get("need_retrain", False)),
    }


def _has_reason(reasons: list[str], *needles: str) -> bool:
    return any(any(needle in reason for needle in needles) for reason in reasons)


def _data_quality_is_critical(data_report: dict[str, Any] | None) -> bool:
    if not data_report:
        return False
    if data_report.get("status") != "critical":
        return False
    reasons = _reasons(data_report)
    return _has_reason(
        reasons,
        "data_quality:",
        "alert:data_quality_critical",
    )


def _min_rows_failed(
    data_report: dict[str, Any] | None,
    prediction_report: dict[str, Any] | None,
    *,
    min_rows: int,
) -> bool:
    data_quality = _metrics(data_report).get("quality", {})
    if isinstance(data_quality, dict) and data_quality.get("row_count") is not None:
        if int(data_quality["row_count"]) < min_rows:
            return True

    prediction_summary = _metrics(prediction_report).get("prediction_summary", {})
    if isinstance(prediction_summary, dict) and prediction_summary.get("count") is not None:
        if int(prediction_summary["count"]) < min_rows:
            return True

    return _has_reason(
        _reasons(data_report) + _reasons(prediction_report),
        "row_count<min_rows",
    )


def _quality_degraded(prediction_report: dict[str, Any] | None, *, threshold: float) -> bool:
    metrics = _metrics(prediction_report)
    ratio = _quality_metric(metrics, "mae_degradation_ratio")
    if ratio is not None:
        return float(ratio) >= threshold
    return _has_reason(_reasons(prediction_report), "quality:mae_degradation")


def _data_drift_critical(data_report: dict[str, Any] | None) -> bool:
    if data_report and data_report.get("need_retrain"):
        return True
    if data_report and data_report.get("status") == "critical":
        return _has_reason(_reasons(data_report), "data_drift:", "psi>critical")
    return False


def _prediction_drift_critical(prediction_report: dict[str, Any] | None) -> bool:
    if not prediction_report:
        return False
    if prediction_report.get("need_retrain") and _has_reason(
        _reasons(prediction_report),
        "prediction_drift:",
        "psi>critical",
    ):
        return True
    return _has_reason(_reasons(prediction_report), "prediction_drift:psi>critical")


def _sustained_anomaly(
    prediction_report: dict[str, Any] | None,
    *,
    min_consecutive_days: int,
) -> bool:
    anomaly = _metrics(prediction_report).get("anomaly", {})
    if isinstance(anomaly, dict):
        consecutive_days = anomaly.get("consecutive_days")
        if consecutive_days is not None and int(consecutive_days) >= min_consecutive_days:
            return True
    return _has_reason(_reasons(prediction_report), "anomaly:consecutive_days>=threshold")


def _single_anomaly_or_change_point(prediction_report: dict[str, Any] | None) -> bool:
    if not prediction_report:
        return False
    if _sustained_anomaly(prediction_report, min_consecutive_days=2):
        return False
    return _has_reason(
        _reasons(prediction_report),
        "anomaly:outlier_detected",
        "anomaly:change_point_detected",
    )


def decide_retrain(
    data_report: dict | None,
    prediction_report: dict | None,
    config: dict,
) -> dict:
    thresholds = _monitoring_thresholds(config)
    reasons: list[str] = []
    reports_used = [
        report
        for report in (
            _report_used("data_report", data_report),
            _report_used("prediction_report", prediction_report),
        )
        if report is not None
    ]

    if _data_quality_is_critical(data_report):
        reasons.append("blocked:data_quality_critical")
        reasons.append("blocked:retrain_on_bad_data_forbidden")
        return {
            "need_retrain": False,
            "decision": "blocked",
            "reasons": sorted(set(reasons + _reasons(data_report))),
            "created_at": utc_now_iso(),
            "reports_used": reports_used,
        }

    if _min_rows_failed(
        data_report,
        prediction_report,
        min_rows=thresholds["min_rows_for_check"],
    ):
        reasons.append("skip:not_enough_rows")
        return {
            "need_retrain": False,
            "decision": "skip",
            "reasons": sorted(set(reasons + _reasons(data_report) + _reasons(prediction_report))),
            "created_at": utc_now_iso(),
            "reports_used": reports_used,
        }

    if _quality_degraded(
        prediction_report,
        threshold=thresholds["mae_degradation_ratio"],
    ):
        reasons.append("quality:mae_degradation")

    if _data_drift_critical(data_report):
        reasons.append("data_drift:critical")

    if _prediction_drift_critical(prediction_report):
        reasons.append("prediction_drift:critical")

    if _sustained_anomaly(
        prediction_report,
        min_consecutive_days=thresholds["anomaly_retrain_min_consecutive_days"],
    ):
        reasons.append("anomaly:consecutive_days>=threshold")

    if reasons:
        return {
            "need_retrain": True,
            "decision": "retrain",
            "reasons": sorted(set(reasons + _reasons(data_report) + _reasons(prediction_report))),
            "created_at": utc_now_iso(),
            "reports_used": reports_used,
        }

    if _single_anomaly_or_change_point(prediction_report):
        reasons.append("skip:single_anomaly_or_change_point")

    return {
        "need_retrain": False,
        "decision": "skip",
        "reasons": sorted(set(reasons + _reasons(data_report) + _reasons(prediction_report))),
        "created_at": utc_now_iso(),
        "reports_used": reports_used,
    }
