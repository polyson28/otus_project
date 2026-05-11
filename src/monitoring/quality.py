from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def data_quality_report(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    date_col: str,
    min_rows: int,
) -> tuple[dict[str, Any], list[str], str]:
    reasons: list[str] = []
    status = "ok"
    metrics: dict[str, Any] = {
        "row_count": int(len(current)),
        "reference_row_count": int(len(reference)),
        "missing_values": {column: int(value) for column, value in current.isna().sum().items()},
        "missing_share": {column: float(value) for column, value in current.isna().mean().items()},
    }

    if len(current) < min_rows:
        status = "critical"
        reasons.append("data_quality:row_count<min_rows")

    if date_col in current.columns:
        dates = pd.to_datetime(current[date_col], errors="coerce")
        invalid_dates = int(dates.isna().sum())
        duplicate_dates = int(dates.duplicated(keep=False).sum())
        metrics.update(
            {
                "invalid_dates": invalid_dates,
                "duplicate_dates": duplicate_dates,
                "min_date": dates.min().isoformat() if dates.notna().any() else None,
                "max_date": dates.max().isoformat() if dates.notna().any() else None,
            }
        )
        if invalid_dates:
            status = "critical"
            reasons.append("data_quality:invalid_dates")
        if duplicate_dates:
            status = "critical"
            reasons.append("data_quality:duplicate_dates")
    else:
        status = "critical"
        metrics["date_column_present"] = False
        reasons.append("data_quality:missing_date_column")

    missing_columns = [
        column for column, share in metrics["missing_share"].items()
        if share > 0
    ]
    if missing_columns and status != "critical":
        status = "warning"
        reasons.append("data_quality:missing_values")

    metrics["numeric_distribution_shift"] = {}
    numeric_columns = sorted(
        set(reference.select_dtypes(include=[np.number]).columns)
        & set(current.select_dtypes(include=[np.number]).columns)
    )
    for column in numeric_columns:
        ref = pd.to_numeric(reference[column], errors="coerce").dropna()
        cur = pd.to_numeric(current[column], errors="coerce").dropna()
        if ref.empty or cur.empty:
            continue
        ref_std = float(ref.std()) or 0.0
        mean_shift_std = abs(float(cur.mean()) - float(ref.mean())) / max(ref_std, 1e-6)
        std_ratio = float(cur.std()) / max(ref_std, 1e-6)
        metrics["numeric_distribution_shift"][column] = {
            "mean_shift_in_reference_std": float(mean_shift_std),
            "std_ratio": float(std_ratio),
        }
        if mean_shift_std >= 3.0 and status == "ok":
            status = "warning"
            reasons.append(f"data_quality:{column}:mean_shift")

    return metrics, reasons, status


def prediction_summary(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "outlier_share": None,
        }

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        outlier_share = 0.0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_share = float(((numeric < lower) | (numeric > upper)).mean())

    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std()) if len(numeric) > 1 else 0.0,
        "outlier_share": outlier_share,
    }


def regression_quality(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    true = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    valid = true.notna() & pred.notna()
    true_values = true[valid].to_numpy(dtype=float)
    pred_values = pred[valid].to_numpy(dtype=float)
    if len(true_values) == 0:
        return {}

    errors = true_values - pred_values
    metrics = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }
    non_zero = true_values != 0
    if non_zero.any():
        metrics["mape"] = float(np.mean(np.abs(errors[non_zero] / true_values[non_zero])))
    return metrics
