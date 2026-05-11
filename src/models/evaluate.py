from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    metrics = {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
    }
    non_zero_mask = y_true_array != 0
    if non_zero_mask.any():
        metrics["mape"] = float(
            np.mean(
                np.abs(
                    (y_true_array[non_zero_mask] - y_pred_array[non_zero_mask])
                    / y_true_array[non_zero_mask]
                )
            )
        )
    if len(y_true_array) >= 2:
        metrics["r2"] = float(r2_score(y_true_array, y_pred_array))
    return metrics


def prefixed_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def date_range_payload(dates: pd.Series | None) -> dict[str, Any]:
    if dates is None or dates.empty:
        return {"start": None, "end": None, "rows": 0}
    parsed = pd.to_datetime(dates, errors="coerce").dropna()
    if parsed.empty:
        return {"start": None, "end": None, "rows": int(len(dates))}
    return {
        "start": parsed.min().isoformat(),
        "end": parsed.max().isoformat(),
        "rows": int(len(dates)),
    }


def backtest_time_windows(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series | None = None,
    *,
    n_windows: int = 3,
    min_window_size: int = 10,
) -> list[dict[str, Any]]:
    if n_windows <= 0 or len(X) < min_window_size:
        return []

    window_size = max(min_window_size, len(X) // n_windows)
    reports: list[dict[str, Any]] = []
    for idx in range(n_windows):
        start = idx * window_size
        end = len(X) if idx == n_windows - 1 else min(len(X), (idx + 1) * window_size)
        if end - start < min_window_size:
            continue
        X_window = X.iloc[start:end]
        y_window = y.iloc[start:end]
        predictions = model.predict(X_window)
        window_dates = dates.iloc[start:end] if dates is not None else None
        reports.append(
            {
                "window": idx + 1,
                "metrics": regression_metrics(y_window, predictions),
                "date_range": date_range_payload(window_dates),
            }
        )
    return reports
