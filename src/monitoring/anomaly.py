from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _numeric_frame(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"value": pd.to_numeric(series, errors="coerce")}).dropna()


def isolation_forest_anomaly_report(
    reference: pd.Series,
    current: pd.Series,
    *,
    contamination: float = 0.05,
) -> dict[str, Any]:
    try:
        from DS_research.anomaly_detector import AnomalyDetector
    except Exception as error:
        LOGGER.warning("Could not import DS_research AnomalyDetector: %s", error)
        return {"available": False, "error": str(error)}

    reference_frame = _numeric_frame(reference)
    current_frame = _numeric_frame(current)
    if len(reference_frame) < 10 or current_frame.empty:
        return {"available": True, "skipped": "not_enough_data"}

    detector = AnomalyDetector(contamination=contamination)
    detector.fit(reference_frame)
    mask = detector.detect(current_frame)
    return {
        "available": True,
        "method": "DS_research.AnomalyDetector(IsolationForest)",
        "anomaly_count": int(mask.sum()),
        "checked_count": int(len(mask)),
        "anomaly_share": float(mask.mean()) if len(mask) else 0.0,
        "anomaly_flags": [bool(value) for value in mask],
    }


def change_point_report(
    values: pd.Series,
    *,
    current_window: int,
    alpha: float = 0.05,
    beta: float = 0.05,
    threshold: float = 2.0,
) -> dict[str, Any]:
    try:
        from DS_research.breakpoints_detector import ChangeFinder
    except Exception as error:
        LOGGER.warning("Could not import DS_research ChangeFinder: %s", error)
        return {"available": False, "error": str(error)}

    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) < 10:
        return {"available": True, "skipped": "not_enough_data"}

    std = float(np.std(numeric)) or 1.0
    detector = ChangeFinder(
        alpha=alpha,
        beta=beta,
        method="sr",
        sigma_diff=1.0 / max(std, 1e-6),
        trsh=threshold,
    )
    for value in numeric:
        detector.feed(float(value))

    recent_states = detector.states[-current_window:] if current_window > 0 else detector.states
    recent_breakpoints = detector.breakpoints[-current_window:] if current_window > 0 else detector.breakpoints
    return {
        "available": True,
        "method": "DS_research.ChangeFinder(sr)",
        "change_point_detected": bool(any(state == 1 for state in recent_states)),
        "red_breakpoint_detected": bool(any(point == "red" for point in recent_breakpoints)),
        "recent_state_count": int(sum(recent_states)),
        "recent_window": int(len(recent_states)),
    }


def consecutive_anomaly_days(
    frame: pd.DataFrame,
    *,
    date_col: str,
    anomaly_flag_col: str,
) -> int:
    if date_col not in frame.columns or anomaly_flag_col not in frame.columns:
        return 0
    prepared = frame[[date_col, anomaly_flag_col]].copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce").dt.date
    prepared = prepared.dropna(subset=[date_col])
    if prepared.empty:
        return 0

    daily = prepared.groupby(date_col)[anomaly_flag_col].max().sort_index()
    consecutive = 0
    for has_anomaly in reversed(daily.tolist()):
        if bool(has_anomaly):
            consecutive += 1
        else:
            break
    return consecutive
