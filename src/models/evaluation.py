from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def liquidity_decision(
    trues: np.ndarray | pd.Series | list[float],
    preds: np.ndarray | pd.Series | list[float],
    key_rate: float = 0.21,
    return_all_values: bool = False,
) -> np.ndarray | float:
    """Business profit metric adapted from DS_research/evaluate_profit.py."""
    day_deposit_rate = key_rate + 0.005
    night_deposit_rate = key_rate - 0.009
    night_loan_rate = key_rate + 0.01

    true_values = np.array(trues, copy=True, dtype=float)
    pred_values = np.array(preds, copy=True, dtype=float)
    profit = np.zeros_like(true_values)

    positive_pred = pred_values > 0
    profit[positive_pred] += day_deposit_rate * pred_values[positive_pred]
    true_values[positive_pred] -= pred_values[positive_pred]
    true_values[~positive_pred] -= pred_values[~positive_pred]

    positive_balance = true_values > 0
    profit[positive_balance] += night_deposit_rate * true_values[positive_balance]
    profit[~positive_balance] += night_loan_rate * true_values[~positive_balance]

    if return_all_values:
        return profit
    return float(profit.sum())


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    key_rate: float = 0.21,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    non_zero_mask = y_true_array != 0

    metrics = {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "profit": float(liquidity_decision(y_true_array, y_pred_array, key_rate=key_rate)),
    }
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


def build_evaluation_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    key_rate: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "metrics": regression_metrics(y_true=y_true, y_pred=y_pred, key_rate=key_rate),
        "n_rows": int(len(y_true)),
        "key_rate": float(key_rate),
    }
    if extra:
        report.update(extra)
    return report
