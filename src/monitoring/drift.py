from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


EPS = 1e-6


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
) -> float | None:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) < 2 or len(cur) < 2:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        minimum = min(ref.min(), cur.min())
        maximum = max(ref.max(), cur.max())
        if minimum == maximum:
            return 0.0
        edges = np.linspace(minimum, maximum, bins + 1)

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_share = ref_counts / max(ref_counts.sum(), 1)
    cur_share = cur_counts / max(cur_counts.sum(), 1)
    return float(np.sum((cur_share - ref_share) * np.log((cur_share + EPS) / (ref_share + EPS))))


def numeric_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    psi_warning: float,
    psi_critical: float,
    exclude_columns: set[str] | None = None,
) -> tuple[dict[str, Any], list[str], str]:
    metrics: dict[str, Any] = {}
    reasons: list[str] = []
    status = "ok"

    excluded = exclude_columns or set()
    numeric_columns = sorted(
        set(reference.select_dtypes(include=[np.number]).columns)
        & set(current.select_dtypes(include=[np.number]).columns)
        - excluded
    )
    for column in numeric_columns:
        ref = reference[column].dropna()
        cur = current[column].dropna()
        psi = population_stability_index(ref, cur)
        ks_statistic = None
        ks_pvalue = None
        if len(ref) >= 2 and len(cur) >= 2:
            ks = ks_2samp(ref, cur)
            ks_statistic = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)

        column_metrics = {
            "psi": psi,
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
            "reference_mean": float(ref.mean()) if len(ref) else None,
            "current_mean": float(cur.mean()) if len(cur) else None,
            "reference_std": float(ref.std()) if len(ref) else None,
            "current_std": float(cur.std()) if len(cur) else None,
        }
        metrics[column] = column_metrics

        if psi is not None and psi >= psi_critical:
            status = "critical"
            reasons.append(f"data_drift:{column}:psi>critical")
        elif psi is not None and psi >= psi_warning and status != "critical":
            status = "warning"
            reasons.append(f"data_drift:{column}:psi>threshold")

    return metrics, reasons, status


def categorical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    max_share_delta_warning: float = 0.2,
    exclude_columns: set[str] | None = None,
) -> tuple[dict[str, Any], list[str], str]:
    metrics: dict[str, Any] = {}
    reasons: list[str] = []
    status = "ok"
    excluded = exclude_columns or set()
    categorical_columns = sorted(
        set(reference.select_dtypes(exclude=[np.number]).columns)
        & set(current.select_dtypes(exclude=[np.number]).columns)
        - excluded
    )

    for column in categorical_columns:
        ref_share = reference[column].astype("string").value_counts(normalize=True, dropna=False)
        cur_share = current[column].astype("string").value_counts(normalize=True, dropna=False)
        values = sorted(set(ref_share.index.astype(str)) | set(cur_share.index.astype(str)))
        deltas = {
            value: float(abs(cur_share.get(value, 0.0) - ref_share.get(value, 0.0)))
            for value in values
        }
        max_delta = max(deltas.values(), default=0.0)
        metrics[column] = {
            "max_share_delta": max_delta,
            "value_share_delta": deltas,
        }
        if max_delta >= max_share_delta_warning:
            status = "warning"
            reasons.append(f"data_drift:{column}:category_share_delta")

    return metrics, reasons, status


def data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    psi_warning: float,
    psi_critical: float,
    exclude_columns: set[str] | None = None,
) -> tuple[dict[str, Any], list[str], str]:
    numeric_metrics, numeric_reasons, numeric_status = numeric_drift(
        reference,
        current,
        psi_warning=psi_warning,
        psi_critical=psi_critical,
        exclude_columns=exclude_columns,
    )
    categorical_metrics, categorical_reasons, categorical_status = categorical_drift(
        reference,
        current,
        exclude_columns=exclude_columns,
    )
    status = "critical" if "critical" in {numeric_status, categorical_status} else (
        "warning" if "warning" in {numeric_status, categorical_status} else "ok"
    )
    return (
        {"numeric": numeric_metrics, "categorical": categorical_metrics},
        numeric_reasons + categorical_reasons,
        status,
    )
