from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.io.s3 import is_s3_uri, read_csv, read_excel, read_json, read_parquet, write_json


STATUS_ORDER = {"ok": 0, "warning": 1, "critical": 2}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def worst_status(*statuses: str) -> str:
    return max(statuses or ("ok",), key=lambda status: STATUS_ORDER.get(status, 0))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def make_report(
    *,
    status: str,
    need_retrain: bool,
    reasons: list[str],
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "status": status,
        "need_retrain": bool(need_retrain),
        "reasons": sorted(set(reasons)),
        "metrics": json_safe(metrics),
        "thresholds": json_safe(thresholds),
    }


def report_uri(base_uri: str, prefix: str, report_date: str | None = None) -> str:
    day = report_date or datetime.now(timezone.utc).date().isoformat()
    filename = f"{prefix}_{day}.json"
    if is_s3_uri(base_uri):
        return f"{base_uri.rstrip('/')}/{filename}"
    return str(Path(base_uri) / filename)


def write_report(report: dict[str, Any], uri: str) -> None:
    write_json(json_safe(report), uri)


def read_table(uri: str) -> pd.DataFrame:
    lower_uri = str(uri).lower()
    if lower_uri.endswith(".parquet"):
        return read_parquet(uri)
    if lower_uri.endswith(".csv"):
        return read_csv(uri)
    if lower_uri.endswith((".xlsx", ".xls")):
        return read_excel(uri)
    if lower_uri.endswith(".json"):
        data = read_json(uri)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        raise ValueError(f"JSON table must be a list or object: {uri}")
    raise ValueError(f"Unsupported table format: {uri}")
