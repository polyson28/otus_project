from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_env_or_config, load_config
from src.io.s3 import read_csv, read_excel, read_parquet, write_json, write_parquet


LOGGER = logging.getLogger("preprocess")


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [normalize_column_name(col) for col in normalized.columns]
    duplicate_columns = normalized.columns[normalized.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate columns after normalization: {duplicate_columns}")
    return normalized


def read_dataframe(uri: str) -> pd.DataFrame:
    suffix = Path(uri.split("?", 1)[0]).suffix.lower()
    LOGGER.info("Reading raw data from %s", uri)

    if suffix in {".xlsx", ".xls"}:
        return read_excel(uri)
    if suffix == ".csv":
        return read_csv(uri)
    if suffix == ".parquet":
        return read_parquet(uri)

    raise ValueError(f"Unsupported input format: {suffix}. Use .xlsx, .csv, or .parquet.")


def quality_report_uri(output_uri: str) -> str:
    if output_uri.startswith("s3://"):
        return output_uri.rsplit("/", 1)[0] + "/quality_report.json"
    return str(Path(output_uri).parent / "quality_report.json")


def coerce_numeric_columns(df: pd.DataFrame, exclude_columns: set[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if col in exclude_columns or pd.api.types.is_numeric_dtype(cleaned[col]):
            continue

        converted = pd.to_numeric(cleaned[col], errors="coerce")
        original_non_null = cleaned[col].notna()
        introduced_missing = converted[original_non_null].isna().sum()
        if introduced_missing == 0 and converted.notna().any():
            cleaned[col] = converted
            LOGGER.info("Converted column to numeric: %s", col)

    return cleaned


def preprocess_dataframe(
    df: pd.DataFrame,
    date_col: str,
    target_col: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows_before = len(df)
    df = normalize_columns(df)

    normalized_date_col = normalize_column_name(date_col)
    normalized_target_col = normalize_column_name(target_col) if target_col else None

    if normalized_date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found after normalization. "
            f"Available columns: {list(df.columns)}"
        )
    if normalized_target_col and normalized_target_col not in df.columns:
        LOGGER.warning(
            "Target column '%s' not found after normalization. Continuing preprocess without target validation.",
            target_col,
        )

    df[normalized_date_col] = pd.to_datetime(df[normalized_date_col], errors="coerce")
    invalid_dates = int(df[normalized_date_col].isna().sum())
    if invalid_dates:
        raise ValueError(f"Date column '{normalized_date_col}' contains {invalid_dates} invalid datetimes.")

    duplicated_dates = int(df.duplicated(subset=[normalized_date_col]).sum())
    if duplicated_dates:
        LOGGER.warning("Found %s duplicated dates. Keeping the last row for each date.", duplicated_dates)

    df = df.sort_values(normalized_date_col)
    df = df.drop_duplicates(subset=[normalized_date_col], keep="last").reset_index(drop=True)
    df = coerce_numeric_columns(df, exclude_columns={normalized_date_col})

    missing_values = {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}
    if missing_values:
        LOGGER.warning("Missing values found after cleaning: %s", missing_values)
    else:
        LOGGER.info("No missing values found after cleaning.")

    report = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns": list(df.columns),
        "missing_values": missing_values,
        "duplicated_dates": duplicated_dates,
        "min_date": df[normalized_date_col].min().isoformat() if not df.empty else None,
        "max_date": df[normalized_date_col].max().isoformat() if not df.empty else None,
    }
    return df, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw time series data into cleaned parquet.")
    parser.add_argument("--input-uri", required=True, help="Input .xlsx, .csv, or .parquet URI/path.")
    parser.add_argument("--output-uri", required=True, help="Output parquet URI/path.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--date-col", default=None, help="Date column name. Overrides config.")
    parser.add_argument("--target-col", default=None, help="Target column name. Overrides config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        config = load_config(args.config)
        date_col = args.date_col or get_env_or_config("DATE_COL", config.get("date_col"))
        target_col = args.target_col or get_env_or_config("TARGET_COL", config.get("target_col"))

        if not date_col:
            raise ValueError("Date column is required. Pass --date-col or set date_col in config.")

        df = read_dataframe(args.input_uri)
        LOGGER.info("Loaded dataframe with shape %s", df.shape)

        cleaned_df, report = preprocess_dataframe(df, date_col=date_col, target_col=target_col)
        report.update(
            {
                "input_uri": args.input_uri,
                "output_uri": args.output_uri,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        LOGGER.info("Writing cleaned parquet to %s", args.output_uri)
        write_parquet(cleaned_df, args.output_uri)

        report_uri = quality_report_uri(args.output_uri)
        LOGGER.info("Writing quality report to %s", report_uri)
        write_json(report, report_uri)

        print(f"OK: preprocessed {args.input_uri}")
        print(f"Rows: {report['rows_before']} -> {report['rows_after']}")
        print(f"Output: {args.output_uri}")
        print(f"Quality report: {report_uri}")
    except Exception as error:
        LOGGER.exception("Preprocess failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
