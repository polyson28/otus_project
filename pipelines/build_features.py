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

from src.config import load_config
from src.features.pipeline import TimeSeriesFeaturePipeline
from src.io.s3 import read_csv, read_excel, read_json, read_parquet, write_json, write_parquet


LOGGER = logging.getLogger("build_features")


def read_dataframe(uri: str) -> pd.DataFrame:
    suffix = Path(uri.split("?", 1)[0]).suffix.lower()
    LOGGER.info("Reading dataframe from %s", uri)

    if suffix in {".xlsx", ".xls"}:
        return read_excel(uri)
    if suffix == ".csv":
        return read_csv(uri)
    if suffix == ".parquet":
        return read_parquet(uri)

    raise ValueError(f"Unsupported dataframe format: {suffix}. Use .xlsx, .csv, or .parquet.")


def read_tax_dates(uri: str | None) -> list[Any] | None:
    if uri is None:
        return None

    suffix = Path(uri.split("?", 1)[0]).suffix.lower()
    LOGGER.info("Reading tax calendar from %s", uri)

    if suffix == ".json":
        payload = read_json(uri)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("tax_dates", "dates", "date"):
                if key in payload:
                    value = payload[key]
                    return value if isinstance(value, list) else [value]
        raise ValueError("Tax calendar JSON must be a list or contain tax_dates/dates/date.")

    calendar_df = read_dataframe(uri)
    if "date" in calendar_df.columns:
        return calendar_df["date"].dropna().tolist()
    if len(calendar_df.columns) == 1:
        return calendar_df.iloc[:, 0].dropna().tolist()

    raise ValueError("Tax calendar table must contain a 'date' column or exactly one column.")


def build_pipeline_from_config(config: dict[str, Any]) -> TimeSeriesFeaturePipeline:
    feature_config = config["feature_engineering"]
    return TimeSeriesFeaturePipeline(
        date_col=config["date_col"],
        target_col=config["target_col"],
        lags=feature_config["lags"],
        rolling_windows=feature_config["rolling_windows"],
        use_calendar_features=feature_config["use_calendar_features"],
        use_tax_features=feature_config["use_tax_features"],
        use_macro_features=feature_config["use_macro_features"],
    )


def build_schema(
    pipeline: TimeSeriesFeaturePipeline,
    source_data_uri: str,
) -> dict[str, Any]:
    return {
        "feature_columns": pipeline.feature_columns,
        "target_col": pipeline.target_col,
        "date_col": pipeline.date_col,
        "lags": pipeline.lags,
        "rolling_windows": pipeline.rolling_windows,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data_uri": source_data_uri,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build time series features from cleaned parquet data.")
    parser.add_argument("--input-uri", required=True, help="Input cleaned parquet URI/path.")
    parser.add_argument("--output-uri", required=True, help="Output features parquet URI/path.")
    parser.add_argument("--feature-pipeline-uri", required=True, help="Feature pipeline joblib URI/path.")
    parser.add_argument("--feature-schema-uri", required=True, help="Feature schema JSON URI/path.")
    parser.add_argument("--macro-uri", default=None, help="Optional macro dataframe URI/path.")
    parser.add_argument("--tax-calendar-uri", default=None, help="Optional tax calendar URI/path.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--mode", choices=["train", "inference"], required=True, help="Feature build mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        config = load_config(args.config)
        df = read_dataframe(args.input_uri)
        macro_df = read_dataframe(args.macro_uri) if args.macro_uri else None
        tax_dates = read_tax_dates(args.tax_calendar_uri)

        if args.mode == "train":
            LOGGER.info("Building features in train mode with fit_transform.")
            pipeline = build_pipeline_from_config(config)
            features = pipeline.fit_transform(df, macro_df=macro_df, tax_dates=tax_dates)
        else:
            LOGGER.info("Building features in inference mode with saved pipeline.")
            pipeline = TimeSeriesFeaturePipeline.load(args.feature_pipeline_uri)
            features = pipeline.transform(df, macro_df=macro_df, tax_dates=tax_dates)

        LOGGER.info("Writing features with shape %s to %s", features.shape, args.output_uri)
        write_parquet(features, args.output_uri)

        LOGGER.info("Writing feature pipeline to %s", args.feature_pipeline_uri)
        pipeline.save(args.feature_pipeline_uri)

        schema = build_schema(pipeline, source_data_uri=args.input_uri)
        LOGGER.info("Writing feature schema to %s", args.feature_schema_uri)
        write_json(schema, args.feature_schema_uri)

        print(f"OK: built features in {args.mode} mode")
        print(f"Features shape: {features.shape}")
        print(f"Features: {args.output_uri}")
        print(f"Feature pipeline: {args.feature_pipeline_uri}")
        print(f"Feature schema: {args.feature_schema_uri}")
    except Exception as error:
        LOGGER.exception("Build features failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
