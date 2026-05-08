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
from src.features.selection import TimeSeriesFeatureSelector
from src.io.s3 import read_csv, read_excel, read_json, read_parquet, write_json, write_parquet


LOGGER = logging.getLogger("select_features")


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


def build_selector_from_config(config: dict[str, Any]) -> TimeSeriesFeatureSelector:
    selection_config = config.get("feature_selection", {})
    methods = selection_config.get("methods", {})
    target_metrics = selection_config.get("target_aware_metrics", {})
    wrapper_config = selection_config.get("wrapper", {})
    return TimeSeriesFeatureSelector(
        target_col=config["target_col"],
        date_col=config["date_col"],
        missing_threshold=float(selection_config.get("missing_threshold", 0.4)),
        correlation_threshold=float(selection_config.get("correlation_threshold", 0.95)),
        remove_constant=bool(methods.get("remove_constant", True)),
        remove_high_missing=bool(methods.get("remove_high_missing", True)),
        remove_high_correlation=bool(methods.get("remove_high_correlation", True)),
        use_mutual_info=bool(target_metrics.get("use_mutual_info", True)),
        use_correlation_with_target=bool(target_metrics.get("use_correlation_with_target", True)),
        max_features=selection_config.get("max_features"),
        always_keep=selection_config.get("always_keep", []),
        min_features_to_select=wrapper_config.get("min_features_to_select"),
        validation_size=float(selection_config.get("validation_size", 0.2)),
        random_state=int(selection_config.get("random_state", 42)),
    )


def count_candidate_features(df: pd.DataFrame, target_col: str, date_col: str) -> int:
    return len([
        col for col in df.select_dtypes(include="number").columns
        if col not in {target_col, date_col}
    ])


def companion_feature_schema_uri(input_uri: str) -> str:
    if "/" not in input_uri:
        return "feature_schema.json"
    return input_uri.rsplit("/", 1)[0] + "/feature_schema.json"


def load_target_frame_from_schema(
    input_uri: str,
    target_col: str,
    date_col: str,
    n_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema_uri = companion_feature_schema_uri(input_uri)
    LOGGER.info("Input is missing date/target columns. Trying companion feature schema: %s", schema_uri)
    schema = read_json(schema_uri)
    source_data_uri = schema.get("source_data_uri")
    if not source_data_uri:
        raise ValueError(
            f"Feature schema {schema_uri} does not contain source_data_uri. "
            "Rebuild features with feature_schema.json or include date/target in features.parquet."
        )

    source_df = read_dataframe(source_data_uri)
    if date_col not in source_df.columns:
        raise ValueError(f"Source dataframe {source_data_uri} is missing date column '{date_col}'.")
    if target_col not in source_df.columns:
        raise ValueError(f"Source dataframe {source_data_uri} is missing target column '{target_col}'.")

    source_df = source_df[[date_col, target_col]].copy()
    source_df[date_col] = pd.to_datetime(source_df[date_col], errors="coerce")
    if source_df[date_col].isna().any():
        raise ValueError(f"Source dataframe {source_data_uri} has invalid datetimes in '{date_col}'.")

    source_df = source_df.sort_values(date_col).reset_index(drop=True)
    if len(source_df) < n_rows:
        raise ValueError(f"Source dataframe has fewer rows than features: {len(source_df)} < {n_rows}")

    aligned = source_df.tail(n_rows).reset_index(drop=True)
    return aligned, {
        "feature_schema_uri": schema_uri,
        "source_data_uri": source_data_uri,
        "source_rows": len(source_df),
        "feature_rows": n_rows,
        "alignment_strategy": "tail_trim_after_feature_lags",
    }


def ensure_train_columns(
    df: pd.DataFrame,
    input_uri: str,
    target_col: str,
    date_col: str,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if date_col in df.columns and target_col in df.columns:
        return df, None

    missing = [col for col in [date_col, target_col] if col not in df.columns]
    LOGGER.warning("Input features are missing train columns: %s", missing)
    target_frame, alignment = load_target_frame_from_schema(
        input_uri=input_uri,
        target_col=target_col,
        date_col=date_col,
        n_rows=len(df),
    )
    enriched = df.reset_index(drop=True).copy()
    if date_col not in enriched.columns:
        enriched[date_col] = target_frame[date_col].values
    if target_col not in enriched.columns:
        enriched[target_col] = target_frame[target_col].values
    return enriched, alignment


def build_selected_features_payload(
    selector: TimeSeriesFeatureSelector,
    target_col: str,
    date_col: str,
    n_features_before: int,
    input_uri: str,
    output_uri: str,
    created_at: str,
) -> dict[str, Any]:
    selected_features = selector.get_selected_features()
    return {
        "selected_features": selected_features,
        "target_col": target_col,
        "date_col": date_col,
        "n_features_before": n_features_before,
        "n_features_after": len(selected_features),
        "created_at": created_at,
        "input_uri": input_uri,
        "output_uri": output_uri,
    }


def build_selection_report(
    selector: TimeSeriesFeatureSelector,
    n_rows: int,
    n_features_before: int,
    created_at: str,
    target_alignment: dict[str, Any] | None,
) -> dict[str, Any]:
    report = selector.get_report()
    dropped = report.get("dropped_features", {})
    return {
        "dropped_constant_features": dropped.get("constant", []),
        "dropped_high_missing_features": dropped.get("high_missing", []),
        "dropped_high_correlation_features": dropped.get("high_correlation", []),
        "feature_scores": report.get("feature_scores", {}),
        "selected_features": selector.get_selected_features(),
        "n_rows": n_rows,
        "n_features_before": n_features_before,
        "n_features_after": len(selector.get_selected_features()),
        "target_alignment": target_alignment,
        "created_at": created_at,
    }


def print_summary(
    selector: TimeSeriesFeatureSelector,
    n_features_before: int,
    output_uri: str,
    selector_uri: str,
    selected_features_uri: str,
    selection_report_uri: str,
) -> None:
    report = selector.get_report()
    dropped = report.get("dropped_features", {})
    print(f"OK: selected {len(selector.get_selected_features())} / {n_features_before} features")
    print(f"Dropped constant: {dropped.get('constant', [])}")
    print(f"Dropped high missing: {dropped.get('high_missing', [])}")
    print(f"Dropped high correlation: {dropped.get('high_correlation', [])}")
    print(f"Selected features parquet: {output_uri}")
    print(f"Feature selector: {selector_uri}")
    print(f"Selected features JSON: {selected_features_uri}")
    print(f"Selection report: {selection_report_uri}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select production model features from features.parquet.")
    parser.add_argument("--input-uri", required=True, help="Input features parquet URI/path.")
    parser.add_argument("--output-uri", required=True, help="Output selected features parquet URI/path.")
    parser.add_argument("--selector-uri", required=True, help="Feature selector joblib URI/path.")
    parser.add_argument("--selected-features-uri", required=True, help="Selected features JSON URI/path.")
    parser.add_argument("--selection-report-uri", required=True, help="Selection report JSON URI/path.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--mode", choices=["train", "inference"], required=True, help="Selection mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        config = load_config(args.config)
        target_col = config["target_col"]
        date_col = config["date_col"]

        df = read_dataframe(args.input_uri)
        n_features_before = count_candidate_features(df, target_col=target_col, date_col=date_col)
        LOGGER.info("Loaded feature frame with shape %s and %s candidate features.", df.shape, n_features_before)

        if args.mode == "train":
            df, target_alignment = ensure_train_columns(
                df=df,
                input_uri=args.input_uri,
                target_col=target_col,
                date_col=date_col,
            )
            selector = build_selector_from_config(config)
            selected_df = selector.fit_transform(df)
            LOGGER.info("Writing feature selector to %s", args.selector_uri)
            selector.save(args.selector_uri)
        else:
            target_alignment = None
            selector = TimeSeriesFeatureSelector.load(args.selector_uri)
            selected_df = selector.transform(df)

        LOGGER.info("Writing selected features with shape %s to %s", selected_df.shape, args.output_uri)
        write_parquet(selected_df, args.output_uri)

        created_at = datetime.now(timezone.utc).isoformat()
        selected_features_payload = build_selected_features_payload(
            selector=selector,
            target_col=target_col,
            date_col=date_col,
            n_features_before=n_features_before,
            input_uri=args.input_uri,
            output_uri=args.output_uri,
            created_at=created_at,
        )
        LOGGER.info("Writing selected features JSON to %s", args.selected_features_uri)
        write_json(selected_features_payload, args.selected_features_uri)

        selection_report = build_selection_report(
            selector=selector,
            n_rows=len(df),
            n_features_before=n_features_before,
            created_at=created_at,
            target_alignment=target_alignment,
        )
        LOGGER.info("Writing selection report to %s", args.selection_report_uri)
        write_json(selection_report, args.selection_report_uri)

        print_summary(
            selector=selector,
            n_features_before=n_features_before,
            output_uri=args.output_uri,
            selector_uri=args.selector_uri,
            selected_features_uri=args.selected_features_uri,
            selection_report_uri=args.selection_report_uri,
        )
    except Exception as error:
        LOGGER.exception("Feature selection failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
