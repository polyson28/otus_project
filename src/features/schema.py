from __future__ import annotations

import pandas as pd


def validate_input_dataframe(
    df: pd.DataFrame,
    date_col: str,
    target_col: str | None = None,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")
    if target_col is not None and target_col not in df.columns:
        raise ValueError(f"Missing required target column: {target_col}")


def validate_feature_frame(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("Feature DataFrame is empty.")
    duplicate_columns = X.columns[X.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Feature DataFrame contains duplicate columns: {duplicate_columns}")
