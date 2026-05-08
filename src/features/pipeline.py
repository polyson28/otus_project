from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.schema import validate_feature_frame, validate_input_dataframe
from src.io.s3 import joblib_dump_to_uri, joblib_load_from_uri


class TimeSeriesFeaturePipeline:
    def __init__(
        self,
        date_col: str,
        target_col: str,
        lags: list[int],
        rolling_windows: list[int],
        use_calendar_features: bool = True,
        use_tax_features: bool = True,
        use_macro_features: bool = True,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.use_calendar_features = use_calendar_features
        self.use_tax_features = use_tax_features
        self.use_macro_features = use_macro_features

        self.feature_columns: list[str] = []
        self.macro_columns_: list[str] = []
        self.macro_numeric_columns_: list[str] = []
        self.macro_scaler_: StandardScaler | None = None

    def fit(
        self,
        df: pd.DataFrame,
        macro_df: pd.DataFrame | None = None,
        tax_dates: list[Any] | None = None,
    ) -> "TimeSeriesFeaturePipeline":
        validate_input_dataframe(df, self.date_col, self.target_col)
        prepared_df = self._prepare_base_frame(df, require_target=True)

        if self.use_macro_features:
            self._fit_macro(prepared_df, macro_df)

        X = self._build_features(prepared_df, macro_df=macro_df, tax_dates=tax_dates)
        self.feature_columns = list(X.columns)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        macro_df: pd.DataFrame | None = None,
        tax_dates: list[Any] | None = None,
    ) -> pd.DataFrame:
        validate_input_dataframe(df, self.date_col)
        prepared_df = self._prepare_base_frame(df, require_target=False)
        if self.target_col not in prepared_df.columns:
            prepared_df[self.target_col] = pd.NA
            warnings.warn(
                f"Missing target column {self.target_col}. Lag and rolling features will be NaN.",
                stacklevel=2,
            )
        X = self._build_features(prepared_df, macro_df=macro_df, tax_dates=tax_dates)

        if self.feature_columns:
            X = self._align_to_fitted_columns(X)

        validate_feature_frame(X)
        return X

    def fit_transform(
        self,
        df: pd.DataFrame,
        macro_df: pd.DataFrame | None = None,
        tax_dates: list[Any] | None = None,
    ) -> pd.DataFrame:
        self.fit(df, macro_df=macro_df, tax_dates=tax_dates)
        X = self.transform(df, macro_df=macro_df, tax_dates=tax_dates)
        return X.dropna(subset=self.feature_columns)

    def save(self, uri: str) -> None:
        joblib_dump_to_uri(self, uri)

    @classmethod
    def load(cls, uri: str) -> "TimeSeriesFeaturePipeline":
        pipeline = joblib_load_from_uri(uri)
        if not isinstance(pipeline, cls):
            raise TypeError(f"Object loaded from {uri} is not a TimeSeriesFeaturePipeline.")
        return pipeline

    def _prepare_base_frame(self, df: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        validate_input_dataframe(df, self.date_col, self.target_col if require_target else None)
        prepared = df.copy()
        prepared[self.date_col] = pd.to_datetime(prepared[self.date_col], errors="coerce")
        if prepared[self.date_col].isna().any():
            raise ValueError(f"Column {self.date_col} contains values that cannot be parsed as datetime.")
        return prepared.sort_values(self.date_col).reset_index(drop=True)

    def _build_features(
        self,
        df: pd.DataFrame,
        macro_df: pd.DataFrame | None,
        tax_dates: list[Any] | None,
    ) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)

        for lag in self.lags:
            X[f"target_lag_{lag}"] = df[self.target_col].shift(lag)

        shifted_target = df[self.target_col].shift(1)
        for window in self.rolling_windows:
            X[f"target_rolling_mean_{window}"] = shifted_target.rolling(window).mean()
            X[f"target_rolling_std_{window}"] = shifted_target.rolling(window).std()

        if self.use_calendar_features:
            self._add_calendar_features(X, df[self.date_col])

        if self.use_tax_features:
            self._add_tax_features(X, df[self.date_col], tax_dates)

        if self.use_macro_features:
            X = self._add_macro_features(X, df, macro_df)

        X.index = df.index
        return X

    def _add_calendar_features(self, X: pd.DataFrame, dates: pd.Series) -> None:
        X["day_of_week"] = dates.dt.dayofweek
        X["day_of_month"] = dates.dt.day
        X["month"] = dates.dt.month
        X["quarter"] = dates.dt.quarter
        X["is_month_start"] = dates.dt.is_month_start.astype(int)
        X["is_month_end"] = dates.dt.is_month_end.astype(int)

    def _add_tax_features(
        self,
        X: pd.DataFrame,
        dates: pd.Series,
        tax_dates: list[Any] | None,
    ) -> None:
        if tax_dates is None:
            X["is_tax_day"] = 0
            return

        parsed_tax_dates = pd.to_datetime(pd.Series(tax_dates), errors="coerce").dropna()
        normalized_tax_dates = set(parsed_tax_dates.dt.normalize())
        X["is_tax_day"] = dates.dt.normalize().isin(normalized_tax_dates).astype(int)

    def _fit_macro(self, df: pd.DataFrame, macro_df: pd.DataFrame | None) -> None:
        macro_source = self._get_macro_source(df, macro_df)
        if macro_source is None:
            self.macro_columns_ = []
            self.macro_numeric_columns_ = []
            self.macro_scaler_ = None
            return

        macro_source = self._prepare_macro_frame(macro_source)
        skipped_columns = [
            col for col in macro_source.columns
            if col != self.date_col and not pd.api.types.is_numeric_dtype(macro_source[col])
        ]
        if skipped_columns:
            warnings.warn(
                f"Skipping non-numeric macro columns: {skipped_columns}",
                stacklevel=2,
            )

        self.macro_columns_ = [
            col for col in macro_source.columns
            if col != self.date_col and pd.api.types.is_numeric_dtype(macro_source[col])
        ]
        self.macro_numeric_columns_ = [
            col for col in self.macro_columns_
            if pd.api.types.is_numeric_dtype(macro_source[col]) and macro_source[col].nunique(dropna=True) > 2
        ]

        if self.macro_numeric_columns_:
            self.macro_scaler_ = StandardScaler()
            self.macro_scaler_.fit(macro_source[self.macro_numeric_columns_])
        else:
            self.macro_scaler_ = None

    def _add_macro_features(
        self,
        X: pd.DataFrame,
        df: pd.DataFrame,
        macro_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        macro_source = self._get_macro_source(df, macro_df)
        if macro_source is None:
            for col in self.macro_columns_:
                X[col] = pd.NA
            if self.macro_columns_:
                warnings.warn(
                    "Macro features were fitted, but macro_df/current macro columns are missing. "
                    "Filled macro features with NaN.",
                    stacklevel=2,
                )
            return X

        macro_source = self._prepare_macro_frame(macro_source)
        macro_source = self._transform_macro_frame(macro_source)
        merged = pd.merge_asof(
            df[[self.date_col]].sort_values(self.date_col),
            macro_source.sort_values(self.date_col),
            on=self.date_col,
            direction="backward",
        )
        merged.index = df.sort_values(self.date_col).index

        for col in self.macro_columns_:
            if col in merged.columns:
                X[col] = merged.loc[X.index, col]
            else:
                X[col] = pd.NA
                warnings.warn(f"Missing macro feature {col}. Filled with NaN.", stacklevel=2)

        return X

    def _get_macro_source(
        self,
        df: pd.DataFrame,
        macro_df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        if macro_df is not None:
            return macro_df

        excluded = {self.date_col, self.target_col}
        macro_columns = [col for col in df.columns if col not in excluded]
        if not macro_columns:
            return None
        return df[[self.date_col] + macro_columns].copy()

    def _prepare_macro_frame(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        validate_input_dataframe(macro_df, self.date_col)
        prepared = macro_df.copy()
        prepared[self.date_col] = pd.to_datetime(prepared[self.date_col], errors="coerce")
        if prepared[self.date_col].isna().any():
            raise ValueError(f"Macro DataFrame column {self.date_col} has invalid datetimes.")
        prepared = prepared.sort_values(self.date_col).drop_duplicates(self.date_col, keep="last")
        return prepared.reset_index(drop=True)

    def _transform_macro_frame(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        transformed = macro_df.copy()
        for col in self.macro_columns_:
            if col not in transformed.columns:
                transformed[col] = pd.NA

        if self.macro_scaler_ is not None and self.macro_numeric_columns_:
            transformed[self.macro_numeric_columns_] = self.macro_scaler_.transform(
                transformed[self.macro_numeric_columns_]
            )

        return transformed[[self.date_col] + self.macro_columns_]

    def _align_to_fitted_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = pd.NA
                warnings.warn(f"Missing feature {col}. Filled with NaN.", stacklevel=2)

        extra_columns = [col for col in X.columns if col not in self.feature_columns]
        if extra_columns:
            warnings.warn(
                f"Dropping unexpected feature columns not seen during fit: {extra_columns}",
                stacklevel=2,
            )

        return X[self.feature_columns]
