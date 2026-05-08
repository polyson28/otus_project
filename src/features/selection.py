from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer

from src.io.s3 import joblib_dump_to_uri, joblib_load_from_uri


LOGGER = logging.getLogger(__name__)


class TimeSeriesFeatureSelector:
    def __init__(
        self,
        target_col: str,
        date_col: str,
        missing_threshold: float = 0.4,
        correlation_threshold: float = 0.95,
        remove_constant: bool = True,
        remove_high_missing: bool = True,
        remove_high_correlation: bool = True,
        use_mutual_info: bool = True,
        use_correlation_with_target: bool = True,
        max_features: int | None = None,
        always_keep: list[str] | None = None,
        min_features_to_select: int | None = None,
        validation_size: float = 0.2,
        random_state: int = 42,
    ):
        self.target_col = target_col
        self.date_col = date_col
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self.remove_constant = remove_constant
        self.remove_high_missing = remove_high_missing
        self.remove_high_correlation = remove_high_correlation
        self.use_mutual_info = use_mutual_info
        self.use_correlation_with_target = use_correlation_with_target
        self.max_features = max_features
        self.always_keep = always_keep or []
        self.min_features_to_select = min_features_to_select
        self.validation_size = validation_size
        self.random_state = random_state

        self.selected_features_: list[str] = []
        self.dropped_features_: dict[str, list[str]] = {}
        self.feature_scores_: dict[str, dict[str, float | None]] = {}
        self.report_: dict[str, Any] = {}
        self.candidate_features_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "TimeSeriesFeatureSelector":
        prepared = self._prepare_input(df, require_target=True)
        fit_frame = self._historical_fit_slice(prepared)
        y = fit_frame[self.target_col]

        technical_columns = {self.date_col}
        candidate_features = [
            col for col in prepared.select_dtypes(include=[np.number]).columns
            if col not in technical_columns and col != self.target_col
        ]
        if not candidate_features:
            raise ValueError("No numeric candidate feature columns found for feature selection.")

        self.candidate_features_ = candidate_features
        self.dropped_features_ = {
            "non_numeric": [
                col for col in prepared.columns
                if col not in technical_columns
                and col != self.target_col
                and col not in candidate_features
            ],
            "high_missing": [],
            "constant": [],
            "high_correlation": [],
            "max_features": [],
        }
        self.feature_scores_ = self._initial_scores(fit_frame, candidate_features)

        selected = list(candidate_features)
        selected = self._remove_high_missing(fit_frame, selected)
        selected = self._remove_constant(fit_frame, selected)

        self._add_target_correlation_scores(fit_frame, selected, y)
        selected = self._remove_high_correlation(fit_frame, selected)
        self._add_mutual_info_scores(fit_frame, selected, y)
        self._update_combined_scores(selected)
        selected = self._apply_max_features(selected)
        selected = self._apply_always_keep(prepared, selected)

        self.selected_features_ = selected
        self.report_ = self._build_report(prepared, fit_frame)
        LOGGER.info("Selected %s of %s candidate features.", len(selected), len(candidate_features))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise ValueError("Feature selector is not fitted. Call fit() or load a fitted selector first.")

        prepared = self._prepare_input(df, require_target=False)
        for feature in self.selected_features_:
            if feature not in prepared.columns:
                warnings.warn(f"Selected feature '{feature}' is missing in input. Filled with NaN.", stacklevel=2)
                LOGGER.warning("Selected feature %s is missing in input. Filled with NaN.", feature)
                prepared[feature] = np.nan

        if self.date_col not in prepared.columns:
            warnings.warn(f"Date column '{self.date_col}' is missing in input. Filled with NaT.", stacklevel=2)
            LOGGER.warning("Date column %s is missing in input. Filled with NaT.", self.date_col)
            prepared[self.date_col] = pd.NaT

        output_columns = [self.date_col] + self.selected_features_
        if self.target_col in prepared.columns:
            output_columns.append(self.target_col)
        return prepared[output_columns].copy()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, uri: str) -> None:
        joblib_dump_to_uri(self, uri)

    @classmethod
    def load(cls, uri: str) -> "TimeSeriesFeatureSelector":
        selector = joblib_load_from_uri(uri)
        if not isinstance(selector, cls):
            raise TypeError(f"Object loaded from {uri} is not a TimeSeriesFeatureSelector.")
        return selector

    def get_selected_features(self) -> list[str]:
        return list(self.selected_features_)

    def get_report(self) -> dict[str, Any]:
        return dict(self.report_)

    def _prepare_input(self, df: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input dataframe is empty.")
        if self.date_col not in df.columns:
            raise ValueError(f"Missing required date column: {self.date_col}")
        if require_target and self.target_col not in df.columns:
            raise ValueError(f"Missing required target column: {self.target_col}")

        prepared = df.copy()
        prepared[self.date_col] = pd.to_datetime(prepared[self.date_col], errors="coerce")
        invalid_dates = int(prepared[self.date_col].isna().sum())
        if invalid_dates:
            raise ValueError(f"Date column '{self.date_col}' contains {invalid_dates} invalid datetimes.")

        prepared = prepared.sort_values(self.date_col).reset_index(drop=True)
        if require_target and prepared[self.target_col].isna().any():
            raise ValueError(f"Target column '{self.target_col}' contains missing values.")
        return prepared

    def _initial_scores(self, df: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float | None]]:
        scores = {}
        for feature in features:
            scores[feature] = {
                "missing_rate": float(df[feature].isna().mean()),
                "n_unique": float(df[feature].nunique(dropna=True)),
                "mutual_info": None,
                "correlation_with_target": None,
                "combined_score": None,
            }
        return scores

    def _historical_fit_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        if 0 < self.validation_size < 1 and len(df) > 1:
            split_idx = max(1, int(len(df) * (1 - self.validation_size)))
            LOGGER.info(
                "Using first %s rows for target-aware feature selection; holding out %s rows.",
                split_idx,
                len(df) - split_idx,
            )
            return df.iloc[:split_idx].copy()
        return df.copy()

    def _remove_high_missing(self, df: pd.DataFrame, features: list[str]) -> list[str]:
        if not self.remove_high_missing:
            return features
        kept = []
        for feature in features:
            missing_rate = self.feature_scores_[feature]["missing_rate"] or 0.0
            if missing_rate > self.missing_threshold and feature not in self.always_keep:
                self.dropped_features_["high_missing"].append(feature)
                LOGGER.warning("Dropped high-missing feature %s: %.4f", feature, missing_rate)
            else:
                kept.append(feature)
        return kept

    def _remove_constant(self, df: pd.DataFrame, features: list[str]) -> list[str]:
        if not self.remove_constant:
            return features
        kept = []
        for feature in features:
            n_unique = int(df[feature].nunique(dropna=True))
            if n_unique <= 1 and feature not in self.always_keep:
                self.dropped_features_["constant"].append(feature)
                LOGGER.warning("Dropped constant feature %s.", feature)
            else:
                kept.append(feature)
        return kept

    def _add_target_correlation_scores(
        self,
        df: pd.DataFrame,
        features: list[str],
        y: pd.Series,
    ) -> None:
        if not self.use_correlation_with_target:
            return
        for feature in features:
            corr = df[feature].corr(y)
            self.feature_scores_[feature]["correlation_with_target"] = None if pd.isna(corr) else float(abs(corr))

    def _remove_high_correlation(self, df: pd.DataFrame, features: list[str]) -> list[str]:
        if not self.remove_high_correlation or len(features) < 2:
            return features

        corr_matrix = df[features].corr().abs()
        original_order = {feature: idx for idx, feature in enumerate(features)}
        dropped: set[str] = set()

        for i, feature in enumerate(features):
            if feature in dropped:
                continue
            for other in features[i + 1:]:
                if other in dropped:
                    continue
                corr_value = corr_matrix.loc[feature, other]
                if pd.isna(corr_value) or corr_value <= self.correlation_threshold:
                    continue

                drop_feature = self._choose_correlated_drop(feature, other, original_order)
                if drop_feature in self.always_keep:
                    drop_feature = other if drop_feature == feature else feature
                if drop_feature in self.always_keep:
                    continue

                dropped.add(drop_feature)
                self.dropped_features_["high_correlation"].append(drop_feature)
                LOGGER.warning(
                    "Dropped high-correlation feature %s from pair (%s, %s), corr=%.4f.",
                    drop_feature,
                    feature,
                    other,
                    corr_value,
                )

        return [feature for feature in features if feature not in dropped]

    def _choose_correlated_drop(
        self,
        feature: str,
        other: str,
        original_order: dict[str, int],
    ) -> str:
        feature_score = self.feature_scores_.get(feature, {}).get("correlation_with_target")
        other_score = self.feature_scores_.get(other, {}).get("correlation_with_target")
        feature_score = -1.0 if feature_score is None else feature_score
        other_score = -1.0 if other_score is None else other_score

        if feature_score < other_score:
            return feature
        if other_score < feature_score:
            return other
        return other if original_order[feature] < original_order[other] else feature

    def _add_mutual_info_scores(self, df: pd.DataFrame, features: list[str], y: pd.Series) -> None:
        if not self.use_mutual_info or not features:
            return
        imputed = SimpleImputer(strategy="median").fit_transform(df[features])
        mi_scores = mutual_info_regression(imputed, y, random_state=self.random_state)
        for feature, score in zip(features, mi_scores):
            self.feature_scores_[feature]["mutual_info"] = float(score)

    def _update_combined_scores(self, features: list[str]) -> None:
        mi_values = {
            feature: self.feature_scores_[feature]["mutual_info"]
            for feature in features
            if self.feature_scores_[feature]["mutual_info"] is not None
        }
        corr_values = {
            feature: self.feature_scores_[feature]["correlation_with_target"]
            for feature in features
            if self.feature_scores_[feature]["correlation_with_target"] is not None
        }
        normalized_mi = self._normalize_scores(mi_values)
        normalized_corr = self._normalize_scores(corr_values)

        for feature in features:
            components = []
            if feature in normalized_mi:
                components.append(normalized_mi[feature])
            if feature in normalized_corr:
                components.append(normalized_corr[feature])
            self.feature_scores_[feature]["combined_score"] = float(np.mean(components)) if components else 0.0

    def _apply_max_features(self, features: list[str]) -> list[str]:
        if self.max_features is None or len(features) <= self.max_features:
            return features

        min_features = self.min_features_to_select or 1
        limit = max(min_features, self.max_features)
        original_order = {feature: idx for idx, feature in enumerate(features)}
        ranked = sorted(
            features,
            key=lambda feature: (
                -(self.feature_scores_[feature]["combined_score"] or 0.0),
                original_order[feature],
            ),
        )
        selected_set = set(ranked[:limit])
        dropped = [feature for feature in features if feature not in selected_set and feature not in self.always_keep]
        self.dropped_features_["max_features"].extend(dropped)
        return [feature for feature in features if feature in selected_set or feature in self.always_keep]

    def _apply_always_keep(self, df: pd.DataFrame, selected: list[str]) -> list[str]:
        result = list(selected)
        for feature in self.always_keep:
            if feature in {self.date_col, self.target_col}:
                continue
            if feature in df.columns and feature not in result and pd.api.types.is_numeric_dtype(df[feature]):
                result.append(feature)
        return result

    def _build_report(self, df: pd.DataFrame, fit_frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "target_col": self.target_col,
            "date_col": self.date_col,
            "rows": len(df),
            "fit_rows": len(fit_frame),
            "validation_rows": len(df) - len(fit_frame),
            "candidate_features": self.candidate_features_,
            "selected_features": self.selected_features_,
            "dropped_features": self.dropped_features_,
            "feature_scores": self.feature_scores_,
            "params": {
                "missing_threshold": self.missing_threshold,
                "correlation_threshold": self.correlation_threshold,
                "remove_constant": self.remove_constant,
                "remove_high_missing": self.remove_high_missing,
                "remove_high_correlation": self.remove_high_correlation,
                "use_mutual_info": self.use_mutual_info,
                "use_correlation_with_target": self.use_correlation_with_target,
                "max_features": self.max_features,
                "min_features_to_select": self.min_features_to_select,
                "validation_size": self.validation_size,
                "always_keep": self.always_keep,
            },
        }

    def _normalize_scores(self, scores: dict[str, float | None]) -> dict[str, float]:
        values = [float(value) for value in scores.values() if value is not None]
        if not values:
            return {}
        min_value = min(values)
        max_value = max(values)
        if max_value == min_value:
            return {feature: 1.0 for feature in scores}
        return {
            feature: (float(value) - min_value) / (max_value - min_value)
            for feature, value in scores.items()
            if value is not None
        }
