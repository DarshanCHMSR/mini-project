from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


def to_1d_array(values: object) -> np.ndarray:
    if values is None:
        array = np.array([], dtype=np.float32)
    elif isinstance(values, np.ndarray):
        array = values.astype(np.float32, copy=False)
    elif isinstance(values, (list, tuple)):
        array = np.asarray(values, dtype=np.float32)
    else:
        array = np.array([], dtype=np.float32)

    array = np.ravel(array).astype(np.float32, copy=False)
    if array.size == 0:
        return array

    finite_mask = np.isfinite(array)
    return array[finite_mask]


def resize_sequence(values: object, target_length: int) -> np.ndarray:
    array = to_1d_array(values)
    if array.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if array.size == 1:
        return np.full(target_length, array[0], dtype=np.float32)
    if array.size == target_length:
        return array.astype(np.float32, copy=False)

    source_positions = np.linspace(0.0, 1.0, num=array.size, dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, array).astype(np.float32)


def sequence_statistics(sequence: np.ndarray) -> dict[str, float]:
    slope = float((sequence[-1] - sequence[0]) / max(sequence.size - 1, 1)) if sequence.size > 1 else 0.0
    return {
        "mean": float(np.mean(sequence)),
        "std": float(np.std(sequence)),
        "min": float(np.min(sequence)),
        "max": float(np.max(sequence)),
        "peak_abs": float(np.max(np.abs(sequence))),
        "slope": slope,
    }


def build_time_series_features(
    df: pd.DataFrame,
    sequence_columns: list[str],
    sequence_length: int,
) -> pd.DataFrame:
    feature_rows: dict[str, list[float]] = {}
    for column in sequence_columns:
        stats_by_name = {
            f"{column}__mean": [],
            f"{column}__std": [],
            f"{column}__min": [],
            f"{column}__max": [],
            f"{column}__peak_abs": [],
            f"{column}__slope": [],
        }
        for values in df[column]:
            sequence = resize_sequence(values, sequence_length)
            stats = sequence_statistics(sequence)
            for suffix, stat_name in (
                ("mean", "mean"),
                ("std", "std"),
                ("min", "min"),
                ("max", "max"),
                ("peak_abs", "peak_abs"),
                ("slope", "slope"),
            ):
                stats_by_name[f"{column}__{suffix}"].append(stats[stat_name])
        feature_rows.update(stats_by_name)

    time_series_features = pd.DataFrame(feature_rows, index=df.index).replace([np.inf, -np.inf], np.nan)
    LOGGER.info(
        "Created %s engineered time-series features from %s DXP signals",
        time_series_features.shape[1],
        len(sequence_columns),
    )
    return time_series_features


def prepare_tabular_features(df: pd.DataFrame, tabular_columns: list[str]) -> pd.DataFrame:
    tabular = df.loc[:, tabular_columns].copy()
    for column in tabular.columns:
        tabular[column] = pd.to_numeric(tabular[column], errors="coerce")

    empty_columns = [column for column in tabular.columns if tabular[column].isna().all()]
    if empty_columns:
        LOGGER.warning("Dropping %s fully empty tabular columns", len(empty_columns))
        tabular = tabular.drop(columns=empty_columns)
    return tabular


def combine_features(tabular_features: pd.DataFrame, time_series_features: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([tabular_features, time_series_features], axis=1)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    LOGGER.info("Combined feature matrix shape: %s", combined.shape)
    return combined


def build_sequence_tensor(
    df: pd.DataFrame,
    sequence_columns: list[str],
    sequence_length: int,
) -> np.ndarray:
    num_rows = len(df)
    num_channels = len(sequence_columns)
    sequences = np.zeros((num_rows, sequence_length, num_channels), dtype=np.float32)

    for row_idx, (_, row) in enumerate(df[sequence_columns].iterrows()):
        for channel_idx, column in enumerate(sequence_columns):
            sequences[row_idx, :, channel_idx] = resize_sequence(row[column], sequence_length)

    LOGGER.info("Built sequence tensor with shape %s", sequences.shape)
    return sequences


@dataclass
class FeaturePreprocessor:
    feature_columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame.loc[:, self.feature_columns]
        imputed = self.imputer.transform(ordered)
        scaled = self.scaler.transform(imputed)
        return pd.DataFrame(scaled, index=ordered.index, columns=self.feature_columns)


@dataclass
class PreprocessorBundle:
    tabular_columns: list[str]
    sequence_columns: list[str]
    sequence_length: int
    tabular_imputer: SimpleImputer
    tabular_scaler: StandardScaler
    sequence_channel_mean: np.ndarray
    sequence_channel_std: np.ndarray
    target_column: str

    def transform_tabular(self, df: pd.DataFrame) -> np.ndarray:
        tabular = df.reindex(columns=self.tabular_columns).copy()
        for column in tabular.columns:
            tabular[column] = pd.to_numeric(tabular[column], errors="coerce")
        transformed = self.tabular_scaler.transform(self.tabular_imputer.transform(tabular))
        return transformed.astype(np.float32)

    def transform_sequences(self, df: pd.DataFrame) -> np.ndarray:
        sequences = build_sequence_tensor(df, self.sequence_columns, self.sequence_length)
        normalized = (sequences - self.sequence_channel_mean.reshape(1, 1, -1)) / self.sequence_channel_std.reshape(
            1, 1, -1
        )
        return normalized.astype(np.float32)

    def save(self, destination: str) -> None:
        joblib.dump(self, destination)

    @classmethod
    def load(cls, source: str) -> "PreprocessorBundle":
        return joblib.load(source)


def fit_feature_preprocessor(train_frame: pd.DataFrame) -> FeaturePreprocessor:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(train_frame)
    scaler.fit(imputed)
    return FeaturePreprocessor(feature_columns=list(train_frame.columns), imputer=imputer, scaler=scaler)


def fit_multimodal_preprocessor(
    train_df: pd.DataFrame,
    tabular_columns: list[str],
    sequence_columns: list[str],
    sequence_length: int,
    target_column: str,
) -> PreprocessorBundle:
    tabular_train = prepare_tabular_features(train_df, tabular_columns)
    tabular_imputer = SimpleImputer(strategy="median")
    tabular_scaler = StandardScaler()
    tabular_imputed = tabular_imputer.fit_transform(tabular_train)
    tabular_scaler.fit(tabular_imputed)

    train_sequences = build_sequence_tensor(train_df, sequence_columns, sequence_length)
    channel_mean = train_sequences.mean(axis=(0, 1))
    channel_std = train_sequences.std(axis=(0, 1))
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)

    LOGGER.info(
        "Fitted multimodal preprocessor for %s tabular and %s sequence columns",
        len(tabular_train.columns),
        len(sequence_columns),
    )
    return PreprocessorBundle(
        tabular_columns=list(tabular_train.columns),
        sequence_columns=sequence_columns,
        sequence_length=sequence_length,
        tabular_imputer=tabular_imputer,
        tabular_scaler=tabular_scaler,
        sequence_channel_mean=channel_mean.astype(np.float32),
        sequence_channel_std=channel_std.astype(np.float32),
        target_column=target_column,
    )
