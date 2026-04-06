from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


def pad_or_truncate_sequence(values: object, sequence_length: int) -> np.ndarray:
    if values is None:
        array = np.array([], dtype=np.float32)
    elif isinstance(values, np.ndarray):
        array = values.astype(np.float32, copy=False)
    elif isinstance(values, (list, tuple)):
        array = np.asarray(values, dtype=np.float32)
    else:
        array = np.array([], dtype=np.float32)

    if array.ndim != 1:
        array = np.ravel(array).astype(np.float32, copy=False)

    finite_mask = np.isfinite(array)
    if not finite_mask.all():
        array = array[finite_mask]

    if array.size >= sequence_length:
        return array[:sequence_length]

    if array.size == 0:
        return np.zeros(sequence_length, dtype=np.float32)

    pad_width = sequence_length - array.size
    return np.pad(array, (0, pad_width), mode="constant", constant_values=0.0)


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
    sequence_columns: Iterable[str],
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
            processed_sequence = pad_or_truncate_sequence(values, sequence_length)
            stats = sequence_statistics(processed_sequence)
            stats_by_name[f"{column}__mean"].append(stats["mean"])
            stats_by_name[f"{column}__std"].append(stats["std"])
            stats_by_name[f"{column}__min"].append(stats["min"])
            stats_by_name[f"{column}__max"].append(stats["max"])
            stats_by_name[f"{column}__peak_abs"].append(stats["peak_abs"])
            stats_by_name[f"{column}__slope"].append(stats["slope"])

        feature_rows.update(stats_by_name)

    time_series_features = pd.DataFrame(feature_rows, index=df.index)
    LOGGER.info(
        "Created %s engineered time-series features from %s DXP signals",
        time_series_features.shape[1],
        len(list(sequence_columns)),
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


def split_and_scale_features(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
):
    stratify = target if target.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)

    x_train_scaled = scaler.fit_transform(x_train_imputed)
    x_test_scaled = scaler.transform(x_test_imputed)

    train_frame = pd.DataFrame(x_train_scaled, columns=features.columns, index=x_train.index)
    test_frame = pd.DataFrame(x_test_scaled, columns=features.columns, index=x_test.index)

    LOGGER.info(
        "Prepared train/test split with train shape %s and test shape %s",
        train_frame.shape,
        test_frame.shape,
    )
    return train_frame, test_frame, y_train, y_test, imputer, scaler
