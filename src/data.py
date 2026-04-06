from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

TABULAR_PREFIXES = ("SET_", "QUA_", "ENV_", "CALC_")
EXCLUDED_PREFIXES = ("CV_", "IR_")


@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame
    target_column: str
    label_columns: list[str]
    tabular_columns: list[str]
    sequence_columns: list[str]


def load_dataset(data_path: str) -> pd.DataFrame:
    LOGGER.info("Loading parquet dataset from %s", data_path)
    return pd.read_parquet(data_path)


def identify_columns(df: pd.DataFrame, target_column: str) -> DatasetBundle:
    label_columns = [col for col in df.columns if col.startswith("LBL_")]
    if target_column not in df.columns:
        if "LBL_NOK" in df.columns:
            target_column = "LBL_NOK"
        elif label_columns:
            target_column = label_columns[0]
        else:
            raise ValueError("No target column was found in the dataset.")

    sequence_columns = [col for col in df.columns if col.startswith("DXP_")]
    tabular_columns = [
        col
        for col in df.columns
        if col.startswith(TABULAR_PREFIXES) and not col.startswith(EXCLUDED_PREFIXES)
    ]

    LOGGER.info(
        "Detected %s tabular columns, %s time-series columns, and %s label columns",
        len(tabular_columns),
        len(sequence_columns),
        len(label_columns),
    )

    return DatasetBundle(
        dataframe=df,
        target_column=target_column,
        label_columns=label_columns,
        tabular_columns=tabular_columns,
        sequence_columns=sequence_columns,
    )


def summarize_dataset(df: pd.DataFrame, target_column: str) -> dict:
    raw_target_distribution = df[target_column].value_counts(dropna=False).to_dict()
    raw_missing_ratio = df.isna().mean().sort_values(ascending=False).head(15).to_dict()
    target_distribution = {str(key): int(value) for key, value in raw_target_distribution.items()}
    missing_ratio = {str(key): float(value) for key, value in raw_missing_ratio.items()}
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_distribution": target_distribution,
        "top_missing_ratio": missing_ratio,
    }
    LOGGER.info("Dataset summary: %s", summary)
    return summary


def ensure_binary_target(series: pd.Series) -> pd.Series:
    values = set(series.dropna().unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Expected a binary target but found values: {sorted(values)}")
    return series.astype(int)


def compute_sequence_length_stats(df: pd.DataFrame, sequence_columns: list[str]) -> dict[str, float]:
    if not sequence_columns:
        return {}

    lengths = []
    for value in df[sequence_columns[0]]:
        try:
            lengths.append(len(value))
        except TypeError:
            lengths.append(0)

    stats = {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "median": float(np.median(lengths)),
        "mean": float(np.mean(lengths)),
    }
    LOGGER.info("Sequence length stats based on %s: %s", sequence_columns[0], stats)
    return stats
