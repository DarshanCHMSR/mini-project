from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

TABULAR_PREFIXES = ("SET_", "QUA_", "ENV_", "CALC_")


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
    label_columns = [column for column in df.columns if column.startswith("LBL_")]
    if target_column not in df.columns:
        target_column = "LBL_NOK" if "LBL_NOK" in df.columns else label_columns[0]

    sequence_columns = [column for column in df.columns if column.startswith("DXP_")]
    tabular_columns = [column for column in df.columns if column.startswith(TABULAR_PREFIXES)]

    LOGGER.info(
        "Detected %s tabular columns, %s time-series columns, and %s label columns",
        len(tabular_columns),
        len(sequence_columns),
        len(label_columns),
    )
    return DatasetBundle(df, target_column, label_columns, tabular_columns, sequence_columns)


def ensure_binary_target(series: pd.Series) -> pd.Series:
    values = set(series.dropna().unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Expected a binary target but found values: {sorted(values)}")
    return series.astype(int)


def summarize_dataset(df: pd.DataFrame, target_column: str) -> dict:
    target_distribution = {str(key): int(value) for key, value in df[target_column].value_counts().to_dict().items()}
    missing_ratio = {
        str(key): float(value)
        for key, value in df.isna().mean().sort_values(ascending=False).head(15).to_dict().items()
    }
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_distribution": target_distribution,
        "top_missing_ratio": missing_ratio,
    }
    LOGGER.info("Dataset summary: %s", summary)
    return summary


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


def create_split_indices(
    target: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
) -> dict[str, list[int]]:
    indices = target.index.to_numpy()
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=target.values if target.nunique() > 1 else None,
    )

    val_fraction = val_size / max(1.0 - test_size, 1e-6)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=random_state,
        stratify=target.loc[train_val_idx].values if target.nunique() > 1 else None,
    )

    split_indices = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    LOGGER.info(
        "Created data splits with train=%s val=%s test=%s",
        len(split_indices["train"]),
        len(split_indices["val"]),
        len(split_indices["test"]),
    )
    return split_indices


def save_split_indices(split_indices: dict[str, list[int]], destination: Path) -> None:
    destination.write_text(json.dumps(split_indices, indent=2), encoding="utf-8")
    LOGGER.info("Saved split indices to %s", destination)
