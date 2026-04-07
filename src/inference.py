from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from src.features import PreprocessorBundle, build_time_series_features, combine_features

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceArtifacts:
    preprocessor: PreprocessorBundle
    fusion_model: torch.jit.ScriptModule
    xgb_model: XGBClassifier
    xgb_preprocessor: object
    selected_model: str
    fusion_threshold: float
    xgb_threshold: float
    feature_sequence_length: int


def _load_serving_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {
            "selected_model": "fusion",
            "thresholds": {"fusion": 0.5, "xgboost": 0.5},
            "feature_sequence_length": 512,
        }
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_inference_artifacts(
    preprocessor_path: str,
    fusion_model_path: str,
    xgb_model_path: str,
    xgb_preprocessor_path: str,
    serving_config_path: str,
) -> InferenceArtifacts:
    preprocessor = PreprocessorBundle.load(preprocessor_path)
    fusion_model = torch.jit.load(fusion_model_path, map_location="cpu")
    fusion_model.eval()

    xgb_model = XGBClassifier()
    xgb_model.load_model(xgb_model_path)
    xgb_preprocessor = joblib.load(xgb_preprocessor_path)

    serving_config = _load_serving_config(serving_config_path)
    selected_model = str(serving_config.get("selected_model", "fusion")).lower()
    thresholds = serving_config.get("thresholds", {})
    fusion_threshold = float(thresholds.get("fusion", 0.5))
    xgb_threshold = float(thresholds.get("xgboost", 0.5))
    feature_sequence_length = int(serving_config.get("feature_sequence_length", 512))

    LOGGER.info(
        "Loaded inference artifacts | selected_model=%s fusion_threshold=%.4f xgb_threshold=%.4f",
        selected_model,
        fusion_threshold,
        xgb_threshold,
    )
    return InferenceArtifacts(
        preprocessor=preprocessor,
        fusion_model=fusion_model,
        xgb_model=xgb_model,
        xgb_preprocessor=xgb_preprocessor,
        selected_model=selected_model,
        fusion_threshold=fusion_threshold,
        xgb_threshold=xgb_threshold,
        feature_sequence_length=feature_sequence_length,
    )


def _build_single_row_frame(tabular: dict[str, float], sequences: dict[str, list[float]], preprocessor: PreprocessorBundle) -> pd.DataFrame:
    row: dict[str, object] = {}
    for column in preprocessor.tabular_columns:
        row[column] = tabular.get(column, np.nan)
    for column in preprocessor.sequence_columns:
        row[column] = sequences.get(column, [])
    return pd.DataFrame([row])


def predict_payload(
    artifacts: InferenceArtifacts,
    tabular: dict[str, float],
    sequences: dict[str, list[float]],
) -> dict:
    frame = _build_single_row_frame(tabular, sequences, artifacts.preprocessor)

    if artifacts.selected_model == "xgboost":
        tabular_frame = frame.reindex(columns=artifacts.preprocessor.tabular_columns).copy()
        for column in tabular_frame.columns:
            tabular_frame[column] = pd.to_numeric(tabular_frame[column], errors="coerce")
        sequence_features = build_time_series_features(
            frame,
            artifacts.preprocessor.sequence_columns,
            artifacts.feature_sequence_length,
        )
        feature_frame = combine_features(tabular_frame, sequence_features)
        feature_frame = feature_frame.reindex(columns=artifacts.xgb_preprocessor.feature_columns)
        transformed = artifacts.xgb_preprocessor.transform(feature_frame)
        probability = float(artifacts.xgb_model.predict_proba(transformed)[:, 1][0])
        threshold = artifacts.xgb_threshold
        return {
            "predicted_label": int(probability >= threshold),
            "defect_probability": probability,
            "threshold": threshold,
            "model_used": "xgboost",
        }

    tabular_array = artifacts.preprocessor.transform_tabular(frame)
    sequence_array = artifacts.preprocessor.transform_sequences(frame)

    with torch.no_grad():
        logits = artifacts.fusion_model(
            torch.from_numpy(sequence_array).float(),
            torch.from_numpy(tabular_array).float(),
        )
        probability = float(torch.sigmoid(logits).cpu().numpy().reshape(-1)[0])

    return {
        "predicted_label": int(probability >= artifacts.fusion_threshold),
        "defect_probability": probability,
        "threshold": artifacts.fusion_threshold,
        "model_used": "fusion",
    }
