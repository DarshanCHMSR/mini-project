from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.features import PreprocessorBundle

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceArtifacts:
    preprocessor: PreprocessorBundle
    model: torch.jit.ScriptModule


def load_inference_artifacts(preprocessor_path: str, model_path: str) -> InferenceArtifacts:
    preprocessor = PreprocessorBundle.load(preprocessor_path)
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    LOGGER.info("Loaded inference artifacts from %s and %s", preprocessor_path, model_path)
    return InferenceArtifacts(preprocessor=preprocessor, model=model)


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
    tabular_array = artifacts.preprocessor.transform_tabular(frame)
    sequence_array = artifacts.preprocessor.transform_sequences(frame)

    with torch.no_grad():
        logits = artifacts.model(
            torch.from_numpy(sequence_array).float(),
            torch.from_numpy(tabular_array).float(),
        )
        probability = torch.sigmoid(logits).cpu().numpy().reshape(-1)[0]

    return {
        "predicted_label": int(probability >= 0.5),
        "defect_probability": float(probability),
        "threshold": 0.5,
    }
