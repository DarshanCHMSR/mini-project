from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import load_config
from src.inference import load_inference_artifacts, predict_payload


class PredictionRequest(BaseModel):
    tabular: dict[str, float] = Field(default_factory=dict)
    sequences: dict[str, list[float]] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def get_artifacts():
    config = load_config()
    return load_inference_artifacts(
        str(config.preprocessor_path),
        str(config.fusion_torchscript_path),
        str(config.xgb_model_path),
        str(config.xgb_preprocessor_path),
        str(config.serving_config_path),
    )


app = FastAPI(title="Defect Detection API", version="2.0.0")
config = load_config()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if config.plots_dir.exists():
    app.mount("/plots", StaticFiles(directory=config.plots_dir), name="plots")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _plot_manifest() -> list[dict[str, str]]:
    return [
        {
            "key": "xgboost_confusion_matrix",
            "title": "XGBoost Confusion Matrix",
            "description": "Baseline classifier confusion matrix on the held-out test split.",
            "path": "/plots/xgboost_confusion_matrix.png",
        },
        {
            "key": "fusion_confusion_matrix",
            "title": "Fusion Confusion Matrix",
            "description": "LSTM fusion model confusion matrix on the held-out test split.",
            "path": "/plots/fusion_confusion_matrix.png",
        },
        {
            "key": "fusion_roc_curve",
            "title": "Fusion ROC Curve",
            "description": "Threshold sensitivity curve for the fusion model.",
            "path": "/plots/fusion_roc_curve.png",
        },
        {
            "key": "fusion_training_curves",
            "title": "Fusion Training Curves",
            "description": "Training and validation loss, accuracy, and F1 over epochs.",
            "path": "/plots/fusion_training_curves.png",
        },
        {
            "key": "xgboost_feature_importance",
            "title": "XGBoost Feature Importance",
            "description": "Top engineered features contributing to the baseline model.",
            "path": "/plots/xgboost_feature_importance.png",
        },
        {
            "key": "model_comparison",
            "title": "Model Comparison",
            "description": "Side-by-side comparison of baseline and fusion model performance.",
            "path": "/plots/model_comparison.png",
        },
        {
            "key": "performance_summary",
            "title": "Performance Summary",
            "description": "Heatmap summary of evaluation metrics for both models.",
            "path": "/plots/performance_summary.png",
        },
    ]


@app.get("/health")
def health() -> dict[str, str]:
    artifacts_ready = config.preprocessor_path.exists() and config.fusion_torchscript_path.exists()
    return {"status": "ok", "artifacts_ready": "true" if artifacts_ready else "false"}


@app.get("/artifacts/schema")
def artifact_schema() -> dict:
    artifacts = get_artifacts()
    preprocessor = artifacts.preprocessor
    suggested_tabular = [
        column
        for column in preprocessor.tabular_columns
        if column.startswith(("SET_", "QUA_", "ENV_"))
    ][:8]
    suggested_sequences = preprocessor.sequence_columns[:6]
    return {
        "target_column": preprocessor.target_column,
        "tabular_columns": preprocessor.tabular_columns,
        "sequence_columns": preprocessor.sequence_columns,
        "suggested_tabular_fields": suggested_tabular,
        "suggested_sequence_fields": suggested_sequences,
        "sequence_length": preprocessor.sequence_length,
    }


@app.get("/artifacts/summary")
def artifact_summary() -> dict:
    comparison = _read_json(config.comparison_metrics_path)
    xgboost_metrics = _read_json(config.xgb_metrics_path)
    fusion_metrics = _read_json(config.fusion_metrics_path)
    metadata = _read_json(config.metadata_path)
    serving = _read_json(config.serving_config_path)
    return {
        "health": health(),
        "comparison": comparison,
        "serving": serving,
        "models": {
            "xgboost": xgboost_metrics,
            "fusion": fusion_metrics,
        },
        "metadata": metadata,
        "plots": _plot_manifest(),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    artifacts = get_artifacts()
    return predict_payload(artifacts, request.tabular, request.sequences)
