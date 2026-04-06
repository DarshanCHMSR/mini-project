from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import load_config
from src.inference import load_inference_artifacts, predict_payload


class PredictionRequest(BaseModel):
    tabular: dict[str, float] = Field(default_factory=dict)
    sequences: dict[str, list[float]] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def get_artifacts():
    config = load_config()
    return load_inference_artifacts(str(config.preprocessor_path), str(config.fusion_torchscript_path))


app = FastAPI(title="Defect Detection API", version="2.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    artifacts = get_artifacts()
    return predict_payload(artifacts, request.tabular, request.sequences)
