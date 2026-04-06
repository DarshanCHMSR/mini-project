from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    project_root: Path
    data_path: Path
    model_path: Path
    log_level: str
    target_column: str
    test_size: float
    random_state: int
    sequence_length: int
    plots_dir: Path
    logs_dir: Path
    metrics_path: Path
    predictions_path: Path
    feature_importance_path: Path
    confusion_matrix_path: Path
    metrics_plot_path: Path


def load_config() -> Config:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    data_path = Path(os.getenv("DATA_PATH", "dataset_V2.parquet"))
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/xgboost_baseline.json"))
    if not data_path.is_absolute():
        data_path = project_root / data_path
    if not model_path.is_absolute():
        model_path = project_root / model_path
    plots_dir = project_root / "plots"
    logs_dir = project_root / "logs"
    artifacts_dir = model_path.parent

    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        project_root=project_root,
        data_path=data_path,
        model_path=model_path,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        target_column=os.getenv("TARGET_COLUMN", "LBL_NOK"),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        sequence_length=int(os.getenv("SEQUENCE_LENGTH", "2048")),
        plots_dir=plots_dir,
        logs_dir=logs_dir,
        metrics_path=artifacts_dir / "metrics.json",
        predictions_path=artifacts_dir / "test_predictions.csv",
        feature_importance_path=plots_dir / "feature_importance.png",
        confusion_matrix_path=plots_dir / "confusion_matrix.png",
        metrics_plot_path=plots_dir / "performance_metrics.png",
    )
