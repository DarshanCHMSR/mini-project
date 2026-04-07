from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    project_root: Path
    data_dir: Path
    processed_dir: Path
    models_dir: Path
    plots_dir: Path
    logs_dir: Path
    data_path: Path
    xgb_model_path: Path
    xgb_preprocessor_path: Path
    fusion_checkpoint_path: Path
    fusion_torchscript_path: Path
    fusion_onnx_path: Path
    preprocessor_path: Path
    split_indices_path: Path
    metadata_path: Path
    xgb_metrics_path: Path
    fusion_metrics_path: Path
    comparison_metrics_path: Path
    serving_config_path: Path
    xgb_predictions_path: Path
    fusion_predictions_path: Path
    xgb_feature_importance_path: Path
    xgb_confusion_matrix_path: Path
    fusion_confusion_matrix_path: Path
    training_curves_path: Path
    roc_curve_path: Path
    comparison_plot_path: Path
    metrics_plot_path: Path
    log_level: str
    target_column: str
    test_size: float
    val_size: float
    random_state: int
    feature_sequence_length: int
    lstm_sequence_length: int
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    lstm_layers: int
    tabular_hidden_dim: int
    fusion_hidden_dim: int
    dropout: float
    use_attention: bool


def _resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def load_config() -> Config:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    models_dir = project_root / "models"
    plots_dir = project_root / "plots"
    logs_dir = project_root / "logs"

    for directory in (data_dir, processed_dir, models_dir, plots_dir, logs_dir, project_root / "api"):
        directory.mkdir(parents=True, exist_ok=True)

    data_path = _resolve_path(project_root, os.getenv("DATA_PATH", "dataset_V2.parquet"))

    return Config(
        project_root=project_root,
        data_dir=data_dir,
        processed_dir=processed_dir,
        models_dir=models_dir,
        plots_dir=plots_dir,
        logs_dir=logs_dir,
        data_path=data_path,
        xgb_model_path=_resolve_path(project_root, os.getenv("XGB_MODEL_PATH", "models/xgboost_baseline.json")),
        xgb_preprocessor_path=_resolve_path(project_root, os.getenv("XGB_PREPROCESSOR_PATH", "models/xgboost_preprocessor.joblib")),
        fusion_checkpoint_path=_resolve_path(project_root, os.getenv("FUSION_CHECKPOINT_PATH", "models/fusion_best.pt")),
        fusion_torchscript_path=_resolve_path(project_root, os.getenv("FUSION_TORCHSCRIPT_PATH", "models/fusion_model.ts")),
        fusion_onnx_path=_resolve_path(project_root, os.getenv("FUSION_ONNX_PATH", "models/fusion_model.onnx")),
        preprocessor_path=_resolve_path(project_root, os.getenv("PREPROCESSOR_PATH", "models/preprocessor_bundle.joblib")),
        split_indices_path=_resolve_path(project_root, os.getenv("SPLIT_INDICES_PATH", "data/processed/split_indices.json")),
        metadata_path=_resolve_path(project_root, os.getenv("METADATA_PATH", "models/run_metadata.json")),
        xgb_metrics_path=_resolve_path(project_root, os.getenv("XGB_METRICS_PATH", "models/xgboost_metrics.json")),
        fusion_metrics_path=_resolve_path(project_root, os.getenv("FUSION_METRICS_PATH", "models/fusion_metrics.json")),
        comparison_metrics_path=_resolve_path(project_root, os.getenv("COMPARISON_METRICS_PATH", "models/comparison_metrics.json")),
        serving_config_path=_resolve_path(project_root, os.getenv("SERVING_CONFIG_PATH", "models/serving_config.json")),
        xgb_predictions_path=_resolve_path(project_root, os.getenv("XGB_PREDICTIONS_PATH", "models/xgboost_test_predictions.csv")),
        fusion_predictions_path=_resolve_path(project_root, os.getenv("FUSION_PREDICTIONS_PATH", "models/fusion_test_predictions.csv")),
        xgb_feature_importance_path=_resolve_path(project_root, os.getenv("XGB_FEATURE_IMPORTANCE_PATH", "plots/xgboost_feature_importance.png")),
        xgb_confusion_matrix_path=_resolve_path(project_root, os.getenv("XGB_CONFUSION_MATRIX_PATH", "plots/xgboost_confusion_matrix.png")),
        fusion_confusion_matrix_path=_resolve_path(project_root, os.getenv("FUSION_CONFUSION_MATRIX_PATH", "plots/fusion_confusion_matrix.png")),
        training_curves_path=_resolve_path(project_root, os.getenv("TRAINING_CURVES_PATH", "plots/fusion_training_curves.png")),
        roc_curve_path=_resolve_path(project_root, os.getenv("ROC_CURVE_PATH", "plots/fusion_roc_curve.png")),
        comparison_plot_path=_resolve_path(project_root, os.getenv("COMPARISON_PLOT_PATH", "plots/model_comparison.png")),
        metrics_plot_path=_resolve_path(project_root, os.getenv("METRICS_PLOT_PATH", "plots/performance_summary.png")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        target_column=os.getenv("TARGET_COLUMN", "LBL_NOK"),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        val_size=float(os.getenv("VAL_SIZE", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        feature_sequence_length=int(os.getenv("FEATURE_SEQUENCE_LENGTH", "512")),
        lstm_sequence_length=int(os.getenv("LSTM_SEQUENCE_LENGTH", "128")),
        batch_size=int(os.getenv("BATCH_SIZE", "32")),
        max_epochs=int(os.getenv("MAX_EPOCHS", "25")),
        patience=int(os.getenv("PATIENCE", "6")),
        learning_rate=float(os.getenv("LEARNING_RATE", "0.001")),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "0.0001")),
        hidden_size=int(os.getenv("HIDDEN_SIZE", "64")),
        lstm_layers=int(os.getenv("LSTM_LAYERS", "2")),
        tabular_hidden_dim=int(os.getenv("TABULAR_HIDDEN_DIM", "128")),
        fusion_hidden_dim=int(os.getenv("FUSION_HIDDEN_DIM", "128")),
        dropout=float(os.getenv("DROPOUT", "0.3")),
        use_attention=os.getenv("USE_ATTENTION", "true").lower() == "true",
    )
