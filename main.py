from __future__ import annotations

import json
import logging
import random

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import Config, load_config
from src.data import (
    compute_sequence_length_stats,
    create_split_indices,
    ensure_binary_target,
    identify_columns,
    load_dataset,
    save_split_indices,
    summarize_dataset,
)
from src.deep_learning import (
    FusionClassifier,
    build_dataloader,
    export_onnx,
    export_torchscript,
    predict_fusion_model,
    train_fusion_model,
)
from src.features import (
    build_time_series_features,
    combine_features,
    fit_feature_preprocessor,
    fit_multimodal_preprocessor,
    prepare_tabular_features,
)
from src.logging_utils import configure_logging
from src.modeling import (
    compute_binary_metrics,
    evaluate_xgboost,
    find_best_threshold,
    save_metrics,
    save_predictions,
    train_xgboost,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_performance_summary,
    plot_roc_curve,
    plot_training_history,
)

LOGGER = logging.getLogger("pipeline")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_frame(frame: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    return frame.loc[indices].copy()


def subset_array(array: np.ndarray, frame_index: pd.Index, indices: list[int]) -> np.ndarray:
    position_map = {index: position for position, index in enumerate(frame_index)}
    positions = [position_map[index] for index in indices]
    return array[positions]


def save_run_metadata(config: Config, summary: dict, sequence_stats: dict, comparison_frame: pd.DataFrame) -> None:
    payload = {
        "data_path": str(config.data_path),
        "target_column": config.target_column,
        "feature_sequence_length": config.feature_sequence_length,
        "lstm_sequence_length": config.lstm_sequence_length,
        "dataset_summary": summary,
        "sequence_stats": sequence_stats,
        "model_comparison": comparison_frame.to_dict(orient="records"),
    }
    config.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved run metadata to %s", config.metadata_path)


def main() -> None:
    config = load_config()
    log_path = configure_logging(config.log_level, config.logs_dir)
    LOGGER.info("Starting Phase 2 multimodal pipeline")
    LOGGER.info("Logging to %s", log_path)
    set_seed(config.random_state)

    dataframe = load_dataset(str(config.data_path))
    bundle = identify_columns(dataframe, config.target_column)
    dataset_summary = summarize_dataset(bundle.dataframe, bundle.target_column)
    sequence_stats = compute_sequence_length_stats(bundle.dataframe, bundle.sequence_columns)
    target = ensure_binary_target(bundle.dataframe[bundle.target_column])

    split_indices = create_split_indices(target, config.test_size, config.val_size, config.random_state)
    save_split_indices(split_indices, config.split_indices_path)

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    tabular_frame = prepare_tabular_features(bundle.dataframe, bundle.tabular_columns)
    engineered_features = build_time_series_features(
        bundle.dataframe,
        bundle.sequence_columns,
        config.feature_sequence_length,
    )
    baseline_frame = combine_features(tabular_frame, engineered_features)

    xgb_preprocessor = fit_feature_preprocessor(subset_frame(baseline_frame, train_idx))
    xgb_train = xgb_preprocessor.transform(subset_frame(baseline_frame, train_idx))
    xgb_val = xgb_preprocessor.transform(subset_frame(baseline_frame, val_idx))
    xgb_test = xgb_preprocessor.transform(subset_frame(baseline_frame, test_idx))
    y_train = target.loc[train_idx]
    y_val = target.loc[val_idx]
    y_test = target.loc[test_idx]

    xgb_model = train_xgboost(xgb_train, y_train, config.random_state)
    xgb_model.save_model(config.xgb_model_path)
    joblib.dump(xgb_preprocessor, config.xgb_preprocessor_path)
    LOGGER.info("Saved XGBoost model to %s", config.xgb_model_path)
    LOGGER.info("Saved XGBoost preprocessor to %s", config.xgb_preprocessor_path)
    _, xgb_val_scores = evaluate_xgboost(xgb_model, xgb_val, y_val)
    xgb_threshold, xgb_best_val_f1 = find_best_threshold(y_val.to_numpy(), xgb_val_scores)
    xgb_metrics, xgb_scores = evaluate_xgboost(xgb_model, xgb_test, y_test)
    xgb_predictions = (xgb_scores >= xgb_threshold).astype(int)
    xgb_metrics = compute_binary_metrics(y_test.to_numpy(), xgb_predictions, xgb_scores)
    xgb_metrics["best_threshold"] = xgb_threshold
    xgb_metrics["best_val_f1"] = xgb_best_val_f1
    save_metrics(xgb_metrics, config.xgb_metrics_path)
    save_predictions(y_test.index, y_test.to_numpy(), xgb_predictions, xgb_scores, config.xgb_predictions_path)
    plot_confusion_matrix(xgb_metrics["confusion_matrix"], "XGBoost Confusion Matrix", config.xgb_confusion_matrix_path)
    plot_feature_importance(xgb_model, list(xgb_train.columns), config.xgb_feature_importance_path)

    multimodal_preprocessor = fit_multimodal_preprocessor(
        subset_frame(bundle.dataframe, train_idx),
        bundle.tabular_columns,
        bundle.sequence_columns,
        config.lstm_sequence_length,
        config.target_column,
    )
    multimodal_preprocessor.save(str(config.preprocessor_path))
    LOGGER.info("Saved preprocessor bundle to %s", config.preprocessor_path)

    all_tabular = multimodal_preprocessor.transform_tabular(bundle.dataframe)
    all_sequences = multimodal_preprocessor.transform_sequences(bundle.dataframe)

    train_sequences = subset_array(all_sequences, bundle.dataframe.index, train_idx)
    val_sequences = subset_array(all_sequences, bundle.dataframe.index, val_idx)
    test_sequences = subset_array(all_sequences, bundle.dataframe.index, test_idx)

    train_tabular = subset_array(all_tabular, bundle.dataframe.index, train_idx)
    val_tabular = subset_array(all_tabular, bundle.dataframe.index, val_idx)
    test_tabular = subset_array(all_tabular, bundle.dataframe.index, test_idx)

    y_train_dl = target.loc[train_idx].to_numpy()
    y_val_dl = target.loc[val_idx].to_numpy()
    y_test_dl = target.loc[test_idx].to_numpy()

    train_loader = build_dataloader(train_sequences, train_tabular, y_train_dl, config.batch_size, shuffle=True)
    val_loader = build_dataloader(val_sequences, val_tabular, y_val_dl, config.batch_size, shuffle=False)
    test_loader = build_dataloader(test_sequences, test_tabular, y_test_dl, config.batch_size, shuffle=False)

    fusion_model = FusionClassifier(
        sequence_input_dim=train_sequences.shape[-1],
        tabular_input_dim=train_tabular.shape[-1],
        hidden_size=config.hidden_size,
        lstm_layers=config.lstm_layers,
        tabular_hidden_dim=config.tabular_hidden_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        dropout=config.dropout,
        use_attention=config.use_attention,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_result = train_fusion_model(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        patience=config.patience,
        checkpoint_path=config.fusion_checkpoint_path,
    )

    fusion_val_scores, fusion_val_targets = predict_fusion_model(fusion_model, val_loader, device)
    fusion_threshold, _ = find_best_threshold(fusion_val_targets.astype(int), fusion_val_scores)
    fusion_scores, fusion_targets = predict_fusion_model(fusion_model, test_loader, device)
    fusion_predictions = (fusion_scores >= fusion_threshold).astype(int)
    fusion_metrics = compute_binary_metrics(fusion_targets.astype(int), fusion_predictions, fusion_scores)
    fusion_metrics["best_epoch"] = training_result.best_epoch
    fusion_metrics["best_val_f1"] = training_result.best_val_f1
    fusion_metrics["best_threshold"] = fusion_threshold
    save_metrics(fusion_metrics, config.fusion_metrics_path)
    save_predictions(pd.Index(test_idx), fusion_targets.astype(int), fusion_predictions, fusion_scores, config.fusion_predictions_path)

    export_torchscript(fusion_model, test_sequences, test_tabular, config.fusion_torchscript_path)
    try:
        export_onnx(fusion_model, test_sequences, test_tabular, config.fusion_onnx_path)
    except Exception as exc:
        LOGGER.warning("ONNX export skipped because it failed: %s", exc)

    plot_confusion_matrix(fusion_metrics["confusion_matrix"], "Fusion Model Confusion Matrix", config.fusion_confusion_matrix_path)
    plot_training_history(training_result.history, config.training_curves_path)
    plot_roc_curve(fusion_targets.astype(int), fusion_scores, config.roc_curve_path)

    comparison_frame = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "accuracy": xgb_metrics["accuracy"],
                "precision": xgb_metrics["precision"],
                "recall": xgb_metrics["recall"],
                "f1_score": xgb_metrics["f1_score"],
                "roc_auc": xgb_metrics["roc_auc"] or 0.0,
            },
            {
                "model": "LSTM Fusion",
                "accuracy": fusion_metrics["accuracy"],
                "precision": fusion_metrics["precision"],
                "recall": fusion_metrics["recall"],
                "f1_score": fusion_metrics["f1_score"],
                "roc_auc": fusion_metrics["roc_auc"] or 0.0,
            },
        ]
    )
    comparison_payload = {
        "xgboost": xgb_metrics,
        "fusion": fusion_metrics,
        "fusion_beats_xgboost": bool(fusion_metrics["f1_score"] > xgb_metrics["f1_score"]),
    }
    config.comparison_metrics_path.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved comparison metrics to %s", config.comparison_metrics_path)

    production_model = "xgboost" if xgb_best_val_f1 >= training_result.best_val_f1 else "fusion"
    serving_payload = {
        "selected_model": production_model,
        "selection_basis": "best_validation_f1",
        "feature_sequence_length": config.feature_sequence_length,
        "thresholds": {
            "xgboost": xgb_threshold,
            "fusion": fusion_threshold,
        },
        "validation_f1": {
            "xgboost": xgb_best_val_f1,
            "fusion": training_result.best_val_f1,
        },
        "test_f1": {
            "xgboost": xgb_metrics["f1_score"],
            "fusion": fusion_metrics["f1_score"],
        },
    }
    config.serving_config_path.write_text(json.dumps(serving_payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved serving config to %s", config.serving_config_path)
    plot_model_comparison(comparison_frame, config.comparison_plot_path)
    plot_performance_summary(comparison_frame, config.metrics_plot_path)

    save_run_metadata(config, dataset_summary, sequence_stats, comparison_frame)
    LOGGER.info(
        "Pipeline complete | xgb_f1=%.4f fusion_f1=%.4f fusion_beats_xgb=%s",
        xgb_metrics["f1_score"],
        fusion_metrics["f1_score"],
        comparison_payload["fusion_beats_xgboost"],
    )


if __name__ == "__main__":
    main()
