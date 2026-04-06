from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import (
    compute_sequence_length_stats,
    ensure_binary_target,
    identify_columns,
    load_dataset,
    summarize_dataset,
)
from src.features import (
    build_time_series_features,
    combine_features,
    prepare_tabular_features,
    split_and_scale_features,
)
from src.logging_utils import configure_logging
from src.modeling import evaluate_model, save_metrics, save_predictions, train_xgboost
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_performance_metrics,
)

LOGGER = logging.getLogger("pipeline")


def save_run_metadata(
    destination: Path,
    config,
    dataset_summary: dict,
    sequence_stats: dict,
    feature_shape: tuple[int, int],
) -> None:
    payload = {
        "data_path": str(config.data_path),
        "target_column": config.target_column,
        "sequence_length": config.sequence_length,
        "feature_shape": {"rows": feature_shape[0], "columns": feature_shape[1]},
        "dataset_summary": dataset_summary,
        "sequence_stats": sequence_stats,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    config = load_config()
    log_path = configure_logging(config.log_level, config.logs_dir)
    LOGGER.info("Starting defect detection baseline pipeline")
    LOGGER.info("Logging to %s", log_path)

    dataframe = load_dataset(str(config.data_path))
    bundle = identify_columns(dataframe, config.target_column)
    dataset_summary = summarize_dataset(bundle.dataframe, bundle.target_column)
    sequence_stats = compute_sequence_length_stats(bundle.dataframe, bundle.sequence_columns)

    target = ensure_binary_target(bundle.dataframe[bundle.target_column])
    tabular_features = prepare_tabular_features(bundle.dataframe, bundle.tabular_columns)
    time_series_features = build_time_series_features(
        bundle.dataframe,
        bundle.sequence_columns,
        config.sequence_length,
    )
    features = combine_features(tabular_features, time_series_features)

    x_train, x_test, y_train, y_test, _, _ = split_and_scale_features(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model = train_xgboost(x_train, y_train, random_state=config.random_state)
    model.save_model(config.model_path)
    LOGGER.info("Saved trained model to %s", config.model_path)

    metrics = evaluate_model(model, x_test, y_test)
    save_metrics(metrics, config.metrics_path)
    save_predictions(model, x_test, y_test, config.predictions_path)

    metadata_path = config.model_path.parent / "run_metadata.json"
    save_run_metadata(metadata_path, config, dataset_summary, sequence_stats, features.shape)
    LOGGER.info("Saved run metadata to %s", metadata_path)

    plot_confusion_matrix(metrics["confusion_matrix"], config.confusion_matrix_path)
    plot_feature_importance(model, list(x_train.columns), config.feature_importance_path)
    plot_performance_metrics(metrics, config.metrics_plot_path)

    report_text = pd.DataFrame(metrics["classification_report"]).transpose().round(4).to_string()
    LOGGER.info("Classification report:\n%s", report_text)
    LOGGER.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
