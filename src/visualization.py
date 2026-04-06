from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def plot_confusion_matrix(confusion: list[list[int]], destination: Path) -> None:
    matrix = np.array(confusion)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()
    LOGGER.info("Saved confusion matrix plot to %s", destination)


def plot_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    destination: Path,
    top_k: int = 20,
) -> None:
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_k)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importance.values, y=importance.index, hue=importance.index, palette="viridis", legend=False)
    plt.title(f"Top {top_k} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()
    LOGGER.info("Saved feature importance plot to %s", destination)


def plot_performance_metrics(metrics: dict, destination: Path) -> None:
    metric_frame = pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "recall", "f1_score"],
            "value": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
            ],
        }
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(data=metric_frame, x="metric", y="value", hue="metric", palette="mako", legend=False)
    plt.ylim(0, 1)
    plt.title("Baseline Performance Metrics")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()
    LOGGER.info("Saved performance metrics plot to %s", destination)
