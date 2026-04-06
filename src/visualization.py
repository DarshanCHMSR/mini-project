from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def plot_confusion_matrix(confusion: list[list[int]], title: str, destination: Path) -> None:
    matrix = np.array(confusion)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()
    LOGGER.info("Saved confusion matrix plot to %s", destination)


def plot_feature_importance(model: XGBClassifier, feature_names: list[str], destination: Path, top_k: int = 20) -> None:
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_k)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importance.values, y=importance.index, hue=importance.index, palette="viridis", legend=False)
    plt.title(f"Top {top_k} XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()
    LOGGER.info("Saved feature importance plot to %s", destination)


def plot_training_history(history: dict[str, list[float]], destination: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    axes[1].plot(epochs, history["train_f1"], label="Train F1", linestyle="--", linewidth=2)
    axes[1].plot(epochs, history["val_f1"], label="Validation F1", linestyle="--", linewidth=2)
    axes[1].set_title("Accuracy and F1 Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved training history plot to %s", destination)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, destination: Path) -> None:
    if len(np.unique(y_true)) < 2:
        LOGGER.warning("Skipping ROC curve because only one class is present.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="Fusion ROC", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()
    LOGGER.info("Saved ROC curve to %s", destination)


def plot_model_comparison(comparison_frame: pd.DataFrame, destination: Path) -> None:
    melted = comparison_frame.melt(id_vars="model", var_name="metric", value_name="value")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="model")
    plt.ylim(0, 1.05)
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()
    LOGGER.info("Saved model comparison plot to %s", destination)


def plot_performance_summary(comparison_frame: pd.DataFrame, destination: Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.heatmap(comparison_frame.set_index("model"), annot=True, fmt=".4f", cmap="YlGnBu", cbar=False)
    plt.title("Performance Summary")
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()
    LOGGER.info("Saved performance summary heatmap to %s", destination)
