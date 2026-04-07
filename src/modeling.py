from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)


def train_xgboost(x_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> XGBClassifier:
    class_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=class_ratio,
        random_state=random_state,
        n_jobs=4,
    )
    model.fit(x_train, y_train)
    LOGGER.info("Finished XGBoost training")
    return model


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = None
    return metrics


def evaluate_xgboost(model: XGBClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray]:
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    metrics = compute_binary_metrics(y_test.to_numpy(), y_pred, y_score)
    LOGGER.info(
        "XGBoost metrics | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%s",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "n/a",
    )
    return metrics, y_score


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.1, 0.9, 81):
        predictions = (y_score >= threshold).astype(int)
        f1 = f1_score(y_true, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def save_metrics(metrics: dict, destination: Path) -> None:
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", destination)


def save_predictions(index: pd.Index, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, destination: Path) -> None:
    pd.DataFrame(
        {
            "index": index.to_list(),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
        }
    ).to_csv(destination, index=False)
    LOGGER.info("Saved predictions to %s", destination)
