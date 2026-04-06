from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> XGBClassifier:
    class_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = XGBClassifier(
        n_estimators=300,
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


def evaluate_model(model: XGBClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    LOGGER.info(
        "Evaluation metrics | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )
    return metrics


def save_metrics(metrics: dict, destination: Path) -> None:
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", destination)


def save_predictions(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    destination: Path,
) -> None:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)
    pd.DataFrame(
        {
            "index": x_test.index,
            "y_true": y_test.values,
            "y_pred": predictions,
            "y_score": probabilities,
        }
    ).to_csv(destination, index=False)
    LOGGER.info("Saved test predictions to %s", destination)
