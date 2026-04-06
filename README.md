# Multi-Sensor Defect Detection Baseline

This project contains Phase 1 of a production-ready machine learning pipeline for defect detection using tabular and time-series data stored in Parquet format.

## Setup

Create and activate the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python main.py
```

## Outputs

- `artifacts/xgboost_baseline.json`: trained XGBoost model
- `artifacts/metrics.json`: evaluation metrics and classification report
- `artifacts/test_predictions.csv`: test set predictions and scores
- `artifacts/run_metadata.json`: dataset and feature metadata
- `plots/confusion_matrix.png`: confusion matrix plot
- `plots/feature_importance.png`: feature importance plot
- `plots/performance_metrics.png`: baseline metric comparison plot
- `logs/pipeline.log`: pipeline logs
