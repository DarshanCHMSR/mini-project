# Multi-Sensor Defect Detection System

End-to-end multimodal defect detection project using tabular process data + long sequence sensor traces.

This repository includes:

- model training pipeline (XGBoost baseline + BiLSTM fusion)
- artifact export and production-serving policy generation
- FastAPI inference service
- React dashboard for metrics and visualization

## 1) Current System Status

This repo has been upgraded with:

- validation-tuned decision thresholds per model (not fixed 0.5)
- model-selection policy for production (`models/serving_config.json`)
- inference routing to selected model (`xgboost` or `fusion`)
- leakage/overfitting diagnostics artifacts in `artifacts/`

Current saved results are strong on random split, but group-aware validation shows harder generalization. Use the anti-leakage sections below before claiming production confidence.

## 2) Data and Target

Expected schema:

- tabular columns: `SET_*`, `QUA_*`, `ENV_*`, `CALC_*`
- sequence columns: `DXP_*`
- labels: `LBL_*`

Default binary target:

- `LBL_NOK = 0`: non-defective
- `LBL_NOK = 1`: defective

Current dataset snapshot (`dataset_V2.parquet`):

- rows: 564
- columns: 364
- tabular columns: 58
- sequence channels: 74

## 3) End-to-End Pipeline (How Model Works)

Main entrypoint: `main.py`

### Stage A: Load and Schema Detection

- Reads parquet dataset
- Detects tabular, sequence, and label columns
- Validates binary target
- Builds stratified train/val/test indices

Code path:

- `src/data.py`

Artifacts:

- `data/processed/split_indices.json`

### Stage B: Baseline Features + XGBoost

- Tabular cleaning: numeric coercion, median imputation, standardization
- Sequence engineering: each `DXP_*` signal is resampled to `FEATURE_SEQUENCE_LENGTH` and summarized by:
  - mean, std, min, max, peak_abs, slope
- Combined feature matrix feeds XGBoost classifier

Code path:

- `src/features.py`
- `src/modeling.py`

Artifacts:

- `models/xgboost_baseline.json`
- `models/xgboost_preprocessor.joblib`
- `models/xgboost_metrics.json`
- `models/xgboost_test_predictions.csv`

### Stage C: Fusion BiLSTM Model

- Sequence branch:
  - bidirectional LSTM
  - attention pooling (if enabled)
- Tabular branch:
  - dense layers + batch norm + dropout
- Fusion head:
  - concat(sequence_embedding, tabular_embedding) -> binary logit
- Training:
  - BCEWithLogits + class weighting
  - AdamW + ReduceLROnPlateau
  - early stopping by validation performance

Code path:

- `src/deep_learning.py`
- `src/features.py`

Artifacts:

- `models/fusion_best.pt`
- `models/fusion_model.ts`
- `models/fusion_model.onnx`
- `models/fusion_metrics.json`
- `models/fusion_test_predictions.csv`

### Stage D: Threshold Tuning + Production Selection

- Finds F1-optimal threshold per model on validation split
- Compares models by validation F1
- Writes serving policy consumed by API

Serving policy artifact:

- `models/serving_config.json`

Contents include:

- selected production model
- per-model thresholds
- validation and test metric snapshot

### Stage E: Report and Plot Generation

- confusion matrices
- ROC curve
- training curves
- feature importance
- model comparison and summary heatmap

Plot outputs (in `plots/`):

- `xgboost_confusion_matrix.png`
- `fusion_confusion_matrix.png`
- `fusion_roc_curve.png`
- `fusion_training_curves.png`
- `xgboost_feature_importance.png`
- `model_comparison.png`
- `performance_summary.png`

## 4) Graphs and How to Read Them

You can view these directly from the filesystem or through the API-mounted route (`/plots/...`).

### Confusion Matrices

- `plots/xgboost_confusion_matrix.png`
- `plots/fusion_confusion_matrix.png`

Use to inspect false positives/false negatives. In production, false negatives are often the highest business risk.

### ROC Curve

- `plots/fusion_roc_curve.png`

Use to understand threshold sensitivity. AUC near 1.0 means good ranking, but not necessarily perfect deployment behavior.

### Training Curves

- `plots/fusion_training_curves.png`

Use to detect optimization instability and early stopping behavior.

### Feature Importance (XGBoost)

- `plots/xgboost_feature_importance.png`

Use to identify dominant engineered features and potential process levers.

### Model Comparison

- `plots/model_comparison.png`
- `plots/performance_summary.png`

Use to compare model family behavior across accuracy, precision, recall, F1, and AUC.

## 5) Anti-Leakage and Overfitting Validation

Additional analysis artifacts:

- `artifacts/model_analysis_report.md`
- `artifacts/overfitting_check_2026-04-07.json`
- `artifacts/anti_leakage_validation_2026-04-07.json`
- `artifacts/anti_leakage_ablation_2026-04-07.json`

Key findings:

- random split remains near-perfect
- label-shuffle sanity check drops strongly (good sign)
- exact row overlap leakage not found
- group-aware validation can drop sharply for some group columns

Conclusion: random-split metrics are optimistic. For real production confidence, prefer group-aware or process-aware validation.

## 6) API Service (FastAPI)

Start API:

```powershell
& "c:/Users/Darsh/OneDrive/Desktop/full stack/mini-project/.venv/Scripts/python.exe" -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `GET /artifacts/schema`
- `GET /artifacts/summary`
- `POST /predict`

`/predict` returns:

- `predicted_label`
- `defect_probability`
- `threshold`
- `model_used`

## 7) Why FastAPI Can "Stop Automatically"

The API itself is stable in current code and was verified healthy (`/health` returns `status=ok`).

Most common reasons it appears to stop:

1. Started in a terminal session that gets closed or reused.
2. Started via a non-background command, then interrupted by another run.
3. Auto-reload mode restarts due to file changes (if using `--reload`).
4. Startup failure due to missing artifacts after clean workspace.

Recommended run pattern:

- use a dedicated terminal for API only
- keep training and API terminals separate
- verify artifacts exist before startup

Quick health check:

```powershell
curl http://127.0.0.1:8000/health
```

## 8) Setup and Run

### Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Train pipeline

```powershell
python main.py
```

### Run API

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### Run frontend

```powershell
cd frontend
npm install
npm run dev
```

## 9) Important Production Notes

- Keep preprocessors and models version-locked with each run.
- Re-run full pipeline after changing feature engineering, sequence length, or target definition.
- Prefer group-aware validation for go-live decisions.
- Keep `models/serving_config.json` as the source of truth for serving model and threshold.
