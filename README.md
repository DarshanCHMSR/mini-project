# Multi-Sensor Defect Detection System

End-to-end machine learning system for industrial defect detection using mixed-modality manufacturing data.

The project includes:

- a full training pipeline with two model families (XGBoost baseline and LSTM fusion)
- reproducible preprocessing and artifact export
- FastAPI inference service for model serving
- React dashboard for model metrics, plots, and live prediction

## Overview

The dataset is expected to contain:

- tabular process columns, typically prefixed with `SET_`, `QUA_`, `ENV_`, `CALC_`
- sequence sensor columns, prefixed with `DXP_`
- label columns, prefixed with `LBL_`

By default, the binary target is `LBL_NOK`:

- `0`: non-defective
- `1`: defective

## Key Capabilities

- Automatic schema discovery for tabular, sequence, and label fields
- Stratified train/validation/test splitting
- Baseline model with engineered sequence statistics + XGBoost
- Deep model with multimodal fusion (BiLSTM sequence branch + tabular branch)
- Export to TorchScript and ONNX
- Saved plots, metrics, predictions, split indices, and run metadata
- API endpoints for health, schema, run summary, and prediction

## Repository Structure

```text
.
|-- api/
|   |-- app.py
|-- data/
|   |-- processed/
|-- frontend/
|   |-- src/
|-- logs/
|-- models/
|-- plots/
|-- src/
|   |-- config.py
|   |-- data.py
|   |-- deep_learning.py
|   |-- features.py
|   |-- inference.py
|   |-- logging_utils.py
|   |-- modeling.py
|   |-- visualization.py
|-- Dockerfile
|-- main.py
|-- requirements.txt
```

## Tech Stack

- Python 3.11
- PyTorch, XGBoost, scikit-learn, pandas, NumPy
- FastAPI + Uvicorn
- React + Vite + Recharts
- Docker (optional runtime packaging)

## Local Setup

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root (or set equivalent system variables).

Recommended minimum:

```env
DATA_PATH=dataset_V2.parquet
TARGET_COLUMN=LBL_NOK
TEST_SIZE=0.2
VAL_SIZE=0.2
RANDOM_STATE=42
FEATURE_SEQUENCE_LENGTH=512
LSTM_SEQUENCE_LENGTH=128
BATCH_SIZE=32
MAX_EPOCHS=25
PATIENCE=6
LEARNING_RATE=0.001
```

All output paths and advanced hyperparameters can also be overridden via environment variables in [src/config.py](./src/config.py).

## Training Pipeline

Run the full pipeline:

```powershell
python main.py
```

Pipeline outputs include:

- trained models in `models/`
- evaluation metrics JSON files
- test prediction CSV files
- visualizations in `plots/`
- split indices in `data/processed/split_indices.json`
- run metadata in `models/run_metadata.json`

## Model Architecture

### 1. Baseline model

- Resamples each `DXP_*` sequence to a fixed engineering length
- Extracts statistical features per sequence channel
- Concatenates with cleaned tabular features
- Trains an XGBoost binary classifier

### 2. Fusion model

- Sequence branch: bidirectional LSTM (+ optional attention pooling)
- Tabular branch: dense encoder with normalization/dropout
- Fusion head: concatenated representation to binary logit

Training uses class weighting, early stopping, and LR scheduling.

## API

Start API server:

```powershell
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Available endpoints:

- `GET /health`
- `GET /artifacts/schema`
- `GET /artifacts/summary`
- `POST /predict`

Example prediction request:

```json
{
  "tabular": {
    "SET_CylinderTemperature": 200.0,
    "QUA_CycleTime": 58.0,
    "ENV_AirTemperature": 22.0
  },
  "sequences": {
    "DXP_Inj1PrsAct": [0.1, 0.2, 0.3],
    "DXP_Inj1VelAct": [0.0, 0.0, 0.1]
  }
}
```

Example response:

```json
{
  "predicted_label": 1,
  "defect_probability": 0.9683,
  "threshold": 0.5
}
```

Inference behavior is resilient to partial payloads:

- missing tabular values are imputed
- missing sequence fields are zero-filled
- sequence lengths are normalized by the saved preprocessor

## Frontend Dashboard

From the `frontend/` directory:

```powershell
npm install
npm run dev
```

Optional frontend environment variable:

```env
VITE_API_BASE_URL=http://localhost:8000
```

The frontend consumes backend endpoints for health, schema, summary, and prediction.

## Docker

Build image:

```powershell
docker build -t defect-detection-api .
```

Run container:

```powershell
docker run -p 8000:8000 defect-detection-api
```

## Current Saved Results

The repository currently includes saved evaluation artifacts where:

- XGBoost baseline scores are perfect on the saved split
- LSTM fusion model performs strongly but does not exceed the saved XGBoost F1 score

Treat these as split-specific results, not guaranteed production performance.

## Operational Notes

- Keep training and inference preprocessing aligned by always using `models/preprocessor_bundle.joblib`.
- Re-run training after changing sequence lengths, feature logic, or target column.
- For robust comparison, prefer cross-validation and leakage checks over a single split.

## Quick Start

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
