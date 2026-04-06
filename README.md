# Multi-Sensor Defect Detection Pipeline

This project now includes:

- Phase 1 baseline with engineered `DXP_*` features and XGBoost
- Phase 2 multimodal deep learning with LSTM fusion over time-series and tabular process data
- Exported inference artifacts plus a FastAPI `/predict` endpoint

## Folder Structure

```text
/project
|-- api/
|-- data/
|   |-- processed/
|-- models/
|-- plots/
|-- logs/
|-- src/
|-- .env
|-- requirements.txt
|-- Dockerfile
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train End to End

```powershell
python main.py
```

## Run API

```powershell
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Example Request

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

Missing fields are handled automatically. Missing tabular values are imputed and missing sequences are zero-filled before resizing and normalization.

## Outputs

- `models/xgboost_baseline.json`
- `models/fusion_best.pt`
- `models/fusion_model.ts`
- `models/fusion_model.onnx` when ONNX export succeeds
- `models/preprocessor_bundle.joblib`
- `models/xgboost_metrics.json`
- `models/fusion_metrics.json`
- `models/comparison_metrics.json`
- `plots/xgboost_feature_importance.png`
- `plots/xgboost_confusion_matrix.png`
- `plots/fusion_confusion_matrix.png`
- `plots/fusion_training_curves.png`
- `plots/fusion_roc_curve.png`
- `plots/model_comparison.png`
- `plots/performance_summary.png`
- `logs/pipeline.log`
