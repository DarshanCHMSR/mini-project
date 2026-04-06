# Multi-Sensor Defect Detection Pipeline

This project is a production-style machine learning pipeline for defect detection using a Parquet dataset that contains:

- tabular process features such as `SET_*`, `QUA_*`, `ENV_*`, and `CALC_*`
- time-series sensor signals stored inside `DXP_*` columns
- defect labels stored in `LBL_*` columns

The current target used by the pipeline is `LBL_NOK`, which is treated as a binary defect label:

- `0` = OK part
- `1` = defective part

The system has two modeling stages:

- Phase 1 baseline: feature engineering over sensor sequences plus XGBoost
- Phase 2 advanced model: LSTM-based multi-modal fusion of time-series and tabular data

It also includes:

- training logs with ETA
- saved evaluation plots
- saved preprocessing artifacts
- TorchScript and ONNX model export
- a FastAPI inference service

## What The Pipeline Does

At a high level, the pipeline does the following when you run [main.py](./main.py):

1. Loads the Parquet dataset from the path defined in [.env](./.env)
2. Detects the important column groups automatically
3. Splits the data into train, validation, and test sets using stratification
4. Builds a Phase 1 baseline using engineered features and XGBoost
5. Builds a Phase 2 fusion model using:
   - an LSTM branch for `DXP_*` sensor sequences
   - a dense branch for tabular process features
   - a fusion head for final classification
6. Evaluates both models on the same test split
7. Saves metrics, predictions, plots, trained models, and preprocessing artifacts
8. Exposes the deep learning model through a FastAPI `/predict` endpoint

## Dataset Assumptions

The code is written around the structure observed in `dataset_V2.parquet`.

### Column groups

- Time-series columns: `DXP_*`
- Tabular columns: `SET_*`, `QUA_*`, `ENV_*`, `CALC_*`
- Labels: `LBL_*`

### Important notes

- Each row represents one manufacturing cycle or part
- Each `DXP_*` cell contains a sequence, typically a NumPy array
- Sequence lengths are not identical across rows, so they must be resized before modeling
- Image-related columns are not used in this project

## Project Structure

```text
/project
|-- api/
|   |-- app.py
|-- data/
|   |-- processed/
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
|-- .env
|-- Dockerfile
|-- main.py
|-- README.md
|-- requirements.txt
```

## Main Components

### [src/config.py](./src/config.py)

Loads environment configuration and defines all important paths and training hyperparameters.

This file controls:

- dataset path
- model output paths
- split sizes
- sequence lengths
- deep learning hyperparameters
- plot and logging locations

### [src/data.py](./src/data.py)

Responsible for:

- loading the Parquet file
- identifying tabular, sequence, and label columns
- validating the binary target
- summarizing missingness and class balance
- creating train, validation, and test split indices

### [src/features.py](./src/features.py)

Responsible for feature preparation for both models.

For Phase 1 it:

- converts tabular values to numeric
- imputes missing tabular values
- rescales sensor sequences to a fixed length
- extracts time-series statistics such as:
  - mean
  - standard deviation
  - minimum
  - maximum
  - absolute peak
  - slope

For Phase 2 it:

- resizes raw `DXP_*` sequences to a fixed LSTM length
- normalizes each sensor channel
- saves preprocessing objects so inference uses the same transformations as training

### [src/modeling.py](./src/modeling.py)

Contains the classical ML stage:

- XGBoost training
- binary evaluation metrics
- metrics and prediction saving

### [src/deep_learning.py](./src/deep_learning.py)

Contains the deep learning system:

- PyTorch dataset and dataloader logic
- LSTM encoder for time-series signals
- dense encoder for tabular input
- fusion classifier head
- attention pooling over LSTM output
- early stopping
- learning rate scheduling
- model checkpointing
- TorchScript export
- ONNX export

### [src/inference.py](./src/inference.py)

Loads the saved preprocessor and TorchScript model and provides a single prediction function for deployment use.

### [src/visualization.py](./src/visualization.py)

Generates:

- confusion matrices
- XGBoost feature importance chart
- training and validation curves
- ROC curve
- model comparison chart
- performance summary heatmap

### [api/app.py](./api/app.py)

Defines the FastAPI service:

- `GET /health`
- `POST /predict`

## How The Models Work

## Phase 1: XGBoost Baseline

The baseline does not directly consume raw sequences. Instead, it summarizes each `DXP_*` signal into engineered statistical features.

### Baseline flow

1. Load tabular columns
2. Resize each `DXP_*` sequence to a fixed feature-engineering length
3. Extract statistics from each resized signal
4. Concatenate engineered time-series features with tabular features
5. Impute missing values
6. Standardize features
7. Train XGBoost on the final feature matrix

### Why this baseline is strong

It gives a simple but powerful benchmark and often works surprisingly well on industrial data because the summary statistics can capture a lot of defect-related signal.

## Phase 2: LSTM Fusion Model

The advanced model keeps more of the raw time-series structure.

### Fusion architecture

The model has three parts.

#### 1. Sequence branch

- input shape: `(sequence_length, number_of_dxp_channels)`
- processed by a bidirectional LSTM
- optionally pooled using attention
- converted into a compact learned sequence representation

#### 2. Tabular branch

- input shape: `(number_of_tabular_features,)`
- processed by dense layers
- uses batch normalization and dropout
- learns a tabular representation

#### 3. Fusion head

- concatenates sequence and tabular representations
- applies dense layers
- outputs a final binary defect score

### Training behavior

The deep learning training loop includes:

- binary cross-entropy with class weighting
- AdamW optimizer
- `ReduceLROnPlateau` scheduler
- early stopping based on validation performance
- checkpoint saving for the best model
- per-epoch ETA logging

## Data Flow End To End

When the pipeline runs, the data moves through these stages:

1. Raw Parquet file is loaded
2. Column groups are detected
3. Split indices are created and saved
4. Baseline features are created for XGBoost
5. A multimodal preprocessor is fitted on training data only
6. Raw tabular and sequence inputs are transformed consistently for train, validation, and test
7. XGBoost is trained and evaluated
8. LSTM fusion model is trained and evaluated
9. Artifacts are exported for deployment

## Setup

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Configuration

Project configuration is stored in [.env](./.env).

### Important variables

- `DATA_PATH`: path to the Parquet file
- `TARGET_COLUMN`: current target column
- `TEST_SIZE`: test split ratio
- `VAL_SIZE`: validation split ratio
- `FEATURE_SEQUENCE_LENGTH`: sequence length used for engineered baseline features
- `LSTM_SEQUENCE_LENGTH`: sequence length used for the LSTM model
- `BATCH_SIZE`: deep learning batch size
- `MAX_EPOCHS`: maximum number of training epochs
- `PATIENCE`: early stopping patience
- `LEARNING_RATE`: learning rate for AdamW

### Output paths

The same `.env` file also defines where the system saves:

- models
- metrics
- predictions
- plots
- split indices
- metadata

## How To Train

Run the full training and evaluation pipeline:

```powershell
python main.py
```

This single command:

- trains the baseline model
- trains the LSTM fusion model
- evaluates both
- saves all outputs
- prepares the API artifacts

## Logging

Logs are written to [logs/pipeline.log](./logs/pipeline.log).

The logger includes:

- timestamp
- log level
- ETA placeholder
- module name
- message

During deep learning training, the log shows:

- epoch number
- train loss
- validation loss
- train accuracy
- validation accuracy
- train F1
- validation F1
- learning rate
- epoch time
- estimated time remaining

## Generated Outputs

### Models

- `models/xgboost_baseline.json`: trained XGBoost baseline
- `models/fusion_best.pt`: PyTorch checkpoint with best validation model
- `models/fusion_model.ts`: TorchScript model for deployment
- `models/fusion_model.onnx`: ONNX export for interoperability
- `models/preprocessor_bundle.joblib`: saved preprocessing pipeline for inference

### Metrics and metadata

- `models/xgboost_metrics.json`
- `models/fusion_metrics.json`
- `models/comparison_metrics.json`
- `models/run_metadata.json`
- `data/processed/split_indices.json`

### Predictions

- `models/xgboost_test_predictions.csv`
- `models/fusion_test_predictions.csv`

### Plots

- `plots/xgboost_feature_importance.png`
- `plots/xgboost_confusion_matrix.png`
- `plots/fusion_confusion_matrix.png`
- `plots/fusion_training_curves.png`
- `plots/fusion_roc_curve.png`
- `plots/model_comparison.png`
- `plots/performance_summary.png`

## Evaluation Outputs Explained

### Confusion matrix

Shows how many parts were correctly and incorrectly classified.

### Training curves

Show whether the fusion model is learning smoothly and whether validation performance is diverging from training performance.

### ROC curve

Shows how well the fusion model separates positive and negative classes across thresholds.

### Model comparison chart

Compares XGBoost and the LSTM fusion model on:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC

## API Usage

Start the API locally:

```powershell
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Health check

```http
GET /health
```

### Prediction endpoint

```http
POST /predict
Content-Type: application/json
```

### Request body

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

### Response body

```json
{
  "predicted_label": 1,
  "defect_probability": 0.9683,
  "threshold": 0.5
}
```

### Missing inputs

The inference layer is robust to partial payloads:

- missing tabular fields are imputed
- missing sequence fields are replaced with zero-filled sequences
- short sequences are resized to the required length

## Docker

You can build the API image with:

```powershell
docker build -t defect-detection-api .
```

Run the container with:

```powershell
docker run -p 8000:8000 defect-detection-api
```

## Current Results

On the currently saved split:

- XGBoost baseline achieved perfect test performance
- LSTM fusion model achieved very strong performance but did not beat the perfect baseline on that split

This means:

- the deep learning system is working correctly end to end
- the comparison pipeline is valid
- the claim that the LSTM model is always better would be misleading on the current data split

## Important Practical Notes

### 1. Perfect baseline scores deserve caution

When a baseline reaches `1.0`, it is worth checking for:

- feature leakage
- duplicate rows or near-duplicates
- easy class separation in the current split
- process-specific correlations that may not generalize

### 2. Training and inference must use the same preprocessing

This project saves the preprocessor bundle intentionally so deployment uses the exact same scaling and normalization learned during training.

### 3. ONNX export warning

The LSTM ONNX export can show a warning about variable batch sizes. The current export is still generated successfully, but if you deploy ONNX broadly, you may want to harden the exported interface further.

## Next Good Improvements

If you want to continue developing this project, the strongest next steps are:

- cross-validation instead of a single split
- leakage checks and duplicate analysis
- threshold tuning based on business cost
- multi-label support for all `LBL_*` targets
- richer sequence encoders such as temporal CNNs or transformers
- experiment tracking with MLflow
- hyperparameter tuning with Optuna

## Quick Start

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

That is enough to train the models, save all artifacts, and serve predictions from the exported fusion model.
