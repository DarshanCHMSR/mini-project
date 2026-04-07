# Model Analysis and Accuracy Improvement Report

Date: 2026-04-07

## 1) Dataset analysis (`dataset_V2.parquet`)

- Shape: 564 rows x 364 columns
- Detected schema:
  - Tabular columns (`SET_`, `QUA_`, `ENV_`, `CALC_`): 58
  - Sequence columns (`DXP_`): 74
  - Label columns (`LBL_`): 8
- Target used: `LBL_NOK`
- Class distribution:
  - Class 0 (non-defective): 409
  - Class 1 (defective): 155
- Sequence length profile (example column `DXP_Inj1PrsAct`):
  - Min: 8041
  - Max: 12079
  - Mean: 10989.26
  - Median: 11387

## 2) Model design and parameter usage

### Baseline model: XGBoost

- Inputs:
  - Numeric tabular features from `SET_`, `QUA_`, `ENV_`, `CALC_`
  - Engineered stats from each `DXP_` signal after resampling to `FEATURE_SEQUENCE_LENGTH`
- Sequence stats per channel:
  - mean, std, min, max, peak_abs, slope
- Preprocessing:
  - Median imputation + standard scaling
- Hyperparameters used:
  - n_estimators=350
  - max_depth=5
  - learning_rate=0.05
  - subsample=0.9
  - colsample_bytree=0.8
  - reg_lambda=1.0
  - scale_pos_weight=neg/pos ratio

### Fusion model: BiLSTM + tabular encoder

- Sequence branch:
  - Bidirectional LSTM (`HIDDEN_SIZE`, `LSTM_LAYERS`)
  - Attention pooling (enabled by `USE_ATTENTION=true`)
- Tabular branch:
  - Dense layers with BatchNorm + Dropout
- Fusion head:
  - Concatenation of sequence + tabular embeddings
  - Dense classifier to binary logit
- Training behavior:
  - BCEWithLogitsLoss with class weighting (`pos_weight`)
  - AdamW (`LEARNING_RATE`, `WEIGHT_DECAY`)
  - ReduceLROnPlateau scheduler
  - Early stopping (`PATIENCE`)

## 3) What was improved

1. Added validation-based threshold tuning for both models instead of always using threshold 0.5.
2. Saved XGBoost preprocessor artifact for production inference parity.
3. Added production serving policy (`models/serving_config.json`) selecting best model by validation F1.
4. Updated inference path to load both models and route predictions through selected model.
5. Added inference response field `model_used` for traceability.

## 4) Accuracy proof (after update)

### Test split metrics

- XGBoost:
  - Accuracy: 1.0000
  - Precision: 1.0000
  - Recall: 1.0000
  - F1: 1.0000
  - ROC-AUC: 1.0000
  - Threshold: 0.10

- Fusion:
  - Accuracy: 0.9912
  - Precision: 0.9688
  - Recall: 1.0000
  - F1: 0.9841
  - ROC-AUC: 0.9980
  - Threshold: 0.50

### Production selection

- Selected model: `xgboost`
- Selection rule: best validation F1 (tie breaks to xgboost in current implementation)
- Result: production endpoint now serves the stronger model for this dataset split.

## 5) Production readiness impact

- Inference now follows validated model selection policy rather than hardcoding fusion.
- Thresholds are explicit, reproducible, and stored with artifacts.
- API summary endpoint exposes serving policy for operational visibility.
- New artifacts support stable deployment:
  - `models/xgboost_preprocessor.joblib`
  - `models/serving_config.json`

## 6) Notes

- These results are split-specific and very high; this can indicate easy separability or potential leakage.
- For stronger production confidence, add k-fold cross-validation and temporal/group split validation if applicable.
