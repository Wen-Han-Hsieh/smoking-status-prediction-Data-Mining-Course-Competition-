# Smoking Status Prediction

Predicting smoking status from health examination and bio-signal data using
gradient-boosted decision trees.

This project was developed for an individual data mining course competition.
The competition restricted model selection to logistic-regression-based and
decision-tree-based methods and used ROC-AUC as the evaluation metric.

## Results

- **Private leaderboard ROC-AUC:** 0.8857
- **Private leaderboard rank:** 25 / 84 individual participants
- **Percentile:** Top 30%
- **Final submitted model:** XGBoost
- **Local validation:** 5-fold Stratified Cross-Validation
- **Hyperparameter optimization:** BayesSearchCV with 70 search iterations

## Problem Description

The objective was to predict the binary target `smoking` using health
examination features, including:

- Age and body measurements
- Eyesight and hearing
- Blood pressure
- Blood sugar and lipid indicators
- Hemoglobin and serum creatinine
- Urine protein
- AST, ALT, and GTP
- Dental caries

The final submission contained the predicted probability of the positive
smoking class for each test-set ID.

## Methodology

### Feature Engineering

The original solution included:

- Reorganizing bilateral eyesight and hearing measurements
- Creating BMI from height and weight
- Creating a systolic-to-diastolic blood-pressure ratio
- Creating an ALT-to-AST liver-function ratio
- Creating an LDL-to-HDL cholesterol ratio
- Adding frequency encoding for age groups
- Applying `log1p` transformation to right-skewed numerical features
- Applying categorical encoding and numerical scaling
- Filtering highly correlated features

The log-transformed features in the archived experiment included:

- Fasting blood sugar
- Triglyceride
- AST
- ALT
- GTP
- ALT-to-AST ratio

### Model Training

The final XGBoost model was tuned using:

- 5-fold `StratifiedKFold`
- ROC-AUC scoring
- 70 Bayesian search iterations
- A fixed random seed of 42

The hyperparameter search included:

- `learning_rate`
- `n_estimators`
- `max_depth`
- `min_child_weight`
- `gamma`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`
- `scale_pos_weight`

### SHAP Analysis

SHAP TreeExplainer was used to rank features by mean absolute SHAP values.

The original experiments considered:

- All available features
- SHAP-ranked feature subsets
- A SHAP top-15 feature subset
- XGBoost
- LightGBM
- CatBoost
- A soft-voting ensemble

The final submitted solution used XGBoost rather than the experimental
ensemble.

## Repository Structure

```text
smoking-status-prediction/
├── README.md
├── requirements.txt
├── archive/
│   ├── README.md
│   ├── original_submission.py
│   ├── original_colab_export.py
│   └── original_colab_execution.pdf
├── reports/
│   └── course_report.pdf
└── data/
    └── README.md
```

## Archived Materials

The `archive/` directory preserves the original course submission and Colab
experiments.

These files contain exploratory code, repeated notebook cells,
environment-specific paths, and intermediate model experiments. They are
retained to document the original development process and are not presented as
a production-ready pipeline.

## Dataset Availability

The competition dataset is not redistributed in this repository.

See data/README.md for the expected local data structure.

## Current Status

This repository currently preserves and documents the original competition
work.

Future improvements may include:

- Separating EDA, feature engineering, training, and prediction
- Replacing Google Drive paths with project-relative paths
- Creating a reproducible preprocessing and training pipeline
- Reproducing SHAP analysis in a clean notebook
- Recording local cross-validation results
- Adding submission-file validation

## Limitations

- The archived code was developed in Google Colab and contains
  environment-specific paths.
- The raw Colab export contains multiple experimental versions and is not
  intended to run as one production script.
- Some preprocessing steps were performed before cross-validation.
- SHAP feature importance describes model behavior and does not establish
  causal relationships.
- Competition performance does not establish clinical validity.

This project is intended for educational and portfolio purposes and should not
be used for medical diagnosis.

## License

The competition dataset remains subject to the terms of its original provider
and is not redistributed in this repository.
