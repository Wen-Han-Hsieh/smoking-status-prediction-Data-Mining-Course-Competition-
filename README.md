# Smoking Status Prediction

A binary classification project for predicting smoking status from health
examination and bio-signal data using gradient-boosted decision trees.

This project was developed for an individual data mining course competition.
The competition restricted model selection to logistic-regression-based and
decision-tree-based methods and evaluated submissions using ROC-AUC.

## Results

- **Private leaderboard ROC-AUC:** 0.8857
- **Private leaderboard rank:** 25 / 84 individual participants
- **Percentile:** Top 30%
- **Final submitted model:** XGBoost
- **Local model selection:** 5-fold Stratified Cross-Validation
- **Hyperparameter optimization:** BayesSearchCV with 70 search iterations

> The ROC-AUC of 0.8857 is the Kaggle private leaderboard score, not the local
> cross-validation score.

## Problem Description

The objective was to predict the binary target `smoking` from health
examination and bio-signal features, including:

- Age, height, weight, and waist circumference
- Left and right eyesight
- Left and right hearing
- Systolic and diastolic blood pressure
- Fasting blood sugar
- Cholesterol, triglyceride, HDL, and LDL
- Hemoglobin and serum creatinine
- Urine protein
- AST, ALT, and GTP
- Dental caries

The final submission contained the predicted probability of the positive
smoking class for each test-set ID.

## Methodology

### Feature Engineering

The original solution included:

- Reorganizing bilateral eyesight and hearing measurements using their minimum
  and maximum values
- Creating BMI from height and weight
- Creating a systolic-to-diastolic blood-pressure ratio
- Creating an ALT-to-AST liver-function ratio
- Creating an LDL-to-HDL cholesterol ratio
- Adding frequency encoding for five-year age groups
- Applying `log1p` transformation to right-skewed numerical features
- Applying one-hot encoding to selected categorical features
- Applying integer encoding to urine-protein levels
- Applying `RobustScaler` to numerical features
- Filtering feature pairs with an absolute Pearson correlation greater than
  0.8

The log-transformed features recorded in the original experiment included:

- Fasting blood sugar
- Triglyceride
- AST
- ALT
- GTP
- ALT-to-AST ratio

### Model Training

XGBoost was selected as the final model to capture nonlinear relationships and
interactions among health indicators.

The model-selection process used:

- 5-fold `StratifiedKFold`
- ROC-AUC scoring
- 70 Bayesian search iterations
- A fixed random seed of 42

The hyperparameter search covered:

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

The final test-set predictions were generated using positive-class
probabilities from `predict_proba`.

### Model Comparison

The main experiments compared:

- XGBoost and LightGBM
- The complete feature set
- SHAP-ranked feature subsets
- A SHAP top-15 feature subset

CatBoost and soft-voting approaches were also explored during development but
were not used in the final submission.

### SHAP Analysis

SHAP TreeExplainer was used to estimate global feature importance by averaging
the absolute SHAP values across samples.

The resulting ranking was used to compare the complete feature set with
reduced feature subsets.

SHAP values were used to explain model behavior and should not be interpreted
as evidence of causal relationships.

## Repository Structure

```text
smoking-status-prediction/
├── README.md
├── archive/
│   ├── README.md
│   ├── original_colab_notebook.ipynb
│   └── original_submission.py
└── report/
    └── course_report.pdf
```

## Project Materials

### Original Colab Notebook

archive/original_colab_notebook.ipynb

The original Google Colab notebook contains:

- Exploratory data analysis
- Feature engineering
- Numerical and categorical preprocessing
- Bayesian hyperparameter optimization
- Model comparisons
- SHAP-based feature analysis
- ROC-AUC evaluation
- Submission generation

### Submission Script

archive/original_submission.py

This script corresponds most closely to the final XGBoost workflow submitted
for the course competition.

### Course Report

report/course_report.pdf

The original written report describes the model-selection rationale,
feature-engineering process, hyperparameter tuning, model evaluation, and
project reflection.

## Repository Notes

The `archive/` directory preserves the original implementation and experiment
history from the course competition.

The notebook was developed iteratively in Google Colab and may contain repeated
cells, intermediate attempts, environment-specific Google Drive paths, and
selected execution errors. It is retained to document the original development
process rather than presented as a production-ready software pipeline.

## Limitations

- The original workflow was developed in Google Colab and contains
  environment-specific paths.
- The notebook includes multiple experimental versions and may not run as a
  clean standalone pipeline without modification.
- Some preprocessing steps were fitted before cross-validation in the original
  implementation.
- Numerical scaling and correlation-based filtering were not isolated through
  complete ablation experiments.
- SHAP values explain model predictions but do not establish causality.
- Performance on a private competition test set does not establish clinical
  validity or generalization to other populations.

## Data and Usage Notice

The dataset was provided for a private course competition and is not included
or redistributed in this repository.

This project is shared for portfolio and educational purposes only. It should
not be used for medical diagnosis or clinical decision-making.
