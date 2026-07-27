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
- **Local model selection:** 5-fold Stratified Cross-Validation
- **Hyperparameter optimization:** BayesSearchCV with 70 search iterations

> The ROC-AUC of 0.8857 is the Kaggle private leaderboard score, not the local
> cross-validation score.

## Problem Description

The objective was to predict the binary target `smoking` using health
examination and bio-signal features, including:

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
- Filtering feature pairs with an absolute correlation greater than 0.8

The log-transformed features recorded in the original experiment included:

- Fasting blood sugar
- Triglyceride
- AST
- ALT
- GTP
- ALT-to-AST ratio

### Model Training

XGBoost was selected as the final model because it can capture nonlinear
relationships and interactions among health indicators.

The model was tuned using:

- 5-fold `StratifiedKFold`
- ROC-AUC scoring
- 70 Bayesian search iterations
- A fixed random seed of 42

The hyperparameter search included:

- `learning_rate`
- `n_estimators`
- `max_depth`
- `min_child_weight
