# Smoking Status Prediction (Data Mining Course Competition)

An XGBoost-based binary classification project for predicting smoking status
from health examination and bio-signal data.

This project was developed for an individual data mining course competition
restricted to logistic-regression-based and decision-tree-based models.
Submissions were evaluated using ROC-AUC.

## Results

- **Private leaderboard rank:** 25 / 84 individual participants
- **Percentile:** Top 30%
- **Final model:** XGBoost
- **Validation:** 5-fold Stratified Cross-Validation
- **Hyperparameter tuning:** Bayesian Optimization with 70 search iterations
- **ROC-AUC:** 0.8857 *(score source to be labeled as CV, public, or private)*

## Dataset

The task is to predict the binary target `smoking` from health examination
features, including:

- Age and body measurements
- Eyesight and hearing
- Blood pressure
- Blood sugar and lipid indicators
- Hemoglobin and kidney-function indicators
- AST, ALT, and GTP
- Urine protein
- Dental caries

Competition data are not redistributed in this repository.

## Feature Engineering

The original solution included:

- Replacing eyesight values greater than 9 with 0.0 as a special-value treatment
- Reorganizing left and right eyesight measurements using their minimum and maximum
- Reorganizing left and right hearing measurements using their minimum and maximum
- Creating BMI from height and weight
- Creating a systolic-to-diastolic blood-pressure ratio
- Creating an ALT-to-AST liver-function ratio
- Creating an LDL-to-HDL cholesterol ratio
- Applying `log1p` to non-negative numerical features with skewness greater than 1
- Applying `RobustScaler` to numerical features
- Applying one-hot encoding to hearing and dental-caries features
- Applying integer encoding to urine-protein levels
- Adding frequency encoding for age groups
- Removing one feature from highly correlated pairs using an absolute correlation
  threshold of 0.8

## Model Training

The final estimator was `XGBClassifier`.

Hyperparameters were optimized using `BayesSearchCV` with:

- 5-fold `StratifiedKFold`
- 70 Bayesian search iterations
- ROC-AUC scoring
- Parallel execution with `n_jobs=-1`
- Fixed random seed of 42

The search space included:

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

## Model and Feature Comparisons

The original project report compared:

- XGBoost and LightGBM
- All features
- Top 15 SHAP-ranked features
- An automatically selected SHAP-based feature subset

The archived Python script contains the final XGBoost workflow.
The LightGBM and SHAP experiments will be added after the original experimental
notebooks are recovered or reproduced.

## Validation Notes

The archived course solution performs numerical scaling, frequency encoding,
and correlation-based feature filtering before cross-validation.

Although these transformations do not directly use the target label, fitting
them on the complete training set allows validation-fold distribution
information to influence preprocessing.

The refactored version of this project will move all stateful preprocessing
inside the cross-validation pipeline.

## Repository Status

This repository distinguishes between:

- `archive/`: the original course submission
- `src/`: the refactored, reproducible implementation
- `notebooks/`: exploratory analysis, model comparison, and SHAP analysis
- `reports/`: cross-validation results and figures

## Limitations

- The original preprocessing workflow was not fully isolated within each
  cross-validation fold.
- The benefit of scaling and correlation filtering for XGBoost requires
  ablation testing.
- SHAP values describe model behavior rather than causal relationships.
- Competition performance does not establish clinical validity.
- This project is for educational purposes and should not be used for medical
  diagnosis.
