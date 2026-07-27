# Original Course Materials

This directory preserves the original implementation developed during the
data mining course competition.

## Files

### `original_colab_notebook.ipynb`

The original Google Colab notebook containing:

- Exploratory data analysis
- Feature engineering
- Numerical and categorical preprocessing
- XGBoost hyperparameter optimization
- LightGBM and CatBoost experiments
- SHAP-based feature analysis
- ROC-AUC evaluation
- Submission generation

The notebook preserves the original development process, including repeated
cells, intermediate experiments, execution errors, and Google Colab-specific
paths. It is provided as an experiment record rather than a production-ready
pipeline.

### `original_submission.py`

The script corresponding most closely to the final XGBoost submission.

It contains the main feature-engineering, five-fold stratified
cross-validation, and Bayesian hyperparameter-optimization workflow.
