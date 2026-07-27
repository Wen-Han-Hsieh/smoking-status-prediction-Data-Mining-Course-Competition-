# Archived Course Materials

This directory preserves the original materials created during the data
mining course competition.

## Files

### `original_submission.py`

The script corresponding most closely to the final XGBoost submission.

It contains:

- Feature engineering
- Numerical and categorical preprocessing
- Correlation-based feature filtering
- XGBoost training
- 5-fold stratified cross-validation
- Bayesian hyperparameter optimization

### `original_colab_export.py`

A raw Python export of the original Google Colab notebook.

It contains multiple experimental versions, including:

- Exploratory data analysis
- XGBoost
- LightGBM
- CatBoost
- SHAP feature importance
- Soft-voting experiments

The file also contains repeated notebook cells, Colab installation commands,
hard-coded Google Drive paths, and intermediate experiments. It is preserved
for historical reference and is not intended to be executed as the main
project pipeline.

### `original_colab_execution.pdf`

A PDF export of the original notebook execution history. It contains selected
outputs, plots, error messages, model experiments, and submission-generation
code.

## Note

These files represent the original course development process. A cleaned and
reproducible implementation may be added separately in the future.
