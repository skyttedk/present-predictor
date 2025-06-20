# Current Context

## Project Status
**Phase**: Addressing Critical Production Prediction Issues
**Last Updated**: June 20, 2025 (Based on ML Expert Letter and current task)

## Current Work Focus
**Resolve Prediction Mismatch**: The primary focus is to fix a critical issue where the CatBoost model's output (trained on aggregated `selection_count`) is misinterpreted in the production prediction logic (`predictor.py`) as a per-employee rate, leading to uniform and incorrect predictions.

**Chosen Approach (Option A - Immediate Fix)**:
- Keep the current CatBoost model.
- Modify the prediction logic in `src/ml/predictor.py` to treat the model's raw output as the total expected quantity for a gift, considering the context of the request (branch code, employee gender counts).
- This involves removing the incorrect scaling by employee ratios and total employee counts as currently implemented in `_aggregate_predictions()`.

## Recent Changes (Reflects Implemented Fixes as of June 20, 2025)
- An ML expert review identified a fundamental mismatch between model training (target is total `selection_count`) and prediction logic (assumes per-employee rate).
- **Implemented Option A Fix (June 20, 2025):** Modified `src/ml/predictor.py`'s `_aggregate_predictions()` method to sum raw model outputs directly, treating them as total expected quantities for their specific context, removing incorrect per-employee scaling.
- An arbitrary `scaling_factor` (previously 0.15) in `predictor.py` was already set to 1.0. The core logic change in Option A addresses the underlying scaling issue.
- Feature engineering discrepancies (default values, NaN handling) were reportedly fixed prior to the expert's letter.
- The system is currently using a single-stage CatBoost Poisson regressor.
- **Updated `brief.md` (June 20, 2025):** Corrected ML model from XGBoost to CatBoost.

## Next Steps
1.  **Thoroughly Test the Implemented Fix (Option A)**: Validate with various scenarios to ensure predictions are varied, sensible, and the uniform prediction issue is resolved.
2.  **Re-evaluate Two-Stage Model**: Post-fix and testing, assess if a two-stage architecture is needed to better handle zero-inflation or improve predictions.
3.  **Improve Confidence Scores**: Develop a more statistically sound method for confidence estimation.
4.  **Investigate Data for Option B**: In parallel, continue investigating the feasibility of obtaining historical employee counts per group to enable training a model on per-employee selection rates (long-term preferred solution).
5.  **Reconcile Dependency Discrepancies**: Align `pyproject.toml` and `requirements.txt` regarding `catboost` vs `xgboost`.