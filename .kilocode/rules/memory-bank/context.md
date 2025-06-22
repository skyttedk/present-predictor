# Current Context

## Project Status
**Phase**: Re-architecting (Implementing Expert Recommendations)
**Last Updated**: June 22, 2025

## Current Work Focus
**CRITICAL RE-ARCHITECTURE REQUIRED**: We have received feedback from an ML expert that our current approach is fundamentally flawed. The model's predictions are unreliable (collapsing to zero or uniform values) due to a structural mismatch between the training target and the business prediction need.

**Expert Diagnosis Summary**:
1.  **Target Mismatch**: The model is trained on `selection_count` (a cumulative historical count), but we need to predict a per-session quantity. This is the root cause of the scale mismatch and unreliable predictions.
2.  **Aggregation Flaw**: The current aggregation logic (`sum(prediction * gender_ratio)`) is incorrect because the model's output is not a probability or rate.
3.  **Zero/Uniform Predictions**: This is a symptom of the model being "confused" by the target mismatch, especially for cold-start (unseen) gift combinations.

## Recent Changes
-   **June 22, 2025**:
    -   ✅ Received detailed feedback from ML expert.
    -   ✅ Confirmed that post-prediction normalization is not the correct solution.
    -   ✅ Identified that the core issue is the training target (`selection_count`).
    -   ❌ Previous attempts to fix aggregation logic were incorrect as they didn't address the root cause.

## Next Steps (Immediate Priority)
1.  **Re-engineer the Training Target**:
    -   The most critical step is to change the target variable from `selection_count` to `selection_rate`.
    -   Formula: `selection_rate = selection_count / total_employees_in_that_group`.
    -   **This requires obtaining `total_employees_in_that_group` for each historical data point.** This may be a new data requirement.

2.  **Retrain the Model**:
    -   Retrain the CatBoost model to predict the new `selection_rate` target.
    -   The loss function should be a standard regression loss (like RMSE), not Poisson, as we are now predicting a rate.

3.  **Correct the Prediction Pipeline**:
    -   Update the `predict` and `_aggregate_predictions` methods in [`src/ml/predictor.py`](src/ml/predictor.py:1).
    -   The new logic will be: `predicted_count = predicted_rate * current_employee_count`.
    -   This will be done for each gender subgroup and then summed.

4.  **Update Documentation**:
    -   Update `architecture.md` and `roadmap.md` to reflect the new plan.

This is a significant pivot from the previous approach and is now the highest priority for the project.