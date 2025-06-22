1. Current Project Overview
1.1. Objective & Application Context
Business Need: Predict per-gift employee selection counts (not normalized probabilities), tailored to company/branch/gender features, for catalog curation and logistics.
Prediction Goal: For an input batch (N gifts, M employees, with gender split), estimate per-gift expected pick counts so that their sum ≈ M, with appropriate discrimination among available products.
1.2. Dataset Characteristics
~178,736 selection events → 98,741 groupwise aggregates.
Features: Employee (shop, branch, gender), product (main/sub-category, brand, color, etc.), shop-level aggregates, and interaction terms.
Data is grouped, not individual-level (i.e., output is aggregated count per feature combination).
Known issues:
Cold starts: Unseen gift/shop/branch combos at prediction time.
Imbalanced feature distribution (e.g., heavy-tail gift popularity).
Shop feature fallbacks (medians).
1.3. Model Architecture, Training, and Infrastructure
Model: CatBoostRegressor, Poisson deviance loss.
Target: selection_count (grouped count), log1p-transformed for stability/training.
Features: 30 total (20 base, 10 interactions).
CV R²: ~0.31 on log-transformed target—a plausible baseline for grouped count prediction.
1.4. Performance Evaluation
Metrics: RMSE, R² on log1p(selection_count).
Validation: Presumably k-fold, but precise data splits not described.
No clear mapping between training/validation outcomes and “live” business metric (sum predicted = #employees) is established.
1.5. Hyperparameter Tuning
Details not specified; CatBoost parameters, categorical encoding, and hash interactions are used.
No mention of tuning via Bayesian/Random search, or targeted metric optimization.
2. Systematic Workflow Analysis
Below is a pointwise diagnosis of structural, conceptual, and tuning issues, mapped to your reported problems:

2.1. Data and Target Mismatch
a. Selection_Count as Target (Cumulative vs. Contextual)
Problem: The model is trained to predict the historical count of selections over an unknown number of employees/events per (shop, branch, gender, product) group.
Implication: At prediction time, the total number of selections across gifts does not necessarily sum (or scale) to the desired number of employees, because:
Historical frequency ≠ per-session probability, unless sample sizes and context are strictly matched.
The model "knows" only the grouped aggregate output, not the actual probability of one employee picking a given gift, nor does it normalize per employee/group.
b. Zero or Uniform Predictions (Critical Symptom)
Cause #1: Model is “confused” by the disconnect—at inference, there is no way to calibrate the true “base rate” for a cold-start gift/group.
Cause #2: Log1p transform compresses small counts; the model “hedges” to 0 or 1 often, especially for low-frequency, sparse, or unseen combinations.
Cause #3: CatBoost Poisson loss (count) assumes target ~ mean count per group, but input grouping structure is not preserved at prediction time.
c. Cold/Unseen Gift Combos
null.

d. Scale Mismatch (Historical Counts vs. Live Prediction)
null.

2.2. Prediction & Aggregation Logic
a. Current Aggregation (_aggregate_predictions)
At present, you compute a prediction for each (gift, gender), then weight by gender ratio. But: since each per-group prediction is per the model's internal group scale, these are not aligned to the per-session employee count.
Aggregating as sum(pred_i * ratio_i) can under- or over-shoot by orders of magnitude due to inconsistent group size scale, explaining the mismatch with business expectation.
b. Lack of Calibration
Normalization: Previous attempt to forcibly normalize post-predicted quantities to sum to #employees was removed, potentially at the cost of losing global constraint but alleviating forced error.
Scaling by Employee Count: This would only make sense if model output is per-capita rate, not arbitrary groupwise count.
Correct Math: Only if your model predicts selection probability per (gift, gender, context) can you safely sum up p(gift|context) * # relevant_employees across groups.
2.3. Modeling Structure & Loss Function
a. Use of Poisson Loss
Suited for event count per interval, only when interval span is comparable for train and prediction.
If training events per group vary wildly (and you do not supply group exposure as a feature), model will simply predict base rate, leading to “mode collapse” at inference.
b. Two-Stage Approach
Recommending as best practice:

Stage 1: Predict probability of selection (per employee/gift/session/group).
Stage 2: Multiply predicted probability by group size (#employees or subgroup) for expected count.
Current Approach: Implied as a single count regression, which is suboptimal for reasons described above.

3. Professional Recommendations & Actionable Steps
3.1. Target Redesign (Most Critical Step)
a. Switch to Rate-Based Target
Target: For each training group, set
$$ \text{selection_rate} = \frac{\text{selection_count}}{\text{total_employees_in_that_group}} $$
Model: Predict selection probability (rate) per gift/employee or per gift/(shop, branch, gender) triple.
Loss: Use regression on selection_rate or log odds for binary classification (if you have individual events rather than aggregates).
b. Advantages
Predicted output is interpretable as “probability per employee per session.”
At inference:
$$ \text{predicted_count_for_gift} = \text{predicted_rate} \times \text{current_employee_count_in_context} $$
Clean mapping from output to business need.
c. When to Use Poisson vs. Regression
Poisson: Only if “exposure” (group size) is a model input, or if all events are per the same denominator.
Regression or Logistic: Recommended when predicting normalized probability/rate.
3.2. Aggregation & Pipeline Correction
Each prediction should be:
expected per-gift count = sum across gender_subgroups(predicted_rate × #present_employees_in_subgroup)

For unseen combos:

Use fallback rates (historical global average for similar gift categories)
Smoothing or Bayesian priors for extreme cold start cases
3.3. Data Preprocessing
Ensure that training “group” definitions match exactly the groups encountered at prediction time (i.e., if model is trained on group-by [shop, branch, gender, gift], infer per batch accordingly).
For cold start, merge gift/employee features with shop-level aggregate rates.
3.4. Diagnosis of Zero Predictions
null.

3.5. Advanced Recommendations
3.5.1. Model Calibration
null.

3.5.2. Ensembling
null.

3.5.3. Uncertainty Estimation
null.

3.5.4. Feature Drift Monitoring
null.

4. Summary Table: Problem → Solution
| Symptom/Issue | Root Cause | Recommended Solution | |--------------------------------------|-------------------------------------------------------|---------------------------------------------| | Zero/flat predictions | Target/count misalignment, cold start, log compression| Use rate-based target, calibration | | Total does not match employee count | Model trains on cumulative counts, not per-employee | Normalize target to selection_rate; multiply by employees at inference| | No discrimination between gifts | Model lacks signal, or is over-regularized | Enhance features, use two-stage modeling | | Cold start | Unseen combos, “other” category collapse | Smoothing, use global or per-category priors| | Aggregation confusion | Math/scaling mismatch | Proper rate × employee scaling per subgroup |

5. Additional Tools & Best Practices
Model Monitoring: Implement tools (e.g., EvidentlyAI) for drift detection and explanation.
Feature Store: Store and version engineered features to ensure consistency across training and prediction.
Reproducibility: Use DVC or MLflow for pipeline, model, and data versioning.
Hyperparameter Optimization: Optuna or CatBoost’s built-in grid search for advanced tuning focused on business-aligned metrics.
Explainability: SHAP values (supported by CatBoost) to diagnose feature influence live.
6. Concrete Next Steps for Your Team
Rebuild training targets to selection_rate (or probability if feasible at event-level).
Retrain CatBoost (or alternative) to predict these rates.
At inference, multiply predicted rate by actual #employees/group to obtain per-gift count.
Optionally soft-normalize per-gift predictions to ensure the total matches expectations.
Add monitoring code to log unseen features, distribution shifts, and model calibration error.
7. Sample Pseudocode for Corrected Pipeline
# After retraining model on selection_rate target

def predict_for_batch(gifts, employees, model):
    gender_counts = Counter(emp['gender'] for emp in employees)
    predictions = []

    for gift in gifts:
        total_count = 0
        for gender, count in gender_counts.items():
            features = assemble_features(gift, gender, ...)  # Same as training
            selection_rate = model.predict(features)
            total_count += selection_rate * count
        predictions.append({'product_id': gift['id'], 'expected_qty': total_count})
    # Optional (if necessary for business logic):
    # Normalize so sum(expected_qty) == len(employees)
    # e.g., with softmax or proportional rescaling
    return predictions
8. References & Further Reading
CatBoost documentation: Regression and Poisson objectives
Machine learning for allocation: Count vs. Rate modeling
Predicting click counts vs. click rates.
9. Open Questions for Further Discussion
Would a hierarchical/Mixed model (by shop, branch) yield improved calibration for rare groups?
Are there use cases where a “cap” (max total picks) is enforced in business logic (i.e., hard constraints on total quantity)?
10. Closing
You’re encountering a classic scale-mismatch stemming from using historical cumulative counts as targets, resulting in non-interpretable and uncalibrated outputs at prediction time. Addressing this with a rate/probability-based target will not only resolve the main issues but also provide a stable foundation for further improvements.

Please reach out if sample code, detailed retraining guidance, or more tailored error analysis on example inference cases is required. A restructured target and aggregation logic will be the fastest and most robust solution to your current challenges.

Let me know if you want detailed sample scripts for target re-computation or advanced CatBoost configuration for this use case.