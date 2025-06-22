# Technical Letter: Predictive Gift Selection System Challenges

## Dear ML Expert,

I'm reaching out for your expertise on some critical challenges we're facing with our predictive gift selection system. Below is a comprehensive technical overview of our current situation.

## System Overview

We're building a B2B gift demand prediction system for Gavefabrikken, where companies offer curated gift selections to employees. The goal is to predict how many units of each gift will be selected given:
- A specific set of available gifts (with attributes)
- Employee demographics (primarily gender distribution)
- Company/branch information

## Current Model Architecture

### Training Pipeline (catboost_trainer.py)
- **Model**: CatBoostRegressor with Poisson loss
- **Target Variable**: `selection_count` - aggregated count of gift selections by (shop, branch, gender, product attributes)
- **Features**:
  - Employee features: shop, branch, gender
  - Product features: main_category, sub_category, brand, color, durability, target_gender, utility_type, usage_type
  - Shop-level features: diversity metrics, most frequent categories/brands
  - Interaction features: hashed branch × main_category interactions
- **Performance**: CV R² ≈ 0.3112 (log-transformed target)

### Prediction Pipeline (predictor.py)
1. For each gift, create feature vectors for each gender group present in employee list
2. Get model predictions for each gender-specific feature vector
3. Aggregate predictions using gender ratios
4. Return final quantity predictions

## Critical Issues We're Facing

### Issue 1: Zero or Uniform Predictions
When testing the API endpoint with real data:
```
Input: 5 gifts, 57 employees
Output: All gifts predicted as 0.0 or 1.0 (no discrimination between products)
```

### Issue 2: Total Quantity Mismatch
- Business expectation: Total predicted quantities should ≈ number of employees
- Current behavior: Total predictions either far exceed employee count or are near zero
- Example: 5 gifts × 1.0 each = 5 total, but we have 57 employees

### Issue 3: Model Target vs Business Need Confusion
The model was trained on `selection_count` which represents:
```python
selection_count = historical_data.groupby(['shop', 'branch', 'gender', ...product_features...]).size()
```

This is the cumulative count across all historical data, not a rate or probability. During prediction:
- We don't have historical context for new gift combinations
- The model outputs raw counts that don't naturally scale to current employee counts

### Issue 4: Aggregation Logic Uncertainty
Current aggregation in `_aggregate_predictions`:
```python
for pred, ratio in zip(predictions, gender_ratios):
    weighted_pred = pred * ratio  # Just weight by gender ratio
    weighted_predictions.append(weighted_pred)
total_prediction = np.sum(weighted_predictions)
```

But should it be:
```python
weighted_pred = pred * ratio * total_employees  # Scale by employees?
```

## Technical Details

### Model Training Stats
- Training samples: 98,741 unique combinations from 178,736 events
- Categorical handling: Native CatBoost categorical features
- Features: 20 base features + 10 interaction hash features
- Target transformation: log1p(selection_count) during training

### Prediction Time Challenges
1. **Cold Start**: Many gift/shop combinations have never been seen
2. **Feature Resolution**: Shop features use historical medians as fallbacks
3. **Scale Mismatch**: Model trained on counts, but needs to output rates/probabilities

### What We've Tried
1. **Normalization**: Post-prediction normalization to match employee count (removed per expert advice)
2. **Scaling in aggregation**: Multiplying by total_employees (unclear if correct)
3. **Debug logging**: Shows model is loaded but outputs very small values

## Key Questions for Your Expertise

1. **Target Variable**: Should we retrain with a rate-based target (selection_count / total_employees_in_shop)?

2. **Aggregation Logic**: Given the model outputs selection counts, what's the correct way to aggregate gender-specific predictions?

3. **Scale Calibration**: How do we bridge between historical cumulative counts and current-session predictions?

4. **Zero Predictions**: Why might a Poisson regression model output near-zero values for all inputs?

5. **Architecture**: Is the current two-step approach (predict by gender, then aggregate) fundamentally flawed?

## Code References

Key files for investigation:
- [`src/ml/catboost_trainer.py`](../src/ml/catboost_trainer.py) - Training pipeline
- [`src/ml/predictor.py`](../src/ml/predictor.py) - Prediction service
- [`scripts/test_comprehensive_predictions.py`](../scripts/test_comprehensive_predictions.py) - Testing script
- [`src/data/historical/present.selection.historic.csv`](../src/data/historical/present.selection.historic.csv) - Training data structure

## Sample Test Case

```python
# Input
presents = [
    {
        'id': '1',
        'item_main_category': 'Home & Kitchen',
        'item_sub_category': 'Kitchen Appliance',
        'brand': 'Tisvilde',
        # ... other attributes
    },
    # ... 4 more presents
]
employees = [
    {'name': 'John Doe', 'gender': 'male'},
    # ... 56 more employees (70% male, 30% female)
]

# Current Output
predictions = [
    {'product_id': '1', 'expected_qty': 0.0},
    {'product_id': '2', 'expected_qty': 0.0},
    # ... all zeros
]
```

## Your Previous Insights

You previously noted:
- R² ≈ 0.31 is respectable given natural constraints
- Recommended CatBoost with Poisson loss (implemented)
- Suggested two-stage model (selection probability × count if selected)
- Noted that normalization might not be the solution

## Request for Help

Could you help us identify:
1. Structural issues in our current approach
2. Why predictions are collapsing to zero
3. The correct mathematical relationship between model output and business need
4. Whether we need to fundamentally restructure the training target

Any insights would be greatly appreciated. I'm happy to provide additional code, data samples, or test outputs as needed.

Best regards,
[Development Team]

---

## Appendix: Current Model Performance

### Training Metrics (from latest run)
```
Best CV Score (RMSE on log target): 1.1398
Feature Importance (Top 5):
1. employee_branch: 25.3%
2. product_main_category: 18.7%
3. employee_shop: 12.4%
4. product_sub_category: 8.9%
5. product_brand: 7.2%
```

### Prediction Pipeline Logs
```
INFO: Making predictions for 5 presents and 57 employees
DEBUG: Employee demographics: {'male': 0.70, 'female': 0.28, 'unisex': 0.02}
INFO: Prediction complete. Total raw demand: 0.0 units