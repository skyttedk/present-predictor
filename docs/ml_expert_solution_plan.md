# ML Expert Solution Plan: Fixing Prediction Mismatch

## Problem Summary

The ML expert identified a critical mismatch:
- **Training Target**: Raw cumulative `selection_count` (total historical selections for shop/product/gender combinations)
- **Inference Expectation**: Per-order demand for a specific customer with known employee count
- **Current Issue**: Model outputs ~1.0 for all products regardless of features, showing no discrimination

## Root Causes

1. **Scale Mismatch**: Model trained on cumulative counts (potentially hundreds/thousands) but outputs ~1.0
2. **Information Loss**: Employee ratio calculated but then dropped before prediction
3. **Model Not Discriminating**: All products get nearly identical predictions regardless of features
4. **Post-processing Band-aids**: Scaling factors and clipping mask but don't fix the fundamental issue

## Solution Options

### Option A: Immediate Fix (1-2 days) ✅ PARTIALLY IMPLEMENTED

**Status**: A scaling factor has been added but the model still shows uniform predictions.

**Current Implementation**:
- Added scaling factor of 4.0 in `_aggregate_predictions()`
- Results: All predictions now show 8 units (still uniform)

**Issue**: The model itself is not discriminating between products - all raw outputs are ~0.99-1.00

**Next Steps for Option A**:
1. Investigate why the model outputs are so uniform
2. Check for feature engineering mismatches between training and inference
3. Validate that the model was properly trained and saved
4. Consider if the Poisson loss function is causing normalization

### Option B: Proper Fix - Retrain with Rate Target (1-2 weeks) 🎯 RECOMMENDED

**Goal**: Retrain model to predict per-employee selection rates

**Requirements**:
1. **Historical Employee Counts**: Need `employees_exposed` per shop/branch/time period
2. **New Target Variable**: `selection_rate = selection_count / employees_exposed`
3. **Updated Feature Engineering**: Keep employee counts/ratios as features
4. **New Training Pipeline**: Use rate as target or Poisson with offset

**Implementation Steps**:

#### Step 1: Obtain Employee Count Data
```python
# Need historical data like:
# shop_id, branch, period, employee_count
# 6210, 621000, 2023-12, 150
# 6210, 621000, 2024-12, 175
```

#### Step 2: Create Rate-Based Training Data
```python
def prepare_rate_based_training_data(selection_data, employee_counts):
    # Merge selection data with employee counts
    merged = pd.merge(
        selection_data,
        employee_counts,
        on=['employee_shop', 'period'],
        how='left'
    )
    
    # Calculate selection rate
    merged['selection_rate'] = merged['selection_count'] / merged['employee_count']
    
    # Alternative: Keep count but add log(employee_count) as offset
    merged['log_exposure'] = np.log(merged['employee_count'])
    
    return merged
```

#### Step 3: Retrain Model with Proper Target
```python
# Option B1: Train on rate directly
model = CatBoostRegressor(
    loss_function='RMSE',  # or 'Poisson' for rates
    # ... other params
)
model.fit(X, y=data['selection_rate'])

# Option B2: Use Poisson with offset
model = CatBoostRegressor(
    loss_function='Poisson',
    # ... other params
)
# Include log_exposure as a feature with weight 1.0
```

#### Step 4: Update Prediction Logic
```python
def _aggregate_predictions(self, predictions: np.ndarray, 
                         employee_counts: Dict[str, int]) -> float:
    """
    Aggregate predictions where model outputs per-employee rates.
    """
    total_prediction = 0
    
    for i, (gender, count) in enumerate(employee_counts.items()):
        if count > 0:
            # Model outputs rate, multiply by actual employee count
            expected_for_gender = predictions[i] * count
            total_prediction += expected_for_gender
    
    return int(round(total_prediction))
```

## Immediate Actions (While Investigating)

1. **Verify Model Training**:
   - Check if the saved model matches training metrics
   - Validate feature importance shows meaningful patterns
   - Test model on training data samples

2. **Debug Feature Pipeline**:
   - Log feature vectors at training vs inference
   - Ensure categorical encoding matches
   - Verify numeric feature scaling/medians

3. **Test Alternative Scaling**:
   - Try different scaling factors based on historical averages
   - Implement dynamic scaling based on shop size

## Long-term Recommendations

1. **Implement Option B**: Retrain with proper rate-based targets
2. **Add Model Monitoring**: Track prediction distributions in production
3. **A/B Testing**: Compare current model vs rate-based model
4. **Documentation**: Clear documentation of what the model predicts

## Business Impact

- **Current State**: Uniform predictions undermine trust
- **Option A Fix**: May provide varied but potentially inaccurate predictions
- **Option B Fix**: Properly scaled, interpretable predictions

## Timeline

- **Week 1**: Debug current model, implement temporary scaling improvements
- **Week 2**: Obtain employee count data, prepare rate-based training data
- **Week 3**: Retrain model with proper targets, validate performance
- **Week 4**: Deploy and monitor new model

## Success Metrics

1. **Prediction Variance**: Products should show different predicted quantities
2. **Scale Accuracy**: Large companies should show proportionally higher demand
3. **Business Validation**: Predictions align with historical patterns
4. **Model Performance**: Maintain or improve R² while fixing scale issues