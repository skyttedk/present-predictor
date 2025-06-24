# Implementation Plan: Log-Exposure Fix for Poisson Model

## Expert Diagnosis Summary
The model shows:
- **Magnitude shift**: Predicts 88 vs actual 57 (+55%)
- **Variance collapse**: All predictions in 3.6-5.4 range vs actual 0-12
- **Root cause**: Missing log-exposure offset in prediction formula

## Implementation Plan

### Phase 1: Update Training Pipeline (catboost_trainer.py)
**Location**: Line ~448 in `prepare_features_for_catboost()`

#### Changes Required:
1. **Add log_exposure calculation**:
   ```python
   # Line 366: After exposure calculation
   X['log_exposure'] = np.log(exposure + 1e-8)
   ```

2. **Update expected_numeric_cols**:
   ```python
   # Line 389-396: Add to expected columns list
   expected_numeric_cols.append('log_exposure')
   ```

3. **Ensure median calculation includes log_exposure**:
   - The existing median calculation logic (lines 398-430) will automatically handle it

### Phase 2: Update Prediction Pipeline (predictor.py)
**Location**: `_create_feature_vector()` method

#### Changes Required:
1. **Add log_exposure to feature vector**:
   ```python
   # In _create_feature_vector() after gender_count calculation
   features['log_exposure'] = np.log(count + 1e-8)  # where count = gender_counts[gender]
   ```

2. **Update numeric_cols list**:
   - Add 'log_exposure' to the numeric columns list
   - Ensure it's included in the 96 hash features + other numeric features

### Phase 3: Model Retraining
1. Execute: `python src/ml/catboost_trainer.py`
2. Model will now learn proper exposure scaling
3. Save new model artifacts with log_exposure feature

### Phase 4: Validation Tests
Create validation script to verify:
- Zeros predicted < 1
- High sellers predicted ≥ 10  
- Total predicted ≈ 57 ± 10%
- Variance spans 0-12 range

## Technical Details

### Why This Works
- **During training**: Model learns coefficient for log_exposure feature
- **During prediction**: log(gender_count) scales predictions appropriately
- **Result**: Predictions automatically adjust to group size

### Critical Implementation Notes
1. **Consistency**: Same feature engineering in train and predict
2. **Small constant**: Use 1e-8 to avoid log(0)
3. **Keep sample_weight**: It's still useful for weighted loss calculation
4. **No multiplication needed**: The sum aggregation remains correct

## Expected Outcomes
After implementation:
- **Magnitude**: Total predictions ≈ actual employee count
- **Variance**: Predictions span full 0-12+ range
- **Discrimination**: Model distinguishes between popular and unpopular items

## Risk Mitigation
1. Backup current model before retraining
2. A/B test with feature flag
3. Validate on multiple test cases
4. Monitor business metrics post-deployment

## Timeline
- Implementation: 2-3 hours
- Retraining: 1-2 hours
- Validation: 1-2 hours
- **Total**: ~6 hours

This fix addresses the fundamental architectural issue preventing proper count predictions.