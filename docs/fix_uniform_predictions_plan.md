# Action Plan: Fix Uniform Predictions Issue

## Problem Summary
The model is producing nearly uniform predictions (~4.0-4.8) for all products, regardless of their actual popularity. This confirms the ML expert's diagnosis of a fundamental mismatch between what the model was trained on (cumulative selection_count) and what we need to predict (per-session quantities).

## Two-Phase Solution Approach

### Phase 1: Fundamental Re-Architecture (Critical - 1 week)
**Goal**: Fix the core issue by changing from count-based to rate-based prediction

#### 1.1 Data Pipeline Changes (Days 1-2)
**Priority**: CRITICAL

1. **Obtain Historical Employee Counts**
   - [ ] Analyze historical data structure to find employee count information
   - [ ] Create script to calculate `total_employees_in_group` for each historical record
   - [ ] If data is missing, work with stakeholders to obtain this critical information

2. **Create New Target Variable**
   ```python
   # In data preprocessing
   df['selection_rate'] = df['selection_count'] / df['total_employees_in_group']
   # This gives us a rate between 0 and 1
   ```

3. **Update Data Pipeline**
   - [ ] Modify `src/data/preprocessor.py` to include employee count calculation
   - [ ] Add validation to ensure rates are between 0 and 1
   - [ ] Update feature engineering to work with rate-based targets

#### 1.2 Model Retraining (Days 3-4)
**File**: `src/ml/catboost_trainer.py`

1. **Switch to Rate-Based Training**
   ```python
   from catboost import CatBoostRegressor
   
   model = CatBoostRegressor(
       iterations=1000,
       loss_function='RMSE',  # Standard regression for rates
       eval_metric='RMSE',
       cat_features=categorical_cols,
       random_state=42
   )
   
   # Train on selection_rate, not selection_count
   model.fit(X_train, y_train_rates)
   ```

2. **Remove Log Transformation**
   - [ ] Remove all `np.log1p()` transformations
   - [ ] Train directly on rates (0-1 scale)

#### 1.3 Prediction Pipeline Fix (Days 4-5)
**File**: `src/ml/predictor.py`

1. **Update Aggregation Logic**
   ```python
   def _aggregate_predictions(self, predictions, gender_counts):
       """
       New logic: Scale rate predictions by actual employee counts
       """
       total_expected = 0
       
       for gender, predicted_rate in predictions.items():
           employee_count = gender_counts.get(gender, 0)
           expected_selections = predicted_rate * employee_count
           total_expected += expected_selections
           
       return total_expected
   ```

2. **Remove Post-Processing Normalization**
   - [ ] Remove any artificial scaling or normalization
   - [ ] Let predictions naturally reflect rates × counts

#### 1.4 Validation (Days 5-6)
1. **Create Test Cases**
   - [ ] Test with known historical examples
   - [ ] Verify predictions show meaningful variation
   - [ ] Ensure sum of predictions is reasonable (not forced to match)

2. **Sanity Checks**
   - [ ] Popular items should have higher predicted quantities
   - [ ] Unpopular items should have lower predicted quantities
   - [ ] Total predictions should be reasonable relative to employee count

### Phase 2: Technical Optimizations (Week 2)
**Goal**: Improve prediction quality using CatBoost best practices

#### 2.1 Implement CatBoost with Native Features (Days 7-8)
```python
from catboost import CatBoostRegressor, Pool

# Create Pool with categorical features specified
train_pool = Pool(
    data=X_train,
    label=y_train_rates,
    cat_features=categorical_indices
)

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)
```

#### 2.2 Add Advanced Features (Days 9-10)
1. **Shop-Level Statistics**
   ```python
   # Historical popularity rates
   df['product_historical_rate'] = (
       df.groupby('product_id')['selection_rate'].transform('mean')
   )
   
   # Shop-specific preferences
   df['shop_product_affinity'] = (
       df.groupby(['employee_shop', 'product_main_category'])['selection_rate']
       .transform('mean')
   )
   ```

2. **Interaction Features**
   - Shop × Category interactions
   - Gender × Product type interactions
   - Branch × Brand preferences

#### 2.3 Two-Stage Model (Optional - Days 11-12)
If single model still shows issues:

```python
# Stage 1: Will anyone select this gift?
binary_model = CatBoostClassifier(
    iterations=1000,
    cat_features=categorical_cols
)

# Stage 2: If selected, what's the rate?
rate_model = CatBoostRegressor(
    iterations=1000,
    loss_function='RMSE',
    cat_features=categorical_cols
)

# Combine: P(selected) × E[rate|selected]
```

## Success Metrics

### Immediate Success (Phase 1)
- [ ] **Variation in predictions**: Range should be meaningful (e.g., 0.5 to 15.0, not 4.0 to 4.8)
- [ ] **Business logic**: Popular items get higher predictions
- [ ] **No forced normalization**: Sum doesn't artificially match employee count

### Long-term Success (Phase 2)
- [ ] **Improved R²**: Target 0.35-0.40 (from current 0.31)
- [ ] **Business validation**: Predictions match domain expert intuition
- [ ] **Production stability**: Consistent performance across different shops

## Implementation Timeline

**Week 1 (Critical)**:
- Mon-Tue: Data pipeline for rates
- Wed-Thu: Model retraining
- Fri: Prediction pipeline fix
- Weekend: Testing & validation

**Week 2 (Optimization)**:
- Mon-Tue: CatBoost implementation
- Wed-Thu: Feature engineering
- Fri: Performance comparison
- Weekend: Documentation & deployment prep

## Immediate Next Steps

1. **Today**: Investigate historical data for employee counts
2. **Tomorrow**: Start implementing rate-based target calculation
3. **This Week**: Complete Phase 1 to fix uniform predictions

## Risk Mitigation

1. **Missing Employee Count Data**
   - Fallback: Estimate from shop size patterns
   - Alternative: Use average shop sizes as proxy

2. **Performance Degradation**
   - Keep current model as backup
   - A/B test new predictions

3. **Timeline Pressure**
   - Focus on Phase 1 first (critical fix)
   - Phase 2 can be iterative improvements