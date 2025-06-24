# Priority 2 Implementation Plan - Critical Post-Implementation Fixes

## Executive Summary
The second-pass technical audit has identified 6 critical blocking issues (C1-C6) that prevent the model from functioning correctly despite our Priority 1 improvements. These must be fixed before the model can work in production.

## Critical Blocking Issues

### C1: Target Mismatch (HIGHEST PRIORITY)
**Problem**: Training uses `selection_count` (Poisson) but predictor interprets output as rate and clips to [0,1]
**Impact**: All predictions are wrong by factor of ~employee_count
**Fix Location**: `src/ml/predictor.py`

### C2: Shop Feature Keys Mismatch
**Problem**: Training saves features as `shop_main_category_diversity_selected` but resolver expects `main_category_diversity`
**Impact**: All shop features fall back to global defaults - diversity signal completely lost
**Fix Location**: `src/ml/catboost_trainer.py` OR `src/ml/shop_features.py`

### C3: Import Error Typo
**Problem**: `return _predictor_instanceshop_features.py.txt` (copy-paste artifact)
**Impact**: ImportError at runtime - API fails to start
**Fix Location**: `src/ml/predictor.py`

### C4: Hash Column Schema Drift
**Problem**: 96 hash columns added but only 32 enumerated in numeric_cols
**Impact**: 64 columns bypass dtype coercion & median-fill, schema warnings flood logs
**Fix Location**: `src/ml/catboost_trainer.py` AND `src/ml/predictor.py`

### C5: Missing Validation Weights
**Problem**: CatBoost receives no weights for validation set or Optuna objective
**Impact**: Early-stopping & hyperparameter optimization biased toward unweighted loss
**Fix Location**: `src/ml/catboost_trainer.py`

### C6: Shop Leakage Persists
**Problem**: Data split still stratified by selection_count, not grouped by shop
**Impact**: Same shops appear in train/val - optimistic metrics
**Fix Location**: `src/ml/catboost_trainer.py`

## Implementation Steps

### Phase 1: Critical Fixes (Day 1-2)

#### Fix C1: Align Prediction with Poisson Target
```python
# In src/ml/predictor.py - _make_catboost_prediction method
# BEFORE (incorrect):
predictions = self.model.predict(test_pool)
predictions = np.clip(predictions, 0, 1)  # Wrong! Treating counts as rates
scaled_predictions = predictions * num_employees_per_gender

# AFTER (correct):
pred_counts = self.model.predict(test_pool)  # Poisson outputs counts directly
pred_counts = np.maximum(pred_counts, 0)     # Ensure non-negative only

# In _aggregate_predictions:
expected_qty = np.sum(pred_counts)  # Direct sum, NO multiplication
```

#### Fix C2: Harmonize Shop Feature Keys
**Option A - Update trainer to match resolver expectations:**
```python
# In src/ml/catboost_trainer.py - compute_and_save_shop_resolver_aggregates
shop_aggregates[shop] = {
    'main_category_diversity': diversity_values['main_category'],  # Remove 'shop_' prefix
    'brand_diversity': diversity_values['brand'],
    'utility_type_diversity': diversity_values['utility_type'],
    'sub_category_diversity': diversity_values['sub_category'],
    'most_frequent_main_category': most_frequent_values['main_category'],
    'most_frequent_brand': most_frequent_values['brand']
}
```

**Option B - Quick patch in resolver:**
```python
# In src/ml/shop_features.py - ShopFeatureResolver.resolve_features
features = {
    'shop_main_category_diversity_selected': 
        shop_info.get('shop_main_category_diversity_selected') or 
        shop_info.get('main_category_diversity', 0),
    'shop_brand_diversity_selected':
        shop_info.get('shop_brand_diversity_selected') or
        shop_info.get('brand_diversity', 0),
    # ... repeat for all features
}
```

#### Fix C3: Simple Typo Fix
```python
# In src/ml/predictor.py - get_predictor function
def get_predictor(model_path: Optional[str] = None) -> PredictorService:
    # ...
    return _predictor_instance  # Remove "shop_features.py.txt" suffix
```

#### Fix C4: Enumerate All Hash Features
```python
# Add constant in BOTH files (trainer and predictor):
INTERACTION_HASH_DIM = 32
NUM_INTERACTION_SETS = 3

# In both src/ml/catboost_trainer.py AND src/ml/predictor.py:
num_hash_features = INTERACTION_HASH_DIM * NUM_INTERACTION_SETS  # 96
numeric_cols += [f'interaction_hash_{i}' for i in range(num_hash_features)]
```

#### Fix C5: Add Validation Weights
```python
# In src/ml/catboost_trainer.py - train_catboost_model
model.fit(
    X_train, y_train,
    sample_weight=exposure_train,
    eval_set=[(X_val, y_val)],
    sample_weight_eval_set=[exposure_val],  # ADD THIS LINE
    cat_features=cat_feature_indices,
    # ...
)

# In Optuna objective function:
def objective(trial):
    # ... model training ...
    val_predictions = model.predict(X_val)
    # Use weighted loss for optimization
    weighted_mse = np.average(
        (val_predictions - y_val) ** 2, 
        weights=exposure_val
    )
    return weighted_mse
```

#### Fix C6: Group-based Data Split
```python
# In src/ml/catboost_trainer.py - Replace current train_test_split
from sklearn.model_selection import GroupShuffleSplit

# Remove stratification by selection_count
gss = GroupShuffleSplit(test_size=0.2, random_state=42, n_splits=1)
train_idx, val_idx = next(gss.split(agg_df, groups=agg_df['employee_shop']))

train_df = agg_df.iloc[train_idx]
val_df = agg_df.iloc[val_idx]

# Verify no shop leakage
train_shops = set(train_df['employee_shop'].unique())
val_shops = set(val_df['employee_shop'].unique())
assert len(train_shops & val_shops) == 0, "Shop leakage detected!"
```

### Phase 2: Quality Improvements (Day 2-3)

#### Update Confidence Calculation
```python
# In src/ml/predictor.py
def _calculate_confidence_poisson(self, predictions: np.ndarray) -> float:
    """Calculate confidence based on Poisson prediction variability"""
    mean_pred = np.mean(predictions)
    if mean_pred > 0:
        # Coefficient of variation for Poisson
        cv = np.std(predictions) / mean_pred
        confidence = 1 / (1 + cv)
    else:
        confidence = 0.5
    return np.clip(confidence, 0.5, 0.95)
```

#### Consistent Metric Calculation
```python
# In src/ml/catboost_trainer.py
def calculate_weighted_metrics(y_true, y_pred, weights):
    """Calculate all metrics with consistent weighting"""
    mae = np.average(np.abs(y_true - y_pred), weights=weights)
    mse = np.average((y_true - y_pred) ** 2, weights=weights)
    rmse = np.sqrt(mse)
    
    # Weighted R²
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}
```

#### Remove Global Singleton
```python
# In src/ml/predictor.py
from functools import lru_cache

@lru_cache(maxsize=1)
def get_predictor(model_path: Optional[str] = None) -> PredictorService:
    """Per-process lazy loading with caching"""
    return PredictorService(model_path)
```

### Phase 3: Validation & Retraining (Day 3)

1. **Version Bump**
   - Update model version to v2.0 in metadata
   - Clear old model artifacts

2. **Full Model Retraining**
   ```bash
   python src/ml/catboost_trainer.py
   ```

3. **End-to-End Testing Suite**
   - Test C1: Verify predictions are counts, not rates
   - Test C2: Confirm shop features load non-zero values
   - Test C3: Import test passes
   - Test C4: All 96 hash columns handled
   - Test C5: Validation metrics use weights
   - Test C6: No shops in both train/val

4. **API Integration Tests**
   ```bash
   python scripts/smoke_test.py
   python scripts/test_api_endpoint_direct.py
   ```

## Success Metrics

1. **Predictions are reasonable counts** (10-50 range, not 0.001-0.999)
2. **Shop features load correctly** (diversity > 0 for most shops)
3. **No runtime errors** from typo or missing columns
4. **Validation uses exposure weights** (check logs)
5. **Zero shop overlap** between train/validation
6. **Business metrics**:
   - MAPE < 20% on holdout
   - Sum of predictions ≈ number of employees (±20%)
   - No systematic over/under prediction

## Testing Checklist

```python
# Quick validation script
def validate_fixes():
    # Test C1: Predictions are counts
    test_pred = predictor.predict(test_request)
    assert all(p.expected_qty > 1 for p in test_pred), "Predictions look like rates!"
    
    # Test C2: Shop features load
    resolver = ShopFeatureResolver()
    features = resolver.resolve_features('2960', 'Bags', 'Toiletry Bag', 'Markberg')
    assert features['shop_main_category_diversity_selected'] > 0
    
    # Test C3: Import works
    from src.ml.predictor import get_predictor
    
    # Test C4: All hash columns
    df = create_features_dataframe(...)
    assert sum(1 for col in df.columns if 'interaction_hash' in col) == 96
    
    # Test C6: No shop leakage
    # Check training logs for assertion
```

## Risk Mitigation

1. **Backup current model** before any changes
2. **Feature branch** for all fixes
3. **Test each fix independently** before combining
4. **Shadow mode** comparison with current model
5. **Rollback plan** with versioned models

## Timeline

- **Day 1 AM**: Fix C1, C3 (critical + simple)
- **Day 1 PM**: Fix C2, C4 (feature-related)
- **Day 2 AM**: Fix C5, C6 (validation improvements)
- **Day 2 PM**: Quality fixes, unit tests
- **Day 3**: Full retrain, integration testing, deployment prep

## Next Steps After Implementation

1. Monitor prediction distributions in production
2. A/B test against current model
3. Collect business feedback on accuracy
4. Plan for next optimization round (Priority 3 items)