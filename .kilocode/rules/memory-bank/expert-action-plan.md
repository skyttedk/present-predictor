# Expert Review Action Plan - Predictive Gift Selection System

## Executive Summary
Based on the 20-year ML specialist's review, we have identified critical issues capping performance at ~69% of empirical ceiling (R² = 0.31 vs potential 0.45). This action plan addresses each issue with specific implementation steps.

## Priority 1: Immediate Fixes (Week 1)
These changes should unlock +0.10-0.14 R² improvement.

### 1.1 Fix Shop Identifier Schema Drift
**Issue**: Training uses `employee_shop = employee_branch`, but inference still distinguishes them.
**Impact**: Resolver returns branch averages instead of shop-specific features, blurring high-signal effects.

**Actions**:
1. **Update predictor.py `_create_feature_vector`**:
   - Set `employee_shop = branch_code` directly
   - Remove `employee_branch` feature entirely (it's now duplicate)
   - Remove the fallback logic that sets shop=None

2. **Simplify ShopFeatureResolver**:
   - Remove branch fallback logic entirely
   - Always use shop (which is now branch) as the key
   - Remove product_relativity features (already excluded from training)

3. **Retrain with consistent schema**:
   - Ensure training and inference use identical shop definition
   - Regenerate shop aggregates with new definition

### 1.2 Switch to Poisson Objective
**Issue**: RMSE on zero-inflated, bounded target is inefficient (~70% zeros).
**Impact**: Small absolute errors on zeros dominate loss, missing business impact differences.

**Actions**:
1. **Modify catboost_trainer.py**:
   ```python
   # Change from:
   loss_function='RMSE'
   # To:
   loss_function='Poisson'
   # Use count as target with exposure
   target = selection_count
   sample_weight = total_employees_in_group
   ```

2. **Update evaluation metrics**:
   - Add Poisson deviance metric
   - Calculate business-weighted MAPE
   - Keep R² for comparison but focus on new metrics

### 1.3 Expand Hyperparameter Search
**Issue**: Limited search space missing key CatBoost parameters.
**Impact**: Estimated +0.02-0.04 R² improvement potential.

**Actions**:
1. **Expand Optuna search space**:
   ```python
   search_space = {
       'iterations': [100, 2000],
       'learning_rate': [0.001, 0.3],
       'depth': [3, 8],  # Shallower for categoricals
       'l2_leaf_reg': [1, 10],
       'random_strength': [0, 10],
       'bagging_temperature': [0, 1],
       'grow_policy': ['Depthwise', 'Lossguide'],
       'bootstrap_type': ['Bayesian', 'Bernoulli'],
       'subsample': [0.6, 1.0],
       'one_hot_max_size': [2, 50],
       'min_data_in_leaf': [1, 100]
   }
   ```

2. **Improve Optuna configuration**:
   - Increase to 300 trials
   - Add MedianPruner
   - Use CatBoostPruningCallback
   - Enable parallelization (10 workers)

### 1.4 Fix Interaction Features
**Issue**: 10 hashed features on 2 tokens = low entropy, high collisions.
**Impact**: <0.5% variance contribution currently.

**Actions**:
1. **Increase hash dimensions**:
   - Change from 10 to 32 dimensions
   - Add second token set: (brand, target_gender)
   - Consider third set: (sub_category, utility_type)

2. **Update both trainer and predictor**:
   ```python
   # Current:
   hasher = FeatureHasher(n_features=10, input_type='string')
   # Change to:
   hasher = FeatureHasher(n_features=32, input_type='string')
   ```

## Priority 2: Near-term Improvements (Weeks 2-3)

### 2.1 Fix Validation Strategy
**Issue**: Stratification on aggregated data can leak shop information.
**Impact**: Overestimated performance, potential future information leakage.

**Actions**:
1. **Implement GroupKFold**:
   - Group by `employee_shop` before aggregation
   - Ensure shops don't appear in both train/val
   - Add time-based holdout if seasonality exists

### 2.2 Remove Residual Leakage
**Issue**: Diversity counts calculated on all rows, product_relativity features still computed.
**Impact**: Minor but measurable performance inflation.

**Actions**:
1. **Fix diversity calculation**:
   - Calculate diversity only on training subset
   - Store as static values like other shop features

2. **Clean up dead features**:
   - Remove product_relativity calculation entirely
   - Remove from resolver and predictor
   - Audit feature importance after retraining

### 2.3 Production Hardening
**Issue**: Thread safety, excessive logging, version mismatch risks.
**Impact**: Production reliability and maintainability.

**Actions**:
1. **Fix predictor caching**:
   - Use `@lru_cache` at function scope
   - Or implement per-worker instances for gunicorn

2. **Reduce log noise**:
   - Change `logger.error` to `logger.debug` for schema messages
   - Add log rotation configuration

3. **Add version checking**:
   - Hash model features and store in metadata
   - Check hash match on predictor init
   - Fail fast on version mismatch

## Implementation Timeline

### Week 1: Critical Fixes
- Day 1-2: Fix shop identifier drift (#1.1)
- Day 3: Switch to Poisson objective (#1.2)
- Day 4: Expand hyperparameter search (#1.3)
- Day 5: Fix interaction features (#1.4)
- Day 6-7: Retrain and validate improvements

### Week 2: Validation & Cleanup
- Day 1-2: Implement GroupKFold (#2.1)
- Day 3: Remove residual leakage (#2.2)
- Day 4-5: Production hardening (#2.3)

### Week 3: Final Validation
- Comprehensive testing with new model
- Performance benchmarking
- Documentation updates
- Deployment preparation

## Success Metrics
- **Primary**: Achieve R² ≥ 0.40 (from current 0.31)
- **Secondary**: 
  - Poisson deviance improvement >20%
  - Business MAPE <15%
  - Zero prediction inflation <5%
- **Operational**: 
  - Inference latency <100ms
  - Model size <100MB
  - Training time <2 hours

## Risk Mitigation
1. **Backup current model** before any changes
2. **A/B test** new model vs current in shadow mode
3. **Monitor** prediction distributions for anomalies
4. **Rollback plan** with version-tagged models

## Expected Outcomes
Based on the expert's analysis:
- **Immediate fixes**: +0.10-0.12 R² improvement
- **Near-term improvements**: +0.02-0.04 R² improvement
- **Total expected**: R² = 0.43-0.45 (95-100% of empirical ceiling)

This represents moving from 69% to ~95% of the realistic performance ceiling, a significant improvement that should translate to meaningful business impact through better inventory management.