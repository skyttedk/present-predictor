# Phase 3.5: CatBoost & Two-Stage Model Implementation Plan

## Overview
**Current State**: XGBoost with CV R² = 0.3112 (log target)  
**Target State**: CatBoost Two-Stage Model with CV R² ≈ 0.36-0.41  
**Expected Gains**:
- CatBoost with Poisson: +2-4 p.p. R²
- Two-Stage Architecture: +3-6 p.p. R² (via RMSE reduction)
- New Features: +1-2 p.p. R²

## Week 1: CatBoost Single-Stage Implementation

### 1.1 Create New CatBoost Notebook
**File**: `notebooks/catboost_implementation.ipynb`

```python
# Key differences from XGBoost notebook:
# 1. No log transformation needed with Poisson loss
# 2. Native categorical feature handling
# 3. Direct count prediction

import catboost
from catboost import CatBoostRegressor

# Model configuration with Poisson loss
model = CatBoostRegressor(
    iterations=1000,
    loss_function='Poisson',  # Count-aware objective
    cat_features=categorical_cols,  # Native categorical handling
    random_state=42,
    verbose=100
)

# Training on raw counts (no log transform!)
model.fit(X_train, y_train,  # y_train is raw selection_count
          eval_set=(X_val, y_val),
          early_stopping_rounds=50)
```

### 1.2 Feature Engineering for CatBoost
**Enhancements to existing features**:

```python
# A. Shop-level share features (non-leaky)
df['product_share_in_shop'] = (
    df.groupby(['employee_shop', 'product_id'])['selection_count'].transform('sum') /
    df.groupby('employee_shop')['selection_count'].transform('sum')
)

df['brand_share_in_shop'] = (
    df.groupby(['employee_shop', 'product_brand'])['selection_count'].transform('sum') /
    df.groupby('employee_shop')['selection_count'].transform('sum')  
)

df['category_share_in_shop'] = (
    df.groupby(['employee_shop', 'product_main_category'])['selection_count'].transform('sum') /
    df.groupby('employee_shop')['selection_count'].transform('sum')
)

# B. Rank features
df['product_rank_in_shop'] = df.groupby('employee_shop')['selection_count'].rank(
    method='dense', ascending=False
)

df['brand_rank_in_shop'] = df.groupby(['employee_shop', 'product_brand'])['selection_count'].transform('sum').rank(
    method='dense', ascending=False
)

# C. Interaction features via hashing (memory efficient)
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=32, input_type='string')

# Create interaction strings
interactions = df.apply(
    lambda x: f"{x['employee_branch']}_{x['product_main_category']}", axis=1
)
interaction_features = hasher.transform(interactions).toarray()

# Add as new columns
for i in range(32):
    df[f'interaction_hash_{i}'] = interaction_features[:, i]
```

### 1.3 CatBoost-Specific Optimizations

```python
# Categorical column specification
categorical_features = [
    'employee_shop', 'employee_branch', 'employee_gender',
    'product_main_category', 'product_sub_category', 'product_brand',
    'product_color', 'product_durability', 'product_target_gender',
    'product_utility_type', 'product_type',
    'shop_most_frequent_main_category_selected',
    'shop_most_frequent_brand_selected'
]

# No need for label encoding with CatBoost!
# Just ensure categorical columns are strings
for col in categorical_features:
    if col in X.columns:
        X[col] = X[col].astype(str)
```

## Week 2: Two-Stage Model Architecture

### 2.1 Create Two-Stage Implementation
**File**: `notebooks/two_stage_catboost.ipynb`

```python
from catboost import CatBoostClassifier, CatBoostRegressor

class TwoStagePredictor:
    def __init__(self, cat_features):
        self.cat_features = cat_features
        
        # Stage 1: Binary classifier
        self.classifier = CatBoostClassifier(
            iterations=500,
            cat_features=cat_features,
            random_state=42,
            verbose=False
        )
        
        # Stage 2: Count regressor (Poisson)
        self.regressor = CatBoostRegressor(
            iterations=1000,
            loss_function='Poisson',
            cat_features=cat_features,
            random_state=42,
            verbose=False
        )
    
    def fit(self, X, y):
        # Create binary target
        y_binary = (y > 0).astype(int)
        
        # Train classifier
        self.classifier.fit(X, y_binary)
        
        # Train regressor on positive samples only
        positive_mask = y > 0
        X_positive = X[positive_mask]
        y_positive = y[positive_mask]
        
        if len(y_positive) > 0:
            self.regressor.fit(X_positive, y_positive)
        
        return self
    
    def predict(self, X):
        # Stage 1: Predict selection probability
        selection_probs = self.classifier.predict_proba(X)[:, 1]
        
        # Stage 2: Predict count if selected
        expected_counts = self.regressor.predict(X)
        
        # Combine: probability × expected count
        final_predictions = selection_probs * expected_counts
        
        return final_predictions
    
    def predict_with_uncertainty(self, X):
        # Get base predictions
        predictions = self.predict(X)
        
        # Estimate uncertainty (simplified)
        selection_probs = self.classifier.predict_proba(X)[:, 1]
        uncertainty = np.sqrt(predictions * (1 - selection_probs))
        
        return predictions, uncertainty
```

### 2.2 Two-Stage Cross-Validation

```python
def two_stage_cv(X, y, cv_folds=5):
    """Custom CV for two-stage model"""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_strata = pd.cut(y, bins=[0, 1, 2, 5, 10, np.inf], 
                      labels=[0, 1, 2, 3, 4], include_lowest=True)
    
    scores = []
    for train_idx, val_idx in cv.split(X, y_strata):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train two-stage model
        model = TwoStagePredictor(cat_features=categorical_features)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

## Week 3: Advanced Features & Optimization

### 3.1 Leave-One-Shop-Out Cross-Validation

```python
def leave_one_shop_out_cv(X, y):
    """Assess cold-start performance"""
    shops = X['employee_shop'].unique()
    scores = []
    
    for test_shop in shops:
        # Train on all shops except one
        train_mask = X['employee_shop'] != test_shop
        test_mask = ~train_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Train model
        model = TwoStagePredictor(cat_features=categorical_features)
        model.fit(X_train, y_train)
        
        # Evaluate on held-out shop
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

### 3.2 Performance Comparison Framework

```python
def compare_models(X, y):
    """Compare XGBoost vs CatBoost vs Two-Stage"""
    results = {}
    
    # 1. Original XGBoost (baseline)
    xgb_model = XGBRegressor(**optimal_xgb_params)
    xgb_scores = cross_val_score(xgb_model, X, np.log1p(y), 
                                cv=cv_stratified, scoring='r2')
    results['XGBoost_Log'] = {
        'cv_r2': xgb_scores.mean(),
        'cv_std': xgb_scores.std()
    }
    
    # 2. CatBoost Single-Stage
    cb_model = CatBoostRegressor(
        iterations=1000,
        loss_function='Poisson',
        cat_features=categorical_features,
        random_state=42,
        verbose=False
    )
    cb_scores = cross_val_score(cb_model, X, y,  # Raw counts
                               cv=cv_stratified, scoring='r2')
    results['CatBoost_Poisson'] = {
        'cv_r2': cb_scores.mean(),
        'cv_std': cb_scores.std()
    }
    
    # 3. Two-Stage CatBoost
    ts_mean, ts_std = two_stage_cv(X, y)
    results['TwoStage_CatBoost'] = {
        'cv_r2': ts_mean,
        'cv_std': ts_std
    }
    
    return results
```

## Implementation Checklist

### Week 1 Tasks:
- [ ] Create `notebooks/catboost_implementation.ipynb`
- [ ] Implement single-stage CatBoost with Poisson loss
- [ ] Add new share and rank features
- [ ] Add interaction features via hashing
- [ ] Compare performance with XGBoost baseline
- [ ] Document performance gains

### Week 2 Tasks:
- [ ] Create `notebooks/two_stage_catboost.ipynb`
- [ ] Implement `TwoStagePredictor` class
- [ ] Add custom two-stage cross-validation
- [ ] Implement uncertainty estimation
- [ ] Test on full dataset
- [ ] Validate expected RMSE reduction

### Week 3 Tasks:
- [ ] Implement Leave-One-Shop-Out CV
- [ ] Create comprehensive model comparison
- [ ] Generate performance visualization dashboard
- [ ] Select best model configuration
- [ ] Prepare for API integration
- [ ] Update memory bank with results

## Expected Deliverables

1. **CatBoost Single-Stage Notebook**
   - Complete implementation with Poisson loss
   - Performance comparison showing +2-4 p.p. gain
   - Feature importance analysis

2. **Two-Stage Model Notebook**
   - Full two-stage implementation
   - Performance metrics showing +3-6 p.p. additional gain
   - Uncertainty quantification

3. **Performance Dashboard**
   - Model comparison visualization
   - CV results across different methodologies
   - Business metric evaluation (MAPE)

4. **Updated ML Module**
   - `src/ml/catboost_model.py`
   - `src/ml/two_stage.py`
   - Integration-ready code

## Success Metrics
- [ ] CatBoost Poisson achieves R² ≥ 0.33
- [ ] Two-Stage model achieves R² ≥ 0.36
- [ ] Combined improvements reach R² ≈ 0.36-0.41
- [ ] Cold-start performance acceptable (LOSO CV)
- [ ] API integration design completed

## Next Steps
1. Start with CatBoost implementation notebook
2. Focus on Poisson loss function benefits
3. Add new features incrementally
4. Validate each improvement step
5. Document all performance gains

This plan provides a clear path from our current XGBoost R² = 0.3112 to the target CatBoost Two-Stage R² ≈ 0.36-0.41, following the ML expert's recommendations exactly.