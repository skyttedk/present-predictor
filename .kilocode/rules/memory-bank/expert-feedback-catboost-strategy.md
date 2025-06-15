# ML Expert Feedback & CatBoost Strategy

## Expert Analysis Summary

### Performance Reality Check
Our ML expert confirmed that our current R² ≈ 0.3112 is actually **respectable** given the natural constraints of the problem:
- **Noise floor**: Selecting 1 gift from ~40-60 options creates pure chance R² ≈ 0.00
- **"Shop mean" oracle**: R² ≈ 0.38-0.45 (optimistic baseline using only shop-level averages)
- **Full leakage oracle**: R² ≈ 0.65-0.70 (theoretical maximum with future knowledge)
- **Realistic ceiling**: R² ≈ 0.45 with current feature set

**Key Insight**: We're at ~69% of the realistic maximum (0.3112 / 0.45), not failing but fighting natural noise and information sparsity.

### Critical Business Constraint Discovered
**Price features won't help** - All gifts in a shop are in the same price range. This eliminates a typical high-value feature source and means we must focus on other signals.

## Quick Wins Action Plan (1-3 weeks)

### 1. Switch to CatBoost with Poisson Loss
**Expected gain**: 2-4 p.p. R² improvement

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=1000,
    loss_function='Poisson',  # Count-aware, no log transform needed
    cat_features=categorical_cols,  # Native categorical handling
    random_state=42
)
```

**Benefits**:
- Better variance handling for count data
- Native categorical feature handling for high-cardinality columns
- No need for log transformation hack

### 2. Implement Two-Stage Model
**Expected gain**: 8-25% RMSE reduction

```python
# Stage 1: Binary Classification
from catboost import CatBoostClassifier
classifier = CatBoostClassifier(
    iterations=500,
    cat_features=categorical_cols,
    random_state=42
)

# Stage 2: Count Regression (positives only)
regressor = CatBoostRegressor(
    iterations=1000,
    loss_function='Poisson',
    cat_features=categorical_cols,
    random_state=42
)

# Final prediction = P(select) × Expected_count
```

**Design Pattern**:
- Stage 1: Predict if gift will be selected (binary)
- Stage 2: Predict how many if selected (count on positives only)
- Combine: selection_probability × expected_count

### 3. Engineer New Features
**Expected gain**: 1-2 p.p. R² improvement

**Quick additions (no extra data)**:
```python
# A. Shop-level shares
df['product_share_in_shop'] = (
    df.groupby(['employee_shop', 'product_id'])['selection_count'].transform('sum') /
    df.groupby('employee_shop')['selection_count'].transform('sum')
)
df['brand_share_in_shop'] = (
    df.groupby(['employee_shop', 'product_brand'])['selection_count'].transform('sum') /
    df.groupby('employee_shop')['selection_count'].transform('sum')
)

# B. Rank features
df['product_rank_in_shop'] = df.groupby('employee_shop')['selection_count'].rank(
    method='dense', ascending=False
)
df['brand_rank_in_shop'] = df.groupby('employee_shop')['product_brand_count'].rank(
    method='dense', ascending=False
)

# C. Interaction features via hashing
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=32, input_type='string')
interactions = hasher.transform(
    df[['employee_branch', 'product_main_category']].apply(
        lambda x: f"{x[0]}_{x[1]}", axis=1
    )
)
```

## Mid-Term Improvements (1-2 months)

### 4. Leave-One-Shop-Out CV
- Quantify cold-start performance
- If gap appears, add hierarchical random effects (shop/product)

### 5. Entity Embeddings
- Replace label encoders with learned embeddings
- TabNet, PyTorch-Tabular, or CatBoost embeddings

### 6. Business Metrics Focus
- Optimize for aggregated MAPE (what drives overstock)
- Not just per-row R²

## Big Bets (Quarter)

### 7. Recommender Formulation
- Two-tower model with side features
- Treat as implicit feedback matrix
- Predict probability, scale to volume

### 8. Text & Image Embeddings
**If data becomes available**:
- Product descriptions → MiniLM/DistilBERT embeddings
- Product images → EfficientNet features

### 9. Monte-Carlo Simulation
- Take predicted distribution
- Run simulations across employee population
- Generate inventory confidence intervals

## Key Takeaways

1. **We're not missing a silver bullet** - fighting natural noise and sparsity
2. **Current R² ≈ 0.31 is respectable** (~69% of realistic maximum)
3. **Clear path to R² ≈ 0.40+** with technical improvements alone
4. **Total expected gain**: 5-10 p.p. → R² ≈ 0.36-0.41

## Implementation Priority

1. **Week 1**: CatBoost with Poisson loss
2. **Week 2**: Two-stage model architecture
3. **Week 3**: New feature engineering
4. **Week 4**: Performance validation & comparison
5. **Month 2**: Production deployment of optimized system

This strategy provides a realistic roadmap from current R² ≈ 0.31 to R² ≈ 0.40+ without requiring new data sources, acknowledging the natural constraints of the problem domain.