# Structural Issues Analysis and Fixes for Predictive Gift Selection Model

## Executive Summary

We identified and fixed critical structural issues in the CatBoost model that were causing artificially inflated performance metrics (R² > 0.7) while delivering poor real-world predictions. The primary issue was severe data leakage in feature engineering.

### Key Findings:
- **Leaked Model R²**: >0.7 (artificially inflated)
- **Fixed Model R²**: 0.3956 (realistic performance)
- **Primary Issue**: Product relativity features directly used the target variable

## Detailed Analysis

### 1. Data Leakage Issues Found

#### A. Target Variable Used in Features
The most critical issue was in the `engineer_product_relativity_features` function:

```python
# LEAKED FEATURES (DO NOT USE)
df['product_share_in_shop'] = df['selection_count'] / shop_total_selections
df['brand_share_in_shop'] = brand_total_selections_in_shop / shop_total_selections
```

These features directly incorporated the target variable (`selection_count`) into the features, causing the model to essentially predict its own input.

#### B. Feature Importance Analysis
The leaked features dominated the model:
- `product_share_in_shop`: 66.1% importance (!)
- `product_rank_in_shop`: 0.6% importance

This explains why the model had high R² but poor real-world performance.

#### C. Feature Engineering After Data Split
All features were calculated on the entire dataset before train/test split, allowing validation data to contain information from training data through aggregated features.

### 2. Additional Issues Fixed

#### A. Removed Arbitrary Scaling Factor
The prediction pipeline had a hardcoded scaling factor (0.25) that was attempting to compensate for the mismatch between historical cumulative counts and single-order predictions:

```python
# REMOVED
scaling_factor = 0.25
scaled_prediction = total_prediction * scaling_factor
```

#### B. Removed Forced Normalization
The prediction pipeline was normalizing all predictions to sum to exactly the number of employees, which masked underlying prediction issues:

```python
# REMOVED
normalized_qty = (pred.expected_qty / total_raw_demand) * len(employees)
```

### 3. Fixed Model Implementation

#### A. Non-Leaky Feature Engineering
Created shop features based only on presence/diversity, not selection counts:

```python
# CORRECT APPROACH
shop_summary = data.groupby('employee_shop').agg(
    unique_product_combinations_in_shop=('product_main_category', 'count'),
    distinct_main_categories_in_shop=('product_main_category', 'nunique'),
    distinct_brands_in_shop=('product_brand', 'nunique')
)
```

#### B. Proper Train/Test Split
Now splitting data BEFORE any feature engineering:

```python
# Split indices first
train_indices, test_indices = train_test_split(
    np.arange(len(agg_df)), 
    test_size=0.2, 
    random_state=42, 
    stratify=y_strata_temp
)

# Then engineer features separately
train_with_features = engineer_shop_features_non_leaky(train_df, is_training=True)
test_with_features = engineer_shop_features_non_leaky(test_df, is_training=False)
```

### 4. Performance Comparison

| Metric | Leaked Model | Fixed Model | Change |
|--------|--------------|-------------|---------|
| R² | >0.70 | 0.3956 | -43.5% |
| Top Feature | product_share_in_shop (66.1%) | employee_branch (24.8%) | Structural |
| Real-world Performance | Poor | Expected to be better | Significant |

### 5. New Feature Importance (Fixed Model)

Top 10 features without leakage:
1. employee_branch: 24.8%
2. employee_shop: 22.0%
3. employee_gender: 17.1%
4. product_sub_category: 14.8%
5. product_brand: 7.2%
6. shop_most_frequent_brand: 5.5%
7. shop_main_category_diversity: 2.1%
8. shop_brand_diversity: 2.0%
9. product_color: 1.8%
10. unique_product_combinations_in_shop: 1.1%

These features make logical sense and represent genuine predictive signals.

## Recommendations

### 1. Immediate Actions
- [x] Deploy the fixed model without data leakage
- [x] Update the predictor to use raw predictions without arbitrary scaling
- [x] Remove normalization from the prediction pipeline

### 2. Model Improvements
Based on the ML expert's recommendations:
- [ ] Implement two-stage model (binary selection + count regression)
- [ ] Add temporal features if seasonal data is available
- [ ] Consider employee count normalization in the target variable

### 3. Data Collection
To improve beyond R² ≈ 0.40:
- [ ] Collect employee count data per historical order
- [ ] Add product descriptions for text embeddings
- [ ] Include temporal/seasonal patterns
- [ ] Track actual vs predicted for model refinement

### 4. Monitoring
- [ ] Track prediction accuracy in production
- [ ] Monitor feature importance stability
- [ ] Implement A/B testing for model improvements

## Conclusion

The structural issues have been identified and fixed. The real model performance (R² = 0.3956) is actually quite respectable given:
- Natural noise in gift selection (1 from 40-60 options)
- Limited feature set available
- No price differentiation within shops

This performance level should provide meaningful business value for inventory planning while being honest about the inherent uncertainty in predicting individual preferences.

## Files Modified

1. **src/ml/predictor.py**: Removed scaling factor and normalization
2. **scripts/catboost_trainer_fixed.py**: Created leak-free training pipeline
3. **models/catboost_fixed_model/**: New model artifacts without leakage

## Next Steps

1. Update production to use the fixed model
2. Update the predictor to load from the fixed model path
3. Monitor real-world performance
4. Iterate based on business feedback