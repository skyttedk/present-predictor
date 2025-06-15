# Technical Letter: Predictive Gift Selection System - Request for Expert ML Guidance

Dear [ML Expert Friend],

I hope this letter finds you well. I'm writing to seek your expert advice on a machine learning project I've been working on for Gavefabrikken, a B2B gift distribution company. We've made significant progress but have hit a performance plateau, and I would greatly value your insights on potential optimization areas.

## Business Context and Problem Statement

Gavefabrikken operates a unique business model where companies provide curated gift selections to their employees through dedicated online portals called "Gift Shops." Each company receives a customized portal featuring a subset of gifts from Gavefabrikken's catalog. During seasonal periods (particularly Christmas), employees access their company's portal to select gifts.

**The Core Challenge**: Accurate demand forecasting for gift quantities, both pre-season and during active periods. Currently, the company relies on manual estimation combined with basic statistical averages, resulting in:
- Inventory imbalances (overstocking and stockouts)
- Significant post-season operational overhead
- Increased costs from back-orders and surplus returns
- Suboptimal customer satisfaction due to unavailable selections

## Data Overview

### Historical Training Data
We have access to historical gift selection data (`present.selection.historic.csv`) containing 178,736 selection events that aggregate to 98,741 unique product-employee combinations. The data structure includes:

**Employee Features (3)**:
- `employee_shop`: Shop identifier (e.g., "2960")
- `employee_branch`: Branch identifier (e.g., "621000")
- `employee_gender`: Employee gender ("male", "female")

**Product Features (8)**:
- `product_main_category`: Primary category (e.g., "Home & Kitchen", "Travel")
- `product_sub_category`: Specific subcategory (e.g., "Cookware", "knife set")
- `product_brand`: Brand name (e.g., "Fiskars", "Kay Bojesen")
- `product_color`: Product color (mostly "NONE" in our data)
- `product_durability`: "durable" or "consumable"
- `product_target_gender`: "unisex", "male", or "female"
- `product_utility_type`: "practical", "aesthetic", or "exclusive"
- `product_type`: "individual" or "shareable"

**Target Variable**: `selection_count` - the number of times a specific product was selected by employees with given demographics in a particular shop.

### Data Characteristics
- **Sparsity**: Many product-employee combinations have low selection counts (mode = 1)
- **Imbalanced Distribution**: Selection counts range from 1 to higher values with a long tail
- **Categorical Nature**: All features are categorical (no continuous variables)
- **Compression Ratio**: 178,736 events → 98,741 unique combinations (1.8x compression)

## Current ML Approach

### Model Architecture
We're using **XGBoost Regressor** with the following configuration:
```python
XGBRegressor(
    n_estimators=1000, max_depth=6, learning_rate=0.03,
    subsample=0.9, colsample_bytree=0.9, reg_alpha=0.3, reg_lambda=0.3,
    gamma=0.1, min_child_weight=8, random_state=42, n_jobs=-1
)
```

### Feature Engineering Pipeline

1. **Original Features (11)**: All categorical features listed above, label-encoded

2. **Shop Assortment Features (NEW)** - engineered to capture shop-level patterns:
   - `shop_main_category_diversity_selected`: Number of distinct main categories selected in shop
   - `shop_brand_diversity_selected`: Number of distinct brands selected in shop
   - `shop_utility_type_diversity_selected`: Number of distinct utility types in shop
   - `shop_sub_category_diversity_selected`: Number of distinct subcategories in shop
   - `shop_most_frequent_main_category_selected`: Most selected main category in shop
   - `shop_most_frequent_brand_selected`: Most selected brand in shop
   - `is_shop_most_frequent_main_category`: Binary - is product the shop's most frequent category?
   - `is_shop_most_frequent_brand`: Binary - is product the shop's most frequent brand?

### Target Transformation
We apply **log transformation** (`np.log1p(selection_count)`) to handle the skewed distribution of selection counts.

### Critical Cross-Validation Methodology
**Key Discovery**: Standard random CV severely underestimates performance due to the selection count distribution. We implemented **Stratified K-Fold CV** with custom binning:
```python
y_strata = pd.cut(y, bins=[0, 1, 2, 5, 10, np.inf], labels=[0, 1, 2, 3, 4])
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

This ensures each fold maintains the selection count distribution, preventing data leakage and providing realistic performance estimates.

## Current Results

### Model Performance Metrics

**With Shop Assortment Features (Current Best)**:
- Stratified CV R² (log target): **0.3112 ± 0.0070**
- Validation R² (log target): 0.3050
- Overfitting (Validation - CV): -0.0062 (minimal)
- Original Scale Metrics (Validation):
  - MAE: 1.3245
  - RMSE: 2.8791
  - R²: ~0.30

**Baseline Model (Original 11 features only)**:
- Stratified CV R² (log target): 0.2947
- Improvement from shop features: +0.0165 (+5.6%)

### Feature Importance (Top 10)
1. `product_sub_category` (0.1823)
2. `product_main_category` (0.1542)
3. `product_brand` (0.1401)
4. `shop_brand_diversity_selected` (0.0891)
5. `employee_shop` (0.0765)
6. `shop_sub_category_diversity_selected` (0.0643)
7. `product_utility_type` (0.0521)
8. `is_shop_most_frequent_brand` (0.0487)
9. `shop_main_category_diversity_selected` (0.0465)
10. `employee_branch` (0.0432)

## Key Technical Challenges and Questions

### 1. Performance Plateau
Despite extensive feature engineering and hyperparameter tuning, we're stuck at R² ≈ 0.31. For business viability, we ideally need R² ≥ 0.6. What fundamental limitations might be preventing better performance?

### 2. Data Characteristics vs. Model Choice
Given that:
- All features are categorical
- Target distribution is highly skewed
- We have moderate data volume (98,741 samples, 19 features)

**Question**: Is XGBoost the optimal choice? Should we consider:
- Deep learning approaches (entity embeddings for categoricals)?
- Bayesian methods?
- Ensemble approaches beyond gradient boosting?
- Different problem formulations (e.g., ranking instead of regression)?

### 3. Feature Engineering Opportunities
We've explored shop-level aggregations, but are there other feature engineering strategies we're missing?
- Interaction features between employee and product attributes?
- Temporal features (if we had timestamps)?
- Graph-based features (employee-product interaction networks)?
- Embedding-based similarities?

### 4. Target Variable Formulation
Currently predicting raw selection counts. Should we consider:
- Predicting selection probability instead?
- Multi-output formulation (probability + quantity)?
- Hierarchical modeling (shop-level → individual predictions)?

### 5. Data Limitations
With 98,741 unique combinations from 178,736 events:
- Is this sufficient data for the complexity we're trying to model?
- Would synthetic data augmentation help?
- Are we missing critical features (price, product images, descriptions)?

### 6. Evaluation Methodology
While stratified CV helped significantly, are there other evaluation considerations?
- Should we use custom business metrics beyond R²?
- Time-based validation splits?
- Leave-one-shop-out cross-validation?

### 7. Business Context Integration
The API will receive:
```json
{
  "branch_no": "123",
  "gifts": [{"product_id": "ABC", "description": "Red ceramic mug"}],
  "employees": [{"name": "John Doe"}]
}
```

We classify descriptions → attributes and names → gender. Is this two-step process introducing noise that limits model performance?

## Specific Technical Details for Reproduction

### Environment
- Python 3.9+
- Key packages: xgboost==1.7.0, scikit-learn==1.3.0, pandas==2.0.0

### Data Processing Pipeline
1. Load raw CSV → Clean (lowercase, strip quotes, fill NA with "NONE")
2. Aggregate by all 11 features → get selection_count
3. Engineer shop features from aggregated data (non-leaky approach)
4. Label encode all categorical features
5. Apply log transformation to target
6. Train/validation split with stratification

### Reproducible Results
The attached notebook (`breakthrough_training.ipynb`) contains the complete pipeline achieving CV R² = 0.3112.

## Questions for Your Expertise

1. **Fundamental Limits**: Given our data characteristics, what's a realistic upper bound for prediction accuracy? Are we approaching the noise floor?

2. **Alternative Approaches**: What radically different ML approaches would you recommend trying?

3. **Feature Engineering**: What feature engineering strategies have you seen work well for similar sparse, categorical, count-prediction problems?

4. **Architecture Recommendations**: Should we consider a two-stage model (classification: will select? → regression: how many)?

5. **Data Strategy**: What additional data would provide the most value? Customer demographics? Historical prices? Product images?

6. **Production Considerations**: Given our moderate performance, how would you recommend framing this to the business? Confidence intervals? Ensemble predictions?

## Closing Thoughts

We've made significant progress from our initial attempts (R² ≈ 0.05 with naive CV) to our current performance (R² ≈ 0.31 with proper methodology). However, I believe we're missing something fundamental that could unlock better performance.

Your expertise in handling similar challenging ML problems would be invaluable. I'm particularly interested in your thoughts on whether we're approaching this problem correctly or if a paradigm shift is needed.

Thank you for taking the time to review this. I'm happy to provide any additional details, data samples, or code that would help with your analysis.

Best regards,
[Your name]

P.S. I've included the full technical context, but please let me know if you need any clarification or additional information about specific aspects of the implementation.

---

## Attachments
- `breakthrough_training.ipynb` - Complete implementation notebook
- `present.selection.historic.csv` - Sample of historical data (available on request)
- Model performance graphs and feature importance plots