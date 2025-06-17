# ML Integration Plan for `/predict` Endpoint

## Overview
This document outlines the implementation plan for integrating the trained CatBoost model into the `/predict` endpoint to provide actual quantity predictions for gift selection.

## Current State
- **Endpoint**: `/predict` transforms input data (CVR lookup, present classification, gender detection)
- **Returns**: `TransformedPredictResponse` with classified presents and employee genders
- **Missing**: Actual ML predictions using the trained CatBoost model

## Target State
- **Keep**: All existing transformation logic (CVR lookup, classification, gender detection)
- **Add**: ML prediction step using CatBoost model
- **Return**: `PredictionResponse` with quantity predictions for each present

## Architecture Overview

```mermaid
graph TD
    A[/predict endpoint] --> B[Data Transformation]
    B --> C[CVR Lookup]
    B --> D[Present Classification]
    B --> E[Gender Detection]
    
    B --> F[ML Predictor Service]
    F --> G[Feature Engineering]
    G --> H[Shop Feature Resolver]
    H --> I[Historical Shop Data]
    H --> J[Similar Shop Proxy]
    
    G --> K[CatBoost Model]
    K --> L[Predictions]
    L --> M[Response Formatting]
```

## Implementation Components

### 1. Shop Feature Resolver (`src/ml/shop_features.py`)

The shop feature resolver handles the challenge of new shops without historical data by using branch/industry codes to find similar shops.

**Key Features**:
- Caches historical shop features from training data
- Maps branch codes to shops for similarity matching
- Provides intelligent fallback strategies

**Fallback Strategy**:
1. Direct lookup for existing shop
2. Average features from similar shops (same branch code)
3. Global average defaults

### 2. ML Predictor Service (`src/ml/predictor.py`)

Core service that handles model loading, feature engineering, and prediction.

**Key Responsibilities**:
- Singleton pattern for efficient model loading
- Feature matrix creation matching training pipeline
- Prediction aggregation across employee-product combinations
- Confidence score calculation

**Feature Engineering Requirements**:
- Base features: 11 categorical columns from training
- Shop features: 8 diversity and frequency features
- Interaction features: 10 hashed interaction features
- Total: ~30 features matching training pipeline

### 3. Feature Engineering Pipeline

The feature engineering must exactly match the training pipeline:

```python
# Required features from training
base_features = [
    'employee_shop', 'employee_branch', 'employee_gender',
    'product_main_category', 'product_sub_category', 'product_brand',
    'product_color', 'product_durability', 'product_target_gender',
    'product_utility_type', 'product_type'
]

shop_features = [
    'shop_main_category_diversity_selected',
    'shop_brand_diversity_selected',
    'shop_utility_type_diversity_selected',
    'shop_sub_category_diversity_selected',
    'shop_most_frequent_main_category_selected',
    'shop_most_frequent_brand_selected',
    'is_shop_most_frequent_main_category',
    'is_shop_most_frequent_brand'
]

# Plus 10 interaction hash features
```

### 4. Prediction Logic

Since the model predicts at the product-employee combination level:

1. For each present:
   - Create feature vectors for each gender group
   - Weight by employee gender distribution
   - Make predictions for each combination
   
2. Aggregate predictions:
   - Sum weighted predictions
   - Scale by total employee count
   - Round to nearest integer

3. Add confidence scores:
   - Based on prediction variance
   - Consider shop data availability
   - Range: 0.0 to 1.0

### 5. API Endpoint Update

Update `/predict` endpoint to:
1. Keep existing transformation logic
2. Add prediction step after transformation
3. Return `PredictionResponse` instead of `TransformedPredictResponse`

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request_data: PredictRequest, current_user: Dict = Depends(get_current_user)):
    # Steps 1-3: Existing transformation logic (keep as is)
    
    # Step 4: ML Predictions (new)
    predictions = predictor.predict(
        branch=branch,
        presents=transformed_presents,
        employees=transformed_employees
    )
    
    # Step 5: Format response
    return PredictionResponse(
        branch_no=request_data.cvr,
        predictions=predictions,
        total_employees=len(request_data.employees),
        processing_time_ms=elapsed_time
    )
```

## Key Challenges and Solutions

### Challenge 1: New Shop Feature Resolution
**Problem**: Model needs shop-level features that don't exist for new shops
**Solution**: Use branch code to find similar shops and average their features

### Challenge 2: Feature Alignment
**Problem**: Features must exactly match training pipeline
**Solution**: Careful mapping and validation of all 30+ features

### Challenge 3: Scaling Predictions
**Problem**: Model predicts per combination, need total per product
**Solution**: Weight by gender distribution and scale by employee count

### Challenge 4: Performance
**Problem**: Multiple predictions per request could be slow
**Solution**: Singleton model loading, batch predictions, caching

## Testing Strategy

### Unit Tests
- Feature engineering correctness
- Shop feature resolver fallback logic
- Prediction aggregation accuracy

### Integration Tests
- Full pipeline with real data
- Edge cases (new shops, single employee, etc.)
- Performance benchmarks

### Validation Tests
- Compare predictions with historical data
- Ensure predictions are reasonable (non-negative, realistic ranges)
- Confidence score validation

## Implementation Steps

1. **Create shop feature resolver** - Load and cache historical features
2. **Create predictor service** - Model loading and prediction logic
3. **Update endpoint** - Integrate predictor with existing transformation
4. **Create tests** - Unit and integration tests
5. **Create POC script** - Test prediction logic in isolation
6. **Performance optimization** - Caching, batching, profiling
7. **Deploy and monitor** - Track predictions in production

## Configuration

### Model Configuration
```python
model_path = "models/catboost_poisson_model/catboost_poisson_model.cbm"
historical_data_path = "src/data/historical/present.selection.historic.csv"
confidence_threshold = 0.7
max_predicted_quantity = 1000
```

### Performance Settings
```python
model_cache_enabled = True
feature_cache_size = 1000
prediction_batch_size = 100
```

## Monitoring and Logging

### Metrics to Track
- Prediction latency per request
- Prediction distribution (mean, median, max)
- Confidence score distribution
- Shop feature resolution success rate
- Cache hit rates

### Logging Requirements
- Log all predictions for analysis
- Track new shops without features
- Monitor prediction outliers
- Performance metrics

## Future Enhancements

1. **Online Learning**: Update shop features as new data arrives
2. **A/B Testing**: Compare with manual predictions
3. **Confidence Intervals**: Provide prediction ranges
4. **Explainability**: Show feature importance per prediction
5. **Feedback Loop**: Incorporate actual vs predicted for model updates