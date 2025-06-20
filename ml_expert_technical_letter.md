# Technical Letter: Critical Issues with Gift Demand Prediction Model

**Date:** June 20, 2025  
**Subject:** Request for Expert Review - Severe Prediction Quality Issues in Production Model  
**System:** Predictive Gift Selection API for Gavefabrikken

## Executive Summary

Our gift demand prediction system is producing severely flawed results in production, with uniform predictions across all products and unrealistic confidence scores. Despite having a well-trained CatBoost model (CV R² ≈ 0.59), the production predictions show no meaningful variance and fail to reflect the underlying data patterns. We need expert guidance to diagnose and resolve these critical issues.

## System Overview

### Business Context
Gavefabrikken operates a B2B gift distribution service where companies provide curated gift selections to employees. Our ML system aims to predict how many units of each gift will be selected by employees at a given company, enabling better inventory management.

### What We're Trying to Achieve
Given:
- Company information (CVR/branch code)
- List of available gifts with attributes
- Employee roster with demographics

We need to predict:
- Expected quantity for each gift
- Confidence scores for predictions

## Current Implementation Architecture

### Model Training (`src/ml/catboost_trainer.py`)
- **Algorithm:** CatBoost Regressor with Poisson loss function
- **Features:** 
  - 11 base features (employee/product attributes)
  - Shop-level diversity features
  - Interaction features (hashed)
- **Target:** Aggregated selection counts from historical data
- **Performance:** CV R² ≈ 0.5894 (respectable given problem constraints)

### Prediction Pipeline (`src/ml/predictor.py`)
1. Receives API request with presents and employees
2. Enriches presents with stored classifications
3. Resolves shop-specific features
4. Creates feature vectors for gender groups
5. Makes predictions and aggregates results
6. Returns expected quantities

## Critical Issues Identified

### 1. Uniform Predictions Across All Products
**Symptom:** All products receive nearly identical predictions (10-11 units)
```json
{
  "predictions": [
    {"product_id": "1", "expected_qty": 11, "confidence_score": 0.93},
    {"product_id": "2", "expected_qty": 11, "confidence_score": 0.94},
    {"product_id": "3", "expected_qty": 11, "confidence_score": 0.93},
    // ... all 19 products show 10-11 units
  ],
  "total_employees": 57
}
```

### 2. Architectural Mismatch
- **Documentation specifies:** Two-stage model (classifier + regressor)
- **Actually implemented:** Single-stage Poisson regressor
- **Missing file:** `src/ml/two_stage.py` (referenced but doesn't exist)

### 3. Arbitrary Scaling Factor
- Found hardcoded `scaling_factor = 0.15` in prediction aggregation
- No documentation or calibration for this value
- When removed (set to 1.0), unclear what the impact will be

### 4. Feature Engineering Discrepancies (Now Fixed)
Previously had mismatches between training and inference:
- Default values for missing attributes (fixed: now consistently "NONE")
- NaN handling for numeric features (fixed: now uses training medians)

### 5. Prediction Aggregation Logic Issues
The current aggregation approach:
```python
# Predictions made per gender group
weighted_predictions = predictions * employee_ratios
expected_per_employee = np.sum(weighted_predictions)
total_prediction = expected_per_employee * total_employees * scaling_factor
```

This assumes the model predicts a rate per employee, but the training target was aggregated counts per unique feature combination.

### 6. Confidence Score Calculation
- Uses heuristic approach based on prediction consistency
- Always returns high confidence (0.93-0.95)
- Not based on actual model uncertainty

## Suspected Root Causes

1. **Model Interpretation Mismatch:** The model was trained on aggregated selection counts but predictions are being interpreted as per-employee rates
2. **Missing Zero-Inflation Handling:** Single-stage model may struggle with the many products that won't be selected at all
3. **Feature Distribution Shift:** Inference features may not match training distribution despite alignment efforts
4. **Scale Mismatch:** The arbitrary scaling factor suggests a fundamental scale issue between training and prediction

## Recent Remediation Attempts

1. **Removed arbitrary scaling factor** (0.15 → 1.0)
2. **Aligned feature engineering:** 
   - Default values now consistent ("NONE")
   - Numeric NaN handling uses training medians
3. **Currently retraining model** with updated pipeline

## Questions for Expert Review

1. Should we implement the documented two-stage architecture? Would this better handle zero-inflation?
2. How should we properly interpret the Poisson regressor's output given our training target?
3. Is our prediction aggregation logic correct for this problem formulation?
4. What's the best approach for calibrating predictions without arbitrary scaling?
5. How can we implement meaningful confidence scores?

## Recommended Files for Review

Please review the following files (in order of importance):

1. **`src/ml/predictor.py`** - Production prediction logic with aggregation
2. **`src/ml/catboost_trainer.py`** - Model training pipeline  
3. **`src/api/main.py`** (lines 145-329) - `/predict` endpoint implementation
4. **`src/ml/shop_features.py`** - Shop-level feature resolution
5. **`test_production_predict.json`** - Example of problematic output
6. **`.kilocode/rules/memory-bank/architecture.md`** - System architecture documentation
7. **`.kilocode/rules/memory-bank/expert-feedback-catboost-strategy.md`** - Your previous recommendations

## Specific Areas Needing Expertise

1. **Prediction Interpretation:** How to correctly interpret Poisson regression output for our use case
2. **Aggregation Strategy:** Proper way to go from per-combination predictions to total quantities
3. **Architecture Decision:** Whether two-stage model is necessary given current issues
4. **Calibration Approach:** Data-driven method to ensure predictions are properly scaled
5. **Confidence Estimation:** Implementing statistically sound uncertainty quantification

## Expected Outcomes

We need the model to:
- Produce varied predictions reflecting actual product preferences
- Generate total quantities that make sense relative to employee count
- Provide meaningful confidence intervals
- Handle new shops/products gracefully

Your expertise has been invaluable in our previous discussions. We're at a critical juncture where the model performs well in validation but fails dramatically in production. Any insights into these issues would be greatly appreciated.

Best regards,  
The Predict.Presents Development Team