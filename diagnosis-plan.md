# Diagnosis Plan: Different Predictions Between Smoke Test and API

## Issue Summary
- Smoke test shows predictions: 4.9-8.1 range, confidence 0.71-0.73
- API shows predictions: 2.2-3.1 range, all confidence 0.56
- Force reload didn't work because it targeted wrong port (8000 vs 9050)

## Root Cause Analysis

### 1. Port Mismatch
- API running on: `localhost:9050`
- Force reload script targets: `localhost:8000`
- Result: Cache was never cleared on the actual running instance

### 2. Possible Multiple Model Files
The consistent lower predictions and uniform confidence (0.56) from the API suggest it's using an older model without the log-exposure fix.

## Action Plan

### Step 1: Update Force Reload Script Port
Edit `scripts/force_api_reload.py` line 47:
```python
# Change from:
api_url = "http://localhost:8000"
# To:
api_url = "http://localhost:9050"
```

### Step 2: Force Reload on Correct Port
```bash
python scripts/force_api_reload.py
```

### Step 3: Verify Model File Integrity
Check if there's only one model file:
```bash
find . -name "*.cbm" -type f
```

### Step 4: Check Model Timestamps
```bash
ls -la models/catboost_poisson_model/
```

### Step 5: Direct API Test
After force reload, test the API again with the same data as smoke test.

## Alternative Solution
If the issue persists, add logging to verify which model file is being loaded:

In `src/ml/predictor.py`, add after line 61:
```python
logger.info(f"Loading CatBoost model from {self.model_path}")
logger.info(f"Model file exists: {os.path.exists(self.model_path)}")
logger.info(f"Model file size: {os.path.getsize(self.model_path) if os.path.exists(self.model_path) else 'N/A'}")
logger.info(f"Model file modified: {os.path.getmtime(self.model_path) if os.path.exists(self.model_path) else 'N/A'}")
```

This will help confirm both processes are loading the same physical file.