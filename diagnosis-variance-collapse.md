# Diagnosis Plan: Variance Collapse Issue Persists

## Problem Statement
After implementing the Priority 3 critical log-exposure fix, the smoke test still shows extremely narrow prediction spread:
- **Observed range**: 2.3-3.2 (0.9 units)
- **Expected range**: 9.60-12.75 (3.15 units)
- **Result**: ~70% less variance than expected

## UPDATE: Cache Was Not The Issue

After restarting the server, the narrow variance persists (still 2.3-3.2 range). This indicates a deeper issue.

## New Hypotheses (Ordered by Likelihood)

### 1. Model File Not Updated
**Hypothesis**: The retrained model with log_exposure fix wasn't actually saved to the expected path
- **Check**: Compare file timestamps of model file vs training time
- **Check**: Verify the model file checksum matches the retrained version
- **Check**: Load model metadata directly to check for log_exposure indicators

### 2. Training Script Didn't Apply Fix
**Hypothesis**: The model was retrained but the log_exposure baseline offset wasn't actually used
- **Check**: Review training logs for baseline parameter usage
- **Check**: Verify training script has the 3-line fix implemented
- **Check**: Check if Pool object was created with baseline parameter

### 3. Wrong Model Path
**Hypothesis**: API might be loading a different model file than expected
- **Check**: Log the actual model path being loaded
- **Check**: Search for multiple .cbm files in the project
- **Check**: Verify model_path in API configuration

### 4. Post-Processing Dampening
**Hypothesis**: Something in the prediction pipeline is artificially constraining variance
- **Check**: Log raw model outputs before any transformations
- **Check**: Verify no clipping or normalization is applied
- **Check**: Check if confidence calculation affects predictions

## SOLUTION: Force Model Reload

### Immediate Fix
The predictor.py already has a function to clear the cache:

```python
def clear_predictor_cache() -> bool:
    """Clear the cached predictor instance to force reload on next access."""
    global _predictor_instance
    
    if _predictor_instance is not None:
        logger.info("Clearing cached predictor instance to force fresh model reload")
        _predictor_instance = None
        return True
    else:
        logger.info("No cached predictor instance found to clear")
        return False
```

### Implementation Steps
1. **Create a script to force model reload** via API endpoint or direct function call
2. **Clear the cache** to force the predictor to load the new model
3. **Verify the new model** is loaded by checking model metadata/timestamps
4. **Run smoke test again** to confirm variance is restored

### Script to Force Model Reload
```python
# scripts/force_model_reload.py
import requests
import json

# Option 1: If there's an admin endpoint
response = requests.post(
    "http://127.0.0.1:9050/admin/reload-model",
    headers={"X-API-Key": "admin-key"}
)

# Option 2: Direct import and clear
from src.ml.predictor import clear_predictor_cache
success = clear_predictor_cache()
print(f"Cache cleared: {success}")

# Option 3: Restart the API service
# This would definitely force a reload
```

### Long-term Solutions
1. **Add Model Version Checking**: Compare file timestamps or checksums on each request
2. **Implement Model Hot-Reload**: Watch for model file changes and auto-reload
3. **Add Admin Endpoint**: Create `/admin/reload-model` endpoint for manual reloads
4. **Remove Singleton Pattern**: Consider per-request model loading with proper caching strategy

## Verification Steps After Fix

1. **Clear the predictor cache** or restart the API
2. **Run smoke test** - should show predictions in 9.60-12.75 range
3. **Check logs** for "Creating new instance" message
4. **Verify model metadata** shows the retrained timestamp

## Expected Results After Fix
- Prediction range: ~3.15 units (e.g., 9.60-12.75)
- Clear discrimination between different product types
- Total predictions around 33% of employee count (not 89.5%)
- Confidence scores showing meaningful variation

## Additional Diagnostics (If Cache Clear Doesn't Work)

### Check Model File
```bash
# Verify model file was actually updated
ls -la models/catboost_poisson_model/
md5sum models/catboost_poisson_model/catboost_poisson_model.cbm
```

### Check for Multiple Model Files
```bash
# Ensure we're loading the right model
find . -name "*.cbm" -type f
```

### Verify Log Exposure in New Model
```python
# Load model directly and check if it uses baseline
import pickle
with open('models/catboost_poisson_model/model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
print(metadata.get('uses_baseline_offset', False))
```

## Summary
The narrow variance is almost certainly due to the API using a cached instance of the OLD model (before the log_exposure fix). The solution is simple: clear the predictor cache or restart the API to force it to load the new model with the fix applied.