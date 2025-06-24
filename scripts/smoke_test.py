# scripts/smoke_test.py

"""
Smoke Test for the Re-architected Prediction Pipeline

This script validates that the rate-based prediction model and the updated
predictor logic are working together as expected.

It checks for:
1. Successful model loading.
2. Non-uniform, non-negative predictions.
3. Correct aggregation logic (total expected_qty is not normalized).
"""

import os
import sys
import logging
import pandas as pd

# Add src to the Python path to allow for absolute imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.ml.predictor import get_predictor
from src.api.schemas.responses import PredictionResult

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def create_test_payload():
    """Creates a sample payload for testing the predictor."""
    branch = "621000"  # A common branch from historical data

    # Sample presents, classified as the predictor would receive them
    presents = [
        {
            "id": "P01",
            "item_main_category": "Home & Kitchen",
            "item_sub_category": "Cookware",
            "brand": "Fiskars",
            "color": "NONE",
            "durability": "durable",
            "target_demographic": "unisex",
            "utility_type": "practical",
            "usage_type": "individual"
        },
        {
            "id": "P02",
            "item_main_category": "Electronics",
            "item_sub_category": "Headphones",
            "brand": "Sony",
            "color": "black",
            "durability": "durable",
            "target_demographic": "unisex",
            "utility_type": "practical",
            "usage_type": "individual"
        },
        {
            "id": "P03",
            "item_main_category": "Travel",
            "item_sub_category": "Hotel Stay",
            "brand": "Comwell",
            "color": "NONE",
            "durability": "consumable",
            "target_demographic": "unisex",
            "utility_type": "exclusive",
            "usage_type": "shareable"
        }
    ]

    # Sample employees
    employees = (
        [{"gender": "male"}] * 50 +
        [{"gender": "female"}] * 45 +
        [{"gender": "unisex"}] * 5
    )
    
    logging.info(f"Created test payload with {len(presents)} presents and {len(employees)} employees for branch '{branch}'.")
    return branch, presents, employees

def run_smoke_test():
    """Executes the smoke test for the prediction pipeline."""
    logging.info("--- Starting Prediction Pipeline Smoke Test ---")
    
    try:
        # 1. Get Predictor Instance
        logging.info("Loading model and initializing predictor...")
        # Use absolute paths to ensure the script runs from any directory
        model_path = os.path.join(BASE_DIR, "models", "catboost_rmse_model", "catboost_rmse_model.cbm")
        historical_data_path = os.path.join(BASE_DIR, "src", "data", "historical", "present.selection.historic.csv")
        
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at: {model_path}")
            logging.error("Please ensure the model has been trained by running `src/ml/catboost_trainer.py`.")
            return False
            
        predictor = get_predictor(model_path=model_path)
        logging.info("Predictor initialized successfully.")

        # 2. Create Test Payload
        branch, presents, employees = create_test_payload()
        total_employees = len(employees)

        # 3. Make Prediction
        logging.info("Making predictions...")
        results = predictor.predict(branch, presents, employees)
        logging.info("Prediction complete.")

        # 4. Validate Results
        logging.info("--- Validation ---")
        
        # Assertion 1: Check if results are a list of dictionaries
        assert isinstance(results, list) and all(isinstance(r, dict) for r in results), \
            f"Expected a list of dictionaries, but got {type(results)}."
        logging.info(f"✅ OK: Prediction returned a list of {len(results)} prediction dictionaries.")

        # Assertion 2: Check if the number of predictions matches the number of presents
        assert len(results) == len(presents), \
            f"Expected {len(presents)} predictions, but got {len(results)}."
        logging.info("✅ OK: Number of predictions matches number of presents.")

        # Assertion 3: Check for non-negative quantities
        quantities = [r['expected_qty'] for r in results]
        assert all(qty >= 0 for qty in quantities), \
            f"Found negative quantities in predictions: {quantities}"
        logging.info("✅ OK: All predicted quantities are non-negative.")

        # Assertion 4: Check for non-uniform predictions
        if len(set(quantities)) == 1 and len(quantities) > 1:
             logging.warning("⚠️ WARNING: All predicted quantities are uniform. This might indicate a problem.")
        else:
            logging.info("✅ OK: Predictions are not uniform.")

        # Assertion 5: Check if total quantity is reasonable (not normalized)
        total_predicted_qty = sum(quantities)
        logging.info(f"Total employees: {total_employees}")
        logging.info(f"Total predicted quantity: {total_predicted_qty:.2f}")
        
        assert total_predicted_qty > 0, "Total predicted quantity should be greater than zero for this test case."
        
        # This is the key check for the new architecture. The sum should NOT equal total_employees.
        # It should be a fraction of it, representing the total expected selections.
        assert total_predicted_qty != total_employees, \
            "Total predicted quantity should not be equal to total employees (old normalization logic)."
        logging.info("✅ OK: Total predicted quantity is not artificially normalized to total employees.")

        logging.info("\n--- Prediction Results ---")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        logging.info("--------------------------\n")

        logging.info("✅ Smoke test passed successfully!")
        return True

    except Exception as e:
        logging.error(f"❌ Smoke test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    test_passed = run_smoke_test()
    if not test_passed:
        sys.exit(1)