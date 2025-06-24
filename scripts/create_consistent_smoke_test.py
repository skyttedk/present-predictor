#!/usr/bin/env python3

"""
Create a consistent smoke test that uses the same classification pipeline as the API
"""

import os
import sys
import logging
import pandas as pd
import asyncio

# Add src to the Python path to allow for absolute imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.ml.predictor import get_predictor
from src.data.classifier import DataClassifier
from src.data.openai_client import create_openai_client
from src.data.gender_classifier import classify_employee_gender
from src.api.schemas.requests import GiftItem

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

async def create_consistent_test_payload():
    """Creates a test payload using the SAME classification pipeline as the API"""
    
    # Raw presents data (same as smoke test but will be classified via OpenAI)
    raw_presents = [
        {"id": "1", "description": "Tisvilde Pizzaovn", "model_name": "Tisvilde Pizzaovn", "model_no": "", "vendor": "GaveFabrikken"},
        {"id": "2", "description": "BodyCare Massagepude", "model_name": "BodyCare Massagepude", "model_no": "", "vendor": "Gavefabrikken"},
        {"id": "3", "description": "Jesper Koch grill sæt, sort, 5 stk", "model_name": "Jesper Koch grill sæt, sort, 5 stk", "model_no": "", "vendor": "GaveFabrikken"}
    ]
    
    # Raw employee names (same as smoke test but will be classified via gender_guesser)
    raw_employees = [
        {"name": "Laimonas Lukosevicius"},
        {"name": "Petra De Laet"}, 
        {"name": "Regimantas Usas"}
    ]
    
    # Step 1: Classify presents using the SAME pipeline as API
    logging.info("🤖 Classifying presents using OpenAI Assistant (same as API)...")
    
    openai_client = None
    try:
        openai_client = create_openai_client()
        classifier = DataClassifier(openai_client)
    except Exception as e:
        logging.warning(f"Failed to create OpenAI client: {e}")
        classifier = DataClassifier()  # Will use fallback classification
    
    # Convert to GiftItem format for classification
    gift_items = [
        GiftItem(product_id=p["id"], description=p["description"])
        for p in raw_presents
    ]
    
    # Classify gifts using the same method as API
    classified_gifts = await classifier._classify_gifts(gift_items)
    
    # Convert to predictor format
    classified_presents = []
    for i, classified_gift in enumerate(classified_gifts):
        attrs = classified_gift.attributes
        present_dict = {
            'id': raw_presents[i]["id"],
            'item_main_category': attrs.itemMainCategory,
            'item_sub_category': attrs.itemSubCategory,
            'color': attrs.color,
            'brand': attrs.brand,
            'vendor': attrs.vendor,
            'target_demographic': attrs.targetDemographic.value,
            'utility_type': attrs.utilityType,
            'usage_type': attrs.usageType,
            'durability': attrs.durability
        }
        classified_presents.append(present_dict)
        
        logging.info(f"📝 Present {present_dict['id']} classified as:")
        logging.info(f"   Main Category: {present_dict['item_main_category']}")
        logging.info(f"   Sub Category: {present_dict['item_sub_category']}")
        logging.info(f"   Brand: {present_dict['brand']}")
        logging.info(f"   Utility: {present_dict['utility_type']}")
    
    # Close OpenAI client
    if openai_client:
        try:
            await openai_client.close()
        except:
            pass
    
    # Step 2: Classify employees using the SAME pipeline as API
    logging.info("👥 Classifying employees using gender_guesser (same as API)...")
    
    processed_employees = []
    for employee in raw_employees:
        gender = classify_employee_gender(employee["name"])
        processed_employees.append({
            'gender': gender.value  # Convert enum to string
        })
        logging.info(f"   {employee['name']} → {gender.value}")
    
    return "469000", classified_presents, processed_employees  # Use same branch as API

async def run_consistent_smoke_test():
    """Executes smoke test using the SAME classification pipeline as API"""
    logging.info("--- Starting CONSISTENT Prediction Pipeline Smoke Test ---")
    logging.info("🔧 Using SAME classification pipeline as API (OpenAI + gender_guesser)")
    
    try:
        # 1. Get Predictor Instance (same model as API)
        logging.info("Loading model...")
        model_path = os.path.join(BASE_DIR, "models", "catboost_poisson_model", "catboost_poisson_model.cbm")
        
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at: {model_path}")
            return False
            
        predictor = get_predictor(model_path=model_path)
        logging.info("Predictor initialized successfully.")

        # 2. Create Test Payload using REAL classification pipeline
        branch, presents, employees = await create_consistent_test_payload()
        total_employees = len(employees)

        # 3. Make Prediction (same method as API)
        logging.info("Making predictions...")
        results = predictor.predict(branch, presents, employees)
        logging.info("Prediction complete.")

        # 4. Analyze Results
        logging.info("--- CONSISTENT TEST RESULTS ---")
        
        quantities = [r['expected_qty'] for r in results]
        confidences = [r['confidence_score'] for r in results]
        total_predicted_qty = sum(quantities)
        
        logging.info(f"📊 Results Summary:")
        logging.info(f"   Quantities range: {min(quantities):.2f} to {max(quantities):.2f}")
        logging.info(f"   Total predicted: {total_predicted_qty:.2f}")
        logging.info(f"   Confidence range: {min(confidences):.2f} to {max(confidences):.2f}")
        logging.info(f"   Total employees: {total_employees}")
        
        logging.info("\n📋 Detailed Results:")
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('expected_qty', ascending=False)
        print(results_df.to_string(index=False))
        
        logging.info(f"\n🎯 COMPARISON EXPECTATION:")
        logging.info(f"   These results should now MATCH the API results!")
        logging.info(f"   Both use the same OpenAI + gender_guesser classification pipeline")
        
        return True

    except Exception as e:
        logging.error(f"❌ Consistent smoke test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    test_passed = asyncio.run(run_consistent_smoke_test())
    if not test_passed:
        sys.exit(1)