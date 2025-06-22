#!/usr/bin/env python3
"""
Debug script to test the prediction pipeline directly and see what's happening at each step.
"""

import sys
import os
import logging
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.predictor import GiftDemandPredictor, get_predictor

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Test data
test_presents = [
    {
        'id': '1',
        'item_main_category': 'Home & Kitchen',
        'item_sub_category': 'Kitchen Appliance',
        'color': 'NONE',
        'brand': 'Tisvilde',
        'vendor': 'GaveFabrikken',
        'target_demographic': 'unisex',
        'utility_type': 'practical',
        'usage_type': 'shareable',
        'durability': 'durable'
    },
    {
        'id': '2',
        'item_main_category': 'Health & Personal Care',
        'item_sub_category': 'Massage',
        'color': 'NONE',
        'brand': 'BodyCare',
        'vendor': 'Gavefabrikken',
        'target_demographic': 'unisex',
        'utility_type': 'practical',
        'usage_type': 'individual',
        'durability': 'durable'
    }
]

test_employees = [
    {'gender': 'male'},
    {'gender': 'male'},
    {'gender': 'female'},
    {'gender': 'female'},
    {'gender': 'male'}
]

def main():
    logger.info("Starting prediction pipeline debug...")
    
    # Initialize predictor
    model_path = "models/catboost_poisson_model/catboost_poisson_model.cbm"
    historical_data_path = "src/data/historical/present.selection.historic.csv"
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    actual_model_path = os.path.join(project_root, model_path)
    actual_historical_data_path = os.path.join(project_root, historical_data_path)
    
    if not os.path.exists(actual_model_path):
        logger.error(f"Model file not found at: {actual_model_path}")
        return
    
    logger.info("Initializing predictor...")
    predictor = get_predictor(
        model_path=actual_model_path,
        historical_data_path=actual_historical_data_path
    )
    
    # Test prediction
    branch = "28892055"
    logger.info(f"Making predictions for branch {branch}")
    logger.info(f"Presents: {len(test_presents)}")
    logger.info(f"Employees: {len(test_employees)}")
    
    # Make predictions
    predictions = predictor.predict(
        branch=branch,
        presents=test_presents,
        employees=test_employees
    )
    
    logger.info("\n=== PREDICTION RESULTS ===")
    total = 0
    for pred in predictions:
        logger.info(f"Product {pred.product_id}: {pred.expected_qty} (confidence: {pred.confidence_score})")
        total += pred.expected_qty
    
    logger.info(f"\nTotal predicted: {total}")
    logger.info(f"Total employees: {len(test_employees)}")
    logger.info(f"Ratio: {total / len(test_employees) if len(test_employees) > 0 else 0:.2f}")
    
    # Also test raw model output
    logger.info("\n=== TESTING RAW MODEL OUTPUT ===")
    
    # Create a simple feature vector
    test_features = predictor._create_feature_vector(
        test_presents[0], 
        'male', 
        branch,
        predictor.shop_resolver.get_shop_features(None, branch, test_presents[0])
    )
    
    logger.info(f"Sample features: {test_features}")
    
    # Create DataFrame
    import pandas as pd
    feature_df = pd.DataFrame([test_features])
    feature_df = predictor._add_interaction_features(feature_df)
    feature_df = predictor._prepare_features(feature_df)
    
    logger.info(f"Feature shape: {feature_df.shape}")
    logger.info(f"Feature columns: {list(feature_df.columns)[:10]}... ({len(feature_df.columns)} total)")
    
    # Make raw prediction
    raw_pred = predictor._make_catboost_prediction(feature_df)
    logger.info(f"Raw model output: {raw_pred}")
    
    # Check if model is loaded
    logger.info(f"\nModel info:")
    logger.info(f"  Model type: {type(predictor.model)}")
    logger.info(f"  Model exists: {predictor.model is not None}")
    if predictor.model:
        logger.info(f"  Model tree count: {predictor.model.tree_count_}")

if __name__ == "__main__":
    main()