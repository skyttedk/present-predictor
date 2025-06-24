#!/usr/bin/env python3
"""
Quick test script to validate that the CatBoost Poisson model loads correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
from ml.predictor import get_predictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_predictor_loading():
    """Test if the predictor loads without errors"""
    try:
        logger.info("Testing predictor loading...")
        
        # Get predictor instance (should load the Poisson model)
        predictor = get_predictor()
        
        logger.info("✅ Predictor loaded successfully!")
        logger.info(f"Model path: {predictor.model_path}")
        logger.info(f"Model RMSE: {predictor.model_rmse}")
        logger.info(f"Numeric medians loaded: {len(predictor.numeric_medians)} features")
        logger.info(f"Expected columns: {len(predictor.expected_columns)} features")
        logger.info(f"Categorical features: {len(predictor.categorical_features)} features")
        
        # Test basic functionality with minimal data
        test_presents = [{
            'id': 'TEST001',
            'item_main_category': 'Home & Kitchen',
            'item_sub_category': 'Cookware',
            'brand': 'TestBrand',
            'color': 'NONE',
            'durability': 'durable',
            'target_demographic': 'unisex',
            'utility_type': 'practical',
            'usage_type': 'individual'
        }]
        
        test_employees = [
            {'gender': 'male'},
            {'gender': 'female'}
        ]
        
        logger.info("Testing prediction with sample data...")
        results = predictor.predict(
            branch="621000",
            presents=test_presents,
            employees=test_employees
        )
        
        logger.info(f"✅ Prediction successful!")
        logger.info(f"Result: {results[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Predictor loading/testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_predictor_loading()
    exit(0 if success else 1)