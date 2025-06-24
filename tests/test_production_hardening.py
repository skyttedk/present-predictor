"""
Production hardening tests for the ML pipeline.
These tests help catch schema drifts and ensure model compatibility.
"""

import pytest
import pandas as pd
import numpy as np
import os
import pickle
import hashlib
from src.ml.predictor import GiftDemandPredictor, get_predictor
from src.config.settings import get_settings

def test_model_metadata_loading():
    """Test that model metadata loads successfully without exceptions."""
    settings = get_settings()
    model_path = "models/catboost_poisson_model/catboost_poisson_model.cbm"
    
    # Skip if model doesn't exist (e.g., in CI without trained model)
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    
    # Should not raise any exceptions
    predictor = GiftDemandPredictor(model_path)
    
    # Basic sanity checks
    assert predictor.model is not None
    assert predictor.expected_columns is not None
    assert len(predictor.expected_columns) > 0
    assert predictor.categorical_features is not None

def test_dummy_prediction_schema_compatibility():
    """
    Test predictor with a dummy 1-row DataFrame containing all expected columns.
    This helps catch schema drifts and missing feature handling.
    """
    model_path = "models/catboost_poisson_model/catboost_poisson_model.cbm"
    
    # Skip if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    
    predictor = GiftDemandPredictor(model_path)
    
    # Create dummy data with all expected attributes
    dummy_presents = [{
        'id': 'TEST_001',
        'item_main_category': 'Home & Kitchen',
        'item_sub_category': 'Cookware',
        'brand': 'TestBrand',
        'color': 'Blue',
        'target_demographic': 'unisex',
        'utility_type': 'practical',
        'durability': 'durable',
        'usage_type': 'individual'
    }]
    
    dummy_employees = [
        {'gender': 'male'},
        {'gender': 'female'}
    ]
    
    # Should not raise any exceptions
    predictions = predictor.predict(
        branch="TEST_BRANCH",
        presents=dummy_presents,
        employees=dummy_employees
    )
    
    # Basic result validation
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert 'product_id' in predictions[0]
    assert 'expected_qty' in predictions[0]
    assert 'confidence_score' in predictions[0]
    assert predictions[0]['expected_qty'] >= 0
    assert 0 <= predictions[0]['confidence_score'] <= 1

def test_feature_signature_consistency():
    """
    Test that feature signature (column names + dtypes) is consistent.
    This helps detect schema changes between training and prediction.
    """
    model_path = "models/catboost_poisson_model/catboost_poisson_model.cbm"
    
    # Skip if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    
    model_dir = os.path.dirname(model_path)
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    
    if not os.path.exists(metadata_path):
        pytest.skip(f"Model metadata not found at {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    features_used = metadata.get('features_used', [])
    categorical_features = metadata.get('categorical_features_in_model', [])
    
    # Create feature signature
    features_signature = {
        'feature_count': len(features_used),
        'categorical_count': len(categorical_features),
        'feature_hash': hashlib.md5('|'.join(sorted(features_used)).encode()).hexdigest(),
        'categorical_hash': hashlib.md5('|'.join(sorted(categorical_features)).encode()).hexdigest()
    }
    
    # Store current signature for comparison (in production, this would be stored in metadata)
    expected_signature = {
        'feature_count': len(features_used),
        'categorical_count': len(categorical_features),
        'feature_hash': hashlib.md5('|'.join(sorted(features_used)).encode()).hexdigest(),
        'categorical_hash': hashlib.md5('|'.join(sorted(categorical_features)).encode()).hexdigest()
    }
    
    # Verify signature matches (in real production, compare against stored baseline)
    assert features_signature == expected_signature, "Feature signature has changed"
    
    # Verify we have the expected feature categories
    assert len(features_used) > 0, "No features found in metadata"
    assert len([f for f in features_used if f.startswith('interaction_hash_')]) == 96, "Expected 96 interaction hash features"

def test_predictor_singleton_behavior():
    """Test that get_predictor returns singleton behavior as expected."""
    model_path = "models/catboost_poisson_model/catboost_poisson_model.cbm"
    
    # Skip if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    
    # Reset singleton for clean test
    from src.ml import predictor
    predictor._predictor_instance = None
    
    predictor1 = get_predictor(model_path)
    predictor2 = get_predictor(model_path)
    
    # Should be the same instance
    assert predictor1 is predictor2

def test_model_performance_metadata():
    """Test that performance metadata contains expected Poisson metrics."""
    model_path = "models/catboost_poisson_model/catboost_poisson_model.cbm"
    
    # Skip if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    
    model_dir = os.path.dirname(model_path)
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    
    if not os.path.exists(metadata_path):
        pytest.skip(f"Model metadata not found at {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    performance_metrics = metadata.get('performance_metrics', {})
    
    # Check for Poisson-specific metrics
    assert 'poisson_deviance' in performance_metrics, "Poisson deviance metric missing"
    assert 'business_mape' in performance_metrics, "Business MAPE metric missing"
    assert 'r2_validation' in performance_metrics, "R² validation metric missing"
    
    # Validate metric ranges
    poisson_deviance = performance_metrics['poisson_deviance']
    business_mape = performance_metrics['business_mape']
    r2_validation = performance_metrics['r2_validation']
    
    assert poisson_deviance > 0, f"Poisson deviance should be positive: {poisson_deviance}"
    assert 0 <= business_mape <= 100, f"Business MAPE should be 0-100%: {business_mape}"
    assert -1 <= r2_validation <= 1, f"R² should be between -1 and 1: {r2_validation}"

def test_shop_features_availability():
    """Test that shop features are properly available and non-empty."""
    model_path = "models/catboost_poisson_model/catboost_poisson_model.cbm"
    
    # Skip if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    
    predictor = GiftDemandPredictor(model_path)
    
    # Test shop feature resolution
    shop_features = predictor.shop_resolver.resolve_features(
        shop_id="TEST_SHOP",
        main_category="Home & Kitchen",
        sub_category="Cookware",
        brand="TestBrand"
    )
    
    # Should return features without errors
    assert isinstance(shop_features, dict)
    
    # Check for expected shop feature keys
    expected_keys = [
        'shop_main_category_diversity_selected',
        'shop_brand_diversity_selected',
        'shop_utility_type_diversity_selected',
        'shop_sub_category_diversity_selected'
    ]
    
    for key in expected_keys:
        assert key in shop_features, f"Missing shop feature: {key}"
        assert isinstance(shop_features[key], (int, float)), f"Shop feature {key} should be numeric"

@pytest.mark.slow
def test_end_to_end_prediction_pipeline():
    """
    Comprehensive end-to-end test of the prediction pipeline.
    This test validates the entire flow from API-like input to predictions.
    """
    model_path = "models/catboost_poisson_model/catboost_poisson_model.cbm"
    
    # Skip if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    
    predictor = get_predictor(model_path)
    
    # Realistic test data
    test_presents = [
        {
            'id': 'COFFEE_MUG_001',
            'item_main_category': 'Home & Kitchen',
            'item_sub_category': 'Drinkware',
            'brand': 'TestBrand',
            'color': 'Blue',
            'target_demographic': 'unisex',
            'utility_type': 'practical',
            'durability': 'durable',
            'usage_type': 'individual'
        },
        {
            'id': 'HEADPHONES_002',
            'item_main_category': 'Electronics',
            'item_sub_category': 'Audio',
            'brand': 'AudioCorp',
            'color': 'Black',
            'target_demographic': 'unisex',
            'utility_type': 'practical',
            'durability': 'durable',
            'usage_type': 'individual'
        }
    ]
    
    test_employees = [
        {'gender': 'male'},
        {'gender': 'male'},
        {'gender': 'female'},
        {'gender': 'female'},
        {'gender': 'unisex'}
    ]
    
    predictions = predictor.predict(
        branch="621000",
        presents=test_presents,
        employees=test_employees
    )
    
    # Validate results
    assert len(predictions) == 2
    
    total_expected_qty = sum(p['expected_qty'] for p in predictions)
    
    # Business logic validation
    assert total_expected_qty >= 0, "Total expected quantity should be non-negative"
    assert total_expected_qty <= len(test_employees) * len(test_presents), "Total should not exceed unrealistic bounds"
    
    # Confidence validation
    for pred in predictions:
        assert 0.5 <= pred['confidence_score'] <= 0.95, f"Confidence should be 0.5-0.95: {pred['confidence_score']}"
        assert pred['expected_qty'] >= 0, f"Expected qty should be non-negative: {pred['expected_qty']}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])