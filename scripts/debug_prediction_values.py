"""
Debug script to examine raw model outputs and understand the scale issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.predictor import GiftDemandPredictor
import numpy as np
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def debug_predictions():
    """Debug the prediction pipeline to see raw values"""
    
    print("Initializing predictor...")
    predictor = GiftDemandPredictor(
        model_path="models/catboost_poisson_model/catboost_poisson_model.cbm",
        historical_data_path="src/data/historical/present.selection.historic.csv"
    )
    
    # Simple test case
    branch = "621000"
    shop_id = branch[:4]  # "6210"
    
    presents = [{
        "id": "TEST001",
        "item_main_category": "Home & Kitchen",
        "item_sub_category": "Cookware",
        "brand": "Fiskars",
        "color": "NONE",
        "durability": "durable",
        "target_demographic": "unisex",
        "utility_type": "practical",
        "usage_type": "individual"
    }]
    
    employees = [{"gender": "male"} for _ in range(10)] + [{"gender": "female"} for _ in range(10)]
    
    print(f"\nTest scenario:")
    print(f"Branch: {branch}, Shop ID: {shop_id}")
    print(f"Employees: 20 total (10 male, 10 female)")
    print(f"Present: {presents[0]['item_main_category']} - {presents[0]['brand']}")
    
    # Get employee stats
    employee_stats = predictor._calculate_employee_stats(employees)
    print(f"\nEmployee stats: {employee_stats}")
    
    # Get shop features
    try:
        shop_features = predictor.shop_resolver.get_shop_features(shop_id, branch, presents[0])
        print(f"\nShop features retrieved: {list(shop_features.keys())}")
    except Exception as e:
        print(f"Error getting shop features: {e}")
        shop_features = {}
    
    # Create feature vectors
    rows = []
    for gender, ratio in employee_stats.items():
        if ratio > 0:
            features = predictor._create_feature_vector(
                presents[0], gender, shop_id, branch, shop_features
            )
            features['employee_ratio'] = ratio
            rows.append(features)
    
    print(f"\nCreated {len(rows)} feature vectors (one per gender)")
    
    # Prepare features
    import pandas as pd
    feature_df = pd.DataFrame(rows)
    feature_df = predictor._add_interaction_features(feature_df)
    employee_ratios = feature_df['employee_ratio'].values
    feature_df = feature_df.drop(columns=['employee_ratio'])
    feature_df = predictor._prepare_features(feature_df)
    
    print(f"\nFeature DataFrame shape: {feature_df.shape}")
    print(f"Columns: {list(feature_df.columns)[:10]}... (showing first 10)")
    
    # Make raw predictions
    try:
        raw_predictions = predictor._make_catboost_prediction(feature_df)
        print(f"\n🔍 RAW MODEL OUTPUTS: {raw_predictions}")
        print(f"   Min: {np.min(raw_predictions):.6f}")
        print(f"   Max: {np.max(raw_predictions):.6f}")
        print(f"   Mean: {np.mean(raw_predictions):.6f}")
        print(f"   Sum: {np.sum(raw_predictions):.6f}")
        
        # Test different scaling factors
        print("\n📊 Testing different scaling factors:")
        for scale in [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]:
            scaled_sum = np.sum(raw_predictions) * scale
            final = max(0, min(scaled_sum, 20))  # 20 employees
            print(f"   Scale {scale:4.2f}: raw_sum={np.sum(raw_predictions):.2f} → "
                  f"scaled={scaled_sum:.2f} → final={final:.0f}")
        
        # Show what the current implementation would produce
        total_prediction = predictor._aggregate_predictions(raw_predictions, 20)
        print(f"\n📌 Current implementation result: {total_prediction}")
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_predictions()