"""
Investigate why the CatBoost model outputs uniform predictions.
This script will help identify if the issue is with:
1. The saved model vs training performance
2. Feature engineering mismatches
3. Categorical encoding issues
4. Model configuration problems
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.predictor import GiftDemandPredictor
from catboost import CatBoostRegressor, Pool

def load_model_metadata():
    """Load and examine model metadata"""
    metadata_path = "models/catboost_poisson_model/model_metadata.pkl"
    
    if not os.path.exists(metadata_path):
        print("❌ Model metadata not found!")
        return None
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print("📊 Model Metadata:")
    print(f"  Model Type: {metadata.get('model_type', 'Unknown')}")
    print(f"  Training Timestamp: {metadata.get('training_timestamp', 'Unknown')}")
    print(f"  Performance Metrics: {metadata.get('performance_metrics', {})}")
    print(f"  Features Used: {len(metadata.get('features_used', []))}")
    print(f"  Categorical Features: {len(metadata.get('categorical_features_in_model', []))}")
    
    return metadata

def test_model_on_training_like_data():
    """Test if model gives varied predictions on training-like data"""
    print("\n🔬 Testing model on varied synthetic data...")
    
    # Load the model directly
    model_path = "models/catboost_poisson_model/catboost_poisson_model.cbm"
    model = CatBoostRegressor()
    model.load_model(model_path)
    
    # Load metadata to get feature structure
    metadata_path = "models/catboost_poisson_model/model_metadata.pkl"
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    features_used = metadata.get('features_used', [])
    categorical_features = metadata.get('categorical_features_in_model', [])
    numeric_medians = metadata.get('numeric_feature_medians', {})
    
    print(f"  Expected features: {len(features_used)}")
    print(f"  Categorical features: {len(categorical_features)}")
    
    # Create synthetic test data with varied characteristics
    test_data = []
    
    # Very different synthetic examples
    examples = [
        {
            'employee_shop': '2960',
            'employee_branch': '621000', 
            'employee_gender': 'male',
            'product_main_category': 'Home & Kitchen',  # Popular category
            'product_sub_category': 'Cookware',
            'product_brand': 'Fiskars',  # Popular brand
            'product_color': 'NONE',
            'product_durability': 'durable',
            'product_target_gender': 'unisex',
            'product_utility_type': 'practical',
            'product_type': 'individual'
        },
        {
            'employee_shop': '2960',
            'employee_branch': '621000',
            'employee_gender': 'female', 
            'product_main_category': 'Wellness',  # Different category
            'product_sub_category': 'Spa Products',
            'product_brand': 'Unknown Brand',  # Rare brand
            'product_color': 'Pink',
            'product_durability': 'consumable',
            'product_target_gender': 'female',
            'product_utility_type': 'aesthetic',
            'product_type': 'individual'
        },
        {
            'employee_shop': '1234',  # Different shop
            'employee_branch': '841100',  # Different branch
            'employee_gender': 'male',
            'product_main_category': 'Tools & DIY',
            'product_sub_category': 'Power Tools',
            'product_brand': 'Bosch',
            'product_color': 'NONE',
            'product_durability': 'durable',
            'product_target_gender': 'male',
            'product_utility_type': 'practical', 
            'product_type': 'individual'
        }
    ]
    
    # Create DataFrame with all required features
    for example in examples:
        # Add shop features (using defaults)
        example.update({
            'shop_main_category_diversity_selected': numeric_medians.get('shop_main_category_diversity_selected', 35),
            'shop_brand_diversity_selected': numeric_medians.get('shop_brand_diversity_selected', 57),
            'shop_utility_type_diversity_selected': numeric_medians.get('shop_utility_type_diversity_selected', 4),
            'shop_sub_category_diversity_selected': numeric_medians.get('shop_sub_category_diversity_selected', 78),
            'unique_product_combinations_in_shop': numeric_medians.get('unique_product_combinations_in_shop', 11171),
            'shop_most_frequent_main_category_selected': 'Home & Kitchen',
            'shop_most_frequent_brand_selected': 'NONE',
            'is_shop_most_frequent_main_category': 1 if example['product_main_category'] == 'Home & Kitchen' else 0,
            'is_shop_most_frequent_brand': 0,
            'product_share_in_shop': numeric_medians.get('product_share_in_shop', 0.0001),
            'brand_share_in_shop': numeric_medians.get('brand_share_in_shop', 0.001),
            'product_rank_in_shop': numeric_medians.get('product_rank_in_shop', 100),
            'brand_rank_in_shop': numeric_medians.get('brand_rank_in_shop', 50),
        })
        
        # Add interaction hash features (zeros for simplicity)
        for i in range(10):
            example[f'interaction_hash_{i}'] = numeric_medians.get(f'interaction_hash_{i}', 0.0)
        
        test_data.append(example)
    
    # Create DataFrame
    test_df = pd.DataFrame(test_data)
    
    # Ensure all required features are present
    for feature in features_used:
        if feature not in test_df.columns:
            if feature in categorical_features:
                test_df[feature] = "NONE"
            else:
                test_df[feature] = numeric_medians.get(feature, 0.0)
    
    # Reorder columns to match training
    test_df = test_df[features_used]
    
    # Ensure correct types
    for feature in categorical_features:
        if feature in test_df.columns:
            test_df[feature] = test_df[feature].astype(str)
    
    print(f"  Test data shape: {test_df.shape}")
    print(f"  Columns: {list(test_df.columns)}")
    
    # Get categorical feature indices
    cat_feature_indices = [test_df.columns.get_loc(col) for col in categorical_features if col in test_df.columns]
    print(f"  Categorical indices: {cat_feature_indices}")
    
    # Create CatBoost Pool
    test_pool = Pool(
        data=test_df,
        cat_features=cat_feature_indices
    )
    
    # Make predictions
    predictions = model.predict(test_pool)
    predictions = np.maximum(0, predictions)
    
    print(f"\n📈 Direct Model Predictions:")
    for i, pred in enumerate(predictions):
        example = examples[i]
        print(f"  Example {i+1}: {example['product_main_category']} ({example['product_target_gender']}) → {pred:.6f}")
    
    # Analyze variation
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    pred_range = np.max(predictions) - np.min(predictions)
    
    print(f"\n📊 Prediction Statistics:")
    print(f"  Mean: {pred_mean:.6f}")
    print(f"  Std Dev: {pred_std:.6f}")
    print(f"  Range: {pred_range:.6f}")
    print(f"  Min: {np.min(predictions):.6f}")
    print(f"  Max: {np.max(predictions):.6f}")
    
    if pred_range < 0.01:
        print("❌ PROBLEM: Predictions are nearly identical!")
        print("   The model is not discriminating between very different inputs.")
    else:
        print("✅ Model shows some discrimination between inputs.")
    
    return predictions

def compare_feature_importance():
    """Load and analyze feature importance from training"""
    print("\n🎯 Feature Importance Analysis:")
    
    metadata_path = "models/catboost_poisson_model/model_metadata.pkl"
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    importance = metadata.get('feature_importance_summary', {})
    
    if importance:
        print("  Top 10 Important Features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_features[:10]):
            print(f"    {i+1:2d}. {feature:<40} {score:8.2f}")
    else:
        print("  ❌ No feature importance data found!")

def check_model_training_logs():
    """Check if there are any training logs available"""
    print("\n📋 Checking for Training Logs:")
    
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("catboost_training_*.log"))
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            print(f"  Latest training log: {latest_log}")
            
            # Read last 20 lines
            with open(latest_log, 'r') as f:
                lines = f.readlines()
            
            print("  Last 20 lines of training log:")
            for line in lines[-20:]:
                print(f"    {line.strip()}")
        else:
            print("  No training logs found.")
    else:
        print("  Logs directory not found.")

def main():
    print("🔍 Investigating Model Issues")
    print("=" * 60)
    
    # 1. Load and examine metadata
    metadata = load_model_metadata()
    if not metadata:
        return
    
    # 2. Test model on synthetic data
    predictions = test_model_on_training_like_data()
    
    # 3. Analyze feature importance
    compare_feature_importance()
    
    # 4. Check training logs
    check_model_training_logs()
    
    print("\n" + "=" * 60)
    print("Investigation complete!")

if __name__ == "__main__":
    main()