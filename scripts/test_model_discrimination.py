"""
Test if the model can discriminate between different product types.
This will help us understand if the uniform predictions are due to:
1. The model not being properly trained
2. Feature engineering issues
3. The scaling problem
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.predictor import GiftDemandPredictor
import numpy as np
import pandas as pd

def test_model_discrimination():
    """Test model's ability to discriminate between very different products"""
    
    print("Initializing predictor...")
    predictor = GiftDemandPredictor(
        model_path="models/catboost_poisson_model/catboost_poisson_model.cbm",
        historical_data_path="src/data/historical/present.selection.historic.csv"
    )
    
    # Create very different products
    test_products = [
        {
            "id": "POPULAR",
            "item_main_category": "Home & Kitchen",  # Most frequent category
            "item_sub_category": "Cookware",
            "brand": "Fiskars",  # Popular brand
            "color": "NONE",
            "durability": "durable",
            "target_demographic": "unisex",
            "utility_type": "practical",
            "usage_type": "individual"
        },
        {
            "id": "MALE_FOCUSED",
            "item_main_category": "Tools & DIY",
            "item_sub_category": "Power Tools",
            "brand": "Bosch",
            "color": "NONE", 
            "durability": "durable",
            "target_demographic": "male",  # Male-specific
            "utility_type": "practical",
            "usage_type": "individual"
        },
        {
            "id": "FEMALE_FOCUSED",
            "item_main_category": "Wellness",
            "item_sub_category": "Spa Products",
            "brand": "L'Occitane",
            "color": "Pink",
            "durability": "consumable",
            "target_demographic": "female",  # Female-specific
            "utility_type": "aesthetic",
            "usage_type": "individual"
        },
        {
            "id": "LUXURY",
            "item_main_category": "Travel",
            "item_sub_category": "Hotel Stay",
            "brand": "Comwell",
            "color": "NONE",
            "durability": "consumable",
            "target_demographic": "unisex",
            "utility_type": "exclusive",  # Luxury item
            "usage_type": "shareable"
        },
        {
            "id": "OBSCURE",
            "item_main_category": "Obscure Category",
            "item_sub_category": "Unknown Sub",
            "brand": "Unknown Brand",
            "color": "Rainbow",
            "durability": "durable",
            "target_demographic": "unisex",
            "utility_type": "sentimental",
            "usage_type": "individual"
        }
    ]
    
    branch = "621000"
    shop_id = branch[:4]
    employees = [{"gender": "male"} for _ in range(60)] + [{"gender": "female"} for _ in range(40)]
    employee_stats = predictor._calculate_employee_stats(employees)
    
    print(f"\nTest Setup:")
    print(f"Branch: {branch}, Shop: {shop_id}")
    print(f"Employees: 100 (60% male, 40% female)")
    print(f"\nTesting {len(test_products)} very different products...")
    print("="*90)
    
    all_predictions = []
    
    for product in test_products:
        # Get shop features for this product
        shop_features = predictor.shop_resolver.get_shop_features(shop_id, branch, product)
        
        # Create feature vectors for each gender
        rows = []
        for gender, ratio in employee_stats.items():
            if ratio > 0:
                features = predictor._create_feature_vector(
                    product, gender, shop_id, branch, shop_features
                )
                features['employee_ratio'] = ratio
                rows.append(features)
        
        # Prepare features
        feature_df = pd.DataFrame(rows)
        feature_df = predictor._add_interaction_features(feature_df)
        employee_ratios = feature_df['employee_ratio'].values
        feature_df = feature_df.drop(columns=['employee_ratio'])
        feature_df = predictor._prepare_features(feature_df)
        
        # Get raw predictions
        raw_predictions = predictor._make_catboost_prediction(feature_df)
        
        # Aggregate
        total_raw = np.sum(raw_predictions)
        scaled = total_raw * 4.0  # Current scaling factor
        final = max(0, min(scaled, 100))
        
        all_predictions.append({
            'product_id': product['id'],
            'category': product['item_main_category'],
            'target': product['target_demographic'],
            'utility': product['utility_type'],
            'raw_male': raw_predictions[0] if len(raw_predictions) > 0 else 0,
            'raw_female': raw_predictions[1] if len(raw_predictions) > 1 else 0,
            'raw_sum': total_raw,
            'scaled': scaled,
            'final': int(round(final))
        })
        
        print(f"\nProduct: {product['id']:<15} ({product['item_main_category']}, {product['target_demographic']})")
        print(f"  Raw predictions: male={raw_predictions[0]:.4f}, female={raw_predictions[1] if len(raw_predictions) > 1 else 0:.4f}")
        print(f"  Sum: {total_raw:.4f} → Scaled: {scaled:.2f} → Final: {int(round(final))}")
    
    # Analyze variation
    print("\n" + "="*90)
    print("ANALYSIS:")
    
    df_results = pd.DataFrame(all_predictions)
    print(f"\nRaw prediction statistics:")
    print(f"  Mean raw sum: {df_results['raw_sum'].mean():.4f}")
    print(f"  Std dev: {df_results['raw_sum'].std():.4f}")
    print(f"  Min: {df_results['raw_sum'].min():.4f}")
    print(f"  Max: {df_results['raw_sum'].max():.4f}")
    print(f"  Range: {df_results['raw_sum'].max() - df_results['raw_sum'].min():.4f}")
    
    # Check if predictions vary
    unique_finals = df_results['final'].nunique()
    if unique_finals == 1:
        print(f"\n❌ PROBLEM: All final predictions are identical ({df_results['final'].iloc[0]})")
        print("   The model is not discriminating between products!")
    else:
        print(f"\n✅ Good: Predictions show {unique_finals} different values")
        print(f"   Range: {df_results['final'].min()} to {df_results['final'].max()}")
    
    # Show sorted results
    print(f"\nPredictions sorted by quantity:")
    sorted_df = df_results.sort_values('final', ascending=False)
    for _, row in sorted_df.iterrows():
        print(f"  {row['product_id']:<15} ({row['target']:<7}, {row['utility']:<10}): {row['final']:3d} units")

if __name__ == "__main__":
    test_model_discrimination()