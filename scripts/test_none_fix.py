"""
Test the NONE classification fix with a fresh predictor instance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.predictor import GiftDemandPredictor

def test_none_classification_fix():
    """Test the fix for NONE classifications with fresh instance"""
    
    print("🔍 Testing NONE Classification Fix")
    print("=" * 60)
    
    # Create a fresh predictor instance (not cached)
    predictor = GiftDemandPredictor(
        model_path="models/catboost_rmse_model/catboost_rmse_model.cbm",
        historical_data_path="src/data/historical/present.selection.historic.csv"
    )
    
    branch = "28892055"
    shop_id = "2889"
    
    # Test products with NONE classifications (like failed API classification)
    none_products = [
        {
            "id": f"{i}",
            "item_main_category": "NONE",
            "item_sub_category": "NONE", 
            "color": "NONE",
            "brand": "NONE",
            "vendor": "NONE",
            "target_demographic": "unisex",
            "utility_type": "practical",
            "usage_type": "individual",
            "durability": "durable"
        }
        for i in range(1, 6)  # 5 products
    ]
    
    employees = [{"gender": "unisex"} for _ in range(57)]
    
    print(f"Testing {len(none_products)} products with NONE classifications")
    print(f"Branch: {branch}, Shop: {shop_id}")
    print(f"Employees: {len(employees)}")
    
    # Test shop feature resolution for NONE products
    print("\nShop feature resolution for NONE products:")
    print("-" * 40)
    
    for i, product in enumerate(none_products):
        shop_features = predictor.shop_resolver.get_shop_features(shop_id, branch, product)
        
        print(f"\nProduct {product['id']}:")
        critical_features = ['product_share_in_shop', 'brand_share_in_shop', 'product_rank_in_shop', 'brand_rank_in_shop']
        for feature in critical_features:
            value = shop_features.get(feature, "NOT FOUND")
            print(f"  {feature:<25}: {value}")
    
    # Make predictions
    print("\n" + "=" * 60)
    print("Making predictions...")
    
    try:
        predictions = predictor.predict(
            branch=branch,
            presents=none_products,
            employees=employees
        )
        
        quantities = [p.expected_qty for p in predictions]
        unique_quantities = set(quantities)
        
        print(f"\nPrediction Results:")
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Quantities: {quantities}")
        print(f"  Unique quantities: {len(unique_quantities)} -> {sorted(unique_quantities)}")
        
        if len(unique_quantities) == 1:
            print(f"\n❌ STILL UNIFORM: All products = {quantities[0]} units")
            print("   The random sampling fix may not be working")
        else:
            print(f"\n✅ SUCCESS: Varied predictions from {min(quantities)} to {max(quantities)}")
            print("   Random sampling fix is working!")
            
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_none_classification_fix()