"""
Test the API predictor instance to see if it's using the old or new ShopFeatureResolver
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.predictor import get_predictor

def test_api_predictor_instance():
    """Test if the API predictor instance has the fix"""
    
    print("🔍 Testing API Predictor Instance")
    print("=" * 60)
    
    # Get the same predictor instance the API uses
    predictor = get_predictor()
    
    # Test the shop feature resolution
    shop_id = "6210"
    branch = "621000"
    
    test_product = {
        "id": "TEST001",
        "item_main_category": "Home & Kitchen",
        "item_sub_category": "Cookware",
        "brand": "Fiskars",
        "color": "NONE",
        "durability": "durable",
        "target_demographic": "unisex",
        "utility_type": "practical",
        "usage_type": "individual"
    }
    
    print(f"Testing with shop {shop_id}, branch {branch}")
    print(f"Product: {test_product['item_main_category']} - {test_product['brand']}")
    
    # Get shop features 
    shop_features = predictor.shop_resolver.get_shop_features(shop_id, branch, test_product)
    
    print(f"\nShop features from API predictor:")
    critical_features = [
        'product_share_in_shop',
        'brand_share_in_shop', 
        'product_rank_in_shop',
        'brand_rank_in_shop'
    ]
    
    for feature in critical_features:
        value = shop_features.get(feature, "NOT FOUND")
        print(f"  {feature:<25}: {value}")
    
    # Check if fix is applied
    product_share = shop_features.get('product_share_in_shop', 0.0)
    if product_share > 0.0:
        print("\n✅ Fix is applied! Product share > 0")
    else:
        print("\n❌ Fix NOT applied! Product share = 0 (old behavior)")
        print("   The API is using a cached predictor instance.")
        print("   Solution: Restart the API server to pick up changes.")
    
    # Test actual prediction
    employees = [{"gender": "male"} for _ in range(10)] + [{"gender": "female"} for _ in range(10)]
    
    try:
        prediction = predictor.predict(
            branch=branch,
            presents=[test_product],
            employees=employees
        )
        
        if prediction and len(prediction) > 0:
            pred_qty = prediction[0].expected_qty
            print(f"\n📊 Prediction result: {pred_qty} units")
            
            if pred_qty > 8:  # Should be higher than default with fix
                print("✅ Prediction shows variation (fix working)")
            else:
                print("❌ Prediction shows default value (fix not working)")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")

if __name__ == "__main__":
    test_api_predictor_instance()