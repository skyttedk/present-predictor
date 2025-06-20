"""
Test with the specific branch from the user's API test to see why it's still uniform
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.predictor import get_predictor

def test_specific_branch():
    """Test with the exact branch number from the user's API test"""
    
    print("🔍 Testing Specific Branch from API Test")
    print("=" * 60)
    
    # Get the same predictor instance the API uses
    predictor = get_predictor()
    
    # Use the exact branch from the user's test
    branch = "28892055"
    shop_id = branch[:4]  # "2889"
    
    print(f"Testing with branch {branch}, shop {shop_id}")
    
    # Test different product types to see if they get different features
    test_products = [
        {
            "id": "1",
            "item_main_category": "Home & Kitchen",
            "brand": "Fiskars",
            "item_sub_category": "Cookware",
            "color": "NONE",
            "durability": "durable",
            "target_demographic": "unisex",
            "utility_type": "practical",
            "usage_type": "individual"
        },
        {
            "id": "2", 
            "item_main_category": "Tools & DIY",
            "brand": "Bosch",
            "item_sub_category": "Power Tools",
            "color": "NONE",
            "durability": "durable", 
            "target_demographic": "male",
            "utility_type": "practical",
            "usage_type": "individual"
        },
        {
            "id": "3",
            "item_main_category": "Travel",
            "brand": "Comwell",
            "item_sub_category": "Hotel Stay",
            "color": "NONE",
            "durability": "consumable",
            "target_demographic": "unisex",
            "utility_type": "exclusive", 
            "usage_type": "shareable"
        }
    ]
    
    employees = [{"gender": "male"} for _ in range(30)] + [{"gender": "female"} for _ in range(27)]
    print(f"Total employees: {len(employees)}")
    
    print("\nTesting shop feature resolution for each product:")
    print("-" * 60)
    
    for product in test_products:
        shop_features = predictor.shop_resolver.get_shop_features(shop_id, branch, product)
        
        print(f"\nProduct {product['id']}: {product['item_main_category']} - {product['brand']}")
        critical_features = [
            'product_share_in_shop',
            'brand_share_in_shop', 
            'product_rank_in_shop',
            'brand_rank_in_shop'
        ]
        
        for feature in critical_features:
            value = shop_features.get(feature, "NOT FOUND")
            print(f"  {feature:<25}: {value}")
    
    print("\n" + "=" * 60)
    print("Making predictions for all products...")
    
    try:
        predictions = predictor.predict(
            branch=branch,
            presents=test_products,
            employees=employees
        )
        
        print(f"\nPrediction Results:")
        print(f"{'Product':<10} {'Category':<15} {'Brand':<10} {'Predicted':<10}")
        print("-" * 50)
        
        for pred in predictions:
            # Find the corresponding product
            product = next(p for p in test_products if p['id'] == pred.product_id)
            print(f"{pred.product_id:<10} {product['item_main_category']:<15} {product['brand']:<10} {pred.expected_qty:<10}")
        
        # Check if all predictions are the same
        quantities = [p.expected_qty for p in predictions]
        if len(set(quantities)) == 1:
            print(f"\n❌ PROBLEM: All predictions are uniform ({quantities[0]} units)")
            print("   This suggests the branch/shop has no historical data")
        else:
            print(f"\n✅ Predictions show variation: {quantities}")
            
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

def check_branch_in_data():
    """Check if this branch exists in historical data"""
    print("\n🔍 Checking if branch exists in historical data")
    print("-" * 60)
    
    # Check if we can find this branch in the shop resolver
    predictor = get_predictor()
    resolver = predictor.shop_resolver
    
    branch = "28892055"
    shop_id = "2889"
    
    print(f"Available branches: {len(resolver.get_available_branches())}")
    print(f"Available shops: {len(resolver.get_available_shops())}")
    
    if branch in resolver.get_available_branches():
        print(f"✅ Branch {branch} found in historical data")
        shop_count = resolver.get_shop_count_by_branch(branch)
        print(f"   {shop_count} shops in this branch")
    else:
        print(f"❌ Branch {branch} NOT found in historical data")
        print("   This explains why all products get default features")
    
    if shop_id in resolver.get_available_shops():
        print(f"✅ Shop {shop_id} found in historical data")
    else:
        print(f"❌ Shop {shop_id} NOT found in historical data")

if __name__ == "__main__":
    test_specific_branch()
    check_branch_in_data()