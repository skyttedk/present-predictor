"""
Debug shop feature resolution to see if product-specific features
are being calculated correctly during inference.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.predictor import GiftDemandPredictor
from src.ml.shop_features import ShopFeatureResolver
import pandas as pd

def debug_shop_feature_resolution():
    """Debug how shop features are resolved for different products"""
    
    print("🔍 Debugging Shop Feature Resolution")
    print("=" * 60)
    
    # Initialize components
    resolver = ShopFeatureResolver("src/data/historical/present.selection.historic.csv")
    
    # Test different products
    test_products = [
        {
            "id": "P001",
            "item_main_category": "Home & Kitchen",
            "item_sub_category": "Cookware",
            "brand": "Fiskars",
            "color": "NONE",
            "durability": "durable",
            "target_demographic": "unisex",
            "utility_type": "practical",
            "usage_type": "individual"
        },
        {
            "id": "P002",
            "item_main_category": "Tools & DIY",
            "item_sub_category": "Power Tools", 
            "brand": "Bosch",
            "color": "NONE",
            "durability": "durable",
            "target_demographic": "male",
            "utility_type": "practical",
            "usage_type": "individual"
        },
        {
            "id": "P003",
            "item_main_category": "Wellness",
            "item_sub_category": "Spa Products",
            "brand": "Unknown Brand",
            "color": "Pink",
            "durability": "consumable",
            "target_demographic": "female",
            "utility_type": "aesthetic",
            "usage_type": "individual"
        }
    ]
    
    shop_id = "6210"
    branch = "621000"
    
    print(f"Testing shop {shop_id}, branch {branch}")
    print("-" * 60)
    
    for product in test_products:
        print(f"\nProduct: {product['id']} - {product['item_main_category']}")
        print(f"  Brand: {product['brand']}")
        print(f"  Target: {product['target_demographic']}")
        
        # Get shop features for this specific product
        features = resolver.get_shop_features(shop_id, branch, product)
        
        # Focus on the critical features
        critical_features = [
            'product_share_in_shop',
            'brand_share_in_shop', 
            'product_rank_in_shop',
            'brand_rank_in_shop',
            'shop_most_frequent_main_category_selected',
            'shop_most_frequent_brand_selected'
        ]
        
        print("  Critical Features:")
        for feature in critical_features:
            value = features.get(feature, "NOT FOUND")
            print(f"    {feature:<40}: {value}")
    
    print("\n" + "=" * 60)
    
    # Check the product relativity lookup
    print("📊 Checking Product Relativity Lookup Table")
    lookup_path = "models/catboost_poisson_model/product_relativity_features.csv"
    
    if os.path.exists(lookup_path):
        lookup_df = pd.read_csv(lookup_path)
        print(f"  Lookup table has {len(lookup_df)} rows")
        print(f"  Columns: {list(lookup_df.columns)}")
        
        # Show some example entries
        print("\n  Sample entries:")
        sample_entries = lookup_df.head(10)
        for _, row in sample_entries.iterrows():
            print(f"    Shop: {row.get('employee_shop', 'N/A')}, "
                  f"Category: {row.get('product_main_category', 'N/A')}, "
                  f"Brand: {row.get('product_brand', 'N/A')}, "
                  f"Share: {row.get('product_share_in_shop', 'N/A'):.6f}")
        
        # Check for specific combinations from our test
        print("\n  Looking for test product combinations:")
        for product in test_products:
            matches = lookup_df[
                (lookup_df['product_main_category'] == product['item_main_category']) &
                (lookup_df['product_brand'] == product['brand'])
            ]
            print(f"    {product['item_main_category']} + {product['brand']}: {len(matches)} matches")
            if len(matches) > 0:
                sample_match = matches.iloc[0]
                print(f"      Example - Share: {sample_match.get('product_share_in_shop', 0):.6f}, "
                      f"Rank: {sample_match.get('product_rank_in_shop', 0)}")
    else:
        print("  ❌ Lookup table not found!")

def compare_training_vs_inference_features():
    """Compare what features look like during training vs inference"""
    
    print("\n🔄 Comparing Training vs Inference Features")
    print("=" * 60)
    
    # Load a sample from the product relativity lookup (represents training data)
    lookup_path = "models/catboost_poisson_model/product_relativity_features.csv"
    
    if not os.path.exists(lookup_path):
        print("❌ Cannot compare - lookup table missing")
        return
    
    lookup_df = pd.read_csv(lookup_path)
    
    # Get a sample entry
    sample_training = lookup_df.iloc[0]
    print("📚 Training Data Sample:")
    print(f"  Shop: {sample_training.get('employee_shop')}")
    print(f"  Branch: {sample_training.get('employee_branch')}")
    print(f"  Category: {sample_training.get('product_main_category')}")
    print(f"  Brand: {sample_training.get('product_brand')}")
    print(f"  Product Share: {sample_training.get('product_share_in_shop', 0):.6f}")
    print(f"  Brand Share: {sample_training.get('brand_share_in_shop', 0):.6f}")
    print(f"  Product Rank: {sample_training.get('product_rank_in_shop', 0)}")
    print(f"  Brand Rank: {sample_training.get('brand_rank_in_shop', 0)}")
    
    # Try to replicate this during inference
    resolver = ShopFeatureResolver("src/data/historical/present.selection.historic.csv")
    
    test_product = {
        "item_main_category": sample_training.get('product_main_category'),
        "brand": sample_training.get('product_brand'),
        "item_sub_category": sample_training.get('product_sub_category', 'Unknown'),
        "color": sample_training.get('product_color', 'NONE'),
        "durability": sample_training.get('product_durability', 'durable'),
        "target_demographic": sample_training.get('product_target_gender', 'unisex'),
        "utility_type": sample_training.get('product_utility_type', 'practical'),
        "usage_type": sample_training.get('product_type', 'individual')
    }
    
    inference_features = resolver.get_shop_features(
        sample_training.get('employee_shop'),
        sample_training.get('employee_branch'),
        test_product
    )
    
    print("\n🔮 Inference Features for Same Product:")
    print(f"  Product Share: {inference_features.get('product_share_in_shop', 'NOT FOUND')}")
    print(f"  Brand Share: {inference_features.get('brand_share_in_shop', 'NOT FOUND')}")
    print(f"  Product Rank: {inference_features.get('product_rank_in_shop', 'NOT FOUND')}")
    print(f"  Brand Rank: {inference_features.get('brand_rank_in_shop', 'NOT FOUND')}")
    
    # Compare
    print("\n📊 Comparison:")
    training_prod_share = sample_training.get('product_share_in_shop', 0)
    inference_prod_share = inference_features.get('product_share_in_shop', 0)
    
    if abs(training_prod_share - inference_prod_share) < 1e-6:
        print("✅ Product shares match!")
    else:
        print(f"❌ Product share mismatch: training={training_prod_share:.6f}, inference={inference_prod_share}")

def main():
    debug_shop_feature_resolution()
    compare_training_vs_inference_features()

if __name__ == "__main__":
    main()