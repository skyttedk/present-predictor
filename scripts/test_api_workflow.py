"""
Test the exact API workflow to identify where uniform predictions come from
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.predictor import get_predictor
from src.data.classifier import DataClassifier
from src.data.openai_client import create_openai_client
from src.data.gender_classifier import classify_employee_gender
from src.api.schemas.requests import GiftItem
import hashlib

async def test_full_api_workflow():
    """Test the complete API workflow that might cause uniform predictions"""
    
    print("🔍 Testing Full API Workflow")
    print("=" * 60)
    
    # Simulate the exact request data structure from the user's test
    branch = "28892055"
    
    # Create fake presents like the API would receive (without classification)
    fake_presents = [
        {"id": i, "description": f"Product {i}", "model_name": f"Model{i}", "model_no": f"No{i}", "vendor": "TestVendor"}
        for i in range(1, 20)  # 19 products like in the user's test
    ]
    
    fake_employees = [
        {"name": f"Employee {i}"} for i in range(57)  # 57 employees like in user's test
    ]
    
    print(f"Simulating API request:")
    print(f"  Branch: {branch}")
    print(f"  Products: {len(fake_presents)}")
    print(f"  Employees: {len(fake_employees)}")
    
    # Step 1: Classify presents (like the API does)
    print("\n1. Classifying presents...")
    
    try:
        openai_client = create_openai_client()
        classifier = DataClassifier(openai_client)
    except Exception as e:
        print(f"   Warning: OpenAI client failed: {e}")
        classifier = DataClassifier()  # Fallback
    
    transformed_presents = []
    
    for present in fake_presents:
        # This mimics the API's classification logic
        present_hash = hashlib.md5(f"{present['description']} - {present['model_name']} - {present['model_no']}.".encode('utf-8')).hexdigest().upper()
        
        # For testing, let's use default attributes (like when classification fails)
        attributes = {
            'item_main_category': 'NONE',
            'item_sub_category': 'NONE', 
            'color': 'NONE',
            'brand': 'NONE',
            'vendor': 'NONE',
            'target_demographic': 'unisex',
            'utility_type': 'practical',
            'usage_type': 'individual',
            'durability': 'durable'
        }
        
        present_dict = {
            'id': present['id'],
            'item_main_category': attributes['item_main_category'],
            'item_sub_category': attributes['item_sub_category'],
            'color': attributes['color'],
            'brand': attributes['brand'],
            'vendor': attributes['vendor'],
            'target_demographic': attributes['target_demographic'],
            'utility_type': attributes['utility_type'],
            'usage_type': attributes['usage_type'],
            'durability': attributes['durability']
        }
        transformed_presents.append(present_dict)
    
    print(f"   Classified {len(transformed_presents)} presents")
    print(f"   Sample classification: {transformed_presents[0]}")
    
    # Step 2: Classify employees (like the API does)
    print("\n2. Classifying employees...")
    
    transformed_employees = []
    for employee in fake_employees:
        gender = classify_employee_gender(employee['name'])
        transformed_employees.append({
            'gender': gender.value  # Convert enum to string
        })
    
    gender_counts = {}
    for emp in transformed_employees:
        gender = emp['gender']
        gender_counts[gender] = gender_counts.get(gender, 0) + 1
    
    print(f"   Employee gender distribution: {gender_counts}")
    
    # Step 3: Make predictions (like the API does)
    print("\n3. Making predictions...")
    
    predictor = get_predictor()
    
    # Test shop feature resolution for the default attributes
    sample_present = transformed_presents[0]
    shop_id = branch[:4]
    shop_features = predictor.shop_resolver.get_shop_features(shop_id, branch, sample_present)
    
    print(f"   Shop features for default product:")
    critical_features = ['product_share_in_shop', 'brand_share_in_shop', 'product_rank_in_shop', 'brand_rank_in_shop']
    for feature in critical_features:
        value = shop_features.get(feature, "NOT FOUND")
        print(f"     {feature:<25}: {value}")
    
    try:
        predictions = predictor.predict(
            branch=branch,
            presents=transformed_presents,
            employees=transformed_employees
        )
        
        print(f"\n📊 Prediction Results:")
        quantities = [p.expected_qty for p in predictions]
        unique_quantities = set(quantities)
        
        print(f"   Total predictions: {len(predictions)}")
        print(f"   Unique quantities: {len(unique_quantities)} -> {sorted(unique_quantities)}")
        print(f"   First 10 predictions: {quantities[:10]}")
        
        if len(unique_quantities) == 1:
            print(f"\n❌ UNIFORM PREDICTIONS: All products = {quantities[0]} units")
            print("   This matches the user's API test result!")
            
            # Analyze why
            print("\n🔍 Analysis:")
            if all(p['item_main_category'] == 'NONE' for p in transformed_presents):
                print("   - All products have category 'NONE' (classification failed)")
            if all(p['brand'] == 'NONE' for p in transformed_presents):
                print("   - All products have brand 'NONE' (classification failed)")
            
            product_shares = [shop_features.get('product_share_in_shop', 0) for _ in transformed_presents]
            if all(share == 0 for share in product_shares):
                print("   - All products get product_share = 0.0 (no fallback match)")
            
        else:
            print(f"\n✅ VARIED PREDICTIONS: Range {min(quantities)} to {max(quantities)}")
            
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await test_full_api_workflow()

if __name__ == "__main__":
    asyncio.run(main())