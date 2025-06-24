#!/usr/bin/env python3

"""
Compare classifications between smoke test manual data and API classification results
"""

import os
import sys
import json
import requests

# Add src to the Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

def get_smoke_test_data():
    """Get the manually classified data from smoke test"""
    
    # Raw presents from smoke test
    raw_presents = [
        {"id": "1", "description": "Tisvilde Pizzaovn", "model_name": "Tisvilde Pizzaovn", "model_no": "", "vendor": "GaveFabrikken"},
        {"id": "2", "description": "BodyCare Massagepude", "model_name": "BodyCare Massagepude", "model_no": "", "vendor": "Gavefabrikken"},
        {"id": "3", "description": "Jesper Koch grill sæt, sort, 5 stk", "model_name": "Jesper Koch grill sæt, sort, 5 stk", "model_no": "", "vendor": "GaveFabrikken"}
    ]
    
    # Manual classifications from smoke test
    manual_classifications = [
        {"id": "1", "item_main_category": "Home & Kitchen", "item_sub_category": "Outdoor Cooking", "brand": "Tisvilde", "color": "NONE", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "shareable"},
        {"id": "2", "item_main_category": "Health & Beauty", "item_sub_category": "Massage", "brand": "BodyCare", "color": "NONE", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "3", "item_main_category": "Home & Kitchen", "item_sub_category": "Grilling", "brand": "Jesper Koch", "color": "sort", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"}
    ]
    
    return raw_presents, manual_classifications

def get_api_classifications():
    """Get API classification results by calling the prediction endpoint"""
    
    raw_presents, _ = get_smoke_test_data()
    
    # Create API payload with just first 3 presents for comparison
    payload = {
        "cvr": "28892055",
        "presents": [
            {"id": p["id"], "description": p["description"], "model_name": p["model_name"], "model_no": p["model_no"], "vendor": p["vendor"]}
            for p in raw_presents[:3]
        ],
        "employees": [
            {"name": "Laimonas Lukosevicius"},
            {"name": "Petra De Laet"},
            {"name": "Regimantas Usas"}
        ]
    }
    
    try:
        print("🔍 Calling API to get classifications...")
        headers = {
            "X-API-Key": "31nl-iINTiAY4bZYlUq53h7qRQ3D_PHIS6aJGSKOYDQ",
            "Content-Type": "application/json"
        }
        response = requests.post("http://127.0.0.1:9050/predict", json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API call successful - got {len(result.get('predictions', []))} predictions")
            return payload, result
        else:
            print(f"❌ API call failed: {response.status_code} - {response.text}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error calling API: {e}")
        return None, None

def compare_classifications():
    """Compare manual vs API classifications"""
    
    print("🔍 Comparing Smoke Test Manual Classifications vs API Classifications")
    print("=" * 80)
    
    # Get data
    raw_presents, manual_classifications = get_smoke_test_data()
    api_payload, api_result = get_api_classifications()
    
    if not api_result:
        print("❌ Could not get API classifications for comparison")
        return
    
    print("\n📊 COMPARISON RESULTS:")
    print("-" * 50)
    
    for i, present in enumerate(raw_presents[:3]):
        print(f"\n🎁 Present {i+1}: {present['description']}")
        print(f"   Raw: {present}")
        
        manual = manual_classifications[i]
        print(f"\n   📝 MANUAL Classification:")
        for key, value in manual.items():
            if key != 'id':
                print(f"      {key}: {value}")
        
        print(f"\n   🤖 API would classify this and predict: (see API predictions)")
        
    if 'predictions' in api_result:
        print(f"\n🎯 API PREDICTIONS (using real classifications):")
        for pred in api_result['predictions'][:3]:
            print(f"   Product {pred['product_id']}: {pred['expected_qty']:.2f} (conf: {pred.get('confidence_score', 'N/A')})")
            
    print(f"\n🔍 KEY INSIGHT:")
    print(f"   The smoke test uses MANUAL classifications (hardcoded)")
    print(f"   The API uses REAL classifications (OpenAI Assistant)")
    print(f"   Different classifications → Different feature vectors → Different predictions!")
    
    print(f"\n💡 SOLUTION:")
    print(f"   1. Use the same classification pipeline in both tests")
    print(f"   2. OR: Update manual classifications to match API results")
    print(f"   3. OR: Test with identical pre-classified data in both flows")

if __name__ == "__main__":
    compare_classifications()