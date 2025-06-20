#!/usr/bin/env python3
"""
Fix script to add missing products for classification before making predictions.
"""

import requests
import json

# Production API details
BASE_URL = "https://predict-presents-api-eu-0c5eca3623c6.herokuapp.com"
API_KEY = "bFzT2ohBOgXHuHGESKYPOIx7D-y9qaZrQSNPF-jxcVY"

def add_sample_products():
    """Add sample products from your request to the database for classification"""
    
    # Sample products from your actual request
    products_to_add = [
        {
            "present_name": "Urban Copenhagen In-ears beige",
            "model_name": "",
            "model_no": "",
            "vendor": ""
        },
        {
            "present_name": "Urban Copenhagen In-ears grøn", 
            "model_name": "",
            "model_no": "",
            "vendor": ""
        },
        {
            "present_name": "Urban Copenhagen In-ears sort",
            "model_name": "",
            "model_no": "",
            "vendor": ""
        },
        {
            "present_name": "Cavalluzzi kabine trolley, hvid",
            "model_name": "Cavalluzzi kabine trolley, hvid",
            "model_no": "",
            "vendor": "TravelGear"
        },
        {
            "present_name": "Bjørn Borg ryggsekk",
            "model_name": "Bjørn Borg ryggsekk", 
            "model_name": "",
            "model_no": "",
            "vendor": "Bjørn Borg"
        }
    ]
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print("🔄 Adding products for classification...")
    print("=" * 50)
    
    for i, product in enumerate(products_to_add, 1):
        print(f"\n{i}. Adding: {product['present_name']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/addPresent",
                headers=headers,
                json=product,
                timeout=30
            )
            
            if response.status_code == 201:
                result = response.json()
                print(f"   ✅ Added successfully (ID: {result.get('id')})")
                print(f"   📝 Hash: {result.get('present_hash')}")
            elif response.status_code == 200:
                result = response.json()
                print(f"   ℹ️  Already exists (ID: {result.get('existing_id')})")
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n⏳ Products added for classification. Wait 2-5 minutes for OpenAI classification...")
    print(f"💡 Check classification status with: GET {BASE_URL}/countPresents")

def check_classification_status():
    """Check how many products are classified vs pending"""
    
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(f"{BASE_URL}/countPresents", headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📊 Classification Status:")
            print(f"   Total presents: {result['total_presents']}")
            
            for status in result['status_counts']:
                print(f"   {status['status']}: {status['count']}")
        else:
            print(f"❌ Failed to check status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")

if __name__ == "__main__":
    add_sample_products()
    check_classification_status()