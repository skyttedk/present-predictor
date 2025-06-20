#!/usr/bin/env python3
"""
Test script to verify on-the-fly classification is working properly.
"""

import asyncio
import requests
import json
import time

# Test configuration
API_BASE_URL = "http://127.0.0.1:8001"
API_KEY = "R5IVGj44RNjmuUjKsBXdQKnPz6_iJdJQqGNOkJTSIbA"

# Test data with products that are NOT in the database
test_data = {
    "cvr": "12345678",
    "presents": [
        {
            "id": 1001,
            "description": "Luxury Coffee Mug Set with Gold Trim",
            "model_name": "Premium Collection",
            "model_no": "PCM-2025",
            "vendor": "Elite Homeware"
        },
        {
            "id": 1002,
            "description": "Organic Cotton Bath Towel Set",
            "model_name": "Spa Series",
            "model_no": "SPA-TOWEL-01",
            "vendor": "Natural Living Co"
        },
        {
            "id": 1003,
            "description": "Smart Fitness Tracker Watch",
            "model_name": "FitPro 5000",
            "model_no": "FP5K-BLK",
            "vendor": "TechFit Solutions"
        }
    ],
    "employees": [
        {"name": "GUNHILD SØRENSEN"},
        {"name": "Per Christian Eidevik"},
        {"name": "Erik Nielsen"},
        {"name": "Anna Hansen"},
        {"name": "Lars Pedersen"}
    ]
}

def test_predict_endpoint():
    """Test the /predict endpoint with on-the-fly classification."""
    
    print("🔍 Testing /predict endpoint with on-the-fly classification")
    print("=" * 70)
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print(f"\n📊 Test Data:")
    print(f"   CVR: {test_data['cvr']}")
    print(f"   Presents: {len(test_data['presents'])} items (NOT in database)")
    print(f"   Employees: {len(test_data['employees'])} people")
    
    for present in test_data['presents']:
        print(f"\n   Present {present['id']}: {present['description']}")
    
    print(f"\n🚀 Sending request to {API_BASE_URL}/predict...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers=headers,
            json=test_data,
            timeout=60  # Longer timeout for classification
        )
        elapsed = (time.time() - start_time) * 1000
        
        print(f"⏱️  Response time: {elapsed:.0f}ms")
        print(f"📥 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ SUCCESS! Predictions received:")
            print(f"   Branch: {result['branch_no']}")
            print(f"   Total employees: {result['total_employees']}")
            print(f"   Processing time: {result['processing_time_ms']:.0f}ms")
            
            print(f"\n📊 Predictions:")
            print("-" * 60)
            print(f"{'Product ID':<12} {'Quantity':<10} {'Confidence':<10} {'Status':<15}")
            print("-" * 60)
            
            total_qty = 0
            unique_quantities = set()
            
            for pred in result['predictions']:
                total_qty += pred['expected_qty']
                unique_quantities.add(pred['expected_qty'])
                
                # Check if quantity is suspiciously uniform
                status = "⚠️ UNIFORM" if len(unique_quantities) == 1 and len(result['predictions']) > 2 else "✅ VARIED"
                
                print(f"{pred['product_id']:<12} {pred['expected_qty']:<10} {pred['confidence_score']:<10.2f} {status}")
            
            print("-" * 60)
            print(f"{'TOTAL':<12} {total_qty:<10}")
            
            # Analysis
            print(f"\n🔍 Analysis:")
            print(f"   Unique quantities: {len(unique_quantities)} different values")
            print(f"   Quantity values: {sorted(unique_quantities)}")
            print(f"   Average per product: {total_qty / len(result['predictions']):.1f}")
            print(f"   Selection rate: {total_qty / (len(result['predictions']) * result['total_employees']) * 100:.1f}%")
            
            if len(unique_quantities) == 1:
                print(f"\n❌ WARNING: All products have the SAME quantity!")
                print(f"   This indicates classification is not working properly.")
                print(f"   Products are likely getting default/NONE values.")
            else:
                print(f"\n✅ SUCCESS: Products have DIFFERENT quantities!")
                print(f"   This indicates classification is working properly.")
                print(f"   Each product is being classified with unique features.")
            
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            error_detail = response.json()
            print(f"   Detail: {json.dumps(error_detail, indent=2)}")
            
            if "missing_presents" in str(error_detail):
                print(f"\n⚠️  The endpoint is requiring pre-classification!")
                print(f"   On-the-fly classification is NOT working.")
            
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out after 60 seconds")
        print(f"   This might indicate OpenAI classification is hanging")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def check_database_for_test_products():
    """Check if test products exist in database."""
    print("\n\n📦 Checking if test products exist in database...")
    print("=" * 70)
    
    # This would need database access to check properly
    # For now, we'll just note that these are new products
    print("ℹ️  Test products are designed to NOT exist in database")
    print("   This forces on-the-fly classification to be used")

if __name__ == "__main__":
    # Make sure API is running
    print("⚠️  Make sure the API is running locally on port 8001!")
    print("   Run: uvicorn src.api.main:app --reload --port 8001")
    
    check_database_for_test_products()
    test_predict_endpoint()