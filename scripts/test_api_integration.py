#!/usr/bin/env python3
"""
Quick integration test for the ML-enabled /predict endpoint
"""

import requests
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_local_predict_endpoint():
    """Test the updated /predict endpoint locally"""
    
    # Local API configuration
    base_url = "http://127.0.0.1:8001"
    api_key = "R5IVGj44RNjmuUjKsBXdQKnPz6_iJdJQqGNOkJTSIbA"
    
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Test data matching our POC
    test_request = {
        "cvr": "12345678",
        "presents": [
            {
                "id": 147748,
                "description": "Travel Luggage White",
                "model_name": "Luggage Model",
                "model_no": "LUG001",
                "vendor": "Cavalluzzi"
            },
            {
                "id": 147757,
                "description": "Travel Backpack",
                "model_name": "Backpack Model", 
                "model_no": "BAG001",
                "vendor": "Bjørn Borg"
            }
        ],
        "employees": [
            {"name": "GUNHILD SØRENSEN"},
            {"name": "Per Christian Eidevik"},
            {"name": "Erik Nielsen"}
        ]
    }
    
    print("=== Testing ML-Enabled /predict Endpoint ===\n")
    print(f"Testing against: {base_url}/predict")
    print(f"Presents: {len(test_request['presents'])}")
    print(f"Employees: {len(test_request['employees'])}")
    
    try:
        # Test the endpoint
        response = requests.post(
            f"{base_url}/predict",
            headers=headers,
            json=test_request,
            timeout=30
        )
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! ML predictions received:")
            print(f"  Branch: {result.get('branch_no')}")
            print(f"  Total employees: {result.get('total_employees')}")
            print(f"  Processing time: {result.get('processing_time_ms', 0):.2f}ms")
            print("\nPredictions:")
            
            total_predicted = 0
            for pred in result.get('predictions', []):
                qty = pred.get('expected_qty', 0)
                conf = pred.get('confidence_score', 0)
                print(f"  Present {pred.get('product_id')}: {qty} units (confidence: {conf})")
                total_predicted += qty
            
            print(f"\nTotal predicted demand: {total_predicted} units")
            
            # Verify response format
            expected_format = [
                {"id": pred['product_id'], "quantity": pred['expected_qty']}
                for pred in result.get('predictions', [])
            ]
            
            print("\nFinal API Response Format:")
            print(json.dumps(expected_format, indent=2))
            
            return True
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to local API server")
        print("Please start the server with: uvicorn src.api.main:app --host 0.0.0.0 --port 8001")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_presents_exist():
    """Check if test presents are in the database"""
    
    base_url = "http://127.0.0.1:8001"
    api_key = "R5IVGj44RNjmuUjKsBXdQKnPz6_iJdJQqGNOkJTSIbA"
    
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Test presents
    test_presents = [
        {
            "present_name": "Travel Luggage White",
            "model_name": "Luggage Model",
            "model_no": "LUG001",
            "vendor": "Cavalluzzi"
        },
        {
            "present_name": "Travel Backpack",
            "model_name": "Backpack Model", 
            "model_no": "BAG001",
            "vendor": "Bjørn Borg"
        }
    ]
    
    print("=== Checking Present Database ===\n")
    
    try:
        for present in test_presents:
            print(f"Adding present: {present['present_name']}")
            response = requests.post(
                f"{base_url}/addPresent",
                headers=headers,
                json=present,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"  ✅ {result.get('message', 'Added successfully')}")
            else:
                print(f"  ❌ Failed: {response.status_code} - {response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking presents: {e}")
        return False

if __name__ == "__main__":
    print("Testing ML Integration for /predict endpoint\n")
    
    # First ensure presents exist
    if test_presents_exist():
        print("\n" + "="*50)
        
        # Then test the ML predictions
        if test_local_predict_endpoint():
            print("\n🎉 Integration test successful!")
            print("The /predict endpoint is working with ML predictions!")
        else:
            print("\n❌ Integration test failed")
    else:
        print("\n❌ Cannot proceed - present setup failed")