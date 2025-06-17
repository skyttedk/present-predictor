#!/usr/bin/env python3
"""
Test script for the /predict endpoint.
This script tests the new predict endpoint with sample data.
"""

import requests
import json
import sys

# Configuration
BASE_URL = "http://127.0.0.1:8001"  # Local development server
API_KEY = "R5IVGj44RNjmuUjKsBXdQKnPz6_iJdJQqGNOkJTSIbA"  # Local API key

# Sample test data
SAMPLE_PREDICT_REQUEST = {
    "cvr": "34445605",  # A real Danish CVR number for testing
    "presents": [
        {
            "id": 147748,
            "description": "Test Mug",
            "model_name": "Ceramic Blue",
            "model_no": "CM-001",
            "vendor": "MugCo"
        },
        {
            "id": 147757,
            "description": "Kitchen Knife Set",
            "model_name": "Stainless Steel",
            "model_no": "KNF-005",
            "vendor": "KitchenPro"
        }
    ],
    "employees": [
        {"name": "GUNHILD SØRENSEN"},
        {"name": "Per Christian Eidevik"},
        {"name": "Ine Opedal"},
        {"name": "Julie Østby-Svendsen"}
    ]
}

# Present to add first (so we have something to predict)
SAMPLE_ADD_PRESENT = {
    "present_name": "Test Mug",
    "model_name": "Ceramic Blue",
    "model_no": "CM-001",
    "vendor": "MugCo"
}

def test_api_connection():
    """Test basic API connection"""
    print("🔍 Testing API connection...")
    try:
        response = requests.get(
            f"{BASE_URL}/test",
            headers={"X-API-Key": API_KEY},
            timeout=10
        )
        if response.status_code == 200:
            print("✅ API connection successful")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ API connection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ API connection error: {e}")
        return False

def add_test_present():
    """Add a test present for classification"""
    print("\n📦 Adding test present...")
    try:
        response = requests.post(
            f"{BASE_URL}/addPresent",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json=SAMPLE_ADD_PRESENT,
            timeout=10
        )
        if response.status_code in [200, 201]:
            print("✅ Test present added successfully")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"⚠️  Present add response: {response.status_code}")
            print(f"   Response: {response.text}")
            return True  # May already exist, that's OK
    except Exception as e:
        print(f"❌ Error adding present: {e}")
        return False

def test_predict_endpoint():
    """Test the new /predict endpoint"""
    print("\n🎯 Testing /predict endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json=SAMPLE_PREDICT_REQUEST,
            timeout=30  # CVR API call may take time
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Predict endpoint successful!")
            result = response.json()
            
            print(f"📊 Results:")
            print(f"   Branch: {result.get('branch', 'N/A')}")
            print(f"   Presents: {len(result.get('presents', []))}")
            print(f"   Employees: {len(result.get('employees', []))}")
            
            # Show sample transformations
            if result.get('presents'):
                print(f"\n📦 Sample Present Transformation:")
                present = result['presents'][0]
                for key, value in present.items():
                    print(f"   {key}: {value}")
            
            if result.get('employees'):
                print(f"\n👥 Employee Gender Classifications:")
                for i, emp in enumerate(result['employees']):
                    original_name = SAMPLE_PREDICT_REQUEST['employees'][i]['name']
                    print(f"   {original_name} → {emp['gender']}")
            
            return True
        else:
            print(f"❌ Predict endpoint failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"   Raw response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing predict endpoint: {e}")
        return False

def test_missing_present():
    """Test predict endpoint with missing present"""
    print("\n🚫 Testing with missing present...")
    
    missing_request = {
        "cvr": "34445605",
        "presents": [
            {
                "id": 999999,
                "description": "Unknown Present",
                "model_name": "Unknown Model",
                "model_no": "UNK-001",
                "vendor": "Unknown Vendor"
            }
        ],
        "employees": [
            {"name": "John Doe"}
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json=missing_request,
            timeout=30
        )
        
        if response.status_code == 400:
            print("✅ Correctly handled missing present")
            error_detail = response.json()
            if "missing_presents" in str(error_detail):
                print("✅ Missing presents properly listed in error")
            else:
                print("⚠️  Missing presents not in expected format")
            return True
        else:
            print(f"⚠️  Unexpected response for missing present: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing missing present: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting /predict endpoint tests...\n")
    
    # Test basic connection
    if not test_api_connection():
        print("❌ Basic API connection failed. Make sure the server is running.")
        sys.exit(1)
    
    # Add test present
    if not add_test_present():
        print("❌ Failed to add test present. Continuing anyway...")
    
    # Test predict endpoint
    predict_success = test_predict_endpoint()
    
    # Test missing present handling
    missing_success = test_missing_present()
    
    # Summary
    print(f"\n📋 Test Summary:")
    print(f"   ✅ API Connection: Passed")
    print(f"   {'✅' if predict_success else '❌'} Predict Endpoint: {'Passed' if predict_success else 'Failed'}")
    print(f"   {'✅' if missing_success else '❌'} Missing Present Handling: {'Passed' if missing_success else 'Failed'}")
    
    if predict_success and missing_success:
        print(f"\n🎉 All tests passed! The /predict endpoint is working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()