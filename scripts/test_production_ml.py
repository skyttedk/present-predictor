#!/usr/bin/env python3
"""
Test the ML-enabled /predict endpoint in production
"""

import requests
import json
import time

def test_production_predict():
    """Test the ML predictions on production European Heroku app"""
    
    # Production API configuration
    base_url = "https://predict-presents-api-eu-0c5eca3623c6.herokuapp.com"
    api_key = "bFzT2ohBOgXHuHGESKYPOIx7D-y9qaZrQSNPF-jxcVY"
    
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Test data
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
            },
            {
                "id": 147758,
                "description": "Kitchen Cutlery Set",
                "model_name": "BBQ Set",
                "model_no": "BBQ001",
                "vendor": "Laguiole"
            }
        ],
        "employees": [
            {"name": "GUNHILD SØRENSEN"},
            {"name": "Per Christian Eidevik"},
            {"name": "Erik Nielsen"},
            {"name": "Anna Hansen"},
            {"name": "Michael Jensen"}
        ]
    }
    
    print("🚀 TESTING PRODUCTION ML INTEGRATION")
    print("="*50)
    print(f"Production URL: {base_url}")
    print(f"Presents: {len(test_request['presents'])}")
    print(f"Employees: {len(test_request['employees'])}")
    print()
    
    try:
        start_time = time.time()
        
        # Make request to production
        response = requests.post(
            f"{base_url}/predict",
            headers=headers,
            json=test_request,
            timeout=30
        )
        
        request_time = (time.time() - start_time) * 1000
        
        print(f"Response Status: {response.status_code}")
        print(f"Request Time: {request_time:.2f}ms")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ SUCCESS! Production ML Predictions:")
            print(f"  CVR: {test_request['cvr']}")
            print(f"  Branch: {result.get('branch_no')}")
            print(f"  Total employees: {result.get('total_employees')}")
            print(f"  Server processing time: {result.get('processing_time_ms', 0):.2f}ms")
            
            print("\n📊 ML Predictions:")
            total_predicted = 0
            predictions = result.get('predictions', [])
            
            for pred in predictions:
                qty = pred.get('expected_qty', 0)
                conf = pred.get('confidence_score', 0)
                print(f"  Present {pred.get('product_id')}: {qty} units (confidence: {conf})")
                total_predicted += qty
            
            print(f"\n🎯 Total predicted demand: {total_predicted} units")
            
            # Show final format
            final_format = [
                {"id": pred['product_id'], "quantity": pred['expected_qty']}
                for pred in predictions
            ]
            
            print("\n📋 Final API Response Format:")
            print(json.dumps(final_format, indent=2))
            
            # Performance assessment
            if request_time < 5000:  # Less than 5 seconds
                print(f"\n🚀 Performance: EXCELLENT ({request_time:.0f}ms)")
            elif request_time < 10000:  # Less than 10 seconds
                print(f"\n⚡ Performance: GOOD ({request_time:.0f}ms)")
            else:
                print(f"\n⏰ Performance: ACCEPTABLE ({request_time:.0f}ms)")
            
            # Validate predictions are reasonable
            if total_predicted > 0 and all(p.get('confidence_score', 0) > 0 for p in predictions):
                print("\n✅ ML INTEGRATION: FULLY OPERATIONAL")
                print("🎉 Production deployment successful!")
                return True
            else:
                print("\n⚠️  Warning: Some predictions may be suboptimal")
                return False
                
        else:
            print(f"\n❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_health_check():
    """Quick health check of production API"""
    
    base_url = "https://predict-presents-api-eu-0c5eca3623c6.herokuapp.com"
    api_key = "bFzT2ohBOgXHuHGESKYPOIx7D-y9qaZrQSNPF-jxcVY"
    
    headers = {"X-API-Key": api_key}
    
    try:
        response = requests.get(f"{base_url}/test", headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ Production API: HEALTHY")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("PRODUCTION ML INTEGRATION TEST")
    print("="*50)
    
    # Health check first
    if test_health_check():
        print()
        
        # Test ML predictions
        if test_production_predict():
            print("\n" + "="*50)
            print("🎉 PRODUCTION DEPLOYMENT: COMPLETE SUCCESS!")
            print("✅ ML-enabled /predict endpoint is live and operational")
            print("🚀 Ready for business use!")
        else:
            print("\n❌ Production ML test failed")
    else:
        print("\n❌ Cannot proceed - production API not accessible")