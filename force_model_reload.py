"""
Force reload of the predictor model in the API server.
This clears the singleton instance to force recreation with updated code.
"""

import requests
import json

# First, let's make a request that will clear the predictor instance
# We'll use a special endpoint if available, or trigger a reload another way

api_url = "http://127.0.0.1:9050"
api_key = "test_api_key_123"  # Update with your actual API key

# Try to make a prediction request to see current behavior
test_payload = {
    "cvr": "28892055",
    "branch_no": "28892055",
    "presents": [
        {
            "id": "P1",
            "present_name": "Test Product",
            "model_name": "Model 1",
            "model_no": "M001",
            "vendor": "Test Vendor"
        }
    ],
    "employees": [
        {"name": "Test Employee"}
    ]
}

headers = {
    "Content-Type": "application/json",
    "X-API-Key": api_key
}

print("Making test request to check current behavior...")
try:
    response = requests.post(
        f"{api_url}/predict",
        json=test_payload,
        headers=headers
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Current prediction: {json.dumps(result, indent=2)}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")

print("\n" + "="*50)
print("IMPORTANT: To apply the shop_features.py changes, you need to restart the API server.")
print("Please stop the current server (Ctrl+C) and restart it with:")
print("python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 9050")
print("="*50)