#!/usr/bin/env python3
"""
Force API Predictor Reload Script

This script forces the API server to reload its predictor instance
by calling a special admin endpoint that clears the cached model.
"""

import httpx
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.settings import settings

async def trigger_predictor_load(api_key: str, api_url: str):
    """Make a prediction to force predictor loading."""
    try:
        async with httpx.AsyncClient() as client:
            test_payload = {
                "cvr": "28892055",
                "presents": [{"id": "test", "description": "test", "model_name": "test", "model_no": "test", "vendor": "test"}],
                "employees": [{"name": "Test"}]
            }
            headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
            response = await client.post(f"{api_url}/predict", headers=headers, json=test_payload, timeout=30.0)
            
            if response.status_code == 200:
                print("✅ Predictor loaded via test prediction")
                return True
            else:
                print(f"⚠️ Test prediction failed with status {response.status_code}")
                return False
                
    except Exception as e:
        print(f"⚠️ Test prediction error: {e}")
        return False

async def force_api_reload():
    """Force the API server to reload its cached predictor instance."""
    
    # Use the provided API key
    api_key = "31nl-iINTiAY4bZYlUq53h7qRQ3D_PHIS6aJGSKOYDQ"
    api_url = "http://localhost:9050"  # Updated to match your running API
    headers = {"X-API-Key": api_key}
    
    async with httpx.AsyncClient() as client:
        try:
            # First test if API is accessible
            print("🔍 Testing API connectivity...")
            response = await client.get(f"{api_url}/test", headers=headers, timeout=10.0)
            if response.status_code != 200:
                print(f"❌ Error: API test failed with status {response.status_code}")
                return False
            print("✅ API is accessible")
            
            # Step 1: Trigger predictor loading 
            print("🔄 Step 1: Triggering predictor load...")
            await trigger_predictor_load(api_key, api_url)
            
            # Step 2: Call the force reload endpoint
            print("🔄 Step 2: Forcing API predictor reload...")
            response = await client.post(f"{api_url}/admin/force-reload-predictor", headers=headers, timeout=30.0)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API predictor reload successful!")
                print(f"📝 Message: {result.get('message', 'No message')}")
                return True
            else:
                print(f"❌ Error: Force reload failed with status {response.status_code}")
                print(f"📝 Response: {response.text}")
                return False
                
        except httpx.TimeoutException:
            print("❌ Error: Request timed out - API server may not be running")
            return False
        except httpx.ConnectError:
            print("❌ Error: Could not connect to API server - is it running on localhost:8000?")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

if __name__ == "__main__":
    import asyncio
    
    print("=== Force API Predictor Reload ===")
    print("This script will force the API server to reload its cached predictor instance")
    print("with the latest retrained model (log-exposure fix).")
    print()
    
    success = asyncio.run(force_api_reload())
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Test the API endpoint again with the same data")
        print("2. Verify predictions now match smoke test results (wider range, higher totals)")
        sys.exit(0)
    else:
        print("\n❌ Failed to reload API predictor. Check:")
        print("1. API server is running on localhost:8000")
        print("2. Admin API key is correct")
        print("3. /admin/force-reload-predictor endpoint exists")
        sys.exit(1)