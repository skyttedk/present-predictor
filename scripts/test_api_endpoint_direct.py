#!/usr/bin/env python3
"""
Direct test of the API endpoint to verify if fixes are working
"""

import sys
import os
sys.path.append('src')

import asyncio
import json
from api.main import predict_endpoint
from api.schemas.requests import PredictRequest, PresentItem, EmployeeItem

def create_test_request():
    """Create test request matching the uniform prediction case"""
    return PredictRequest(
        cvr="28892055",
        presents=[
            PresentItem(
                id=i, 
                description=f"Test Product {i}", 
                model_name="Model", 
                model_no="001", 
                vendor="TestVendor"
            ) for i in range(1, 6)
        ],
        employees=[
            EmployeeItem(name=f"Employee {i}") for i in range(1, 58)
        ]
    )

async def test_api_endpoint():
    """Test the API endpoint directly"""
    print("🔍 Testing API Endpoint Directly")
    print("=" * 50)
    
    # Mock user for authentication
    mock_user = {'id': 1, 'username': 'test_user'}
    
    # Create test request
    request = create_test_request()
    
    try:
        print(f"📝 Request: {len(request.presents)} presents, {len(request.employees)} employees")
        print(f"   CVR: {request.cvr}")
        
        # Call the API endpoint function directly
        result = await predict_endpoint(request, mock_user)
        
        print("\n📊 API Response:")
        print(f"   Branch: {result.branch_no}")
        print(f"   Total employees: {result.total_employees}")
        print(f"   Processing time: {result.processing_time_ms:.2f}ms")
        
        # Extract prediction quantities
        quantities = [p.expected_qty for p in result.predictions]
        unique_quantities = sorted(set(quantities))
        
        print(f"\n🎯 Predictions:")
        print(f"   Quantities: {quantities}")
        print(f"   Unique values: {len(unique_quantities)} -> {unique_quantities}")
        print(f"   Range: {min(quantities)} to {max(quantities)}")
        
        # Check if varied or uniform
        if len(unique_quantities) == 1:
            print(f"\n❌ UNIFORM PREDICTIONS: All products = {unique_quantities[0]}")
            print("   API is NOT picking up the fixes!")
        else:
            print(f"\n✅ VARIED PREDICTIONS: Range {min(quantities)}-{max(quantities)}")
            print("   API is working correctly!")
            
        return result
        
    except Exception as e:
        print(f"\n❌ API Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_api_endpoint())
    
    if result:
        print(f"\n📋 Full Response Summary:")
        print(f"   Predictions count: {len(result.predictions)}")
        for i, pred in enumerate(result.predictions[:5]):  # Show first 5
            print(f"   Product {pred.product_id}: {pred.expected_qty} units (confidence: {pred.confidence_score})")
        if len(result.predictions) > 5:
            print(f"   ... and {len(result.predictions) - 5} more")