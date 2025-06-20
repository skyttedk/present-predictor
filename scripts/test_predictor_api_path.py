#!/usr/bin/env python3
"""
Test the exact predictor path used by the API
"""

import sys
import os
sys.path.append('src')

# Direct imports without relative imports
from ml.predictor import get_predictor

def test_predictor_directly():
    """Test the predictor that the API calls"""
    print("🔍 Testing Predictor Used by API")
    print("=" * 50)
    
    try:
        # Get predictor instance (same call as API)
        predictor = get_predictor()
        print("✅ Predictor instance created successfully")
        
        # Test data matching your API call
        branch = "28892055"
        
        # Create presents with NONE values (like API classification failures)
        presents = [
            {
                'id': i,
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
            for i in range(1, 6)
        ]
        
        # Create employees
        employees = [{'gender': 'unisex'} for _ in range(57)]
        
        print(f"📝 Test data: {len(presents)} presents, {len(employees)} employees")
        print(f"   Branch: {branch}")
        print(f"   Sample present: {presents[0]}")
        
        # Make predictions
        predictions = predictor.predict(branch, presents, employees)
        
        # Extract quantities
        quantities = [p.expected_qty for p in predictions]
        unique_quantities = sorted(set(quantities))
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Quantities: {quantities}")
        print(f"   Unique values: {len(unique_quantities)} -> {unique_quantities}")
        print(f"   Range: {min(quantities)} to {max(quantities)}")
        
        # Check results
        if len(unique_quantities) == 1:
            print(f"\n❌ UNIFORM PREDICTIONS: All products = {unique_quantities[0]}")
            print("   The fix is NOT working through predictor!")
        else:
            print(f"\n✅ VARIED PREDICTIONS: Range {min(quantities)}-{max(quantities)}")
            print("   The fix IS working through predictor!")
            
        return predictions
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    predictions = test_predictor_directly()
    
    if predictions:
        print(f"\n📋 Detailed Results:")
        for pred in predictions:
            print(f"   Product {pred.product_id}: {pred.expected_qty} units (confidence: {pred.confidence_score})")