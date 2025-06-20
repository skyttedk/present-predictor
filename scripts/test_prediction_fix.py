"""
Test script to verify the prediction fix addresses the uniform prediction issue.
Tests with various scenarios to ensure predictions are now varied and reasonable.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.predictor import get_predictor
from src.api.schemas.responses import PredictionResult
import json

def test_predictions():
    """Test predictions with different scenarios"""
    
    print("Initializing predictor...")
    predictor = get_predictor()
    
    # Test scenarios with different employee counts and gift selections
    test_scenarios = [
        {
            "name": "Small company (20 employees)",
            "branch": "621000",
            "employees": [{"gender": "male"} for _ in range(12)] + [{"gender": "female"} for _ in range(8)],
            "presents": [
                {
                    "id": "P001",
                    "item_main_category": "Home & Kitchen",
                    "item_sub_category": "Cookware",
                    "brand": "Fiskars",
                    "color": "NONE",
                    "durability": "durable",
                    "target_demographic": "unisex",
                    "utility_type": "practical",
                    "usage_type": "individual"
                },
                {
                    "id": "P002",
                    "item_main_category": "Tools & DIY",
                    "item_sub_category": "Power Tools",
                    "brand": "Bosch",
                    "color": "NONE",
                    "durability": "durable",
                    "target_demographic": "male",
                    "utility_type": "practical",
                    "usage_type": "individual"
                },
                {
                    "id": "P003",
                    "item_main_category": "Home & Decor",
                    "item_sub_category": "Candles",
                    "brand": "Yankee",
                    "color": "NONE",
                    "durability": "consumable",
                    "target_demographic": "female",
                    "utility_type": "aesthetic",
                    "usage_type": "individual"
                }
            ]
        },
        {
            "name": "Large company (200 employees)",
            "branch": "841100",
            "employees": [{"gender": "male"} for _ in range(120)] + [{"gender": "female"} for _ in range(80)],
            "presents": [
                {
                    "id": "P101",
                    "item_main_category": "Home & Kitchen",
                    "item_sub_category": "Cookware",
                    "brand": "Fiskars",
                    "color": "NONE",
                    "durability": "durable",
                    "target_demographic": "unisex",
                    "utility_type": "practical",
                    "usage_type": "individual"
                },
                {
                    "id": "P102",
                    "item_main_category": "Travel",
                    "item_sub_category": "Hotel Stay",
                    "brand": "Comwell",
                    "color": "NONE",
                    "durability": "consumable",
                    "target_demographic": "unisex",
                    "utility_type": "exclusive",
                    "usage_type": "shareable"
                }
            ]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"Testing: {scenario['name']}")
        print(f"Branch: {scenario['branch']}")
        print(f"Employees: {len(scenario['employees'])} total")
        print(f"Presents: {len(scenario['presents'])} options")
        print("-" * 60)
        
        try:
            predictions = predictor.predict(
                branch=scenario['branch'],
                presents=scenario['presents'],
                employees=scenario['employees']
            )
            
            total_predicted = sum(p.expected_qty for p in predictions)
            print(f"\nPrediction Results:")
            print(f"{'Product ID':<10} {'Category':<20} {'Target':<10} {'Predicted':<10} {'Confidence':<10}")
            print("-" * 70)
            
            for i, pred in enumerate(predictions):
                present = scenario['presents'][i]
                print(f"{pred.product_id:<10} {present['item_main_category']:<20} "
                      f"{present['target_demographic']:<10} {pred.expected_qty:<10} "
                      f"{pred.confidence_score:<10.2f}")
            
            print(f"\nTotal predicted: {total_predicted} / {len(scenario['employees'])} employees")
            print(f"Average selection rate: {total_predicted / len(scenario['employees']):.2%}")
            
            # Check for uniform predictions (the issue we're trying to fix)
            unique_predictions = len(set(p.expected_qty for p in predictions))
            if unique_predictions == 1:
                print("⚠️  WARNING: All predictions are uniform! Fix may not be working.")
            else:
                print("✅ Predictions show variation - fix appears to be working!")
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Test completed!")

if __name__ == "__main__":
    test_predictions()