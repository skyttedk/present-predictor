#!/usr/bin/env python3
"""
Comprehensive test script for prediction functionality.
Tests both API endpoint and local predictor with proper error handling.
"""

import json
import logging
import requests
import time
import sys
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Optional

# Add src to path for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.predictor import GiftDemandPredictor, get_predictor
from src.api.schemas.responses import PredictionResult
from src.data.classifier import DataClassifier
from src.api.schemas.requests import PredictionRequest, GiftItem, Employee as APIEmployee
from src.config.settings import get_settings

# --- Configuration ---
MODEL_PATH = "models/catboost_poisson_model/catboost_poisson_model.cbm"
HISTORICAL_DATA_PATH = "src/data/historical/present.selection.historic.csv"
API_BASE_URL = "http://localhost:9050"
API_KEY = "31nl-iINTiAY4bZYlUq53h7qRQ3D_PHIS6aJGSKOYDQ"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Test Payload ---
test_payload = {
    "cvr": "28892055",
    "presents": [
        {
            "id": "1",
            "description": "Tisvilde Pizzaovn",
            "model_name": "Tisvilde Pizzaovn",
            "model_no": "",
            "vendor": "GaveFabrikken"
        },
        {
            "id": "2",
            "description": "BodyCare Massagepude",
            "model_name": "BodyCare Massagepude",
            "model_no": "",
            "vendor": "Gavefabrikken"
        },
        {
            "id": "3",
            "description": "Jesper Koch grill sæt, sort, 5 stk",
            "model_name": "Jesper Koch grill sæt, sort, 5 stk",
            "model_no": "",
            "vendor": "GaveFabrikken"
        },
        {
            "id": "4",
            "description": "Coleman men care skægtrimmer og travel shaver",
            "model_name": "Coleman men care skægtrimmer og travel shaver",
            "model_no": "",
            "vendor": "GaveFabrikken"
        },
        {
            "id": "5",
            "description": "FineSmile IQ tandbørste, silver",
            "model_name": "FineSmile IQ tandbørste, silver",
            "model_no": "",
            "vendor": "NDP Group"
        }
    ],
    "employees": [
        {"name": "Laimonas Lukosevicius"},
        {"name": "Petra De Laet"},
        {"name": "Regimantas Usas"},
        {"name": "Marius Melvold"},
        {"name": "Poul Torndahl"},
        {"name": "Hallvard Banken"},
        {"name": "Daniel Løsnesløkken"},
        {"name": "Magnus Ebenstrand"},
        {"name": "Tina Blaabjerg"},
        {"name": "Sanne Olsen"},
        {"name": "Thomas Hindhede"},
        {"name": "Magus Gupefjäll"},
        {"name": "Pål Aulesjord"},
        {"name": "Martin Berg"},
        {"name": "Henning Poulsen"},
        {"name": "Alexander Normann"},
        {"name": "Henrik Stærsholm"},
        {"name": "Ann-Sofie Åhman"},
        {"name": "Juha Södervik"},
        {"name": "Danny Borød"},
        {"name": "Kai Huhtala"},
        {"name": "Søren Pedersen"},
        {"name": "Tina Kindvall"},
        {"name": "Leila Eskandari"},
        {"name": "Johnny Winther"},
        {"name": "Mats Dovander"},
        {"name": "Patrik Larsson"},
        {"name": "Kia Dahl Andersen"},
        {"name": "Cecilie Dietiker Vonk"},
        {"name": "Trine Syversen"},
        {"name": "Anders Ahlstam"},
        {"name": "Dorthe Niewald"},
        {"name": "Malene Pedersen"},
        {"name": "Natasha Tyyskä"},
        {"name": "Åsa Hörnell"},
        {"name": "Reine Ringblom"},
        {"name": "Torben Villadsen"},
        {"name": "Otto Svensson"},
        {"name": "Vegard Hagen"},
        {"name": "Susanne Sundsberg"},
        {"name": "Ronny Israelsson"},
        {"name": "Kaj Nyman"},
        {"name": "Bengt Gerhardsen"},
        {"name": "Marianne List Nissen"},
        {"name": "Navneet Kaur"},
        {"name": "Jens Rask"},
        {"name": "Alexander Nylander"},
        {"name": "Jan Kristiansen"},
        {"name": "Marianne Wallanger"},
        {"name": "Tommy Sundsjöö"},
        {"name": "Lars Jagervall"},
        {"name": "Claus Andersen"},
        {"name": "Christer Jönsson"},
        {"name": "Thomas Olesen"},
        {"name": "Fredrik Alvarsson"},
        {"name": "Gvido Musperts"},
        {"name": "Søren Wander Jensen"}
    ]
}

def test_health_check():
    """Test if the server is running by checking endpoints."""
    health_urls = [
        f"{API_BASE_URL}/",
        f"{API_BASE_URL}/health",
        f"{API_BASE_URL}/docs"
    ]
    
    print("\nCHECKING SERVER AVAILABILITY:")
    
    for url in health_urls:
        try:
            response = requests.get(url, timeout=5)
            print(f"  ✓ {url} - Status: {response.status_code}")
            if response.status_code in [200, 404]:  # 404 is OK for some endpoints
                return True
        except requests.exceptions.RequestException:
            print(f"  ✗ {url} - Not responding")
    
    return False

def analyze_predictions(predictions: List[Dict], num_employees: int):
    """Analyze prediction results."""
    if not predictions:
        print("No predictions to analyze")
        return
    
    quantities = [p.get('expected_qty', 0) for p in predictions]
    total_expected = sum(quantities)
    
    print(f"\nPREDICTION ANALYSIS:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Total expected quantity: {total_expected:.1f}")
    print(f"  Number of employees: {num_employees}")
    print(f"  Ratio (expected/employees): {total_expected/num_employees:.2f}")
    
    if quantities:
        print(f"\nSTATISTICS:")
        print(f"  Min quantity: {min(quantities):.1f}")
        print(f"  Max quantity: {max(quantities):.1f}")
        print(f"  Average quantity: {sum(quantities)/len(quantities):.1f}")
        
        # Check for normalization
        if abs(total_expected - num_employees) < 0.1:
            print(f"  ✅ Total quantity correctly normalized to employee count")
        else:
            print(f"  ⚠️  Total quantity ({total_expected:.1f}) differs from employee count ({num_employees})")
        
        # Check for variation
        unique_values = set(quantities)
        if len(unique_values) == 1:
            print(f"  ⚠️  WARNING: All predictions are identical ({quantities[0]:.1f})")
        elif len(unique_values) < 3:
            print(f"  ⚠️  WARNING: Only {len(unique_values)} unique prediction values")
        else:
            print(f"  ✅ Good: {len(unique_values)} unique prediction values")

def test_api_endpoint():
    """Test the /predict API endpoint."""
    endpoint = f"{API_BASE_URL}/predict"
    
    print("\n" + "=" * 80)
    print(f"TESTING API ENDPOINT")
    print(f"URL: {endpoint}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Prepare headers with API key
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-API-Key": API_KEY
    }
    
    print(f"\nREQUEST SUMMARY:")
    print(f"CVR: {test_payload['cvr']}")
    print(f"Number of presents: {len(test_payload['presents'])}")
    print(f"Number of employees: {len(test_payload['employees'])}")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Make the request
        response = requests.post(
            endpoint,
            json=test_payload,
            headers=headers,
            timeout=120  # 2 minute timeout
        )
        
        # Record end time
        response_time = time.time() - start_time
        
        print(f"\nResponse received in {response_time:.2f} seconds")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                
                # Print raw response for debugging
                print(f"\nRAW RESPONSE DATA:")
                print(json.dumps(response_data, indent=2))
                
                # Handle both old (list) and new (dict) response formats
                if isinstance(response_data, dict):
                    print(f"✅ SUCCESS - Response format: dict")
                    
                    # Extract predictions from the dict
                    predictions = response_data.get('predictions', [])
                    total_employees = response_data.get('total_employees', len(test_payload['employees']))
                    
                    print(f"\nResponse contains:")
                    print(f"  - branch_no: {response_data.get('branch_no')}")
                    print(f"  - predictions: {len(predictions)} items")
                    print(f"  - total_employees: {total_employees}")
                    print(f"  - processing_time_ms: {response_data.get('processing_time_ms', 'N/A')}")
                    
                elif isinstance(response_data, list):
                    print(f"✅ SUCCESS - Response format: list (legacy)")
                    predictions = response_data
                    total_employees = len(test_payload['employees'])
                else:
                    print(f"❌ UNEXPECTED RESPONSE FORMAT: {type(response_data)}")
                    return
                
                # Analyze predictions
                analyze_predictions(predictions, total_employees)
                
                # Show sample predictions
                print(f"\nSAMPLE PREDICTIONS:")
                print(f"{'ID':<6} {'Expected Qty':<12} {'Description':<50}")
                print("-" * 70)
                
                for i, pred in enumerate(predictions[:5]):
                    product_id = pred.get('product_id', 'N/A')
                    expected_qty = pred.get('expected_qty', 0)
                    
                    # Find description
                    description = "Unknown"
                    for present in test_payload['presents']:
                        if str(present['id']) == str(product_id):
                            description = present['description'][:47] + "..." if len(present['description']) > 47 else present['description']
                            break
                    
                    print(f"{product_id:<6} {expected_qty:<12.2f} {description:<50}")
                
                if len(predictions) > 5:
                    print(f"... and {len(predictions) - 5} more")
                    
            except json.JSONDecodeError as e:
                print(f"❌ FAILED TO PARSE JSON RESPONSE")
                print(f"JSONDecodeError: {e}")
                print(f"Raw response: {response.text[:500]}...")
                
        else:
            print(f"❌ REQUEST FAILED - Status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw response: {response.text}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ CONNECTION ERROR: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

async def test_local_predictor():
    """Test the predictor directly without API."""
    print("\n" + "=" * 80)
    print("TESTING LOCAL PREDICTOR")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check for model and data files
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    actual_model_path = os.path.join(project_root, MODEL_PATH)
    actual_historical_data_path = os.path.join(project_root, HISTORICAL_DATA_PATH)
    
    if not os.path.exists(actual_model_path):
        logger.error(f"Model file not found at: {actual_model_path}")
        return
    
    if not os.path.exists(actual_historical_data_path):
        logger.warning(f"Historical data file not found at: {actual_historical_data_path}")
    
    try:
        # Step 1: Initialize classifier and predictor
        logger.info("Initializing DataClassifier...")
        classifier = DataClassifier()
        
        logger.info(f"Initializing predictor with model: {actual_model_path}")
        predictor = get_predictor(
            model_path=actual_model_path,
            historical_data_path=actual_historical_data_path
        )
        
        # Step 2: Prepare request
        api_gifts = [GiftItem(product_id=p["id"], description=p["description"]) for p in test_payload["presents"][:5]]  # Test with first 5
        api_employees = [APIEmployee(name=e["name"]) for e in test_payload["employees"]]
        
        prediction_request = PredictionRequest(
            branch_no=test_payload["cvr"],
            gifts=api_gifts,
            employees=api_employees
        )
        
        # Step 3: Classify data
        logger.info("Classifying request data (this may take time for OpenAI calls)...")
        classified_gifts, processed_employees, classification_stats = await classifier.classify_request(prediction_request)
        
        logger.info(f"Classification completed:")
        logger.info(f"  - Successful gifts: {classification_stats.successful_gifts}/{classification_stats.total_gifts}")
        logger.info(f"  - High confidence employees: {classification_stats.high_confidence_employees}/{classification_stats.total_employees}")
        
        # Step 4: Transform for predictor
        presents_for_predictor = []
        for cg in classified_gifts:
            present_dict = {"id": cg.product_id}
            if cg.attributes:
                present_dict.update(cg.attributes.model_dump())
            else:
                # Fallback attributes
                default_attrs = {
                    'item_main_category': 'NONE', 'item_sub_category': 'NONE', 
                    'brand': 'NONE', 'color': 'NONE', 'durability': 'durable',
                    'target_demographic': 'unisex', 'utility_type': 'practical', 
                    'usage_type': 'individual'
                }
                present_dict.update(default_attrs)
            presents_for_predictor.append(present_dict)
        
        # Fix: ProcessedEmployee already has gender as string, not enum
        employees_for_predictor = []
        for pe in processed_employees:
            # Check if gender is already a string or has .value attribute
            if hasattr(pe.gender, 'value'):
                gender = pe.gender.value
            else:
                gender = pe.gender
            employees_for_predictor.append({"name": pe.name, "gender": gender})
        
        # Step 5: Make predictions
        logger.info(f"Making predictions for {len(presents_for_predictor)} presents and {len(employees_for_predictor)} employees...")
        
        results: List[PredictionResult] = predictor.predict(
            branch=test_payload["cvr"],
            presents=presents_for_predictor,
            employees=employees_for_predictor
        )
        
        # Step 6: Analyze results
        if results:
            predictions_dict = [
                {"product_id": r.product_id, "expected_qty": r.expected_qty}
                for r in results
            ]
            analyze_predictions(predictions_dict, len(employees_for_predictor))
            
            print(f"\nDETAILED RESULTS:")
            for result in results:
                print(f"  Product {result.product_id}: {result.expected_qty:.2f} (confidence: {result.confidence_score:.2f})")
        else:
            logger.warning("No results returned from predictor")
            
    except Exception as e:
        logger.error(f"Error during local predictor test: {e}")
        import traceback
        traceback.print_exc()

async def main(test_mode: str = "both"):
    """Main test function."""
    print("=" * 80)
    print("COMPREHENSIVE PREDICTION TESTING")
    print(f"Test mode: {test_mode}")
    print("=" * 80)
    
    # Ensure settings are loaded
    settings = get_settings()
    
    if test_mode in ["api", "both"]:
        # Test API endpoint
        if test_health_check():
            print("\n✅ Server is running, proceeding with API test...")
            test_api_endpoint()
        else:
            print("\n❌ Server not available for API testing")
            if test_mode == "api":
                return
    
    if test_mode in ["local", "both"]:
        # Test local predictor
        print("\n\nTesting local predictor...")
        if not settings.openai.api_key:
            print("⚠️  WARNING: OPENAI_API_KEY not found. Classification may use fallbacks.")
        await test_local_predictor()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test prediction functionality")
    parser.add_argument(
        "--mode", 
        choices=["api", "local", "both"], 
        default="both",
        help="Which tests to run (default: both)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run async main function
    asyncio.run(main(args.mode))