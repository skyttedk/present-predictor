#!/usr/bin/env python3
"""
Smoke test for the prediction system.
Tests the end-to-end prediction pipeline by calling the API endpoint.
"""

import sys
import requests
import json
import logging
import pandas as pd
from pathlib import Path

# --- Configuration ---
API_KEY = "31nl-iINTiAY4bZYlUq53h7qRQ3D_PHIS6aJGSKOYDQ"  # Replace with actual API key

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def create_test_payload():
    """Create the exact test payload as provided by user."""
    # Exact payload as provided by user
    request_data = {
        "cvr": "28892055",
        "presents": [
            {
                "id": 1,
                "description": "Tisvilde Pizzaovn",
                "model_name": "Tisvilde Pizzaovn",
                "model_no": "",
                "vendor": "GaveFabrikken",
                "order_count": 7
            },
            {
                "id": 2,
                "description": "BodyCare Massagepude",
                "model_name": "BodyCare Massagepude",
                "model_no": "",
                "vendor": "Gavefabrikken",
                "order_count": 9
            },
            {
                "id": 3,
                "description": "Jesper Koch grill sæt, sort, 5 stk",
                "model_name": "Jesper Koch grill sæt, sort, 5 stk",
                "model_no": "",
                "vendor": "GaveFabrikken",
                "order_count": 1
            },
            {
                "id": 4,
                "description": "Coleman men care skægtrimmer og travel shaver",
                "model_name": "Coleman men care skægtrimmer og travel shaver",
                "model_no": "",
                "vendor": "GaveFabrikken",
                "order_count": 3
            },
            {
                "id": 5,
                "description": "FineSmile IQ tandbørste, silver",
                "model_name": "FineSmile IQ tandbørste, silver",
                "model_no": "",
                "vendor": "NDP Group",
                "order_count": 0
            },
            {
                "id": 6,
                "description": "Jesper Koch aluminiumsfad 32 x 25 cm",
                "model_name": "Jesper Koch aluminiumsfad",
                "model_no": "",
                "vendor": "GaveFabrikken",
                "order_count": 3
            },
            {
                "id": 7,
                "description": "iiFun tlf oplader",
                "model_name": "iiFun tlf oplader",
                "model_no": "",
                "vendor": "GaveFabrikken",
                "order_count": 2
            },
            {
                "id": 8,
                "description": "By Lassen kubus micro - 4 stk.",
                "model_name": "Kubus Micro - 4 stk",
                "model_no": "",
                "vendor": "By Lassen",
                "order_count": 0
            },
            {
                "id": 9,
                "description": "Håndklæder 2+2 light oak BCI",
                "model_name": "Håndklæder 2+2 light oak BCI",
                "model_no": "",
                "vendor": "GEORG JENSEN DAMASK",
                "order_count": 8
            },
            {
                "id": 10,
                "description": "Dyberg Larsen DL12 gulvlampe sort",
                "model_name": "Dyberg Larsen DL12 gulvlampe sort",
                "model_no": "",
                "vendor": "Dyberg Larsen",
                "order_count": 12
            },
            {
                "id": 11,
                "description": "Kähler Omagio Circulare vase H31",
                "model_name": "Kähler Omagio Circulare vase H31",
                "model_no": "",
                "vendor": "Rosendahl Design Group A/S",
                "order_count": 1
            },
            {
                "id": 12,
                "description": "Urban Copenhagen In-ears - vælg mellem farver",
                "model_name": "Urban Copenhagen In-ears beige",
                "model_no": "",
                "vendor": "GaveFabrikken",
                "order_count": 0
            },
            {
                "id": 13,
                "description": "Urban Copenhagen In-ears - vælg mellem farver",
                "model_name": "Urban Copenhagen In-ears grøn",
                "model_no": "",
                "vendor": "GaveFabrikken",
                "order_count": 1
            },
            {
                "id": 14,
                "description": "Urban Copenhagen In-ears - vælg mellem farver",
                "model_name": "Urban Copenhagen In-ears sort",
                "model_no": "",
                "vendor": "GaveFabrikken",
                "order_count": 1
            },
            {
                "id": 15,
                "description": "Morsø sort Fossil pande 24cm",
                "model_name": "Morsø sort Fossil pande 24cm",
                "model_no": "",
                "vendor": "F & H",
                "order_count": 3
            },
            {
                "id": 16,
                "description": "Tobias Jacobsen Solcellelamper - 3 stk",
                "model_name": "Tobias Jacobsen Solcellelamper - 3 stk",
                "model_no": "",
                "vendor": "GaveFabrikken",
                "order_count": 3
            },
            {
                "id": 17,
                "description": "Royal Copenhagen History mix æggebægre 3 stk",
                "model_name": "Royal Copenhagen History mix æggebægre 3 stk",
                "model_no": "",
                "vendor": "Fiskars",
                "order_count": 1
            },
            {
                "id": 18,
                "description": "Caterpillar skruetrækker bitsæt",
                "model_name": "Caterpillar skruetrækker bitsæt",
                "model_no": "",
                "vendor": "Dangaard",
                "order_count": 2
            },
            {
                "id": 19,
                "description": "Ordo Sonic+ tandbørste hvid m. sølv",
                "model_name": "Ordo Sonic+ tandbørste hvid m. sølv",
                "model_no": "",
                "vendor": "Dangaard",
                "order_count": 0
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
    
    logging.info(f"Created test payload with {len(request_data['presents'])} presents and {len(request_data['employees'])} employees for CVR '{request_data['cvr']}'.")
    
    return request_data


def run_smoke_test():
    """Run the smoke test by calling the API endpoint."""
    logging.info("--- Starting Prediction Pipeline Smoke Test (API) ---")
    
    try:
        # Get test data
        request_data = create_test_payload()
        total_employees = len(request_data['employees'])
        
        logging.info(f"📊 Test Data:")
        logging.info(f"   • Presents: {len(request_data['presents'])}")
        logging.info(f"   • Employees: {total_employees}")
        logging.info("")
        
        # Make API request
        logging.info("🌐 Making API request...")
        api_url = "http://127.0.0.1:9050/predict"
        
        response = requests.post(
            api_url,
            json=request_data,
            headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
            timeout=120  # Increased timeout for OpenAI classification
        )
        
        if response.status_code != 200:
            logging.error(f"❌ API request failed with status {response.status_code}")
            logging.error(f"Response: {response.text}")
            return False
            
        results = response.json()
        logging.info("✅ API request completed successfully.")
        logging.info("")

        # 4. Validate Results
        logging.info("--- Validation ---")
        
        # Assertion 1: Check if results contain a predictions field
        if not isinstance(results, dict):
            logging.error(f"❌ Expected dict response, got {type(results)}")
            return False
        
        if 'predictions' not in results:
            logging.error(f"❌ Response missing 'predictions' field. Keys: {list(results.keys())}")
            return False
            
        predictions = results['predictions']
        
        if not isinstance(predictions, list):
            logging.error(f"❌ Expected list of predictions, got {type(predictions)}")
            return False
        
        logging.info(f"✅ Response format validation passed (dict with {len(predictions)} predictions)")
        
        # Assertion 2: Check if all required fields are present
        required_fields = ['product_id', 'expected_qty']
        for i, prediction in enumerate(predictions):
            if not isinstance(prediction, dict):
                logging.error(f"❌ Prediction {i} is not a dictionary: {prediction}")
                return False
            
            for field in required_fields:
                if field not in prediction:
                    logging.error(f"❌ Missing required field '{field}' in prediction {i}: {prediction}")
                    return False
        
        logging.info("✅ All predictions have required fields")
        
        # Assertion 3: Check if expected_qty is reasonable (between 0 and employee count)
        total_predicted = 0
        for prediction in predictions:
            qty = prediction['expected_qty']
            total_predicted += qty
            
            if qty < 0:
                logging.error(f"❌ Negative prediction found: {prediction}")
                return False
            
            if qty > total_employees:
                logging.warning(f"⚠️  High prediction (>{total_employees} employees): {prediction}")
        
        logging.info(f"✅ All predictions are non-negative")
        logging.info(f"   • Total predicted quantity: {total_predicted:.1f}")
        logging.info(f"   • Average per employee: {total_predicted/total_employees:.2f}")
        
        # Display results summary
        logging.info("")
        logging.info("--- Results Summary ---")
        
        # Sort by expected quantity for easier reading
        sorted_predictions = sorted(predictions, key=lambda x: x['expected_qty'], reverse=True)
        
        logging.info("Top 10 predicted gifts:")
        for i, prediction in enumerate(sorted_predictions[:10]):
            logging.info(f"   {i+1:2d}. ID {prediction['product_id']:2s}: {prediction['expected_qty']:5.1f} units")
        
        logging.info("")
        logging.info("--- Summary Statistics ---")
        quantities = [p['expected_qty'] for p in predictions]
        logging.info(f"   • Total presents: {len(predictions)}")
        logging.info(f"   • Total employees: {total_employees}")
        logging.info(f"   • Total predicted: {sum(quantities):.1f}")
        logging.info(f"   • Average per present: {sum(quantities)/len(predictions):.1f}")
        logging.info(f"   • Max prediction: {max(quantities):.1f}")
        logging.info(f"   • Min prediction: {min(quantities):.1f}")
        
        # Check if we have reasonable distribution
        zero_predictions = len([q for q in quantities if q == 0])
        if zero_predictions == len(predictions):
            logging.warning("⚠️  All predictions are zero - model may not be working correctly")
        elif zero_predictions > len(predictions) * 0.8:
            logging.warning(f"⚠️  {zero_predictions}/{len(predictions)} predictions are zero - high sparsity")
        else:
            logging.info(f"✅ Good prediction distribution ({zero_predictions}/{len(predictions)} zero predictions)")
        
        logging.info("")
        logging.info("🎉 Smoke test completed successfully!")
        return True
        
    except requests.exceptions.Timeout:
        logging.error("❌ API request timed out")
        return False
    except requests.exceptions.ConnectionError:
        logging.error("❌ Could not connect to API - is it running on http://127.0.0.1:9050?")
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)