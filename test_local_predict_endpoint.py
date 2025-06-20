#!/usr/bin/env python3
"""
Test script for the local predict endpoint.
Tests the /predict endpoint with the provided payload on localhost:8000.
"""

import json
import requests
import time
from datetime import datetime

# Test payload from user
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
        },
        {
            "id": "6",
            "description": "Jesper Koch aluminiumsfad 32 x 25 cm",
            "model_name": "Jesper Koch aluminiumsfad",
            "model_no": "",
            "vendor": "GaveFabrikken"
        },
        {
            "id": "7",
            "description": "iiFun tlf oplader",
            "model_name": "iiFun tlf oplader",
            "model_no": "",
            "vendor": "GaveFabrikken"
        },
        {
            "id": "8",
            "description": "By Lassen kubus micro - 4 stk.",
            "model_name": "Kubus Micro - 4 stk",
            "model_no": "",
            "vendor": "By Lassen"
        },
        {
            "id": "9",
            "description": "Håndklæder 2+2 light oak BCI",
            "model_name": "Håndklæder 2+2 light oak BCI",
            "model_no": "",
            "vendor": "GEORG JENSEN DAMASK"
        },
        {
            "id": "10",
            "description": "Dyberg Larsen DL12 gulvlampe sort",
            "model_name": "Dyberg Larsen DL12 gulvlampe sort",
            "model_no": "",
            "vendor": "Dyberg Larsen"
        },
        {
            "id": "11",
            "description": "Kähler Omagio Circulare vase H31",
            "model_name": "Kähler Omagio Circulare vase H31",
            "model_no": "",
            "vendor": "Rosendahl Design Group A/S"
        },
        {
            "id": "12",
            "description": "Urban Copenhagen In-ears - vælg mellem farver",
            "model_name": "Urban Copenhagen In-ears beige",
            "model_no": "",
            "vendor": "GaveFabrikken"
        },
        {
            "id": "13",
            "description": "Urban Copenhagen In-ears - vælg mellem farver",
            "model_name": "Urban Copenhagen In-ears grøn",
            "model_no": "",
            "vendor": "GaveFabrikken"
        },
        {
            "id": "14",
            "description": "Urban Copenhagen In-ears - vælg mellem farver",
            "model_name": "Urban Copenhagen In-ears sort",
            "model_no": "",
            "vendor": "GaveFabrikken"
        },
        {
            "id": "15",
            "description": "Morsø sort Fossil pande 24cm",
            "model_name": "Morsø sort Fossil pande 24cm",
            "model_no": "",
            "vendor": "F & H"
        },
        {
            "id": "16",
            "description": "Tobias Jacobsen Solcellelamper - 3 stk",
            "model_name": "Tobias Jacobsen Solcellelamper - 3 stk",
            "model_no": "",
            "vendor": "GaveFabrikken"
        },
        {
            "id": "17",
            "description": "Royal Copenhagen History mix æggebægre 3 stk",
            "model_name": "Royal Copenhagen History mix æggebægre 3 stk",
            "model_no": "",
            "vendor": "Fiskars"
        },
        {
            "id": "18",
            "description": "Caterpillar skruetrækker bitsæt",
            "model_name": "Caterpillar skruetrækker bitsæt",
            "model_no": "",
            "vendor": "Dangaard"
        },
        {
            "id": "19",
            "description": "Ordo Sonic+ tandbørste hvid m. sølv",
            "model_name": "Ordo Sonic+ tandbørste hvid m. sølv",
            "model_no": "",
            "vendor": "Dangaard"
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

def test_predict_endpoint():
    """Test the /predict endpoint with the provided payload."""
    
    # Local development server URL
    base_url = "http://localhost:9050"
    endpoint = f"{base_url}/predict"
    
    print("=" * 80)
    print(f"TESTING PREDICT ENDPOINT")
    print(f"URL: {endpoint}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Print request summary
    print(f"\nREQUEST SUMMARY:")
    print(f"CVR: {test_payload['cvr']}")
    print(f"Number of presents: {len(test_payload['presents'])}")
    print(f"Number of employees: {len(test_payload['employees'])}")
    
    print(f"\nSample presents:")
    for i, present in enumerate(test_payload['presents'][:3]):
        print(f"  {i+1}. ID: {present['id']} - {present['description'][:50]}...")
    if len(test_payload['presents']) > 3:
        print(f"  ... and {len(test_payload['presents']) - 3} more")
    
    print(f"\nSample employees:")
    for i, employee in enumerate(test_payload['employees'][:5]):
        print(f"  {i+1}. {employee['name']}")
    if len(test_payload['employees']) > 5:
        print(f"  ... and {len(test_payload['employees']) - 5} more")
    
    # Prepare headers with API key
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-API-Key": "31nl-iINTiAY4bZYlUq53h7qRQ3D_PHIS6aJGSKOYDQ"
    }
    
    print(f"\n" + "-" * 80)
    print("SENDING REQUEST...")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Make the request
        response = requests.post(
            endpoint,
            json=test_payload,
            headers=headers,
            timeout=120  # 2 minute timeout for ML processing
        )
        
        # Record end time
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"Response received in {response_time:.2f} seconds")
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'Not specified')}")
        
        print(f"\n" + "-" * 80)
        print("RESPONSE ANALYSIS:")
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                
                # Analyze the response
                if isinstance(response_data, list):
                    print(f"✅ SUCCESS - Received predictions for {len(response_data)} items")
                    
                    print(f"\nPREDICTION RESULTS:")
                    print(f"{'ID':<6} {'Expected Qty':<12} {'Description':<50}")
                    print("-" * 70)
                    
                    total_expected = 0
                    for prediction in response_data:
                        if isinstance(prediction, dict):
                            product_id = prediction.get('product_id', 'N/A')
                            expected_qty = prediction.get('expected_qty', 0)
                            
                            # Find the description from the original request
                            description = "Unknown"
                            for present in test_payload['presents']:
                                if str(present['id']) == str(product_id):
                                    description = present['description'][:47] + "..." if len(present['description']) > 47 else present['description']
                                    break
                            
                            print(f"{product_id:<6} {expected_qty:<12.1f} {description:<50}")
                            total_expected += expected_qty
                    
                    print("-" * 70)
                    print(f"{'TOTAL':<6} {total_expected:<12.1f}")
                    
                    # Statistics
                    quantities = [p.get('expected_qty', 0) for p in response_data if isinstance(p, dict)]
                    if quantities:
                        print(f"\nSTATISTICS:")
                        print(f"  Min quantity: {min(quantities):.1f}")
                        print(f"  Max quantity: {max(quantities):.1f}")
                        print(f"  Average quantity: {sum(quantities)/len(quantities):.1f}")
                        print(f"  Total expected: {sum(quantities):.1f}")
                        print(f"  Expected per employee: {sum(quantities)/len(test_payload['employees']):.2f}")
                        
                        # Check for uniform predictions (potential issue)
                        unique_values = set(quantities)
                        if len(unique_values) == 1:
                            print(f"  ⚠️  WARNING: All predictions are identical ({quantities[0]:.1f})")
                        elif len(unique_values) < 3:
                            print(f"  ⚠️  WARNING: Only {len(unique_values)} unique prediction values")
                        else:
                            print(f"  ✅ Good: {len(unique_values)} unique prediction values")
                
                else:
                    print(f"❌ UNEXPECTED RESPONSE FORMAT")
                    print(f"Expected list, got: {type(response_data)}")
                    print(f"Response: {json.dumps(response_data, indent=2)}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ FAILED TO PARSE JSON RESPONSE")
                print(f"JSONDecodeError: {e}")
                print(f"Raw response content: {response.text[:500]}...")
                
        else:
            print(f"❌ REQUEST FAILED")
            print(f"Status: {response.status_code}")
            
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw response: {response.text}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ CONNECTION ERROR")
        print(f"Error: {e}")
        print(f"\nPossible issues:")
        print(f"  - Server not running on localhost:8000")
        print(f"  - Server startup still in progress")
        print(f"  - Network connection issue")
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR")
        print(f"Error: {e}")
        print(f"Type: {type(e)}")
    
    print(f"\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

def test_health_check():
    """Test if the server is running by checking a health endpoint."""
    health_urls = [
        "http://localhost:9050/",
        "http://localhost:9050/health",
        "http://localhost:9050/docs"
    ]
    
    print("\nCHECKING SERVER AVAILABILITY:")
    
    for url in health_urls:
        try:
            response = requests.get(url, timeout=5)
            print(f"  OK {url} - Status: {response.status_code}")
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            print(f"  FAIL {url} - Not responding")
    
    return False

if __name__ == "__main__":
    print("Testing Local Predict Endpoint")
    print("==============================")
    
    # First check if server is available
    if test_health_check():
        print("\nSUCCESS: Server appears to be running")
        test_predict_endpoint()
    else:
        print("\nERROR: Server does not appear to be running on localhost:9050")
        print("\nPlease ensure:")
        print("  1. Development server is started")
        print("  2. Server is running on port 9050")
        print("  3. No firewall blocking the connection")