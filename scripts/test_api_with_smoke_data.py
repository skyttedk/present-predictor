#!/usr/bin/env python3
"""
Test API with exact same data as smoke test to compare results
"""

import httpx
import json
import asyncio

async def test_api_with_smoke_data():
    """Test API with the exact data from smoke test"""
    
    api_url = "http://localhost:9050"
    api_key = "31nl-iINTiAY4bZYlUq53h7qRQ3D_PHIS6aJGSKOYDQ"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    # Exact same payload as smoke test
    payload = {
        "cvr": "28892055",
        "presents": [
            {"id": "1", "description": "Tisvilde Pizzaovn", "model_name": "Tisvilde Pizzaovn", "model_no": "", "vendor": "GaveFabrikken"},
            {"id": "2", "description": "BodyCare Massagepude", "model_name": "BodyCare Massagepude", "model_no": "", "vendor": "Gavefabrikken"},
            {"id": "3", "description": "Jesper Koch grill sæt, sort, 5 stk", "model_name": "Jesper Koch grill sæt, sort, 5 stk", "model_no": "", "vendor": "GaveFabrikken"},
            {"id": "4", "description": "Coleman men care skægtrimmer og travel shaver", "model_name": "Coleman men care skægtrimmer og travel shaver", "model_no": "", "vendor": "GaveFabrikken"},
            {"id": "5", "description": "FineSmile IQ tandbørste, silver", "model_name": "FineSmile IQ tandbørste, silver", "model_no": "", "vendor": "NDP Group"},
            {"id": "6", "description": "Jesper Koch aluminiumsfad 32 x 25 cm", "model_name": "Jesper Koch aluminiumsfad", "model_no": "", "vendor": "GaveFabrikken"},
            {"id": "7", "description": "iiFun tlf oplader", "model_name": "iiFun tlf oplader", "model_no": "", "vendor": "GaveFabrikken"},
            {"id": "8", "description": "By Lassen kubus micro - 4 stk.", "model_name": "Kubus Micro - 4 stk", "model_no": "", "vendor": "By Lassen"},
            {"id": "9", "description": "Håndklæder 2+2 light oak BCI", "model_name": "Håndklæder 2+2 light oak BCI", "model_no": "", "vendor": "GEORG JENSEN DAMASK"},
            {"id": "10", "description": "Dyberg Larsen DL12 gulvlampe sort", "model_name": "Dyberg Larsen DL12 gulvlampe sort", "model_no": "", "vendor": "Dyberg Larsen"},
            {"id": "11", "description": "Kähler Omagio Circulare vase H31", "model_name": "Kähler Omagio Circulare vase H31", "model_no": "", "vendor": "Rosendahl Design Group A/S"},
            {"id": "12", "description": "Urban Copenhagen In-ears - vælg mellem farver", "model_name": "Urban Copenhagen In-ears beige", "model_no": "", "vendor": "GaveFabrikken"},
            {"id": "13", "description": "Urban Copenhagen In-ears - vælg mellem farver", "model_name": "Urban Copenhagen In-ears grøn", "model_no": "", "vendor": "GaveFabrikken"},
            {"id": "14", "description": "Urban Copenhagen In-ears - vælg mellem farver", "model_name": "Urban Copenhagen In-ears sort", "model_no": "", "vendor": "GaveFabrikken"},
            {"id": "15", "description": "Morsø sort Fossil pande 24cm", "model_name": "Morsø sort Fossil pande 24cm", "model_no": "", "vendor": "F & H"},
            {"id": "16", "description": "Tobias Jacobsen Solcellelamper - 3 stk", "model_name": "Tobias Jacobsen Solcellelamper - 3 stk", "model_no": "", "vendor": "GaveFabrikken"},
            {"id": "17", "description": "Royal Copenhagen History mix æggebægre 3 stk", "model_name": "Royal Copenhagen History mix æggebægre 3 stk", "model_no": "", "vendor": "Fiskars"},
            {"id": "18", "description": "Caterpillar skruetrækker bitsæt", "model_name": "Caterpillar skruetrækker bitsæt", "model_no": "", "vendor": "Dangaard"},
            {"id": "19", "description": "Ordo Sonic+ tandbørste hvid m. sølv", "model_name": "Ordo Sonic+ tandbørste hvid m. sølv", "model_no": "", "vendor": "Dangaard"}
        ],
        "employees": [
            {"name": "Laimonas Lukosevicius"}, {"name": "Petra De Laet"}, {"name": "Regimantas Usas"}, {"name": "Marius Melvold"},
            {"name": "Poul Torndahl"}, {"name": "Hallvard Banken"}, {"name": "Daniel Løsnesløkken"}, {"name": "Magnus Ebenstrand"},
            {"name": "Tina Blaabjerg"}, {"name": "Sanne Olsen"}, {"name": "Thomas Hindhede"}, {"name": "Magus Gupefjäll"},
            {"name": "Pål Aulesjord"}, {"name": "Martin Berg"}, {"name": "Henning Poulsen"}, {"name": "Alexander Normann"},
            {"name": "Henrik Stærsholm"}, {"name": "Ann-Sofie Åhman"}, {"name": "Juha Södervik"}, {"name": "Danny Borød"},
            {"name": "Kai Huhtala"}, {"name": "Søren Pedersen"}, {"name": "Tina Kindvall"}, {"name": "Leila Eskandari"},
            {"name": "Johnny Winther"}, {"name": "Mats Dovander"}, {"name": "Patrik Larsson"}, {"name": "Kia Dahl Andersen"},
            {"name": "Cecilie Dietiker Vonk"}, {"name": "Trine Syversen"}, {"name": "Anders Ahlstam"}, {"name": "Dorthe Niewald"},
            {"name": "Malene Pedersen"}, {"name": "Natasha Tyyskä"}, {"name": "Åsa Hörnell"}, {"name": "Reine Ringblom"},
            {"name": "Torben Villadsen"}, {"name": "Otto Svensson"}, {"name": "Vegard Hagen"}, {"name": "Susanne Sundsberg"},
            {"name": "Ronny Israelsson"}, {"name": "Kaj Nyman"}, {"name": "Bengt Gerhardsen"}, {"name": "Marianne List Nissen"},
            {"name": "Navneet Kaur"}, {"name": "Jens Rask"}, {"name": "Alexander Nylander"}, {"name": "Jan Kristiansen"},
            {"name": "Marianne Wallanger"}, {"name": "Tommy Sundsjöö"}, {"name": "Lars Jagervall"}, {"name": "Claus Andersen"},
            {"name": "Christer Jönsson"}, {"name": "Thomas Olesen"}, {"name": "Fredrik Alvarsson"}, {"name": "Gvido Musperts"},
            {"name": "Søren Wander Jensen"}
        ]
    }
    
    print("🔍 Testing API with exact smoke test data...")
    print(f"📝 Payload: {len(payload['presents'])} presents, {len(payload['employees'])} employees")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(f"{api_url}/predict", headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ API Response received!")
                print(f"📊 Branch: {result['branch_no']}")
                print(f"👥 Total employees: {result['total_employees']}")
                print(f"⏱️ Processing time: {result['processing_time_ms']:.2f}ms")
                
                # Extract predictions
                predictions = result['predictions']
                quantities = [p['expected_qty'] for p in predictions]
                confidences = [p['confidence_score'] for p in predictions]
                
                print(f"\n🎯 Prediction Analysis:")
                print(f"   Quantities range: {min(quantities):.2f} to {max(quantities):.2f}")
                print(f"   Total predicted: {sum(quantities):.2f}")
                print(f"   Confidence range: {min(confidences):.2f} to {max(confidences):.2f}")
                print(f"   Unique quantities: {len(set([round(q, 2) for q in quantities]))}")
                
                # Show top predictions
                print(f"\n📋 Top 10 Predictions:")
                sorted_preds = sorted(predictions, key=lambda x: x['expected_qty'], reverse=True)
                for i, pred in enumerate(sorted_preds[:10]):
                    print(f"   {i+1:2d}. Product {pred['product_id']:2s}: {pred['expected_qty']:8.2f} (conf: {pred['confidence_score']:.2f})")
                
                # Compare with smoke test results
                print(f"\n🔍 Comparison with Smoke Test:")
                print("   Smoke test range: 4.90 to 8.16")
                print("   Smoke test confidence: 0.71 to 0.73")
                if max(quantities) > 4.0 and max(confidences) > 0.65:
                    print("   ✅ Results look similar to smoke test!")
                else:
                    print("   ❌ Results still different from smoke test")
                
                return result
                
            else:
                print(f"❌ API Error: Status {response.status_code}")
                print(f"📝 Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

if __name__ == "__main__":
    result = asyncio.run(test_api_with_smoke_data())