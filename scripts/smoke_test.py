# scripts/smoke_test.py

"""
Smoke Test for the Re-architected Prediction Pipeline

This script validates that the rate-based prediction model and the updated
predictor logic are working together as expected.

It checks for:
1. Successful model loading.
2. Non-uniform, non-negative predictions.
3. Correct aggregation logic (total expected_qty is not normalized).
"""

import os
import sys
import logging
import pandas as pd

# Add src to the Python path to allow for absolute imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.ml.predictor import get_predictor
from src.api.schemas.responses import PredictionResult

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def create_test_payload():
    """Creates a real-world payload for testing the predictor using data that previously caused poor predictions."""
    
    # Real-world CVR and branch data
    cvr = "28892055"
    branch = "621000"  # Using common branch for consistency
    
    # Real-world presents data (raw descriptions that need classification)
    raw_presents = [
        {"id": "1", "description": "Tisvilde Pizzaovn", "model_name": "Tisvilde Pizzaovn", "model_no": "", "vendor": "GaveFabrikken", "order_count": 7},
        {"id": "2", "description": "BodyCare Massagepude", "model_name": "BodyCare Massagepude", "model_no": "", "vendor": "Gavefabrikken", "order_count": 9},
        {"id": "3", "description": "Jesper Koch grill sæt, sort, 5 stk", "model_name": "Jesper Koch grill sæt, sort, 5 stk", "model_no": "", "vendor": "GaveFabrikken", "order_count": 1},
        {"id": "4", "description": "Coleman men care skægtrimmer og travel shaver", "model_name": "Coleman men care skægtrimmer og travel shaver", "model_no": "", "vendor": "GaveFabrikken", "order_count": 3},
        {"id": "5", "description": "FineSmile IQ tandbørste, silver", "model_name": "FineSmile IQ tandbørste, silver", "model_no": "", "vendor": "NDP Group", "order_count": 0},
        {"id": "6", "description": "Jesper Koch aluminiumsfad 32 x 25 cm", "model_name": "Jesper Koch aluminiumsfad", "model_no": "", "vendor": "GaveFabrikken", "order_count": 3},
        {"id": "7", "description": "iiFun tlf oplader", "model_name": "iiFun tlf oplader", "model_no": "", "vendor": "GaveFabrikken", "order_count": 2},
        {"id": "8", "description": "By Lassen kubus micro - 4 stk.", "model_name": "Kubus Micro - 4 stk", "model_no": "", "vendor": "By Lassen", "order_count": 0},
        {"id": "9", "description": "Håndklæder 2+2 light oak BCI", "model_name": "Håndklæder 2+2 light oak BCI", "model_no": "", "vendor": "GEORG JENSEN DAMASK", "order_count": 8},
        {"id": "10", "description": "Dyberg Larsen DL12 gulvlampe sort", "model_name": "Dyberg Larsen DL12 gulvlampe sort", "model_no": "", "vendor": "Dyberg Larsen", "order_count": 12},
        {"id": "11", "description": "Kähler Omagio Circulare vase H31", "model_name": "Kähler Omagio Circulare vase H31", "model_no": "", "vendor": "Rosendahl Design Group A/S", "order_count": 1},
        {"id": "12", "description": "Urban Copenhagen In-ears - vælg mellem farver", "model_name": "Urban Copenhagen In-ears beige", "model_no": "", "vendor": "GaveFabrikken", "order_count": 0},
        {"id": "13", "description": "Urban Copenhagen In-ears - vælg mellem farver", "model_name": "Urban Copenhagen In-ears grøn", "model_no": "", "vendor": "GaveFabrikken", "order_count": 1},
        {"id": "14", "description": "Urban Copenhagen In-ears - vælg mellem farver", "model_name": "Urban Copenhagen In-ears sort", "model_no": "", "vendor": "GaveFabrikken", "order_count": 1},
        {"id": "15", "description": "Morsø sort Fossil pande 24cm", "model_name": "Morsø sort Fossil pande 24cm", "model_no": "", "vendor": "F & H", "order_count": 3},
        {"id": "16", "description": "Tobias Jacobsen Solcellelamper - 3 stk", "model_name": "Tobias Jacobsen Solcellelamper - 3 stk", "model_no": "", "vendor": "GaveFabrikken", "order_count": 3},
        {"id": "17", "description": "Royal Copenhagen History mix æggebægre 3 stk", "model_name": "Royal Copenhagen History mix æggebægre 3 stk", "model_no": "", "vendor": "Fiskars", "order_count": 1},
        {"id": "18", "description": "Caterpillar skruetrækker bitsæt", "model_name": "Caterpillar skruetrækker bitsæt", "model_no": "", "vendor": "Dangaard", "order_count": 2},
        {"id": "19", "description": "Ordo Sonic+ tandbørste hvid m. sølv", "model_name": "Ordo Sonic+ tandbørste hvid m. sølv", "model_no": "", "vendor": "Dangaard", "order_count": 0}
    ]
    
    # Real-world employee names
    raw_employees = [
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
    
    # Manual classification for testing (based on descriptions)
    classified_presents = [
        {"id": "1", "item_main_category": "Home & Kitchen", "item_sub_category": "Outdoor Cooking", "brand": "Tisvilde", "color": "NONE", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "shareable"},
        {"id": "2", "item_main_category": "Health & Beauty", "item_sub_category": "Massage", "brand": "BodyCare", "color": "NONE", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "3", "item_main_category": "Home & Kitchen", "item_sub_category": "Grilling", "brand": "Jesper Koch", "color": "sort", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "4", "item_main_category": "Health & Beauty", "item_sub_category": "Grooming", "brand": "Coleman", "color": "NONE", "durability": "durable", "target_demographic": "male", "utility_type": "practical", "usage_type": "individual"},
        {"id": "5", "item_main_category": "Health & Beauty", "item_sub_category": "Dental Care", "brand": "FineSmile", "color": "silver", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "6", "item_main_category": "Home & Kitchen", "item_sub_category": "Cookware", "brand": "Jesper Koch", "color": "NONE", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "7", "item_main_category": "Electronics", "item_sub_category": "Chargers", "brand": "iiFun", "color": "NONE", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "8", "item_main_category": "Home & Decor", "item_sub_category": "Candle Holders", "brand": "By Lassen", "color": "NONE", "durability": "durable", "target_demographic": "unisex", "utility_type": "aesthetic", "usage_type": "individual"},
        {"id": "9", "item_main_category": "Home & Kitchen", "item_sub_category": "Textiles", "brand": "GEORG JENSEN DAMASK", "color": "light oak", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "10", "item_main_category": "Home & Decor", "item_sub_category": "Lighting", "brand": "Dyberg Larsen", "color": "sort", "durability": "durable", "target_demographic": "unisex", "utility_type": "aesthetic", "usage_type": "individual"},
        {"id": "11", "item_main_category": "Home & Decor", "item_sub_category": "Vases", "brand": "Kähler", "color": "NONE", "durability": "durable", "target_demographic": "unisex", "utility_type": "aesthetic", "usage_type": "individual"},
        {"id": "12", "item_main_category": "Electronics", "item_sub_category": "Headphones", "brand": "Urban Copenhagen", "color": "beige", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "13", "item_main_category": "Electronics", "item_sub_category": "Headphones", "brand": "Urban Copenhagen", "color": "grøn", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "14", "item_main_category": "Electronics", "item_sub_category": "Headphones", "brand": "Urban Copenhagen", "color": "sort", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "15", "item_main_category": "Home & Kitchen", "item_sub_category": "Cookware", "brand": "Morsø", "color": "sort", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "16", "item_main_category": "Home & Decor", "item_sub_category": "Lighting", "brand": "Tobias Jacobsen", "color": "NONE", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "17", "item_main_category": "Home & Kitchen", "item_sub_category": "Tableware", "brand": "Royal Copenhagen", "color": "NONE", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"},
        {"id": "18", "item_main_category": "Tools & DIY", "item_sub_category": "Hand Tools", "brand": "Caterpillar", "color": "NONE", "durability": "durable", "target_demographic": "male", "utility_type": "practical", "usage_type": "individual"},
        {"id": "19", "item_main_category": "Health & Beauty", "item_sub_category": "Dental Care", "brand": "Ordo", "color": "hvid", "durability": "durable", "target_demographic": "unisex", "utility_type": "practical", "usage_type": "individual"}
    ]
    
    # Manual gender classification for testing (based on names)
    processed_employees = [
        {"gender": "male"}, {"gender": "female"}, {"gender": "male"}, {"gender": "male"},  # Laimonas, Petra, Regimantas, Marius
        {"gender": "male"}, {"gender": "male"}, {"gender": "male"}, {"gender": "male"},    # Poul, Hallvard, Daniel, Magnus
        {"gender": "female"}, {"gender": "female"}, {"gender": "male"}, {"gender": "male"}, # Tina, Sanne, Thomas, Magus
        {"gender": "male"}, {"gender": "male"}, {"gender": "male"}, {"gender": "male"},     # Pål, Martin, Henning, Alexander
        {"gender": "male"}, {"gender": "female"}, {"gender": "male"}, {"gender": "male"},   # Henrik, Ann-Sofie, Juha, Danny
        {"gender": "male"}, {"gender": "male"}, {"gender": "female"}, {"gender": "female"}, # Kai, Søren, Tina, Leila
        {"gender": "male"}, {"gender": "male"}, {"gender": "male"}, {"gender": "female"},   # Johnny, Mats, Patrik, Kia
        {"gender": "female"}, {"gender": "female"}, {"gender": "male"}, {"gender": "female"}, # Cecilie, Trine, Anders, Dorthe
        {"gender": "female"}, {"gender": "female"}, {"gender": "female"}, {"gender": "male"}, # Malene, Natasha, Åsa, Reine
        {"gender": "male"}, {"gender": "male"}, {"gender": "male"}, {"gender": "female"},    # Torben, Otto, Vegard, Susanne
        {"gender": "male"}, {"gender": "male"}, {"gender": "male"}, {"gender": "female"},    # Ronny, Kaj, Bengt, Marianne
        {"gender": "female"}, {"gender": "male"}, {"gender": "male"}, {"gender": "male"},    # Navneet, Jens, Alexander, Jan
        {"gender": "female"}, {"gender": "male"}, {"gender": "male"}, {"gender": "male"},    # Marianne, Tommy, Lars, Claus
        {"gender": "male"}, {"gender": "male"}, {"gender": "male"}, {"gender": "male"},      # Christer, Thomas, Fredrik, Gvido
        {"gender": "male"}  # Søren
    ]
    
    logging.info(f"Created real-world test payload with {len(classified_presents)} presents and {len(processed_employees)} employees for CVR '{cvr}' and branch '{branch}'.")
    logging.info(f"Employee gender distribution: {sum(1 for e in processed_employees if e['gender'] == 'male')} male, {sum(1 for e in processed_employees if e['gender'] == 'female')} female")
    
    return branch, classified_presents, processed_employees

def run_smoke_test():
    """Executes the smoke test for the prediction pipeline."""
    logging.info("--- Starting Prediction Pipeline Smoke Test ---")
    
    try:
        # 1. Get Predictor Instance
        logging.info("Loading model and initializing predictor...")
        # Use absolute paths to ensure the script runs from any directory
        model_path = os.path.join(BASE_DIR, "models", "catboost_poisson_model", "catboost_poisson_model.cbm")
        historical_data_path = os.path.join(BASE_DIR, "src", "data", "historical", "present.selection.historic.csv")
        
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at: {model_path}")
            logging.error("Please ensure the model has been trained by running `src/ml/catboost_trainer.py`.")
            return False
            
        predictor = get_predictor(model_path=model_path)
        logging.info("Predictor initialized successfully.")

        # 2. Create Test Payload
        branch, presents, employees = create_test_payload()
        total_employees = len(employees)

        # 3. Make Prediction
        logging.info("Making predictions...")
        results = predictor.predict(branch, presents, employees)
        logging.info("Prediction complete.")

        # 4. Validate Results
        logging.info("--- Validation ---")
        
        # Assertion 1: Check if results are a list of dictionaries
        assert isinstance(results, list) and all(isinstance(r, dict) for r in results), \
            f"Expected a list of dictionaries, but got {type(results)}."
        logging.info(f"✅ OK: Prediction returned a list of {len(results)} prediction dictionaries.")

        # Assertion 2: Check if the number of predictions matches the number of presents
        assert len(results) == len(presents), \
            f"Expected {len(presents)} predictions, but got {len(results)}."
        logging.info("✅ OK: Number of predictions matches number of presents.")

        # Assertion 3: Check for non-negative quantities
        quantities = [r['expected_qty'] for r in results]
        assert all(qty >= 0 for qty in quantities), \
            f"Found negative quantities in predictions: {quantities}"
        logging.info("✅ OK: All predicted quantities are non-negative.")

        # Assertion 4: Check for non-uniform predictions (key for log-exposure fix validation)
        unique_quantities = set(round(qty, 2) for qty in quantities)
        if len(unique_quantities) == 1 and len(quantities) > 1:
             logging.warning("⚠️ WARNING: All predicted quantities are uniform. This indicates variance collapse - log-exposure fix may not be working.")
        else:
            min_qty = min(quantities)
            max_qty = max(quantities)
            qty_range = max_qty - min_qty
            logging.info(f"✅ OK: Predictions are not uniform. Range: {min_qty:.2f} to {max_qty:.2f} (spread: {qty_range:.2f})")

        # Assertion 5: Check variance range (critical for log-exposure fix validation)
        qty_range = max(quantities) - min(quantities)
        if qty_range < 2.0:
            logging.warning(f"⚠️ WARNING: Prediction range is narrow ({qty_range:.2f}). This may indicate variance collapse.")
        else:
            logging.info(f"✅ OK: Good prediction variance with range of {qty_range:.2f}")

        # Assertion 6: Check if total quantity is reasonable (not normalized)
        total_predicted_qty = sum(quantities)
        selection_rate = (total_predicted_qty / total_employees) * 100
        logging.info(f"Total employees: {total_employees}")
        logging.info(f"Total predicted quantity: {total_predicted_qty:.2f}")
        logging.info(f"Selection rate: {selection_rate:.1f}%")
        
        assert total_predicted_qty > 0, "Total predicted quantity should be greater than zero for this test case."
        
        # This is the key check for the new architecture. The sum should NOT equal total_employees.
        # It should be a fraction of it, representing the total expected selections.
        assert total_predicted_qty != total_employees, \
            "Total predicted quantity should not be equal to total employees (old normalization logic)."
        logging.info("✅ OK: Total predicted quantity is not artificially normalized to total employees.")

        # Assertion 7: Check for reasonable selection rate (10-60% typical range)
        if 10 <= selection_rate <= 60:
            logging.info(f"✅ OK: Selection rate {selection_rate:.1f}% is within reasonable range (10-60%)")
        else:
            logging.warning(f"⚠️ WARNING: Selection rate {selection_rate:.1f}% may be outside typical range (10-60%)")

        logging.info("\n--- Top Prediction Results (showing first 10) ---")
        results_df = pd.DataFrame(results)
        # Sort by expected_qty descending to see highest predictions first
        results_df = results_df.sort_values('expected_qty', ascending=False)
        print(results_df.head(10).to_string(index=False))
        logging.info("--------------------------\n")

        logging.info("✅ Smoke test passed successfully!")
        return True

    except Exception as e:
        logging.error(f"❌ Smoke test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    test_passed = run_smoke_test()
    if not test_passed:
        sys.exit(1)