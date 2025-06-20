import logging
import sys
import os
import asyncio # For running async classification
from typing import List, Dict

# Add src to path to allow direct imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.predictor import GiftDemandPredictor, get_predictor
from src.api.schemas.responses import PredictionResult
# from src.data.gender_classifier import EnhancedGenderClassifier, classify_employee_gender # No longer needed directly here
from src.data.classifier import DataClassifier # For gift and employee classification
from src.api.schemas.requests import PredictionRequest, GiftItem, Employee as APIEmployee # Rename to avoid conflict
from src.config.settings import get_settings # To ensure settings (like OpenAI key) are loaded

# --- Configuration ---
MODEL_PATH = "models/catboost_poisson_model/catboost_poisson_model.cbm"
HISTORICAL_DATA_PATH = "src/data/historical/present.selection.historic.csv"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Test Payload (Raw Descriptions) ---
# Presents will now only have 'id' and 'description'. Classification will happen in the script.
test_payload = {
    "cvr": "28892055",
    "presents": [
        {
            "id": "1",
            "description": "Tisvilde Pizzaovn"
        },
        {
            "id": "2",
            "description": "BodyCare Massagepude"
        },
        {
            "id": "3",
            "description": "Jesper Koch grill sæt, sort, 5 stk"
        },
        # Add more presents with raw descriptions if needed
        # Example:
        # {
        #     "id": "4",
        #     "description": "Royal Copenhagen Blue Fluted Mega Bowl 21cm"
        # },
        # {
        #     "id": "5",
        #     "description": "Georg Jensen Bernadotte thermos 1L"
        # }
    ],
    "employees": [ # Employee names remain the same, gender will be classified
        {"name": "Laimonas Lukosevicius"}, {"name": "Petra De Laet"}, {"name": "Regimantas Usas"},
        {"name": "Marius Melvold"}, {"name": "Poul Torndahl"}, {"name": "Hallvard Banken"},
        {"name": "Daniel Løsnesløkken"}, {"name": "Magnus Ebenstrand"}, {"name": "Tina Blaabjerg"},
        {"name": "Sanne Olsen"}, {"name": "Thomas Hindhede"}, {"name": "Magus Gupefjäll"},
        {"name": "Pål Aulesjord"}, {"name": "Martin Berg"}, {"name": "Henning Poulsen"},
        {"name": "Alexander Normann"}, {"name": "Henrik Stærsholm"}, {"name": "Ann-Sofie Åhman"},
        {"name": "Juha Södervik"}, {"name": "Danny Borød"}, {"name": "Kai Huhtala"},
        {"name": "Søren Pedersen"}, {"name": "Tina Kindvall"}, {"name": "Leila Eskandari"},
        {"name": "Johnny Winther"}, {"name": "Mats Dovander"}, {"name": "Patrik Larsson"},
        {"name": "Kia Dahl Andersen"}, {"name": "Cecilie Dietiker Vonk"}, {"name": "Trine Syversen"},
        {"name": "Anders Ahlstam"}, {"name": "Dorthe Niewald"}, {"name": "Malene Pedersen"},
        {"name": "Natasha Tyyskä"}, {"name": "Åsa Hörnell"}, {"name": "Reine Ringblom"},
        {"name": "Torben Villadsen"}, {"name": "Otto Svensson"}, {"name": "Vegard Hagen"},
        {"name": "Susanne Sundsberg"}, {"name": "Ronny Israelsson"}, {"name": "Kaj Nyman"},
        {"name": "Bengt Gerhardsen"}, {"name": "Marianne List Nissen"}, {"name": "Navneet Kaur"},
        {"name": "Jens Rask"}, {"name": "Alexander Nylander"}, {"name": "Jan Kristiansen"},
        {"name": "Marianne Wallanger"}, {"name": "Tommy Sundsjöö"}, {"name": "Lars Jagervall"},
        {"name": "Claus Andersen"}, {"name": "Christer Jönsson"}, {"name": "Thomas Olesen"},
        {"name": "Fredrik Alvarsson"}, {"name": "Gvido Musperts"}, {"name": "Søren Wander Jensen"}
    ]
}

# Removed process_employees_for_predictor as DataClassifier will handle it.

async def main(): # Main function is now async
    logger.info("Starting local predictor test script with internal classification...")
    
    # Ensure settings are loaded (especially for OpenAI API Key)
    settings = get_settings()
    if not settings.openai.api_key:
        logger.warning("OPENAI_API_KEY not found in environment or .env file. Gift classification will likely use fallbacks.")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    actual_model_path = os.path.join(project_root, MODEL_PATH)
    actual_historical_data_path = os.path.join(project_root, HISTORICAL_DATA_PATH)

    if not os.path.exists(actual_model_path):
        logger.error(f"Model file not found at: {actual_model_path}")
        return
    if not os.path.exists(actual_historical_data_path):
        logger.error(f"Historical data file not found at: {actual_historical_data_path}")

    try:
        # --- Step 1: Classify Data ---
        logger.info("Initializing DataClassifier...")
        # DataClassifier will create its own OpenAI client if one isn't passed
        classifier = DataClassifier()

        # Prepare PredictionRequest for DataClassifier
        api_gifts = [GiftItem(product_id=p["id"], description=p["description"]) for p in test_payload["presents"]]
        api_employees = [APIEmployee(name=e["name"]) for e in test_payload["employees"]]
        
        prediction_request = PredictionRequest(
            branch_no=test_payload["cvr"], # Using cvr as branch_no for consistency
            gifts=api_gifts,
            employees=api_employees
        )

        logger.info("Classifying request data (gifts and employees)...")
        classified_gifts_result, processed_employees_result, classification_stats = await classifier.classify_request(prediction_request)
        
        logger.info(f"Classification stats: {classification_stats}")
        logger.debug(f"Classified gifts: {classified_gifts_result}")
        logger.debug(f"Processed employees: {processed_employees_result}")

        # Transform classified data into the format expected by GiftDemandPredictor
        presents_for_predictor = []
        for cg in classified_gifts_result:
            present_dict = {"id": cg.product_id} # Predictor expects 'id', not 'product_id' at the top level
            if cg.attributes:
                # Spread the attributes from the GiftAttributes model into the dictionary
                # Predictor's _create_feature_vector uses present.get('item_main_category', 'NONE'), etc.
                present_dict.update(cg.attributes.model_dump())
            else: # Fallback if attributes are None (should not happen with current DataClassifier logic)
                logger.warning(f"Gift {cg.product_id} has no attributes after classification. Using NONE for all.")
                # Add default NONE attributes if necessary, matching predictor expectations
                default_attrs = {
                    'item_main_category': 'NONE', 'item_sub_category': 'NONE', 'brand': 'NONE',
                    'color': 'NONE', 'durability': 'NONE', 'target_demographic': 'NONE',
                    'utility_type': 'NONE', 'usage_type': 'NONE'
                }
                present_dict.update(default_attrs)
            presents_for_predictor.append(present_dict)

        employees_for_predictor = [{"name": pe.name, "gender": pe.gender.value} for pe in processed_employees_result]

        # --- Step 2: Make Predictions ---
        logger.info(f"Initializing predictor with model: {actual_model_path}")
        predictor = get_predictor(
            model_path=actual_model_path,
            historical_data_path=actual_historical_data_path
        )
        
        cvr = test_payload["cvr"]
        logger.info(f"Calling predictor.predict for CVR: {cvr}")
        logger.debug(f"Presents being sent to predictor: {presents_for_predictor}")
        logger.debug(f"Employees being sent to predictor: {employees_for_predictor}")

        results: List[PredictionResult] = predictor.predict(
            branch=cvr,
            presents=presents_for_predictor,
            employees=employees_for_predictor
        )

        logger.info("--- Prediction Results ---")
        if results:
            for result in results:
                logger.info(f"Product ID: {result.product_id}, Expected Qty: {result.expected_qty}, Confidence: {result.confidence_score}")
        else:
            logger.info("No results returned from predictor.")

    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())