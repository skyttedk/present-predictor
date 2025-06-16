import asyncio
import logging
from typing import List, Dict, Any

from src.database.presents import get_pending_classification_presents, update_classified_present_attributes
# Import OpenAI client instead of classifier function
from src.data.openai_client import create_openai_client, OpenAIClassificationError, GiftAttributes
from src.database.db import get_db # Corrected import

logger = logging.getLogger(__name__)

async def fetch_pending_present_attributes():
    """
    Fetches attributes for presents with 'pending_classification' status
    from OpenAI and updates the database.
    """
    logger.info("Scheduler task: Starting to fetch pending present attributes.")
    openai_client = None
    try:
        # get_pending_classification_presents and update_classified_present_attributes
        # now manage their own DB connections via the get_db context manager.
        # No need for explicit db connection management here.
        pending_presents: List[Dict[str, Any]] = get_pending_classification_presents()

        if not pending_presents:
            logger.info("Scheduler task: No presents pending classification found.")
            return

        logger.info(f"Scheduler task: Found {len(pending_presents)} presents pending classification.")
        
        openai_client = create_openai_client() # Create client once

        for present in pending_presents:
            present_hash = present["present_hash"]
            present_name = present.get("present_name")
            # present_vendor is the vendor of the present itself, used in description if available
            present_vendor_original = present.get("present_vendor")
            model_name = present.get("model_name")
            model_no = present.get("model_no")
            
            logger.info(f"Scheduler task: Processing present_hash: {present_hash}")

            if not all([present_name, model_name, model_no]):
                logger.error(
                    f"Scheduler task: Missing required fields (name, model, model_no) for present_hash {present_hash}. "
                    f"Present data: {present}"
                )
                update_classified_present_attributes(
                    present_hash=present_hash,
                    attributes={},
                    new_status="error_missing_details"
                )
                continue

            description_parts = [present_name, model_name, model_no]
            if present_vendor_original:
                description_parts.append(f"Vendor: {present_vendor_original}")
            description_for_openai = " | ".join(filter(None, description_parts))
            
            thread_id_val, run_id_val = None, None # Initialize for potential error logging

            try:
                attributes_model, thread_id_val, run_id_val = await openai_client.classify_product(description_for_openai)
                
                if attributes_model:
                    attributes_dict = attributes_model.model_dump() # Convert Pydantic model to dict
                    update_classified_present_attributes(
                        present_hash=present_hash,
                        attributes=attributes_dict,
                        new_status="success",
                        thread_id=thread_id_val,
                        run_id=run_id_val
                    )
                else: # Should ideally be caught by OpenAIClassificationError if classify_product ensures it
                    logger.warning(f"Scheduler task: No attributes model returned for present_hash: {present_hash} (Thread: {thread_id_val}, Run: {run_id_val})")
                    update_classified_present_attributes(
                        present_hash=present_hash,
                        attributes={},
                        new_status="error_openai_nodata",
                        thread_id=thread_id_val,
                        run_id=run_id_val
                    )

            except OpenAIClassificationError as e:
                logger.error(f"Scheduler task: OpenAI classification error for present_hash {present_hash}: {e}", exc_info=False) # exc_info=False as error includes IDs
                # Extract thread_id and run_id from error if possible, or use the ones from the call
                update_classified_present_attributes(
                    present_hash=present_hash,
                    attributes={},
                    new_status="error_openai_api",
                    thread_id=thread_id_val, # Log IDs if available
                    run_id=run_id_val
                )
            except Exception as e:
                logger.error(f"Scheduler task: Unexpected error processing present_hash {present_hash}: {e}", exc_info=True)
                update_classified_present_attributes(
                    present_hash=present_hash,
                    attributes={},
                    new_status="error_processing_failed",
                    thread_id=thread_id_val, # Log IDs if available
                    run_id=run_id_val
                )
            
            await asyncio.sleep(1)

        logger.info("Scheduler task: Finished fetching pending present attributes.")

    except Exception as e:
        logger.error(f"Scheduler task: An unexpected error occurred in main task logic: {e}", exc_info=True)
    finally:
        if openai_client:
            await openai_client.close()
        # No explicit db.close() needed here as connections are managed by get_db context manager

if __name__ == "__main__":
    # Example of how to run the task manually for testing
    # Configure logging for standalone run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # This is a simplified setup. In a real scenario, ensure DB path and OpenAI keys are configured.
    # from src.config.settings import settings # Ensure settings are loaded if needed by underlying functions
    # print(f"DB Path for manual run: {settings.DATABASE_URL}")
    
    async def main():
        await fetch_pending_present_attributes()

    asyncio.run(main())