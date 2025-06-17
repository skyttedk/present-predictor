from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import APIKeyHeader
from typing import Dict
import hashlib
import time

from .schemas.requests import AddPresentRequest
from .schemas.responses import CSVImportResponse, CSVImportSummary, DeleteAllPresentsResponse
from ..database.users import authenticate_user
from ..database import db_factory
from ..database.csv_import import import_presents_from_csv
import logging
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from ..scheduler.tasks import fetch_pending_present_attributes
from ..config.settings import settings

# Configure logging
# Ensure settings.LOG_LEVEL is defined in your src.config.settings
# Defaulting to INFO if not found or for simplicity in this example.
LOG_LEVEL = getattr(settings, "LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app_instance: FastAPI): # Renamed app to app_instance to avoid conflict
    # Startup
    logger.info("Application startup: Checking database...")
    try:
        # Check if tables exist before initializing
        check_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'present_attributes'"
        result = db_factory.execute_query(check_query)
        
        if not result:
            logger.info("Database tables not found. Initializing database...")
            db_factory.init_database()
            logger.info("Application startup: Database initialized.")
        else:
            logger.info("Database tables already exist. Skipping initialization.")
    except Exception as e:
        logger.error(f"Database check/initialization failed: {e}")
        # Continue anyway - database might be accessible but check failed
    
    logger.info("Application startup: Starting scheduler...")
    scheduler.add_job(fetch_pending_present_attributes, "interval", minutes=2, id="fetch_attributes_job")
    scheduler.start()
    logger.info("Application startup: Scheduler started. Job 'fetch_attributes_job' scheduled every 2 minutes.")
    
    yield
    
    # Shutdown
    logger.info("Application shutdown: Stopping scheduler...")
    if scheduler.running:
        scheduler.shutdown(wait=False)
    logger.info("Application shutdown: Scheduler stopped.")
    logger.info("Application shutdown complete.")

app = FastAPI(
    title="Predictive Gift Selection API",
    description="API for predicting gift selection quantities and managing scheduled tasks.",
    version="0.1.1", # Bump version
    lifespan=lifespan
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(api_key: str = Depends(api_key_header)) -> Dict:
    """
    Dependency to authenticate user via API key.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: API key is missing",
            headers={"WWW-Authenticate": "Header"},
        )
    user = authenticate_user(api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Header"},
        )
    return user

@app.get("/test")
async def test_endpoint(current_user: Dict = Depends(get_current_user)):
    """
    A simple test endpoint, protected by API key.
    """
    return {"message": f"Test endpoint is working! Hello {current_user.get('username')}!"}

@app.post("/addPresent", status_code=status.HTTP_201_CREATED)
async def add_present(
    request_data: AddPresentRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Adds a new present to the system for classification.
    Calculates a hash based on present details. If the hash already exists,
    the operation is skipped. Otherwise, the present is added with a
    'pending_classification' status.
    """
    present_name = request_data.present_name
    model_name = request_data.model_name
    model_no = request_data.model_no
    vendor = request_data.vendor

    # Implement user-specified hashing logic
    use_vendor = bool(vendor and vendor.lower() != 'gavefabrikken')
    if use_vendor:
        text_to_hash = f"{present_name} - {model_name} - {model_no}. Vendor {vendor}"
    else:
        text_to_hash = f"{present_name} - {model_name} - {model_no}."
    
    calculated_hash = hashlib.md5(text_to_hash.encode('utf-8')).hexdigest()

    # Check if hash exists
    query_check = "SELECT id FROM present_attributes WHERE present_hash = %s"
    existing_present = db_factory.execute_query(query_check, (calculated_hash,))

    if existing_present:
        # Use HTTPException for a more standard error response, or return 200 with a message
        # For now, returning 200 as per "skip insert. you get the point"
        return {
            "message": "Present with this combination of details already exists.",
            "present_hash": calculated_hash,
            "existing_id": existing_present[0]['id']
        }

    # If not existing, insert new present for classification
    query_insert = """
        INSERT INTO present_attributes
        (present_name, model_name, model_no, present_vendor, present_hash, classification_status)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    params_insert = (
        present_name,
        model_name,
        model_no,
        vendor,  # This is the 'present_vendor' from the request
        calculated_hash,
        'pending_classification'
    )
    
    try:
        new_id = db_factory.execute_write(query_insert, params_insert)
        return {
            "message": "Present added for classification.",
            "id": new_id,
            "present_hash": calculated_hash
        }
    except Exception as e: # Consider more specific exception handling
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add present to database: {str(e)}"
        )


@app.post("/addPresentsProcessed", response_model=CSVImportResponse, status_code=status.HTTP_201_CREATED)
async def add_presents_processed(
    file: UploadFile = File(..., description="CSV file containing pre-classified present attributes"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Import pre-classified present attributes from a CSV file.
    
    This endpoint allows bulk import of present attributes that have already been
    classified through external processes. The CSV must contain all required columns
    including classifications.
    
    Required CSV columns:
    - present_hash, present_name, present_vendor, model_name, model_no
    - item_main_category, item_sub_category, color, brand, vendor
    - target_demographic, utility_type, durability, usage_type
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV file"
        )
    
    # Read file content
    try:
        content = await file.read()
        csv_content = content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading file: {str(e)}"
        )
    
    if not csv_content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file is empty"
        )
    
    # Import presents
    try:
        result = import_presents_from_csv(csv_content)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create summary
        summary = CSVImportSummary(
            imported_count=result['imported_count'],
            skipped_count=result['skipped_count'],
            error_count=result['error_count'],
            total_processed=result['total_processed']
        )
        
        # Determine success message
        if result['error_count'] == 0:
            message = "CSV import completed successfully"
        elif result['imported_count'] > 0:
            message = f"CSV import completed with {result['error_count']} errors"
        else:
            message = "CSV import failed - no presents were imported"
        
        return CSVImportResponse(
            message=message,
            summary=summary,
            processing_time_ms=processing_time_ms
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during import: {str(e)}"
        )


@app.post("/deleteAllPresents", response_model=DeleteAllPresentsResponse, status_code=status.HTTP_200_OK)
async def delete_all_presents(
    current_user: Dict = Depends(get_current_user)
):
    """
    Delete all presents from the present_attributes table.
    
    ⚠️ WARNING: This endpoint will permanently delete ALL present data!
    This is intended for testing purposes only.
    """
    start_time = time.time()
    
    try:
        # Get count before deletion for response
        count_query = "SELECT COUNT(*) as count FROM present_attributes"
        count_result = db_factory.execute_query(count_query)
        total_count = count_result[0]['count'] if count_result else 0
        
        # Delete all presents
        delete_query = "DELETE FROM present_attributes"
        deleted_count = db_factory.execute_write(delete_query)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return DeleteAllPresentsResponse(
            message=f"All presents deleted successfully. Removed {deleted_count} records.",
            deleted_count=deleted_count,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting presents: {str(e)}"
        )