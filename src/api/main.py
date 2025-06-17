from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import APIKeyHeader
from typing import Dict
import hashlib
import time

from .schemas.requests import AddPresentRequest, CreateUserRequest, DeleteUserRequest, PredictRequest
from .schemas.responses import CSVImportResponse, CSVImportSummary, DeleteAllPresentsResponse, CountPresentsResponse, PresentCountByStatus, CreateUserResponse, ListUsersResponse, DeleteUserResponse, UserInfo, TailLogsResponse, ApiLogEntry, TransformedPredictResponse, TransformedPresent, TransformedEmployee
from ..data.cvr_client import fetch_industry_code
from ..data.gender_classifier import classify_employee_gender
from ..database.presents import get_present_by_hash
from ..database.users import authenticate_user, is_admin_user, create_user, list_users, delete_user, count_users
from ..database.api_logs import get_recent_logs
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

async def get_admin_user(api_key: str = Depends(api_key_header)) -> Dict:
    """
    Dependency to authenticate admin user via API key.
    """
    # Check if no users exist yet (special case for first user creation)
    user_count = count_users()
    if user_count == 0:
        # Allow creation of first user without any API key authentication
        # Return a dummy user object for first user creation
        return {"id": 0, "username": "system", "is_admin": True}
    
    # Normal authentication required when users exist
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: API key is missing",
            headers={"WWW-Authenticate": "Header"},
        )
    
    # Normal admin check for existing users
    if not is_admin_user(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    
    user = authenticate_user(api_key)
    return user

@app.get("/test")
async def test_endpoint(current_user: Dict = Depends(get_current_user)):
    """
    A simple test endpoint, protected by API key.
    """
    return {"message": f"Test endpoint is working! Hello {current_user.get('username')}!"}


@app.post("/predict", response_model=TransformedPredictResponse)
async def predict_endpoint(
    request_data: PredictRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Transform input data for prediction.
    
    This endpoint:
    1. Fetches industry code from CVR API
    2. Looks up present classifications from database
    3. Classifies employee genders
    4. Returns transformed data (no ML predictions yet)
    """
    
    # Step 1: Fetch industry code from CVR API
    branch = await fetch_industry_code(request_data.cvr)
    if not branch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not fetch industry code for CVR: {request_data.cvr}"
        )
    
    # Step 2: Transform presents
    transformed_presents = []
    missing_presents = []
    
    for present in request_data.presents:
        # Calculate hash using the same logic as /addPresent
        use_vendor = bool(present.vendor and present.vendor.lower() != 'gavefabrikken')
        if use_vendor:
            text_to_hash = f"{present.description} - {present.model_name} - {present.model_no}. Vendor {present.vendor}"
        else:
            text_to_hash = f"{present.description} - {present.model_name} - {present.model_no}."
        
        present_hash = hashlib.md5(text_to_hash.encode('utf-8')).hexdigest().upper()
        
        # Lookup in database
        attributes = get_present_by_hash(present_hash)
        
        if not attributes:
            missing_presents.append({
                "id": present.id,
                "description": present.description,
                "hash": present_hash
            })
            continue
        
        transformed_presents.append(TransformedPresent(
            id=present.id,
            item_main_category=attributes.get('item_main_category', ''),
            item_sub_category=attributes.get('item_sub_category', ''),
            color=attributes.get('color', ''),
            brand=attributes.get('brand', ''),
            vendor=attributes.get('vendor', ''),
            target_demographic=attributes.get('target_demographic', ''),
            utility_type=attributes.get('utility_type', ''),
            usage_type=attributes.get('usage_type', ''),
            durability=attributes.get('durability', '')
        ))
    
    # Check if any presents were not found
    if missing_presents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Some presents not found in database. Please classify them first using /addPresent endpoint.",
                "missing_presents": missing_presents
            }
        )
    
    # Step 3: Transform employees
    transformed_employees = []
    for employee in request_data.employees:
        gender = classify_employee_gender(employee.name)
        transformed_employees.append(TransformedEmployee(
            gender=gender.value  # Convert enum to string
        ))
    
    return TransformedPredictResponse(
        branch=branch,
        presents=transformed_presents,
        employees=transformed_employees
    )


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
    
    calculated_hash = hashlib.md5(text_to_hash.encode('utf-8')).hexdigest().upper()

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
@app.get("/countPresents", response_model=CountPresentsResponse, status_code=status.HTTP_200_OK)
async def count_presents(
    current_user: Dict = Depends(get_current_user)
):
    """
    Count presents grouped by classification status.
    
    Returns the total number of presents in the system and a breakdown
    by classification status (success, pending_classification, error_openai_api, etc.).
    """
    start_time = time.time()
    
    try:
        # Get count of presents grouped by status
        count_query = """
            SELECT 
                classification_status,
                COUNT(*) as count
            FROM present_attributes 
            GROUP BY classification_status
            ORDER BY classification_status
        """
        status_results = db_factory.execute_query(count_query)
        
        # Get total count
        total_query = "SELECT COUNT(*) as total FROM present_attributes"
        total_result = db_factory.execute_query(total_query)
        total_count = total_result[0]['total'] if total_result else 0
        
        # Format the results
        status_counts = []
        for row in status_results:
            status_counts.append(PresentCountByStatus(
                status=row['classification_status'],
                count=row['count']
            ))
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return CountPresentsResponse(
            message="Present counts retrieved successfully",
            total_presents=total_count,
            status_counts=status_counts,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error counting presents: {str(e)}"
        )


# User Management Endpoints

@app.post("/createUser", response_model=CreateUserResponse, status_code=status.HTTP_201_CREATED)
async def create_user_endpoint(
    request_data: CreateUserRequest,
    admin_user: Dict = Depends(get_admin_user)
):
    """
    Create a new user with API key.
    
    The first user created automatically becomes an admin.
    Subsequent users require admin authentication to create.
    
    Returns the new user information and API key (only shown once).
    """
    try:
        user_data = create_user(request_data.username)
        
        # Format timestamps for response
        user_info = UserInfo(
            id=user_data["user_id"],
            username=user_data["username"],
            is_active=True,
            is_admin=user_data["is_admin"],
            created_at=str(admin_user.get('created_at', '')),  # Use current time or similar
            updated_at=str(admin_user.get('updated_at', ''))
        )
        
        # Get fresh user data with timestamps
        fresh_users = list_users()
        fresh_user = next((u for u in fresh_users if u['id'] == user_data["user_id"]), None)
        if fresh_user:
            user_info.created_at = str(fresh_user['created_at'])
            user_info.updated_at = str(fresh_user['updated_at'])
        
        return CreateUserResponse(
            message="User created successfully",
            user=user_info,
            api_key=user_data["api_key"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )


@app.get("/users", response_model=ListUsersResponse, status_code=status.HTTP_200_OK)
async def list_users_endpoint(
    active_only: bool = False,
    admin_user: Dict = Depends(get_admin_user)
):
    """
    List all users in the system.
    
    Requires admin authentication.
    
    Args:
        active_only: If True, only return active users
    """
    try:
        users_data = list_users(active_only=active_only)
        
        # Convert to UserInfo format
        users = []
        for user_data in users_data:
            user_info = UserInfo(
                id=user_data["id"],
                username=user_data["username"],
                is_active=user_data["is_active"],
                is_admin=user_data.get("is_admin", False),
                created_at=str(user_data["created_at"]),
                updated_at=str(user_data["updated_at"])
            )
            users.append(user_info)
        
        return ListUsersResponse(
            message="Users retrieved successfully",
            users=users,
            total_count=len(users)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving users: {str(e)}"
        )


@app.post("/deleteUser", response_model=DeleteUserResponse, status_code=status.HTTP_200_OK)
async def delete_user_endpoint(
    request_data: DeleteUserRequest,
    admin_user: Dict = Depends(get_admin_user)
):
    """
    Delete a user from the system.
    
    Requires admin authentication.
    Permanently removes the user and all related data.
    """
    username_to_delete = request_data.username
    
    # Prevent admin from deleting themselves
    if admin_user.get('username') == username_to_delete:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own user account"
        )
    
    try:
        success = delete_user(username_to_delete)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{username_to_delete}' not found"
            )
        
        return DeleteUserResponse(
            message="User deleted successfully",
            username=username_to_delete
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting user: {str(e)}"
        )


@app.get("/logs/tail", response_model=TailLogsResponse, status_code=status.HTTP_200_OK)
async def get_tail_logs(
    limit: int = 10,
    admin_user: Dict = Depends(get_admin_user)
):
    """
    Get recent API log entries (tail logs).
    
    Requires admin authentication.
    Returns the most recent log entries from the API call log.
    
    Args:
        limit: Number of log entries to return (default: 10, max: 100)
    """
    # Validate limit parameter
    if limit < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be at least 1"
        )
    if limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit cannot exceed 100"
        )
    
    start_time = time.time()
    
    try:
        logs_data = get_recent_logs(limit=limit)
        
        # Convert to ApiLogEntry format
        logs = []
        for log_data in logs_data:
            log_entry = ApiLogEntry(
                id=log_data["id"],
                date_time=log_data["date_time"],
                username=log_data["username"],
                api_route=log_data["api_route"],
                response_status_code=log_data["response_status_code"],
                response_time_ms=log_data["response_time_ms"],
                error_message=log_data["error_message"],
                request_payload=log_data["request_payload"],
                response_payload=log_data["response_payload"]
            )
            logs.append(log_entry)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return TailLogsResponse(
            message="Recent logs retrieved successfully",
            logs=logs,
            total_count=len(logs),
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving logs: {str(e)}"
        )