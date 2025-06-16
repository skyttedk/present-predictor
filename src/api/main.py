from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Dict
import hashlib

from src.api.schemas.requests import AddPresentRequest
from src.database.users import authenticate_user
from src.database.db import execute_query, execute_write

app = FastAPI(
    title="Predictive Gift Selection API",
    description="API for predicting gift selection quantities.",
    version="0.1.0",
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
    query_check = "SELECT id FROM present_attributes WHERE present_hash = ?"
    existing_present = execute_query(query_check, (calculated_hash,))

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
        VALUES (?, ?, ?, ?, ?, ?)
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
        new_id = execute_write(query_insert, params_insert)
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