# /predict Endpoint Implementation Plan

## Overview
This document outlines the implementation plan for the new `/predict` endpoint that transforms input data (CVR, presents, employees) into a format suitable for ML prediction.

## Endpoint Specification

### Request Format
```json
POST /predict
{
  "cvr": "12345678",
  "presents": [
    {
      "id": 147748,
      "description": "Cavalluzzi kabine trolley, hvid",
      "model_name": "Cavalluzzi kabine trolley, hvid",
      "model_no": "",
      "vendor": "TravelGear"
    }
  ],
  "employees": [
    {
      "name": "GUNHILD SØRENSEN"
    }
  ]
}
```

### Response Format
```json
{
  "branch": "621000",
  "presents": [
    {
      "id": 147748,
      "item_main_category": "Travel",
      "item_sub_category": "Luggage",
      "color": "white",
      "brand": "Cavalluzzi",
      "vendor": "TravelGear",
      "target_demographic": "unisex",
      "utility_type": "practical",
      "usage_type": "individual",
      "durability": "durable"
    }
  ],
  "employees": [
    {
      "gender": "female"
    }
  ]
}
```

## Implementation Steps

### 1. Create CVR API Client Module
**File**: `src/data/cvr_client.py`

```python
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def fetch_industry_code(cvr: str) -> Optional[str]:
    """
    Fetch industry code from Danish CVR API.
    
    Args:
        cvr: Danish CVR number
        
    Returns:
        Industry code as string, or None if not found
    """
    url = f"https://cvrapi.dk/api?search={cvr}&country=dk"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                industry_code = data.get("industrycode")
                if industry_code:
                    logger.info(f"Found industry code {industry_code} for CVR {cvr}")
                    return str(industry_code)
                else:
                    logger.warning(f"No industry code in response for CVR {cvr}")
            else:
                logger.error(f"CVR API returned status {response.status_code} for CVR {cvr}")
                
    except httpx.TimeoutException:
        logger.error(f"Timeout fetching industry code for CVR {cvr}")
    except Exception as e:
        logger.error(f"Error fetching industry code for CVR {cvr}: {e}")
    
    return None
```

### 2. Add Request/Response Models

**File**: `src/api/schemas/requests.py` (additions)

```python
class PredictPresent(BaseModel):
    """Present item for prediction request."""
    id: int = Field(..., description="Present ID")
    description: str = Field(..., description="Present description (name)")
    model_name: str = Field(..., description="Model name")
    model_no: str = Field(..., description="Model number")
    vendor: str = Field(..., description="Vendor name")
    
    @validator('*', pre=True)
    def strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

class PredictEmployee(BaseModel):
    """Employee for prediction request."""
    name: str = Field(..., description="Employee name")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Employee name cannot be empty")
        return v.strip()

class PredictRequest(BaseModel):
    """Request model for /predict endpoint."""
    cvr: str = Field(..., description="Danish CVR number")
    presents: List[PredictPresent] = Field(..., min_items=1)
    employees: List[PredictEmployee] = Field(..., min_items=1)
    
    @validator('cvr')
    def validate_cvr(cls, v):
        # Basic CVR validation (8 digits)
        v = v.strip()
        if not v.isdigit() or len(v) != 8:
            raise ValueError("CVR must be 8 digits")
        return v
```

**File**: `src/api/schemas/responses.py` (additions)

```python
class TransformedPresent(BaseModel):
    """Transformed present with classification attributes."""
    id: int
    item_main_category: str
    item_sub_category: str
    color: str
    brand: str
    vendor: str
    target_demographic: str
    utility_type: str
    usage_type: str
    durability: str

class TransformedEmployee(BaseModel):
    """Transformed employee with gender."""
    gender: str

class TransformedPredictResponse(BaseModel):
    """Response model for /predict endpoint."""
    branch: str
    presents: List[TransformedPresent]
    employees: List[TransformedEmployee]
```

### 3. Add Database Lookup Function

**File**: `src/database/presents.py` (addition)

```python
def get_present_by_hash(present_hash: str) -> Optional[Dict[str, Any]]:
    """
    Get present attributes by hash, only if successfully classified.
    
    Args:
        present_hash: MD5 hash of present details
        
    Returns:
        Present attributes dict or None if not found/not classified
    """
    query = """
        SELECT 
            item_main_category,
            item_sub_category,
            color,
            brand,
            vendor,
            target_demographic,
            utility_type,
            usage_type,
            durability
        FROM present_attributes
        WHERE present_hash = %s
        AND classification_status = 'success'
        LIMIT 1
    """
    
    results = db_factory.execute_query(query, (present_hash,))
    return results[0] if results else None
```

### 4. Implement the Endpoint

**File**: `src/api/main.py` (addition)

```python
# Add imports
from .schemas.requests import PredictRequest
from .schemas.responses import TransformedPredictResponse, TransformedPresent, TransformedEmployee
from ..data.cvr_client import fetch_industry_code
from ..data.gender_classifier import classify_employee_gender
from ..database.presents import get_present_by_hash

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
        
        present_hash = hashlib.md5(text_to_hash.encode('utf-8')).hexdigest()
        
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
```

### 5. Update Dependencies

**File**: `requirements.txt` (addition)
```
httpx>=0.24.0
```

## Data Flow

```mermaid
flowchart TD
    A[Client Request] --> B[/predict Endpoint]
    B --> C[Validate Request]
    C --> D[Fetch CVR Industry Code]
    D --> E[Transform Presents]
    E --> F[Transform Employees]
    F --> G[Return Response]
    
    D --> D1[CVR API Call]
    D1 --> D2[Extract industrycode]
    
    E --> E1[Calculate Hash]
    E1 --> E2[Database Lookup]
    E2 --> E3{Found?}
    E3 -->|Yes| E4[Add to Response]
    E3 -->|No| E5[Add to Missing List]
    
    F --> F1[Gender Classification]
    F1 --> F2[Use gender_guesser]
```

## Key Considerations

### Hash Calculation
- Uses the same logic as `/addPresent` endpoint
- Includes vendor in hash if vendor != 'gavefabrikken'
- MD5 hash of formatted string

### Error Handling
1. **CVR Not Found**: Returns 400 with clear error message
2. **Present Not Found**: Returns 400 with list of missing presents
3. **Validation Errors**: Handled by Pydantic models

### Performance
- Async CVR API call for non-blocking operation
- Single database query per present
- Batch gender classification for employees

## Testing

### Unit Tests
1. Test CVR API client with mock responses
2. Test hash calculation consistency
3. Test gender classification accuracy
4. Test request validation

### Integration Tests
1. Test full endpoint flow with real data
2. Test error scenarios
3. Test performance with large batches

### Example Test Cases

**Valid Request**:
```python
def test_predict_valid_request():
    response = client.post("/predict", json={
        "cvr": "12345678",
        "presents": [{
            "id": 1,
            "description": "Test Mug",
            "model_name": "Ceramic Blue",
            "model_no": "CM-001",
            "vendor": "MugCo"
        }],
        "employees": [{"name": "John Doe"}]
    })
    assert response.status_code == 200
    assert response.json()["branch"] == "621000"
```

**Missing Present**:
```python
def test_predict_missing_present():
    response = client.post("/predict", json={
        "cvr": "12345678",
        "presents": [{
            "id": 999,
            "description": "Unknown Item",
            "model_name": "Unknown",
            "model_no": "UNK-001",
            "vendor": "Unknown"
        }],
        "employees": [{"name": "John Doe"}]
    })
    assert response.status_code == 400
    assert "missing_presents" in response.json()["detail"]
```

## Future Enhancements

1. **Caching**: Add Redis caching for CVR lookups
2. **Batch Processing**: Optimize database queries for multiple presents
3. **ML Integration**: Add actual prediction logic once model is ready
4. **Monitoring**: Add metrics for API performance and accuracy

## Security Considerations

1. **API Key Authentication**: Already implemented via `get_current_user`
2. **Input Validation**: Handled by Pydantic models
3. **Rate Limiting**: Consider adding for CVR API calls
4. **Error Messages**: Ensure no sensitive data in error responses

## Deployment Notes

1. Ensure `httpx` is installed in production
2. Test CVR API connectivity from production environment
3. Monitor API response times
4. Set up alerts for high error rates