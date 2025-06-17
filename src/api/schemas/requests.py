"""
API request models for the Predictive Gift Selection System.
Defines the structure for incoming API requests.
"""

from typing import List
from pydantic import BaseModel, Field, validator, field_validator


class GiftItem(BaseModel):
    """Model for gift items in API requests."""
    
    product_id: str = Field(..., description="Unique product identifier", min_length=1)
    description: str = Field(..., description="Product description for classification", min_length=1)
    
    @validator('product_id')
    def validate_product_id(cls, v):
        """Validate product ID format."""
        if not v.strip():
            raise ValueError("Product ID cannot be empty")
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        """Validate description content."""
        if not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()


class Employee(BaseModel):
    """Model for employee information in API requests."""
    
    name: str = Field(..., description="Employee full name", min_length=1)
    
    @validator('name')
    def validate_name(cls, v):
        """Validate employee name."""
        if not v.strip():
            raise ValueError("Employee name cannot be empty")
        return v.strip()


class PredictionRequest(BaseModel):
    """
    Main API request model for demand prediction.
    
    This represents Step 1 of the three-step processing pipeline:
    Raw request input with branch info, gifts, and employees.
    """
    
    branch_no: str = Field(..., description="Branch number identifier", min_length=1)
    gifts: List[GiftItem] = Field(..., description="List of gifts to predict demand for", min_items=1)
    employees: List[Employee] = Field(..., description="List of employees", min_items=1)
    
    @validator('branch_no')
    def validate_branch_no(cls, v):
        """Validate branch number format."""
        if not v.strip():
            raise ValueError("Branch number cannot be empty")
        return v.strip()
    
    @validator('gifts')
    def validate_gifts(cls, v):
        """Validate gifts list."""
        if not v:
            raise ValueError("At least one gift must be provided")
        
        # Check for duplicate product IDs
        product_ids = [gift.product_id for gift in v]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Duplicate product IDs are not allowed")
        
        return v
    
    @validator('employees')
    def validate_employees(cls, v):
        """Validate employees list."""
        if not v:
            raise ValueError("At least one employee must be provided")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "branch_no": "621000",
                "gifts": [
                    {
                        "product_id": "ABC123",
                        "description": "Red ceramic mug with handle"
                    },
                    {
                        "product_id": "DEF456", 
                        "description": "Kitchen knife set stainless steel"
                    }
                ],
                "employees": [
                    {"name": "John Doe"},
                    {"name": "Jane Smith"},
                    {"name": "Erik Nielsen"}
                ]
            }
        }

class AddPresentRequest(BaseModel):
    """Model for adding a new present for classification."""
    present_name: str = Field(..., description="Name of the present", min_length=1)
    model_name: str = Field(..., description="Model name of the present", min_length=1)
    model_no: str = Field(..., description="Model number of the present", min_length=1)
    vendor: str = Field(..., description="Vendor of the present", min_length=1)

    @validator('*', pre=True, always=True)
    def strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator('present_name', 'model_name', 'model_no', 'vendor', mode='before')
    def check_not_empty(cls, v, info):
        if isinstance(v, str) and not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        if v is None: # Handles cases where field might be None if not caught by Field(...)
            raise ValueError(f"{info.field_name} cannot be None")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "present_name": "Luxury Coffee Mug",
                "model_name": "Grande Series",
                "model_no": "CMG-500X",
                "vendor": "Premium Homewares"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Model for batch prediction requests (future enhancement)."""
    
    requests: List[PredictionRequest] = Field(..., description="List of prediction requests", min_items=1, max_items=100)
    
    @validator('requests')
    def validate_batch_size(cls, v):
        """Validate batch size limits."""
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "requests": [
                    {
                        "branch_no": "621000",
                        "gifts": [{"product_id": "ABC123", "description": "Red mug"}],
                        "employees": [{"name": "John Doe"}]
                    },
                    {
                        "branch_no": "841100", 
                        "gifts": [{"product_id": "DEF456", "description": "Kitchen knife"}],
                        "employees": [{"name": "Jane Smith"}]
                    }
                ]
            }
        }


class CSVUploadRequest(BaseModel):
    """Model for CSV file upload metadata (if needed for future enhancements)."""
    
    description: str = Field(default="", description="Optional description of the CSV import")
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "Bulk import of pre-classified present attributes"
            }
        }


class CreateUserRequest(BaseModel):
    """Model for creating a new user."""
    username: str = Field(..., description="Unique username for the new user", min_length=1, max_length=50)
    
    @field_validator('username', mode='before')
    def validate_username(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Username cannot be empty")
            if len(v) < 3:
                raise ValueError("Username must be at least 3 characters long")
            # Basic username validation (alphanumeric, underscore, hyphen)
            if not v.replace('_', '').replace('-', '').isalnum():
                raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe"
            }
        }


class DeleteUserRequest(BaseModel):
    """Model for deleting a user."""
    username: str = Field(..., description="Username to delete", min_length=1)
    
    @field_validator('username', mode='before')
    def validate_username(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Username cannot be empty")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe"
            }
        }