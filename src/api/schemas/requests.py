"""
API request models for the Predictive Gift Selection System.
Defines the structure for incoming API requests.
"""

from typing import List
from pydantic import BaseModel, Field, validator


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