"""
API response models for the Predictive Gift Selection System.
Defines the structure for API responses.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class PredictionResult(BaseModel):
    """
    Model for individual prediction results.
    
    This represents the final output of Step 3 in the processing pipeline.
    """
    
    product_id: str = Field(..., description="Product identifier from the request")
    expected_qty: int = Field(..., description="Predicted demand quantity", ge=0)
    confidence_score: Optional[float] = Field(None, description="Prediction confidence (0-1)", ge=0.0, le=1.0)
    
    @validator('expected_qty')
    def validate_quantity(cls, v):
        """Ensure quantity is non-negative."""
        if v < 0:
            raise ValueError("Expected quantity cannot be negative")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "ABC123",
                "expected_qty": 15,
                "confidence_score": 0.85
            }
        }


class PredictionResponse(BaseModel):
    """
    Main API response model for demand predictions.
    Contains the list of prediction results and metadata.
    """
    
    branch_no: str = Field(..., description="Branch number from request")
    predictions: List[PredictionResult] = Field(..., description="List of demand predictions")
    total_employees: int = Field(..., description="Total number of employees processed", ge=0)
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds", ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    @validator('predictions')
    def validate_predictions(cls, v):
        """Validate predictions list."""
        if not v:
            raise ValueError("At least one prediction must be provided")
        
        # Check for duplicate product IDs
        product_ids = [pred.product_id for pred in v]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Duplicate product IDs in predictions")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "branch_no": "621000",
                "predictions": [
                    {
                        "product_id": "ABC123",
                        "expected_qty": 15,
                        "confidence_score": 0.85
                    },
                    {
                        "product_id": "DEF456",
                        "expected_qty": 8,
                        "confidence_score": 0.92
                    }
                ],
                "total_employees": 25,
                "processing_time_ms": 150.5,
                "timestamp": "2025-12-06T18:00:00Z"
            }
        }


class ClassificationDebugInfo(BaseModel):
    """Debug information for gift classification (development mode)."""
    
    product_id: str = Field(..., description="Product identifier")
    original_description: str = Field(..., description="Original product description")
    classified_attributes: Dict[str, Any] = Field(..., description="Classified attributes")
    classification_confidence: Optional[float] = Field(None, description="Classification confidence")
    processing_notes: Optional[List[str]] = Field(None, description="Processing notes or warnings")


class EmployeeDebugInfo(BaseModel):
    """Debug information for employee processing (development mode)."""
    
    original_name: str = Field(..., description="Original employee name")
    classified_gender: str = Field(..., description="Classified gender")
    gender_confidence: Optional[float] = Field(None, description="Gender classification confidence")
    processing_notes: Optional[List[str]] = Field(None, description="Processing notes")


class PredictionDebugResponse(BaseModel):
    """
    Extended response model with debug information (development mode only).
    Includes detailed information about the classification and processing steps.
    """
    
    # Main response data
    branch_no: str = Field(..., description="Branch number from request")
    predictions: List[PredictionResult] = Field(..., description="List of demand predictions")
    total_employees: int = Field(..., description="Total number of employees processed")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    # Debug information
    classification_debug: List[ClassificationDebugInfo] = Field(..., description="Gift classification debug info")
    employee_debug: List[EmployeeDebugInfo] = Field(..., description="Employee processing debug info")
    feature_engineering_info: Optional[Dict[str, Any]] = Field(None, description="Feature engineering details")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model prediction details")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction requests."""
    
    total_requests: int = Field(..., description="Total number of requests processed", ge=0)
    successful_predictions: int = Field(..., description="Number of successful predictions", ge=0)
    failed_predictions: int = Field(..., description="Number of failed predictions", ge=0)
    results: List[PredictionResponse] = Field(..., description="Individual prediction results")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details for failed predictions")
    total_processing_time_ms: Optional[float] = Field(None, description="Total processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    database_connected: Optional[bool] = Field(None, description="Database connection status")
    dependencies: Optional[Dict[str, str]] = Field(None, description="Dependency versions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2025-12-06T18:00:00Z",
                "model_loaded": True,
                "database_connected": None,
                "dependencies": {
                    "fastapi": "0.115.12",
                    "xgboost": "3.0.2",
                    "pandas": "2.0.0"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error_code: str = Field(..., description="Error code identifier")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    
    class Config:
        json_schema_extra = {
        }
        protected_namespaces = ()


class ValidationErrorResponse(BaseModel):
    """Validation error response with field-specific details."""
    
    error_code: str = Field(default="VALIDATION_ERROR", description="Error code")
    error_message: str = Field(..., description="General error message")
    validation_errors: List[Dict[str, Any]] = Field(..., description="Field-specific validation errors")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "VALIDATION_ERROR",
                "error_message": "Request validation failed",
                "validation_errors": [
                    {
                        "field": "gifts.0.product_id",
                        "message": "Product ID cannot be empty",
                        "input_value": ""
                    },
                    {
                        "field": "employees",
                        "message": "At least one employee must be provided",
                        "input_value": []
                    }
                ],
                "timestamp": "2025-12-06T18:00:00Z"
            }
        }


class CSVImportSummary(BaseModel):
    """Summary statistics for CSV import operation."""
    
    imported_count: int = Field(..., description="Number of presents successfully imported", ge=0)
    skipped_count: int = Field(..., description="Number of presents skipped (already exist)", ge=0)
    error_count: int = Field(..., description="Number of presents with import errors", ge=0)
    total_processed: int = Field(..., description="Total number of rows processed", ge=0)
    
    @validator('total_processed')
    def validate_total(cls, v, values):
        """Ensure total matches sum of other counts."""
        expected_total = values.get('imported_count', 0) + values.get('skipped_count', 0) + values.get('error_count', 0)
        if v != expected_total:
            raise ValueError(f"Total processed ({v}) should equal sum of imported, skipped, and error counts ({expected_total})")
        return v


class CSVImportResponse(BaseModel):
    """Response model for CSV import operation."""
    
    message: str = Field(..., description="Success or failure message")
    summary: CSVImportSummary = Field(..., description="Import operation summary")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds", ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Import timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "CSV import completed successfully",
                "summary": {
                    "imported_count": 150,
                    "skipped_count": 25,
                    "error_count": 5,
                    "total_processed": 180
                },
                "processing_time_ms": 2500.0,
                "timestamp": "2025-06-17T08:00:00Z"
            }
        }


class DeleteAllPresentsResponse(BaseModel):
    """Response model for delete all presents operation."""
    
    message: str = Field(..., description="Success message")
    deleted_count: int = Field(..., description="Number of presents deleted", ge=0)
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds", ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Delete timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "All presents deleted successfully",
                "deleted_count": 8913,
                "processing_time_ms": 120.5,
                "timestamp": "2025-06-17T09:31:00Z"
            }
        }