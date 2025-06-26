# src/api/schemas/optimal_requests.py

"""
Optimal API Request/Response Schemas

This module defines the optimal API request format that provides perfect alignment
between training data and prediction requests using CVR and direct employee counts.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class OptimalGiftItem(BaseModel):
    """
    Gift item for optimal prediction request.
    """
    id: str = Field(..., description="Unique gift identifier")
    description: str = Field(..., description="Gift description")
    model_name: Optional[str] = Field(None, description="Model name")
    model_no: Optional[str] = Field(None, description="Model number")
    vendor: Optional[str] = Field(None, description="Vendor name")

    class Config:
        schema_extra = {
            "example": {
                "id": "1",
                "description": "Tisvilde Pizzaovn",
                "model_name": "Tisvilde Pizzaovn",
                "model_no": "",
                "vendor": "GaveFabrikken"
            }
        }


class OptimalPredictionRequest(BaseModel):
    """
    Optimal prediction request format with CVR and direct employee counts.
    
    This format provides perfect alignment with training data structure and
    eliminates the need for real-time gender classification.
    """
    cvr: str = Field(..., description="Company CVR number for context")
    male_count: int = Field(..., ge=0, description="Number of male employees")
    female_count: int = Field(..., ge=0, description="Number of female employees")
    presents: List[OptimalGiftItem] = Field(..., min_items=1, max_items=100, description="List of gifts to predict")

    @validator('cvr')
    def validate_cvr(cls, v):
        """Validate CVR format (Danish company registration number)."""
        if not v or len(v.strip()) == 0:
            raise ValueError("CVR cannot be empty")
        # Basic CVR validation (8 digits)
        cvr_clean = v.strip()
        if not cvr_clean.isdigit() or len(cvr_clean) != 8:
            raise ValueError("CVR must be 8 digits")
        return cvr_clean

    @validator('male_count', 'female_count')
    def validate_employee_counts(cls, v):
        """Validate employee counts are reasonable."""
        if v < 0:
            raise ValueError("Employee count cannot be negative")
        if v > 10000:
            raise ValueError("Employee count seems unreasonably high (>10000)")
        return v

    @validator('presents')
    def validate_total_employees(cls, v, values):
        """Validate that total employees is reasonable."""
        if 'male_count' in values and 'female_count' in values:
            total_employees = values['male_count'] + values['female_count']
            if total_employees == 0:
                raise ValueError("Total employee count cannot be zero")
            if total_employees > 10000:
                raise ValueError("Total employee count seems unreasonably high (>10000)")
        return v

    class Config:
        schema_extra = {
            "example": {
                "cvr": "28892055",
                "male_count": 12,
                "female_count": 11,
                "presents": [
                    {
                        "id": "1",
                        "description": "Tisvilde Pizzaovn",
                        "model_name": "Tisvilde Pizzaovn",
                        "model_no": "",
                        "vendor": "GaveFabrikken"
                    },
                    {
                        "id": "2",
                        "description": "Fiskars Knife Set",
                        "model_name": "Professional Chef Set",
                        "model_no": "FK-2024",
                        "vendor": "Fiskars"
                    }
                ]
            }
        }


class GenderBreakdown(BaseModel):
    """
    Gender-specific prediction breakdown.
    """
    predicted_rate: float = Field(..., description="Predicted selection rate for this gender")
    expected_count: float = Field(..., description="Expected number of selections for this gender")
    exposure: int = Field(..., description="Number of employees in this gender group")


class OptimalPredictionResult(BaseModel):
    """
    Enhanced prediction result with detailed breakdown.
    """
    product_id: str = Field(..., description="Product identifier")
    expected_qty: float = Field(..., description="Total expected quantity across all employees")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    gender_breakdown: Dict[str, GenderBreakdown] = Field(..., description="Gender-specific predictions")
    total_exposure: int = Field(..., description="Total number of employees")
    error: Optional[str] = Field(None, description="Error message if prediction failed")

    class Config:
        schema_extra = {
            "example": {
                "product_id": "1",
                "expected_qty": 8.5,
                "confidence_score": 0.82,
                "gender_breakdown": {
                    "male": {
                        "predicted_rate": 0.42,
                        "expected_count": 5.04,
                        "exposure": 12
                    },
                    "female": {
                        "predicted_rate": 0.31,
                        "expected_count": 3.41,
                        "exposure": 11
                    }
                },
                "total_exposure": 23
            }
        }


class OptimalPredictionResponse(BaseModel):
    """
    Complete prediction response.
    """
    predictions: List[OptimalPredictionResult] = Field(..., description="Prediction results")
    request_summary: Dict[str, Any] = Field(..., description="Summary of the request")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "product_id": "1",
                        "expected_qty": 8.5,
                        "confidence_score": 0.82,
                        "gender_breakdown": {
                            "male": {
                                "predicted_rate": 0.42,
                                "expected_count": 5.04,
                                "exposure": 12
                            },
                            "female": {
                                "predicted_rate": 0.31,
                                "expected_count": 3.41,
                                "exposure": 11
                            }
                        },
                        "total_exposure": 23
                    }
                ],
                "request_summary": {
                    "cvr": "28892055",
                    "total_employees": 23,
                    "male_count": 12,
                    "female_count": 11,
                    "num_gifts": 1
                },
                "model_info": {
                    "model_type": "CatBoost Regressor (Enhanced - RMSE)",
                    "target": "selection_rate",
                    "architecture": "Three-file company-level architecture"
                },
                "processing_time_ms": 45.2
            }
        }


class BusinessValidationMetrics(BaseModel):
    """
    Business logic validation metrics for the predictions.
    """
    total_predicted_selections: float = Field(..., description="Sum of all expected quantities")
    total_employees: int = Field(..., description="Total number of employees")
    overall_selection_rate: float = Field(..., description="Total selections / total employees")
    rate_validation: str = Field(..., description="Whether the rate is within expected business range")
    recommendations: List[str] = Field(default=[], description="Business recommendations based on predictions")

    class Config:
        schema_extra = {
            "example": {
                "total_predicted_selections": 42.3,
                "total_employees": 23,
                "overall_selection_rate": 1.84,
                "rate_validation": "REASONABLE",
                "recommendations": [
                    "Selection rate is within expected range (0.5-2.5)",
                    "Consider inventory planning based on these predictions"
                ]
            }
        }


class OptimalPredictionResponseWithValidation(OptimalPredictionResponse):
    """
    Extended response that includes business validation metrics.
    """
    business_validation: BusinessValidationMetrics = Field(..., description="Business logic validation")


# Legacy request format for backward compatibility
class LegacyPredictionRequest(BaseModel):
    """
    Legacy prediction request format (for backward compatibility).
    """
    branch_no: str = Field(..., description="Branch number")
    gifts: List[Dict[str, str]] = Field(..., description="List of gifts")
    employees: List[Dict[str, str]] = Field(..., description="List of employees")

    class Config:
        schema_extra = {
            "example": {
                "branch_no": "12600",
                "gifts": [
                    {
                        "product_id": "1",
                        "description": "Tisvilde Pizzaovn"
                    }
                ],
                "employees": [
                    {"name": "Lars Nielsen"},
                    {"name": "Anna Müller"}
                ]
            }
        }


class ConversionResponse(BaseModel):
    """
    Response for converting legacy request to optimal format.
    """
    optimal_request: OptimalPredictionRequest = Field(..., description="Converted optimal request")
    gender_classification_results: List[Dict[str, Any]] = Field(..., description="Gender classification details")
    conversion_warnings: List[str] = Field(default=[], description="Any warnings during conversion")

    class Config:
        schema_extra = {
            "example": {
                "optimal_request": {
                    "cvr": "28892055",
                    "male_count": 1,
                    "female_count": 1,
                    "presents": [
                        {
                            "id": "1",
                            "description": "Tisvilde Pizzaovn",
                            "model_name": "",
                            "model_no": "",
                            "vendor": ""
                        }
                    ]
                },
                "gender_classification_results": [
                    {"name": "Lars Nielsen", "gender": "male", "confidence": "high"},
                    {"name": "Anna Müller", "gender": "female", "confidence": "high"}
                ],
                "conversion_warnings": []
            }
        }