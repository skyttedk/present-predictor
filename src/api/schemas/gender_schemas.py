"""
Request and response schemas for gender classification endpoints
"""

from pydantic import BaseModel, Field
from typing import List

class GenderClassificationRequest(BaseModel):
    """Request model for single name gender classification"""
    name: str = Field(..., description="Full name to classify", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Lars Nielsen"
            }
        }

class GenderBatchRequest(BaseModel):
    """Request model for batch gender classification"""
    names: List[str] = Field(..., description="List of names to classify", min_length=1, max_length=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "names": ["Lars Nielsen", "Anna Müller", "John Smith"]
            }
        }

class GenderClassificationResponse(BaseModel):
    """Response model for single gender classification result"""
    name: str = Field(..., description="Original name provided")
    gender: str = Field(..., description="Classified gender: 'male', 'female', or 'unknown'")
    confidence: str = Field(..., description="Confidence level: 'high' or 'low'")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Lars Nielsen",
                "gender": "male",
                "confidence": "high",
                "processing_time_ms": 2.5
            }
        }

class GenderBatchResponse(BaseModel):
    """Response model for batch gender classification"""
    results: List[GenderClassificationResponse] = Field(..., description="List of classification results")
    total_processed: int = Field(..., description="Total number of names processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "name": "Lars Nielsen",
                        "gender": "male", 
                        "confidence": "high",
                        "processing_time_ms": 15.2
                    },
                    {
                        "name": "Anna Müller",
                        "gender": "female",
                        "confidence": "high", 
                        "processing_time_ms": 15.2
                    }
                ],
                "total_processed": 2,
                "processing_time_ms": 15.2
            }
        }