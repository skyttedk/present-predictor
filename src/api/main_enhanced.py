"""
Enhanced FastAPI Application for Predictive Gift Selection System

This API implements the optimal request structure with direct CVR + male_count/female_count
inputs for perfect training/prediction alignment.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Dict, Any
import time
import logging

from .schemas.optimal_requests import (
    OptimalPredictionRequest,
    OptimalPredictionResponse,
    OptimalPredictionResult,
    GenderBreakdown
)
from .schemas.gender_schemas import (
    GenderClassificationRequest, 
    GenderBatchRequest, 
    GenderClassificationResponse, 
    GenderBatchResponse
)
from ..data.gender_classifier import classify_employee_gender_detailed, batch_classify_employee_genders
from ..ml.predictor_enhanced import EnhancedPredictor
from ..database.users import authenticate_user, is_admin_user
from ..config.settings import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings and predictor
settings = get_settings()
enhanced_predictor = EnhancedPredictor()

app = FastAPI(
    title="Enhanced Predictive Gift Selection API",
    description="Enhanced API with optimal CVR + direct employee count structure for perfect training/prediction alignment",
    version="2.0.0"
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(api_key: str = Depends(api_key_header)) -> Dict:
    """Dependency to authenticate user via API key."""
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
    """Dependency to authenticate admin user via API key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: API key is missing",
            headers={"WWW-Authenticate": "Header"},
        )
    
    if not is_admin_user(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    
    user = authenticate_user(api_key)
    return user


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Enhanced Predictive Gift Selection API",
        "version": "2.0.0",
        "architecture": "Three-file company-level architecture",
        "optimal_features": [
            "Direct CVR + male_count/female_count inputs",
            "Company-level granularity",
            "Zero data leakage",
            "Perfect training/prediction alignment",
            "CatBoost Regressor with RMSE loss"
        ],
        "endpoints": {
            "prediction": "/predict/optimal",
            "gender_classification": "/classify/gender",
            "batch_gender_classification": "/classify/gender/batch",
            "model_info": "/model/info"
        }
    }


@app.get("/test")
async def test_endpoint(current_user: Dict = Depends(get_current_user)):
    """Test endpoint to verify authentication."""
    return {
        "message": f"Enhanced API is working! Hello {current_user.get('username')}!",
        "architecture": "Company-level optimal structure",
        "timestamp": time.time()
    }


@app.post("/predict/optimal", response_model=OptimalPredictionResponse)
async def predict_optimal(
    request_data: OptimalPredictionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Make optimal predictions using CVR + direct employee counts.
    
    This endpoint implements the optimal request structure:
    - Direct CVR for company identification
    - Direct male_count and female_count (no employee name processing needed)
    - Perfect alignment with training data structure
    - Company-level granularity for accurate modeling
    """
    start_time = time.time()
    
    logger.info(f"Optimal prediction request: CVR={request_data.cvr}, "
               f"Male={request_data.male_count}, Female={request_data.female_count}, "
               f"Presents={len(request_data.presents)}")
    
    try:
        # Validate inputs
        if request_data.male_count < 0 or request_data.female_count < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Employee counts cannot be negative"
            )
        
        if request_data.male_count + request_data.female_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Total employee count cannot be zero"
            )
        
        if not request_data.presents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Presents list cannot be empty"
            )
        
        # Make predictions using enhanced predictor
        prediction_result = enhanced_predictor.predict_optimal(request_data)
        
        # Convert to response format
        predictions = []
        for pred in prediction_result['predictions']:
            # Extract gender breakdown if available
            gender_breakdown = {}
            if 'gender_breakdown' in pred:
                gb = pred['gender_breakdown']
                for gender in ['male', 'female']:
                    if gender in gb:
                        gender_breakdown[gender] = GenderBreakdown(
                            predicted_rate=gb[gender].get('predicted_rate', 0.0),
                            expected_count=gb[gender].get('expected_count', 0.0),
                            exposure=gb[gender].get('exposure', 0)
                        )
            
            prediction = OptimalPredictionResult(
                product_id=pred['product_id'],
                expected_qty=pred['expected_qty'],
                confidence_score=pred.get('confidence_score', 0.0),
                gender_breakdown=gender_breakdown,
                total_exposure=pred.get('total_exposure', request_data.male_count + request_data.female_count)
            )
            predictions.append(prediction)
        
        # Create request summary as dict
        request_summary = {
            "cvr": request_data.cvr,
            "total_employees": request_data.male_count + request_data.female_count,
            "male_count": request_data.male_count,
            "female_count": request_data.female_count,
            "num_gifts": len(request_data.presents)
        }
        
        # Create model info as dict
        model_info_data = enhanced_predictor.get_model_info()
        model_info = {
            "model_type": model_info_data.get('model_type', 'CatBoost Regressor (Enhanced - RMSE)'),
            "target": model_info_data.get('target', 'selection_rate'),
            "architecture": model_info_data.get('architecture', 'Three-file company-level architecture'),
            "model_loaded": model_info_data.get('model_loaded', False)
        }
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = OptimalPredictionResponse(
            predictions=predictions,
            request_summary=request_summary,
            model_info=model_info,
            processing_time_ms=round(processing_time_ms, 2)
        )
        
        logger.info(f"Optimal prediction completed: {len(predictions)} predictions in {processing_time_ms:.2f}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimal prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction service failed: {str(e)}"
        )


@app.post("/classify/gender", response_model=GenderClassificationResponse)
async def classify_gender_endpoint(
    request_data: GenderClassificationRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Classify the gender of a single name using enhanced Danish gender classification.
    
    This endpoint is useful for preparing employee data in external applications
    before making optimal prediction requests.
    """
    start_time = time.time()
    
    try:
        # Classify the gender
        gender_result = classify_employee_gender_detailed(request_data.name)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Gender classified for name '{request_data.name}': {gender_result.gender} ({processing_time_ms:.2f}ms)")
        
        return GenderClassificationResponse(
            name=request_data.name,
            gender=gender_result.gender,
            confidence=gender_result.confidence,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Gender classification failed for name '{request_data.name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gender classification failed: {str(e)}"
        )


@app.post("/classify/gender/batch", response_model=GenderBatchResponse)
async def classify_gender_batch_endpoint(
    request_data: GenderBatchRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Classify the gender of multiple names in a single request.
    
    This endpoint is ideal for batch processing employee lists to get gender counts
    that can then be used in optimal prediction requests.
    """
    start_time = time.time()
    
    if not request_data.names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Names list cannot be empty"
        )
    
    if len(request_data.names) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 1000 names"
        )
    
    try:
        # Process all names
        gender_results = batch_classify_employee_genders(request_data.names)
        
        # Convert to response format
        results = []
        for name, gender_result in zip(request_data.names, gender_results):
            result = GenderClassificationResponse(
                name=name,
                gender=gender_result.gender,
                confidence=gender_result.confidence,
                processing_time_ms=0  # Will be set to total time
            )
            results.append(result)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update processing time for all results
        for result in results:
            result.processing_time_ms = processing_time_ms
        
        # Calculate gender counts for convenience
        male_count = sum(1 for r in results if r.gender == "male")
        female_count = sum(1 for r in results if r.gender == "female")
        
        logger.info(f"Batch gender classification completed: {len(results)} names processed "
                   f"({male_count} male, {female_count} female) in {processing_time_ms:.2f}ms")
        
        response = GenderBatchResponse(
            results=results,
            total_processed=len(results),
            processing_time_ms=processing_time_ms
        )
        
        # Add gender summary to logs for convenience
        logger.info(f"Gender summary: Male={male_count}, Female={female_count}")
        
        return response
        
    except Exception as e:
        logger.error(f"Batch gender classification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch gender classification failed: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info(current_user: Dict = Depends(get_current_user)):
    """
    Get information about the loaded enhanced model.
    
    Returns details about the model architecture, training configuration,
    and current status.
    """
    try:
        model_info = enhanced_predictor.get_model_info()
        
        return {
            "message": "Model information retrieved successfully",
            "model_info": model_info,
            "api_version": "2.0.0",
            "architecture_features": [
                "Company-level granularity (CVR-based)",
                "Direct employee counts (no name processing)",
                "Three-file optimal data structure",
                "Zero data leakage prevention",
                "CatBoost Regressor with RMSE loss",
                "Selection rate modeling with exposure scaling"
            ],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model information: {str(e)}"
        )


@app.post("/admin/reload-model", status_code=status.HTTP_200_OK)
async def reload_model(admin_user: Dict = Depends(get_admin_user)):
    """
    Force reload of the enhanced predictor model.
    
    This endpoint reinitializes the enhanced predictor instance,
    which will reload the model from disk.
    
    Requires admin authentication.
    """
    start_time = time.time()
    
    try:
        global enhanced_predictor
        
        # Create new predictor instance
        enhanced_predictor = EnhancedPredictor()
        
        # Get model info to verify reload
        model_info = enhanced_predictor.get_model_info()
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return {
            "message": "Enhanced predictor model reloaded successfully",
            "model_loaded": model_info.get('model_loaded', False),
            "model_type": model_info.get('model_type', 'Unknown'),
            "processing_time_ms": processing_time_ms,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)