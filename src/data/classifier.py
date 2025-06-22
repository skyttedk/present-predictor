"""
Main data classification module for the Predictive Gift Selection System.
Combines OpenAI gift classification and enhanced gender classification.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass

from .openai_client import OpenAIAssistantClient, classify_product_description, OpenAIClassificationError
from .gender_classifier import (
    get_gender_classifier, 
    classify_employee_gender_detailed, 
    batch_classify_employee_genders,
    GenderClassificationResult
)
from .schemas.data_models import (
    GiftAttributes, 
    ClassifiedGift, 
    ProcessedEmployee, 
    HistoricalRecord,
    TargetDemographic
)
from ..api.schemas.requests import PredictionRequest, GiftItem, Employee

logger = logging.getLogger(__name__)


@dataclass
class ClassificationStats:
    """Statistics for classification operations."""
    total_gifts: int = 0
    successful_gifts: int = 0
    failed_gifts: int = 0
    total_employees: int = 0
    high_confidence_employees: int = 0
    low_confidence_employees: int = 0
    processing_time_ms: float = 0.0


class DataClassifier:
    """
    Main data classifier that handles both gift and employee classification.
    Implements the three-step processing pipeline.
    """
    
    def __init__(self, openai_client: Optional[OpenAIAssistantClient] = None):
        """Initialize the data classifier."""
        self.openai_client = openai_client
        self.gender_classifier = get_gender_classifier()
        self.stats = ClassificationStats()
    
    async def classify_request(self, request: PredictionRequest) -> Tuple[List[ClassifiedGift], List[ProcessedEmployee], ClassificationStats]:
        """
        Classify a complete prediction request.
        
        Args:
            request: The prediction request to classify
            
        Returns:
            Tuple of (classified_gifts, processed_employees, classification_stats)
        """
        import time
        start_time = time.time()
        
        try:
            # Classify gifts and employees in parallel
            gifts_task = self._classify_gifts(request.gifts)
            employees_task = self._classify_employees(request.employees)
            
            classified_gifts, processed_employees = await asyncio.gather(
                gifts_task, employees_task
            )
            
            # Update statistics
            self.stats.processing_time_ms = (time.time() - start_time) * 1000
            
            return classified_gifts, processed_employees, self.stats
            
        except Exception as e:
            logger.error(f"Failed to classify request: {e}")
            raise
    
    async def _classify_gifts(self, gifts: List[GiftItem]) -> List[ClassifiedGift]:
        """Classify all gifts in the request."""
        self.stats.total_gifts = len(gifts)
        classified_gifts = []
        
        # Use OpenAI client or create one if not provided
        if self.openai_client:
            client = self.openai_client
            should_close = False
        else:
            from .openai_client import create_openai_client
            client = create_openai_client()
            should_close = True
        
        try:
            # Classify each gift
            for gift in gifts:
                try:
                    # Classify using OpenAI Assistant
                    attributes, _, _ = await client.classify_product(gift.description)
                    
                    classified_gift = ClassifiedGift(
                        product_id=gift.product_id,
                        description=gift.description,
                        attributes=attributes,
                        confidence_score=0.85  # Default confidence for successful classification
                    )
                    classified_gifts.append(classified_gift)
                    self.stats.successful_gifts += 1
                    
                except OpenAIClassificationError as e:
                    logger.warning(f"Failed to classify gift '{gift.product_id}': {e}")
                    # Create fallback classification
                    fallback_attributes = self._create_fallback_gift_attributes(gift.description)
                    classified_gift = ClassifiedGift(
                        product_id=gift.product_id,
                        description=gift.description,
                        attributes=fallback_attributes,
                        confidence_score=0.3  # Low confidence for fallback
                    )
                    classified_gifts.append(classified_gift)
                    self.stats.failed_gifts += 1
                
                except Exception as e:
                    logger.error(f"Unexpected error classifying gift '{gift.product_id}': {e}")
                    # Create fallback classification
                    fallback_attributes = self._create_fallback_gift_attributes(gift.description)
                    classified_gift = ClassifiedGift(
                        product_id=gift.product_id,
                        description=gift.description,
                        attributes=fallback_attributes,
                        confidence_score=0.1  # Very low confidence for error fallback
                    )
                    classified_gifts.append(classified_gift)
                    self.stats.failed_gifts += 1
        
        finally:
            if should_close:
                await client.close()
        
        return classified_gifts
    
    async def _classify_employees(self, employees: List[Employee]) -> List[ProcessedEmployee]:
        """Classify all employees in the request."""
        self.stats.total_employees = len(employees)
        
        # Extract names
        names = [emp.name for emp in employees]
        
        # Classify in batch
        gender_results = batch_classify_employee_genders(names)
        
        # Convert to ProcessedEmployee objects
        processed_employees = []
        for employee, gender_result in zip(employees, gender_results):
            processed_employee = ProcessedEmployee(
                name=employee.name,
                gender=gender_result.gender
            )
            processed_employees.append(processed_employee)
            
            # Update stats
            if gender_result.confidence == "high":
                self.stats.high_confidence_employees += 1
            else:
                self.stats.low_confidence_employees += 1
        
        return processed_employees
    
    def _create_fallback_gift_attributes(self, description: str) -> GiftAttributes:
        """
        Create fallback gift attributes when classification fails.
        Uses simple heuristics based on the description.
        """
        desc_lower = description.lower()
        
        # Simple keyword-based classification
        if any(word in desc_lower for word in ['mug', 'cup', 'kitchen', 'cook']):
            main_category = "Home & Kitchen"
            sub_category = "Cookware"
            utility_type = "practical"
        elif any(word in desc_lower for word in ['bag', 'purse', 'wallet']):
            main_category = "Bags"
            sub_category = "General"
            utility_type = "practical"
        elif any(word in desc_lower for word in ['decoration', 'decor', 'ornament']):
            main_category = "Home & Decor"
            sub_category = "Decoration"
            utility_type = "aesthetic"
        else:
            main_category = "General"
            sub_category = "Unknown"
            utility_type = "practical"
        
        # Extract color if mentioned
        color = "NONE"
        color_words = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'orange']
        for color_word in color_words:
            if color_word in desc_lower:
                color = color_word.capitalize()
                break
        
        return GiftAttributes(
            itemMainCategory=main_category,
            itemSubCategory=sub_category,
            color=color,
            brand="NONE",
            vendor="NONE",
            valuePrice=0.0,
            targetDemographic=TargetDemographic.UNISEX,
            utilityType=utility_type,
            durability="durable",
            usageType="individual"
        )
    
    def convert_to_historical_format(
        self, 
        classified_gifts: List[ClassifiedGift], 
        processed_employees: List[ProcessedEmployee], 
        branch_no: str
    ) -> List[HistoricalRecord]:
        """
        Convert classified data to historical record format for ML processing.
        
        Args:
            classified_gifts: Classified gift data
            processed_employees: Processed employee data
            branch_no: Branch number from request
            
        Returns:
            List of HistoricalRecord objects ready for ML processing
        """
        historical_records = []
        
        # Extract shop and branch from branch_no
        # For now, use the same value for both (can be enhanced later)
        employee_shop = "2960"  # Default shop value
        employee_branch = branch_no
        
        # Create records for each gift-employee combination
        for gift in classified_gifts:
            for employee in processed_employees:
                record = gift.to_historical_format(
                    employee_shop=employee_shop,
                    employee_branch=employee_branch,
                    employee_gender=employee.gender
                )
                historical_records.append(record)
        
        return historical_records


async def classify_prediction_request(request: PredictionRequest) -> Tuple[List[ClassifiedGift], List[ProcessedEmployee], ClassificationStats]:
    """
    Convenience function to classify a prediction request.
    
    Args:
        request: The prediction request to classify
        
    Returns:
        Tuple of (classified_gifts, processed_employees, classification_stats)
    """
    classifier = DataClassifier()
    return await classifier.classify_request(request)


def validate_classification_completeness(
    classified_gifts: List[ClassifiedGift], 
    processed_employees: List[ProcessedEmployee],
    original_request: PredictionRequest
) -> Dict[str, Any]:
    """
    Validate that classification is complete and consistent.
    
    Args:
        classified_gifts: Classified gifts
        processed_employees: Processed employees
        original_request: Original request for comparison
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "gift_count_match": len(classified_gifts) == len(original_request.gifts),
        "employee_count_match": len(processed_employees) == len(original_request.employees),
        "missing_gift_ids": [],
        "low_confidence_gifts": [],
        "low_confidence_employees": []
    }
    
    # Check gift count
    if not validation_results["gift_count_match"]:
        validation_results["errors"].append(
            f"Gift count mismatch: {len(classified_gifts)} classified vs {len(original_request.gifts)} requested"
        )
        validation_results["valid"] = False
    
    # Check employee count
    if not validation_results["employee_count_match"]:
        validation_results["errors"].append(
            f"Employee count mismatch: {len(processed_employees)} processed vs {len(original_request.employees)} requested"
        )
        validation_results["valid"] = False
    
    # Check for missing gift IDs
    original_gift_ids = {gift.product_id for gift in original_request.gifts}
    classified_gift_ids = {gift.product_id for gift in classified_gifts}
    missing_ids = original_gift_ids - classified_gift_ids
    if missing_ids:
        validation_results["missing_gift_ids"] = list(missing_ids)
        validation_results["errors"].append(f"Missing gift IDs: {missing_ids}")
        validation_results["valid"] = False
    
    # Check for low confidence classifications
    for gift in classified_gifts:
        if gift.confidence_score and gift.confidence_score < 0.5:
            validation_results["low_confidence_gifts"].append({
                "product_id": gift.product_id,
                "confidence": gift.confidence_score
            })
    
    if validation_results["low_confidence_gifts"]:
        validation_results["warnings"].append(
            f"Low confidence gift classifications: {len(validation_results['low_confidence_gifts'])}"
        )
    
    return validation_results