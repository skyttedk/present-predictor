"""
Enhanced gender classification for employee names.
Based on the user's Flask implementation with Danish name support.
"""

import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

try:
    import gender_guesser.detector as gender
    GENDER_GUESSER_AVAILABLE = True
except ImportError:
    GENDER_GUESSER_AVAILABLE = False
    gender = None

from .schemas.data_models import TargetDemographic

logger = logging.getLogger(__name__)


@dataclass
class GenderClassificationResult:
    """Result of gender classification."""
    name: str
    normalized_name: str
    gender: TargetDemographic
    confidence: str  # 'high', 'medium', 'low'
    method: str  # 'danish_dict', 'gender_guesser', 'fallback'


class GenderClassificationError(Exception):
    """Custom exception for gender classification errors."""
    pass


class EnhancedGenderClassifier:
    """
    Enhanced gender classifier with Danish name support.
    Based on the user's Flask implementation.
    """
    
    # Enhanced Danish names dictionary for better coverage
    DANISH_NAMES = {
        # Female names
        'ea': 'female', 'my': 'female', 'freja': 'female', 'saga': 'female',
        'alba': 'female', 'liv': 'female', 'naja': 'female', 'maja': 'female',
        'clara': 'female', 'frida': 'female', 'vigga': 'female', 'luna': 'female',
        'nova': 'female', 'aya': 'female', 'nora': 'female', 'alma': 'female',
        
        # Male names
        'bo': 'male', 'aksel': 'male', 'villads': 'male', 'malthe': 'male',
        'viggo': 'male', 'theodor': 'male', 'bertram': 'male', 'storm': 'male',
        'august': 'male', 'magnus': 'male', 'felix': 'male', 'noah': 'male',
        'lucas': 'male', 'oscar': 'male', 'victor': 'male', 'emil': 'male'
    }
    
    def __init__(self):
        """Initialize the gender classifier."""
        if not GENDER_GUESSER_AVAILABLE:
            logger.warning("gender-guesser package not available. Install with: pip install gender-guesser")
            self.detector = None
        else:
            self.detector = gender.Detector()
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize name to proper case (first letter uppercase, rest lowercase).
        Handles compound names with hyphens and spaces.
        """
        if not name:
            return name
        
        # Remove extra whitespace and convert to proper case
        name = name.strip()
        if not name:
            return name
        
        # Handle hyphenated names like "Anne-Marie"
        if '-' in name:
            parts = name.split('-')
            return '-'.join([part.capitalize() for part in parts if part])
        
        # Handle compound names with spaces like "Anna Lisa"
        if ' ' in name:
            parts = name.split()
            return ' '.join([part.capitalize() for part in parts if part])
        
        return name.capitalize()

    def extract_first_name(self, full_name: str) -> str:
        """
        Extract just the first name from a full name string.
        This is critical because gender-guesser works best with first names only.
        
        Examples:
        - "GUNHILD SØRENSEN" -> "Gunhild"
        - "Per Christian Eidevik" -> "Per"
        - "Anne-Marie Hansen" -> "Anne-Marie"
        """
        if not full_name:
            return full_name
        
        # Normalize first
        normalized = self.normalize_name(full_name)
        
        # Split on spaces and take the first part
        # This handles "First Last" -> "First"
        parts = normalized.split()
        if not parts:
            return normalized
        
        first_part = parts[0]
        
        # Handle hyphenated first names like "Anne-Marie Hansen"
        # If the first part contains a hyphen, it's likely a compound first name
        if '-' in first_part:
            return first_part  # Keep the whole hyphenated name
        
        return first_part
    
    def classify_gender(self, name: str) -> GenderClassificationResult:
        """
        Enhanced gender detection with Danish name support and case handling.
        CRITICAL: Extracts first name only as gender-guesser works best with first names.
        
        Args:
            name: The full name to classify (e.g., "GUNHILD SØRENSEN")
            
        Returns:
            GenderClassificationResult: Classification result with confidence and method
        """
        if not name:
            return GenderClassificationResult(
                name=name,
                normalized_name="",
                gender=TargetDemographic.UNISEX,
                confidence="low",
                method="fallback"
            )
        
        # CRITICAL: Extract first name only for classification
        first_name = self.extract_first_name(name)
        normalized_name = self.normalize_name(first_name)
        original_lower = first_name.lower().strip()
        
        # Method 1: Try the standard detector first with first name only
        if self.detector:
            try:
                result = self.detector.get_gender(normalized_name)
                if result in ['male', 'female']:
                    return GenderClassificationResult(
                        name=name,  # Keep original full name
                        normalized_name=normalized_name,  # Show extracted first name
                        gender=TargetDemographic(result),
                        confidence="high",
                        method="gender_guesser"
                    )
            except Exception as e:
                logger.debug(f"Gender guesser failed for '{normalized_name}': {e}")
        
        # Method 2: Check our Danish names dictionary (with first name only)
        if original_lower in self.DANISH_NAMES:
            gender_result = self.DANISH_NAMES[original_lower]
            return GenderClassificationResult(
                name=name,  # Keep original full name
                normalized_name=normalized_name,  # Show extracted first name
                gender=TargetDemographic(gender_result),
                confidence="high",
                method="danish_dict"
            )
        
        # Method 3: Try with Denmark country code specifically
        if self.detector:
            try:
                result = self.detector.get_gender(normalized_name, 'denmark')
                if result in ['male', 'female']:
                    return GenderClassificationResult(
                        name=name,  # Keep original full name
                        normalized_name=normalized_name,  # Show extracted first name
                        gender=TargetDemographic(result),
                        confidence="medium",
                        method="gender_guesser"
                    )
            except Exception as e:
                logger.debug(f"Gender guesser with Denmark code failed for '{normalized_name}': {e}")
        
        # Method 4: Final fallback with uncertain results
        if self.detector:
            try:
                result = self.detector.get_gender(normalized_name)
                if result in ['mostly_male', 'andy']:  # Andy means androgynous but lean male
                    return GenderClassificationResult(
                        name=name,  # Keep original full name
                        normalized_name=normalized_name,  # Show extracted first name
                        gender=TargetDemographic.MALE,
                        confidence="low",
                        method="gender_guesser"
                    )
                elif result in ['mostly_female']:
                    return GenderClassificationResult(
                        name=name,  # Keep original full name
                        normalized_name=normalized_name,  # Show extracted first name
                        gender=TargetDemographic.FEMALE,
                        confidence="low",
                        method="gender_guesser"
                    )
            except Exception as e:
                logger.debug(f"Final gender guesser attempt failed for '{normalized_name}': {e}")
        
        # Ultimate fallback
        return GenderClassificationResult(
            name=name,  # Keep original full name
            normalized_name=normalized_name,  # Show extracted first name
            gender=TargetDemographic.UNISEX,
            confidence="low",
            method="fallback"
        )
    
    def batch_classify(self, names: list[str]) -> list[GenderClassificationResult]:
        """
        Classify multiple names in batch.
        
        Args:
            names: List of names to classify
            
        Returns:
            List of GenderClassificationResult objects
        """
        results = []
        for name in names:
            try:
                result = self.classify_gender(name)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify name '{name}': {e}")
                # Add fallback result
                results.append(GenderClassificationResult(
                    name=name,
                    normalized_name=self.normalize_name(name) if name else "",
                    gender=TargetDemographic.UNISEX,
                    confidence="low",
                    method="error_fallback"
                ))
        return results
    
    def get_confidence_stats(self, results: list[GenderClassificationResult]) -> Dict[str, int]:
        """
        Get statistics on classification confidence.
        
        Args:
            results: List of classification results
            
        Returns:
            Dictionary with confidence statistics
        """
        stats = {'high': 0, 'medium': 0, 'low': 0}
        for result in results:
            if result.confidence in stats:
                stats[result.confidence] += 1
        return stats


# Global classifier instance
_classifier_instance: Optional[EnhancedGenderClassifier] = None


def get_gender_classifier() -> EnhancedGenderClassifier:
    """Get the global gender classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EnhancedGenderClassifier()
    return _classifier_instance


def classify_employee_gender(name: str) -> TargetDemographic:
    """
    Convenience function to classify a single employee name.
    
    Args:
        name: Employee name to classify
        
    Returns:
        TargetDemographic: Classified gender
    """
    classifier = get_gender_classifier()
    result = classifier.classify_gender(name)
    return result.gender


def classify_employee_gender_detailed(name: str) -> GenderClassificationResult:
    """
    Classify employee gender with detailed results.
    
    Args:
        name: Employee name to classify
        
    Returns:
        GenderClassificationResult: Detailed classification result
    """
    classifier = get_gender_classifier()
    return classifier.classify_gender(name)


def batch_classify_employee_genders(names: list[str]) -> list[GenderClassificationResult]:
    """
    Classify multiple employee names in batch.
    
    Args:
        names: List of employee names
        
    Returns:
        List of detailed classification results
    """
    classifier = get_gender_classifier()
    return classifier.batch_classify(names)