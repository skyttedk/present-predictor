"""
API data preprocessing utilities for the Predictive Gift Selection System.
Lightweight processing for the three-step API pipeline only.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .schemas.data_models import ClassifiedGift, ProcessedEmployee
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class APIDataProcessor:
    """
    Lightweight data processor for API requests only.
    Transforms classified API data into model-ready format.
    """
    
    def __init__(self, label_encoders: Optional[Dict[str, LabelEncoder]] = None):
        """
        Initialize API data processor.
        
        Args:
            label_encoders: Pre-trained label encoders from notebook training
        """
        self.settings = get_settings()
        self.label_encoders = label_encoders or {}
        
    def prepare_api_features(
        self, 
        gifts: List[ClassifiedGift], 
        employees: List[ProcessedEmployee],
        branch_no: str
    ) -> pd.DataFrame:
        """
        Convert API request data to model input format.
        
        Args:
            gifts: List of classified gifts from Step 2
            employees: List of processed employees from Step 2  
            branch_no: Branch number from API request
            
        Returns:
            DataFrame ready for model prediction
        """
        # Create feature combinations for each gift-employee pair
        features_list = []
        
        for gift in gifts:
            for employee in employees:
                features = {
                    'employee_shop': branch_no,
                    'employee_branch': branch_no,
                    'employee_gender': employee.gender,
                    'product_main_category': gift.item_main_category,
                    'product_sub_category': gift.item_sub_category,
                    'product_brand': gift.brand,
                    'product_color': gift.color,
                    'product_durability': gift.durability,
                    'product_target_gender': gift.target_demographics,
                    'product_utility_type': gift.utility_type,
                    'product_type': gift.usage_type
                }
                features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Apply label encoding using pre-trained encoders
        encoded_df = self._apply_label_encoding(features_df)
        
        logger.info(f"Prepared {len(encoded_df)} feature combinations for prediction")
        return encoded_df
    
    def _apply_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pre-trained label encoders to categorical features.
        
        Args:
            df: Features DataFrame
            
        Returns:
            Encoded features DataFrame
        """
        encoded_df = df.copy()
        
        for column in df.columns:
            if column in self.label_encoders:
                le = self.label_encoders[column]
                try:
                    # Handle unknown categories by using the first known category
                    encoded_values = []
                    for value in df[column]:
                        if value in le.classes_:
                            encoded_values.append(le.transform([value])[0])
                        else:
                            logger.warning(f"Unknown category '{value}' in {column}, using default")
                            encoded_values.append(0)  # Use first category as default
                    
                    encoded_df[column] = encoded_values
                    
                except Exception as e:
                    logger.error(f"Error encoding {column}: {e}")
                    # Fallback: use simple numeric encoding
                    unique_values = df[column].unique()
                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                    encoded_df[column] = df[column].map(value_map)
            else:
                logger.warning(f"No label encoder found for {column}, using simple encoding")
                unique_values = df[column].unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                encoded_df[column] = df[column].map(value_map)
        
        return encoded_df
    
    def aggregate_predictions(
        self, 
        predictions: List[float], 
        gifts: List[ClassifiedGift],
        employees: List[ProcessedEmployee]
    ) -> Dict[str, float]:
        """
        Aggregate predictions per gift across all employees.
        
        Args:
            predictions: Model predictions for each gift-employee combination
            gifts: Original gift list
            employees: Original employee list
            
        Returns:
            Dictionary mapping product_id to expected quantity
        """
        # Reshape predictions into gift x employee matrix
        n_employees = len(employees)
        gift_predictions = {}
        
        for i, gift in enumerate(gifts):
            # Sum predictions across all employees for this gift
            start_idx = i * n_employees
            end_idx = start_idx + n_employees
            gift_total = sum(predictions[start_idx:end_idx])
            gift_predictions[gift.product_id] = max(1, round(gift_total))  # Minimum 1
        
        logger.info(f"Aggregated predictions for {len(gift_predictions)} gifts")
        return gift_predictions


def load_label_encoders(encoders_path: str = "models/label_encoders.pkl") -> Dict[str, LabelEncoder]:
    """
    Load pre-trained label encoders from notebook training.
    
    Args:
        encoders_path: Path to saved encoders
        
    Returns:
        Dictionary of label encoders
    """
    import pickle
    from pathlib import Path
    
    if not Path(encoders_path).exists():
        logger.warning(f"Label encoders not found at {encoders_path}")
        return {}
    
    try:
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        logger.info(f"Loaded {len(encoders)} label encoders")
        return encoders
    except Exception as e:
        logger.error(f"Error loading label encoders: {e}")
        return {}


# Convenience function for API endpoints
def process_api_request(
    gifts: List[ClassifiedGift],
    employees: List[ProcessedEmployee], 
    branch_no: str,
    encoders_path: str = "models/label_encoders.pkl"
) -> pd.DataFrame:
    """
    One-step processing for API requests.
    
    Args:
        gifts: Classified gifts from Step 2
        employees: Processed employees from Step 2
        branch_no: Branch number
        encoders_path: Path to trained encoders
        
    Returns:
        Model-ready features DataFrame
    """
    # Load encoders trained from notebook
    label_encoders = load_label_encoders(encoders_path)
    
    # Process data
    processor = APIDataProcessor(label_encoders)
    features = processor.prepare_api_features(gifts, employees, branch_no)
    
    return features