"""
Gift Demand Predictor Service

Main ML service for predicting gift demand quantities using CatBoost model.
Handles feature engineering, shop context resolution, and prediction aggregation.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.feature_extraction import FeatureHasher
from typing import List, Dict, Optional
import logging
import os
import pickle

from .shop_features import ShopFeatureResolver
from ..api.schemas.responses import PredictionResult

logger = logging.getLogger(__name__)

class GiftDemandPredictor:
    """
    Main predictor service for gift demand using CatBoost model.
    
    Handles:
    - Model loading and caching
    - Feature engineering matching training pipeline  
    - Shop feature resolution with fallbacks
    - Prediction aggregation across employee-product combinations
    """
    
    def __init__(self, model_path: str, historical_data_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.shop_resolver = None
        self.hasher = FeatureHasher(n_features=10, input_type='string')
        
        self.numeric_medians: Dict[str, float] = {} # Initialize numeric_medians
        
        # Feature configuration matching training pipeline
        self.expected_columns = [
            'employee_shop', 'employee_branch', 'employee_gender',
            'product_main_category', 'product_sub_category', 'product_brand',
            'product_color', 'product_durability', 'product_target_gender',
            'product_utility_type', 'product_type',
            'shop_main_category_diversity_selected', 'shop_brand_diversity_selected',
            'shop_utility_type_diversity_selected', 'shop_sub_category_diversity_selected',
            'shop_most_frequent_main_category_selected', 'shop_most_frequent_brand_selected',
            'unique_product_combinations_in_shop',
            'is_shop_most_frequent_main_category', 'is_shop_most_frequent_brand',
            # New product relativity features
            'product_share_in_shop', 'brand_share_in_shop',
            'product_rank_in_shop', 'brand_rank_in_shop'
        ] + [f'interaction_hash_{i}' for i in range(10)]
        
        # Categorical features (must match training)
        self.categorical_features = [
            'employee_shop', 'employee_branch', 'employee_gender',
            'product_main_category', 'product_sub_category', 'product_brand',
            'product_color', 'product_durability', 'product_target_gender',
            'product_utility_type', 'product_type',
            'shop_most_frequent_main_category_selected', 'shop_most_frequent_brand_selected'
        ]
        
        # Initialize components
        self._load_model()
        self._initialize_shop_resolver(historical_data_path)
        
        logger.info("GiftDemandPredictor initialized successfully")
    
    def _load_model(self):
        """Load CatBoost model and associated metadata (including numeric_medians)"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            logger.info(f"Loading CatBoost model from {self.model_path}")
            self.model = CatBoostRegressor()
            self.model.load_model(self.model_path)
            logger.info("Model loaded successfully.")

            # Load metadata, specifically numeric_medians
            model_dir = os.path.dirname(self.model_path)
            metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
            
            if not os.path.exists(metadata_path):
                logger.warning(f"Model metadata file not found at {metadata_path}. Numeric medians will be empty.")
                self.numeric_medians = {}
            else:
                logger.info(f"Loading model metadata from {metadata_path}")
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.numeric_medians = metadata.get('numeric_feature_medians', {})
                if not self.numeric_medians:
                    logger.warning("Numeric medians not found or empty in metadata.")
                else:
                    logger.info(f"Numeric medians loaded successfully: {list(self.numeric_medians.keys())}")

        except FileNotFoundError as e:
            logger.error(f"Error loading model/metadata: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading model/metadata: {e}")
            # Fallback for numeric_medians if any other error occurs during loading
            if not hasattr(self, 'numeric_medians') or self.numeric_medians is None:
                 self.numeric_medians = {}
                 logger.warning("numeric_medians initialized to empty due to an error during loading.")
            raise
    
    def _initialize_shop_resolver(self, historical_data_path: Optional[str]):
        """Initialize shop feature resolver"""
        try:
            self.shop_resolver = ShopFeatureResolver(historical_data_path)
            logger.info("Shop feature resolver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize shop resolver: {e}")
            raise
    
    def predict(self, branch: str, presents: List[Dict], employees: List[Dict]) -> List[PredictionResult]:
        """
        Make predictions for gift demand.
        
        Args:
            branch: Industry/branch code (e.g., "621000")
            presents: List of present dictionaries with classification attributes
            employees: List of employee dictionaries with gender information
            
        Returns:
            List of PredictionResult objects with quantity predictions
        """
        try:
            logger.info(f"Making predictions for {len(presents)} presents and {len(employees)} employees")
            
            # Calculate employee demographics
            employee_stats = self._calculate_employee_stats(employees)
            logger.debug(f"Employee demographics: {employee_stats}")
            
            raw_predictions = []
            
            for present in presents:
                try:
                    # Step 1: Get raw, un-normalized prediction for each present
                    shop_and_product_features = self.shop_resolver.get_shop_features(None, branch, present)
                    
                    prediction = self._predict_for_present(
                        present, employee_stats, branch, shop_and_product_features, len(employees)
                    )
                    raw_predictions.append(prediction)
                    
                except Exception as e:
                    logger.error(f"Failed to predict for present {present.get('id', 'unknown')}: {e}")
                    raw_predictions.append(PredictionResult(
                        product_id=str(present.get('id', 'unknown')),
                        expected_qty=0,
                        confidence_score=0.0
                    ))

            # Step 2: Normalize predictions so their sum equals total_employees
            total_raw_demand = sum(p.expected_qty for p in raw_predictions)
            
            final_predictions = []
            if total_raw_demand > 0:
                for pred in raw_predictions:
                    normalized_qty = (pred.expected_qty / total_raw_demand) * len(employees)
                    final_predictions.append(PredictionResult(
                        product_id=pred.product_id,
                        expected_qty=int(round(normalized_qty)),
                        confidence_score=pred.confidence_score
                    ))
            else:
                # If all raw predictions are zero, distribute employees evenly
                logger.warning("All raw predictions were zero. Distributing demand evenly.")
                equal_share = len(employees) / len(presents) if len(presents) > 0 else 0
                for pred in raw_predictions:
                    final_predictions.append(PredictionResult(
                        product_id=pred.product_id,
                        expected_qty=int(round(equal_share)),
                        confidence_score=0.1 # Low confidence
                    ))

            # Ensure the sum exactly matches total_employees due to rounding
            total_predicted = sum(p.expected_qty for p in final_predictions)
            if total_predicted != len(employees) and len(final_predictions) > 0:
                diff = len(employees) - total_predicted
                final_predictions[0].expected_qty += diff

            logger.info(f"Prediction complete. Total normalized demand: {sum(p.expected_qty for p in final_predictions)} units")
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return zero predictions for all presents
            return [
                PredictionResult(
                    product_id=str(present.get('id', 'unknown')),
                    expected_qty=0,
                    confidence_score=0.0
                )
                for present in presents
            ]
    
    def _calculate_employee_stats(self, employees: List[Dict]) -> Dict[str, float]:
        """Calculate employee gender distribution"""
        gender_counts = {'male': 0, 'female': 0, 'unisex': 0}
        
        for emp in employees:
            gender = emp.get('gender', 'unisex').lower()
            if gender in gender_counts:
                gender_counts[gender] += 1
            else:
                gender_counts['unisex'] += 1
        
        total = len(employees)
        if total == 0:
            return {'male': 0.0, 'female': 0.0, 'unisex': 1.0}
        
        return {
            gender: count / total
            for gender, count in gender_counts.items()
        }
    
    def _predict_for_present(self, present: Dict, employee_stats: Dict[str, float],
                           branch: str, shop_features: Dict,
                           total_employees: int) -> PredictionResult:
        """Make prediction for a single present"""
        
        present_id = present.get('id', 'unknown')
        logger.debug(f"Predicting for present {present_id}")
        
        # Create feature vectors for each gender group with non-zero representation
        rows = []
        gender_ratios = []
        for gender, ratio in employee_stats.items():
            if ratio > 0:
                features = self._create_feature_vector(
                    present, gender, branch, shop_features
                )
                rows.append(features)
                gender_ratios.append(ratio)
        
        if not rows:
            logger.warning(f"No valid employee data for present {present_id}")
            return PredictionResult(
                product_id=str(present_id),
                expected_qty=0,
                confidence_score=0.0
            )
        
        # Create DataFrame and add interaction features
        feature_df = pd.DataFrame(rows)
        feature_df = self._add_interaction_features(feature_df)
        
        # Prepare features for model
        feature_df = self._prepare_features(feature_df)
        
        try:
            # Make predictions using CatBoost Pool
            predictions = self._make_catboost_prediction(feature_df)
            
            # Aggregate predictions with proper gender ratio weighting
            total_prediction = self._aggregate_predictions(
                predictions, gender_ratios, total_employees
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(predictions, len(rows))
            
            return PredictionResult(
                product_id=str(present_id),
                expected_qty=int(round(max(0, total_prediction))),
                confidence_score=round(confidence, 2)
            )
            
        except Exception as e:
            logger.error(f"Prediction error for present {present_id}: {e}")
            return PredictionResult(
                product_id=str(present_id),
                expected_qty=0,
                confidence_score=0.0
            )
    
    def _create_feature_vector(self, present: Dict, employee_gender: str,
                             branch: str, shop_features: Dict) -> Dict:
        """Create feature vector for one present-employee combination"""
        
        # Base features matching training data structure
        # Since we don't have a real shop_id during prediction, use branch code as placeholder
        features = {
            'employee_shop': branch,  # Use branch as shop placeholder
            'employee_branch': branch,
            'employee_gender': employee_gender,
            'product_main_category': present.get('item_main_category', 'NONE'),
            'product_sub_category': present.get('item_sub_category', 'NONE'),
            'product_brand': present.get('brand', 'NONE'),
            'product_color': present.get('color', 'NONE'),
            'product_durability': present.get('durability', 'NONE'),
            'product_target_gender': present.get('target_demographic', 'NONE'),
            'product_utility_type': present.get('utility_type', 'NONE'),
            'product_type': present.get('usage_type', 'NONE')
        }
        
        # Add shop-level features
        features.update(shop_features)
        
        # Add binary indicator features
        features['is_shop_most_frequent_main_category'] = int(
            present.get('item_main_category') == shop_features.get('shop_most_frequent_main_category_selected')
        )
        features['is_shop_most_frequent_brand'] = int(
            present.get('brand') == shop_features.get('shop_most_frequent_brand_selected')
        )
        
        return features
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hashed interaction features"""
        # Create interaction strings
        interactions = df.apply(
            lambda x: f"{x['employee_branch']}_{x['product_main_category']}", 
            axis=1
        )
        
        # Hash interactions
        hashed_features = self.hasher.transform(
            interactions.apply(lambda x: [x])
        ).toarray()
        
        # Add as columns
        for i in range(hashed_features.shape[1]):
            df[f'interaction_hash_{i}'] = hashed_features[:, i]
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure features match model expectations"""
        
        # Convert categorical columns to string
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("NONE")
        
        # Ensure numeric columns
        numeric_cols = [
            'shop_main_category_diversity_selected', 'shop_brand_diversity_selected',
            'shop_utility_type_diversity_selected', 'shop_sub_category_diversity_selected',
            'unique_product_combinations_in_shop',
            'is_shop_most_frequent_main_category', 'is_shop_most_frequent_brand',
            # Add new relativity features to numeric handling
            'product_share_in_shop', 'brand_share_in_shop',
            'product_rank_in_shop', 'brand_rank_in_shop'
        ] + [f'interaction_hash_{i}' for i in range(10)]
        
        for col in numeric_cols:
            if col in df.columns:
                # Fill NaNs using loaded medians, fallback to 0 if median not available for a specific column
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(self.numeric_medians.get(col, 0))
            # Ensure the column exists even if not in input, fill with median or 0
            elif col in self.numeric_medians:
                 df[col] = self.numeric_medians[col]
            else: # Fallback if column is expected but not in input and no median
                 df[col] = 0.0


        # Add missing columns with defaults
        for col in self.expected_columns:
            if col not in df.columns: # This handles columns that might not have been in numeric_cols or categorical_features explicitly
                if col.startswith('interaction_hash_'):
                    df[col] = self.numeric_medians.get(col, 0.0) # Use median if available for hashes too
                elif col in self.categorical_features:
                    df[col] = "NONE"
                elif col in self.numeric_medians: # Other numeric columns expected
                    df[col] = self.numeric_medians[col]
                else: # Absolute fallback for unexpected missing columns
                    df[col] = 0.0
                    logger.warning(f"Column '{col}' was missing from input and no median found. Defaulted to 0.0.")
        
        # Reorder columns to match training
        df = df[self.expected_columns]
        
        return df
    
    def _make_catboost_prediction(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Make prediction using CatBoost with proper categorical handling"""
        
        # Get categorical feature indices
        cat_feature_indices = [
            feature_df.columns.get_loc(col) for col in self.categorical_features 
            if col in feature_df.columns
        ]
        
        # Create CatBoost Pool
        test_pool = Pool(
            data=feature_df,
            cat_features=cat_feature_indices
        )
        
        # Make predictions
        predictions = self.model.predict(test_pool)
        predictions = np.maximum(0, predictions)  # Ensure non-negative
        
        return predictions
    
    def _aggregate_predictions(self, predictions: np.ndarray, gender_ratios: List[float],
                             total_employees: int) -> float:
        """
        Aggregate model predictions to total quantity using proper gender weighting.
        
        IMPORTANT: The model was trained on historical selection_count which represents
        cumulative counts across all historical data for a shop/product/gender combination.
        
        For each gender group, we weight the prediction by:
        prediction_for_gender * gender_ratio * total_employees
        
        This gives us the expected quantity for each gender subgroup, which we sum
        to get the total expected demand.
        """
        
        if len(predictions) != len(gender_ratios):
            logger.error(f"Mismatch: {len(predictions)} predictions vs {len(gender_ratios)} ratios")
            return 0.0
        
        # Calculate weighted predictions for each gender group
        weighted_predictions = []
        for pred, ratio in zip(predictions, gender_ratios):
            # Each prediction represents expected rate for that gender
            # Scale by the number of employees in that gender group
            gender_employees = ratio * total_employees
            weighted_pred = pred * gender_employees
            weighted_predictions.append(weighted_pred)
        
        # Sum all gender-specific predictions
        total_prediction = np.sum(weighted_predictions)
        
        # Apply a scaling factor to account for the difference between
        # historical cumulative counts and single-order predictions
        # This converts from historical aggregated counts to per-order rates
        scaling_factor = 0.25  # Adjusted: converts cumulative counts to selection rates
        
        scaled_prediction = total_prediction * scaling_factor
        
        # Ensure non-negative and reasonable bounds
        # Cap at total employees as upper bound (100% selection rate)
        final_prediction = max(0, min(scaled_prediction, total_employees))
        
        # HEROKU DEBUG: Enhanced logging to see exact calculation values
        logger.info(f"HEROKU DEBUG - Aggregation: raw_predictions={predictions.tolist()}, "
                   f"gender_ratios={gender_ratios}, weighted_predictions={weighted_predictions}, "
                   f"total_sum={total_prediction:.4f}, scaled={scaled_prediction:.4f}, "
                   f"final={final_prediction:.4f}, total_employees={total_employees}")
        
        logger.debug(f"Aggregation: weighted_sum={total_prediction:.2f}, "
                    f"scaled={scaled_prediction:.2f}, final={final_prediction:.2f}")
        
        return final_prediction
    
    def _calculate_confidence(self, predictions: np.ndarray, num_groups: int) -> float:
        """Calculate prediction confidence score"""
        
        if len(predictions) == 0:
            return 0.0
        
        if num_groups == 1:
            # Single prediction - base confidence
            return 0.8
        
        # Multi-group prediction - confidence based on consistency
        prediction_mean = np.mean(predictions)
        prediction_std = np.std(predictions)
        
        # Higher consistency = higher confidence
        if prediction_mean > 0:
            consistency_factor = 1 - (prediction_std / prediction_mean)
            consistency_factor = np.clip(consistency_factor, 0, 1)
        else:
            consistency_factor = 0.5
        
        # Base confidence (0.7) + consistency bonus (up to 0.25)
        confidence = 0.7 + (0.25 * consistency_factor)
        
        return min(0.95, confidence)

# Force fresh instances to avoid caching issues during development
_predictor_instance = None

def get_predictor(model_path: str = "models/catboost_poisson_model/catboost_poisson_model.cbm",
                 historical_data_path: Optional[str] = "src/data/historical/present.selection.historic.csv") -> GiftDemandPredictor:
    """
    Get predictor instance. Creates fresh instance to ensure latest code changes are applied.
    
    Args:
        model_path: Path to trained CatBoost model
        historical_data_path: Path to historical selection data
        
    Returns:
        GiftDemandPredictor instance
    """
    global _predictor_instance
    
    # Always create fresh instance to avoid caching issues
    logger.info("Creating fresh predictor instance to ensure latest code changes")
    _predictor_instance = GiftDemandPredictor(model_path, historical_data_path)
    
    return _predictor_instance