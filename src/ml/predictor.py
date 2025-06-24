"""
Gift Demand Predictor Service

Main ML service for predicting gift demand selection rates using a CatBoost model.
Handles feature engineering, shop context resolution, and scales predicted rates to expected quantities.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.feature_extraction import FeatureHasher
from typing import List, Dict # Optional is not needed for Python 3.10+ with `| None`
import logging
import os
import pickle

from .shop_features import ShopFeatureResolver
# PredictionResult is no longer directly used for internal return types
# from ..api.schemas.responses import PredictionResult

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
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.shop_resolver = None
        self.hasher = FeatureHasher(n_features=10, input_type='string')
        self.model_rmse: float | None = None # To store model RMSE, using Python 3.10+ union type
        
        self.numeric_medians: Dict[str, float] = {} # Initialize numeric_medians
        self.expected_columns: List[str] = [] # To be loaded from metadata
        self.categorical_features: List[str] = [] # To be loaded from metadata
        
        # Initialize components
        self._load_model() # This also loads numeric_medians from metadata
        self._initialize_shop_resolver() # Uses model_dir derived from self.model_path
        
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

                # Load model RMSE from metadata
                performance_metrics = metadata.get('performance_metrics', {})
                self.model_rmse = performance_metrics.get('rmse_validation')
                if self.model_rmse is not None:
                    logger.info(f"Model RMSE loaded successfully: {self.model_rmse:.4f}")
                else:
                    logger.warning("Model RMSE (rmse_validation) not found in metadata. Confidence score might be affected.")
                    self.model_rmse = 0.3 # Default RMSE if not found
                
                # Load expected_columns and categorical_features from metadata
                self.expected_columns = metadata.get('features_used', [])
                self.categorical_features = metadata.get('categorical_features_in_model', [])

                if not self.expected_columns:
                    logger.error("CRITICAL: 'features_used' not found in model metadata. Predictor may not function correctly.")
                    # Potentially raise an error here or fallback to a default, but error is safer.
                    # For now, it will proceed with an empty list, likely causing downstream errors.
                else:
                    logger.info(f"Expected columns loaded successfully ({len(self.expected_columns)} columns).")

                if not self.categorical_features:
                    logger.warning("'categorical_features_in_model' not found or empty in model metadata.")
                    # This might be acceptable if the model has no categorical features,
                    # or it might indicate an issue with metadata.
                else:
                    logger.info(f"Categorical features loaded successfully ({len(self.categorical_features)} features).")

        except FileNotFoundError as e:
            logger.error(f"Error loading model/metadata: {e}")
            self.model_rmse = 0.3 # Default RMSE on file not found
            # If metadata file is not found, expected_columns and categorical_features will remain empty.
            # This will likely cause errors downstream, which is a desired failure mode.
            logger.error("CRITICAL: Model metadata file not found. Expected columns and categorical features could not be loaded.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading model/metadata: {e}")
            # Fallback for numeric_medians if any other error occurs during loading
            if not hasattr(self, 'numeric_medians') or self.numeric_medians is None:
                 self.numeric_medians = {}
                 logger.warning("numeric_medians initialized to empty due to an error during loading.")
            if not hasattr(self, 'model_rmse') or self.model_rmse is None:
                 self.model_rmse = 0.3 # Default RMSE on other errors
                 logger.warning(f"model_rmse initialized to default {self.model_rmse} due to an error during loading.")
            # Ensure expected_columns and categorical_features are at least empty lists on error
            if not hasattr(self, 'expected_columns') or self.expected_columns is None:
                self.expected_columns = []
                logger.error("CRITICAL: Expected columns list is empty due to an error during model/metadata loading.")
            if not hasattr(self, 'categorical_features') or self.categorical_features is None:
                self.categorical_features = []
                logger.warning("Categorical features list is empty due to an error during model/metadata loading.")
            raise
    
    def _initialize_shop_resolver(self):
        """Initialize shop feature resolver using aggregates from the model directory."""
        try:
            model_dir = os.path.dirname(self.model_path)
            if not model_dir or not os.path.isdir(model_dir): # Ensure model_dir is valid
                logger.error(f"Invalid model directory derived from model_path: '{model_dir}'. Cannot initialize ShopFeatureResolver.")
                # Set a non-functional resolver or raise an error
                # For now, let's allow it to proceed but it will use hardcoded defaults.
                self.shop_resolver = ShopFeatureResolver(model_dir) # It will log warnings
            else:
                self.shop_resolver = ShopFeatureResolver(model_dir)
            logger.info("Shop feature resolver initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize shop resolver: {e}")
            # Consider how to handle this: raise, or allow fallback to default resolver state
            # For now, let it proceed; the resolver itself has fallbacks.
            # If self.shop_resolver is not set, calls to it will fail later.
            # Ensure it's at least instantiated with some default state if critical.
            if not hasattr(self, 'shop_resolver') or self.shop_resolver is None:
                 model_dir_fallback = os.path.dirname(self.model_path) if self.model_path else "."
                 self.shop_resolver = ShopFeatureResolver(model_dir_fallback) # Attempt with potentially invalid path, resolver handles it
                 logger.warning("ShopFeatureResolver initialized with potentially invalid path due to earlier error.")
            # raise # Optionally re-raise if resolver is critical for any operation
    
    def predict(self, branch: str, presents: List[Dict], employees: List[Dict]) -> List[Dict]:
        """
        Make predictions for gift demand.
        
        Args:
            branch: Industry/branch code (e.g., "621000")
            presents: List of present dictionaries with classification attributes
            employees: List of employee dictionaries with gender information
            
        Returns:
            List of prediction dictionaries with quantity predictions
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
                        present, employee_stats, branch, shop_and_product_features
                    )
                    raw_predictions.append(prediction)
                    
                except Exception as e:
                    logger.error(f"Failed to predict for present {present.get('id', 'unknown')}: {e}")
                    raw_predictions.append({
                        "product_id": str(present.get('id', 'unknown')),
                        "expected_qty": 0,
                        "confidence_score": 0.0
                    })

            # Return raw predictions without normalization
            logger.info(f"Prediction complete. Total raw demand: {sum(p['expected_qty'] for p in raw_predictions)} units")
            
            return raw_predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return zero predictions for all presents
            return [
                {
                    "product_id": str(present.get('id', 'unknown')),
                    "expected_qty": 0,
                    "confidence_score": 0.0
                }
                for present in presents
            ]
    
    def _calculate_employee_stats(self, employees: List[Dict]) -> Dict[str, int]:
        """Calculate employee gender counts."""
        gender_counts = {'male': 0, 'female': 0, 'unisex': 0}
        
        for emp in employees:
            gender = emp.get('gender', 'unisex').lower()
            if gender in gender_counts:
                gender_counts[gender] += 1
            else:
                gender_counts['unisex'] += 1
        
        return gender_counts
    
    def _predict_for_present(self, present: Dict, employee_gender_counts: Dict[str, int],
                           branch: str, shop_features: Dict) -> Dict:
        """Make prediction for a single present, returning a dictionary."""
        
        present_id = present.get('id', 'unknown')
        logger.debug(f"Predicting for present {present_id}")
        
        # Create feature vectors for each gender group with non-zero representation
        rows = []
        gender_counts_list = []
        for gender, count in employee_gender_counts.items():
            if count > 0:
                features = self._create_feature_vector(
                    present, gender, branch, shop_features
                )
                rows.append(features)
                gender_counts_list.append(count)
        
        if not rows:
            logger.warning(f"No valid employee data for present {present_id}")
            return {
                "product_id": str(present_id),
                "expected_qty": 0,
                "confidence_score": 0.0
            }
        
        # Create DataFrame and add interaction features
        feature_df = pd.DataFrame(rows)
        feature_df = self._add_interaction_features(feature_df)
        
        # Prepare features for model
        feature_df = self._prepare_features(feature_df)
        
        try:
            # Make predictions using CatBoost Pool
            predicted_rates = self._make_catboost_prediction(feature_df)
            
            # Aggregate predictions with proper gender ratio weighting
            total_prediction = self._aggregate_predictions(
                predicted_rates, gender_counts_list
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(predicted_rates, len(rows))
            
            return {
                "product_id": str(present_id),
                "expected_qty": total_prediction,
                "confidence_score": round(confidence, 2)
            }
            
        except Exception as e:
            logger.error(f"Prediction error for present {present_id}: {e}")
            return {
                "product_id": str(present_id),
                "expected_qty": 0,
                "confidence_score": 0.0
            }
    
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
        """Add hashed interaction features using sub-tokens."""
        if df.empty:
            # Add empty hash columns if df is empty to maintain schema
            for i in range(self.hasher.n_features):
                df[f'interaction_hash_{i}'] = 0.0
            return df

        # Create sub-tokens for better hashing
        # Ensure columns exist, use 'NONE' as fallback
        interaction_tokens = df.apply(
            lambda x: [
                f"branch_{x.get('employee_branch', 'NONE')}",
                f"cat_{x.get('product_main_category', 'NONE')}"
            ],
            axis=1
        )
        
        # Hash interactions
        hashed_features = self.hasher.transform(interaction_tokens).toarray()
        
        # Add as columns
        for i in range(hashed_features.shape[1]):
            df[f'interaction_hash_{i}'] = hashed_features[:, i]
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure features match model expectations"""
        
        initial_columns = set(df.columns) # Store initial columns for logging later
        
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
            'product_share_in_shop', 'brand_share_in_shop',
            'product_rank_in_shop', 'brand_rank_in_shop'
        ] + [f'interaction_hash_{i}' for i in range(10)]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(self.numeric_medians.get(col, 0.0 if not col.endswith('_rank_in_shop') else 99.0))
            elif col in self.numeric_medians: # Ensure the column exists even if not in input, fill with median
                 df[col] = self.numeric_medians[col]
            else: # Fallback if column is expected but not in input and no median
                 default_val = 0.0 if not col.endswith('_rank_in_shop') else 99.0
                 df[col] = default_val
                 logger.error(f"CRITICAL: Numeric column '{col}' was expected but not found in input df AND its median was not in loaded numeric_medians. Defaulted to {default_val}. This may indicate a schema mismatch or an issue with saved model artifacts.")


        # Add any other missing columns from self.expected_columns with appropriate defaults
        for col in self.expected_columns:
            if col not in df.columns:
                if col in self.categorical_features:
                    df[col] = "NONE"
                elif col in self.numeric_medians: # Check medians first for any numeric column
                    df[col] = self.numeric_medians[col]
                elif col.endswith('_rank_in_shop'): # Specific default for rank features
                    df[col] = 99.0
                    # No warning needed here if it's a rank feature and median is missing, as 99.0 is a defined fallback.
                else: # General numeric default for other columns not in numeric_medians
                    df[col] = 0.0
                    logger.error(f"CRITICAL: Expected column '{col}' was missing from input DataFrame AND its median was not in loaded numeric_medians. Defaulted to 0.0. This may indicate a schema mismatch or an issue with saved model artifacts.")
                # Original warning for missing column, now more contextual based on median presence
                # This specific warning for defaulting based on iloc[0] is less reliable now with more robust defaulting.
                # The CRITICAL error above is more important.
                # if not (col in self.numeric_medians or col in self.categorical_features or col.endswith('_rank_in_shop')):
                #      logger.warning(f"Column '{col}' was missing from input DataFrame and not found in medians/categoricals. Defaulted to '{df[col].iloc[0] if len(df)>0 and isinstance(df[col], pd.Series) else df[col] }'.")
        
        # Reorder columns to match training, ensure all expected columns are present
        df = df.reindex(columns=self.expected_columns, fill_value=0) # fill_value for any still missed numeric
        # Post-reindex fill for categoricals that might have been added as 0 by reindex
        for col in self.categorical_features:
            if col in df.columns and df[col].dtype != 'object' and df[col].dtype != 'str':
                 # If a categorical column was added by reindex (e.g. as 0), convert to "NONE"
                 if (df[col] == 0).all(): # Check if it was filled with 0
                      df[col] = "NONE"
                 df[col] = df[col].astype(str)

        # Log columns that were added because they were in expected_columns but not in initial_columns
        added_columns = set(self.expected_columns) - initial_columns
        if added_columns:
            logger.info(f"Columns added to DataFrame to match expected schema (filled with defaults/medians): {sorted(list(added_columns))}")
        
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
        predictions = np.clip(predictions, 0, 1)  # Ensure predictions are rates [0,1]
        
        return predictions
    
    def _aggregate_predictions(self, predicted_rates: np.ndarray, gender_counts: List[int]) -> float:
        """
        Aggregate model predictions (selection rates) to total quantity.
        """
        
        if len(predicted_rates) != len(gender_counts):
            logger.error(f"Mismatch: {len(predicted_rates)} predictions vs {len(gender_counts)} counts")
            return 0.0
        
        # Calculate expected quantity for each gender group
        expected_quantities = []
        for rate, count in zip(predicted_rates, gender_counts):
            expected_quantities.append(rate * count)
            
        # Sum all gender-specific expected quantities
        total_expected_qty = np.sum(expected_quantities)
        
        return max(0, total_expected_qty)
    
    def _calculate_confidence(self, predictions: np.ndarray, num_groups: int) -> float:
        """
        Calculate prediction confidence score based on model RMSE.
        A lower RMSE (better model performance) results in higher confidence.
        """
        
        if self.model_rmse is None:
            logger.warning("Model RMSE is not available. Returning default low confidence.")
            return 0.5 # Default low confidence if RMSE is missing
            
        # Map RMSE to confidence: 0.5 + 0.45 * exp(-4.05 * RMSE)
        # This maps RMSE to a range of [0.5, 0.95]
        # RMSE = 0 -> confidence = 0.5 + 0.45 * 1 = 0.95
        # RMSE = 0.2 -> confidence = 0.5 + 0.45 * exp(-0.81) ~ 0.5 + 0.45 * 0.444 ~ 0.5 + 0.20 = 0.70
        # RMSE = 0.4 -> confidence = 0.5 + 0.45 * exp(-1.62) ~ 0.5 + 0.45 * 0.197 ~ 0.5 + 0.09 = 0.59
        # RMSE -> large, confidence -> 0.5
        
        k = 4.05
        confidence = 0.5 + 0.45 * np.exp(-k * self.model_rmse)
        
        # Ensure confidence is within a sensible range, e.g., [0.5, 0.95]
        confidence = np.clip(confidence, 0.5, 0.95)
        
        return float(confidence)

# Force fresh instances to avoid caching issues during development
_predictor_instance = None

def get_predictor(model_path: str = "models/catboost_rmse_model/catboost_rmse_model.cbm") -> GiftDemandPredictor:
    """
    Get predictor instance. Creates fresh instance to ensure latest code changes are applied.
    
    Args:
        model_path: Path to trained CatBoost model (e.g., "models/catboost_rmse_model/catboost_rmse_model.cbm")
        
    Returns:
        GiftDemandPredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        logger.info(f"No existing predictor instance found. Creating new instance with model: {model_path}")
        _predictor_instance = GiftDemandPredictor(model_path)
        logger.info("Predictor instance created and cached.")
    else:
        # Optional: Check if model_path has changed, though typically it won't for a running server.
        # If it could change, logic to re-initialize would be needed here or via a separate reload mechanism.
        logger.debug("Returning cached predictor instance.")
        
    return _predictor_instance