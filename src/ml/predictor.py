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
        # Interaction hasher constants for clarity
        self.HASH_DIM_PER_SET = 32
        self.NUM_HASH_SETS = 3
        self.hasher = FeatureHasher(n_features=self.HASH_DIM_PER_SET, input_type='string')
        self.model_rmse: float | None = None # To store model RMSE for legacy compatibility
        self.model_poisson: float | None = None # To store model Poisson deviance (primary metric)
        
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

                # Load model performance metrics from metadata
                performance_metrics = metadata.get('performance_metrics', {})
                self.model_rmse = performance_metrics.get('rmse_validation')
                self.model_poisson = performance_metrics.get('poisson_deviance')
                
                if self.model_rmse is not None:
                    logger.info(f"Model RMSE loaded successfully: {self.model_rmse:.4f}")
                else:
                    logger.warning("Model RMSE (rmse_validation) not found in metadata. Confidence score might be affected.")
                    self.model_rmse = 0.3 # Default RMSE if not found
                    
                if self.model_poisson is not None:
                    logger.info(f"Model Poisson deviance loaded successfully: {self.model_poisson:.4f}")
                else:
                    logger.warning("Model Poisson deviance not found in metadata.")
                    self.model_poisson = 3.5 # Default Poisson deviance if not found
                
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
                    # Step 1: Get shop features using simplified resolver
                    shop_features = self.shop_resolver.resolve_features(
                        shop_id=branch,  # Shop ID is now branch (consolidated)
                        main_category=present.get('item_main_category', 'NONE'),
                        sub_category=present.get('item_sub_category', 'NONE'),
                        brand=present.get('brand', 'NONE')
                    )
                    
                    prediction = self._predict_for_present(
                        present, employee_stats, branch, shop_features
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
                    present, gender, branch, shop_features, count
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
            # Make predictions using CatBoost Pool - model outputs counts now
            predicted_counts = self._make_catboost_prediction(feature_df)
            
            # Aggregate predictions - counts are summed directly
            total_prediction = self._aggregate_predictions(
                predicted_counts, gender_counts_list
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(predicted_counts)
            
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
                             branch: str, shop_features: Dict, count: int) -> Dict:
        """Create feature vector for one present-employee combination"""
        
        # Base features matching training data structure
        # Shop and branch are now identical per expert feedback
        employee_shop = branch  # Shop and branch are now consolidated
        
        features = {
            'employee_shop': employee_shop,
            'employee_gender': employee_gender,
            'product_main_category': present.get('item_main_category', 'NONE'),
            'product_sub_category': present.get('item_sub_category', 'NONE'),
            'product_brand': present.get('brand', 'NONE'),
            'product_color': present.get('color', 'NONE'),
            'product_durability': present.get('durability', 'NONE'),
            'product_target_gender': present.get('target_demographic', 'NONE'),
            'product_utility_type': present.get('utility_type', 'NONE'),
            'product_type': present.get('usage_type', 'NONE'),
            'log_exposure': np.log(count + 1e-8)  # CRITICAL: Add log_exposure feature
        }
        
        # Add shop-level features (binary indicators are already included in shop_features)
        features.update(shop_features)
        
        return features
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hashed interaction features using sub-tokens."""
        if df.empty:
            # Add empty hash columns if df is empty to maintain schema
            total_hash_features = self.HASH_DIM_PER_SET * self.NUM_HASH_SETS  # 96 total
            for i in range(total_hash_features):
                df[f'interaction_hash_{i}'] = 0.0
            return df

        # Create multiple interaction sets for better signal capture
        # First set: shop x main_category (existing)
        interaction1 = df.apply(
            lambda x: [
                f"shop_{x.get('employee_shop', 'NONE')}",
                f"cat_{x.get('product_main_category', 'NONE')}"
            ],
            axis=1
        )
        
        # Second set: brand x target_gender
        interaction2 = df.apply(
            lambda x: [
                f"brand_{x.get('product_brand', 'NONE')}",
                f"gender_{x.get('product_target_gender', 'NONE')}"
            ],
            axis=1
        )
        
        # Third set: sub_category x utility_type
        interaction3 = df.apply(
            lambda x: [
                f"subcat_{x.get('product_sub_category', 'NONE')}",
                f"utility_{x.get('product_utility_type', 'NONE')}"
            ],
            axis=1
        )
        
        # Hash all interaction sets
        hash1 = self.hasher.transform(interaction1).toarray()
        hash2 = self.hasher.transform(interaction2).toarray()
        hash3 = self.hasher.transform(interaction3).toarray()
        
        # Combine all hashes
        all_hashes = np.hstack([hash1, hash2, hash3])
        
        # Add as columns
        for i in range(all_hashes.shape[1]):
            df[f'interaction_hash_{i}'] = all_hashes[:, i]
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure features match model expectations"""
        
        initial_columns = set(df.columns) # Store initial columns for logging later
        
        # Preserve log_exposure for baseline offset (will be used in _make_catboost_prediction)
        log_exposure_preserved = None
        if 'log_exposure' in df.columns:
            log_exposure_preserved = df['log_exposure'].copy()
        
        # Convert categorical columns to string
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("NONE")
        
        # Ensure numeric columns - Fix C4: Enumerate all 96 hash features (32 * 3 interaction sets)
        num_hash_features = self.HASH_DIM_PER_SET * self.NUM_HASH_SETS  # 96
        
        numeric_cols = [
            'shop_main_category_diversity_selected', 'shop_brand_diversity_selected',
            'shop_utility_type_diversity_selected', 'shop_sub_category_diversity_selected',
            'unique_product_combinations_in_shop',
            'is_shop_most_frequent_main_category', 'is_shop_most_frequent_brand',
            'product_share_in_shop', 'brand_share_in_shop',
            'product_rank_in_shop', 'brand_rank_in_shop'
            # log_exposure removed - now used as baseline offset, not feature
        ] + [f'interaction_hash_{i}' for i in range(num_hash_features)]
        
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

        # Restore log_exposure for baseline offset usage
        if log_exposure_preserved is not None:
            df['log_exposure'] = log_exposure_preserved

        # Log columns that were added because they were in expected_columns but not in initial_columns
        added_columns = set(self.expected_columns) - initial_columns
        if added_columns:
            logger.info(f"Columns added to DataFrame to match expected schema (filled with defaults/medians): {sorted(list(added_columns))}")
        
        return df
    
    def _make_catboost_prediction(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Make prediction using CatBoost with proper baseline offset for Poisson model"""
        
        # Extract log_exposure for baseline offset (keep a copy of offset)
        log_offset = feature_df['log_exposure'].values  # <-- 1
        
        # Drop log_exposure from feature set fed to the model
        feature_df_nolog = feature_df.drop(columns=['log_exposure'])  # <-- 2
        
        # Get categorical feature indices
        cat_feature_indices = [
            feature_df_nolog.columns.get_loc(col) for col in self.categorical_features
            if col in feature_df_nolog.columns
        ]
        
        # Create CatBoost Pool with baseline offset
        test_pool = Pool(
            data=feature_df_nolog,
            cat_features=cat_feature_indices,
            baseline=log_offset  # <-- 3
        )
        
        # Make predictions - Poisson model outputs counts directly, not rates
        pred_counts = self.model.predict(test_pool)
        pred_counts = np.maximum(pred_counts, 0)  # Ensure non-negative counts only
        
        return pred_counts
    
    def _aggregate_predictions(self, predicted_counts: np.ndarray, gender_counts: List[int]) -> float:
        """
        Aggregate model predictions (selection counts) to total quantity.
        Model now predicts counts directly, so we sum them without multiplication.
        """
        
        if len(predicted_counts) != len(gender_counts):
            logger.error(f"Mismatch: {len(predicted_counts)} predictions vs {len(gender_counts)} counts")
            return 0.0
        
        # Model predicts counts directly - sum without multiplication by employee counts
        total_expected_qty = np.sum(predicted_counts)
        
        return max(0, total_expected_qty)
    
    def _calculate_confidence_poisson(self, predictions: np.ndarray) -> float:
        """
        Calculate confidence based on Poisson prediction variability for count data.
        """
        mean_pred = np.mean(predictions)
        if mean_pred > 0:
            # Coefficient of variation for Poisson
            cv = np.std(predictions) / mean_pred
            confidence = 1 / (1 + cv)
        else:
            confidence = 0.5
        return np.clip(confidence, 0.5, 0.95)
    
    def _calculate_confidence(self, predictions: np.ndarray) -> float:
        """
        Calculate prediction confidence score - updated for Poisson model.
        Removed unused num_groups parameter as recommended by expert.
        """
        # Use Poisson-based confidence calculation since model now predicts counts
        return self._calculate_confidence_poisson(predictions)

# Force fresh instances to avoid caching issues during development
_predictor_instance = None

def get_predictor(model_path: str = "models/catboost_poisson_model/catboost_poisson_model.cbm") -> GiftDemandPredictor:
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


def clear_predictor_cache() -> bool:
    """
    Clear the cached predictor instance to force reload on next access.
    
    Returns:
        bool: True if a cached instance was cleared, False if no instance was cached
    """
    global _predictor_instance
    
    if _predictor_instance is not None:
        logger.info("Clearing cached predictor instance to force fresh model reload")
        _predictor_instance = None
        return True
    else:
        logger.info("No cached predictor instance found to clear")
        return False