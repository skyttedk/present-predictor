"""
XGBoost model implementation for the Predictive Gift Selection System.
Handles model training, prediction, and evaluation.
"""

import logging
import pickle
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from ..config.model_config import get_model_config, get_xgboost_params
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class DemandPredictor:
    """
    XGBoost-based demand prediction model.
    Predicts selection counts based on historical patterns.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """Initialize the demand predictor."""
        if model_config is None:
            model_config = get_xgboost_params()
        
        self.model_config = model_config
        self.model = XGBRegressor(**model_config)
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.label_encoders: Optional[Dict[str, Any]] = None
        self.training_stats: Dict[str, Any] = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the XGBoost model on historical data.
        
        Args:
            X: Feature matrix
            y: Target variable (selection counts)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training statistics and metrics
        """
        logger.info(f"Training XGBoost model with {len(X)} samples and {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle small datasets (< 10 samples) differently
        if len(X) < 10:
            logger.warning(f"Small dataset detected ({len(X)} samples). Using full dataset for training.")
            
            # Train on full dataset for small datasets
            self.model.fit(X, y, verbose=False)
            
            # Calculate metrics on full dataset
            y_pred = self.model.predict(X)
            full_metrics = self._calculate_metrics(y, y_pred, "full dataset")
            
            # Use leave-one-out cross-validation for small datasets
            cv_scores = self._perform_cross_validation(X, y, cv_folds=min(len(X), 5))
            
            # Store training statistics
            self.training_stats = {
                'model_config': self.model_config,
                'training_samples': len(X),
                'validation_samples': 0,  # No validation split for small datasets
                'feature_count': X.shape[1],
                'feature_names': self.feature_names,
                'train_metrics': full_metrics,
                'validation_metrics': {'note': 'No validation split due to small dataset'},
                'cross_validation': cv_scores,
                'feature_importance': self.get_feature_importance(),
                'small_dataset_warning': True
            }
            
            logger.info(f"Model training completed on small dataset. Full dataset R²: {full_metrics['r2']:.4f}")
            
        else:
            # Normal training for larger datasets
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=None
            )
            
            # Train the model
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Make predictions for evaluation
            y_train_pred = self.model.predict(X_train)
            y_val_pred = self.model.predict(X_val)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, "training")
            val_metrics = self._calculate_metrics(y_val, y_val_pred, "validation")
            
            # Cross-validation
            cv_scores = self._perform_cross_validation(X, y)
            
            # Store training statistics
            self.training_stats = {
                'model_config': self.model_config,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': X.shape[1],
                'feature_names': self.feature_names,
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'cross_validation': cv_scores,
                'feature_importance': self.get_feature_importance(),
                'small_dataset_warning': False
            }
            
            logger.info(f"Model training completed. Validation R²: {val_metrics['r2']:.4f}")
        
        self.is_trained = True
        return self.training_stats
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted selection counts
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        if list(X.columns) != self.feature_names:
            raise ValueError(f"Feature names don't match. Expected: {self.feature_names}")
        
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_single(self, features: Dict[str, Any]) -> float:
        """
        Make a single prediction from feature dictionary.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Predicted selection count
        """
        # Create DataFrame from features
        df = pd.DataFrame([features])
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        prediction = self.predict(df)[0]
        return float(prediction)
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
        }
        
        # Calculate MAPE only if no zero values
        if not (y_true == 0).any():
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        
        logger.info(f"{dataset_name.capitalize()} metrics - MAE: {metrics['mae']:.4f}, "
                   f"RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        config = get_model_config()
        cv_folds = config.training.cv_folds
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='r2')
        
        cv_results = {
            'mean_r2': cv_scores.mean(),
            'std_r2': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        logger.info(f"Cross-validation R² - Mean: {cv_results['mean_r2']:.4f} "
                   f"(±{cv_results['std_r2']:.4f})")
        
        return cv_results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            return {}
        
        importance_scores = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importance_scores))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_model(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save the trained model and metadata.
        
        Args:
            model_path: Path to save the model
            metadata_path: Path to save metadata (optional)
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        if metadata_path is None:
            metadata_path = model_path.replace('.pkl', '_metadata.json')
        
        metadata = {
            'model_config': self.model_config,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'is_trained': self.is_trained
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load a trained model and metadata.
        
        Args:
            model_path: Path to the saved model
            metadata_path: Path to the metadata file (optional)
        """
        # Load the model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        if metadata_path is None:
            metadata_path = model_path.replace('.pkl', '_metadata.json')
        
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.model_config = metadata.get('model_config', {})
            self.feature_names = metadata.get('feature_names', [])
            self.training_stats = metadata.get('training_stats', {})
            self.is_trained = metadata.get('is_trained', True)
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_prediction_explanation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get explanation for a prediction (feature contributions).
        
        Args:
            features: Feature dictionary
            
        Returns:
            Explanation dictionary with feature contributions
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        prediction = self.predict_single(features)
        feature_importance = self.get_feature_importance()
        
        # Create explanation
        explanation = {
            'prediction': prediction,
            'features_used': features,
            'feature_importance': feature_importance,
            'top_contributing_features': dict(list(feature_importance.items())[:5])
        }
        
        return explanation


def train_demand_prediction_model(X: pd.DataFrame, y: pd.Series) -> DemandPredictor:
    """
    Convenience function to train a demand prediction model.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Trained DemandPredictor instance
    """
    predictor = DemandPredictor()
    predictor.train(X, y)
    return predictor


def load_trained_model(model_path: str) -> DemandPredictor:
    """
    Load a previously trained model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded DemandPredictor instance
    """
    predictor = DemandPredictor()
    predictor.load_model(model_path)
    return predictor