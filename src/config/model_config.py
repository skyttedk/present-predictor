"""
Machine learning model configuration for the Predictive Gift Selection System.
Contains XGBoost model parameters and ML pipeline settings.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class XGBoostConfig(BaseModel):
    """XGBoost model hyperparameters configuration."""
    
    # Core XGBoost parameters
    n_estimators: int = Field(default=100, description="Number of boosting rounds")
    max_depth: int = Field(default=6, description="Maximum depth of trees")
    learning_rate: float = Field(default=0.1, description="Boosting learning rate")
    subsample: float = Field(default=1.0, description="Subsample ratio of training instances")
    colsample_bytree: float = Field(default=1.0, description="Subsample ratio of columns")
    
    # Regularization parameters
    reg_alpha: float = Field(default=0.0, description="L1 regularization term")
    reg_lambda: float = Field(default=1.0, description="L2 regularization term")
    gamma: float = Field(default=0.0, description="Minimum loss reduction for split")
    
    # Tree construction parameters
    min_child_weight: int = Field(default=1, description="Minimum sum of instance weight")
    max_delta_step: int = Field(default=0, description="Maximum delta step for weight estimation")
    
    # General parameters
    objective: str = Field(default="reg:squarederror", description="Learning objective")
    eval_metric: str = Field(default="rmse", description="Evaluation metric")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    n_jobs: int = Field(default=-1, description="Number of parallel threads")
    
    # Early stopping
    early_stopping_rounds: Optional[int] = Field(default=10, description="Early stopping rounds")
    
    def to_xgboost_params(self) -> Dict[str, Any]:
        """Convert to XGBoost parameter dictionary."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "max_delta_step": self.max_delta_step,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "random_state": self.seed,
            "n_jobs": self.n_jobs,
        }


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering pipeline configuration."""
    
    # Categorical encoding
    categorical_encoding_method: str = Field(
        default="onehot", 
        description="Method for categorical encoding (onehot, label, target)"
    )
    handle_unknown_categories: str = Field(
        default="ignore", 
        description="How to handle unknown categories (ignore, error)"
    )
    
    # Numerical scaling
    numerical_scaling_method: str = Field(
        default="standard", 
        description="Method for numerical scaling (standard, minmax, robust)"
    )
    
    # Feature selection
    feature_selection_enabled: bool = Field(
        default=True, 
        description="Whether to perform feature selection"
    )
    feature_selection_method: str = Field(
        default="mutual_info", 
        description="Feature selection method (mutual_info, chi2, f_regression)"
    )
    max_features: Optional[int] = Field(
        default=None, 
        description="Maximum number of features to select"
    )
    
    # Target encoding
    target_encoding_categories: List[str] = Field(
        default_factory=lambda: ["Item Main Category", "Item Sub Category", "Color"],
        description="Categories to apply target encoding"
    )
    
    # Temporal features
    extract_temporal_features: bool = Field(
        default=True, 
        description="Whether to extract temporal features from dates"
    )
    temporal_features: List[str] = Field(
        default_factory=lambda: ["year", "month", "quarter", "week", "day_of_week"],
        description="Temporal features to extract"
    )


class DataValidationConfig(BaseModel):
    """Data validation and quality checks configuration."""
    
    # Missing value handling
    missing_value_threshold: float = Field(
        default=0.3, 
        description="Maximum allowed missing value ratio per column"
    )
    missing_value_strategy: str = Field(
        default="drop", 
        description="Strategy for handling missing values (drop, impute)"
    )
    
    # Outlier detection
    outlier_detection_enabled: bool = Field(
        default=True, 
        description="Whether to perform outlier detection"
    )
    outlier_method: str = Field(
        default="iqr", 
        description="Outlier detection method (iqr, zscore, isolation_forest)"
    )
    outlier_threshold: float = Field(
        default=1.5, 
        description="Threshold for outlier detection"
    )
    
    # Data consistency checks
    min_records_per_category: int = Field(
        default=5, 
        description="Minimum records required per category"
    )
    max_categorical_cardinality: int = Field(
        default=100, 
        description="Maximum allowed unique values for categorical features"
    )


class ModelTrainingConfig(BaseModel):
    """Model training pipeline configuration."""
    
    # Data splitting
    train_test_split_ratio: float = Field(
        default=0.8, 
        description="Training data ratio for train/test split"
    )
    validation_split_ratio: float = Field(
        default=0.2, 
        description="Validation data ratio from training set"
    )
    stratify_split: bool = Field(
        default=True, 
        description="Whether to stratify splits by target categories"
    )
    
    # Cross-validation
    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    cv_shuffle: bool = Field(default=True, description="Whether to shuffle CV folds")
    cv_random_state: int = Field(default=42, description="Random state for CV")
    
    # Model evaluation
    evaluation_metrics: List[str] = Field(
        default_factory=lambda: ["mae", "rmse", "r2", "mape"],
        description="Metrics for model evaluation"
    )
    
    # Model persistence
    model_save_path: str = Field(
        default="models/xgboost_demand_predictor.pkl",
        description="Path to save trained model"
    )
    feature_names_path: str = Field(
        default="models/feature_names.json",
        description="Path to save feature names"
    )


class PredictionConfig(BaseModel):
    """Prediction service configuration."""
    
    # Prediction constraints
    min_prediction_value: float = Field(
        default=0.0, 
        description="Minimum allowed prediction value"
    )
    max_prediction_value: float = Field(
        default=10000.0, 
        description="Maximum allowed prediction value"
    )
    
    # Prediction rounding
    round_predictions: bool = Field(
        default=True, 
        description="Whether to round predictions to integers"
    )
    
    # Confidence intervals
    provide_confidence_intervals: bool = Field(
        default=False, 
        description="Whether to provide prediction confidence intervals"
    )
    confidence_level: float = Field(
        default=0.95, 
        description="Confidence level for intervals"
    )
    
    # Batch prediction
    max_batch_size: int = Field(
        default=1000, 
        description="Maximum batch size for bulk predictions"
    )


class ModelConfig(BaseModel):
    """Main model configuration combining all ML-related settings."""
    
    # Component configurations
    xgboost: XGBoostConfig = XGBoostConfig()
    feature_engineering: FeatureEngineeringConfig = FeatureEngineeringConfig()
    data_validation: DataValidationConfig = DataValidationConfig()
    training: ModelTrainingConfig = ModelTrainingConfig()
    prediction: PredictionConfig = PredictionConfig()
    
    # Global ML settings
    random_seed: int = Field(default=42, description="Global random seed")
    reproducible_results: bool = Field(
        default=True, 
        description="Whether to ensure reproducible results"
    )
    
    # Model versioning
    model_version: str = Field(default="1.0.0", description="Model version")
    model_name: str = Field(
        default="gavefabrikken_demand_predictor", 
        description="Model name identifier"
    )
    
    class Config:
        env_prefix = "MODEL_"
        case_sensitive = False
        protected_namespaces = ()


# Global model configuration instance
model_config = ModelConfig()


def get_model_config() -> ModelConfig:
    """Get model configuration instance."""
    return model_config


def get_xgboost_params() -> Dict[str, Any]:
    """Get XGBoost parameters as dictionary."""
    return model_config.xgboost.to_xgboost_params()


def update_model_config(**kwargs) -> ModelConfig:
    """Update model configuration with new values."""
    global model_config
    for key, value in kwargs.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
    return model_config