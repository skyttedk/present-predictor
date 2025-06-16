"""
Application settings for the Predictive Gift Selection System.
Handles environment variables and application configuration.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """API-related configuration settings."""
    
    title: str = "Predictive Gift Selection API"
    description: str = "ML-powered demand prediction system for Gavefabrikken"
    version: str = "0.1.0"
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    reload: bool = Field(default=False, env="API_RELOAD")
    
    # CORS settings
    allow_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")
    allow_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")
    allow_methods: list[str] = Field(default=["*"], env="CORS_METHODS")
    allow_headers: list[str] = Field(default=["*"], env="CORS_HEADERS")


class DatabaseSettings(BaseSettings):
    """Database configuration settings (for future use)."""
    
    url: Optional[str] = Field(default=None, env="DATABASE_URL")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")


class DataSettings(BaseSettings):
    """Data processing configuration settings."""
    
    # Data file paths
    historical_data_path: str = Field(default="data/historical", env="HISTORICAL_DATA_PATH")
    model_storage_path: str = Field(default="models", env="MODEL_STORAGE_PATH")
    
    # Data processing settings
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    batch_size: int = Field(default=1000, env="DATA_BATCH_SIZE")
    validation_split: float = Field(default=0.2, env="VALIDATION_SPLIT")
    
    # Feature engineering settings
    categorical_encoding: str = Field(default="onehot", env="CATEGORICAL_ENCODING")
    handle_missing_values: str = Field(default="drop", env="HANDLE_MISSING_VALUES")


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    max_file_size_mb: int = Field(default=10, env="LOG_MAX_FILE_SIZE_MB")
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")


class OpenAISettings(BaseSettings):
    """OpenAI specific configuration settings."""
    api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    assistant_id: str = Field(default="asst_BuFvA6iXF4xSyQ4px7Q5zjiN", env="OPENAI_ASSISTANT_ID")

    class Config:
        # env_prefix = "OPENAI_" # Not using prefix as .env keys are direct
        pass

class CatBoostModelSettings(BaseSettings):
    """CatBoost specific model configuration settings."""
    iterations: int = Field(default=1000, env="CATBOOST_ITERATIONS")
    loss_function: str = Field(default="Poisson", env="CATBOOST_LOSS_FUNCTION")
    eval_metric: str = Field(default="R2", env="CATBOOST_EVAL_METRIC")
    learning_rate: float = Field(default=0.05, env="CATBOOST_LEARNING_RATE")
    depth: int = Field(default=6, env="CATBOOST_DEPTH")
    l2_leaf_reg: float = Field(default=3.0, env="CATBOOST_L2_LEAF_REG")
    early_stopping_rounds: int = Field(default=50, env="CATBOOST_EARLY_STOPPING_ROUNDS")
    # CATBOOST_RANDOM_SEED can be part of general ModelSettings if a single seed is preferred.
    # If a specific seed for CatBoost is needed, add:
    # catboost_random_seed: int = Field(default=42, env="CATBOOST_RANDOM_SEED")


    class Config:
        pass # No env_prefix needed as .env keys are direct and mapped via 'env'

class ModelSettings(BaseSettings):
    """General model and training configuration settings."""
    random_seed: int = Field(default=42, env="MODEL_RANDOM_SEED")
    reproducible_results: bool = Field(default=True, env="MODEL_REPRODUCIBLE_RESULTS")
    version: str = Field(default="1.0.0", env="MODEL_VERSION")
    name: str = Field(default="gavefabrikken_demand_predictor", env="MODEL_NAME")

    # Feature Engineering
    feature_engineering_categorical_encoding_method: str = Field(default="onehot", env="MODEL_FEATURE_ENGINEERING_CATEGORICAL_ENCODING_METHOD")
    feature_engineering_numerical_scaling_method: str = Field(default="standard", env="MODEL_FEATURE_ENGINEERING_NUMERICAL_SCALING_METHOD")
    feature_engineering_feature_selection_enabled: bool = Field(default=True, env="MODEL_FEATURE_ENGINEERING_FEATURE_SELECTION_ENABLED")
    feature_engineering_extract_temporal_features: bool = Field(default=True, env="MODEL_FEATURE_ENGINEERING_EXTRACT_TEMPORAL_FEATURES")

    # Data Validation
    data_validation_missing_value_threshold: float = Field(default=0.3, env="MODEL_DATA_VALIDATION_MISSING_VALUE_THRESHOLD")
    data_validation_outlier_detection_enabled: bool = Field(default=True, env="MODEL_DATA_VALIDATION_OUTLIER_DETECTION_ENABLED")
    data_validation_outlier_method: str = Field(default="iqr", env="MODEL_DATA_VALIDATION_OUTLIER_METHOD")
    data_validation_min_records_per_category: int = Field(default=5, env="MODEL_DATA_VALIDATION_MIN_RECORDS_PER_CATEGORY")

    # Model Training
    training_train_test_split_ratio: float = Field(default=0.8, env="MODEL_TRAINING_TRAIN_TEST_SPLIT_RATIO")
    training_validation_split_ratio: float = Field(default=0.2, env="MODEL_TRAINING_VALIDATION_SPLIT_RATIO")
    training_cv_folds: int = Field(default=5, env="MODEL_TRAINING_CV_FOLDS")
    training_model_save_path: str = Field(default="models/catboost_poisson_model", env="MODEL_TRAINING_MODEL_SAVE_PATH")

    # Prediction Settings
    prediction_min_prediction_value: float = Field(default=0.0, env="MODEL_PREDICTION_MIN_PREDICTION_VALUE")
    prediction_max_prediction_value: float = Field(default=10000.0, env="MODEL_PREDICTION_MAX_PREDICTION_VALUE")
    prediction_round_predictions: bool = Field(default=True, env="MODEL_PREDICTION_ROUND_PREDICTIONS")
    prediction_max_batch_size: int = Field(default=1000, env="MODEL_PREDICTION_MAX_BATCH_SIZE")
    
    class Config:
        pass # No env_prefix needed

class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Component settings
    api: APISettings = APISettings()
    database: DatabaseSettings = DatabaseSettings()
    data: DataSettings = DataSettings()
    logging: LoggingSettings = LoggingSettings()
    security: SecuritySettings = SecuritySettings()
    openai: OpenAISettings = OpenAISettings()
    model_general: ModelSettings = ModelSettings()
    model_catboost: CatBoostModelSettings = CatBoostModelSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = 'ignore'  # Allow and ignore extra fields from .env at the top level
        protected_namespaces = ()
        
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ("development", "dev")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() in ("production", "prod")
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment.lower() in ("testing", "test")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global settings
    settings = Settings()
    return settings