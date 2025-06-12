"""
Configuration validation for the Predictive Gift Selection System.
Validates configuration settings and provides helpful error messages.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .settings import Settings, get_settings
from .model_config import ModelConfig, get_model_config

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigValidator:
    """Validates application and model configuration."""
    
    def __init__(self, settings: Optional[Settings] = None, model_config: Optional[ModelConfig] = None):
        self.settings = settings or get_settings()
        self.model_config = model_config or get_model_config()
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> bool:
        """Validate all configuration sections. Returns True if valid."""
        self.errors.clear()
        self.warnings.clear()
        
        # Validate individual sections
        self._validate_environment()
        self._validate_api_settings()
        self._validate_data_settings()
        self._validate_logging_settings()
        self._validate_security_settings()
        self._validate_model_config()
        self._validate_paths()
        
        # Log results
        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        if self.errors:
            for error in self.errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def _validate_environment(self) -> None:
        """Validate environment settings."""
        valid_environments = ["development", "testing", "production"]
        if self.settings.environment.lower() not in valid_environments:
            self.errors.append(
                f"Invalid environment '{self.settings.environment}'. "
                f"Must be one of: {', '.join(valid_environments)}"
            )
        
        # Production-specific validations
        if self.settings.is_production:
            if self.settings.debug:
                self.warnings.append("Debug mode is enabled in production environment")
            
            if self.settings.security.secret_key == "dev-secret-key-change-in-production":
                self.errors.append("Default secret key detected in production environment")
    
    def _validate_api_settings(self) -> None:
        """Validate API configuration."""
        api = self.settings.api
        
        # Port validation
        if not (1 <= api.port <= 65535):
            self.errors.append(f"Invalid API port {api.port}. Must be between 1 and 65535")
        
        # CORS validation
        if self.settings.is_production and "*" in api.allow_origins:
            self.warnings.append("Wildcard CORS origins detected in production")
    
    def _validate_data_settings(self) -> None:
        """Validate data processing settings."""
        data = self.settings.data
        
        # File size validation
        if data.max_file_size_mb <= 0:
            self.errors.append("Maximum file size must be positive")
        
        # Batch size validation
        if data.batch_size <= 0:
            self.errors.append("Batch size must be positive")
        
        # Validation split validation
        if not (0.0 < data.validation_split < 1.0):
            self.errors.append("Validation split must be between 0 and 1")
        
        # Encoding method validation
        valid_encodings = ["onehot", "label", "target"]
        if data.categorical_encoding not in valid_encodings:
            self.errors.append(
                f"Invalid categorical encoding '{data.categorical_encoding}'. "
                f"Must be one of: {', '.join(valid_encodings)}"
            )
    
    def _validate_logging_settings(self) -> None:
        """Validate logging configuration."""
        logging_config = self.settings.logging
        
        # Log level validation
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if logging_config.level.upper() not in valid_levels:
            self.errors.append(
                f"Invalid log level '{logging_config.level}'. "
                f"Must be one of: {', '.join(valid_levels)}"
            )
        
        # File size validation
        if logging_config.max_file_size_mb <= 0:
            self.errors.append("Log file size must be positive")
        
        # Backup count validation
        if logging_config.backup_count < 0:
            self.errors.append("Log backup count cannot be negative")
    
    def _validate_security_settings(self) -> None:
        """Validate security configuration."""
        security = self.settings.security
        
        # Secret key validation
        if len(security.secret_key) < 16:
            self.errors.append("Secret key must be at least 16 characters long")
        
        # Token expiration validation
        if security.access_token_expire_minutes <= 0:
            self.errors.append("Access token expiration must be positive")
        
        # Rate limiting validation
        if security.rate_limit_enabled:
            if security.rate_limit_requests <= 0:
                self.errors.append("Rate limit requests must be positive")
            if security.rate_limit_period <= 0:
                self.errors.append("Rate limit period must be positive")
    
    def _validate_model_config(self) -> None:
        """Validate machine learning model configuration."""
        # XGBoost parameters validation
        xgb = self.model_config.xgboost
        
        if xgb.n_estimators <= 0:
            self.errors.append("Number of estimators must be positive")
        
        if xgb.max_depth <= 0:
            self.errors.append("Maximum depth must be positive")
        
        if not (0.0 < xgb.learning_rate <= 1.0):
            self.errors.append("Learning rate must be between 0 and 1")
        
        if not (0.0 < xgb.subsample <= 1.0):
            self.errors.append("Subsample ratio must be between 0 and 1")
        
        if not (0.0 < xgb.colsample_bytree <= 1.0):
            self.errors.append("Column sample ratio must be between 0 and 1")
        
        # Training configuration validation
        training = self.model_config.training
        
        if not (0.0 < training.train_test_split_ratio < 1.0):
            self.errors.append("Train/test split ratio must be between 0 and 1")
        
        if not (0.0 < training.validation_split_ratio < 1.0):
            self.errors.append("Validation split ratio must be between 0 and 1")
        
        if training.cv_folds < 2:
            self.errors.append("Cross-validation folds must be at least 2")
        
        # Feature engineering validation
        fe = self.model_config.feature_engineering
        valid_categorical_methods = ["onehot", "label", "target"]
        if fe.categorical_encoding_method not in valid_categorical_methods:
            self.errors.append(
                f"Invalid categorical encoding method '{fe.categorical_encoding_method}'"
            )
        
        valid_scaling_methods = ["standard", "minmax", "robust"]
        if fe.numerical_scaling_method not in valid_scaling_methods:
            self.errors.append(
                f"Invalid numerical scaling method '{fe.numerical_scaling_method}'"
            )
    
    def _validate_paths(self) -> None:
        """Validate file and directory paths."""
        # Data paths
        data_path = Path(self.settings.data.historical_data_path)
        model_path = Path(self.settings.data.model_storage_path)
        
        # Create directories if they don't exist (development only)
        if self.settings.is_development:
            try:
                data_path.mkdir(parents=True, exist_ok=True)
                model_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.warnings.append(f"Could not create directories: {e}")
        
        # Log file path validation
        if self.settings.logging.file_path:
            log_path = Path(self.settings.logging.file_path)
            log_dir = log_path.parent
            
            if self.settings.is_development:
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self.warnings.append(f"Could not create log directory: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation results summary."""
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


def validate_configuration() -> bool:
    """Validate the current application configuration."""
    validator = ConfigValidator()
    return validator.validate_all()


def validate_and_raise() -> None:
    """Validate configuration and raise exception if invalid."""
    validator = ConfigValidator()
    if not validator.validate_all():
        error_messages = "\n".join(validator.errors)
        raise ConfigurationError(f"Configuration validation failed:\n{error_messages}")


def get_validation_summary() -> Dict[str, Any]:
    """Get configuration validation summary."""
    validator = ConfigValidator()
    validator.validate_all()
    return validator.get_validation_summary()