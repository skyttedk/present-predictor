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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
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