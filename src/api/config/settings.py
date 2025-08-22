"""
API Configuration Manager
WBS 2.4: Environment-based configuration for API backend
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from functools import lru_cache

class APISettings(BaseSettings):
    """API configuration settings"""
    
    # Application Settings
    app_name: str = Field(default="Web Content Analyzer API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Settings
    host: str = Field(default="127.0.0.1", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    reload: bool = Field(default=True, env="API_RELOAD")
    workers: int = Field(default=1, env="API_WORKERS")
    
    # CORS Settings
    cors_origins: list = Field(
        default=["http://localhost:8501", "http://127.0.0.1:8501"], 
        env="CORS_ORIGINS"
    )
    cors_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")
    cors_methods: list = Field(default=["*"], env="CORS_METHODS")
    cors_headers: list = Field(default=["*"], env="CORS_HEADERS")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=10, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    
    # Analysis Settings
    max_concurrent_analyses: int = Field(default=3, env="MAX_CONCURRENT_ANALYSES")
    analysis_timeout: int = Field(default=300, env="ANALYSIS_TIMEOUT")  # seconds
    max_content_length: int = Field(default=1000000, env="MAX_CONTENT_LENGTH")  # bytes
    
    # LLM Settings
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")
    llm_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    llm_max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.3, env="LLM_TEMPERATURE")
    
    # Storage Settings
    storage_type: str = Field(default="file", env="STORAGE_TYPE")  # file, database, etc.
    storage_path: str = Field(default="./data", env="STORAGE_PATH")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Security Settings
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    security_headers_enabled: bool = Field(default=True, env="SECURITY_HEADERS_ENABLED")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("cors_methods", pre=True)
    def parse_cors_methods(cls, v):
        """Parse CORS methods from string or list"""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator("cors_headers", pre=True)
    def parse_cors_headers(cls, v):
        """Parse CORS headers from string or list"""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting"""
        valid_environments = ["development", "testing", "staging", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v.lower()
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level setting"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == "production"
    
    def get_uvicorn_config(self) -> Dict[str, Any]:
        """Get configuration for Uvicorn server"""
        return {
            "host": self.host,
            "port": self.port,
            "reload": self.reload and self.is_development,
            "workers": 1 if self.is_development else self.workers,
            "log_level": self.log_level.lower(),
            "access_log": True
        }
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": self.cors_origins,
            "allow_credentials": self.cors_credentials,
            "allow_methods": self.cors_methods,
            "allow_headers": self.cors_headers
        }


@lru_cache()
def get_settings() -> APISettings:
    """Get cached API settings instance"""
    return APISettings()


def configure_logging(settings: APISettings) -> None:
    """Configure application logging"""
    import logging
    import logging.config
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": settings.log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": settings.log_level
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": settings.log_level,
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            },
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    # Add file handler if log file is specified
    if settings.log_file:
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": settings.log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "detailed",
            "level": settings.log_level
        }
        
        # Update loggers to use file handler
        for logger_name in logging_config["loggers"]:
            logging_config["loggers"][logger_name]["handlers"].append("file")
    
    logging.config.dictConfig(logging_config)


def create_env_template():
    """Create a template .env file with all available settings"""
    
    env_template = """# Web Content Analyzer API Configuration

# Application Settings
APP_NAME=Web Content Analyzer API
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=false

# Server Settings
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true
API_WORKERS=1

# CORS Settings
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
CORS_CREDENTIALS=true
CORS_METHODS=*
CORS_HEADERS=*

# Rate Limiting
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=60
RATE_LIMIT_ENABLED=true

# Analysis Settings
MAX_CONCURRENT_ANALYSES=3
ANALYSIS_TIMEOUT=300
MAX_CONTENT_LENGTH=1000000

# LLM Settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your-openai-api-key-here
LLM_MAX_TOKENS=2000
LLM_TEMPERATURE=0.3

# Storage Settings
STORAGE_TYPE=file
STORAGE_PATH=./data

# Logging Settings
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
# LOG_FILE=./logs/api.log

# Security Settings
SECRET_KEY=your-secret-key-here
SECURITY_HEADERS_ENABLED=true
"""
    
    env_file_path = ".env.template"
    with open(env_file_path, "w") as f:
        f.write(env_template.strip())
    
    print(f"Environment template created at {env_file_path}")
    print("Copy this file to .env and update the values as needed")


if __name__ == "__main__":
    # Create environment template
    create_env_template()
    
    # Show current settings
    settings = get_settings()
    print("\nCurrent API Settings:")
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    print(f"Host: {settings.host}:{settings.port}")
    print(f"CORS Origins: {settings.cors_origins}")
    print(f"Rate Limiting: {settings.rate_limit_requests} requests per {settings.rate_limit_window}s")
