"""
Configuration Service Implementation
Provides configuration management with environment variable support.
"""
import os
import logging
from typing import Optional, List, Dict, Any
from src.application.interfaces.configuration import (
    IConfigurationService, IEnvironmentService, ILoggingService,
    ScrapingConfig, SecurityConfig, AnalysisConfig
)


class EnvironmentService(IEnvironmentService):
    """
    Environment service for managing environment variables.
    Provides type-safe access to environment configuration.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def get_string(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get string environment variable."""
        value = os.getenv(key, default)
        if value is not None:
            self._logger.debug(f"Environment variable {key} retrieved")
        return value
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get integer environment variable."""
        value = os.getenv(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                self._logger.warning(f"Invalid integer value for {key}: {value}, using default")
                return default
        return default
    
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Get boolean environment variable."""
        value = os.getenv(key)
        if value is not None:
            return value.lower() in ('true', '1', 'yes', 'on')
        return default
    
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Get float environment variable."""
        value = os.getenv(key)
        if value is not None:
            try:
                return float(value)
            except ValueError:
                self._logger.warning(f"Invalid float value for {key}: {value}, using default")
                return default
        return default
    
    def get_list(self, key: str, separator: str = ",", default: Optional[List[str]] = None) -> Optional[List[str]]:
        """Get list environment variable."""
        value = os.getenv(key)
        if value is not None:
            return [item.strip() for item in value.split(separator) if item.strip()]
        return default or []
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        env = os.getenv('ENVIRONMENT', 'development').lower()
        return env in ('development', 'dev', 'local')
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        env = os.getenv('ENVIRONMENT', 'development').lower()
        return env in ('production', 'prod')


class LoggingService(ILoggingService):
    """
    Logging service for configuring application logging.
    Uses standard Python logging with configurable levels.
    """
    
    def __init__(self, env_service: IEnvironmentService):
        self._env_service = env_service
        self._configured = False
    
    def configure_logging(self, config: Dict[str, Any] = None) -> None:
        """Configure logging with given settings."""
        if self._configured:
            return
        
        # Default configuration
        default_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        
        if config:
            default_config.update(config)
        
        # Get level from environment
        log_level = self._env_service.get_string('LOG_LEVEL', default_config['level'])
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=default_config['format'],
            datefmt=default_config['datefmt']
        )
        
        # Set specific logger levels if in development
        if self._env_service.is_development():
            # More verbose logging in development
            logging.getLogger('src').setLevel(logging.DEBUG)
        
        self._configured = True
        logging.getLogger(__name__).info(f"Logging configured with level: {log_level}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger instance for given name."""
        if not self._configured:
            self.configure_logging()
        return logging.getLogger(name)
    
    def set_log_level(self, level: str) -> None:
        """Set global log level."""
        try:
            numeric_level = getattr(logging, level.upper())
            logging.getLogger().setLevel(numeric_level)
            logging.getLogger(__name__).info(f"Log level set to: {level}")
        except AttributeError:
            logging.getLogger(__name__).error(f"Invalid log level: {level}")


class ConfigurationService(IConfigurationService):
    """
    Main configuration service providing application configuration.
    Combines default values with environment variable overrides.
    """
    
    def __init__(self, env_service: IEnvironmentService):
        self._env_service = env_service
        self._logger = logging.getLogger(__name__)
        
        # Load configurations
        self._scraping_config = self._load_scraping_config()
        self._security_config = self._load_security_config()
        self._analysis_config = self._load_analysis_config()
    
    def get_scraping_config(self) -> ScrapingConfig:
        """Get scraping configuration."""
        return self._scraping_config
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return self._security_config
    
    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis configuration."""
        return self._analysis_config
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a configuration value."""
        self._logger.info(f"Updating config: {section}.{key} = {value}")
        
        if section == "scraping":
            if hasattr(self._scraping_config, key):
                setattr(self._scraping_config, key, value)
            else:
                self._logger.warning(f"Unknown scraping config key: {key}")
        elif section == "security":
            if hasattr(self._security_config, key):
                setattr(self._security_config, key, value)
            else:
                self._logger.warning(f"Unknown security config key: {key}")
        elif section == "analysis":
            if hasattr(self._analysis_config, key):
                setattr(self._analysis_config, key, value)
            else:
                self._logger.warning(f"Unknown analysis config key: {key}")
        else:
            self._logger.warning(f"Unknown config section: {section}")
    
    def reload_config(self) -> None:
        """Reload configuration from source."""
        self._logger.info("Reloading configuration")
        self._scraping_config = self._load_scraping_config()
        self._security_config = self._load_security_config()
        self._analysis_config = self._load_analysis_config()
    
    def _load_scraping_config(self) -> ScrapingConfig:
        """Load scraping configuration from environment."""
        return ScrapingConfig(
            user_agent=self._env_service.get_string(
                'SCRAPING_USER_AGENT', 
                'WebContentAnalyzer/1.0'
            ),
            timeout=self._env_service.get_int('SCRAPING_TIMEOUT', 30),
            max_retries=self._env_service.get_int('SCRAPING_MAX_RETRIES', 3),
            delay_between_requests=self._env_service.get_float('SCRAPING_DELAY', 1.0),
            respect_robots_txt=self._env_service.get_bool('SCRAPING_RESPECT_ROBOTS', True),
            max_content_size=self._env_service.get_int('SCRAPING_MAX_SIZE', 10_000_000),
            allowed_content_types=self._env_service.get_list(
                'SCRAPING_ALLOWED_TYPES',
                default=['text/html', 'application/xhtml+xml']
            )
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration from environment."""
        return SecurityConfig(
            block_private_ips=self._env_service.get_bool('SECURITY_BLOCK_PRIVATE_IPS', True),
            block_local_networks=self._env_service.get_bool('SECURITY_BLOCK_LOCAL', True),
            allowed_domains=self._env_service.get_list('SECURITY_ALLOWED_DOMAINS', default=[]),
            blocked_domains=self._env_service.get_list('SECURITY_BLOCKED_DOMAINS', default=[]),
            max_redirects=self._env_service.get_int('SECURITY_MAX_REDIRECTS', 5),
            validate_ssl=self._env_service.get_bool('SECURITY_VALIDATE_SSL', True),
            allowed_ports=self._env_service.get_list(
                'SECURITY_ALLOWED_PORTS',
                default=['80', '443', '8000', '8080']
            )
        )
    
    def _load_analysis_config(self) -> AnalysisConfig:
        """Load analysis configuration from environment."""
        return AnalysisConfig(
            enable_sentiment_analysis=self._env_service.get_bool('ANALYSIS_SENTIMENT', True),
            enable_keyword_extraction=self._env_service.get_bool('ANALYSIS_KEYWORDS', True),
            enable_content_summarization=self._env_service.get_bool('ANALYSIS_SUMMARY', True),
            max_keywords=self._env_service.get_int('ANALYSIS_MAX_KEYWORDS', 10),
            summary_max_length=self._env_service.get_int('ANALYSIS_SUMMARY_LENGTH', 500),
            llm_model=self._env_service.get_string('ANALYSIS_LLM_MODEL', 'gpt-3.5-turbo'),
            llm_timeout=self._env_service.get_int('ANALYSIS_LLM_TIMEOUT', 60)
        )
