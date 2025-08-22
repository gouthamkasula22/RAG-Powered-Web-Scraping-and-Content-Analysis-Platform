"""
Configuration and settings interfaces following SOLID principles.
These interfaces define contracts for configuration management and settings.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ScrapingConfig:
    """Configuration for scraping operations."""
    user_agent: str = "WebContentAnalyzer/1.0"
    timeout: int = 30
    max_retries: int = 3
    delay_between_requests: float = 1.0
    respect_robots_txt: bool = True
    max_content_size: int = 10_000_000  # 10MB
    allowed_content_types: List[str] = None
    
    def __post_init__(self):
        if self.allowed_content_types is None:
            self.allowed_content_types = ["text/html", "application/xhtml+xml"]


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    block_private_ips: bool = True
    block_local_networks: bool = True
    allowed_domains: List[str] = None
    blocked_domains: List[str] = None
    max_redirects: int = 5
    validate_ssl: bool = True
    allowed_ports: List[int] = None
    
    def __post_init__(self):
        if self.allowed_domains is None:
            self.allowed_domains = []
        if self.blocked_domains is None:
            self.blocked_domains = []
        if self.allowed_ports is None:
            self.allowed_ports = [80, 443, 8000, 8080]


@dataclass
class AnalysisConfig:
    """Configuration for analysis operations."""
    enable_sentiment_analysis: bool = True
    enable_keyword_extraction: bool = True
    enable_content_summarization: bool = True
    max_keywords: int = 10
    summary_max_length: int = 500
    llm_model: str = "gpt-3.5-turbo"
    llm_timeout: int = 60


class IConfigurationService(ABC):
    """
    Interface for configuration management.
    Implements Single Responsibility Principle for configuration handling.
    """
    
    @abstractmethod
    def get_scraping_config(self) -> ScrapingConfig:
        """
        Get scraping configuration.
        
        Returns:
            ScrapingConfig: Current scraping configuration
        """
        pass
    
    @abstractmethod
    def get_security_config(self) -> SecurityConfig:
        """
        Get security configuration.
        
        Returns:
            SecurityConfig: Current security configuration
        """
        pass
    
    @abstractmethod
    def get_analysis_config(self) -> AnalysisConfig:
        """
        Get analysis configuration.
        
        Returns:
            AnalysisConfig: Current analysis configuration
        """
        pass
    
    @abstractmethod
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            section: Configuration section (scraping, security, analysis)
            key: Configuration key
            value: New value
        """
        pass
    
    @abstractmethod
    def reload_config(self) -> None:
        """
        Reload configuration from source (file, env vars, etc.).
        """
        pass


class IEnvironmentService(ABC):
    """
    Interface for environment variable management.
    Abstracts environment configuration details.
    """
    
    @abstractmethod
    def get_string(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get string environment variable.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            
        Returns:
            Environment variable value or default
        """
        pass
    
    @abstractmethod
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get integer environment variable.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            
        Returns:
            Environment variable value as int or default
        """
        pass
    
    @abstractmethod
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        Get boolean environment variable.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            
        Returns:
            Environment variable value as bool or default
        """
        pass
    
    @abstractmethod
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        Get float environment variable.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            
        Returns:
            Environment variable value as float or default
        """
        pass
    
    @abstractmethod
    def get_list(self, key: str, separator: str = ",", default: Optional[List[str]] = None) -> Optional[List[str]]:
        """
        Get list environment variable.
        
        Args:
            key: Environment variable key
            separator: List item separator
            default: Default value if not found
            
        Returns:
            Environment variable value as list or default
        """
        pass
    
    @abstractmethod
    def is_development(self) -> bool:
        """
        Check if running in development environment.
        
        Returns:
            bool: True if development environment
        """
        pass
    
    @abstractmethod
    def is_production(self) -> bool:
        """
        Check if running in production environment.
        
        Returns:
            bool: True if production environment
        """
        pass


class ILoggingService(ABC):
    """
    Interface for logging service.
    Abstracts logging implementation details.
    """
    
    @abstractmethod
    def configure_logging(self, config: Dict[str, Any]) -> None:
        """
        Configure logging with given settings.
        
        Args:
            config: Logging configuration
        """
        pass
    
    @abstractmethod
    def get_logger(self, name: str) -> Any:
        """
        Get logger instance for given name.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        pass
    
    @abstractmethod
    def set_log_level(self, level: str) -> None:
        """
        Set global log level.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        pass
