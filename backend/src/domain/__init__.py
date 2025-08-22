"""
Domain package - Core business logic and entities.
This layer contains the heart of the business logic with no external dependencies.
"""

# Core Models
from .models import (
    URLInfo,
    ScrapedContent, 
    ScrapingResult,
    ScrapingRequest,
    ContentMetrics,
    ContentType,
    ScrapingStatus
)

# Domain Services
from .services import (
    ContentClassificationService,
    ContentQualityService,
    URLAnalysisService
)

# Domain Exceptions
from .exceptions import (
    DomainError,
    ValidationError,
    ContentQualityError,
    URLSecurityError,
    ContentTooLargeError,
    UnsupportedContentTypeError,
    ScrapingTimeoutError,
    NetworkError,
    ConfigurationError,
    ErrorSeverity
)

__all__ = [
    # Models
    'URLInfo',
    'ScrapedContent',
    'ScrapingResult', 
    'ScrapingRequest',
    'ContentMetrics',
    'ContentType',
    'ScrapingStatus',
    
    # Services
    'ContentClassificationService',
    'ContentQualityService', 
    'URLAnalysisService',
    
    # Exceptions
    'DomainError',
    'ValidationError',
    'ContentQualityError',
    'URLSecurityError',
    'ContentTooLargeError',
    'UnsupportedContentTypeError',
    'ScrapingTimeoutError',
    'NetworkError',
    'ConfigurationError',
    'ErrorSeverity'
]

# Package metadata
__version__ = "1.0.0"
__description__ = "Domain layer for Web Content Analyzer - Business logic and entities"
