"""
Domain-specific exceptions with comprehensive error handling.
These represent business rule violations and domain errors.
"""
from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for proper error categorization"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DomainError(Exception):
    """
    Base exception for all domain errors with enhanced error handling.
    Provides structured error information for better debugging and monitoring.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        inner_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.context = context or {}
        self.inner_exception = inner_exception
        
    def __str__(self) -> str:
        """Enhanced string representation with context"""
        base_msg = f"[{self.error_code}] {self.message}"
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            base_msg += f" (Context: {context_str})"
            
        if self.inner_exception:
            base_msg += f" (Caused by: {self.inner_exception})"
            
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "context": self.context,
            "inner_exception": str(self.inner_exception) if self.inner_exception else None
        }


class ValidationError(DomainError):
    """
    Raised when domain validation rules are violated.
    Used for business rule validation failures.
    """
    
    def __init__(
        self, 
        message: str, 
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if field_name:
            context['field_name'] = field_name
        if field_value is not None:
            context['field_value'] = str(field_value)
        
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'DOMAIN_VALIDATION_ERROR')
        
        super().__init__(message, **kwargs)


class ContentQualityError(DomainError):
    """
    Raised when content doesn't meet quality standards.
    Includes specific quality metrics that failed.
    """
    
    def __init__(
        self, 
        message: str,
        quality_metrics: Optional[Dict[str, Any]] = None,
        minimum_standards: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if quality_metrics:
            context['quality_metrics'] = quality_metrics
        if minimum_standards:
            context['minimum_standards'] = minimum_standards
        
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'CONTENT_QUALITY_ERROR')
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.LOW)
        
        super().__init__(message, **kwargs)


class URLSecurityError(DomainError):
    """
    Raised when URL poses security risks (SSRF, malicious domains, etc.).
    Critical security error that should be monitored.
    """
    
    def __init__(
        self, 
        message: str,
        url: Optional[str] = None,
        security_check_failed: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if url:
            context['url'] = url
        if security_check_failed:
            context['security_check_failed'] = security_check_failed
        
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'URL_SECURITY_ERROR')
        kwargs['severity'] = ErrorSeverity.CRITICAL  # Always critical for security
        
        super().__init__(message, **kwargs)


class ContentTooLargeError(DomainError):
    """
    Raised when content exceeds size limits.
    Helps prevent memory and performance issues.
    """
    
    def __init__(
        self, 
        message: str,
        content_size: Optional[int] = None,
        size_limit: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if content_size is not None:
            context['content_size'] = content_size
        if size_limit is not None:
            context['size_limit'] = size_limit
        
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'CONTENT_TOO_LARGE_ERROR')
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.MEDIUM)
        
        super().__init__(message, **kwargs)


class UnsupportedContentTypeError(DomainError):
    """
    Raised when content type is not supported for analysis.
    Indicates content that cannot be processed by the system.
    """
    
    def __init__(
        self, 
        message: str,
        content_type: Optional[str] = None,
        supported_types: Optional[list] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if content_type:
            context['content_type'] = content_type
        if supported_types:
            context['supported_types'] = supported_types
        
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'UNSUPPORTED_CONTENT_TYPE_ERROR')
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.LOW)
        
        super().__init__(message, **kwargs)


class ScrapingTimeoutError(DomainError):
    """
    Raised when scraping operations timeout.
    Indicates network or performance issues.
    """
    
    def __init__(
        self, 
        message: str,
        timeout_seconds: Optional[float] = None,
        url: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if timeout_seconds is not None:
            context['timeout_seconds'] = timeout_seconds
        if url:
            context['url'] = url
        
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'SCRAPING_TIMEOUT_ERROR')
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.MEDIUM)
        
        super().__init__(message, **kwargs)


class NetworkError(DomainError):
    """
    Raised when network-related errors occur during scraping.
    Covers DNS resolution, connection failures, etc.
    """
    
    def __init__(
        self, 
        message: str,
        url: Optional[str] = None,
        network_error_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if url:
            context['url'] = url
        if network_error_type:
            context['network_error_type'] = network_error_type
        
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'NETWORK_ERROR')
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.MEDIUM)
        
        super().__init__(message, **kwargs)


class ConfigurationError(DomainError):
    """
    Raised when configuration issues are detected.
    Indicates system setup or configuration problems.
    """
    
    def __init__(
        self, 
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = str(config_value)
        
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'CONFIGURATION_ERROR')
        kwargs['severity'] = ErrorSeverity.HIGH  # Config errors are serious
        
        super().__init__(message, **kwargs)


class LLMProviderError(DomainError):
    """
    Raised when LLM provider operations fail.
    Includes provider-specific error details.
    """
    
    def __init__(
        self, 
        message: str,
        provider_name: Optional[str] = None,
        provider_error_code: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if provider_name:
            context['provider_name'] = provider_name
        if provider_error_code:
            context['provider_error_code'] = provider_error_code
        
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'LLM_PROVIDER_ERROR')
        kwargs['severity'] = ErrorSeverity.HIGH
        
        super().__init__(message, **kwargs)


class LLMAnalysisError(DomainError):
    """
    Raised when LLM content analysis fails.
    Includes analysis-specific context.
    """
    
    def __init__(
        self, 
        message: str,
        analysis_type: Optional[str] = None,
        content_length: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if analysis_type:
            context['analysis_type'] = analysis_type
        if content_length:
            context['content_length'] = content_length
        
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'LLM_ANALYSIS_ERROR')
        
        super().__init__(message, **kwargs)
