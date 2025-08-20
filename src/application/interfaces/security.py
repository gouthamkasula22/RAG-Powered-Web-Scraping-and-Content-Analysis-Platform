"""
Security-related interfaces following SOLID principles.
These interfaces define contracts for security validation and protection.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.domain import URLInfo, ScrapingRequest, ScrapingResult


class IURLValidator(ABC):
    """
    Interface for URL validation and security checking.
    Implements Dependency Inversion Principle - high-level modules depend on abstractions.
    """
    
    @abstractmethod
    def is_valid_url(self, url: str) -> bool:
        """
        Check if URL is syntactically valid and follows proper format.
        
        Args:
            url: URL string to validate
            
        Returns:
            bool: True if URL is valid format
        """
        pass
    
    @abstractmethod
    def is_safe_url(self, url: str) -> bool:
        """
        Check if URL is safe to scrape (prevents SSRF attacks).
        
        Args:
            url: URL string to check for security
            
        Returns:
            bool: True if URL is safe to scrape
        """
        pass
    
    @abstractmethod
    def validate_domain(self, domain: str) -> bool:
        """
        Validate if domain is allowed for scraping.
        
        Args:
            domain: Domain name to validate
            
        Returns:
            bool: True if domain is allowed
        """
        pass
    
    @abstractmethod
    def get_validation_errors(self, url: str) -> List[str]:
        """
        Get detailed validation errors for debugging.
        
        Args:
            url: URL to validate
            
        Returns:
            List[str]: List of validation error messages
        """
        pass


class ISecurityService(ABC):
    """
    Interface for comprehensive security services.
    Provides security orchestration and policy enforcement.
    """
    
    @abstractmethod
    def check_security_policy(self, request: ScrapingRequest) -> Dict[str, Any]:
        """
        Check if scraping request complies with security policies.
        
        Args:
            request: Scraping request to validate
            
        Returns:
            Dict containing security check results
        """
        pass
    
    @abstractmethod
    def sanitize_url(self, url: str) -> str:
        """
        Sanitize URL to remove potential security threats.
        
        Args:
            url: URL to sanitize
            
        Returns:
            str: Sanitized URL
        """
        pass
    
    @abstractmethod
    def is_rate_limited(self, domain: str) -> bool:
        """
        Check if domain is currently rate limited.
        
        Args:
            domain: Domain to check
            
        Returns:
            bool: True if rate limited
        """
        pass


class IScrapingProxy(ABC):
    """
    Interface for secure scraping proxy implementing Proxy Pattern.
    Controls access to scraping operations with security validation.
    """
    
    @abstractmethod
    async def secure_scrape(self, request: ScrapingRequest) -> ScrapingResult:
        """
        Perform secure scraping with all security checks applied.
        
        Args:
            request: Scraping request with parameters
            
        Returns:
            ScrapingResult: Result of secure scraping operation
        """
        pass
    
    @abstractmethod
    def get_security_report(self, url: str) -> Dict[str, Any]:
        """
        Generate security analysis report for URL.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dict containing security analysis
        """
        pass
