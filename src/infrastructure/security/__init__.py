"""
Security Infrastructure Implementation
Implements security services with SSRF prevention and URL validation.
"""
import logging
import ipaddress
import socket
from urllib.parse import urlparse, urlunparse
from typing import Set, List, Optional, Dict, Any
import re
from src.domain import ValidationError, URLSecurityError, ScrapingRequest
from src.application.interfaces.security import IURLValidator, ISecurityService
from src.application.interfaces.configuration import SecurityConfig


class URLValidator(IURLValidator):
    """
    Concrete implementation of URL validator with SSRF prevention.
    Implements comprehensive security checks for web scraping.
    """
    
    def __init__(self, security_config: SecurityConfig):
        self._config = security_config
        self._logger = logging.getLogger(__name__)
        
        # Private IP ranges for SSRF prevention
        self._private_networks = [
            ipaddress.IPv4Network('10.0.0.0/8'),
            ipaddress.IPv4Network('172.16.0.0/12'),
            ipaddress.IPv4Network('192.168.0.0/16'),
            ipaddress.IPv4Network('127.0.0.0/8'),  # Loopback
            ipaddress.IPv4Network('169.254.0.0/16'),  # Link-local
            ipaddress.IPv6Network('::1/128'),  # IPv6 loopback
            ipaddress.IPv6Network('fc00::/7'),  # IPv6 private
            ipaddress.IPv6Network('fe80::/10'),  # IPv6 link-local
        ]
        
        # Blocked domains and patterns
        self._blocked_patterns = [
            r'localhost',
            r'.*\.local$',
            r'.*\.internal$',
            r'.*\.corp$',
            r'.*\.lan$'
        ]
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is syntactically valid and follows proper format."""
        try:
            return self._validate_url_format(url)
        except Exception:
            return False
    
    def is_safe_url(self, url: str) -> bool:
        """Check if URL is safe to scrape (prevents SSRF attacks)."""
        try:
            # Run the async validation synchronously for the interface
            import asyncio
            return asyncio.run(self._is_safe_url_async(url))
        except Exception:
            return False
    
    def validate_domain(self, domain: str) -> bool:
        """Validate if domain is allowed for scraping."""
        try:
            # Run the async validation synchronously for the interface
            import asyncio
            return asyncio.run(self._validate_domain(domain))
        except Exception:
            return False
    
    def get_validation_errors(self, url: str) -> List[str]:
        """Get detailed validation errors for debugging."""
        errors = []
        try:
            if not self._validate_url_format(url):
                errors.append("Invalid URL format")
                return errors
            
            parsed = urlparse(url)
            
            if not self._validate_scheme(parsed.scheme):
                errors.append(f"Unsupported scheme: {parsed.scheme}")
            
            if not self._validate_port(parsed.port):
                errors.append(f"Blocked port: {parsed.port}")
            
            # Domain validation would need async, so we'll skip detailed IP checks
            if parsed.hostname and parsed.hostname.lower() in self._config.blocked_domains:
                errors.append(f"Blocked domain: {parsed.hostname}")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    async def _is_safe_url_async(self, url: str) -> bool:
        """Async version of safety check."""
        try:
            parsed = urlparse(url)
            return (await self._validate_domain(parsed.hostname) and 
                    await self._validate_ip_address(parsed.hostname))
        except Exception:
            return False
    
    async def validate_url(self, url: str) -> bool:
        """
        Validate URL with comprehensive security checks.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL is valid and safe
            
        Raises:
            ValidationError: If URL format is invalid
            SecurityError: If URL violates security policies
        """
        try:
            self._logger.debug(f"Validating URL: {url}")
            
            # Basic format validation
            if not self._validate_url_format(url):
                raise ValidationError(
                    message="Invalid URL format",
                    context={"field": "url", "value": url}
                )
            
            parsed = urlparse(url)
            
            # Scheme validation
            if not self._validate_scheme(parsed.scheme):
                raise URLSecurityError(
                    message=f"Unsupported URL scheme: {parsed.scheme}",
                    context={"url": url, "scheme": parsed.scheme}
                )
            
            # Port validation
            if not self._validate_port(parsed.port):
                raise URLSecurityError(
                    message=f"Blocked port: {parsed.port}",
                    context={"url": url, "port": parsed.port}
                )
            
            # Domain validation
            if not await self._validate_domain(parsed.hostname):
                raise URLSecurityError(
                    message=f"Blocked domain: {parsed.hostname}",
                    context={"url": url, "domain": parsed.hostname}
                )
            
            # IP address validation (SSRF prevention)
            if not await self._validate_ip_address(parsed.hostname):
                raise URLSecurityError(
                    message=f"Private/local IP address blocked: {parsed.hostname}",
                    context={"url": url, "hostname": parsed.hostname}
                )
            
            self._logger.debug(f"URL validation passed: {url}")
            return True
            
        except (ValidationError, URLSecurityError):
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error validating URL {url}: {e}")
            raise URLSecurityError(
                message="URL validation failed due to unexpected error",
                context={"url": url, "error": str(e)}
            )
    
    def sanitize_url(self, url: str) -> str:
        """
        Sanitize URL by removing potentially dangerous components.
        
        Args:
            url: URL to sanitize
            
        Returns:
            str: Sanitized URL
        """
        try:
            parsed = urlparse(url.strip())
            
            # Remove fragment and clean query parameters
            sanitized_query = self._sanitize_query_params(parsed.query)
            
            # Rebuild URL without fragment
            sanitized = urlunparse((
                parsed.scheme,
                parsed.netloc.lower(),  # Normalize domain case
                parsed.path,
                parsed.params,
                sanitized_query,
                ''  # Remove fragment
            ))
            
            self._logger.debug(f"URL sanitized: {url} -> {sanitized}")
            return sanitized
            
        except Exception as e:
            self._logger.warning(f"Failed to sanitize URL {url}: {e}")
            return url
    
    def get_domain_from_url(self, url: str) -> Optional[str]:
        """
        Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Optional[str]: Domain or None if extraction fails
        """
        try:
            parsed = urlparse(url)
            return parsed.hostname.lower() if parsed.hostname else None
        except Exception:
            return None
    
    def _validate_url_format(self, url: str) -> bool:
        """Validate basic URL format."""
        if not url or not isinstance(url, str):
            return False
        
        if len(url) > 2048:  # Maximum reasonable URL length
            return False
        
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    def _validate_scheme(self, scheme: str) -> bool:
        """Validate URL scheme."""
        allowed_schemes = {'http', 'https'}
        return scheme.lower() in allowed_schemes
    
    def _validate_port(self, port: Optional[int]) -> bool:
        """Validate port number."""
        if port is None:
            return True  # Default ports are OK
        
        return port in self._config.allowed_ports
    
    async def _validate_domain(self, hostname: str) -> bool:
        """Validate domain name."""
        if not hostname:
            return False
        
        hostname = hostname.lower()
        
        # Check blocked domains
        if hostname in self._config.blocked_domains:
            return False
        
        # Check allowed domains (if specified)
        if self._config.allowed_domains and hostname not in self._config.allowed_domains:
            return False
        
        # Check blocked patterns
        for pattern in self._blocked_patterns:
            if re.match(pattern, hostname):
                return False
        
        return True
    
    async def _validate_ip_address(self, hostname: str) -> bool:
        """Validate IP address to prevent SSRF attacks."""
        if not hostname:
            return False
        
        try:
            # Try to resolve hostname to IP
            ip_addresses = await self._resolve_hostname(hostname)
            
            for ip_str in ip_addresses:
                try:
                    ip = ipaddress.ip_address(ip_str)
                    
                    # Check if IP is in private/local ranges
                    if self._config.block_private_ips:
                        for network in self._private_networks:
                            if ip in network:
                                self._logger.warning(f"Blocked private IP: {ip} for hostname {hostname}")
                                return False
                    
                    # Check for localhost variations
                    if self._config.block_local_networks:
                        if ip.is_loopback or ip.is_link_local or ip.is_private:
                            self._logger.warning(f"Blocked local IP: {ip} for hostname {hostname}")
                            return False
                
                except ValueError:
                    # Invalid IP address
                    continue
            
            return True
            
        except Exception as e:
            self._logger.warning(f"Could not resolve hostname {hostname}: {e}")
            # If we can't resolve, allow it (DNS might be down)
            return True
    
    async def _resolve_hostname(self, hostname: str) -> List[str]:
        """Resolve hostname to IP addresses."""
        try:
            # Use socket.getaddrinfo for async-compatible DNS resolution
            import asyncio
            loop = asyncio.get_event_loop()
            
            def resolve():
                try:
                    addr_info = socket.getaddrinfo(hostname, None)
                    return [info[4][0] for info in addr_info]
                except socket.gaierror:
                    return []
            
            return await loop.run_in_executor(None, resolve)
            
        except Exception:
            return []
    
    def _sanitize_query_params(self, query: str) -> str:
        """Sanitize query parameters."""
        if not query:
            return query
        
        # Remove potentially dangerous parameters
        dangerous_params = {'callback', 'jsonp', 'redirect', 'url', 'goto'}
        
        try:
            from urllib.parse import parse_qs, urlencode
            params = parse_qs(query, keep_blank_values=False)
            
            # Filter out dangerous parameters
            safe_params = {k: v for k, v in params.items() 
                          if k.lower() not in dangerous_params}
            
            return urlencode(safe_params, doseq=True)
            
        except Exception:
            # If parsing fails, return empty query
            return ''


class SecurityService(ISecurityService):
    """
    Main security service implementing comprehensive security policies.
    Orchestrates security validation and policy enforcement.
    """
    
    def __init__(self, url_validator: IURLValidator, security_config: SecurityConfig):
        self._url_validator = url_validator
        self._config = security_config
        self._logger = logging.getLogger(__name__)
        
        # Rate limiting storage (in production, use Redis or similar)
        self._rate_limits = {}
    
    def check_security_policy(self, request: ScrapingRequest) -> Dict[str, Any]:
        """Check if scraping request complies with security policies."""
        policy_results = {
            "compliant": True,
            "violations": [],
            "warnings": []
        }
        
        try:
            # URL validation
            if not self._url_validator.is_valid_url(request.url):
                policy_results["compliant"] = False
                policy_results["violations"].append("Invalid URL format")
            
            if not self._url_validator.is_safe_url(request.url):
                policy_results["compliant"] = False
                policy_results["violations"].append("URL violates security policies")
            
            # Request parameter validation
            if request.timeout and request.timeout > 300:
                policy_results["compliant"] = False
                policy_results["violations"].append("Timeout exceeds maximum allowed")
            
            # Rate limiting check
            domain = self._extract_domain(request.url)
            if self.is_rate_limited(domain):
                policy_results["compliant"] = False
                policy_results["violations"].append("Domain is rate limited")
            
        except Exception as e:
            policy_results["compliant"] = False
            policy_results["violations"].append(f"Policy check error: {str(e)}")
        
        return policy_results
    
    def sanitize_url(self, url: str) -> str:
        """Sanitize URL to remove potential security threats."""
        # Delegate to URL validator's sanitize method
        return self._url_validator.sanitize_url(url)
    
    def is_rate_limited(self, domain: str) -> bool:
        """Check if domain is currently rate limited."""
        return not self.check_rate_limit(domain, limit=60, window=3600)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for rate limiting."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return "unknown"
    
    async def validate_url(self, url: str) -> bool:
        """
        Comprehensive URL validation with security checks.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL passes all security checks
        """
        self._logger.debug(f"Security service validating URL: {url}")
        
        # Delegate to URL validator
        result = await self._url_validator.validate_url(url)
        
        # Additional security checks could be added here
        # (rate limiting, reputation checks, etc.)
        
        return result
    
    async def check_rate_limit(self, identifier: str, limit: int = 100, window: int = 3600) -> bool:
        """
        Check rate limiting for requests.
        
        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            limit: Maximum requests per window
            window: Time window in seconds
            
        Returns:
            bool: True if within rate limit
        """
        import time
        current_time = time.time()
        
        if identifier not in self._rate_limits:
            self._rate_limits[identifier] = []
        
        # Clean old entries
        self._rate_limits[identifier] = [
            timestamp for timestamp in self._rate_limits[identifier]
            if current_time - timestamp < window
        ]
        
        # Check limit
        if len(self._rate_limits[identifier]) >= limit:
            self._logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Add current request
        self._rate_limits[identifier].append(current_time)
        return True
    
    async def validate_content_type(self, content_type: str) -> bool:
        """
        Validate content type is safe for processing.
        
        Args:
            content_type: MIME content type
            
        Returns:
            bool: True if content type is safe
        """
        if not content_type:
            return False
        
        safe_types = {
            'text/html',
            'application/xhtml+xml',
            'text/plain',
            'application/xml',
            'text/xml'
        }
        
        # Normalize content type (remove charset, etc.)
        normalized = content_type.split(';')[0].strip().lower()
        
        is_safe = normalized in safe_types
        
        if not is_safe:
            self._logger.warning(f"Unsafe content type blocked: {content_type}")
        
        return is_safe
    
    async def validate_response_size(self, size: int) -> bool:
        """
        Validate response size is within acceptable limits.
        
        Args:
            size: Response size in bytes
            
        Returns:
            bool: True if size is acceptable
        """
        max_size = 50 * 1024 * 1024  # 50MB default limit
        
        if size > max_size:
            self._logger.warning(f"Response size too large: {size} bytes")
            return False
        
        return True
    
    def get_security_headers(self) -> dict:
        """
        Get security headers for HTTP requests.
        
        Returns:
            dict: Security headers
        """
        return {
            'User-Agent': 'WebContentAnalyzer/1.0 (Security Enabled)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',  # Do Not Track
            'Connection': 'close',  # Prevent connection reuse
            'Upgrade-Insecure-Requests': '1'
        }
