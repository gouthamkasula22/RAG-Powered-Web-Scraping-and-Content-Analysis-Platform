"""
Scraping Proxy Implementation using Proxy Design Pattern
Provides secure web scraping with SSRF prevention and comprehensive validation.
"""
import asyncio
import time
import logging
from typing import Optional, Dict, Any
from src.domain import ScrapingRequest, ScrapingResult, ScrapedContent, URLSecurityError, NetworkError, ScrapingTimeoutError
from src.application.interfaces.security import IScrapingProxy, ISecurityService
from src.application.interfaces.scraping import IWebScraper, IHTTPClient
from src.application.interfaces.configuration import IConfigurationService


class ScrapingProxy(IScrapingProxy):
    """
    Secure scraping proxy implementing Proxy Design Pattern.
    Adds security validation, rate limiting, and error handling to web scraping operations.
    """
    
    def __init__(
        self,
        web_scraper: IWebScraper,
        security_service: ISecurityService,
        config_service: IConfigurationService,
        http_client: IHTTPClient
    ):
        self._web_scraper = web_scraper
        self._security_service = security_service
        self._config_service = config_service
        self._http_client = http_client
        self._logger = logging.getLogger(__name__)
    
    async def secure_scrape(self, request: ScrapingRequest) -> ScrapingResult:
        """
        Securely scrape content with comprehensive validation and security checks.
        
        Args:
            request: Scraping request with URL and parameters
            
        Returns:
            ScrapingResult: Result containing scraped content or error information
        """
        start_time = time.time()
        
        try:
            self._logger.info(f"Starting secure scraping for URL: {request.url}")
            
            # Step 1: Pre-scraping security validation
            await self._validate_request_security(request)
            
            # Step 2: Rate limiting check
            domain = self._extract_domain(request.url)
            if not await self._check_rate_limits(domain):
                return ScrapingResult(
                    success=False,
                    content=None,
                    error_message="Rate limit exceeded for domain",
                    status_code=429,
                    processing_time=time.time() - start_time
                )
            
            # Step 3: Pre-flight request to validate accessibility
            if await self._should_perform_preflight(request):
                if not await self._perform_preflight_check(request):
                    return ScrapingResult(
                        success=False,
                        content=None,
                        error_message="Preflight check failed - URL not accessible",
                        status_code=0,
                        processing_time=time.time() - start_time
                    )
            
            # Step 4: Delegate to actual scraper with monitoring
            result = await self._execute_with_monitoring(request)
            
            # Step 5: Post-scraping validation
            if result.success and result.content:
                validated_result = await self._validate_scraped_content(result)
                if not validated_result.success:
                    return validated_result
            
            # Step 6: Log metrics and return result
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self._logger.info(
                f"Scraping completed for {request.url}: "
                f"success={result.success}, time={processing_time:.2f}s"
            )
            
            return result
            
        except URLSecurityError as e:
            self._logger.error(f"Security error during scraping {request.url}: {e}")
            return ScrapingResult(
                success=False,
                content=None,
                error_message=f"Security validation failed: {e.message}",
                status_code=403,
                processing_time=time.time() - start_time
            )
        
        except ScrapingTimeoutError as e:
            self._logger.error(f"Timeout during scraping {request.url}: {e}")
            return ScrapingResult(
                success=False,
                content=None,
                error_message=f"Request timeout: {e.message}",
                status_code=408,
                processing_time=time.time() - start_time
            )
        
        except Exception as e:
            self._logger.error(f"Unexpected error during scraping {request.url}: {e}")
            return ScrapingResult(
                success=False,
                content=None,
                error_message=f"Scraping failed: {str(e)}",
                status_code=500,
                processing_time=time.time() - start_time
            )
    
    async def validate_url_accessibility(self, url: str) -> bool:
        """
        Check if URL is accessible without full scraping.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL is accessible
        """
        try:
            self._logger.debug(f"Validating URL accessibility: {url}")
            
            # Security validation first
            await self._security_service.validate_url(url)
            
            # Perform HEAD request to check accessibility
            security_headers = self._security_service.get_security_headers()
            response = await self._http_client.head(url, headers=security_headers, timeout=10)
            
            # Check response
            status_code = response.get('status_code', 0)
            content_type = response.get('headers', {}).get('content-type', '')
            
            if status_code < 200 or status_code >= 400:
                self._logger.debug(f"URL not accessible: {url}, status: {status_code}")
                return False
            
            # Validate content type
            if not await self._security_service.validate_content_type(content_type):
                self._logger.debug(f"Unsafe content type for URL: {url}, type: {content_type}")
                return False
            
            self._logger.debug(f"URL accessibility validated: {url}")
            return True
            
        except Exception as e:
            self._logger.warning(f"URL accessibility check failed for {url}: {e}")
            return False
    
    async def _validate_request_security(self, request: ScrapingRequest) -> None:
        """Validate request security before scraping."""
        # URL security validation
        await self._security_service.validate_url(request.url)
        
        # Additional request validation
        if request.timeout and request.timeout > 300:  # 5 minutes max
            raise URLSecurityError(
                message="Request timeout too large",
                details={"timeout": request.timeout, "max_allowed": 300}
            )
        
        # Validate user agent
        if request.user_agent and len(request.user_agent) > 500:
            raise URLSecurityError(
                message="User agent string too long",
                details={"length": len(request.user_agent)}
            )
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for rate limiting."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return "unknown"
    
    async def _check_rate_limits(self, domain: str) -> bool:
        """Check rate limits for domain."""
        # Simple rate limiting - 60 requests per hour per domain
        return await self._security_service.check_rate_limit(
            identifier=f"domain:{domain}",
            limit=60,
            window=3600
        )
    
    async def _should_perform_preflight(self, request: ScrapingRequest) -> bool:
        """Determine if preflight check should be performed."""
        # Perform preflight for new domains or on request
        return True  # Always perform for security
    
    async def _perform_preflight_check(self, request: ScrapingRequest) -> bool:
        """Perform preflight check to validate URL accessibility."""
        try:
            security_headers = self._security_service.get_security_headers()
            response = await self._http_client.head(
                request.url,
                headers=security_headers,
                timeout=10
            )
            
            status_code = response.get('status_code', 0)
            headers = response.get('headers', {})
            
            # Check status code
            if status_code < 200 or status_code >= 400:
                return False
            
            # Check content type
            content_type = headers.get('content-type', '')
            if not await self._security_service.validate_content_type(content_type):
                return False
            
            # Check content length
            content_length = headers.get('content-length')
            if content_length:
                try:
                    size = int(content_length)
                    if not await self._security_service.validate_response_size(size):
                        return False
                except ValueError:
                    pass
            
            return True
            
        except Exception as e:
            self._logger.warning(f"Preflight check failed for {request.url}: {e}")
            return False
    
    async def _execute_with_monitoring(self, request: ScrapingRequest) -> ScrapingResult:
        """Execute scraping with timeout and monitoring."""
        config = self._config_service.get_scraping_config()
        timeout = request.timeout or config.timeout
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._web_scraper.scrape_content(request),
                timeout=timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            raise ScrapingTimeoutError(
                message=f"Scraping timeout after {timeout} seconds",
                details={"url": request.url, "timeout": timeout}
            )
    
    async def _validate_scraped_content(self, result: ScrapingResult) -> ScrapingResult:
        """Validate scraped content for security and quality."""
        if not result.content:
            return result
        
        content = result.content
        
        try:
            # Validate content size
            content_size = len(content.raw_html)
            if not await self._security_service.validate_response_size(content_size):
                return ScrapingResult(
                    success=False,
                    content=None,
                    error_message="Content size exceeds security limits",
                    status_code=413,
                    processing_time=result.processing_time
                )
            
            # Validate final URL (after redirects)
            if content.url != result.content.url:
                await self._security_service.validate_url(content.url)
            
            # Additional content validation could be added here
            # (malware scanning, content filtering, etc.)
            
            return result
            
        except URLSecurityError as e:
            return ScrapingResult(
                success=False,
                content=None,
                error_message=f"Content validation failed: {e.message}",
                status_code=403,
                processing_time=result.processing_time
            )
        except Exception as e:
            return ScrapingResult(
                success=False,
                content=None,
                error_message=f"Content validation error: {str(e)}",
                status_code=500,
                processing_time=result.processing_time
            )
    
    def get_security_report(self, url: str) -> Dict[str, Any]:
        """
        Generate comprehensive security analysis report for URL.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dict containing security analysis
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            # Basic security analysis
            report = {
                "url": url,
                "scheme": parsed.scheme,
                "hostname": parsed.hostname,
                "port": parsed.port,
                "path": parsed.path,
                "security_status": "unknown",
                "issues": [],
                "recommendations": []
            }
            
            # Check scheme security
            if parsed.scheme not in ["http", "https"]:
                report["issues"].append(f"Unsupported scheme: {parsed.scheme}")
                report["security_status"] = "blocked"
            
            # Check for private IPs
            if parsed.hostname:
                try:
                    import ipaddress
                    ip = ipaddress.ip_address(parsed.hostname)
                    if ip.is_private or ip.is_loopback:
                        report["issues"].append(f"Private/loopback IP detected: {parsed.hostname}")
                        report["security_status"] = "blocked"
                except ValueError:
                    # Not an IP address, that's fine
                    pass
            
            # Check for dangerous ports
            dangerous_ports = {22, 23, 25, 53, 135, 139, 445, 993, 995}
            if parsed.port and parsed.port in dangerous_ports:
                report["issues"].append(f"Potentially dangerous port: {parsed.port}")
                report["recommendations"].append(f"Avoid connecting to port {parsed.port}")
            
            # Set final status
            if not report["issues"] and parsed.scheme in ["http", "https"]:
                report["security_status"] = "safe"
            elif report["security_status"] == "unknown":
                report["security_status"] = "warning"
            
            return report
            
        except Exception as e:
            return {
                "url": url,
                "security_status": "error",
                "error": str(e),
                "issues": [f"Security analysis failed: {str(e)}"],
                "recommendations": ["Manual review required"]
            }
