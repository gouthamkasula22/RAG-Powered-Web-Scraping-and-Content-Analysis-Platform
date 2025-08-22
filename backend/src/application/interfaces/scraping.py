"""
Web scraping interfaces following SOLID principles.
These interfaces define contracts for content extraction and scraping operations.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from ...domain import ScrapingRequest, ScrapingResult, ScrapedContent


class IWebScraper(ABC):
    """
    Interface for web scraping operations.
    Implements Strategy Pattern for different scraping approaches.
    """
    
    @abstractmethod
    async def scrape_content(self, request: ScrapingRequest) -> ScrapingResult:
        """
        Scrape content from URL with given parameters.
        
        Args:
            request: Scraping request with URL and parameters
            
        Returns:
            ScrapingResult: Result containing scraped content or error info
        """
        pass
    
    @abstractmethod
    def supports_url(self, url: str) -> bool:
        """
        Check if this scraper can handle the given URL.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if scraper can handle this URL
        """
        pass
    
    @abstractmethod
    def get_scraper_name(self) -> str:
        """
        Get name/identifier for this scraper.
        
        Returns:
            str: Scraper name
        """
        pass


class IContentExtractor(ABC):
    """
    Interface for content extraction from HTML.
    Implements Strategy Pattern for different extraction strategies.
    """
    
    @abstractmethod
    def extract_content(self, html: str, url: str) -> Optional[ScrapedContent]:
        """
        Extract structured content from HTML.
        
        Args:
            html: Raw HTML content
            url: Source URL for the content
            
        Returns:
            Optional[ScrapedContent]: Extracted content or None if failed
        """
        pass
    
    @abstractmethod
    def can_extract(self, html: str, url: str) -> bool:
        """
        Check if this extractor can handle the given HTML/URL.
        
        Args:
            html: HTML content to check
            url: Source URL
            
        Returns:
            bool: True if extractor can handle this content
        """
        pass
    
    @abstractmethod
    def get_extractor_name(self) -> str:
        """
        Get name/identifier for this extractor.
        
        Returns:
            str: Extractor name
        """
        pass


class IHTTPClient(ABC):
    """
    Interface for HTTP operations.
    Abstracts HTTP client implementation details.
    """
    
    @abstractmethod
    async def get(self, url: str, headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
        """
        Perform HTTP GET request.
        
        Args:
            url: URL to request
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            
        Returns:
            Dict containing response data
        """
        pass
    
    @abstractmethod
    async def head(self, url: str, headers: Dict[str, str] = None, timeout: int = 10) -> Dict[str, Any]:
        """
        Perform HTTP HEAD request to check URL accessibility.
        
        Args:
            url: URL to check
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            
        Returns:
            Dict containing response headers and status
        """
        pass
