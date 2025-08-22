"""
Production web scraper implementation using requests and BeautifulSoup
"""
import asyncio
import time
from datetime import datetime
from typing import Optional
import logging
import re

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPING_DEPS = True
except ImportError:
    HAS_SCRAPING_DEPS = False

from src.domain.models import (
    ScrapingResult, URLInfo, ContentMetrics, ScrapedContent, 
    ContentType, ScrapingStatus, ScrapingRequest
)
from src.application.interfaces.scraping import IWebScraper

logger = logging.getLogger(__name__)


class ProductionWebScraper(IWebScraper):
    """Production web scraper using requests and BeautifulSoup"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set realistic user agent
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    async def scrape_content(self, request: ScrapingRequest) -> ScrapingResult:
        """Scrape content from a URL"""
        if not HAS_SCRAPING_DEPS:
            return self._create_fallback_result(str(request.url), "Missing dependencies: requests, beautifulsoup4")
        
        start_time = time.time()
        url = str(request.url)
        
        try:
            logger.info(f"🌐 Scraping URL: {url}")
            
            # Validate URL
            url_info = URLInfo.from_url(url)
            
            # Perform HTTP request with retries
            response = None
            last_error = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self.session.get(url, timeout=self.timeout, allow_redirects=True)
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    last_error = e
                    if attempt < self.max_retries:
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        raise
            
            if not response:
                raise Exception(f"Failed to get response after {self.max_retries + 1} attempts")
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            scraped_content = self._extract_content(soup, url_info)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Successfully scraped {url} in {processing_time:.2f}s")
            
            return ScrapingResult(
                content=scraped_content,
                status=ScrapingStatus.SUCCESS,
                error_message=None,
                processing_time_seconds=processing_time,
                attempt_count=1
            )
            
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {self.timeout}s"
            logger.error(f"❌ {error_msg} for {url}")
            return self._create_error_result(url, ScrapingStatus.TIMEOUT, error_msg, time.time() - start_time)
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 0
            if status_code == 404:
                status = ScrapingStatus.NOT_FOUND
                error_msg = f"Page not found (HTTP 404): The requested URL does not exist"
            elif status_code == 403:
                status = ScrapingStatus.FORBIDDEN
                error_msg = f"Access forbidden (HTTP 403): The server is refusing access to this content"
            elif status_code == 503:
                status = ScrapingStatus.FAILED
                error_msg = f"Service unavailable (HTTP 503): The website is temporarily blocking requests. This is common with e-commerce sites like Amazon. Try a different URL or wait before retrying."
            elif status_code == 429:
                status = ScrapingStatus.FAILED
                error_msg = f"Rate limited (HTTP 429): Too many requests. The website is blocking due to high request frequency."
            elif status_code >= 500:
                status = ScrapingStatus.FAILED
                error_msg = f"Server error (HTTP {status_code}): The website's server is experiencing issues"
            elif status_code >= 400:
                status = ScrapingStatus.FAILED
                error_msg = f"Client error (HTTP {status_code}): {str(e)}"
            else:
                status = ScrapingStatus.FAILED
                error_msg = f"HTTP {status_code}: {str(e)}"
            
            logger.error(f"❌ {error_msg} for {url}")
            return self._create_error_result(url, status, error_msg, time.time() - start_time)
            
        except Exception as e:
            error_msg = f"Scraping failed: {str(e)}"
            logger.error(f"❌ {error_msg} for {url}")
            return self._create_error_result(url, ScrapingStatus.FAILED, error_msg, time.time() - start_time)
    
    def _extract_content(self, soup: BeautifulSoup, url_info: URLInfo) -> ScrapedContent:
        """Extract structured content from BeautifulSoup object"""
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else url_info.domain
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Extract main content
        main_content = ""
        
        # Try to find main content areas
        content_selectors = [
            'main', 'article', '[role="main"]', '.content', '.main-content',
            '.post-content', '.entry-content', '.article-content', 'body'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem.get_text(separator=' ', strip=True)
                if len(main_content.split()) > 50:  # Ensure substantial content
                    break
        
        # Fallback to body text if no main content found
        if not main_content or len(main_content.split()) < 20:
            main_content = soup.get_text(separator=' ', strip=True)
        
        # Extract headings
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = heading.get_text().strip()
            if text:
                headings.append(text)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http'):
                links.append(href)
        
        # Extract meta description
        meta_desc = None
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_desc_tag and meta_desc_tag.get('content'):
            meta_desc = meta_desc_tag['content'].strip()
        
        # Extract meta keywords
        meta_keywords = []
        meta_keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords_tag and meta_keywords_tag.get('content'):
            keywords = meta_keywords_tag['content'].strip()
            meta_keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        
        # Determine content type
        content_type = self._determine_content_type(soup, url_info, title)
        
        # Calculate metrics
        metrics = ContentMetrics.calculate(
            content=main_content,
            links=links,
            headings=headings
        )
        
        return ScrapedContent(
            url_info=url_info,
            title=title,
            headings=headings,
            main_content=main_content,
            links=links,
            meta_description=meta_desc,
            meta_keywords=meta_keywords,
            content_type=content_type,
            metrics=metrics,
            scraped_at=datetime.now(),
            status=ScrapingStatus.SUCCESS
        )
    
    def _determine_content_type(self, soup: BeautifulSoup, url_info: URLInfo, title: str) -> ContentType:
        """Determine content type based on page structure and URL"""
        
        # Check for article markers
        if soup.find('article') or soup.find('[role="article"]'):
            return ContentType.ARTICLE
        
        # Check URL patterns
        path = url_info.path.lower()
        if any(keyword in path for keyword in ['/blog/', '/post/', '/article/', '/news/']):
            if '/blog/' in path:
                return ContentType.BLOG_POST
            elif '/news/' in path:
                return ContentType.NEWS
            else:
                return ContentType.ARTICLE
        
        # Check for product pages
        if any(keyword in path for keyword in ['/product/', '/item/', '/shop/']):
            return ContentType.PRODUCT
        
        # Check for documentation
        if any(keyword in path for keyword in ['/docs/', '/documentation/', '/api/', '/guide/']):
            return ContentType.DOCUMENTATION
        
        # Check if it's a homepage
        if path in ['/', '/home', '/index'] or len(path.strip('/')) == 0:
            return ContentType.HOMEPAGE
        
        # Default to article
        return ContentType.ARTICLE
    
    def _create_error_result(self, url: str, status: ScrapingStatus, error_msg: str, processing_time: float) -> ScrapingResult:
        """Create error result for failed scraping"""
        return ScrapingResult(
            content=None,
            status=status,
            error_message=error_msg,
            processing_time_seconds=processing_time,
            attempt_count=self.max_retries + 1
        )
    
    def _create_fallback_result(self, url: str, error_msg: str) -> ScrapingResult:
        """Create fallback result when dependencies are missing"""
        return ScrapingResult(
            content=None,
            status=ScrapingStatus.FAILED,
            error_message=error_msg,
            processing_time_seconds=0.0,
            attempt_count=1
        )
    
    async def secure_scrape(self, url: str) -> ScrapingResult:
        """Legacy method for backwards compatibility"""
        request = ScrapingRequest(url=url)
        return await self.scrape_content(request)
