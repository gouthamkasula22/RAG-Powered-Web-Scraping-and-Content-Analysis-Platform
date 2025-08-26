"""
Web Scraping Implementation using BeautifulSoup
Provides content extraction from HTML with robust error handling.
"""
import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse
from src.domain import ScrapingRequest, ScrapingResult, ScrapedContent, NetworkError
from src.application.interfaces.scraping import IWebScraper, IHTTPClient, IContentExtractor


class HTTPClient(IHTTPClient):
    """
    Async HTTP client implementation using aiohttp.
    Provides robust HTTP operations with proper error handling.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=True  # Enable SSL verification
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                trust_env=True  # Use system proxy settings
            )
        return self._session
    
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
        session = await self._get_session()
        
        try:
            self._logger.debug(f"HTTP GET: {url}")
            
            custom_timeout = aiohttp.ClientTimeout(total=timeout)
            
            async with session.get(url, headers=headers, timeout=custom_timeout) as response:
                content = await response.read()
                
                # Get encoding from response
                encoding = response.charset or 'utf-8'
                
                # Decode content
                try:
                    text = content.decode(encoding)
                except UnicodeDecodeError:
                    # Fallback to utf-8 with error handling
                    text = content.decode('utf-8', errors='replace')
                
                result = {
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'content': content,
                    'text': text,
                    'url': str(response.url),  # Final URL after redirects
                    'encoding': encoding,
                    'size': len(content)
                }
                
                self._logger.debug(f"HTTP GET completed: {url}, status: {response.status}, size: {len(content)}")
                return result
                
        except asyncio.TimeoutError:
            self._logger.error(f"HTTP GET timeout: {url}")
            raise NetworkError(
                message=f"HTTP request timeout after {timeout} seconds",
                details={"url": url, "timeout": timeout}
            )
        except aiohttp.ClientError as e:
            self._logger.error(f"HTTP GET client error: {url}, error: {e}")
            raise NetworkError(
                message=f"HTTP client error: {str(e)}",
                details={"url": url, "error": str(e)}
            )
        except Exception as e:
            self._logger.error(f"HTTP GET unexpected error: {url}, error: {e}")
            raise NetworkError(
                message=f"HTTP request failed: {str(e)}",
                details={"url": url, "error": str(e)}
            )
    
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
        session = await self._get_session()
        
        try:
            self._logger.debug(f"HTTP HEAD: {url}")
            
            custom_timeout = aiohttp.ClientTimeout(total=timeout)
            
            async with session.head(url, headers=headers, timeout=custom_timeout) as response:
                result = {
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'url': str(response.url)
                }
                
                self._logger.debug(f"HTTP HEAD completed: {url}, status: {response.status}")
                return result
                
        except asyncio.TimeoutError:
            self._logger.error(f"HTTP HEAD timeout: {url}")
            raise NetworkError(
                message=f"HTTP HEAD timeout after {timeout} seconds",
                details={"url": url, "timeout": timeout}
            )
        except aiohttp.ClientError as e:
            self._logger.error(f"HTTP HEAD client error: {url}, error: {e}")
            raise NetworkError(
                message=f"HTTP client error: {str(e)}",
                details={"url": url, "error": str(e)}
            )
        except Exception as e:
            self._logger.error(f"HTTP HEAD unexpected error: {url}, error: {e}")
            raise NetworkError(
                message=f"HTTP HEAD failed: {str(e)}",
                details={"url": url, "error": str(e)}
            )
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class BeautifulSoupExtractor(IContentExtractor):
    """
    Content extractor using BeautifulSoup for HTML parsing.
    Implements robust content extraction with error handling.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def extract_content(self, html: str, url: str) -> Optional[ScrapedContent]:
        """
        Extract structured content from HTML using BeautifulSoup.
        
        Args:
            html: Raw HTML content
            url: Source URL for the content
            
        Returns:
            Optional[ScrapedContent]: Extracted content or None if failed
        """
        try:
            self._logger.debug(f"Extracting content from HTML, size: {len(html)} characters")
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            self._clean_soup(soup)
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract meta description
            meta_description = self._extract_meta_description(soup)
            
            # Extract language
            language = self._extract_language(soup)
            
            # Extract text content
            text_content = self._extract_text_content(soup)
            
            # Extract links
            links = self._extract_links(soup, url)
            
            # Extract images
            images = self._extract_images(soup, url)
            
            # Extract headings
            headings = self._extract_headings(soup)
            
            # Create ScrapedContent object
            content = ScrapedContent(
                url=url,
                title=title,
                text_content=text_content,
                raw_html=html,
                meta_description=meta_description,
                language=language,
                links=links,
                images=images,
                headings=headings,
                scraped_at=datetime.now(timezone.utc)
            )
            
            self._logger.debug(f"Content extraction completed: title='{title[:50]}...', text_length={len(text_content)}")
            return content
            
        except Exception as e:
            self._logger.error(f"Content extraction failed for URL {url}: {e}")
            return None
    
    def can_extract(self, html: str, url: str) -> bool:
        """
        Check if this extractor can handle the given HTML/URL.
        
        Args:
            html: HTML content to check
            url: Source URL
            
        Returns:
            bool: True if extractor can handle this content
        """
        if not html or not isinstance(html, str):
            return False
        
        # Check if it looks like HTML
        html_lower = html.lower().strip()
        return (html_lower.startswith('<!doctype html') or 
                html_lower.startswith('<html') or
                '<html' in html_lower[:500])
    
    def get_extractor_name(self) -> str:
        """
        Get name/identifier for this extractor.
        
        Returns:
            str: Extractor name
        """
        return "BeautifulSoupExtractor"
    
    def _clean_soup(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from soup."""
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'aside']):
            element.decompose()
        
        # Remove comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()
        
        # Remove hidden elements
        for element in soup.find_all(attrs={'style': lambda x: x and 'display:none' in x.replace(' ', '')}):
            element.decompose()
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return "No title found"
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc.get('content').strip()
        
        # Fallback to Open Graph description
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc and og_desc.get('content'):
            return og_desc.get('content').strip()
        
        return ""
    
    def _extract_language(self, soup: BeautifulSoup) -> str:
        """Extract page language."""
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            return html_tag.get('lang')
        
        # Check meta tag
        meta_lang = soup.find('meta', attrs={'http-equiv': 'content-language'})
        if meta_lang and meta_lang.get('content'):
            return meta_lang.get('content')
        
        return "en"  # Default to English
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content."""
        # Focus on main content areas
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_=lambda x: x and 'content' in x.lower()) or
            soup.find('body')
        )
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:  # Filter out very short lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and resolve links."""
        links = []
        link_tags = soup.find_all('a', href=True)
        
        for link in link_tags:
            href = link.get('href')
            if href:
                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)
                if absolute_url not in links:
                    links.append(absolute_url)
        
        return links[:100]  # Limit to first 100 links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and resolve image URLs."""
        images = []
        img_tags = soup.find_all('img', src=True)
        
        for img in img_tags:
            src = img.get('src')
            if src:
                # Resolve relative URLs
                absolute_url = urljoin(base_url, src)
                if absolute_url not in images:
                    images.append(absolute_url)
        
        return images[:50]  # Limit to first 50 images
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract headings (h1-h6)."""
        headings = []
        
        for level in range(1, 7):  # h1 to h6
            heading_tags = soup.find_all(f'h{level}')
            for heading in heading_tags:
                text = heading.get_text().strip()
                if text:
                    headings.append(text)
        
        return headings[:20]  # Limit to first 20 headings


class WebScraper(IWebScraper):
    async def crawl(self, start_url: str, max_depth: int = 2, max_pages: int = 20) -> List[ScrapingResult]:
        """Recursively crawl child links starting from start_url."""
        visited = set()
        results = []
        queue = [(start_url, 0)]

        while queue and len(results) < max_pages:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue
            visited.add(url)
            req = ScrapingRequest(url=url)
            result = await self.scrape_content(req)
            results.append(result)
            # Only crawl further if successful and content has links
            if result.success and hasattr(result.content, 'links'):
                for link in result.content.links:
                    if link not in visited and len(results) + len(queue) < max_pages:
                        queue.append((link, depth + 1))
        return results
    """
    Main web scraper implementation using BeautifulSoup and aiohttp.
    Coordinates HTTP requests and content extraction.
    """
    
    def __init__(self, http_client: IHTTPClient, content_extractor: IContentExtractor):
        self._http_client = http_client
        self._content_extractor = content_extractor
        self._logger = logging.getLogger(__name__)
    
    async def scrape_content(self, request: ScrapingRequest) -> ScrapingResult:
        """
        Scrape content from URL with given parameters.
        
        Args:
            request: Scraping request with URL and parameters
            
        Returns:
            ScrapingResult: Result containing scraped content or error info
        """
        start_time = time.time()
        
        try:
            self._logger.info(f"Scraping content from: {request.url}")
            
            # Prepare headers
            headers = {
                'User-Agent': request.user_agent or 'WebContentAnalyzer/1.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Perform HTTP request
            response = await self._http_client.get(
                request.url,
                headers=headers,
                timeout=request.timeout or 30
            )
            
            # Check response status
            if response['status_code'] >= 400:
                return ScrapingResult(
                    success=False,
                    content=None,
                    error_message=f"HTTP error: {response['status_code']}",
                    status_code=response['status_code'],
                    processing_time=time.time() - start_time
                )
            
            # Extract content
            final_url = response['url']
            html = response['text']
            
            if not self._content_extractor.can_extract(html, final_url):
                return ScrapingResult(
                    success=False,
                    content=None,
                    error_message="Content type not supported for extraction",
                    status_code=response['status_code'],
                    processing_time=time.time() - start_time
                )
            
            content = self._content_extractor.extract_content(html, final_url)
            
            if not content:
                return ScrapingResult(
                    success=False,
                    content=None,
                    error_message="Failed to extract content from HTML",
                    status_code=response['status_code'],
                    processing_time=time.time() - start_time
                )
            
            # Success
            processing_time = time.time() - start_time
            self._logger.info(f"Content scraped successfully from {request.url}, processing time: {processing_time:.2f}s")
            
            return ScrapingResult(
                success=True,
                content=content,
                error_message=None,
                status_code=response['status_code'],
                processing_time=processing_time
            )
            
        except NetworkError:
            # Re-raise NetworkError as-is
            raise
        except Exception as e:
            self._logger.error(f"Scraping failed for {request.url}: {e}")
            return ScrapingResult(
                success=False,
                content=None,
                error_message=f"Scraping error: {str(e)}",
                status_code=0,
                processing_time=time.time() - start_time
            )
    
    def supports_url(self, url: str) -> bool:
        """
        Check if this scraper can handle the given URL.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if scraper can handle this URL
        """
        try:
            parsed = urlparse(url)
            return parsed.scheme in ['http', 'https']
        except Exception:
            return False
    
    def get_scraper_name(self) -> str:
        """
        Get name/identifier for this scraper.
        
        Returns:
            str: Scraper name
        """
        return "BeautifulSoupWebScraper"
