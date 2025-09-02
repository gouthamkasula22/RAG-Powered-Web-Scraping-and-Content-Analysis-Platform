"""
Production web scraper implementation using requests and BeautifulSoup
"""
print("🔥 PRODUCTION SCRAPER LOADED - VERSION 2 🔥")
import asyncio
import time
from datetime import datetime
from typing import Optional
import logging
import re
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPING_DEPS = True
except ImportError:
    HAS_SCRAPING_DEPS = False

from src.domain.models import (
    ScrapingResult, URLInfo, ContentMetrics, ScrapedContent, 
    ContentType, ScrapingStatus, ScrapingRequest, ExtractedImage
)
from src.infrastructure.image_processor import ImageProcessor
from src.application.interfaces.scraping import IWebScraper

logger = logging.getLogger(__name__)


class ProductionWebScraper(IWebScraper):
    """Production web scraper using requests and BeautifulSoup"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.image_processor = ImageProcessor()
        
        # Check if image dependencies are available
        from src.infrastructure.image_processor import HAS_IMAGE_DEPS
        logger.info(f"🔧 ProductionWebScraper initialized. Image deps available: {HAS_IMAGE_DEPS}")
        
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
            logger.info(f"🟢 Calling _extract_content for {url_info.url} (extract_images={request.extract_images})")
            try:
                scraped_content = await self._extract_content(soup, url_info, request)
            except Exception as e:
                logger.error(f"❌ Exception in _extract_content: {e}")
                import traceback
                traceback.print_exc()
                scraped_content = None
            
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
    
    async def _extract_content(self, soup: BeautifulSoup, url_info: URLInfo, request: ScrapingRequest) -> ScrapedContent:
        """Extract structured content from BeautifulSoup object"""
        
        logger.info(f"🎯 _extract_content called for {url_info.url}")
        logger.info(f"🎯 extract_images requested: {request.extract_images}")
        
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
        
        # Extract images from the HTML (only if requested)
        extracted_images = []
        if request.extract_images:
            try:
                logger.info(f"🔍 Starting image extraction for {url_info.url}")
                logger.info(f"🔧 ImageProcessor instance: {self.image_processor}")
                logger.info(f"🌐 BeautifulSoup object type: {type(soup)}")
                logger.info(f"📄 HTML content length: {len(str(soup))} characters")
                
                # Check if we have image dependencies
                from src.infrastructure.image_processor import HAS_IMAGE_DEPS
                logger.info(f"📦 HAS_IMAGE_DEPS: {HAS_IMAGE_DEPS}")
                
                extracted_images = self.image_processor.extract_images_from_html(
                    soup=soup,
                    base_url=url_info.url,
                    website_id=url_info.domain
                )
                logger.info(f"🖼️ Image extraction complete: found {len(extracted_images)} images")
                
                # Download images if any were found - using sync approach for now
                if extracted_images:
                    logger.info(f"📥 Starting synchronous image download for {len(extracted_images)} images...")
                    try:
                        downloaded_images = []
                        for i, img in enumerate(extracted_images):
                            downloaded_img = self._download_image_sync(img, url_info.domain, i)
                            downloaded_images.append(downloaded_img)
                        
                        extracted_images = downloaded_images
                        successful_downloads = len([img for img in extracted_images if img.local_path])
                        logger.info(f"✅ Image download complete: {successful_downloads}/{len(extracted_images)} images saved")
                        
                    except Exception as download_error:
                        logger.error(f"❌ Image download failed: {download_error}")
                        import traceback
                        traceback.print_exc()
                
                # Log details about the first few images
                for i, img in enumerate(extracted_images[:3]):
                    logger.info(f"  Image {i+1}: {img.info.original_url} - {img.info.image_type} - Downloaded: {bool(img.local_path)}")
                    
            except Exception as e:
                logger.error(f"❌ Image extraction failed: {e}")
                import traceback
                traceback.print_exc()
                extracted_images = []
        else:
            logger.info(f"⏭️ Image extraction skipped (not requested) for {url_info.url}")
        
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
        
        # Log final state before creating ScrapedContent
        logger.info(f"📊 Final extraction results:")
        logger.info(f"  - Title: {title[:50]}...")
        logger.info(f"  - Main content length: {len(main_content)} chars")
        logger.info(f"  - Headings count: {len(headings)}")
        logger.info(f"  - Links count: {len(links)}")
        logger.info(f"  - Images count: {len(extracted_images)}")
        
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
            status=ScrapingStatus.SUCCESS,
            images=extracted_images
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
        request = ScrapingRequest(url=url, extract_images=True)
        return await self.scrape_content(request)
    
    def supports_url(self, url: str) -> bool:
        """Check if this scraper can handle the given URL"""
        # This production scraper can handle most HTTP/HTTPS URLs
        return url.startswith(('http://', 'https://'))
    
    def get_scraper_name(self) -> str:
        """Get name/identifier for this scraper"""
        return "ProductionWebScraper"
    
    def _download_image_sync(self, extracted_image: ExtractedImage, website_id: str, index: int) -> ExtractedImage:
        """Synchronously download a single image and create thumbnail"""
        try:
            import requests
            
            url = extracted_image.info.original_url
            logger.info(f"📥 Downloading image {index+1}: {url}")
            
            # Create directories
            images_dir = Path("data/images") / website_id
            originals_dir = images_dir / "originals"
            thumbnails_dir = images_dir / "thumbnails"
            originals_dir.mkdir(parents=True, exist_ok=True)
            thumbnails_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            filename = self._generate_filename(url, index)
            original_path = originals_dir / filename
            thumbnail_path = thumbnails_dir / f"thumb_{filename}"
            
            # Download with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, timeout=10, headers=headers, stream=True)
            response.raise_for_status()
            
            # Save original
            with open(original_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✅ Downloaded {original_path} ({original_path.stat().st_size} bytes)")
            
            # Create thumbnail
            self._create_thumbnail_sync(original_path, thumbnail_path)
            
            # Update extracted image
            extracted_image.local_path = str(original_path)
            extracted_image.thumbnail_path = str(thumbnail_path)
            extracted_image.download_status = ScrapingStatus.SUCCESS
            
            return extracted_image
            
        except Exception as e:
            logger.error(f"❌ Failed to download image {index+1}: {e}")
            extracted_image.download_status = ScrapingStatus.FAILED
            extracted_image.download_error = str(e)
            return extracted_image
    
    def _create_thumbnail_sync(self, original_path, thumbnail_path):
        """Create thumbnail synchronously"""
        try:
            from PIL import Image
            
            with Image.open(original_path) as img:
                # Convert RGBA to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                # Create thumbnail
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
                
            logger.info(f"✅ Created thumbnail {thumbnail_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create thumbnail: {e}")
            raise
    
    def _generate_filename(self, url: str, index: int) -> str:
        """Generate safe filename from URL"""
        from urllib.parse import urlparse
        import re
        
        parsed = urlparse(url)
        path = parsed.path
        
        # Extract filename from path
        if path and '.' in path:
            filename = path.split('/')[-1]
            # Clean filename
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        else:
            # Generate filename from domain and index
            domain = parsed.netloc.replace('.', '_')
            filename = f"{domain}_{index:03d}.jpg"
        
        # Ensure it has an extension
        if '.' not in filename:
            filename += '.jpg'
            
        # Add index prefix
        name_part, ext = filename.rsplit('.', 1)
        filename = f"{index:03d}_{name_part}.{ext}"
        
        return filename
