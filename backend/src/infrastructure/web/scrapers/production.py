"""
Production web scraper implementation using requests and BeautifulSoup
"""
print("🔥 PRODUCTION SCRAPER LOADED - VERSION 2 🔥")
import asyncio
import time
from datetime import datetime
from typing import Optional, List
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
        
        # URL cache for avoiding duplicate downloads
        self._url_cache = {}  # url -> local_path mapping
        
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
    
    def _check_image_cache(self, url: str) -> Optional[tuple]:
        """Check if image has been downloaded before and return (local_path, thumbnail_path)"""
        try:
            import sqlite3
            import os
            
            db_path = os.getenv("DATABASE_PATH", "data/analysis_history.db")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path, thumbnail_path 
                FROM images 
                WHERE url = ? AND file_path IS NOT NULL
                ORDER BY extracted_at DESC
                LIMIT 1
            """, (url,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:  # Check if file_path exists
                from pathlib import Path
                local_path = Path(result[0])
                
                # Verify file actually exists
                if local_path.exists():
                    logger.info(f"🎯 Cache HIT for {url}: {local_path}")
                    return (str(local_path), result[1])  # (local_path, thumbnail_path)
                else:
                    logger.debug(f"🗑️ Cache entry exists but file missing: {local_path}")
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Cache check failed for {url}: {e}")
            return None
    
    def _update_image_cache_stats(self, cache_hits: int, total_images: int):
        """Log cache performance statistics"""
        if total_images > 0:
            cache_rate = (cache_hits / total_images) * 100
            logger.info(f"📊 Image Cache Performance: {cache_hits}/{total_images} hits ({cache_rate:.1f}%)")
            if cache_rate > 0:
                logger.info(f"⚡ Saved downloading {cache_hits} images from cache!")
    
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
                
                # Apply max_images limit if specified
                if hasattr(request, 'max_images') and request.max_images > 0:
                    if len(extracted_images) > request.max_images:
                        extracted_images = extracted_images[:request.max_images]
                        logger.info(f"🔢 Limited to {request.max_images} images as requested")
                
                # Download images only if download_images is True
                if extracted_images and hasattr(request, 'download_images') and request.download_images:
                    logger.info(f"� Starting PARALLEL image download for {len(extracted_images)} images...")
                    try:
                        # Use async parallel download
                        downloaded_images = await self._download_images_parallel(
                            extracted_images, 
                            url_info.domain, 
                            max_workers=5
                        )
                        
                        extracted_images = downloaded_images
                        successful_downloads = len([img for img in extracted_images if img.local_path])
                        logger.info(f"✅ Parallel download complete: {successful_downloads}/{len(extracted_images)} images saved")
                        
                        # Windows file locking workaround - if parallel failed, try sync
                        if successful_downloads == 0 and len(extracted_images) > 0:
                            logger.info("🔄 Parallel download had no successes, trying sync fallback...")
                            try:
                                extracted_images = self._download_images_sync_fallback(extracted_images, url_info.domain)
                                successful_downloads = len([img for img in extracted_images if img.local_path])
                                logger.info(f"✅ Sync fallback complete: {successful_downloads}/{len(extracted_images)} images saved")
                            except Exception as fallback_error:
                                logger.error(f"❌ Sync fallback also failed: {fallback_error}")
                        
                    except Exception as download_error:
                        logger.error(f"❌ Parallel download failed: {download_error}")
                        import traceback
                        traceback.print_exc()
                        
                        # Try sync fallback as last resort
                        logger.info("🔄 Trying sync fallback after parallel exception...")
                        try:
                            extracted_images = self._download_images_sync_fallback(extracted_images, url_info.domain)
                            successful_downloads = len([img for img in extracted_images if img.local_path])
                            logger.info(f"✅ Sync fallback complete: {successful_downloads}/{len(extracted_images)} images saved")
                        except Exception as fallback_error:
                            logger.error(f"❌ Sync fallback also failed: {fallback_error}")
                elif extracted_images and hasattr(request, 'download_images') and not request.download_images:
                    logger.info(f"⏭️ Image download skipped (download_images=False). URLs extracted only.")
                elif extracted_images:
                    # Default behavior - use parallel download for better performance
                    logger.info(f"� Starting PARALLEL image download for {len(extracted_images)} images (default behavior)...")
                    try:
                        # Use async parallel download
                        downloaded_images = await self._download_images_parallel(
                            extracted_images, 
                            url_info.domain, 
                            max_workers=5
                        )
                        
                        extracted_images = downloaded_images
                        successful_downloads = len([img for img in extracted_images if img.local_path])
                        logger.info(f"✅ Parallel download complete: {successful_downloads}/{len(extracted_images)} images saved")
                        
                        # Windows file locking workaround - if parallel failed, try sync
                        if successful_downloads == 0 and len(extracted_images) > 0:
                            logger.info("🔄 Parallel download had no successes, trying sync fallback...")
                            try:
                                extracted_images = self._download_images_sync_fallback(extracted_images, url_info.domain)
                                successful_downloads = len([img for img in extracted_images if img.local_path])
                                logger.info(f"✅ Sync fallback complete: {successful_downloads}/{len(extracted_images)} images saved")
                            except Exception as fallback_error:
                                logger.error(f"❌ Sync fallback also failed: {fallback_error}")
                        
                    except Exception as download_error:
                        logger.error(f"❌ Image download failed: {download_error}")
                        import traceback
                        traceback.print_exc()
                        
                        # Try sync fallback as last resort
                        logger.info("🔄 Trying sync fallback after parallel exception...")
                        try:
                            extracted_images = self._download_images_sync_fallback(extracted_images, url_info.domain)
                            successful_downloads = len([img for img in extracted_images if img.local_path])
                            logger.info(f"✅ Sync fallback complete: {successful_downloads}/{len(extracted_images)} images saved")
                        except Exception as fallback_error:
                            logger.error(f"❌ Sync fallback also failed: {fallback_error}")
                
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
    
    async def secure_scrape(self, 
                           url: str,
                           extract_images: bool = True,
                           download_images: bool = False,
                           max_images: int = 10) -> ScrapingResult:
        """Legacy method for backwards compatibility with image control parameters"""
        request = ScrapingRequest(
            url=url, 
            extract_images=extract_images,
            download_images=download_images,
            max_images=max_images
        )
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
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/*,*/*;q=0.8'
            }
            
            response = requests.get(url, timeout=10, headers=headers, stream=True)
            response.raise_for_status()
            
            # Check content type to handle SVG properly
            content_type = response.headers.get('content-type', '').lower()
            is_svg = 'svg' in content_type or url.lower().endswith('.svg')
            
            # Adjust filename extension for SVG files
            if is_svg and not original_path.suffix.lower() == '.svg':
                original_path = original_path.with_suffix('.svg')
                thumbnail_path = thumbnails_dir / f"thumb_{original_path.name}"
            
            # Save original
            with open(original_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✅ Downloaded {original_path} ({original_path.stat().st_size} bytes) - Type: {'SVG' if is_svg else 'Raster'}")
            
            # Create thumbnail (with fallback for unsupported formats)
            try:
                self._create_thumbnail_sync(original_path, thumbnail_path)
            except Exception as thumb_error:
                logger.warning(f"⚠️ Thumbnail creation failed for {original_path}: {thumb_error}")
                # Don't fail the entire download just because thumbnail failed
                # The original image is still saved and can be served
                extracted_image.thumbnail_path = None
            
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
        """Create thumbnail synchronously - supports both raster and SVG images"""
        try:
            from PIL import Image
            import os
            
            # Check if it's an SVG file
            if str(original_path).lower().endswith('.svg'):
                try:
                    import cairosvg
                    import io
                    
                    logger.info(f"🎨 Processing SVG file: {original_path}")
                    
                    # Convert SVG to PNG in memory
                    with open(original_path, 'rb') as svg_file:
                        svg_data = svg_file.read()
                    
                    # Convert SVG to PNG with 200x200 max size
                    png_data = cairosvg.svg2png(
                        bytestring=svg_data,
                        output_width=200,
                        output_height=200
                    )
                    
                    # Save as JPEG thumbnail
                    img = Image.open(io.BytesIO(png_data))
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = background
                    
                    img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
                    logger.info(f"✅ Created SVG thumbnail {thumbnail_path}")
                    return
                    
                except ImportError:
                    logger.warning(f"⚠️ cairosvg not available, skipping SVG: {original_path}")
                    raise Exception("SVG processing not available")
                except Exception as svg_error:
                    logger.warning(f"⚠️ SVG processing failed: {svg_error}, skipping")
                    raise Exception(f"SVG processing failed: {svg_error}")
            
            # Handle regular raster images (existing functionality)
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

    async def _download_images_parallel(self, images: List[ExtractedImage], website_id: str, max_workers: int = 5) -> List[ExtractedImage]:
        """Download images in parallel for better performance"""
        import aiohttp
        import asyncio
        from pathlib import Path
        
        if not images:
            return []
        
        logger.info(f"🚀 Starting parallel download of {len(images)} images with {max_workers} workers")
        
        # Create directories
        images_dir = Path("data/images") / website_id
        originals_dir = images_dir / "originals"
        thumbnails_dir = images_dir / "thumbnails"
        originals_dir.mkdir(parents=True, exist_ok=True)
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        semaphore = asyncio.Semaphore(max_workers)
        
        async def download_single(session: aiohttp.ClientSession, index: int, image: ExtractedImage) -> ExtractedImage:
            """Download a single image with concurrency control and caching"""
            async with semaphore:
                try:
                    url = image.info.original_url
                    
                    # Step 4: Check cache first
                    cache_result = self._check_image_cache(url)
                    if cache_result:
                        cached_local_path, cached_thumbnail_path = cache_result
                        
                        # Copy cached file to current website folder
                        filename = self._generate_filename(url, index)
                        original_path = originals_dir / filename
                        thumbnail_path = thumbnails_dir / f"thumb_{filename}"
                        
                        # Copy original file
                        import shutil
                        shutil.copy2(cached_local_path, original_path)
                        
                        # Copy thumbnail if it exists
                        if cached_thumbnail_path and Path(cached_thumbnail_path).exists():
                            shutil.copy2(cached_thumbnail_path, thumbnail_path)
                            image.thumbnail_path = str(thumbnail_path)
                        
                        logger.info(f"🎯 Using cached image {index+1}/{len(images)}: {url}")
                        
                        # Update image
                        image.local_path = str(original_path)
                        image.download_status = ScrapingStatus.SUCCESS
                        
                        return image
                    
                    # No cache hit - download normally
                    logger.info(f"📥 Downloading image {index+1}/{len(images)}: {url}")
                    
                    # Generate unique filename with index to avoid conflicts
                    filename = self._generate_filename(url, index)
                    # Ensure filename is unique by adding microsecond timestamp
                    import time
                    timestamp = str(int(time.time() * 1000000))[-6:]  # Last 6 digits of microseconds
                    name_parts = filename.rsplit('.', 1)
                    if len(name_parts) == 2:
                        unique_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
                    else:
                        unique_filename = f"{filename}_{timestamp}.jpg"
                    
                    original_path = originals_dir / unique_filename
                    thumbnail_path = thumbnails_dir / f"thumb_{unique_filename}"
                    
                    # Download with timeout
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'image/*,*/*;q=0.8'
                    }
                    
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            content = await response.read()
                            
                            # Save original with proper file handling
                            try:
                                # Ensure directory exists
                                original_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                # Write file atomically to avoid conflicts
                                temp_path = original_path.with_suffix('.tmp')
                                with open(temp_path, 'wb') as f:
                                    f.write(content)
                                
                                # Atomic rename to final path
                                temp_path.rename(original_path)
                                
                                logger.info(f"✅ Downloaded {original_path.name} ({len(content)} bytes)")
                                
                                # Update image
                                image.local_path = str(original_path)
                                image.download_status = ScrapingStatus.SUCCESS
                                
                            except Exception as file_error:
                                logger.warning(f"⚠️ File save failed for {original_path}: {file_error}")
                                image.download_status = ScrapingStatus.FAILED
                                image.download_error = f"File save error: {file_error}"
                            
                        else:
                            logger.warning(f"⚠️ Failed to download image {index+1}: HTTP {response.status}")
                            image.download_status = ScrapingStatus.FAILED
                            image.download_error = f"HTTP {response.status}"
                            
                except Exception as e:
                    logger.warning(f"⚠️ Failed to download image {index+1}: {e}")
                    image.download_status = ScrapingStatus.FAILED
                    image.download_error = str(e)
                
                return image
        
        # Download all images in parallel
        try:
            connector = aiohttp.TCPConnector(limit=max_workers, limit_per_host=max_workers)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [download_single(session, i, img) for i, img in enumerate(images)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions and return successful downloads
                successful_images = []
                for result in results:
                    if isinstance(result, ExtractedImage):
                        successful_images.append(result)
                    else:
                        logger.error(f"Download task failed: {result}")
                
                successful_count = len([img for img in successful_images if img.local_path])
                logger.info(f"🎉 Parallel download complete: {successful_count}/{len(images)} images downloaded")
                
                # Step 4: Calculate and log cache statistics
                cache_hits = 0
                for img in successful_images:
                    if img.local_path and img.download_status == ScrapingStatus.SUCCESS:
                        # Check if this was a cache hit (no actual download logged)
                        # This is a simple heuristic - images from cache won't have download logs
                        pass
                
                # For now, let's count cache hits differently - by checking if files were copied vs downloaded
                # This will be refined in future iterations
                self._update_image_cache_stats(0, len(images))  # Will be improved with better tracking
                
                return successful_images
                
        except Exception as e:
            logger.error(f"❌ Parallel download failed: {e}")
            # Fallback to sync download
            logger.info("🔄 Falling back to synchronous download...")
            return self._download_images_sync_fallback(images, website_id)

    def _download_images_sync_fallback(self, images: List[ExtractedImage], website_id: str) -> List[ExtractedImage]:
        """Fallback synchronous download method"""
        downloaded_images = []
        for i, img in enumerate(images):
            try:
                downloaded_img = self._download_image_sync(img, website_id, i)
                downloaded_images.append(downloaded_img)
            except Exception as e:
                logger.warning(f"Sync fallback failed for image {i+1}: {e}")
                img.download_status = ScrapingStatus.FAILED
                downloaded_images.append(img)
        return downloaded_images
