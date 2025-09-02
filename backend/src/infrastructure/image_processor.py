"""
Image processing utilities for web scraping.
Handles image extraction, downloading, and optimization.
"""
import asyncio
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import logging

try:
    import requests
    from bs4 import BeautifulSoup, Tag
    from PIL import Image, ImageOps
    HAS_IMAGE_DEPS = True
except ImportError:
    HAS_IMAGE_DEPS = False

from src.domain.models import (
    ImageInfo, ExtractedImage, ImageType, ImageFormat, ScrapingStatus
)

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image extraction and processing for web scraping"""
    
    def __init__(self, 
                 storage_path: str = "data/images",
                 thumbnail_size: Tuple[int, int] = (200, 200),
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 timeout: int = 30):
        """
        Initialize image processor
        
        Args:
            storage_path: Base directory for storing images
            thumbnail_size: Size for generated thumbnails (width, height)
            max_file_size: Maximum file size to download in bytes
            timeout: Request timeout in seconds
        """
        self.storage_path = Path(storage_path)
        self.thumbnail_size = thumbnail_size
        self.max_file_size = max_file_size
        self.timeout = timeout
        
        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "originals").mkdir(exist_ok=True)
        (self.storage_path / "thumbnails").mkdir(exist_ok=True)
        
        # Setup session for image downloads
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.9'
        })
    
    def extract_images_from_html(self, 
                                soup: BeautifulSoup, 
                                base_url: str,
                                website_id: str) -> List[ExtractedImage]:
        """
        Extract all images from HTML content using multiple techniques
        
        Args:
            soup: BeautifulSoup object of the HTML
            base_url: Base URL for resolving relative image URLs
            website_id: ID for organizing stored images
            
        Returns:
            List of ExtractedImage objects
        """
        logger.info(f"🔍 Starting comprehensive image extraction from {base_url}")
        
        if not HAS_IMAGE_DEPS:
            logger.warning("Image processing dependencies not available")
            return []

        images = []
        found_urls = set()  # Prevent duplicates

        # 1. Traditional img tags
        img_tags = soup.find_all('img')
        logger.info(f"🔍 Found {len(img_tags)} img tags")
        
        for img_tag in img_tags:
            try:
                image_info = self._extract_image_info(img_tag, base_url)
                if image_info and image_info.original_url not in found_urls:
                    found_urls.add(image_info.original_url)
                    extracted_image = ExtractedImage(info=image_info)
                    images.append(extracted_image)
            except Exception as e:
                logger.debug(f"Failed to extract img tag: {e}")
                continue

        # 2. Picture elements with source sets
        picture_tags = soup.find_all('picture')
        logger.info(f"🔍 Found {len(picture_tags)} picture elements")
        
        for picture in picture_tags:
            try:
                # Look for source tags first, then img
                sources = picture.find_all('source')
                for source in sources:
                    srcset = source.get('srcset')
                    if srcset:
                        urls = self._extract_srcset_urls(srcset)
                        for url in urls:
                            full_url = urljoin(base_url, url)
                            if full_url not in found_urls:
                                found_urls.add(full_url)
                                image_info = ImageInfo(
                                    original_url=full_url,
                                    alt_text="",
                                    image_type=ImageType.CONTENT
                                )
                                images.append(ExtractedImage(info=image_info))
                                break  # Just take first URL from srcset
            except Exception as e:
                logger.debug(f"Failed to extract picture element: {e}")
                continue

        # 3. Meta tags (Open Graph, Twitter Cards)
        meta_images = self._extract_meta_images(soup, base_url)
        for img_url in meta_images:
            if img_url not in found_urls:
                found_urls.add(img_url)
                image_info = ImageInfo(
                    original_url=img_url,
                    alt_text="Meta image",
                    image_type=ImageType.CONTENT
                )
                images.append(ExtractedImage(info=image_info))

        # 4. CSS background images (basic extraction)
        bg_images = self._extract_background_images(soup, base_url)
        for img_url in bg_images:
            if img_url not in found_urls:
                found_urls.add(img_url)
                image_info = ImageInfo(
                    original_url=img_url,
                    alt_text="Background image",
                    image_type=ImageType.DECORATIVE
                )
                images.append(ExtractedImage(info=image_info))

        logger.info(f"🎉 Comprehensive extraction complete: {len(images)} unique images found")
        return images
    
    def _extract_image_info(self, img_tag: Tag, base_url: str) -> Optional[ImageInfo]:
        """Extract image information from img tag with modern lazy loading support"""
        # Try multiple src attributes (common lazy loading patterns)
        src_attrs = [
            'src', 'data-src', 'data-lazy-src', 'data-original', 
            'data-srcset', 'data-lazy', 'data-echo', 'data-original-src'
        ]
        
        src = None
        for attr in src_attrs:
            src = img_tag.get(attr)
            if src:
                break
        
        if not src:
            return None
        
        # Handle srcset (take first URL)
        if 'srcset' in src or ',' in src:
            urls = self._extract_srcset_urls(src)
            if urls:
                src = urls[0]
        
        # Resolve relative URLs
        full_url = urljoin(base_url, src)
        
        # Validate URL
        if not self._is_valid_image_url(full_url):
            return None
        
        # Extract attributes
        alt_text = img_tag.get('alt', '').strip()
        title = img_tag.get('title', '').strip() or None
        width = self._safe_int(img_tag.get('width'))
        height = self._safe_int(img_tag.get('height'))
        loading = img_tag.get('loading')
        
        # Determine image format from URL
        image_format = self._detect_image_format(full_url)
        
        # Classify image type
        image_type = self._classify_image_type(img_tag, alt_text, src)
        
        # Extract context (surrounding text)
        context = self._extract_context(img_tag)
        
        # Check if decorative
        is_decorative = self._is_decorative_image(img_tag, alt_text)
        
        return ImageInfo(
            original_url=full_url,
            alt_text=alt_text or None,
            title=title,
            width=width,
            height=height,
            image_format=image_format,
            image_type=image_type,
            context=context,
            is_decorative=is_decorative,
            loading_attribute=loading
        )
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert value to integer"""
        if not value:
            return None
        try:
            # Handle CSS units (remove px, em, etc.)
            if isinstance(value, str):
                value = re.sub(r'[^\d]', '', value)
            return int(value) if value else None
        except (ValueError, TypeError):
            return None
    
    def _detect_image_format(self, url: str) -> ImageFormat:
        """Detect image format from URL"""
        url_lower = url.lower()
        if '.jpg' in url_lower or '.jpeg' in url_lower:
            return ImageFormat.JPEG
        elif '.png' in url_lower:
            return ImageFormat.PNG
        elif '.webp' in url_lower:
            return ImageFormat.WEBP
        elif '.gif' in url_lower:
            return ImageFormat.GIF
        elif '.svg' in url_lower:
            return ImageFormat.SVG
        elif '.bmp' in url_lower:
            return ImageFormat.BMP
        else:
            return ImageFormat.UNKNOWN
    
    def _classify_image_type(self, img_tag: Tag, alt_text: str, src: str) -> ImageType:
        """Classify the type of image based on context and attributes"""
        src_lower = src.lower()
        alt_lower = alt_text.lower() if alt_text else ''
        
        # Check for logo
        if any(indicator in src_lower for indicator in ['logo', 'brand']):
            return ImageType.LOGO
        if any(indicator in alt_lower for indicator in ['logo', 'brand', 'company']):
            return ImageType.LOGO
        
        # Check for icons
        if any(indicator in src_lower for indicator in ['icon', 'favicon', 'sprite']):
            return ImageType.ICON
        if 'icon' in alt_lower:
            return ImageType.ICON
        
        # Check for hero images
        if any(indicator in src_lower for indicator in ['hero', 'banner', 'header']):
            return ImageType.HERO
        
        # Check for thumbnails
        if any(indicator in src_lower for indicator in ['thumb', 'preview', 'small']):
            return ImageType.THUMBNAIL
        
        # Check for product images
        if any(indicator in src_lower for indicator in ['product', 'item', 'catalog']):
            return ImageType.PRODUCT
        
        # Check for avatars
        if any(indicator in src_lower for indicator in ['avatar', 'profile', 'user']):
            return ImageType.AVATAR
        if any(indicator in alt_lower for indicator in ['avatar', 'profile', 'user']):
            return ImageType.AVATAR
        
        # Check CSS classes for more context
        css_classes = img_tag.get('class', [])
        if isinstance(css_classes, list):
            classes_str = ' '.join(css_classes).lower()
            if any(indicator in classes_str for indicator in ['hero', 'banner']):
                return ImageType.HERO
            elif any(indicator in classes_str for indicator in ['logo', 'brand']):
                return ImageType.LOGO
            elif any(indicator in classes_str for indicator in ['icon']):
                return ImageType.ICON
        
        # Default to content image
        return ImageType.CONTENT
    
    def _extract_context(self, img_tag: Tag) -> Optional[str]:
        """Extract surrounding text context for the image"""
        try:
            # Look for parent elements with text
            parent = img_tag.parent
            context_parts = []
            
            # Check figure caption
            figure = img_tag.find_parent('figure')
            if figure:
                caption = figure.find('figcaption')
                if caption:
                    context_parts.append(caption.get_text().strip())
            
            # Check for nearby text in parent elements
            if parent and parent.name not in ['html', 'body']:
                parent_text = parent.get_text().strip()
                if parent_text and len(parent_text) < 200:  # Avoid long text blocks
                    context_parts.append(parent_text)
            
            return ' '.join(context_parts)[:500] if context_parts else None  # Limit length
        except Exception as e:
            logger.warning(f"Failed to extract context: {e}")
            return None
    
    def _is_decorative_image(self, img_tag: Tag, alt_text: str) -> bool:
        """Determine if image is purely decorative"""
        # Empty alt attribute often indicates decorative image
        if alt_text == "":
            return True
        
        # Check for decorative indicators in src
        src = img_tag.get('src', '').lower()
        decorative_indicators = ['decoration', 'spacer', 'divider', 'bullet', 'bg-']
        if any(indicator in src for indicator in decorative_indicators):
            return True
        
        # Very small images are often decorative
        width = self._safe_int(img_tag.get('width'))
        height = self._safe_int(img_tag.get('height'))
        if width and height and width <= 20 and height <= 20:
            return True
        
        return False
    
    async def download_images(self, 
                            extracted_images: List[ExtractedImage], 
                            website_id: str) -> List[ExtractedImage]:
        """
        Download images and create thumbnails
        
        Args:
            extracted_images: List of images to download
            website_id: ID for organizing stored images
            
        Returns:
            Updated list with download status and local paths
        """
        if not HAS_IMAGE_DEPS:
            logger.warning("Image processing dependencies not available")
            return extracted_images
        
        # Create website-specific directories
        website_dir = self.storage_path / website_id
        originals_dir = website_dir / "originals"
        thumbnails_dir = website_dir / "thumbnails"
        
        for dir_path in [website_dir, originals_dir, thumbnails_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Download images concurrently
        download_tasks = []
        for i, extracted_image in enumerate(extracted_images):
            task = self._download_single_image(
                extracted_image, 
                originals_dir, 
                thumbnails_dir, 
                i
            )
            download_tasks.append(task)
        
        # Execute downloads
        results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # Update extracted_images with results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Download failed for image {i}: {result}")
                extracted_images[i].download_status = ScrapingStatus.FAILED
                extracted_images[i].download_error = str(result)
            else:
                extracted_images[i] = result
        
        logger.info(f"Downloaded {sum(1 for img in extracted_images if img.is_downloaded)} out of {len(extracted_images)} images")
        
        return extracted_images
    
    async def _download_single_image(self, 
                                   extracted_image: ExtractedImage,
                                   originals_dir: Path,
                                   thumbnails_dir: Path,
                                   index: int) -> ExtractedImage:
        """Download a single image and create thumbnail"""
        try:
            url = extracted_image.info.original_url
            
            # Generate filename
            filename = self._generate_filename(url, index)
            original_path = originals_dir / filename
            thumbnail_path = thumbnails_dir / f"thumb_{filename}"
            
            # Download original image
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_size:
                raise ValueError(f"File too large: {content_length} bytes")
            
            # Save original
            with open(original_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Get file size
            file_size = original_path.stat().st_size
            
            # Create thumbnail using PIL
            await self._create_thumbnail(original_path, thumbnail_path)
            
            # Update extracted image
            extracted_image.local_path = str(original_path)
            extracted_image.thumbnail_path = str(thumbnail_path)
            extracted_image.download_status = ScrapingStatus.SUCCESS
            
            # Update image info with actual file size
            extracted_image.info = ImageInfo(
                original_url=extracted_image.info.original_url,
                alt_text=extracted_image.info.alt_text,
                title=extracted_image.info.title,
                width=extracted_image.info.width,
                height=extracted_image.info.height,
                file_size=file_size,
                image_format=extracted_image.info.image_format,
                image_type=extracted_image.info.image_type,
                context=extracted_image.info.context,
                is_decorative=extracted_image.info.is_decorative,
                loading_attribute=extracted_image.info.loading_attribute
            )
            
            return extracted_image
            
        except Exception as e:
            logger.error(f"Failed to download image {url}: {e}")
            extracted_image.download_status = ScrapingStatus.FAILED
            extracted_image.download_error = str(e)
            return extracted_image
    
    async def _create_thumbnail(self, original_path: Path, thumbnail_path: Path):
        """Create thumbnail from original image"""
        try:
            with Image.open(original_path) as img:
                # Convert RGBA to RGB if necessary (for JPEG compatibility)
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                # Create thumbnail
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
                
        except Exception as e:
            logger.error(f"Failed to create thumbnail for {original_path}: {e}")
            raise
    
    def _generate_filename(self, url: str, index: int) -> str:
        """Generate safe filename from URL"""
        parsed = urlparse(url)
        path = parsed.path
        
        # Extract filename from path
        if path:
            filename = os.path.basename(path)
            if filename and '.' in filename:
                # Clean filename
                name, ext = filename.rsplit('.', 1)
                safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:50]
                return f"{index:03d}_{safe_name}.{ext}"
        
        # Fallback filename
        return f"{index:03d}_image.jpg"
    
    def cleanup_old_images(self, website_id: str, days_old: int = 7):
        """Remove old downloaded images to save space"""
        try:
            website_dir = self.storage_path / website_id
            if not website_dir.exists():
                return
            
            import time
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            
            for image_file in website_dir.rglob('*'):
                if image_file.is_file() and image_file.stat().st_mtime < cutoff_time:
                    image_file.unlink()
                    logger.debug(f"Removed old image: {image_file}")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old images: {e}")

    def _extract_srcset_urls(self, srcset: str) -> List[str]:
        """Extract URLs from srcset attribute"""
        urls = []
        # srcset format: "url1 width1, url2 width2, ..."
        parts = srcset.split(',')
        for part in parts:
            url = part.strip().split()[0]  # Take URL part before width descriptor
            if url:
                urls.append(url)
        return urls

    def _extract_meta_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract images from meta tags (Open Graph, Twitter Cards)"""
        urls = []
        
        # Open Graph images
        og_images = soup.find_all('meta', property=['og:image', 'og:image:url'])
        for meta in og_images:
            content = meta.get('content')
            if content:
                full_url = urljoin(base_url, content)
                urls.append(full_url)
        
        # Twitter Card images
        twitter_images = soup.find_all('meta', attrs={'name': ['twitter:image', 'twitter:image:src']})
        for meta in twitter_images:
            content = meta.get('content')
            if content:
                full_url = urljoin(base_url, content)
                urls.append(full_url)
        
        return urls

    def _extract_background_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract background images from inline styles"""
        urls = []
        
        # Find elements with style attributes
        elements_with_style = soup.find_all(attrs={'style': True})
        for element in elements_with_style:
            style = element.get('style', '')
            # Look for background-image: url(...)
            import re
            bg_matches = re.findall(r'background-image\s*:\s*url\(["\']?(.*?)["\']?\)', style, re.IGNORECASE)
            for match in bg_matches:
                full_url = urljoin(base_url, match)
                if self._is_valid_image_url(full_url):
                    urls.append(full_url)
        
        return urls

    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL looks like a valid image URL"""
        if not url or len(url) > 2000:  # Reasonable URL length limit
            return False
        
        # Skip data URLs, javascript, etc.
        if url.startswith(('data:', 'javascript:', 'mailto:', '#')):
            return False
        
        # Check for common image extensions or formats
        image_patterns = [
            r'\.(jpg|jpeg|png|gif|svg|webp|bmp|tiff)(\?|$)',  # File extensions
            r'/image/',  # Path contains 'image'
            r'\.amazonaws\.com.*\.(jpg|jpeg|png|gif|svg|webp)',  # AWS S3 images
            r'cloudinary\.com',  # Cloudinary CDN
            r'imgix\.net',  # Imgix CDN
        ]
        
        for pattern in image_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        return False
