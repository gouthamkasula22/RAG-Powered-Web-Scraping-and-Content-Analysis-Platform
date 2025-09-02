"""
Domain models representing the core business entities.
These are pure business objects with no external dependencies.
Follows Single Responsibility Principle - each model has one clear purpose.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from urllib.parse import urlparse
import re
import logging

# Use standard logging until loguru is installed
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type classification for different website types"""
    ARTICLE = "article"
    HOMEPAGE = "homepage" 
    PRODUCT = "product"
    BLOG_POST = "blog_post"
    NEWS = "news"
    DOCUMENTATION = "documentation"
    FORUM = "forum"
    SOCIAL_MEDIA = "social_media"
    UNKNOWN = "unknown"


class ImageType(Enum):
    """Image classification types for different website images"""
    LOGO = "logo"
    HERO = "hero"
    CONTENT = "content"
    THUMBNAIL = "thumbnail"
    ICON = "icon"
    BANNER = "banner"
    PRODUCT = "product"
    AVATAR = "avatar"
    BACKGROUND = "background"
    DECORATION = "decoration"
    UNKNOWN = "unknown"


class ImageFormat(Enum):
    """Supported image formats"""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    SVG = "svg"
    BMP = "bmp"
    UNKNOWN = "unknown"


class ScrapingStatus(Enum):
    """Status codes for scraping operations"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    INVALID_URL = "invalid_url"
    NETWORK_ERROR = "network_error"
    CONTENT_TOO_LARGE = "content_too_large"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class URLInfo:
    """
    Value object for URL information.
    Immutable to ensure data integrity.
    """
    url: str
    domain: str
    is_secure: bool
    path: str
    query_params: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_url(cls, url: str) -> 'URLInfo':
        """
        Factory method to create URLInfo from URL string.
        Handles URL parsing and validation with proper error handling.
        """
        try:
            logger.debug(f"Parsing URL: {url}")
            parsed = urlparse(url.strip())
            
            if not parsed.netloc:
                logger.error(f"Invalid URL - no domain found: {url}")
                raise ValueError(f"Invalid URL format: {url}")
            
            # Parse query parameters with error handling
            query_params = {}
            if parsed.query:
                try:
                    for param in parsed.query.split('&'):
                        if '=' in param:
                            key, value = param.split('=', 1)
                            query_params[key] = value
                except Exception as e:
                    logger.warning(f"Failed to parse query parameters for {url}: {e}")
            
            url_info = cls(
                url=url,
                domain=parsed.netloc.lower(),
                is_secure=parsed.scheme == 'https',
                path=parsed.path or '/',
                query_params=query_params
            )
            
            logger.debug(f"Successfully parsed URL: {url} -> domain: {url_info.domain}")
            return url_info
            
        except Exception as e:
            logger.error(f"Failed to parse URL {url}: {e}")
            raise ValueError(f"Failed to parse URL: {url}") from e
    
    @property
    def base_domain(self) -> str:
        """Extract base domain (remove subdomains) with error handling"""
        try:
            parts = self.domain.split('.')
            if len(parts) >= 2:
                base = '.'.join(parts[-2:])
                logger.debug(f"Extracted base domain: {base} from {self.domain}")
                return base
            return self.domain
        except Exception as e:
            logger.warning(f"Failed to extract base domain from {self.domain}: {e}")
            return self.domain
    
    @property
    def is_root_page(self) -> bool:
        """Check if this is a root/homepage URL"""
        root_paths = ['/', '/index.html', '/index.php', '/home', '/index.htm']
        is_root = self.path in root_paths
        logger.debug(f"URL {self.url} is_root_page: {is_root}")
        return is_root


@dataclass(frozen=True)
class ImageInfo:
    """
    Value object representing extracted image information.
    Immutable to ensure data integrity.
    """
    original_url: str
    alt_text: Optional[str] = None
    title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None  # in bytes
    image_format: ImageFormat = ImageFormat.UNKNOWN
    image_type: ImageType = ImageType.UNKNOWN
    context: Optional[str] = None  # Surrounding text context
    is_decorative: bool = False  # True if image is purely decorative
    loading_attribute: Optional[str] = None  # lazy, eager, etc.
    
    @property
    def aspect_ratio(self) -> Optional[float]:
        """Calculate aspect ratio if dimensions are available"""
        if self.width and self.height and self.height > 0:
            return self.width / self.height
        return None
    
    @property
    def is_optimized(self) -> bool:
        """Check if image appears to be optimized based on format and size"""
        if not self.file_size:
            return False
        
        # Consider WebP as optimized format
        if self.image_format == ImageFormat.WEBP:
            return True
        
        # Basic size-based optimization check (< 100KB for content images)
        if self.image_type in [ImageType.CONTENT, ImageType.THUMBNAIL]:
            return self.file_size < 100_000  # 100KB
        
        # Stricter for icons (< 50KB)
        if self.image_type == ImageType.ICON:
            return self.file_size < 50_000  # 50KB
        
        return self.file_size < 200_000  # 200KB for other types
    
    @property
    def has_accessibility_info(self) -> bool:
        """Check if image has proper accessibility information"""
        if self.is_decorative:
            return self.alt_text == "" or self.alt_text is None
        return bool(self.alt_text and self.alt_text.strip())


@dataclass
class ExtractedImage:
    """
    Represents an image extracted and processed from a website.
    Includes both original metadata and processed file information.
    """
    info: ImageInfo
    local_path: Optional[str] = None  # Path to saved image file
    thumbnail_path: Optional[str] = None  # Path to generated thumbnail
    download_status: ScrapingStatus = ScrapingStatus.SUCCESS
    download_error: Optional[str] = None
    extracted_at: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None  # Database ID (set after saving)
    
    @property
    def is_downloaded(self) -> bool:
        """Check if image has been successfully downloaded"""
        return (self.download_status == ScrapingStatus.SUCCESS and 
                self.local_path is not None)
    
    @property
    def has_thumbnail(self) -> bool:
        """Check if thumbnail has been generated"""
        return (self.is_downloaded and 
                self.thumbnail_path is not None)


@dataclass
class ContentMetrics:
    """
    Value object for content quality metrics.
    Helps determine content value and readability.
    """
    word_count: int
    sentence_count: int
    paragraph_count: int
    link_count: int
    image_count: int
    heading_count: int
    reading_time_minutes: float
    
    @classmethod
    def calculate(cls, content: str, links: List[str], headings: List[str]) -> 'ContentMetrics':
        """Calculate metrics from content with comprehensive error handling"""
        try:
            logger.debug("Calculating content metrics")
            
            # Safe word counting
            words = len(content.split()) if content else 0
            
            # Safe sentence counting with fallback
            try:
                sentences = len(re.findall(r'[.!?]+', content)) if content else 0
                sentences = max(sentences, 1)  # Avoid division by zero
            except Exception as e:
                logger.warning(f"Failed to count sentences, using fallback: {e}")
                sentences = max(1, words // 15)  # Fallback: ~15 words per sentence
            
            # Safe paragraph counting
            try:
                paragraphs = len([p for p in content.split('\n\n') if p.strip()]) if content else 0
                paragraphs = max(paragraphs, 1)  # Avoid division by zero
            except Exception as e:
                logger.warning(f"Failed to count paragraphs, using fallback: {e}")
                paragraphs = max(1, sentences // 3)  # Fallback: ~3 sentences per paragraph
            
            # Safe image counting
            try:
                image_count = content.count('<img') + content.count('[image]') if content else 0
            except Exception as e:
                logger.warning(f"Failed to count images: {e}")
                image_count = 0
            
            # Calculate reading time (average 200 words per minute)
            reading_time = words / 200.0 if words > 0 else 0.0
            
            metrics = cls(
                word_count=words,
                sentence_count=sentences,
                paragraph_count=paragraphs,
                link_count=len(links) if links else 0,
                image_count=image_count,
                heading_count=len(headings) if headings else 0,
                reading_time_minutes=reading_time
            )
            
            logger.debug(f"Content metrics calculated: {words} words, {sentences} sentences, "
                        f"{paragraphs} paragraphs, {reading_time:.1f}min reading time")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate content metrics: {e}")
            # Return minimal safe metrics
            return cls(
                word_count=0,
                sentence_count=1,
                paragraph_count=1,
                link_count=0,
                image_count=0,
                heading_count=0,
                reading_time_minutes=0.0
            )
    
    @property
    def content_density_score(self) -> float:
        """Calculate content density (0-10 scale) with error handling"""
        try:
            if self.word_count == 0:
                return 0.0
            
            # Good content has balanced text-to-link ratio
            link_ratio = self.link_count / max(self.word_count / 100, 1)
            heading_ratio = self.heading_count / max(self.paragraph_count, 1)
            
            # Normalize to 0-10 scale with safety checks
            density = min(10, (self.word_count / 100) * (1 + heading_ratio) * (1 - min(link_ratio, 0.5)))
            density = max(0, density)  # Ensure non-negative
            
            score = round(density, 2)
            logger.debug(f"Content density score calculated: {score}")
            return score
            
        except Exception as e:
            logger.warning(f"Failed to calculate content density score: {e}")
            return 0.0


@dataclass
class ScrapedContent:
    """
    Aggregate root for scraped website content.
    Contains all information extracted from a webpage.
    """
    url_info: URLInfo
    title: str
    headings: List[str]
    main_content: str
    links: List[str]
    meta_description: Optional[str]
    meta_keywords: List[str]
    content_type: ContentType
    metrics: ContentMetrics
    scraped_at: datetime
    status: ScrapingStatus
    images: List[ExtractedImage] = field(default_factory=list)  # New field for extracted images
    
    def __post_init__(self):
        """Validate content after creation with proper error handling"""
        try:
            logger.debug(f"Validating scraped content for URL: {self.url_info.url}")
            
            if not self.is_valid_content():
                error_msg = f"Invalid content for URL: {self.url_info.url} - " \
                           f"title: '{self.title[:50]}', content_length: {len(self.main_content)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            logger.debug(f"Content validation successful for URL: {self.url_info.url}")
            
        except Exception as e:
            logger.error(f"Content validation failed: {e}")
            raise
    
    def is_valid_content(self) -> bool:
        """
        Business rule: Valid content must meet minimum quality standards.
        This is a domain business rule with comprehensive validation.
        """
        try:
            # Check title
            if not self.title or not self.title.strip():
                logger.debug("Content validation failed: empty title")
                return False
            
            # Check main content
            if not self.main_content or not self.main_content.strip():
                logger.debug("Content validation failed: empty main content")
                return False
            
            # Check minimum content length
            if len(self.main_content) < 100:
                logger.debug(f"Content validation failed: content too short ({len(self.main_content)} chars)")
                return False
            
            # Check minimum word count
            if self.metrics.word_count < 20:
                logger.debug(f"Content validation failed: too few words ({self.metrics.word_count})")
                return False
            
            logger.debug("Content validation passed all checks")
            return True
            
        except Exception as e:
            logger.error(f"Error during content validation: {e}")
            return False
    
    def is_substantial_content(self) -> bool:
        """Check if content is substantial enough for analysis"""
        try:
            is_substantial = (
                self.metrics.word_count >= 200 and
                self.metrics.content_density_score >= 3.0 and
                len(self.headings) >= 1
            )
            
            logger.debug(f"Content substantiality check: {is_substantial} "
                        f"(words: {self.metrics.word_count}, density: {self.metrics.content_density_score}, "
                        f"headings: {len(self.headings)})")
            
            return is_substantial
            
        except Exception as e:
            logger.warning(f"Error checking content substantiality: {e}")
            return False
    
    def get_content_summary(self, max_words: int = 50) -> str:
        """Get a brief summary of the content with error handling"""
        try:
            if not self.main_content:
                return ""
            
            words = self.main_content.split()
            if len(words) <= max_words:
                return self.main_content
            
            summary_words = words[:max_words]
            summary = ' '.join(summary_words) + '...'
            
            logger.debug(f"Generated content summary: {len(summary_words)} words")
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate content summary: {e}")
            return self.main_content[:200] + "..." if self.main_content else ""
    
    def extract_key_phrases(self, min_length: int = 3) -> List[str]:
        """Extract potential key phrases from headings and content with error handling"""
        try:
            phrases = []
            
            # Extract from headings
            if self.headings:
                for heading in self.headings:
                    try:
                        if heading and len(heading.split()) >= min_length:
                            phrases.append(heading.strip())
                    except Exception as e:
                        logger.warning(f"Error processing heading '{heading}': {e}")
            
            # Extract from meta keywords
            if self.meta_keywords:
                try:
                    phrases.extend([kw.strip() for kw in self.meta_keywords 
                                  if kw and kw.strip()])
                except Exception as e:
                    logger.warning(f"Error processing meta keywords: {e}")
            
            # Remove duplicates and empty phrases
            unique_phrases = list(set([p for p in phrases if p]))
            
            logger.debug(f"Extracted {len(unique_phrases)} key phrases")
            return unique_phrases
            
        except Exception as e:
            logger.error(f"Failed to extract key phrases: {e}")
            return []


@dataclass
class ScrapingResult:
    """
    Result of a scraping operation.
    Encapsulates success/failure with proper error handling.
    """
    content: Optional[ScrapedContent]
    status: ScrapingStatus
    error_message: Optional[str]
    processing_time_seconds: float
    attempt_count: int = 1
    
    def __post_init__(self):
        """Validate result consistency"""
        try:
            if self.status == ScrapingStatus.SUCCESS and self.content is None:
                logger.warning("Inconsistent scraping result: SUCCESS status but no content")
                self.status = ScrapingStatus.FAILED
                self.error_message = "Success status but no content available"
                
        except Exception as e:
            logger.error(f"Error validating scraping result: {e}")
    
    @property
    def is_success(self) -> bool:
        """Check if scraping was successful"""
        return self.status == ScrapingStatus.SUCCESS and self.content is not None
    
    @property
    def is_retryable(self) -> bool:
        """Check if this failure can be retried with smart retry logic"""
        try:
            retryable_statuses = {
                ScrapingStatus.TIMEOUT,
                ScrapingStatus.NETWORK_ERROR,
                ScrapingStatus.FAILED
            }
            
            is_retryable = (
                self.status in retryable_statuses and 
                self.attempt_count < 3 and
                self.processing_time_seconds < 120  # Don't retry if it took too long
            )
            
            logger.debug(f"Retry check: status={self.status}, attempts={self.attempt_count}, "
                        f"time={self.processing_time_seconds}s, retryable={is_retryable}")
            
            return is_retryable
            
        except Exception as e:
            logger.error(f"Error checking if result is retryable: {e}")
            return False
    
    def with_retry(self, new_attempt_count: int) -> 'ScrapingResult':
        """Create a new result with updated attempt count"""
        try:
            logger.debug(f"Creating retry result: attempt {new_attempt_count}")
            
            return ScrapingResult(
                content=self.content,
                status=self.status,
                error_message=self.error_message,
                processing_time_seconds=self.processing_time_seconds,
                attempt_count=new_attempt_count
            )
            
        except Exception as e:
            logger.error(f"Failed to create retry result: {e}")
            return self


@dataclass
class ScrapingRequest:
    """
    Value object representing a scraping request.
    Contains all parameters needed for scraping operation.
    """
    url: str
    timeout_seconds: int = 30
    max_content_length: int = 1_000_000  # 1MB limit
    follow_redirects: bool = True
    extract_images: bool = False
    extract_scripts: bool = False
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate request parameters with comprehensive error handling"""
        try:
            logger.debug(f"Validating scraping request for URL: {self.url}")
            
            # Validate timeout
            if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
                raise ValueError(f"Invalid timeout: {self.timeout_seconds}. Must be positive number.")
            
            if self.timeout_seconds > 300:  # 5 minutes max
                logger.warning(f"Timeout {self.timeout_seconds}s is very high, capping at 300s")
                self.timeout_seconds = 300
            
            # Validate content length
            if not isinstance(self.max_content_length, int) or self.max_content_length <= 0:
                raise ValueError(f"Invalid max_content_length: {self.max_content_length}")
            
            # Validate headers
            if self.custom_headers:
                for key, value in self.custom_headers.items():
                    if not isinstance(key, str) or not isinstance(value, str):
                        raise ValueError(f"Invalid header: {key}={value}. Must be strings.")
            
            logger.debug(f"Scraping request validation successful")
            
        except Exception as e:
            logger.error(f"Scraping request validation failed: {e}")
            raise
    
    @property
    def url_info(self) -> URLInfo:
        """Get URL info for this request with error handling"""
        try:
            return URLInfo.from_url(self.url)
        except Exception as e:
            logger.error(f"Failed to create URLInfo from request URL {self.url}: {e}")
            raise ValueError(f"Invalid URL in scraping request: {self.url}") from e


# ==========================================
# CONTENT ANALYSIS MODELS (WBS 2.2)
# ==========================================

class AnalysisStatus(Enum):
    """Status of analysis process"""
    PENDING = "pending"
    PROCESSING = "processing"
    SCRAPING = "scraping"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisType(Enum):
    """Types of analysis supported"""
    COMPREHENSIVE = "comprehensive"
    BASIC = "basic"
    SEO_FOCUSED = "seo_focused"
    UX_FOCUSED = "ux_focused"
    CONTENT_QUALITY = "content_quality"
    COMPETITIVE = "competitive"


class QualityLevel(Enum):
    """Quality level for analysis operations"""
    FAST = "fast"
    BALANCED = "balanced" 
    HIGH = "high"


@dataclass
class AnalysisMetrics:
    """Quantified analysis metrics"""
    content_quality_score: float  # 1-10
    seo_score: float              # 1-10
    ux_score: float               # 1-10
    readability_score: float      # 1-10
    engagement_score: float       # 1-10
    overall_score: float          # 1-10

    def __post_init__(self):
        """Validate scores are within 1-10 range"""
        scores = [
            self.content_quality_score, self.seo_score, self.ux_score,
            self.readability_score, self.engagement_score, self.overall_score
        ]
        for score in scores:
            if not 1.0 <= score <= 10.0:
                raise ValueError(f"Score {score} must be between 1.0 and 10.0")


@dataclass
class AnalysisInsights:
    """Structured analysis insights"""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    # Metadata
    url: str
    analysis_id: str
    analysis_type: AnalysisType
    status: AnalysisStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    content_id: Optional[int] = None  # Database ID for scraped content
    
    # Source Data
    scraped_content: Optional[ScrapedContent] = None
    
    # LLM Response (using Any to avoid circular imports)
    llm_response: Optional[Any] = None
    
    # Structured Analysis
    executive_summary: str = ""
    metrics: Optional[AnalysisMetrics] = None
    insights: Optional[AnalysisInsights] = None
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Performance Data
    processing_time: float = 0.0
    cost: float = 0.0
    provider_used: str = ""
    
    # Error Handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate analysis result data"""
        if not self.url:
            raise ValueError("URL is required for analysis result")
        if not self.analysis_id:
            raise ValueError("Analysis ID is required")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")
        if self.cost < 0:
            raise ValueError("Cost cannot be negative")
            
    @property
    def success(self):
        """Determine if the analysis was successful"""
        return self.status == AnalysisStatus.COMPLETED


# Export all classes for proper imports
__all__ = [
    'ContentType', 'ImageType', 'ImageFormat', 'ScrapingStatus',
    'URLInfo', 'ImageInfo', 'ExtractedImage', 'ScrapedContent',
    'AnalysisType', 'AnalysisStatus', 'AnalysisMetrics', 'AnalysisInsights',
    'AnalysisResult'
]


