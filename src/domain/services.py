"""
Domain services for business logic that doesn't belong to any single entity.
These are stateless services that operate on domain objects with comprehensive error handling.
"""
from typing import List, Set, Optional
import re
import logging

# Use standard logging until loguru is installed
logger = logging.getLogger(__name__)

# Import our domain models and exceptions
from .models import ScrapedContent, ContentType, URLInfo, ContentMetrics
from .exceptions import (
    ContentQualityError, 
    UnsupportedContentTypeError,
    ValidationError,
    ErrorSeverity
)


class ContentClassificationService:
    """
    Domain service for classifying content types.
    Implements business rules for content categorization with error handling.
    """
    
    # Keywords that indicate different content types
    ARTICLE_INDICATORS = {
        'author', 'published', 'article', 'story', 'report', 'analysis',
        'journalist', 'correspondent', 'by', 'written', 'opinion'
    }
    
    NEWS_INDICATORS = {
        'news', 'breaking', 'update', 'bulletin', 'press', 'release',
        'headline', 'latest', 'developing', 'alert', 'announced'
    }
    
    PRODUCT_INDICATORS = {
        'price', 'buy', 'product', 'cart', 'purchase', 'add to cart', 'shop',
        'order', 'checkout', 'inventory', 'stock', 'sale', 'discount'
    }
    
    BLOG_INDICATORS = {
        'blog', 'post', 'comment', 'share', 'tag', 'category',
        'personal', 'diary', 'thoughts', 'opinion', 'lifestyle'
    }
    
    DOC_INDICATORS = {
        'documentation', 'guide', 'tutorial', 'manual', 'reference', 'api',
        'how-to', 'instructions', 'help', 'docs', 'readme', 'wiki'
    }
    
    FORUM_INDICATORS = {
        'forum', 'discussion', 'thread', 'reply', 'topic', 'community',
        'question', 'answer', 'solved', 'user', 'member'
    }
    
    @classmethod
    def classify_content(cls, content: ScrapedContent) -> ContentType:
        """
        Classify content based on various signals with comprehensive error handling.
        Business logic for content type determination.
        """
        try:
            logger.debug(f"Classifying content for URL: {content.url_info.url}")
            
            # Prepare text for analysis with error handling
            text_to_analyze = cls._prepare_analysis_text(content)
            url_path = content.url_info.path.lower()
            
            # URL-based classification (highest priority)
            url_classification = cls._classify_by_url(url_path, text_to_analyze)
            if url_classification != ContentType.UNKNOWN:
                logger.debug(f"Content classified by URL: {url_classification}")
                return url_classification
            
            # Homepage detection
            if content.url_info.is_root_page:
                logger.debug("Content classified as HOMEPAGE (root page)")
                return ContentType.HOMEPAGE
            
            # Content-based classification
            content_classification = cls._classify_by_content(text_to_analyze)
            logger.debug(f"Content classified by content analysis: {content_classification}")
            
            return content_classification
            
        except Exception as e:
            logger.error(f"Failed to classify content for {content.url_info.url}: {e}")
            return ContentType.UNKNOWN
    
    @classmethod
    def _prepare_analysis_text(cls, content: ScrapedContent) -> str:
        """Prepare text for analysis with safe string handling"""
        try:
            text_parts = []
            
            # Add title
            if content.title:
                text_parts.append(content.title)
            
            # Add headings
            if content.headings:
                text_parts.extend([h for h in content.headings if h])
            
            # Add content sample (first 500 chars to avoid performance issues)
            if content.main_content:
                text_parts.append(content.main_content[:500])
            
            # Add meta description
            if content.meta_description:
                text_parts.append(content.meta_description)
            
            # Join and clean text
            full_text = ' '.join(text_parts).lower()
            
            # Remove extra whitespace and special characters for analysis
            cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
            
            logger.debug(f"Prepared {len(cleaned_text)} characters for analysis")
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"Error preparing analysis text: {e}")
            return ""
    
    @classmethod
    def _classify_by_url(cls, url_path: str, text_content: str) -> ContentType:
        """Classify content based on URL patterns"""
        try:
            # Blog/Article patterns
            if cls._matches_patterns(url_path, ['/blog/', '/post/', '/article/', '/story/']):
                if cls._contains_keywords(text_content, cls.NEWS_INDICATORS):
                    return ContentType.NEWS
                elif cls._contains_keywords(text_content, cls.BLOG_INDICATORS):
                    return ContentType.BLOG_POST
                else:
                    return ContentType.ARTICLE
            
            # Product patterns
            if cls._matches_patterns(url_path, ['/product/', '/item/', '/shop/', '/store/']):
                return ContentType.PRODUCT
            
            # Documentation patterns
            if cls._matches_patterns(url_path, ['/docs/', '/documentation/', '/guide/', '/api/', '/help/']):
                return ContentType.DOCUMENTATION
            
            # Forum patterns
            if cls._matches_patterns(url_path, ['/forum/', '/community/', '/discussion/', '/thread/']):
                return ContentType.FORUM
            
            # News patterns
            if cls._matches_patterns(url_path, ['/news/', '/press/', '/media/']):
                return ContentType.NEWS
            
            return ContentType.UNKNOWN
            
        except Exception as e:
            logger.warning(f"Error in URL-based classification: {e}")
            return ContentType.UNKNOWN
    
    @classmethod
    def _classify_by_content(cls, text_content: str) -> ContentType:
        """Classify content based on textual analysis"""
        try:
            if not text_content:
                return ContentType.UNKNOWN
            
            # Calculate keyword matches for each category
            classifications = {
                ContentType.PRODUCT: cls._contains_keywords(text_content, cls.PRODUCT_INDICATORS),
                ContentType.NEWS: cls._contains_keywords(text_content, cls.NEWS_INDICATORS),
                ContentType.DOCUMENTATION: cls._contains_keywords(text_content, cls.DOC_INDICATORS),
                ContentType.FORUM: cls._contains_keywords(text_content, cls.FORUM_INDICATORS),
                ContentType.BLOG_POST: cls._contains_keywords(text_content, cls.BLOG_INDICATORS),
                ContentType.ARTICLE: cls._contains_keywords(text_content, cls.ARTICLE_INDICATORS),
            }
            
            # Return the classification with the highest score
            for content_type, has_indicators in classifications.items():
                if has_indicators:
                    return content_type
            
            return ContentType.UNKNOWN
            
        except Exception as e:
            logger.warning(f"Error in content-based classification: {e}")
            return ContentType.UNKNOWN
    
    @staticmethod
    def _matches_patterns(text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given patterns with error handling"""
        try:
            return any(pattern in text for pattern in patterns)
        except Exception as e:
            logger.warning(f"Error matching patterns: {e}")
            return False
    
    @staticmethod
    def _contains_keywords(text: str, keywords: Set[str]) -> bool:
        """Check if text contains any of the keywords with error handling"""
        try:
            if not text or not keywords:
                return False
            
            # Split text into words and check intersection
            text_words = set(text.lower().split())
            matches = keywords.intersection(text_words)
            
            return len(matches) > 0
            
        except Exception as e:
            logger.warning(f"Error checking keywords: {e}")
            return False


class ContentQualityService:
    """
    Domain service for assessing content quality.
    Implements business rules for content quality evaluation with detailed error handling.
    """
    
    # Configurable quality standards
    MIN_WORD_COUNT = 100
    MIN_READING_TIME = 0.5  # minutes
    MAX_LINK_DENSITY = 0.15  # links per 100 words
    MIN_SUBSTANTIAL_WORDS = 200
    MIN_DENSITY_SCORE = 3.0
    MIN_HEADINGS = 1
    
    @classmethod
    def validate_content_quality(cls, content: ScrapedContent) -> None:
        """
        Validate that content meets quality standards.
        Raises ContentQualityError if standards are not met.
        """
        try:
            logger.debug(f"Validating content quality for: {content.url_info.url}")
            
            metrics = content.metrics
            quality_issues = []
            
            # Check word count
            if metrics.word_count < cls.MIN_WORD_COUNT:
                quality_issues.append(f"Word count too low: {metrics.word_count} < {cls.MIN_WORD_COUNT}")
            
            # Check reading time
            if metrics.reading_time_minutes < cls.MIN_READING_TIME:
                quality_issues.append(f"Reading time too short: {metrics.reading_time_minutes:.1f}min < {cls.MIN_READING_TIME}min")
            
            # Check link density (too many links = low quality)
            link_density = metrics.link_count / max(metrics.word_count / 100, 1)
            if link_density > cls.MAX_LINK_DENSITY:
                quality_issues.append(f"Link density too high: {link_density:.2f} > {cls.MAX_LINK_DENSITY}")
            
            # Check for empty title or content
            if not content.title.strip():
                quality_issues.append("Empty or whitespace-only title")
            
            if not content.main_content.strip():
                quality_issues.append("Empty or whitespace-only content")
            
            # If there are quality issues, raise exception with details
            if quality_issues:
                error_message = f"Content quality validation failed: {'; '.join(quality_issues)}"
                
                raise ContentQualityError(
                    message=error_message,
                    quality_metrics={
                        'word_count': metrics.word_count,
                        'reading_time_minutes': metrics.reading_time_minutes,
                        'link_density': link_density,
                        'title_length': len(content.title.strip()),
                        'content_length': len(content.main_content.strip())
                    },
                    minimum_standards={
                        'min_word_count': cls.MIN_WORD_COUNT,
                        'min_reading_time': cls.MIN_READING_TIME,
                        'max_link_density': cls.MAX_LINK_DENSITY
                    },
                    context={'url': content.url_info.url}
                )
            
            logger.debug("Content quality validation passed")
            
        except ContentQualityError:
            raise  # Re-raise quality errors
        except Exception as e:
            logger.error(f"Error during content quality validation: {e}")
            raise ContentQualityError(
                message=f"Failed to validate content quality: {e}",
                context={'url': content.url_info.url},
                inner_exception=e
            )
    
    @classmethod
    def calculate_quality_score(cls, content: ScrapedContent) -> float:
        """
        Calculate overall content quality score (0-10) with comprehensive metrics.
        Business logic for quality assessment.
        """
        try:
            logger.debug(f"Calculating quality score for: {content.url_info.url}")
            
            # Start with base validation
            try:
                cls.validate_content_quality(content)
                base_score = 2.0  # Base score for passing validation
            except ContentQualityError:
                logger.debug("Content failed basic quality validation")
                return 0.0
            
            metrics = content.metrics
            score = base_score
            
            # Word count score (0-3 points)
            if metrics.word_count >= 1000:
                score += 3.0
            elif metrics.word_count >= 500:
                score += 2.0
            elif metrics.word_count >= cls.MIN_WORD_COUNT:
                score += 1.0
            
            # Structure score (0-2 points)
            if metrics.heading_count >= 5:
                score += 2.0
            elif metrics.heading_count >= 3:
                score += 1.5
            elif metrics.heading_count >= 1:
                score += 1.0
            
            # Content density score (0-3 points)
            density = metrics.content_density_score
            if density >= 8:
                score += 3.0
            elif density >= 6:
                score += 2.0
            elif density >= 4:
                score += 1.0
            
            # Readability score (0-2 points)
            try:
                avg_sentence_length = metrics.word_count / metrics.sentence_count
                if 10 <= avg_sentence_length <= 20:  # Optimal range
                    score += 2.0
                elif 8 <= avg_sentence_length <= 25:  # Acceptable range
                    score += 1.0
            except ZeroDivisionError:
                logger.warning("No sentences found for readability calculation")
            
            # Cap at 10.0
            final_score = min(10.0, score)
            
            logger.debug(f"Quality score calculated: {final_score:.2f}")
            return final_score
            
        except Exception as e:
            logger.error(f"Failed to calculate quality score: {e}")
            return 0.0
    
    @classmethod
    def is_suitable_for_analysis(cls, content: ScrapedContent) -> bool:
        """
        Determine if content is suitable for LLM analysis.
        More permissive than quality validation.
        """
        try:
            metrics = content.metrics
            
            # Basic requirements for analysis
            has_minimum_content = (
                metrics.word_count >= 50 and  # Lower threshold for analysis
                len(content.title.strip()) > 0 and
                len(content.main_content.strip()) > 100
            )
            
            # Check content type suitability
            unsuitable_types = {ContentType.UNKNOWN}
            is_suitable_type = content.content_type not in unsuitable_types
            
            result = has_minimum_content and is_suitable_type
            
            logger.debug(f"Content suitability for analysis: {result} "
                        f"(words: {metrics.word_count}, type: {content.content_type})")
            
            return result
            
        except Exception as e:
            logger.warning(f"Error checking content suitability: {e}")
            return False


class URLAnalysisService:
    """
    Domain service for analyzing URL characteristics.
    Helps determine scraping strategy and content expectations.
    """
    
    @classmethod
    def is_likely_content_page(cls, url_info: URLInfo) -> bool:
        """
        Determine if URL is likely to contain substantial content.
        Business rule for content page identification.
        """
        try:
            logger.debug(f"Analyzing URL for content likelihood: {url_info.url}")
            
            path = url_info.path.lower()
            
            # Positive indicators
            content_indicators = [
                '/article/', '/post/', '/blog/', '/news/', '/story/',
                '/guide/', '/tutorial/', '/review/', '/analysis/',
                '/documentation/', '/docs/', '/help/'
            ]
            
            if any(indicator in path for indicator in content_indicators):
                logger.debug("URL has positive content indicators")
                return True
            
            # Negative indicators
            non_content_indicators = [
                '/api/', '/admin/', '/login/', '/register/', '/cart/',
                '/checkout/', '/search/', '/category/', '/tag/',
                '/static/', '/assets/', '/images/', '/css/', '/js/',
                '.json', '.xml', '.pdf', '.jpg', '.png', '.gif', '.css', '.js'
            ]
            
            if any(indicator in path for indicator in non_content_indicators):
                logger.debug("URL has negative content indicators")
                return False
            
            # Check if it's a deep page (more likely to have content)
            path_depth = len([p for p in path.split('/') if p])
            is_deep = path_depth >= 2
            
            # Check for query parameters (might indicate dynamic content)
            has_query_params = bool(url_info.query_params)
            
            result = is_deep and not has_query_params
            
            logger.debug(f"Content likelihood analysis: {result} "
                        f"(depth: {path_depth}, query_params: {has_query_params})")
            
            return result
            
        except Exception as e:
            logger.warning(f"Error analyzing URL content likelihood: {e}")
            return True  # Default to trying to scrape
    
    @classmethod
    def estimate_scraping_complexity(cls, url_info: URLInfo) -> str:
        """
        Estimate scraping complexity based on URL characteristics.
        Returns: 'simple', 'moderate', 'complex'
        """
        try:
            logger.debug(f"Estimating scraping complexity for: {url_info.url}")
            
            domain = url_info.domain.lower()
            
            # Simple sites (usually static content)
            simple_patterns = [
                'github.io', 'blogspot.com', 'wordpress.com', 
                'medium.com', 'substack.com', 'ghost.io'
            ]
            if any(pattern in domain for pattern in simple_patterns):
                logger.debug("Classified as simple scraping")
                return 'simple'
            
            # Complex sites (heavy JS, dynamic content, anti-scraping)
            complex_patterns = [
                'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
                'reddit.com', 'discord.com', 'slack.com', 'notion.so'
            ]
            if any(pattern in domain for pattern in complex_patterns):
                logger.debug("Classified as complex scraping")
                return 'complex'
            
            # Check for signs of dynamic content
            has_complex_query = len(url_info.query_params) > 3
            has_hash = '#' in url_info.url
            
            if has_complex_query or has_hash:
                logger.debug("Classified as moderate scraping (dynamic indicators)")
                return 'moderate'
            
            # Check domain complexity by TLD and structure
            if (domain.count('.') > 2 or  # Complex subdomain structure
                any(tld in domain for tld in ['.gov', '.edu', '.mil'])):  # Official sites
                logger.debug("Classified as moderate scraping (domain complexity)")
                return 'moderate'
            
            logger.debug("Classified as simple scraping (default)")
            return 'simple'
            
        except Exception as e:
            logger.warning(f"Error estimating scraping complexity: {e}")
            return 'moderate'  # Safe default
    
    @classmethod
    def suggest_timeout(cls, url_info: URLInfo) -> int:
        """
        Suggest appropriate timeout based on URL complexity analysis.
        Returns timeout in seconds.
        """
        try:
            complexity = cls.estimate_scraping_complexity(url_info)
            
            timeout_map = {
                'simple': 15,
                'moderate': 30,
                'complex': 60
            }
            
            suggested_timeout = timeout_map.get(complexity, 30)
            
            logger.debug(f"Suggested timeout for {complexity} site: {suggested_timeout}s")
            return suggested_timeout
            
        except Exception as e:
            logger.warning(f"Error suggesting timeout: {e}")
            return 30  # Safe default
