"""Utility module to fix ScrapedContent issues in tests"""

from datetime import datetime
import uuid
from typing import List, Optional
from enum import Enum

# Import necessary classes from the project
from src.domain.models import (
    URLInfo,
    ContentMetrics,
    ScrapedContent,
    ContentType,
    ScrapingStatus,
    AnalysisType
)

# Create enums for tests since they appear to be defined locally in test files
class QualityLevel(Enum):
    """Quality level for analysis"""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"

def create_test_url_info(url: str = "https://example.com") -> URLInfo:
    """Create a test URLInfo instance"""
    return URLInfo.from_url(url)

def create_test_content_metrics(
    word_count: int = 500,
    sentence_count: int = 25,
    paragraph_count: int = 5,
    link_count: int = 10,
    image_count: int = 2,
    heading_count: int = 3,
    reading_time_minutes: float = 2.5
) -> ContentMetrics:
    """Create a test ContentMetrics instance"""
    return ContentMetrics(
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        link_count=link_count,
        image_count=image_count,
        heading_count=heading_count,
        reading_time_minutes=reading_time_minutes
    )

def create_test_scraped_content(
    url: str = "https://example.com",
    title: str = "Example Title",
    main_content: str = "This is example content for testing purposes. It needs to be long enough to pass validation. " * 5,
    headings: Optional[List[str]] = None,
    links: Optional[List[str]] = None,
    meta_description: str = "Example meta description",
    meta_keywords: Optional[List[str]] = None,
    content_type: ContentType = ContentType.ARTICLE,
    status: ScrapingStatus = ScrapingStatus.SUCCESS,
    scraped_at: Optional[datetime] = None
) -> ScrapedContent:
    """Create a test ScrapedContent instance with proper fields"""
    if headings is None:
        headings = ["Header 1", "Header 2"]
    if links is None:
        links = ["https://example.com/link1", "https://example.com/link2"]
    if meta_keywords is None:
        meta_keywords = ["test", "example"]
    if scraped_at is None:
        scraped_at = datetime.now()

    url_info = create_test_url_info(url)
    metrics = create_test_content_metrics()  # Use metrics, not content_metrics
    
    return ScrapedContent(
        url_info=url_info,
        title=title,
        headings=headings,
        main_content=main_content,
        links=links,
        meta_description=meta_description,
        meta_keywords=meta_keywords,
        content_type=content_type,
        metrics=metrics,  # This is the key fix - use metrics not content_metrics
        scraped_at=scraped_at,
        status=status
    )

def create_test_analysis_result(
    analysis_id: str = None,
    url: str = "https://example.com",
    title: str = "Example Title",
    summary: str = "This is a test summary",
    key_points: Optional[List[str]] = None,
    sentiment: str = "positive",
    content_type: str = "article",
    topics: Optional[List[str]] = None,
    readability_score: float = 8.5,
    created_at: Optional[datetime] = None,
    llm_response: str = "Test LLM response",
    quality_level: QualityLevel = QualityLevel.STANDARD,
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
):
    """Create a test AnalysisResult instance with proper fields"""
    if analysis_id is None:
        analysis_id = str(uuid.uuid4())
    if key_points is None:
        key_points = ["Key point 1", "Key point 2"]
    if topics is None:
        topics = ["Topic 1", "Topic 2"]
    if created_at is None:
        created_at = datetime.now()

    # Create a dictionary instead since we don't have direct access to the AnalysisResult class
    return {
        "analysis_id": analysis_id,
        "url": url,
        "title": title,
        "summary": summary,
        "key_points": key_points,
        "sentiment": sentiment,
        "content_type": content_type,
        "topics": topics,
        "readability_score": readability_score,
        "created_at": created_at,
        "llm_response": llm_response,
        "quality_level": quality_level.value,
        "analysis_type": analysis_type.value
    }

def create_test_scraping_request(
    url: str = "https://example.com",
    timeout_seconds: int = 30,
    max_retries: int = 3,
    quality_level: QualityLevel = QualityLevel.STANDARD
):
    """Create a test ScrapingRequest instance with proper fields"""
    return {
        "url": url,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "quality_level": quality_level.value
    }

# Simple test to verify the fix works
def test_create_scraped_content():
    """Test that ScrapedContent can be created with the right parameters"""
    content = create_test_scraped_content()
    assert content.title == "Example Title"
    assert content.url_info.url == "https://example.com"
    assert isinstance(content.metrics, ContentMetrics)
    assert content.metrics.word_count == 500
    
    # Additional test to make sure we're fixing the right issue
    metrics = create_test_content_metrics(word_count=1000)
    content2 = ScrapedContent(
        url_info=create_test_url_info(),
        title="Test",
        headings=["H1"],
        main_content="This content is long enough to pass validation. It needs to be longer than 50 characters to satisfy the requirements.",
        links=["https://example.com"],
        meta_description="Description",
        meta_keywords=["key"],
        content_type=ContentType.ARTICLE,
        metrics=metrics,  # Using metrics parameter, not content_metrics
        scraped_at=datetime.now(),
        status=ScrapingStatus.SUCCESS
    )
    assert content2.metrics.word_count == 1000
