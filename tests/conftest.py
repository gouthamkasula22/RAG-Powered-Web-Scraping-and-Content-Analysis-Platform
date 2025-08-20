"""
Pytest configuration and shared fixtures.
"""
import pytest
import logging
from datetime import datetime
from src.domain import URLInfo, ContentMetrics, ContentType, ScrapingStatus


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def sample_url_info():
    """Fixture providing a sample URLInfo object"""
    return URLInfo.from_url("https://example.com/test-article")


@pytest.fixture
def sample_content_metrics():
    """Fixture providing sample content metrics"""
    content = "This is a test article. " * 20  # 100 words
    return ContentMetrics.calculate(
        content=content,
        links=["https://example.com/link1", "https://example.com/link2"],
        headings=["Main Heading", "Sub Heading"]
    )


@pytest.fixture
def sample_scraped_content(sample_url_info, sample_content_metrics):
    """Fixture providing a complete ScrapedContent object"""
    from src.domain import ScrapedContent
    
    return ScrapedContent(
        url_info=sample_url_info,
        title="Sample Test Article",
        headings=["Main Heading", "Sub Heading"],
        main_content="This is a test article. " * 20,  # 100 words
        links=["https://example.com/link1", "https://example.com/link2"],
        meta_description="This is a test article for unit testing",
        meta_keywords=["test", "article", "sample"],
        content_type=ContentType.ARTICLE,
        metrics=sample_content_metrics,
        scraped_at=datetime.now(),
        status=ScrapingStatus.SUCCESS
    )
