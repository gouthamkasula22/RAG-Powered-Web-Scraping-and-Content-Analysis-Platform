"""
Unit tests for domain models.
Tests the core business logic without external dependencies.
"""
import pytest
from datetime import datetime
from src.domain import (
    URLInfo, ScrapedContent, ScrapingResult, ScrapingRequest,
    ContentMetrics, ContentType, ScrapingStatus,
    ContentClassificationService, ContentQualityService, URLAnalysisService,
    ValidationError, ContentQualityError
)


class TestURLInfo:
    """Test URLInfo value object"""
    
    def test_create_from_valid_url(self):
        """Test creating URLInfo from valid URL"""
        url = "https://example.com/article/test?param=value"
        url_info = URLInfo.from_url(url)
        
        assert url_info.url == url
        assert url_info.domain == "example.com"
        assert url_info.is_secure is True
        assert url_info.path == "/article/test"
        assert url_info.query_params == {"param": "value"}
    
    def test_create_from_invalid_url(self):
        """Test creating URLInfo from invalid URL"""
        with pytest.raises(ValueError):
            URLInfo.from_url("not-a-url")
    
    def test_base_domain_extraction(self):
        """Test base domain extraction"""
        url_info = URLInfo.from_url("https://blog.subdomain.example.com/test")
        assert url_info.base_domain == "example.com"
    
    def test_is_root_page(self):
        """Test root page detection"""
        root_urls = [
            "https://example.com/",
            "https://example.com/index.html",
            "https://example.com/home"
        ]
        
        for url in root_urls:
            url_info = URLInfo.from_url(url)
            assert url_info.is_root_page is True
        
        non_root = URLInfo.from_url("https://example.com/article/test")
        assert non_root.is_root_page is False


class TestContentMetrics:
    """Test ContentMetrics calculations"""
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation"""
        content = "This is a test article. It has multiple sentences! And some content."
        links = ["https://example.com", "https://test.com"]
        headings = ["Main Title", "Subtitle"]
        
        metrics = ContentMetrics.calculate(content, links, headings)
        
        assert metrics.word_count == 12
        assert metrics.sentence_count == 3
        assert metrics.link_count == 2
        assert metrics.heading_count == 2
        assert metrics.reading_time_minutes == 0.06  # 12/200
    
    def test_calculate_empty_content(self):
        """Test metrics with empty content"""
        metrics = ContentMetrics.calculate("", [], [])
        
        assert metrics.word_count == 0
        assert metrics.sentence_count == 1  # Minimum to avoid division by zero
        assert metrics.paragraph_count == 1
        assert metrics.link_count == 0
    
    def test_content_density_score(self):
        """Test content density score calculation"""
        content = " ".join(["word"] * 200)  # 200 words
        metrics = ContentMetrics.calculate(content, [], ["Heading"])
        
        score = metrics.content_density_score
        assert 0 <= score <= 10


class TestScrapedContent:
    """Test ScrapedContent aggregate"""
    
    def create_valid_content(self):
        """Helper to create valid scraped content"""
        url_info = URLInfo.from_url("https://example.com/article")
        # Create content that meets quality standards (200+ words for substantial content)
        content = " ".join(["This is a test article with enough content for analysis."] * 25)  # 250 words
        # Reduce links to avoid link density issues (1 link per 250 words = 0.4/100 > 0.15 limit)
        metrics = ContentMetrics.calculate(content, [], ["Test Heading"])  # No links
        
        return ScrapedContent(
            url_info=url_info,
            title="Test Article",
            headings=["Test Heading"],
            main_content=content,
            links=[],  # No links to avoid density issues
            meta_description="Test description",
            meta_keywords=["test", "article"],
            content_type=ContentType.ARTICLE,
            metrics=metrics,
            scraped_at=datetime.now(),
            status=ScrapingStatus.SUCCESS
        )
    
    def test_valid_content_creation(self):
        """Test creating valid scraped content"""
        content = self.create_valid_content()
        assert content.is_valid_content() is True
        assert content.is_substantial_content() is True
    
    def test_invalid_content_creation(self):
        """Test creating invalid scraped content raises error"""
        url_info = URLInfo.from_url("https://example.com/test")
        metrics = ContentMetrics.calculate("", [], [])
        
        with pytest.raises(ValueError):
            ScrapedContent(
                url_info=url_info,
                title="",  # Empty title
                headings=[],
                main_content="",  # Empty content
                links=[],
                meta_description=None,
                meta_keywords=[],
                content_type=ContentType.UNKNOWN,
                metrics=metrics,
                scraped_at=datetime.now(),
                status=ScrapingStatus.SUCCESS
            )
    
    def test_content_summary(self):
        """Test content summary generation"""
        content = self.create_valid_content()
        summary = content.get_content_summary(max_words=5)
        
        assert len(summary.split()) <= 6  # 5 words + "..."
        assert summary.endswith("...")
    
    def test_extract_key_phrases(self):
        """Test key phrase extraction"""
        content = self.create_valid_content()
        phrases = content.extract_key_phrases(min_length=2)  # Lower requirement
        
        # Check that we get some phrases from headings and keywords
        assert len(phrases) > 0
        # Check that meta keywords are included
        assert "test" in phrases
        assert "article" in phrases


class TestScrapingResult:
    """Test ScrapingResult"""
    
    def test_successful_result(self):
        """Test successful scraping result"""
        content = TestScrapedContent().create_valid_content()
        result = ScrapingResult(
            content=content,
            status=ScrapingStatus.SUCCESS,
            error_message=None,
            processing_time_seconds=1.5
        )
        
        assert result.is_success is True
        assert result.is_retryable is False
    
    def test_failed_retryable_result(self):
        """Test failed but retryable result"""
        result = ScrapingResult(
            content=None,
            status=ScrapingStatus.TIMEOUT,
            error_message="Connection timeout",
            processing_time_seconds=30.0,
            attempt_count=1
        )
        
        assert result.is_success is False
        assert result.is_retryable is True
    
    def test_retry_creation(self):
        """Test creating retry result"""
        original = ScrapingResult(
            content=None,
            status=ScrapingStatus.FAILED,
            error_message="Test error",
            processing_time_seconds=5.0,
            attempt_count=1
        )
        
        retry = original.with_retry(2)
        assert retry.attempt_count == 2
        assert retry.status == ScrapingStatus.FAILED


class TestScrapingRequest:
    """Test ScrapingRequest validation"""
    
    def test_valid_request(self):
        """Test creating valid scraping request"""
        request = ScrapingRequest(
            url="https://example.com/test",
            timeout_seconds=30,
            max_content_length=1000000
        )
        
        assert request.url == "https://example.com/test"
        assert request.timeout_seconds == 30
        url_info = request.url_info
        assert url_info.domain == "example.com"
    
    def test_invalid_timeout(self):
        """Test invalid timeout validation"""
        with pytest.raises(ValueError):
            ScrapingRequest(
                url="https://example.com/test",
                timeout_seconds=0  # Invalid timeout
            )
    
    def test_timeout_capping(self):
        """Test timeout gets capped at maximum"""
        request = ScrapingRequest(
            url="https://example.com/test",
            timeout_seconds=500  # Too high
        )
        
        assert request.timeout_seconds == 300  # Should be capped


class TestContentClassificationService:
    """Test content classification service"""
    
    def test_classify_article_content(self):
        """Test article classification"""
        content = TestScrapedContent().create_valid_content()
        content.url_info = URLInfo.from_url("https://example.com/article/test")
        
        classification = ContentClassificationService.classify_content(content)
        assert classification == ContentType.ARTICLE
    
    def test_classify_homepage(self):
        """Test homepage classification"""
        content = TestScrapedContent().create_valid_content()
        content.url_info = URLInfo.from_url("https://example.com/")
        
        classification = ContentClassificationService.classify_content(content)
        assert classification == ContentType.HOMEPAGE
    
    def test_classify_news_content(self):
        """Test news classification"""
        content = TestScrapedContent().create_valid_content()
        content.title = "Breaking News: Important Update"
        content.url_info = URLInfo.from_url("https://example.com/news/update")
        
        classification = ContentClassificationService.classify_content(content)
        assert classification == ContentType.NEWS


class TestContentQualityService:
    """Test content quality service"""
    
    def test_validate_quality_success(self):
        """Test successful quality validation"""
        content = TestScrapedContent().create_valid_content()
        
        # Should not raise exception
        ContentQualityService.validate_content_quality(content)
    
    def test_validate_quality_failure(self):
        """Test quality validation failure"""
        # Create content that will fail validation without triggering ScrapedContent validation
        url_info = URLInfo.from_url("https://example.com/test")
        # Content long enough for ScrapedContent but too short for quality standards
        short_content = " ".join(["Short content."] * 15)  # ~30 words, >100 chars but <100 words
        metrics = ContentMetrics.calculate(short_content, [], [])
        
        content = ScrapedContent(
            url_info=url_info,
            title="Test",
            headings=[],
            main_content=short_content,
            links=[],
            meta_description=None,
            meta_keywords=[],
            content_type=ContentType.ARTICLE,
            metrics=metrics,
            scraped_at=datetime.now(),
            status=ScrapingStatus.SUCCESS
        )
        
        with pytest.raises(ContentQualityError):
            ContentQualityService.validate_content_quality(content)
    
    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        content = TestScrapedContent().create_valid_content()
        score = ContentQualityService.calculate_quality_score(content)
        
        assert 0 <= score <= 10


class TestURLAnalysisService:
    """Test URL analysis service"""
    
    def test_is_likely_content_page(self):
        """Test content page likelihood"""
        # Positive case
        content_url = URLInfo.from_url("https://example.com/article/interesting-topic")
        assert URLAnalysisService.is_likely_content_page(content_url) is True
        
        # Negative case
        api_url = URLInfo.from_url("https://example.com/api/data.json")
        assert URLAnalysisService.is_likely_content_page(api_url) is False
    
    def test_estimate_scraping_complexity(self):
        """Test scraping complexity estimation"""
        # Simple site
        simple_url = URLInfo.from_url("https://user.github.io/blog/post")
        assert URLAnalysisService.estimate_scraping_complexity(simple_url) == "simple"
        
        # Complex site
        complex_url = URLInfo.from_url("https://twitter.com/user/status/123")
        assert URLAnalysisService.estimate_scraping_complexity(complex_url) == "complex"
        
        # Regular site
        regular_url = URLInfo.from_url("https://example.com/article")
        complexity = URLAnalysisService.estimate_scraping_complexity(regular_url)
        assert complexity in ["simple", "moderate", "complex"]
    
    def test_suggest_timeout(self):
        """Test timeout suggestion"""
        url = URLInfo.from_url("https://example.com/test")
        timeout = URLAnalysisService.suggest_timeout(url)
        
        assert isinstance(timeout, int)
        assert 10 <= timeout <= 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
