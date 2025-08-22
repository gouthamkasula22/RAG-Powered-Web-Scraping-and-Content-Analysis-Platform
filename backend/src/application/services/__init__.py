"""
Application Services - Service Layer Implementation
These services orchestrate domain operations and implement application use cases.
"""
import logging
from typing import Dict, Any, Optional
from ...domain import (
    ScrapingRequest, ScrapingResult, ScrapedContent, ContentMetrics,
    URLAnalysisService, ContentClassificationService, ContentQualityService,
    ValidationError, URLSecurityError, NetworkError
)
from ...application.interfaces.security import ISecurityService, IScrapingProxy
from ...application.interfaces.scraping import IWebScraper
from ...application.interfaces.analysis import IContentAnalyzer, ILLMService, IReportGenerator
from ...application.interfaces.configuration import IConfigurationService


class WebContentAnalysisService:
    """
    Main application service that orchestrates web content analysis operations.
    Implements Application Service pattern and coordinates between layers.
    """
    
    def __init__(
        self,
        security_service: ISecurityService,
        scraping_proxy: IScrapingProxy,
        content_analyzer: IContentAnalyzer,
        llm_service: ILLMService,
        report_generator: IReportGenerator,
        config_service: IConfigurationService
    ):
        self._security_service = security_service
        self._scraping_proxy = scraping_proxy
        self._content_analyzer = content_analyzer
        self._llm_service = llm_service
        self._report_generator = report_generator
        self._config_service = config_service
        
        # Domain services
        self._url_analysis_service = URLAnalysisService()
        self._classification_service = ContentClassificationService()
        self._quality_service = ContentQualityService()
        
        self._logger = logging.getLogger(__name__)
    
    async def analyze_url(self, url: str, analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main use case: Analyze content from a URL.
        
        Args:
            url: URL to analyze
            analysis_options: Optional analysis configuration
            
        Returns:
            Dict containing complete analysis results
            
        Raises:
            ValidationError: If URL is invalid
            SecurityError: If URL violates security policies
            ContentError: If content cannot be processed
        """
        try:
            self._logger.info(f"Starting analysis for URL: {url}")
            
            # Step 1: Validate URL and check security
            await self._security_service.validate_url(url)
            self._logger.debug(f"URL security validation passed: {url}")
            
            # Step 2: Analyze URL structure
            url_analysis = self._url_analysis_service.analyze_url(url)
            self._logger.debug(f"URL analysis complete: complexity={url_analysis.complexity_score}")
            
            # Step 3: Create scraping request
            scraping_config = self._config_service.get_scraping_config()
            request = ScrapingRequest(
                url=url,
                timeout=scraping_config.timeout,
                user_agent=scraping_config.user_agent,
                respect_robots_txt=scraping_config.respect_robots_txt
            )
            
            # Step 4: Scrape content securely
            scraping_result = await self._scraping_proxy.scrape_content(request)
            if not scraping_result.success:
                raise NetworkError(
                    message=f"Failed to scrape content: {scraping_result.error_message}",
                    details={"url": url, "error": scraping_result.error_message}
                )
            
            content = scraping_result.content
            self._logger.info(f"Content scraped successfully: {len(content.raw_html)} characters")
            
            # Step 5: Classify and assess content quality
            content_type = self._classification_service.classify_content(content)
            quality_metrics = self._quality_service.assess_quality(content)
            
            self._logger.debug(f"Content classified as: {content_type}, quality: {quality_metrics.overall_score}")
            
            # Step 6: Perform LLM-based analysis if requested
            llm_analysis = {}
            analysis_config = self._config_service.get_analysis_config()
            
            if analysis_options is None:
                analysis_options = {}
            
            if analysis_options.get("sentiment", analysis_config.enable_sentiment_analysis):
                llm_analysis["sentiment"] = await self._llm_service.analyze_sentiment(content.text_content)
                self._logger.debug("Sentiment analysis completed")
            
            if analysis_options.get("keywords", analysis_config.enable_keyword_extraction):
                max_keywords = analysis_options.get("max_keywords", analysis_config.max_keywords)
                llm_analysis["keywords"] = await self._llm_service.extract_keywords(
                    content.text_content, max_keywords
                )
                self._logger.debug(f"Extracted {len(llm_analysis['keywords'])} keywords")
            
            if analysis_options.get("summary", analysis_config.enable_content_summarization):
                max_length = analysis_options.get("summary_length", analysis_config.summary_max_length)
                llm_analysis["summary"] = await self._llm_service.summarize_content(
                    content.text_content, max_length
                )
                self._logger.debug("Content summarization completed")
            
            # Step 7: Generate comprehensive report
            analysis_results = {
                "url_analysis": {
                    "original_url": url,
                    "final_url": content.url,
                    "complexity_score": url_analysis.complexity_score,
                    "is_secure": url_analysis.is_secure,
                    "domain": url_analysis.domain
                },
                "content_classification": {
                    "type": content_type,
                    "confidence": 0.95  # Would be provided by actual classifier
                },
                "quality_metrics": {
                    "overall_score": quality_metrics.overall_score,
                    "readability_score": quality_metrics.readability_score,
                    "content_length": quality_metrics.content_length,
                    "structure_score": quality_metrics.structure_score,
                    "metadata_score": quality_metrics.metadata_score
                },
                "llm_analysis": llm_analysis,
                "metadata": {
                    "title": content.title,
                    "description": content.meta_description,
                    "language": content.language,
                    "scraped_at": content.scraped_at.isoformat(),
                    "processing_time": scraping_result.processing_time
                }
            }
            
            # Step 8: Generate final report
            report_format = analysis_options.get("report_format", "json")
            final_report = await self._report_generator.generate_report(
                content, analysis_results, report_format
            )
            
            self._logger.info(f"Analysis completed successfully for URL: {url}")
            return final_report
            
        except (ValidationError, URLSecurityError, NetworkError) as e:
            self._logger.error(f"Analysis failed for URL {url}: {e}")
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error during analysis of {url}: {e}")
            raise NetworkError(
                message="Unexpected error during content analysis",
                details={"url": url, "error": str(e)}
            )
    
    async def validate_url_only(self, url: str) -> Dict[str, Any]:
        """
        Use case: Validate URL without performing full analysis.
        
        Args:
            url: URL to validate
            
        Returns:
            Dict containing validation results
        """
        try:
            self._logger.debug(f"Validating URL: {url}")
            
            # Security validation
            await self._security_service.validate_url(url)
            
            # URL structure analysis
            url_analysis = self._url_analysis_service.analyze_url(url)
            
            return {
                "valid": True,
                "url_analysis": {
                    "complexity_score": url_analysis.complexity_score,
                    "is_secure": url_analysis.is_secure,
                    "domain": url_analysis.domain
                },
                "security_check": "passed"
            }
            
        except (ValidationError, URLSecurityError) as e:
            self._logger.warning(f"URL validation failed for {url}: {e}")
            return {
                "valid": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def get_content_preview(self, url: str) -> Dict[str, Any]:
        """
        Use case: Get basic content preview without full analysis.
        
        Args:
            url: URL to preview
            
        Returns:
            Dict containing content preview
        """
        try:
            self._logger.debug(f"Getting content preview for: {url}")
            
            # Security validation
            await self._security_service.validate_url(url)
            
            # Quick scrape with minimal processing
            scraping_config = self._config_service.get_scraping_config()
            request = ScrapingRequest(
                url=url,
                timeout=min(scraping_config.timeout, 10),  # Shorter timeout for preview
                user_agent=scraping_config.user_agent
            )
            
            scraping_result = await self._scraping_proxy.scrape_content(request)
            if not scraping_result.success:
                raise NetworkError(
                    message=f"Failed to preview content: {scraping_result.error_message}",
                    details={"url": url}
                )
            
            content = scraping_result.content
            content_type = self._classification_service.classify_content(content)
            
            # Basic preview info
            preview = {
                "title": content.title,
                "description": content.meta_description,
                "content_type": content_type,
                "content_length": len(content.text_content),
                "has_images": len(content.images) > 0,
                "has_links": len(content.links) > 0,
                "language": content.language,
                "final_url": content.url
            }
            
            self._logger.debug(f"Content preview generated for: {url}")
            return preview
            
        except (ValidationError, URLSecurityError, NetworkError) as e:
            self._logger.error(f"Preview failed for URL {url}: {e}")
            raise


class ScrapingOrchestrationService:
    """
    Service for orchestrating complex scraping operations.
    Handles batch processing and advanced scraping scenarios.
    """
    
    def __init__(
        self,
        scraping_proxy: IScrapingProxy,
        security_service: ISecurityService,
        config_service: IConfigurationService
    ):
        self._scraping_proxy = scraping_proxy
        self._security_service = security_service
        self._config_service = config_service
        self._logger = logging.getLogger(__name__)
    
    async def batch_scrape_urls(self, urls: list[str]) -> Dict[str, ScrapingResult]:
        """
        Scrape multiple URLs in batch.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            Dict mapping URLs to their scraping results
        """
        results = {}
        scraping_config = self._config_service.get_scraping_config()
        
        for url in urls:
            try:
                self._logger.debug(f"Batch scraping URL: {url}")
                
                # Validate each URL
                await self._security_service.validate_url(url)
                
                # Create request
                request = ScrapingRequest(
                    url=url,
                    timeout=scraping_config.timeout,
                    user_agent=scraping_config.user_agent,
                    respect_robots_txt=scraping_config.respect_robots_txt
                )
                
                # Scrape with delay between requests
                result = await self._scraping_proxy.scrape_content(request)
                results[url] = result
                
                # Respect delay between requests
                if scraping_config.delay_between_requests > 0:
                    import asyncio
                    await asyncio.sleep(scraping_config.delay_between_requests)
                
            except Exception as e:
                self._logger.error(f"Batch scraping failed for {url}: {e}")
                results[url] = ScrapingResult(
                    success=False,
                    content=None,
                    error_message=str(e),
                    status_code=0,
                    processing_time=0.0
                )
        
        self._logger.info(f"Batch scraping completed: {len(results)} URLs processed")
        return results
