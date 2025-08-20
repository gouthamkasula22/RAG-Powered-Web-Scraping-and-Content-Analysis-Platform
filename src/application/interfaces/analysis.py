"""
Analysis and reporting interfaces following SOLID principles.
These interfaces define contracts for content analysis and report generation.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.domain import ScrapedContent, ContentMetrics


class IContentAnalyzer(ABC):
    """
    Interface for analyzing scraped content.
    Implements Strategy Pattern for different analysis approaches.
    """
    
    @abstractmethod
    async def analyze_content(self, content: ScrapedContent) -> Dict[str, Any]:
        """
        Analyze scraped content and extract insights.
        
        Args:
            content: Scraped content to analyze
            
        Returns:
            Dict containing analysis results
        """
        pass
    
    @abstractmethod
    def get_analyzer_name(self) -> str:
        """
        Get name/identifier for this analyzer.
        
        Returns:
            str: Analyzer name
        """
        pass
    
    @abstractmethod
    def supports_content_type(self, content_type: str) -> bool:
        """
        Check if this analyzer can handle the given content type.
        
        Args:
            content_type: Content type to check
            
        Returns:
            bool: True if analyzer can handle this content type
        """
        pass


class ILLMService(ABC):
    """
    Interface for LLM-based content analysis.
    Abstracts LLM provider implementation details.
    """
    
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing sentiment analysis results
        """
        pass
    
    @abstractmethod
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract key keywords from text.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        pass
    
    @abstractmethod
    async def summarize_content(self, text: str, max_length: int = 500) -> str:
        """
        Generate summary of given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in characters
            
        Returns:
            str: Generated summary
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get name/identifier for the LLM model.
        
        Returns:
            str: Model name
        """
        pass


class IReportGenerator(ABC):
    """
    Interface for generating analysis reports.
    Implements Strategy Pattern for different report formats.
    """
    
    @abstractmethod
    async def generate_report(
        self, 
        content: ScrapedContent, 
        analysis: Dict[str, Any],
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate analysis report.
        
        Args:
            content: Original scraped content
            analysis: Analysis results
            format_type: Output format (json, html, pdf, etc.)
            
        Returns:
            Dict containing generated report
        """
        pass
    
    @abstractmethod
    def supports_format(self, format_type: str) -> bool:
        """
        Check if this generator supports the given format.
        
        Args:
            format_type: Format to check
            
        Returns:
            bool: True if format is supported
        """
        pass
    
    @abstractmethod
    def get_generator_name(self) -> str:
        """
        Get name/identifier for this report generator.
        
        Returns:
            str: Generator name
        """
        pass


class IMetricsCollector(ABC):
    """
    Interface for collecting and storing metrics.
    Follows Single Responsibility Principle for metrics handling.
    """
    
    @abstractmethod
    async def record_scraping_metrics(self, url: str, metrics: ContentMetrics) -> None:
        """
        Record metrics for a scraping operation.
        
        Args:
            url: URL that was scraped
            metrics: Collected metrics
        """
        pass
    
    @abstractmethod
    async def record_analysis_metrics(self, analyzer_name: str, duration: float, success: bool) -> None:
        """
        Record metrics for an analysis operation.
        
        Args:
            analyzer_name: Name of analyzer used
            duration: Operation duration in seconds
            success: Whether operation succeeded
        """
        pass
    
    @abstractmethod
    async def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get metrics summary for specified time period.
        
        Args:
            hours: Number of hours to include in summary
            
        Returns:
            Dict containing metrics summary
        """
        pass
