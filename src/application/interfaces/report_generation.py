"""
Report generation interfaces following SOLID principles.
Defines contracts for structured analysis report creation and management.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime

# For now, let's use Any for AnalysisResult to avoid import issues during testing
# In production, this would import from the proper domain models


class ReportFormat(Enum):
    """Supported report output formats"""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    CSV = "csv"


class ReportTemplate(Enum):
    """Available report templates"""
    COMPREHENSIVE = "comprehensive"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    COMPARATIVE = "comparative"
    SUMMARY = "summary"


class ReportPriority(Enum):
    """Report generation priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class IReportGenerator(ABC):
    """
    Interface for generating structured analysis reports.
    Supports multiple formats and templates with caching capabilities.
    """
    
    @abstractmethod
    async def generate_report(
        self,
        analysis_result: Any,
        template: ReportTemplate = ReportTemplate.COMPREHENSIVE,
        format_type: ReportFormat = ReportFormat.JSON,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured analysis report.
        
        Args:
            analysis_result: Analysis result to generate report from
            template: Report template to use
            format_type: Output format for the report
            options: Additional generation options
            
        Returns:
            Dict containing generated report and metadata
        """
        pass
    
    @abstractmethod
    async def generate_comparative_report(
        self,
        analysis_results: List[Any],
        template: ReportTemplate = ReportTemplate.COMPARATIVE,
        format_type: ReportFormat = ReportFormat.JSON,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comparative analysis report for multiple websites.
        
        Args:
            analysis_results: List of analysis results to compare
            template: Report template to use
            format_type: Output format for the report
            options: Additional generation options
            
        Returns:
            Dict containing comparative report and metadata
        """
        pass
    
    @abstractmethod
    async def generate_bulk_reports(
        self,
        analysis_results: List[Any],
        template: ReportTemplate = ReportTemplate.COMPREHENSIVE,
        format_type: ReportFormat = ReportFormat.JSON,
        priority: ReportPriority = ReportPriority.NORMAL
    ) -> List[Dict[str, Any]]:
        """
        Generate reports for multiple analyses with optimization.
        
        Args:
            analysis_results: List of analysis results
            template: Report template to use
            format_type: Output format for reports
            priority: Generation priority level
            
        Returns:
            List of generated reports
        """
        pass
    
    @abstractmethod
    def validate_template(self, template_content: str, template_type: ReportTemplate) -> bool:
        """
        Validate report template structure and syntax.
        
        Args:
            template_content: Template content to validate
            template_type: Type of template being validated
            
        Returns:
            bool: True if template is valid
        """
        pass
    
    @abstractmethod
    async def get_cached_report(self, analysis_id: str, template: ReportTemplate, format_type: ReportFormat) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached report if available.
        
        Args:
            analysis_id: Analysis identifier
            template: Report template used
            format_type: Report format
            
        Returns:
            Cached report or None if not found
        """
        pass


class IReportTemplateManager(ABC):
    """
    Interface for managing report templates.
    Handles template loading, validation, and customization.
    """
    
    @abstractmethod
    def load_template(self, template_type: ReportTemplate) -> str:
        """
        Load report template content.
        
        Args:
            template_type: Type of template to load
            
        Returns:
            Template content as string
        """
        pass
    
    @abstractmethod
    def register_custom_template(self, name: str, content: str, base_template: ReportTemplate) -> bool:
        """
        Register a custom report template.
        
        Args:
            name: Custom template name
            content: Template content
            base_template: Base template this extends
            
        Returns:
            bool: True if registration successful
        """
        pass
    
    @abstractmethod
    def get_available_templates(self) -> List[str]:
        """
        Get list of available template names.
        
        Returns:
            List of template names
        """
        pass


class IReportCache(ABC):
    """
    Interface for report caching mechanism.
    Provides efficient storage and retrieval of generated reports.
    """
    
    @abstractmethod
    async def store_report(
        self,
        analysis_id: str,
        template: ReportTemplate,
        format_type: ReportFormat,
        report_data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Store generated report in cache.
        
        Args:
            analysis_id: Analysis identifier
            template: Report template used
            format_type: Report format
            report_data: Generated report data
            ttl_seconds: Time to live in seconds
            
        Returns:
            bool: True if stored successfully
        """
        pass
    
    @abstractmethod
    async def retrieve_report(
        self,
        analysis_id: str,
        template: ReportTemplate,
        format_type: ReportFormat
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached report.
        
        Args:
            analysis_id: Analysis identifier
            template: Report template used
            format_type: Report format
            
        Returns:
            Cached report or None if not found
        """
        pass
    
    @abstractmethod
    async def invalidate_cache(self, analysis_id: str) -> bool:
        """
        Invalidate all cached reports for an analysis.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            bool: True if invalidation successful
        """
        pass
    
    @abstractmethod
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dict containing cache statistics
        """
        pass


class IComparativeAnalyzer(ABC):
    """
    Interface for comparative analysis between multiple websites.
    Identifies differentiators and similarities.
    """
    
    @abstractmethod
    async def compare_analyses(self, analyses: List[Any]) -> Dict[str, Any]:
        """
        Perform comparative analysis between multiple results.
        
        Args:
            analyses: List of analysis results to compare
            
        Returns:
            Dict containing comparative analysis results
        """
        pass
    
    @abstractmethod
    def identify_differentiators(self, analyses: List[Any], min_differentiators: int = 3) -> List[Dict[str, Any]]:
        """
        Identify key differentiators between websites.
        
        Args:
            analyses: List of analysis results
            min_differentiators: Minimum number of differentiators to identify
            
        Returns:
            List of differentiator insights
        """
        pass
    
    @abstractmethod
    def calculate_similarity_scores(self, analyses: List[Any]) -> Dict[str, float]:
        """
        Calculate similarity scores between analyses.
        
        Args:
            analyses: List of analysis results
            
        Returns:
            Dict mapping analysis pairs to similarity scores
        """
        pass
