"""
Report generation data models and schemas.
Defines structured formats for analysis reports with comprehensive metrics.
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid


class ReportType(Enum):
    """Types of analysis reports"""
    INDIVIDUAL = "individual"
    COMPARATIVE = "comparative"
    BULK = "bulk"
    SUMMARY = "summary"


class AnalysisDimension(Enum):
    """Analysis dimensions for comprehensive reporting"""
    CONTENT_QUALITY = "content_quality"
    SEO_OPTIMIZATION = "seo_optimization"
    USER_EXPERIENCE = "user_experience"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MOBILE_RESPONSIVENESS = "mobile_responsiveness"
    ENGAGEMENT_POTENTIAL = "engagement_potential"


class ReportFormat(Enum):
    """Supported report output formats"""
    HTML = "html"
    JSON = "json"
    PDF = "pdf"
    TEXT = "text"


class ReportTemplate(Enum):
    """Available report templates"""
    INDIVIDUAL_ANALYSIS = "individual_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_REPORT = "technical_report"
    BULK_SUMMARY = "bulk_summary"


class Priority(Enum):
    """Priority levels for recommendations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EffortLevel(Enum):
    """Effort levels for implementation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ImpactLevel(Enum):
    """Expected impact levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ImprovementCategory(Enum):
    """Categories for improvements"""
    SEO = "seo"
    CONTENT = "content"
    UX = "ux"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    SECURITY = "security"
    TECHNICAL = "technical"


@dataclass
class ReportMetadata:
    """Report generation metadata"""
    report_id: str
    generated_at: datetime
    generator_version: str
    template_used: str
    format_type: str
    generation_time_ms: float
    cache_hit: bool = False
    custom_options: Optional[Dict[str, Any]] = None


@dataclass
class DimensionScore:
    """Score for a specific analysis dimension"""
    dimension: AnalysisDimension
    score: float  # 0-10 scale
    weight: float  # Importance weight
    details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class ExecutiveSummary:
    """Executive summary with key insights"""
    summary_text: str  # <200 words
    key_metrics: Dict[str, Union[float, str]]
    top_strengths: List[str]  # Top 3 strengths
    critical_issues: List[str]  # Top 3 issues
    priority_actions: List[str]  # Top 3 actions
    overall_assessment: str  # Brief overall assessment


@dataclass
class ComparativeInsight:
    """Insight from comparative analysis"""
    insight_type: str
    title: str
    description: str
    affected_sites: List[str]
    significance_score: float  # 0-10
    supporting_data: Dict[str, Any]


@dataclass
class WebsiteComparison:
    """Comparison data for a single website"""
    url: str
    site_name: str
    overall_score: float
    dimension_scores: Dict[AnalysisDimension, float]
    rank_position: int
    strengths: List[str]
    weaknesses: List[str]
    differentiators: List[str]


@dataclass
class ImprovementItem:
    """Individual improvement recommendation"""
    category: ImprovementCategory
    title: str
    description: str
    priority: Priority
    effort_level: EffortLevel
    expected_impact: ImpactLevel
    timeline_weeks: int
    dependencies: List[str]
    success_metrics: List[str]


@dataclass
class AnalysisReport:
    """Individual website analysis report"""
    # Required fields
    metadata: ReportMetadata
    url: str
    site_name: str
    analysis_timestamp: datetime
    overall_score: float
    dimension_scores: Dict[AnalysisDimension, float]
    executive_summary: ExecutiveSummary
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    improvement_roadmap: List[ImprovementItem]
    technical_details: Dict[str, Any]
    appendices: Dict[str, Any]
    
    # Optional field with default
    report_type: ReportType = ReportType.INDIVIDUAL


@dataclass
class ComparativeReport:
    """Comparative analysis report for multiple websites"""
    # Report metadata
    metadata: ReportMetadata
    
    # Comparison overview
    comparison_summary: str
    websites_analyzed: int
    comparison_criteria: List[str]
    
    # Executive summary
    executive_summary: ExecutiveSummary
    
    # Website comparisons
    website_comparisons: List[WebsiteComparison]
    
    # Comparative insights
    key_differentiators: List[ComparativeInsight]
    similarity_analysis: Dict[str, Any]
    market_positioning: Dict[str, Any]
    
    # Rankings and performance
    overall_rankings: List[Dict[str, Any]]
    dimension_rankings: Dict[AnalysisDimension, List[Dict[str, Any]]]
    
    # Insights and recommendations
    comparative_insights: List[str]
    cross_site_recommendations: List[Dict[str, Any]]
    best_practices_identified: List[Dict[str, Any]]
    
    # Optional field with default
    report_type: ReportType = ReportType.COMPARATIVE


@dataclass
class BulkReportSummary:
    """Summary for bulk report generation"""
    # Report metadata  
    metadata: ReportMetadata
    
    # Bulk processing stats
    total_reports_generated: int
    successful_reports: int
    failed_reports: int
    total_urls_analyzed: int
    average_generation_time_ms: float
    common_issues_identified: List[str]
    bulk_insights: List[str]
    performance_metrics: Dict[str, str]
    
    # Optional fields with defaults
    report_type: ReportType = ReportType.BULK


class ReportGenerationRequest:
    """Request for report generation"""
    analysis_ids: List[str]
    template_name: str
    format_type: str
    options: Dict[str, Any]
    priority: str = "normal"
    callback_url: Optional[str] = None
    requester_id: str = ""
    
    def __post_init__(self):
        if not self.requester_id:
            self.requester_id = str(uuid.uuid4())


@dataclass
class ReportGenerationResponse:
    """Response from report generation"""
    request_id: str
    status: str  # "pending", "processing", "completed", "failed"
    report_id: Optional[str] = None
    download_url: Optional[str] = None
    error_message: Optional[str] = None
    generation_time_ms: Optional[float] = None
    file_size_bytes: Optional[int] = None
    expires_at: Optional[datetime] = None


# Template schemas for validation
COMPREHENSIVE_TEMPLATE_SCHEMA = {
    "required_sections": [
        "metadata",
        "executive_summary", 
        "overall_assessment",
        "dimension_analysis",
        "content_quality",
        "seo_optimization",
        "user_experience", 
        "accessibility",
        "performance",
        "security",
        "recommendations",
        "technical_details"
    ],
    "optional_sections": [
        "comparative_analysis",
        "industry_benchmarks",
        "improvement_roadmap"
    ]
}

EXECUTIVE_TEMPLATE_SCHEMA = {
    "required_sections": [
        "executive_summary",
        "key_metrics",
        "top_recommendations",
        "overall_score"
    ],
    "max_length": 1000,  # words
    "summary_max_length": 200  # words
}

COMPARATIVE_TEMPLATE_SCHEMA = {
    "required_sections": [
        "comparison_overview",
        "website_rankings",
        "key_differentiators", 
        "similarity_analysis",
        "recommendations"
    ],
    "min_websites": 2,
    "max_websites": 10
}

TEMPLATE_SCHEMAS = {
    "comprehensive": COMPREHENSIVE_TEMPLATE_SCHEMA,
    "executive": EXECUTIVE_TEMPLATE_SCHEMA,
    "comparative": COMPARATIVE_TEMPLATE_SCHEMA
}
