"""
API Request Models
WBS 2.4: Request validation and schema definitions
"""

from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum

class AnalysisTypeEnum(str, Enum):
    COMPREHENSIVE = "comprehensive"
    SEO_FOCUSED = "seo_focused"
    UX_FOCUSED = "ux_focused"
    CONTENT_QUALITY = "content_quality"
    TECHNICAL = "technical"

class QualityPreferenceEnum(str, Enum):
    SPEED = "speed"           # Free APIs only
    BALANCED = "balanced"     # Mix of free + premium
    PREMIUM = "premium"       # Premium APIs for best quality

class ReportFormatEnum(str, Enum):
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    CSV = "csv"

class AnalysisRequest(BaseModel):
    """Request model for website analysis"""
    
    url: HttpUrl = Field(
        ..., 
        description="Website URL to analyze",
        example="https://example.com"
    )
    
    analysis_type: AnalysisTypeEnum = Field(
        default=AnalysisTypeEnum.COMPREHENSIVE,
        description="Type of analysis to perform"
    )
    
    quality_preference: QualityPreferenceEnum = Field(
        default=QualityPreferenceEnum.BALANCED,
        description="Quality vs speed preference"
    )
    
    max_cost: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum cost for analysis in USD"
    )
    
    include_screenshots: bool = Field(
        default=False,
        description="Include website screenshots in analysis"
    )
    
    custom_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom analysis options"
    )
    
    @validator('url')
    def validate_url(cls, v):
        """Validate URL format and accessibility"""
        url_str = str(v)
        
        # Check for valid schemes
        if not url_str.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        
        # Additional URL validation can be added here
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://example.com",
                "analysis_type": "comprehensive",
                "quality_preference": "balanced",
                "max_cost": 0.05,
                "include_screenshots": False
            }
        }

class BulkAnalysisRequest(BaseModel):
    """Request model for bulk website analysis"""
    
    urls: List[HttpUrl] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of URLs to analyze (max 10)"
    )
    
    analysis_type: AnalysisTypeEnum = Field(
        default=AnalysisTypeEnum.COMPREHENSIVE,
        description="Type of analysis to perform"
    )
    
    quality_preference: QualityPreferenceEnum = Field(
        default=QualityPreferenceEnum.BALANCED,
        description="Quality vs speed preference"
    )
    
    max_cost_per_url: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum cost per URL in USD"
    )
    
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Processing priority (1=highest, 5=lowest)"
    )

class ReportGenerationRequest(BaseModel):
    """Request model for report generation"""
    
    analysis_id: str = Field(
        ...,
        description="Analysis ID to generate report for"
    )
    
    report_format: ReportFormatEnum = Field(
        default=ReportFormatEnum.JSON,
        description="Output format for the report"
    )
    
    include_raw_data: bool = Field(
        default=False,
        description="Include raw analysis data in report"
    )
    
    custom_template: Optional[str] = Field(
        default=None,
        description="Custom report template name"
    )

class ComparativeAnalysisRequest(BaseModel):
    """Request model for comparative analysis"""
    
    analysis_ids: List[str] = Field(
        ...,
        min_items=2,
        max_items=5,
        description="List of analysis IDs to compare (2-5 analyses)"
    )
    
    comparison_dimensions: List[str] = Field(
        default=["seo", "content", "ux", "performance"],
        description="Dimensions to compare"
    )
    
    report_format: ReportFormatEnum = Field(
        default=ReportFormatEnum.JSON,
        description="Output format for comparison report"
    )
