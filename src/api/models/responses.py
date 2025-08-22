"""
API Response Models
WBS 2.4: Response schemas and data structures
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class AnalysisStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    
    analysis_id: str = Field(..., description="Unique analysis identifier")
    url: str = Field(..., description="Analyzed website URL")
    status: AnalysisStatusEnum = Field(..., description="Analysis status")
    
    # Core results
    executive_summary: Optional[str] = Field(None, description="Executive summary of findings")
    overall_score: Optional[float] = Field(None, ge=0, le=10, description="Overall score (0-10)")
    
    # Detailed metrics
    metrics: Optional[Dict[str, float]] = Field(None, description="Detailed analysis metrics")
    insights: Optional[Dict[str, List[str]]] = Field(None, description="Analysis insights")
    recommendations: Optional[List[str]] = Field(None, description="Improvement recommendations")
    
    # Technical details
    analysis_type: str = Field(..., description="Type of analysis performed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    cost: Optional[float] = Field(None, description="Analysis cost in USD")
    provider_used: Optional[str] = Field(None, description="AI provider used")
    
    # Timestamps
    created_at: datetime = Field(..., description="Analysis creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Analysis completion timestamp")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "abc123def456",
                "url": "https://example.com",
                "status": "completed",
                "executive_summary": "Website shows strong SEO performance...",
                "overall_score": 8.5,
                "metrics": {
                    "seo_score": 9.2,
                    "content_quality": 8.1,
                    "ux_score": 8.7,
                    "performance": 7.8
                },
                "analysis_type": "comprehensive",
                "processing_time": 45.2,
                "cost": 0.032,
                "created_at": "2025-08-21T10:30:00Z"
            }
        }

class BulkAnalysisResponse(BaseModel):
    """Response model for bulk analysis"""
    
    bulk_id: str = Field(..., description="Bulk analysis identifier")
    total_urls: int = Field(..., description="Total number of URLs")
    completed: int = Field(default=0, description="Number of completed analyses")
    failed: int = Field(default=0, description="Number of failed analyses")
    
    status: str = Field(..., description="Overall bulk analysis status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    results: List[AnalysisResponse] = Field(default=[], description="Individual analysis results")
    
    total_cost: float = Field(default=0.0, description="Total cost of bulk analysis")
    average_processing_time: Optional[float] = Field(None, description="Average processing time")

class ReportResponse(BaseModel):
    """Response model for generated reports"""
    
    report_id: str = Field(..., description="Unique report identifier")
    analysis_id: str = Field(..., description="Source analysis ID")
    report_format: str = Field(..., description="Report format")
    
    # Report content
    content: Optional[str] = Field(None, description="Report content (for text formats)")
    download_url: Optional[str] = Field(None, description="Download URL for binary formats")
    
    # Metadata
    generated_at: datetime = Field(..., description="Report generation timestamp")
    file_size: Optional[int] = Field(None, description="Report file size in bytes")
    expires_at: Optional[datetime] = Field(None, description="Report expiration timestamp")

class ErrorResponse(BaseModel):
    """Standard error response model"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid URL format provided",
                "details": {"field": "url", "value": "invalid-url"},
                "timestamp": "2025-08-21T10:30:00Z"
            }
        }

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    services: Dict[str, Dict[str, Any]] = Field(..., description="Individual service statuses")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")

class AnalysisStatusResponse(BaseModel):
    """Response for analysis status checks"""
    
    analysis_id: str = Field(..., description="Analysis identifier")
    status: AnalysisStatusEnum = Field(..., description="Current analysis status")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    current_step: Optional[str] = Field(None, description="Current processing step")
    error_message: Optional[str] = Field(None, description="Error message if failed")
