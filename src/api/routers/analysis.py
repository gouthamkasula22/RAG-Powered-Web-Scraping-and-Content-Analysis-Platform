"""
Analysis API Router
WBS 2.4: Main analysis endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from typing import Optional, List
import uuid
import asyncio
from datetime import datetime

from ..models.requests import AnalysisRequest, BulkAnalysisRequest
from ..models.responses import AnalysisResponse, BulkAnalysisResponse, AnalysisStatusResponse
from ..dependencies.services import get_service_container
from ..services.analysis_orchestrator import AnalysisOrchestrator

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def create_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    service_container = Depends(get_service_container)
):
    """
    Create a new website analysis
    
    - **url**: Website URL to analyze
    - **analysis_type**: Type of analysis (comprehensive, seo_focused, etc.)
    - **quality_preference**: Speed vs quality trade-off
    - **max_cost**: Maximum cost limit in USD
    """
    
    try:
        # Create analysis orchestrator
        orchestrator = AnalysisOrchestrator(service_container)
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Create initial response
        response = AnalysisResponse(
            analysis_id=analysis_id,
            url=str(request.url),
            status="pending",
            analysis_type=request.analysis_type.value,
            created_at=datetime.utcnow()
        )
        
        # Start analysis in background
        background_tasks.add_task(
            orchestrator.execute_analysis,
            analysis_id=analysis_id,
            request=request
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create analysis: {str(e)}"
        )

@router.get("/analyze/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    service_container = Depends(get_service_container)
):
    """Get analysis results by ID"""
    
    try:
        # Get analysis from storage
        analysis_service = service_container.get_analysis_service()
        result = await analysis_service.get_analysis(analysis_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis {analysis_id} not found"
            )
        
        # Convert to response model
        return AnalysisResponse(
            analysis_id=result.analysis_id,
            url=result.url,
            status=result.status.value,
            executive_summary=result.executive_summary,
            overall_score=result.metrics.overall_score if result.metrics else None,
            metrics={
                "seo_score": result.metrics.seo_score,
                "content_quality": result.metrics.content_quality_score,
                "ux_score": result.metrics.ux_score,
                "performance": result.metrics.performance_score,
                "readability": result.metrics.readability_score,
                "engagement": result.metrics.engagement_score
            } if result.metrics else None,
            insights={
                "strengths": result.insights.strengths,
                "weaknesses": result.insights.weaknesses,
                "opportunities": result.insights.opportunities,
                "key_findings": result.insights.key_findings
            } if result.insights else None,
            recommendations=result.insights.recommendations if result.insights else None,
            analysis_type=result.analysis_type.value,
            processing_time=getattr(result, 'processing_time', None),
            cost=getattr(result, 'cost', None),
            provider_used=getattr(result, 'provider_used', None),
            created_at=result.created_at,
            completed_at=getattr(result, 'completed_at', None),
            error_message=result.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve analysis: {str(e)}"
        )

@router.get("/analyze/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    analysis_id: str,
    service_container = Depends(get_service_container)
):
    """Get analysis status and progress"""
    
    try:
        orchestrator = AnalysisOrchestrator(service_container)
        status_info = await orchestrator.get_analysis_status(analysis_id)
        
        if not status_info:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis {analysis_id} not found"
            )
        
        return AnalysisStatusResponse(**status_info)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analysis status: {str(e)}"
        )

@router.post("/analyze/bulk", response_model=BulkAnalysisResponse)
async def create_bulk_analysis(
    request: BulkAnalysisRequest,
    background_tasks: BackgroundTasks,
    service_container = Depends(get_service_container)
):
    """Create bulk analysis for multiple URLs"""
    
    try:
        orchestrator = AnalysisOrchestrator(service_container)
        bulk_id = str(uuid.uuid4())
        
        # Create bulk analysis response
        response = BulkAnalysisResponse(
            bulk_id=bulk_id,
            total_urls=len(request.urls),
            status="pending"
        )
        
        # Start bulk analysis in background
        background_tasks.add_task(
            orchestrator.execute_bulk_analysis,
            bulk_id=bulk_id,
            request=request
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create bulk analysis: {str(e)}"
        )

@router.get("/analyze/bulk/{bulk_id}", response_model=BulkAnalysisResponse)
async def get_bulk_analysis(
    bulk_id: str,
    service_container = Depends(get_service_container)
):
    """Get bulk analysis results"""
    
    try:
        orchestrator = AnalysisOrchestrator(service_container)
        result = await orchestrator.get_bulk_analysis(bulk_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Bulk analysis {bulk_id} not found"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve bulk analysis: {str(e)}"
        )

@router.delete("/analyze/{analysis_id}")
async def cancel_analysis(
    analysis_id: str,
    service_container = Depends(get_service_container)
):
    """Cancel a running analysis"""
    
    try:
        orchestrator = AnalysisOrchestrator(service_container)
        success = await orchestrator.cancel_analysis(analysis_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis {analysis_id} not found or cannot be cancelled"
            )
        
        return {"message": f"Analysis {analysis_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel analysis: {str(e)}"
        )

@router.get("/analyses", response_model=List[AnalysisResponse])
async def list_analyses(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    service_container = Depends(get_service_container)
):
    """List recent analyses with pagination"""
    
    try:
        analysis_service = service_container.get_analysis_service()
        analyses = await analysis_service.list_analyses(
            limit=limit,
            offset=offset,
            status=status
        )
        
        # Convert to response models
        return [
            AnalysisResponse(
                analysis_id=analysis.analysis_id,
                url=analysis.url,
                status=analysis.status.value,
                executive_summary=analysis.executive_summary,
                overall_score=analysis.metrics.overall_score if analysis.metrics else None,
                analysis_type=analysis.analysis_type.value,
                created_at=analysis.created_at,
                error_message=analysis.error_message
            )
            for analysis in analyses
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list analyses: {str(e)}"
        )
