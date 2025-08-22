"""
Reports API Router
WBS 2.4: Report generation and management endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
import uuid
from datetime import datetime

from ..models.requests import ReportGenerationRequest, ComparativeAnalysisRequest
from ..models.responses import ReportResponse, AnalysisResponse
from ..dependencies.services import get_service_container

router = APIRouter()

@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportGenerationRequest,
    service_container = Depends(get_service_container)
):
    """
    Generate a report for an analysis
    
    - **analysis_id**: ID of the analysis to generate report for
    - **report_format**: Output format (json, pdf, html, csv)
    - **include_raw_data**: Include raw analysis data
    - **custom_template**: Custom template name (optional)
    """
    
    try:
        report_service = service_container.get_report_service()
        
        # Generate report
        report = await report_service.generate_report(
            analysis_id=request.analysis_id,
            format_type=request.report_format,
            include_raw_data=request.include_raw_data,
            custom_template=request.custom_template
        )
        
        if not report:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis {request.analysis_id} not found"
            )
        
        # Create response
        return ReportResponse(
            report_id=str(uuid.uuid4()),
            analysis_id=request.analysis_id,
            report_format=request.report_format.value,
            content=report.get('content') if request.report_format.value in ['json', 'html'] else None,
            download_url=report.get('download_url') if request.report_format.value in ['pdf', 'csv'] else None,
            generated_at=datetime.utcnow(),
            file_size=report.get('file_size'),
            expires_at=report.get('expires_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )

@router.post("/comparative", response_model=ReportResponse)
async def generate_comparative_report(
    request: ComparativeAnalysisRequest,
    service_container = Depends(get_service_container)
):
    """
    Generate comparative analysis report
    
    - **analysis_ids**: List of analysis IDs to compare (2-5)
    - **comparison_dimensions**: Dimensions to include in comparison
    - **report_format**: Output format
    """
    
    try:
        report_service = service_container.get_report_service()
        
        # Generate comparative report
        report = await report_service.generate_comparative_report(
            analysis_ids=request.analysis_ids,
            comparison_dimensions=request.comparison_dimensions,
            format_type=request.report_format
        )
        
        if not report:
            raise HTTPException(
                status_code=404,
                detail="One or more analyses not found"
            )
        
        # Create response
        return ReportResponse(
            report_id=str(uuid.uuid4()),
            analysis_id=f"comparative_{len(request.analysis_ids)}_sites",
            report_format=request.report_format.value,
            content=report.get('content') if request.report_format.value in ['json', 'html'] else None,
            download_url=report.get('download_url') if request.report_format.value in ['pdf', 'csv'] else None,
            generated_at=datetime.utcnow(),
            file_size=report.get('file_size')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate comparative report: {str(e)}"
        )

@router.get("/templates")
async def list_report_templates(
    service_container = Depends(get_service_container)
):
    """List available report templates"""
    
    try:
        report_service = service_container.get_report_service()
        templates = await report_service.list_templates()
        
        return {
            "templates": templates,
            "total": len(templates),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list templates: {str(e)}"
        )

@router.get("/history")
async def get_report_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    analysis_id: Optional[str] = Query(default=None),
    service_container = Depends(get_service_container)
):
    """Get report generation history"""
    
    try:
        report_service = service_container.get_report_service()
        
        history = await report_service.get_report_history(
            limit=limit,
            offset=offset,
            analysis_id=analysis_id
        )
        
        return {
            "reports": history,
            "total": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get report history: {str(e)}"
        )

@router.delete("/reports/{report_id}")
async def delete_report(
    report_id: str,
    service_container = Depends(get_service_container)
):
    """Delete a generated report"""
    
    try:
        report_service = service_container.get_report_service()
        
        success = await report_service.delete_report(report_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Report {report_id} not found"
            )
        
        return {"message": f"Report {report_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete report: {str(e)}"
        )
