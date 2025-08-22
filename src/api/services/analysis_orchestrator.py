"""
Analysis Orchestrator
WBS 2.4: Orchestrates end-to-end analysis workflow
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import time

from ..models.requests import AnalysisRequest, BulkAnalysisRequest
from ..models.responses import AnalysisResponse, BulkAnalysisResponse
from ...domain.models import AnalysisType, AnalysisStatus

logger = logging.getLogger(__name__)

class AnalysisOrchestrator:
    """
    Orchestrates the complete analysis workflow:
    1. Web scraping
    2. Content analysis
    3. Report generation
    4. Result storage
    """
    
    def __init__(self, service_container):
        self.service_container = service_container
        self.active_analyses = {}  # Track running analyses
        self.bulk_analyses = {}    # Track bulk operations
    
    async def execute_analysis(self, analysis_id: str, request: AnalysisRequest):
        """Execute complete analysis workflow"""
        
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting analysis {analysis_id} for {request.url}")
            
            # Update status to processing
            self.active_analyses[analysis_id] = {
                "status": "processing",
                "progress": 10,
                "current_step": "initializing",
                "start_time": start_time,
                "url": str(request.url)
            }
            
            # Step 1: Web Scraping
            await self._update_progress(analysis_id, 20, "scraping_content")
            scraping_service = self.service_container.get_scraping_service()
            
            scraped_content = await scraping_service.scrape_url(
                url=str(request.url),
                include_images=request.include_screenshots,
                max_content_length=50000  # Limit content size
            )
            
            if not scraped_content:
                raise Exception("Failed to scrape website content")
            
            # Step 2: Content Analysis
            await self._update_progress(analysis_id, 50, "analyzing_content")
            analysis_service = self.service_container.get_analysis_service()
            
            # Convert request to analysis type
            analysis_type = AnalysisType(request.analysis_type.value)
            
            analysis_result = await analysis_service.analyze_content(
                scraped_content=scraped_content,
                analysis_type=analysis_type,
                quality_preference=request.quality_preference.value,
                max_cost=request.max_cost
            )
            
            # Step 3: Report Generation
            await self._update_progress(analysis_id, 80, "generating_report")
            report_service = self.service_container.get_report_service()
            
            report = await report_service.generate_report(
                analysis_result=analysis_result,
                template_type="comprehensive",
                format_type="json"
            )
            
            # Step 4: Store Results
            await self._update_progress(analysis_id, 95, "storing_results")
            storage_service = self.service_container.get_storage_service()
            
            # Create final result
            final_result = {
                "analysis_id": analysis_id,
                "url": str(request.url),
                "status": "completed",
                "analysis_type": request.analysis_type.value,
                "executive_summary": analysis_result.executive_summary,
                "overall_score": analysis_result.metrics.overall_score if analysis_result.metrics else None,
                "metrics": {
                    "seo_score": analysis_result.metrics.seo_score,
                    "content_quality": analysis_result.metrics.content_quality_score,
                    "ux_score": analysis_result.metrics.ux_score,
                    "performance": analysis_result.metrics.performance_score,
                    "readability": analysis_result.metrics.readability_score,
                    "engagement": analysis_result.metrics.engagement_score
                } if analysis_result.metrics else None,
                "insights": {
                    "strengths": analysis_result.insights.strengths,
                    "weaknesses": analysis_result.insights.weaknesses,
                    "opportunities": analysis_result.insights.opportunities,
                    "key_findings": analysis_result.insights.key_findings
                } if analysis_result.insights else None,
                "recommendations": analysis_result.insights.recommendations if analysis_result.insights else None,
                "processing_time": time.time() - start_time,
                "cost": getattr(analysis_result, 'cost', 0.0),
                "provider_used": getattr(analysis_result, 'provider_used', 'mixed'),
                "created_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat()
            }
            
            # Store in mock storage
            await storage_service.set(f"analysis:{analysis_id}", final_result)
            
            # Update final status
            await self._update_progress(analysis_id, 100, "completed")
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Analysis {analysis_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Analysis {analysis_id} failed: {e}")
            
            # Update error status
            self.active_analyses[analysis_id] = {
                "status": "failed",
                "progress": 0,
                "current_step": "failed",
                "error_message": str(e),
                "start_time": start_time,
                "url": str(request.url)
            }
            
            # Store error result
            error_result = {
                "analysis_id": analysis_id,
                "url": str(request.url),
                "status": "failed",
                "error_message": str(e),
                "analysis_type": request.analysis_type.value,
                "created_at": datetime.utcnow().isoformat(),
                "processing_time": time.time() - start_time
            }
            
            storage_service = self.service_container.get_storage_service()
            await storage_service.set(f"analysis:{analysis_id}", error_result)
    
    async def execute_bulk_analysis(self, bulk_id: str, request: BulkAnalysisRequest):
        """Execute bulk analysis for multiple URLs"""
        
        logger.info(f"🚀 Starting bulk analysis {bulk_id} for {len(request.urls)} URLs")
        
        # Initialize bulk tracking
        self.bulk_analyses[bulk_id] = {
            "total_urls": len(request.urls),
            "completed": 0,
            "failed": 0,
            "status": "processing",
            "results": [],
            "start_time": time.time()
        }
        
        # Process URLs concurrently (with limit)
        semaphore = asyncio.Semaphore(3)  # Limit concurrent analyses
        
        async def process_url(url):
            async with semaphore:
                analysis_id = str(uuid.uuid4())
                
                # Create individual analysis request
                individual_request = AnalysisRequest(
                    url=url,
                    analysis_type=request.analysis_type,
                    quality_preference=request.quality_preference,
                    max_cost=request.max_cost_per_url
                )
                
                try:
                    await self.execute_analysis(analysis_id, individual_request)
                    
                    # Get result
                    storage_service = self.service_container.get_storage_service()
                    result = await storage_service.get(f"analysis:{analysis_id}")
                    
                    self.bulk_analyses[bulk_id]["results"].append(result)
                    
                    if result.get("status") == "completed":
                        self.bulk_analyses[bulk_id]["completed"] += 1
                    else:
                        self.bulk_analyses[bulk_id]["failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Bulk analysis URL {url} failed: {e}")
                    self.bulk_analyses[bulk_id]["failed"] += 1
        
        # Execute all analyses
        await asyncio.gather(*[process_url(url) for url in request.urls])
        
        # Update final status
        self.bulk_analyses[bulk_id]["status"] = "completed"
        self.bulk_analyses[bulk_id]["processing_time"] = time.time() - self.bulk_analyses[bulk_id]["start_time"]
        
        logger.info(f"✅ Bulk analysis {bulk_id} completed")
    
    async def get_analysis_status(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis status and progress"""
        
        # Check active analyses
        if analysis_id in self.active_analyses:
            analysis = self.active_analyses[analysis_id]
            
            return {
                "analysis_id": analysis_id,
                "status": analysis["status"],
                "progress": analysis["progress"],
                "current_step": analysis["current_step"],
                "estimated_completion": self._estimate_completion(analysis),
                "error_message": analysis.get("error_message")
            }
        
        # Check storage for completed analyses
        storage_service = self.service_container.get_storage_service()
        result = await storage_service.get(f"analysis:{analysis_id}")
        
        if result:
            return {
                "analysis_id": analysis_id,
                "status": result["status"],
                "progress": 100 if result["status"] == "completed" else 0,
                "current_step": result["status"],
                "error_message": result.get("error_message")
            }
        
        return None
    
    async def get_bulk_analysis(self, bulk_id: str) -> Optional[BulkAnalysisResponse]:
        """Get bulk analysis status and results"""
        
        if bulk_id not in self.bulk_analyses:
            return None
        
        bulk_data = self.bulk_analyses[bulk_id]
        
        return BulkAnalysisResponse(
            bulk_id=bulk_id,
            total_urls=bulk_data["total_urls"],
            completed=bulk_data["completed"],
            failed=bulk_data["failed"],
            status=bulk_data["status"],
            results=[
                AnalysisResponse(**result) for result in bulk_data["results"]
            ],
            total_cost=sum(result.get("cost", 0) for result in bulk_data["results"]),
            average_processing_time=bulk_data.get("processing_time", 0) / max(bulk_data["total_urls"], 1)
        )
    
    async def cancel_analysis(self, analysis_id: str) -> bool:
        """Cancel a running analysis"""
        
        if analysis_id in self.active_analyses:
            self.active_analyses[analysis_id]["status"] = "cancelled"
            logger.info(f"🚫 Analysis {analysis_id} cancelled")
            return True
        
        return False
    
    async def _update_progress(self, analysis_id: str, progress: int, step: str):
        """Update analysis progress"""
        
        if analysis_id in self.active_analyses:
            self.active_analyses[analysis_id]["progress"] = progress
            self.active_analyses[analysis_id]["current_step"] = step
            
        # Small delay to simulate work
        await asyncio.sleep(0.1)
    
    def _estimate_completion(self, analysis: Dict[str, Any]) -> Optional[datetime]:
        """Estimate analysis completion time"""
        
        if analysis["status"] in ["completed", "failed", "cancelled"]:
            return None
        
        elapsed = time.time() - analysis["start_time"]
        progress = analysis["progress"]
        
        if progress > 0:
            total_estimated = elapsed * (100 / progress)
            remaining = total_estimated - elapsed
            return datetime.utcnow() + timedelta(seconds=remaining)
        
        return None
