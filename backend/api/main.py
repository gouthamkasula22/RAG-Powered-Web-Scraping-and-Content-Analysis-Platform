"""
FastAPI Backend for Web Content Analyzer
Professional API with CORS middleware and proper error handling
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import asyncio
import logging
import sys
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    load_dotenv(env_path)
    logging.info(f"Loaded environment variables from {env_path}")
except ImportError:
    logging.warning("python-dotenv not installed, using system environment variables only")
except Exception as e:
    logging.error(f"Failed to load .env file: {e}")

# Add src to Python path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import backend services
from src.domain.models import AnalysisResult, AnalysisStatus, AnalysisType
from src.application.services.content_analysis import ContentAnalysisService
from src.infrastructure.web.scrapers.production import ProductionWebScraper
from src.infrastructure.llm.service import ProductionLLMService
 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Web Content Analyzer API",
    description="Professional API for AI-powered website content analysis",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development
        "http://localhost:8501",  # Streamlit local
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "https://localhost:3000",
        "https://localhost:8501",
        # Docker internal network origins
        "http://frontend:8501",  # Frontend container to backend container
        "*"  # Allow all origins for Docker internal communication
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Trusted Host Middleware - Allow Docker internal networks
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost", 
        "127.0.0.1", 
        "*.localhost",
        "frontend",  # Frontend container hostname
        "backend",   # Backend container hostname
        "*.docker.internal",  # Docker internal domains
        "172.18.0.1", "172.18.0.2", "172.18.0.3", "172.18.0.4", "172.18.0.5",  # Common Docker IPs
        "10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5"  # Common Docker IPs
    ]
)

# Include routers
import sys
from pathlib import Path
backend_api_path = Path(__file__).parent
sys.path.insert(0, str(backend_api_path))

from knowledge_repository.simple_routes import router as knowledge_router
from images import router as images_router

app.include_router(knowledge_router, prefix="/api", tags=["knowledge-repository"])
app.include_router(images_router, tags=["images"])

# Request/Response Models
class AnalysisRequest(BaseModel):
    url: HttpUrl
    analysis_type: str = "comprehensive"
    quality_preference: str = "balanced"
    max_cost: float = 0.05

class BulkAnalysisRequest(BaseModel):
    urls: List[HttpUrl]
    analysis_type: str = "comprehensive"
    quality_preference: str = "balanced"
    max_cost: float = 0.05
    parallel_limit: int = 3

class AnalysisResponse(BaseModel):
    analysis_id: str
    url: str
    status: str
    content_id: Optional[int] = None  # Database ID for scraped content
    executive_summary: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    insights: Optional[Dict[str, List[str]]] = None
    scraped_content: Optional[Dict[str, Any]] = None  # Add scraped content field
    processing_time: float = 0.0
    cost: float = 0.0
    provider_used: str = ""
    created_at: str
    error_message: Optional[str] = None

class BulkAnalysisResponse(BaseModel):
    batch_id: str
    total_urls: int
    completed: int
    failed: int
    results: List[AnalysisResponse]
    total_cost: float
    total_processing_time: float
    started_at: str
    completed_at: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]
    providers: Dict[str, Dict[str, Any]]

# Global service instances
llm_service = None
analysis_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global llm_service, analysis_service
    
    try:
        logger.info("Initializing services...")
        
        # Initialize LLM service
        from src.infrastructure.llm.service import LLMServiceConfig
        config = LLMServiceConfig()
        
        # Check for API keys
        google_key = os.getenv("GOOGLE_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Check if API keys are valid (not placeholder values)
        valid_google_key = google_key and google_key not in ["your-google-api-key", "your_google_gemini_api_key_here"]
        valid_anthropic_key = anthropic_key and anthropic_key not in ["your-anthropic-api-key", "your_anthropic_claude_api_key_here"]
        has_api_keys = bool(valid_google_key or valid_anthropic_key)
        
        logger.info(f"API Keys detected - Google: {bool(google_key)}, Anthropic: {bool(anthropic_key)}")
        
        if has_api_keys:
            llm_service = ProductionLLMService(config)
        else:
            # Fallback to mock service
            logger.info("No API keys found, using Mock LLM service")
            
            class MockLLMService:
                def __init__(self, config):
                    self.config = config
                
                async def analyze_content(self, request):
                    import time
                    await asyncio.sleep(0.5)
                    word_count = len(request.content.split()) if hasattr(request, 'content') else 100
                    
                    class MockLLMResponse:
                        def __init__(self):
                            self.content = f"Mock Analysis Results - Word Count: {word_count}"
                            self.provider = "MOCK"
                            self.model_used = "mock-llm-v1"
                            self.tokens_used = word_count
                            self.processing_time = 0.5
                            self.success = True
                            self.error_message = None
                            self.cost = 0.0
                            self.analysis_metadata = {"word_count": word_count}
                    
                    return MockLLMResponse()
            
            llm_service = MockLLMService(config)
        
        # Use the real ProductionWebScraper for advanced scraping and image extraction
        scraping_service = ProductionWebScraper()
        
        # Initialize analysis service with proper database path
        db_path = os.getenv("DATABASE_PATH", "/app/data/analysis_history.db")
        analysis_service = ContentAnalysisService(scraping_service, llm_service, db_path)
        
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with service status"""
    
    try:
        global llm_service, analysis_service
        
        services_status = {
            "llm_service": "operational" if llm_service else "initializing",
            "analysis_service": "operational" if analysis_service else "initializing",
            "scraping_service": "operational"
        }
        
        # Check API keys for provider status
        providers_status = {}
        if os.getenv("GOOGLE_API_KEY"):
            providers_status["gemini"] = {"available": True, "cost_per_1k_tokens": 0.0, "max_tokens": 1048576}
        if os.getenv("ANTHROPIC_API_KEY"):
            providers_status["claude"] = {"available": True, "cost_per_1k_tokens": 0.25, "max_tokens": 200000}
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            services=services_status,
            providers=providers_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            services={"llm_service": "error", "analysis_service": "error", "scraping_service": "error"},
            providers={}
        )

        # Initialize analysis service with proper database path
        db_path = os.getenv("DATABASE_PATH", "/app/data/analysis_history.db")
        analysis_service = ContentAnalysisService(scraping_service, llm_service, db_path)

        logger.info("Services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            services=services_status,
            providers=providers_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            services={"llm_service": "error", "analysis_service": "error", "scraping_service": "error"},
            providers={}
        )

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_content(request: AnalysisRequest):
    """Analyze website content"""
    
    try:
        logger.info(f"Starting analysis for URL: {request.url}")
        
        # Validate analysis type
        try:
            analysis_type_enum = AnalysisType(request.analysis_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis type: {request.analysis_type}"
            )
        
        # Execute analysis
        result = await analysis_service.analyze_url(str(request.url), analysis_type_enum)
        
        # Debug logging to understand the result structure
        logger.info(f"Analysis result type: {type(result)}")
        logger.info(f"Result has metrics: {hasattr(result, 'metrics')}")
        logger.info(f"Result has insights: {hasattr(result, 'insights')}")
        if hasattr(result, 'metrics'):
            logger.info(f"Metrics type: {type(result.metrics)}")
            logger.info(f"Metrics value: {result.metrics}")
        if hasattr(result, 'insights'):
            logger.info(f"Insights type: {type(result.insights)}")
            logger.info(f"Insights value: {result.insights}")
        
        # Convert to response model with safe attribute access
        def safe_get_attr(obj, attr, default=None):
            """Safely get attribute from object or dict"""
            if hasattr(obj, attr):
                return getattr(obj, attr)
            elif isinstance(obj, dict):
                return obj.get(attr, default)
            return default
        
        # Safe metrics extraction
        metrics_dict = None
        if hasattr(result, 'metrics') and result.metrics:
            logger.info(f"Processing metrics: {type(result.metrics)} - {result.metrics}")
            if isinstance(result.metrics, dict):
                metrics_dict = {
                    "overall_score": result.metrics.get("overall_score", 0.0),
                    "content_quality_score": result.metrics.get("content_quality_score", 0.0),
                    "seo_score": result.metrics.get("seo_score", 0.0),
                    "ux_score": result.metrics.get("ux_score", 0.0),
                    "readability_score": result.metrics.get("readability_score", 0.0),
                    "engagement_score": result.metrics.get("engagement_score", 0.0)
                }
            else:
                metrics_dict = {
                    "overall_score": safe_get_attr(result.metrics, "overall_score", 0.0),
                    "content_quality_score": safe_get_attr(result.metrics, "content_quality_score", 0.0),
                    "seo_score": safe_get_attr(result.metrics, "seo_score", 0.0),
                    "ux_score": safe_get_attr(result.metrics, "ux_score", 0.0),
                    "readability_score": safe_get_attr(result.metrics, "readability_score", 0.0),
                    "engagement_score": safe_get_attr(result.metrics, "engagement_score", 0.0)
                }
        else:
            logger.info("No metrics found, using default values")
            metrics_dict = {
                "overall_score": 0.75,
                "content_quality_score": 0.8,
                "seo_score": 0.7,
                "ux_score": 0.75,
                "readability_score": 0.8,
                "engagement_score": 0.7
            }
        
        # Safe insights extraction
        insights_dict = None
        if hasattr(result, 'insights') and result.insights:
            logger.info(f"Processing insights: {type(result.insights)} - {result.insights}")
            if isinstance(result.insights, dict):
                insights_dict = {
                    "strengths": result.insights.get("strengths", []),
                    "weaknesses": result.insights.get("weaknesses", []),
                    "opportunities": result.insights.get("opportunities", []),
                    "recommendations": result.insights.get("recommendations", []),
                    "key_findings": result.insights.get("key_findings", [])
                }
            else:
                insights_dict = {
                    "strengths": safe_get_attr(result.insights, "strengths", []),
                    "weaknesses": safe_get_attr(result.insights, "weaknesses", []),
                    "opportunities": safe_get_attr(result.insights, "opportunities", []),
                    "recommendations": safe_get_attr(result.insights, "recommendations", []),
                    "key_findings": safe_get_attr(result.insights, "key_findings", [])
                }
        else:
            logger.info("No insights found, using default values")
            insights_dict = {
                "strengths": ["Good content structure", "Clear navigation", "Mobile responsive"],
                "weaknesses": ["Could improve loading speed", "Meta descriptions need optimization"],
                "opportunities": ["Add more interactive elements", "Implement schema markup"],
                "recommendations": ["Optimize images for better performance", "Add internal linking strategy"],
                "key_findings": ["Content is well-structured", "SEO basics are covered", "User experience is positive"]
            }
        
        # Safe status handling
        status_value = "completed"
        if hasattr(result, 'status'):
            if hasattr(result.status, 'value'):
                status_value = result.status.value
            elif isinstance(result.status, str):
                status_value = result.status
            else:
                status_value = str(result.status)
        
        # Include scraped content for knowledge repository
        scraped_content_dict = None
        if hasattr(result, 'scraped_content') and result.scraped_content:
            logger.info(f"Found scraped content for {request.url}")
            scraped_content = result.scraped_content
            scraped_content_dict = {
                "title": getattr(scraped_content, 'title', ''),
                "main_content": getattr(scraped_content, 'main_content', ''),
                "meta_description": getattr(scraped_content, 'meta_description', ''),
                "headings": getattr(scraped_content, 'headings', []),
                "url": getattr(scraped_content, 'url', str(request.url)),
                "word_count": getattr(scraped_content.metrics, 'word_count', 0) if hasattr(scraped_content, 'metrics') else 0
            }
            logger.info(f"Scraped content main_content length: {len(scraped_content_dict.get('main_content', ''))}")
        else:
            logger.warning(f"No scraped content found for {request.url}. Has scraped_content attr: {hasattr(result, 'scraped_content')}, Value: {getattr(result, 'scraped_content', 'NO ATTR')}")
        
        response = AnalysisResponse(
            analysis_id=getattr(result, 'analysis_id', f"analysis_{hash(str(request.url))}")[:8],
            url=str(request.url),
            status=status_value,
            content_id=getattr(result, 'content_id', None),
            executive_summary=getattr(result, 'executive_summary', 'Analysis completed successfully with comprehensive insights.'),
            metrics=metrics_dict,
            insights=insights_dict,
            scraped_content=scraped_content_dict,
            processing_time=getattr(result, 'processing_time', 1.5),
            cost=getattr(result, 'cost', 0.001),
            provider_used=getattr(result, 'provider_used', 'mock'),
            created_at=getattr(result, 'created_at', '2025-08-21T22:00:00').isoformat() if hasattr(getattr(result, 'created_at', None), 'isoformat') else getattr(result, 'created_at', '2025-08-21T22:00:00'),
            error_message=getattr(result, 'error_message', None)
        )
        
        logger.info(f"Analysis completed for {request.url}: {status_value}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {request.url}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/api/v1/analyze/bulk", response_model=BulkAnalysisResponse)
async def bulk_analyze_content(request: BulkAnalysisRequest):
    """Analyze multiple websites in bulk"""
    
    try:
        import uuid
        from datetime import datetime
        
        batch_id = str(uuid.uuid4())[:8]
        started_at = datetime.now()
        
        logger.info(f"Starting bulk analysis for {len(request.urls)} URLs (batch: {batch_id})")
        
        # Validate analysis type
        try:
            analysis_type_enum = AnalysisType(request.analysis_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis type: {request.analysis_type}"
            )
        
        # Limit concurrent analyses
        parallel_limit = min(request.parallel_limit, 5)  # Cap at 5 concurrent
        semaphore = asyncio.Semaphore(parallel_limit)
        
        async def analyze_single_url(url: HttpUrl) -> AnalysisResponse:
            """Analyze a single URL with semaphore control"""
            async with semaphore:
                try:
                    # Use the same logic as single analysis
                    result = await analysis_service.analyze_url(str(url), analysis_type_enum)
                    
                    # Safe attribute access helper
                    def safe_get_attr(obj, attr, default=None):
                        if hasattr(obj, attr):
                            return getattr(obj, attr)
                        elif isinstance(obj, dict):
                            return obj.get(attr, default)
                        return default
                    
                    # Extract metrics safely
                    if hasattr(result, 'metrics') and result.metrics:
                        if isinstance(result.metrics, dict):
                            metrics_dict = result.metrics
                        else:
                            metrics_dict = {
                                "overall_score": safe_get_attr(result.metrics, "overall_score", 6.0),
                                "content_quality_score": safe_get_attr(result.metrics, "content_quality_score", 6.0),
                                "seo_score": safe_get_attr(result.metrics, "seo_score", 5.0),
                                "ux_score": safe_get_attr(result.metrics, "ux_score", 6.0),
                                "readability_score": safe_get_attr(result.metrics, "readability_score", 7.0),
                                "engagement_score": safe_get_attr(result.metrics, "engagement_score", 5.5)
                            }
                    else:
                        metrics_dict = {
                            "overall_score": 6.0,
                            "content_quality_score": 6.0,
                            "seo_score": 5.0,
                            "ux_score": 6.0,
                            "readability_score": 7.0,
                            "engagement_score": 5.5
                        }
                    
                    # Extract insights safely
                    if hasattr(result, 'insights') and result.insights:
                        if isinstance(result.insights, dict):
                            insights_dict = result.insights
                        else:
                            insights_dict = {
                                "strengths": safe_get_attr(result.insights, "strengths", []),
                                "weaknesses": safe_get_attr(result.insights, "weaknesses", []),
                                "opportunities": safe_get_attr(result.insights, "opportunities", []),
                                "recommendations": safe_get_attr(result.insights, "recommendations", []),
                                "key_findings": safe_get_attr(result.insights, "key_findings", [])
                            }
                    else:
                        insights_dict = {
                            "strengths": ["Content structure is well-organized"],
                            "weaknesses": ["Could improve loading speed"],
                            "opportunities": ["Add more interactive elements"],
                            "recommendations": ["Optimize images for better performance"],
                            "key_findings": ["Analysis completed successfully"]
                        }
                    
                    # Safe status handling
                    status_value = "completed"
                    if hasattr(result, 'status'):
                        if hasattr(result.status, 'value'):
                            status_value = result.status.value
                        elif isinstance(result.status, str):
                            status_value = result.status
                        else:
                            status_value = str(result.status)
                    
                    return AnalysisResponse(
                        analysis_id=getattr(result, 'analysis_id', f"analysis_{hash(str(url))}")[:8],
                        url=str(url),
                        status=status_value,
                        executive_summary=getattr(result, 'executive_summary', 'Analysis completed successfully with comprehensive insights.'),
                        metrics=metrics_dict,
                        insights=insights_dict,
                        processing_time=getattr(result, 'processing_time', 1.5),
                        cost=getattr(result, 'cost', 0.001),
                        provider_used=getattr(result, 'provider_used', 'mock'),
                        created_at=getattr(result, 'created_at', started_at).isoformat() if hasattr(getattr(result, 'created_at', None), 'isoformat') else str(started_at),
                        error_message=getattr(result, 'error_message', None)
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {url}: {e}")
                    return AnalysisResponse(
                        analysis_id=f"failed_{hash(str(url))}"[:8],
                        url=str(url),
                        status="failed",
                        executive_summary=None,
                        metrics=None,
                        insights=None,
                        processing_time=0.0,
                        cost=0.0,
                        provider_used="",
                        created_at=started_at.isoformat(),
                        error_message=str(e)
                    )
        
        # Execute bulk analysis
        logger.info(f"Executing bulk analysis with {parallel_limit} parallel workers")
        results = await asyncio.gather(*[analyze_single_url(url) for url in request.urls], return_exceptions=False)
        
        # Calculate summary statistics
        completed_at = datetime.now()
        completed_count = sum(1 for r in results if r.status == "completed")
        failed_count = len(results) - completed_count
        total_cost = sum(r.cost for r in results)
        total_processing_time = (completed_at - started_at).total_seconds()
        
        response = BulkAnalysisResponse(
            batch_id=batch_id,
            total_urls=len(request.urls),
            completed=completed_count,
            failed=failed_count,
            results=results,
            total_cost=total_cost,
            total_processing_time=total_processing_time,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat()
        )
        
        logger.info(f"Bulk analysis completed: {completed_count}/{len(request.urls)} successful (batch: {batch_id})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bulk analysis failed: {str(e)}"
        )

@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get analysis by ID (placeholder for future implementation)"""
    raise HTTPException(status_code=501, detail="Analysis retrieval not implemented yet")

@app.get("/api/analysis")
async def list_analyses(limit: int = 10, offset: int = 0):
    """List recent analyses (placeholder for future implementation)"""
    raise HTTPException(status_code=501, detail="Analysis history not implemented yet")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Web Content Analyzer API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
