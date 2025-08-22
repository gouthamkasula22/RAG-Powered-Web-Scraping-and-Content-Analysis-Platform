"""
FastAPI Application Entry Point
WBS 2.4: Web Content Analyzer REST API
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Import routers
from src.api.routers import analysis, reports, health

# Import middleware
from src.api.middleware.rate_limiting import RateLimitingMiddleware
from src.api.middleware.error_handling import (
    ErrorHandlerMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware
)

# Import configuration
from src.api.config.settings import get_settings, configure_logging

# Import services
from src.api.dependencies.services import ServiceContainer

# Configure logging
settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)

# Service container instance
service_container = ServiceContainer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    
    # Startup
    logger.info("Starting Web Content Analyzer API...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    try:
        # Initialize services
        await service_container.initialize()
        logger.info("Services initialized successfully")
        
        # Store service container in app state
        app.state.services = service_container
        app.state.settings = settings
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Web Content Analyzer API...")
        
        try:
            await service_container.shutdown()
            logger.info("Services shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="REST API for analyzing web content with AI-powered insights",
    version=settings.app_version,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    lifespan=lifespan,
    debug=settings.debug
)

# Add middleware in correct order (last added = first executed)

# 1. Security headers (outermost)
if settings.security_headers_enabled:
    app.add_middleware(SecurityHeadersMiddleware)

# 2. Error handling
app.add_middleware(ErrorHandlerMiddleware)

# 3. Request logging
app.add_middleware(RequestLoggingMiddleware)

# 4. CORS middleware
cors_config = settings.get_cors_config()
app.add_middleware(
    CORSMiddleware,
    **cors_config
)

# 5. Rate limiting (innermost)
if settings.rate_limit_enabled:
    app.add_middleware(RateLimitingMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "docs": "/docs" if settings.is_development else "Documentation disabled in production",
        "health": "/health",
        "api_version": "v1"
    }

@app.get("/info")
async def app_info():
    """Application information endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "features": {
            "rate_limiting": settings.rate_limit_enabled,
            "security_headers": settings.security_headers_enabled,
            "cors_enabled": True,
            "api_docs": settings.is_development
        },
        "endpoints": {
            "health": "/health",
            "analysis": "/api/v1/analysis",
            "reports": "/api/v1/reports",
            "docs": "/docs" if settings.is_development else None
        }
    }

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "not_found",
            "message": f"The requested endpoint {request.url.path} was not found",
            "available_endpoints": [
                "/",
                "/info",
                "/health",
                "/api/v1/analysis",
                "/api/v1/reports"
            ] + (["/docs", "/redoc"] if settings.is_development else [])
        }
    )
