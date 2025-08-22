"""
Error Handling Middleware
WBS 2.4: Centralized error handling and logging
"""

import logging
import traceback
import time
import uuid
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Centralized error handling middleware
    Catches all unhandled exceptions and returns structured error responses
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process request with error handling"""
        
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Add request ID to state for logging
        request.state.request_id = request_id
        
        try:
            # Log incoming request
            logger.info(
                f"Request {request_id}: {request.method} {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'}"
            )
            
            # Process request
            response = await call_next(request)
            
            # Log successful response
            duration = time.time() - start_time
            logger.info(
                f"Request {request_id}: {response.status_code} "
                f"completed in {duration:.3f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            duration = time.time() - start_time
            
            logger.warning(
                f"Request {request_id}: HTTP {e.status_code} - {e.detail} "
                f"in {duration:.3f}s"
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": self._get_error_type(e.status_code),
                    "message": str(e.detail),
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path)
                },
                headers={"X-Request-ID": request_id}
            )
            
        except ValidationError as e:
            # Handle Pydantic validation errors
            duration = time.time() - start_time
            
            logger.warning(
                f"Request {request_id}: Validation error - {str(e)} "
                f"in {duration:.3f}s"
            )
            
            return JSONResponse(
                status_code=422,
                content={
                    "error": "validation_error",
                    "message": "Request validation failed",
                    "details": self._format_validation_errors(e),
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path)
                },
                headers={"X-Request-ID": request_id}
            )
            
        except Exception as e:
            # Handle unexpected errors
            duration = time.time() - start_time
            
            # Log full exception details
            logger.error(
                f"Request {request_id}: Unhandled exception in {duration:.3f}s",
                exc_info=True
            )
            
            # Don't expose internal error details in production
            is_development = self._is_development_mode()
            
            error_details = {
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url.path)
            }
            
            # Add debug info in development mode
            if is_development:
                error_details["debug"] = {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc()
                }
            
            return JSONResponse(
                status_code=500,
                content=error_details,
                headers={"X-Request-ID": request_id}
            )
    
    def _get_error_type(self, status_code: int) -> str:
        """Get error type based on status code"""
        
        error_types = {
            400: "bad_request",
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            405: "method_not_allowed",
            409: "conflict",
            422: "validation_error",
            429: "rate_limit_exceeded",
            500: "internal_server_error",
            502: "bad_gateway",
            503: "service_unavailable",
            504: "gateway_timeout"
        }
        
        return error_types.get(status_code, "unknown_error")
    
    def _format_validation_errors(self, validation_error: ValidationError) -> list:
        """Format Pydantic validation errors for API response"""
        
        formatted_errors = []
        
        for error in validation_error.errors():
            formatted_errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        return formatted_errors
    
    def _is_development_mode(self) -> bool:
        """Check if running in development mode"""
        import os
        return os.getenv("ENVIRONMENT", "production").lower() in ["development", "dev", "debug"]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware for API monitoring
    """
    
    async def dispatch(self, request: Request, call_next):
        """Log request details"""
        
        start_time = time.time()
        
        # Log request details
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")
        
        logger.info(
            f"Incoming request: {request.method} {request.url.path} "
            f"from {client_ip} - {user_agent}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response details
        duration = time.time() - start_time
        
        logger.info(
            f"Response: {response.status_code} for {request.method} {request.url.path} "
            f"in {duration:.3f}s"
        )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses
    """
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers"""
        
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # HSTS header (only for HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # CSP header for API
        response.headers["Content-Security-Policy"] = "default-src 'none'; script-src 'none'; object-src 'none';"
        
        return response
