"""
Rate Limiting Middleware
WBS 2.4: API rate limiting and request throttling
"""

import time
import asyncio
from collections import defaultdict, deque
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window approach
    Limits requests per IP address
    """
    
    def __init__(self, app, calls: int = 10, period: int = 60):
        """
        Initialize rate limiter
        
        Args:
            calls: Number of calls allowed
            period: Time period in seconds
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients: Dict[str, deque] = defaultdict(deque)
        self.cleanup_interval = 300  # Clean up every 5 minutes
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Skip rate limiting for health checks
        if request.url.path in ["/api/health/ping", "/api/health/live", "/api/health/ready"]:
            return await call_next(request)
        
        current_time = time.time()
        
        # Clean up old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Check rate limit
        if await self._is_rate_limited(client_ip, current_time):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            
            # Calculate retry after
            client_requests = self.clients[client_ip]
            if client_requests:
                oldest_request = client_requests[0]
                retry_after = int(self.period - (current_time - oldest_request)) + 1
            else:
                retry_after = self.period
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Limit: {self.calls} per {self.period} seconds",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Record this request
        self.clients[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.calls - len(self.clients[client_ip]))
        reset_time = int(current_time + self.period)
        
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        response.headers["X-RateLimit-Window"] = str(self.period)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        
        # Check for forwarded headers (for reverse proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    async def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client has exceeded rate limit"""
        
        client_requests = self.clients[client_ip]
        
        # Remove requests outside the time window
        while client_requests and current_time - client_requests[0] > self.period:
            client_requests.popleft()
        
        # Check if limit exceeded
        return len(client_requests) >= self.calls
    
    async def _cleanup_old_entries(self, current_time: float):
        """Clean up old request records"""
        
        clients_to_remove = []
        
        for client_ip, requests in self.clients.items():
            # Remove old requests
            while requests and current_time - requests[0] > self.period * 2:  # Keep extra buffer
                requests.popleft()
            
            # Remove empty clients
            if not requests:
                clients_to_remove.append(client_ip)
        
        for client_ip in clients_to_remove:
            del self.clients[client_ip]
        
        if clients_to_remove:
            logger.debug(f"Cleaned up {len(clients_to_remove)} inactive clients")


class AdaptiveRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting with adaptive limits based on endpoint
    """
    
    def __init__(self, app):
        super().__init__(app)
        
        # Different limits for different endpoints
        self.endpoint_limits = {
            "/api/analysis/analyze": {"calls": 5, "period": 60},      # Strict for analysis
            "/api/analysis/bulk": {"calls": 1, "period": 300},        # Very strict for bulk
            "/api/reports/generate": {"calls": 10, "period": 60},     # Moderate for reports
            "default": {"calls": 20, "period": 60}                   # Default for other endpoints
        }
        
        self.clients: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
    
    async def dispatch(self, request: Request, call_next):
        """Process request with adaptive rate limiting"""
        
        client_ip = self._get_client_ip(request)
        endpoint = self._get_endpoint_key(request.url.path)
        
        # Get limits for this endpoint
        limits = self.endpoint_limits.get(endpoint, self.endpoint_limits["default"])
        calls = limits["calls"]
        period = limits["period"]
        
        current_time = time.time()
        
        # Check rate limit for this endpoint
        if await self._is_rate_limited(client_ip, endpoint, current_time, calls, period):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests to {endpoint}. Limit: {calls} per {period} seconds",
                    "endpoint": endpoint
                }
            )
        
        # Record request
        self.clients[client_ip][endpoint].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add headers
        remaining = max(0, calls - len(self.clients[client_ip][endpoint]))
        response.headers["X-RateLimit-Limit"] = str(calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Endpoint"] = endpoint
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _get_endpoint_key(self, path: str) -> str:
        """Get endpoint key for rate limiting"""
        
        # Match specific endpoints
        for endpoint in self.endpoint_limits:
            if endpoint != "default" and path.startswith(endpoint):
                return endpoint
        
        return "default"
    
    async def _is_rate_limited(self, client_ip: str, endpoint: str, current_time: float, calls: int, period: int) -> bool:
        """Check rate limit for specific endpoint"""
        
        client_requests = self.clients[client_ip][endpoint]
        
        # Clean old requests
        while client_requests and current_time - client_requests[0] > period:
            client_requests.popleft()
        
        return len(client_requests) >= calls
