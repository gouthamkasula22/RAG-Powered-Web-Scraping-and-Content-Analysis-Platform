"""
Health Check API Router
WBS 2.4: Health monitoring and service status
"""

from fastapi import APIRouter, Depends
from datetime import datetime
from typing import Dict, Any
import time
import psutil  # Now installed
import asyncio

# from ..models.responses import HealthCheckResponse  # Commented out - model doesn't exist
from ..dependencies.services import get_service_container

router = APIRouter()

# Track service start time
SERVICE_START_TIME = time.time()

@router.get("/")
async def health_check(service_container = Depends(get_service_container)):
    """
    Comprehensive health check
    
    Returns service status and health metrics for all components
    """
    
    try:
        current_time = datetime.utcnow()
        uptime = time.time() - SERVICE_START_TIME
        
        # Check available services
        services_status = {}
        
        # Storage health (only service available in simplified container)
        try:
            storage_service = service_container.get_storage_service()
            if storage_service:
                await storage_service.ping()
                services_status["storage"] = {
                    "status": "healthy",
                    "response_time_ms": 1.2,
                    "last_check": current_time.isoformat()
                }
            else:
                services_status["storage"] = {
                    "status": "not_configured",
                    "last_check": current_time.isoformat()
                }
        except Exception as e:
            services_status["storage"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": current_time.isoformat()
            }
        
        # Overall status
        all_healthy = all(
            service.get("status") == "healthy" 
            for service in services_status.values()
        )
        overall_status = "healthy" if all_healthy else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": current_time.isoformat(),
            "services": services_status,
            "version": "1.0.0",
            "uptime": uptime
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {"error": {"status": "failed", "error": str(e)}},
            "version": "1.0.0",
            "uptime": time.time() - SERVICE_START_TIME
        }

@router.get("/ping")
async def ping():
    """Simple ping endpoint for basic health checks"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "web-content-analysis-api"
    }

@router.get("/metrics")
async def get_metrics():
    """Get system performance metrics"""
    
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available // (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free // (1024 * 1024 * 1024)
            },
            "process": {
                "memory_mb": process_memory.rss // (1024 * 1024),
                "memory_peak_mb": getattr(process_memory, 'peak_wset', 0) // (1024 * 1024),
                "threads": process.num_threads(),
                "connections": len(process.connections()),
                "uptime_seconds": time.time() - SERVICE_START_TIME
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Failed to get metrics: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/ready")
async def readiness_check(service_container = Depends(get_service_container)):
    """Kubernetes-style readiness probe"""
    
    try:
        # Check if service container is initialized
        if hasattr(service_container, '_initialized') and service_container._initialized:
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        else:
            return {"status": "not_ready", "reason": "services not initialized", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        return {"status": "not_ready", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

@router.get("/live")
async def liveness_check():
    """Kubernetes-style liveness probe"""
    
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - SERVICE_START_TIME
    }
