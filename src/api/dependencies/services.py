"""
Service Container - Dependency Injection
WBS 2.4: Service management and dependency injection
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

"""
Service Container - Dependency Injection
WBS 2.4: Service management and dependency injection

Note: This is a simplified version using only existing backend services.
For the main application, services are configured in backend/api/main.py
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ServiceContainer:
    """
    Simplified service container for dependency injection
    Currently a placeholder - main services are in backend/api/main.py
    """
    
    def __init__(self):
        self._services = {}
        self._initialized = False
        self._initialization_time = None
    
    async def initialize(self):
        """Initialize all services"""
        if self._initialized:
            return
        
        try:
            logger.info("🔧 Initializing service container...")
            start_time = datetime.utcnow()
            
            # Initialize mock services for now
            await self._initialize_mock_services()
            
            self._initialized = True
            self._initialization_time = datetime.utcnow()
            
            elapsed = (self._initialization_time - start_time).total_seconds()
            logger.info(f"✅ Service container initialized in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize service container: {e}")
            raise
    
    async def _initialize_mock_services(self):
        """Initialize mock services"""
        
        # Mock Storage Service
        self._services['storage'] = MockStorageService()
        
        logger.info("✅ Mock services initialized")
    
    def get_storage_service(self):
        """Get storage service"""
        return self._services.get('storage')
    
    async def health_check(self) -> dict:
        """Check health of all services"""
        
        if not self._initialized:
            return {"status": "not_initialized"}
        
        health_status = {}
        
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'health_check'):
                    await service.health_check()
                    health_status[service_name] = {"status": "healthy"}
                else:
                    # Basic existence check
                    health_status[service_name] = {"status": "healthy"}
            except Exception as e:
                health_status[service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return health_status
    
    async def shutdown(self):
        """Gracefully shutdown all services"""
        
        logger.info("🔄 Shutting down service container...")
        
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'shutdown'):
                    await service.shutdown()
                logger.debug(f"✅ {service_name} service shut down")
            except Exception as e:
                logger.error(f"❌ Error shutting down {service_name}: {e}")
        
        self._services.clear()
        self._initialized = False
        
        logger.info("✅ Service container shut down complete")


class MockStorageService:
    """Mock storage service for development"""
    
    def __init__(self):
        self._data = {}
    
    async def ping(self):
        """Health check ping"""
        return True
    
    async def get(self, key: str):
        """Get data by key"""
        return self._data.get(key)
    
    async def set(self, key: str, value):
        """Set data by key"""
        self._data[key] = value
        return True
    
    async def delete(self, key: str):
        """Delete data by key"""
        return self._data.pop(key, None) is not None


# Global service container instance
_service_container: Optional[ServiceContainer] = None

async def get_service_container() -> ServiceContainer:
    """Get or create service container instance"""
    global _service_container
    
    if _service_container is None:
        _service_container = ServiceContainer()
        await _service_container.initialize()
    
    return _service_container

async def shutdown_service_container():
    """Shutdown global service container"""
    global _service_container
    
    if _service_container is not None:
        await _service_container.shutdown()
        _service_container = None


class MockStorageService:
    """Mock storage service for development"""
    
    def __init__(self):
        self._data = {}
    
    async def ping(self):
        """Health check ping"""
        return True
    
    async def get(self, key: str):
        """Get data by key"""
        return self._data.get(key)
    
    async def set(self, key: str, value):
        """Set data by key"""
        self._data[key] = value
        return True
    
    async def delete(self, key: str):
        """Delete data by key"""
        return self._data.pop(key, None) is not None


# Global service container instance
_service_container: Optional[ServiceContainer] = None

async def get_service_container() -> ServiceContainer:
    """Get or create service container instance"""
    global _service_container
    
    if _service_container is None:
        _service_container = ServiceContainer()
        await _service_container.initialize()
    
    return _service_container

async def shutdown_service_container():
    """Shutdown global service container"""
    global _service_container
    
    if _service_container is not None:
        await _service_container.shutdown()
        _service_container = None
