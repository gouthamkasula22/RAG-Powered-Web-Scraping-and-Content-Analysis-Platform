"""
Application Interfaces

Contains abstract interfaces following the Dependency Inversion Principle.
These interfaces define contracts between layers.
"""
"""
Update application interfaces __init__.py
"""
from .llm import (
    ILLMProvider,
    ILLMService, 
    ILLMAnalysisService,
    LLMProvider,
    LLMRequest,
    LLMResponse
)

__all__ = [
    # Existing interfaces...
    'IURLValidator',
    'IWebScraper',
    'IScrapingProxy',
    'IContentExtractor',
    'IHTTPClient',
    'ISecurityService',
    'IConfigurationService',
    'IEnvironmentService',
    'ILoggingService',
    
    # New LLM interfaces
    'ILLMProvider',
    'ILLMService',
    'ILLMAnalysisService',
    'LLMProvider',
    'LLMRequest', 
    'LLMResponse'
]