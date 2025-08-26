"""
Production Multi-Provider LLM Service
Intelligent orchestration of Gemini + Claude providers with cost optimization.
"""
from typing import List, Dict, Optional, Any
import asyncio
import time
import os
import logging
from dataclasses import dataclass

from ...application.interfaces.llm import (
    ILLMService, ILLMProvider, LLMRequest, LLMResponse, 
    LLMProvider, AnalysisRequest
)
from ...domain.exceptions import LLMProviderError, ConfigurationError

logger = logging.getLogger(__name__)

@dataclass
class LLMServiceConfig:
    """Production LLM service configuration"""
    primary_provider: str = "gemini"  # Free tier for most requests
    premium_provider: str = "claude"  # Premium for large content
    max_cost_per_request: float = 0.05  # $0.05 default limit
    large_content_threshold: int = 20000  # Switch to Claude for >20k chars
    max_retry_attempts: int = 3
    timeout_seconds: int = 60

class ProductionLLMService(ILLMService):
    """Production-ready multi-provider LLM service"""
    
    def __init__(self, config: LLMServiceConfig):
        """Initialize with production configuration"""
        self.config = config
        self.providers: Dict[str, ILLMProvider] = {}
    def _initialize_providers(self):
        """Initialize production LLM providers"""
        try:
            # Initialize Gemini (primary - free)
            gemini_key = os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                from .gemini import GeminiProvider
                self.providers["gemini"] = GeminiProvider(
                    api_key=gemini_key,
                    model_name=os.getenv("GEMINI_MODEL", "gemini-pro")
                )
                logger.info("✅ Gemini provider initialized (free tier)")
            
            # Initialize Claude (premium - paid)
            claude_key = os.getenv("ANTHROPIC_API_KEY")
            if claude_key:
                self.providers["claude"] = ClaudeProvider(
                    api_key=claude_key,
                    model_name=os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
                )
                logger.info("✅ Claude provider initialized (premium tier)")
                
        except Exception as e:
            raise ConfigurationError(f"LLM provider initialization failed: {e}")
    
    async def analyze_content(self, request: AnalysisRequest) -> LLMResponse:
        """Analyze content using optimal provider based on size and cost"""
        provider = self._select_optimal_provider(request)
        if not provider:
            raise LLMProviderError("No suitable LLM provider available")
        # Create LLM request
        llm_request = LLMRequest(
            prompt=self._create_analysis_prompt(request),
            system_message=self._get_system_message(request.analysis_type),
            max_tokens=4096,
            temperature=0.3  # Lower for more consistent analysis
        )
        # Execute with fallback
        return await self._execute_with_fallback(llm_request, request.max_cost)
    
    def _select_optimal_provider(self, request: AnalysisRequest) -> Optional[ILLMProvider]:
        """Select best provider based on content size, cost, and quality preferences"""
        
        content_length = len(request.content)
        
        # For large content or premium quality preference, use Claude if available
        if (content_length > self.config.large_content_threshold or 
            request.quality_preference == "premium" or
            request.require_high_context):
            
            claude = self.providers.get("claude")
            if claude and claude.is_available():
                # Check cost constraints
                estimated_cost = claude.estimate_cost(content_length // 4)  # Rough token estimate
                if estimated_cost <= request.max_cost:
                    logger.info(f"🎯 Selected Claude for large/premium content ({content_length} chars, ~${estimated_cost:.4f})")
                    return claude
        
        # Default to Gemini (free tier)
        gemini = self.providers.get("gemini")
        if gemini and gemini.is_available():
            if gemini.can_handle_content_size(content_length):
                logger.info(f"🎯 Selected Gemini for standard content ({content_length} chars, free)")
                return gemini
        
        # Fallback to any available provider
        for provider in self.providers.values():
            if provider.is_available():
                logger.warning(f"⚠️ Fallback to {provider.__class__.__name__}")
                return provider
                
        return None
    
    async def _execute_with_fallback(self, request: LLMRequest, max_cost: float) -> LLMResponse:
        """Execute request with intelligent fallback"""
        
        # Try providers in order of preference
        provider_order = self._get_provider_order()
        
        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider or not provider.is_available():
                continue
                
            try:
                logger.info(f"🚀 Attempting analysis with {provider_name}")
                response = await provider.generate_response(request)
                
                if response.success:
                    if response.cost <= max_cost:
                        logger.info(f"✅ Analysis completed with {provider_name} (${response.cost:.4f})")
                        return response
                    else:
                        logger.warning(f"💰 {provider_name} cost ${response.cost:.4f} exceeds limit ${max_cost:.4f}")
                        continue
                else:
                    logger.warning(f"⚠️ {provider_name} failed: {response.error_message}")
                    continue
                    
            except Exception as e:
                logger.error(f"❌ {provider_name} error: {str(e)}")
                continue
        
        # All providers failed
        raise LLMProviderError("All LLM providers failed or exceeded cost limits")
    
    def _get_provider_order(self) -> List[str]:
        """Get provider order based on configuration"""
        available_providers = [name for name, provider in self.providers.items() 
                             if provider.is_available()]
        
        # Prefer free providers first, then premium
        order = []
        if "gemini" in available_providers:
            order.append("gemini")
        if "claude" in available_providers:
            order.append("claude")
            
        return order
    
    def _create_analysis_prompt(self, request: AnalysisRequest) -> str:
        """Create detailed analysis prompt based on request type"""
        
        prompts = {
            "comprehensive": f"""
Analyze the following web content comprehensively. Provide:

1. **Content Summary**: Main topics and key points
2. **Content Type**: Article, product page, blog, news, etc.
3. **Quality Assessment**: Writing quality, credibility, bias detection
4. **Key Insights**: Important information and takeaways
5. **SEO Analysis**: Keywords, structure, optimization level
6. **Audience**: Target audience and reading level
7. **Recommendations**: Improvements or actions suggested

Content to analyze:
{request.content}

Please provide a structured, detailed analysis in markdown format.
""",
            
            "summary": f"""
Provide a concise summary of this web content:

{request.content}

Focus on the main points, key information, and overall purpose.
""",
            
            "seo": f"""
Analyze this web content for SEO optimization:

{request.content}

Provide insights on keywords, structure, meta information, and optimization recommendations.
""",
            
            "sentiment": f"""
Analyze the sentiment and tone of this web content:

{request.content}

Determine overall sentiment, emotional tone, and communication style.
"""
        }
        
        return prompts.get(request.analysis_type, prompts["comprehensive"])
    
    def _get_system_message(self, analysis_type: str) -> str:
        """Get system message for analysis type"""
        return """You are an expert web content analyst. Provide accurate, insightful, and actionable analysis. 
Be objective, thorough, and structure your responses clearly. Focus on practical insights that help users 
understand and improve their content strategy."""
    
    async def get_available_providers(self) -> List[LLMProvider]:
        """Get list of currently available providers"""
        available = []
        for name, provider in self.providers.items():
            if provider.is_available():
                available.append(LLMProvider(name))
        return available
    
    def get_cost_estimate(self, content: str) -> Dict[str, float]:
        """Get cost estimates for all providers"""
        estimates = {}
        content_tokens = len(content) // 4  # Rough estimate
        
        for name, provider in self.providers.items():
            if provider.is_available():
                estimates[name] = provider.estimate_cost(content_tokens)
                
        return estimates
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        status = {
            "timestamp": time.time(),
            "providers": {},
            "total_available": 0
        }
        
        for name, provider in self.providers.items():
            is_available = provider.is_available()
            status["providers"][name] = {
                "available": is_available,
                "max_tokens": provider.get_max_tokens(),
                "cost_per_1k": provider.estimate_cost(1000)
            }
            if is_available:
                status["total_available"] += 1
                
        status["healthy"] = status["total_available"] > 0
        return status