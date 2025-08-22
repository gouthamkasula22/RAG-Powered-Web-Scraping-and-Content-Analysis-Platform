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
    """Production LLM service configuration - Updated for Gemini-2.0-Flash"""
    primary_provider: str = "gemini"  # Free tier with 1M token context
    premium_provider: str = "claude"  # Premium for specialized analysis
    max_cost_per_request: float = 0.05  # $0.05 default limit
    
    # Updated thresholds for Gemini-2.0-Flash (1M tokens vs old 30K)
    large_content_threshold: int = 500000  # Switch to Claude for >500K chars (massive content)
    gemini_max_chars: int = 3000000  # ~750K tokens (safe limit with response buffer)
    claude_preferred_threshold: int = 100000  # Use Claude for >100K chars if high quality needed
    
    max_retry_attempts: int = 3
    timeout_seconds: int = 60

class ProductionLLMService(ILLMService):
    """Production-ready multi-provider LLM service"""
    
    def __init__(self, config: LLMServiceConfig):
        """Initialize with production configuration"""
        self.config = config
        self.providers: Dict[str, ILLMProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize production LLM providers"""
        try:
            # Initialize Gemini (primary - free with 1M token context)
            gemini_key = os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                from .gemini import GeminiProvider
                self.providers["gemini"] = GeminiProvider(
                    api_key=gemini_key,
                    model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                )
                logger.info("✅ Gemini-2.0-Flash provider initialized (free tier, 1M tokens)")
            
            # Initialize Claude (premium - paid)
            claude_key = os.getenv("ANTHROPIC_API_KEY")
            if claude_key:
                from .claude import ClaudeProvider
                self.providers["claude"] = ClaudeProvider(
                    api_key=claude_key,
                    model_name=os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
                )
                logger.info("✅ Claude provider initialized (premium tier)")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM providers: {e}")
            raise ConfigurationError(f"LLM provider initialization failed: {e}")
    
    async def analyze_content(self, request: AnalysisRequest) -> LLMResponse:
        """Analyze content using optimal provider based on size and cost"""
        
        # Choose optimal provider
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
        """Select best provider based on content size, cost, and quality preferences - Updated for Gemini-2.0-Flash"""
        
        content_length = len(request.content)
        
        # Strategy updated for Gemini-2.0-Flash capabilities:
        # - Gemini can now handle up to 1M tokens (3M+ chars)
        # - Claude is now mainly for specialized analysis or cost concerns
        # - Quality preference determines provider choice more than size
        
        logger.info(f"🔍 Selecting provider for content: {content_length} chars")
        
        # Check if content exceeds Gemini's practical limits
        if content_length > self.config.gemini_max_chars:
            logger.info(f"📏 Content very large ({content_length} chars), considering Claude")
            claude = self.providers.get("claude")
            if claude and claude.is_available():
                estimated_cost = claude.estimate_cost(content_length // 4)
                if estimated_cost <= request.max_cost:
                    logger.info(f"🎯 Selected Claude for massive content ({content_length} chars, ~${estimated_cost:.4f})")
                    return claude
        
        # Premium quality preference - use Claude for better analysis quality
        if request.quality_preference == "premium":
            claude = self.providers.get("claude")
            if claude and claude.is_available():
                estimated_cost = claude.estimate_cost(content_length // 4)
                if estimated_cost <= request.max_cost:
                    logger.info(f"🎯 Selected Claude for premium quality (~${estimated_cost:.4f})")
                    return claude
                else:
                    logger.info(f"💰 Claude too expensive for premium request (${estimated_cost:.4f} > ${request.max_cost:.4f}), using Gemini")
        
        # For large content that still fits in Gemini but might benefit from Claude's accuracy
        if (content_length > self.config.claude_preferred_threshold and 
            request.quality_preference == "balanced"):
            
            claude = self.providers.get("claude")
            if claude and claude.is_available():
                estimated_cost = claude.estimate_cost(content_length // 4)
                # Use Claude if cost is reasonable for the improved quality
                if estimated_cost <= min(request.max_cost, 0.02):  # Max $0.02 for balanced mode
                    logger.info(f"🎯 Selected Claude for balanced large content (~${estimated_cost:.4f})")
                    return claude
        
        # Default to Gemini (free tier) - now handles much larger content
        gemini = self.providers.get("gemini")
        if gemini and gemini.is_available():
            if content_length <= self.config.gemini_max_chars:
                logger.info(f"🎯 Selected Gemini-2.0-Flash ({content_length} chars, free)")
                return gemini
            else:
                logger.warning(f"⚠️ Content too large for Gemini ({content_length} chars > {self.config.gemini_max_chars})")
        
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
        logger.info(f"🔄 Provider order: {provider_order}")
        
        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider or not provider.is_available():
                logger.warning(f"⚠️ Provider {provider_name} not available")
                continue
                
            try:
                logger.info(f"🚀 Attempting analysis with {provider_name}")
                logger.info(f"📝 Request prompt length: {len(request.prompt)} chars")
                
                # Add timeout to catch hanging calls
                import asyncio
                response = await asyncio.wait_for(
                    provider.generate_response(request), 
                    timeout=60.0  # 60 second timeout
                )
                
                logger.info(f"📬 Received response from {provider_name}: success={response.success}")
                
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
                    
            except asyncio.TimeoutError:
                logger.error(f"⏰ {provider_name} timeout after 60 seconds")
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