from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

class LLMProvider(Enum):
    """Production LLM providers"""
    GEMINI = "gemini"
    CLAUDE = "claude"
    MOCK = "mock"

@dataclass
class LLMRequest:
    """Standardized request format for all LLM providers"""
    prompt: str
    system_message: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    model_name: Optional[str] = None
    
@dataclass
class LLMResponse:
    """Standardized response format with cost tracking"""
    content: str
    provider: LLMProvider
    model_used: str
    tokens_used: Optional[int] = None
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    cost: float = 0.0
    analysis_metadata: Optional[Dict[str, Any]] = None
    
    def __init__(self, content: str, provider: LLMProvider, model_used: str, 
                 tokens_used: Optional[int] = None, processing_time: float = 0.0,
                 success: bool = True, error_message: Optional[str] = None,
                 cost: float = 0.0, analysis_metadata: Optional[Dict[str, Any]] = None,
                 # Backward compatibility parameters
                 provider_used: Optional[str] = None, cost_estimate: Optional[float] = None,
                 **kwargs):
        """Initialize with backward compatibility for old parameter names"""
        self.content = content
        
        # Handle provider parameter (accept both new and old formats)
        if provider_used and not isinstance(provider, LLMProvider):
            # Convert string to LLMProvider enum
            if provider_used == "mock":
                self.provider = LLMProvider.MOCK
            elif provider_used == "gemini":
                self.provider = LLMProvider.GEMINI
            elif provider_used == "claude":
                self.provider = LLMProvider.CLAUDE
            else:
                self.provider = LLMProvider.MOCK
        else:
            self.provider = provider
            
        self.model_used = model_used
        self.tokens_used = tokens_used
        self.processing_time = processing_time
        self.success = success
        self.error_message = error_message
        
        # Handle cost parameter (accept both new and old names)
        self.cost = cost_estimate if cost_estimate is not None else cost
        
        self.analysis_metadata = analysis_metadata or {}

@dataclass
class AnalysisRequest:
    """Enhanced analysis request with cost and quality preferences"""
    content: str
    analysis_type: str = "comprehensive"
    max_cost: float = 0.05
    quality_preference: str = "balanced"  # "speed", "balanced", "premium"
    require_high_context: bool = False

class ILLMProvider(ABC):
    """Interface for all LLM providers"""
    
    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum token limit for this provider"""
        pass
    
    @abstractmethod
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for given token count"""
        pass

class ILLMService(ABC):
    """Production LLM service interface"""
    
    @abstractmethod
    async def analyze_content(self, request: AnalysisRequest) -> LLMResponse:
        """Analyze content using optimal provider"""
        pass
    
    @abstractmethod
    async def get_available_providers(self) -> List[LLMProvider]:
        """Get list of currently available providers"""
        pass
    
    @abstractmethod
    def get_cost_estimate(self, content: str) -> Dict[str, float]:
        """Get cost estimates for all providers"""
        pass

class ILLMAnalysisService(ABC):
    """High-level analysis service interface"""
    
    @abstractmethod
    async def analyze_scraped_content(self, scraped_content: Any) -> Dict[str, Any]:
        """Analyze scraped content and return structured results"""
        pass
    
    @abstractmethod
    async def generate_content_report(self, scraped_content: Any, 
                                    analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate comprehensive content analysis report"""
        pass