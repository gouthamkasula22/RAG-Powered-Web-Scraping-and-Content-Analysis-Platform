"""
Claude LLM Provider Implementation
Production-ready Anthropic Claude integration for premium content analysis.
"""
import asyncio
import time
from typing import Optional, Dict, Any
import logging
from dataclasses import asdict

try:
    import anthropic
except ImportError:
    anthropic = None

from ...application.interfaces.llm import (
    ILLMProvider, LLMRequest, LLMResponse, LLMProvider
)
from ...domain.exceptions import LLMProviderError, ConfigurationError

logger = logging.getLogger(__name__)

class ClaudeProvider(ILLMProvider):
    """Anthropic Claude Haiku provider for premium analysis"""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-haiku-20240307"):
        """Initialize Claude provider
        
        Args:
            api_key: Anthropic API key
            model_name: Claude model to use (claude-3-haiku, claude-3-sonnet)
        """
        if not api_key:
            raise ConfigurationError("Anthropic API key is required")
            
        if anthropic is None:
            raise ConfigurationError("anthropic package not installed")
            
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens_limit = 200000  # Claude 3 Haiku context window
        self.tokens_per_request = 4096  # Safe output limit
        
        # Rate limiting configuration
        self.requests_per_minute = 60
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Cost tracking for Claude Haiku
        self.cost_per_1k_input_tokens = 0.00025  # $0.00025 per 1K input tokens
        self.cost_per_1k_output_tokens = 0.00125  # $0.00125 per 1K output tokens
        
        # Initialize client
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response from Claude
        
        Args:
            request: Standardized LLM request
            
        Returns:
            LLMResponse with Claude's response
        """
        start_time = time.time()
        
        try:
            # Validate request
            if not request.prompt.strip():
                raise LLMProviderError("Empty prompt provided")
                
            # Prepare Claude messages format
            messages = self._prepare_messages(request)
            
            # Make API call with retries
            response = await self._make_api_call(messages, request)
            
            # Process response
            content = response.content[0].text.strip()
            processing_time = time.time() - start_time
            
            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.CLAUDE,
                model_used=self.model_name,
                tokens_used=input_tokens + output_tokens,
                processing_time=processing_time,
                success=True,
                cost=cost,
                analysis_metadata={
                    "model": self.model_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return LLMResponse(
                content="",
                provider=LLMProvider.CLAUDE,
                model_used=self.model_name,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                cost=0.0
            )
    
    def _prepare_messages(self, request: LLMRequest) -> list:
        """Prepare Claude messages format"""
        messages = []
        
        # Claude uses system message separately and messages format
        if request.system_message:
            # System message is handled separately in Claude
            pass
            
        messages.append({
            "role": "user",
            "content": request.prompt
        })
        
        return messages
    
    async def _make_api_call(self, messages: list, request: LLMRequest) -> Any:
        """Make API call to Claude with retries"""
        
        for attempt in range(self.max_retries):
            try:
                # Prepare API parameters
                api_params = {
                    "model": self.model_name,
                    "max_tokens": min(request.max_tokens, self.tokens_per_request),
                    "temperature": request.temperature,
                    "messages": messages
                }
                
                # Add system message if provided
                if request.system_message:
                    api_params["system"] = request.system_message
                
                # Make the API call
                response = await self.client.messages.create(**api_params)
                return response
                
            except anthropic.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Claude rate limit hit, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise LLMProviderError(f"Claude rate limit exceeded: {str(e)}")
                    
            except anthropic.APIError as e:
                raise LLMProviderError(f"Claude API error: {str(e)}")
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LLMProviderError(f"Claude request failed: {str(e)}")
                await asyncio.sleep(self.retry_delay)
                
        raise LLMProviderError("Max retries exceeded")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output_tokens
        return input_cost + output_cost
    
    def is_available(self) -> bool:
        """Check if Claude provider is configured and available"""
        return bool(self.api_key and anthropic)
    
    def get_max_tokens(self) -> int:
        """Get maximum token limit"""
        return self.max_tokens_limit
    
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for token count (assuming 80% input, 20% output)"""
        input_tokens = int(tokens * 0.8)
        output_tokens = int(tokens * 0.2)
        return self._calculate_cost(input_tokens, output_tokens)
    
    def can_handle_content_size(self, content_length: int) -> bool:
        """Check if content fits within token limits"""
        # Rough estimate: ~4 characters per token
        estimated_tokens = content_length // 4
        return estimated_tokens <= self.max_tokens_limit * 0.8  # Leave room for response
    
    def is_cost_effective_for_content(self, content_length: int, max_cost: float) -> bool:
        """Check if analysis would be cost-effective"""
        estimated_tokens = content_length // 4
        estimated_cost = self.estimate_cost(estimated_tokens)
        return estimated_cost <= max_cost
