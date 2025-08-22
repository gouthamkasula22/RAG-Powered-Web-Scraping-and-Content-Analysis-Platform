"""
Gemini LLM Provider Implementation
Production-ready Google Gemini integration with advanced features.
"""
import asyncio
import time
from typing import Optional, Dict, Any
import aiohttp
import json
import logging
from dataclasses import asdict

from ...application.interfaces.llm import (
    ILLMProvider, LLMRequest, LLMResponse, LLMProvider
)
from ...domain.exceptions import LLMProviderError, ConfigurationError

logger = logging.getLogger(__name__)

class GeminiProvider(ILLMProvider):
    """Google Gemini Pro provider with production features"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """Initialize Gemini provider
        
        Args:
            api_key: Google API key
            model_name: Gemini model to use (gemini-2.0-flash, gemini-pro)
        """
        if not api_key:
            raise ConfigurationError("Gemini API key is required")
            
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # Updated for Gemini-2.0-Flash capabilities
        if "2.0-flash" in model_name:
            self.max_tokens_limit = 1000000  # 1M tokens input context
            self.tokens_per_request = 8192   # Output limit
        else:
            # Fallback for older models
            self.max_tokens_limit = 30720
            self.tokens_per_request = 8192
        
        # Rate limiting and retry configuration
        self.requests_per_minute = 60
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Cost tracking (Gemini is free up to limits)
        self.cost_per_1k_tokens = 0.0  # Free tier
        
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response from Gemini Pro
        
        Args:
            request: Standardized LLM request
            
        Returns:
            LLMResponse with Gemini's response
        """
        start_time = time.time()
        
        try:
            # Validate request
            if not request.prompt.strip():
                raise LLMProviderError("Empty prompt provided")
                
            # Prepare Gemini API request
            api_request = self._prepare_api_request(request)
            
            # Make API call with retries
            response_data = await self._make_api_call(api_request)
            
            # Process response
            content = self._extract_content(response_data)
            processing_time = time.time() - start_time
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GEMINI,
                model_used=self.model_name,
                tokens_used=self._estimate_tokens(request.prompt + content),
                processing_time=processing_time,
                success=True,
                cost=0.0,  # Free tier
                analysis_metadata={
                    "model": self.model_name,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return LLMResponse(
                content="",
                provider=LLMProvider.GEMINI,
                model_used=self.model_name,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                cost=0.0
            )
    
    def _prepare_api_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Prepare Gemini API request format"""
        
        # Combine system message and prompt
        full_prompt = request.prompt
        if request.system_message:
            full_prompt = f"{request.system_message}\n\n{request.prompt}"
        
        return {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": min(request.max_tokens, self.tokens_per_request),
                "topP": 0.8,
                "topK": 10,
                "stopSequences": []
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
    
    async def _make_api_call(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Gemini API with retries"""
        
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        logger.info(f"🌐 Making Gemini API call to: {url}")
        logger.info(f"🔑 API key configured: {bool(self.api_key)}")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 Attempt {attempt + 1}/{self.max_retries}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, 
                        headers=headers, 
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        logger.info(f"📡 Response status: {response.status}")
                        
                        if response.status == 200:
                            response_json = await response.json()
                            logger.info(f"✅ Successful response received")
                            return response_json
                        
                        elif response.status == 429:
                            # Rate limit hit
                            wait_time = self.retry_delay * (2 ** attempt)
                            logger.warning(f"Rate limit hit, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                            
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ API error {response.status}: {error_text}")
                            raise LLMProviderError(f"Gemini API error {response.status}: {error_text}")
                            
            except aiohttp.ClientError as e:
                logger.error(f"🌐 Network error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise LLMProviderError(f"Network error: {str(e)}")
                await asyncio.sleep(self.retry_delay)
                
        raise LLMProviderError("Max retries exceeded")
    
    def _extract_content(self, response_data: Dict[str, Any]) -> str:
        """Extract text content from Gemini response"""
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                raise LLMProviderError("No candidates in response")
                
            content_parts = candidates[0].get("content", {}).get("parts", [])
            if not content_parts:
                raise LLMProviderError("No content parts in response")
                
            return content_parts[0].get("text", "").strip()
            
        except (KeyError, IndexError) as e:
            raise LLMProviderError(f"Invalid response format: {str(e)}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4
    
    def is_available(self) -> bool:
        """Check if Gemini provider is configured and available"""
        return bool(self.api_key)
    
    def get_max_tokens(self) -> int:
        """Get maximum token limit"""
        return self.max_tokens_limit
    
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for token count (free for Gemini)"""
        return 0.0  # Free tier
    
    def can_handle_content_size(self, content_length: int) -> bool:
        """Check if content fits within token limits - Updated for Gemini-2.0-Flash"""
        estimated_tokens = self._estimate_tokens(content_length * "x")  # Rough estimate
        # Use 80% of max tokens to leave room for response and safety buffer
        max_safe_tokens = self.max_tokens_limit * 0.8
        return estimated_tokens <= max_safe_tokens