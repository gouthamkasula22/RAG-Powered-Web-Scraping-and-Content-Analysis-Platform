"""
HuggingFace local/remote LLM provider implementation
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time
import asyncio
import os
from typing import Optional, Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

class HuggingFaceProvider:
    """HuggingFace LLM provider for local and remote models"""
    
    # Recommended free models for content analysis
    RECOMMENDED_MODELS = {
        "microsoft/DialoGPT-medium": {"size": "medium", "type": "conversational"},
        "microsoft/DialoGPT-large": {"size": "large", "type": "conversational"},
        "EleutherAI/gpt-neo-1.3B": {"size": "1.3B", "type": "generative"},
        "EleutherAI/gpt-neo-2.7B": {"size": "2.7B", "type": "generative"},
        "google/flan-t5-base": {"size": "220M", "type": "instruction"},
        "google/flan-t5-large": {"size": "770M", "type": "instruction"},
    }
    
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._max_tokens = 512  # Conservative default
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize HuggingFace model"""
        try:
            logger.info(f"Loading HuggingFace model: {self.model_name} on {self.device}")
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # For T5 models, use text2text-generation pipeline
            if "t5" in self.model_name.lower():
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self._max_tokens = 512
            else:
                # For other models, use text-generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self._max_tokens = 1024
            
            logger.info(f"HuggingFace model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {self.model_name}: {e}")
            self.pipeline = None
    
    def is_available(self) -> bool:
        """Check if HuggingFace model is loaded"""
        return self.pipeline is not None
    
    def get_max_tokens(self) -> int:
        """Get max tokens for this model"""
        return self._max_tokens
    
    def estimate_cost(self, tokens: int) -> float:
        """HuggingFace models are free"""
        return 0.0
    
    async def generate_response(self, request) -> 'LLMResponse':
        """Generate response using HuggingFace model"""
        # Import here to avoid circular imports
        from src.application.interfaces.llm import LLMResponse, LLMProvider
        
        if not self.is_available():
            return LLMResponse(
                content="",
                provider=LLMProvider.HUGGINGFACE,
                model_used=self.model_name,
                success=False,
                error_message="HuggingFace model not available"
            )
        
        start_time = time.time()
        
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(request)
            
            # Generate response asynchronously
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                request.max_tokens,
                request.temperature
            )
            
            processing_time = time.time() - start_time
            
            # Extract generated text
            generated_text = self._extract_generated_text(response, prompt)
            
            return LLMResponse(
                content=generated_text,
                provider=LLMProvider.HUGGINGFACE,
                model_used=self.model_name,
                tokens_used=len(self.tokenizer.encode(generated_text)) if self.tokenizer else None,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"HuggingFace generation failed: {str(e)}"
            logger.error(error_msg)
            
            return LLMResponse(
                content="",
                provider=LLMProvider.HUGGINGFACE,
                model_used=self.model_name,
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )
    
    def _prepare_prompt(self, request) -> str:
        """Prepare prompt based on model type"""
        if "t5" in self.model_name.lower():
            # T5 models work better with instruction format
            if request.system_message:
                return f"{request.system_message} {request.prompt}"
            return request.prompt
        else:
            # Other models use conversational format
            if request.system_message:
                return f"System: {request.system_message}\nUser: {request.prompt}\nAssistant:"
            return f"User: {request.prompt}\nAssistant:"
    
    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float):
        """Synchronous generation for executor"""
        generation_kwargs = {
            "max_new_tokens": min(max_tokens, self._max_tokens),
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        if "t5" in self.model_name.lower():
            # T5 models
            generation_kwargs.pop("pad_token_id", None)  # T5 handles padding differently
            return self.pipeline(prompt, **generation_kwargs)
        else:
            # Other generative models
            generation_kwargs.update({
                "return_full_text": False,  # Only return generated part
                "num_return_sequences": 1,
            })
            return self.pipeline(prompt, **generation_kwargs)
    
    def _extract_generated_text(self, response, original_prompt: str) -> str:
        """Extract clean generated text from model response"""
        if isinstance(response, list) and len(response) > 0:
            if "generated_text" in response[0]:
                generated = response[0]["generated_text"]
                # For non-T5 models, remove the original prompt
                if not "t5" in self.model_name.lower():
                    generated = generated.replace(original_prompt, "").strip()
                return generated
            elif "text" in response[0]:
                return response[0]["text"].strip()
        
        return str(response).strip()