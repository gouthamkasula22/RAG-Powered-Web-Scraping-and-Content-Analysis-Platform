#!/usr/bin/env python3
"""
Test script to isolate LLMResponse constructor issue
"""
import sys
import os
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not available")

# Test LLMResponse constructor
try:
    from src.application.interfaces.llm import LLMResponse, LLMProvider
    
    print("✅ Successfully imported LLMResponse and LLMProvider")
    
    # Test correct constructor
    response = LLMResponse(
        content="Test content",
        provider=LLMProvider.MOCK,
        model_used="test-model",
        tokens_used=100,
        processing_time=1.0,
        success=True,
        error_message=None,
        cost=0.0,
        analysis_metadata={}
    )
    
    print("✅ LLMResponse constructor works correctly")
    print(f"Response: {response}")
    
except Exception as e:
    print(f"❌ Error with LLMResponse: {e}")
    import traceback
    traceback.print_exc()

# Test the actual services
try:
    from src.infrastructure.llm.service import LLMServiceConfig, ProductionLLMService
    
    config = LLMServiceConfig()
    
    # Check API keys
    google_key = os.getenv("GOOGLE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    print(f"🔑 Google API Key: {'✅' if google_key else '❌'}")
    print(f"🔑 Anthropic API Key: {'✅' if anthropic_key else '❌'}")
    
    if google_key or anthropic_key:
        print("🚀 Testing ProductionLLMService initialization...")
        llm_service = ProductionLLMService(config)
        print("✅ ProductionLLMService initialized successfully")
    else:
        print("⚠️ No API keys found, skipping ProductionLLMService test")
        
except Exception as e:
    print(f"❌ Error with ProductionLLMService: {e}")
    import traceback
    traceback.print_exc()
