#!/usr/bin/env python3
"""
Debug script to test LLM service directly
"""
import sys
import os
import asyncio

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

async def test_llm_direct():
    """Test LLM service directly"""
    print("🧪 Testing LLM Service Direct Integration...")
    
    try:
        # Import what we need
        from backend.src.infrastructure.llm.service import ProductionLLMService, LLMServiceConfig
        from backend.src.application.interfaces.llm import AnalysisRequest
        
        # Create config
        config = LLMServiceConfig()
        
        # Initialize service
        print("🚀 Initializing LLM service...")
        service = ProductionLLMService(config)
        
        # Check providers
        print(f"📊 Service has {len(service.providers)} providers")
        for name, provider in service.providers.items():
            print(f"   • {name}: {type(provider).__name__}")
            print(f"     Available: {provider.is_available()}")
            
            # Check if provider has generate_response method
            if hasattr(provider, 'generate_response'):
                print(f"     Has generate_response: ✅")
            else:
                print(f"     Has generate_response: ❌")
        
        # Test a simple request
        print("\n🧠 Testing question answering...")
        
        test_prompt = """Based on the following context, answer the question:

Context: Bala Nemani is the President and Group CEO of the company.

Question: Who is Bala Nemani?

Answer directly and concisely."""
        
        request = AnalysisRequest(
            content=test_prompt,
            analysis_type="question_answering",
            max_cost=0.05,
            quality_preference="balanced"
        )
        
        print("📝 Sending request to LLM service...")
        response = await service.analyze_content(request)
        
        print(f"📬 Response received:")
        print(f"   Success: {response.success}")
        if response.success:
            print(f"   Content: {response.content}")
            print(f"   Provider: {response.provider_used}")
            print(f"   Cost: ${response.cost:.4f}")
        else:
            print(f"   Error: {response.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_direct())
    sys.exit(0 if success else 1)
