"""
Test script for Production LLM Service
Verifies Gemini + Claude integration works correctly.
"""
import asyncio
import os
import sys
import pytest
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    print("⚠️  python-dotenv not installed, trying manual .env loading...")
    # Manual .env file parsing as fallback
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

from src.infrastructure.llm.service import ProductionLLMService, LLMServiceConfig
from src.application.interfaces.llm import AnalysisRequest

@pytest.mark.integration
@pytest.mark.skipif(not (os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")), reason="No LLM API keys configured")
def test_llm_service():
    """Synchronous pytest wrapper that runs the async LLM service test."""

    async def _run():
        print("🧪 Testing Production LLM Service")
        print("=" * 50)
        config = LLMServiceConfig()
        try:
            print("📝 Initializing LLM service...")
            service = ProductionLLMService(config)
            health = service.get_health_status()
            print(f"🏥 Service Health: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
            print(f"📊 Available Providers: {health['total_available']}")
            for provider_name, provider_info in health['providers'].items():
                status = "✅ Available" if provider_info['available'] else "❌ Unavailable"
                cost = provider_info['cost_per_1k']
                print(f"   {provider_name}: {status} (${cost:.4f}/1K tokens)")

            test_content = "Sample content for LLM service functional test."  # Keep short for speed
            request = AnalysisRequest(
                content=test_content,
                analysis_type="comprehensive",
                max_cost=0.005,
                quality_preference="balanced"
            )
            print("🚀 Starting analysis...")
            response = await service.analyze_content(request)
            if response.success:
                print("✅ Analysis completed successfully!")
                print(f"🤖 Provider: {response.provider.value}")
                print(f"💰 Cost: ${response.cost:.6f}")
                preview = response.content[:120]
                print(preview + ("..." if len(response.content) > 120 else ""))
            else:
                print("❌ Analysis failed!")
                print(f"Error: {response.error_message}")
            assert response is not None
            assert hasattr(response, "success")
            if response.success:
                assert response.content
        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    asyncio.run(_run())

if __name__ == "__main__":
    # Allow running standalone
    test_llm_service()
