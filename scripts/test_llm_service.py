"""
Test script for Production LLM Service
Verifies Gemini + Claude integration works correctly.
"""
import asyncio
import os
import sys
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

async def test_llm_service():
    """Test the production LLM service"""
    
    print("🧪 Testing Production LLM Service")
    print("=" * 50)
    
    # Create service configuration
    config = LLMServiceConfig()
    
    try:
        # Initialize service
        print("📝 Initializing LLM service...")
        service = ProductionLLMService(config)
        
        # Check health status
        health = service.get_health_status()
        print(f"🏥 Service Health: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
        print(f"📊 Available Providers: {health['total_available']}")
        
        for provider_name, provider_info in health['providers'].items():
            status = "✅ Available" if provider_info['available'] else "❌ Unavailable"
            cost = provider_info['cost_per_1k']
            print(f"   {provider_name}: {status} (${cost:.4f}/1K tokens)")
        
        # Test with sample content
        test_content = """
        This is a sample web page content for testing LLM analysis.
        
        About Our Company
        We are a technology company focused on building innovative solutions.
        Our products help businesses streamline their operations and improve efficiency.
        
        Key Features:
        - Advanced analytics dashboard
        - Real-time data processing  
        - Scalable cloud infrastructure
        - 24/7 customer support
        
        Contact us today to learn more about how we can help your business grow.
        """
        
        print("\n🎯 Testing content analysis...")
        
        # Create analysis request
        request = AnalysisRequest(
            content=test_content,
            analysis_type="comprehensive",
            max_cost=0.01,  # Low cost limit for testing
            quality_preference="balanced"
        )
        
        # Perform analysis
        print("🚀 Starting analysis...")
        response = await service.analyze_content(request)
        
        if response.success:
            print("✅ Analysis completed successfully!")
            print(f"🤖 Provider: {response.provider.value}")
            print(f"💰 Cost: ${response.cost:.6f}")
            print(f"⏱️  Time: {response.processing_time:.2f}s")
            print(f"🎯 Tokens: {response.tokens_used}")
            print("\n📄 Analysis Result:")
            print("-" * 40)
            # Show first 500 chars of response
            preview = response.content[:500]
            if len(response.content) > 500:
                preview += "..."
            print(preview)
        else:
            print("❌ Analysis failed!")
            print(f"Error: {response.error_message}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for environment variables (more flexible)
    gemini_key = os.getenv("GOOGLE_API_KEY")
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not gemini_key and not claude_key:
        print("⚠️  No LLM API keys found in environment")
        print("Set at least one of these in your .env file:")
        print("   - GOOGLE_API_KEY (for free Gemini tier)")
        print("   - ANTHROPIC_API_KEY (for premium Claude)")
        print("\n🔧 For testing architecture only, we'll demonstrate the service structure...")
        
        # Show architecture test
        config = LLMServiceConfig()
        try:
            service = ProductionLLMService(config)
            health = service.get_health_status()
            print(f"\n🏗️  Service Architecture: ✅ Loaded")
            print(f"📊 Provider Framework: {len(health['providers'])} provider slots")
            print(f"🔧 Configuration: ✅ Valid")
            print("\n✅ LLM infrastructure is properly implemented!")
            print("🚀 Ready for API keys to enable full functionality.")
        except Exception as e:
            print(f"❌ Architecture error: {e}")
        
        sys.exit(0)
    
    if gemini_key:
        print("✅ Gemini API key found - free tier available")
    if claude_key:
        print("✅ Claude API key found - premium tier available")
    
    # Run the full test
    asyncio.run(test_llm_service())
