#!/usr/bin/env python3
"""
Test script for RAG with direct LLM provider access
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

async def test_rag_with_llm():
    """Test RAG system with LLM integration"""
    print("🧪 Testing RAG with Direct LLM Provider Access...")
    
    try:
        from frontend.streamlit.components.rag_knowledge_repository import RAGKnowledgeRepository
        
        # Initialize RAG system
        print("🚀 Initializing RAG system...")
        rag_system = RAGKnowledgeRepository()
        
        if not rag_system.llm_service or not rag_system.llm_service.providers:
            print("❌ No LLM providers available")
            return False
            
        print(f"✅ LLM providers available: {list(rag_system.llm_service.providers.keys())}")
        
        # Test direct LLM call
        print("\n🧠 Testing direct LLM response...")
        
        question = "Who is Bala Nemani?"
        context = "Bala Nemani is President - Group CEO of the company. He leads the organization and oversees all operations."
        
        try:
            response = await rag_system._generate_llm_response(question, context, [])
            print(f"✅ LLM Response: {response}")
            
            # Test the response generation method
            print("\n🔄 Testing full RAG response generation...")
            result = rag_system._generate_contextual_response(question, context, [])
            print(f"📝 Method: {result['method']}")
            print(f"💬 Response: {result['response']}")
            
        except Exception as e:
            print(f"❌ LLM response failed: {e}")
            return False
        
        print("\n🎉 RAG LLM integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rag_with_llm())
    sys.exit(0 if success else 1)
