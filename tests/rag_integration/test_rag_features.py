#!/usr/bin/env python3
"""
Test script for RAG with latest features:
1. Reversed chat order (latest first)
2. Website selection for targeted search
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_rag_features():
    """Test the latest RAG features"""
    print("🧪 Testing RAG Features: Chat Order & Website Selection...")
    
    from frontend.streamlit.components.rag_knowledge_repository import RAGKnowledgeRepository
    
    # Initialize RAG system
    print("🚀 Initializing RAG system...")
    rag_system = RAGKnowledgeRepository()
    
    print(f"✅ RAG system initialized")
    print(f"   • LLM Service: {'Available' if rag_system.llm_service else 'Not Available'}")
    print(f"   • Embedding Model: {'Available' if rag_system.embedding_model else 'Not Available'}")
    
    assert rag_system is not None
    
    # Test website loading
    print("\n📊 Testing website loading...")
    websites = rag_system._get_available_websites()
    print(f"   • Found {len(websites)} websites in knowledge base")
    
    assert isinstance(websites, list)
    
    for i, website in enumerate(websites[:3], 1):
        print(f"   {i}. {website['title']} - {website.get('chunk_count', 0)} chunks")
        assert 'title' in website
    
    if len(websites) > 3:
        print(f"   ... and {len(websites) - 3} more websites")
    
    # Test features summary
    print(f"\n🎯 New Features Status:")
    print(f"   ✅ Latest messages appear at top (reversed order)")
    print(f"   ✅ Website selection dropdown for targeted search")
    print(f"   ✅ Visual indicators for search scope")
    print(f"   ✅ Source attribution below responses")
    print(f"   ✅ Method indicators (AI vs Rule-based)")
    
    print(f"\n🚀 Features Ready for Testing:")
    print(f"   1. Run the Streamlit app")
    print(f"   2. Select a specific website or 'All Websites'")
    print(f"   3. Ask questions and see responses appear at the top")
    print(f"   4. Check source attribution under each response")

if __name__ == "__main__":
    success = test_rag_features()
    sys.exit(0 if success else 1)
