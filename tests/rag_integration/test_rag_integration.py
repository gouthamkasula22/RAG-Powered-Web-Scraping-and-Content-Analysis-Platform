#!/usr/bin/env python3
"""
Test script for RAG Knowledge Repository integration
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_rag_system():
    """Test the RAG system integration"""
    print("🧪 Testing RAG Knowledge Repository Integration...")
    
    try:
        # Import the RAG system
        from frontend.streamlit.components.rag_knowledge_repository import RAGKnowledgeRepository
        print("✅ RAG system imported successfully")
        
        # Initialize the system
        rag_system = RAGKnowledgeRepository()
        print("✅ RAG system initialized successfully")
        
        # Check LLM service status
        if rag_system.llm_service:
            if rag_system.llm_service.providers:
                providers = list(rag_system.llm_service.providers.keys())
                print(f"✅ LLM service active with providers: {', '.join(providers)}")
            else:
                print("⚠️ LLM service initialized but no providers available (missing API keys)")
        else:
            print("ℹ️ LLM service not available, using rule-based responses")
        
        # Test basic functionality
        print("\n🔍 Testing basic retrieval...")
        
        # Check if database exists and has content
        database_file = os.path.join(project_root, "data", "knowledge_repository.db")
        if os.path.exists(database_file):
            print(f"✅ Database found: {database_file}")
            
            # Test search functionality
            results = rag_system._retrieve_relevant_chunks("leadership team", top_k=3)
            print(f"✅ Search test completed - found {len(results)} results")
            
        else:
            print(f"⚠️ Database not found: {database_file}")
            print("💡 You may need to analyze some websites first to populate the knowledge base")
        
        print("\n🎉 RAG system integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)
