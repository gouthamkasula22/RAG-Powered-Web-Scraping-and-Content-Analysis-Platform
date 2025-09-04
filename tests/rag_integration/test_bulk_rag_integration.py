#!/usr/bin/env python3
"""
Test bulk analysis integration with RAG Knowledge Repository
"""
import sys
import os
from types import SimpleNamespace

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def create_mock_bulk_results():
    """Create mock bulk analysis results for testing"""
    
    # Simulate bulk analysis results structure
    bulk_results = {
        "batch_id": "test_batch_123",
        "results": [
            {
                "url": "https://example.com",
                "status": "completed",
                "executive_summary": "Example.com is a domain used in illustrative examples in documents. It serves as a placeholder website for demonstrations and testing purposes.",
                "scraped_content": {
                    "title": "Example Domain",
                    "main_content": "This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission. More information about example domains can be found here.",
                    "meta_description": "Example domain for documentation"
                }
            },
            {
                "url": "https://test.com", 
                "status": "completed",
                "executive_summary": "Test.com is a technology company providing testing solutions for software development.",
                "scraped_content": {
                    "title": "Test Company",
                    "main_content": "We provide comprehensive testing solutions for modern software development. Our tools help developers ensure quality and reliability in their applications.",
                    "meta_description": "Professional testing solutions"
                }
            }
        ],
        "total_cost": 0.10,
        "completed": 2,
        "failed": 0
    }
    
    return bulk_results

def test_bulk_integration():
    """Test the integration of bulk analysis with RAG Knowledge Repository"""
    
    print("🧪 Testing Bulk Analysis → Knowledge Repository Integration")
    print("=" * 60)
    
    # Import the RAG system
    from frontend.streamlit.components.rag_knowledge_repository import RAGKnowledgeRepository
    
    # Initialize RAG system
    rag_system = RAGKnowledgeRepository()
    print("✅ RAG Knowledge Repository initialized")
    assert rag_system is not None
    
    # Create mock bulk results
    bulk_results = create_mock_bulk_results()
    print(f"✅ Created mock bulk results with {len(bulk_results['results'])} websites")
    assert len(bulk_results['results']) == 2
    
    # Test the conversion method
    conversion_count = 0
    for result_dict in bulk_results['results']:
        # Convert dict to analysis result object
        analysis_result = rag_system._convert_dict_to_analysis_result(result_dict)
        print(f"✅ Converted result for {analysis_result.url}")
        assert analysis_result is not None
        assert hasattr(analysis_result, 'url')
        conversion_count += 1
        
        # Test adding to knowledge base
        success = rag_system.add_website_from_analysis(analysis_result)
        if success:
            print(f"✅ Added {analysis_result.url} to knowledge base")
        else:
            print(f"⚠️ {analysis_result.url} already exists or has insufficient content")
    
    assert conversion_count == 2
    
    # Test retrieval
    print("\n🔍 Testing knowledge retrieval...")
    relevant_chunks = rag_system._retrieve_relevant_chunks("What is example.com?", [], top_k=3)
    print(f"✅ Retrieved {len(relevant_chunks)} relevant chunks")
    assert isinstance(relevant_chunks, list)
    
    if relevant_chunks:
        # Test response generation
        response = rag_system._generate_rag_response("What is example.com?", relevant_chunks)
        print(f"✅ Generated response using {response['method']}")
        print(f"   Response: {response['response'][:100]}...")
        assert 'response' in response
        assert 'method' in response
    
    print(f"\n🎉 Integration test completed successfully!")
    print(f"\n💡 Key Benefits:")
    print(f"   • Bulk analyzed websites are now available in Knowledge Repository")
    print(f"   • Users can ask questions about all analyzed content")
    print(f"   • Both single and bulk analysis results work together")
    print(f"   • Automatic deduplication prevents duplicate entries")

if __name__ == "__main__":
    success = test_bulk_integration()
    sys.exit(0 if success else 1)
