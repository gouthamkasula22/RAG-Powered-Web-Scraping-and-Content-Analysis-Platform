"""
Demo script to show RAG Knowledge Repository functionality
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def demo_rag_system():
    """Demonstrate the RAG system capabilities"""
    print("🎭 RAG Knowledge Repository Demo")
    print("=" * 50)
    
    try:
        from frontend.streamlit.components.rag_knowledge_repository import RAGKnowledgeRepository
        
        # Initialize the system
        print("🚀 Initializing RAG system...")
        rag_system = RAGKnowledgeRepository()
        
        # Show system status
        print(f"\n📊 System Status:")
        print(f"   • LLM Service: {'Available' if rag_system.llm_service else 'Not Available'}")
        if rag_system.llm_service:
            provider_count = len(rag_system.llm_service.providers)
            print(f"   • LLM Providers: {provider_count} configured")
            if provider_count == 0:
                print("   • Status: Using rule-based responses (no API keys)")
            else:
                providers = list(rag_system.llm_service.providers.keys())
                print(f"   • Active Providers: {', '.join(providers)}")
        
        print(f"   • Embedding Model: {'Available' if rag_system.embedding_model else 'Not Available'}")
        
        # Check database status
        database_file = os.path.join(project_root, "data", "knowledge_repository.db")
        if os.path.exists(database_file):
            print(f"   • Knowledge Database: Available ({database_file})")
            
            # Get some stats
            import sqlite3
            conn = sqlite3.connect(database_file)
            cursor = conn.cursor()
            
            try:
                cursor.execute("SELECT COUNT(*) FROM knowledge_chunks")
                chunk_count = cursor.fetchone()[0]
                print(f"   • Knowledge Chunks: {chunk_count} stored")
                
                cursor.execute("SELECT COUNT(DISTINCT website_url) FROM knowledge_chunks")
                website_count = cursor.fetchone()[0]
                print(f"   • Websites Analyzed: {website_count}")
                
            except Exception as e:
                print(f"   • Database Status: Present but may need initialization")
            finally:
                conn.close()
        else:
            print(f"   • Knowledge Database: Not found")
            print(f"   • Note: Analyze some websites first to populate the knowledge base")
        
        print(f"\n🧠 Testing Question Answering...")
        
        # Test questions
        test_questions = [
            "Who is the CEO?",
            "What is the company about?",
            "Tell me about the leadership team",
            "What services do they offer?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            
            # Retrieve relevant chunks
            relevant_chunks = rag_system._retrieve_relevant_chunks(question, top_k=3)
            
            if relevant_chunks:
                print(f"   📚 Found {len(relevant_chunks)} relevant chunks")
                
                # Generate response
                result = rag_system._generate_rag_response(question, relevant_chunks)
                response = result["response"]
                method = result["method"]
                
                print(f"   🤖 Response Method: {method}")
                print(f"   💬 Answer: {response[:200]}{'...' if len(response) > 200 else ''}")
            else:
                print(f"   🔍 No relevant content found for this question")
        
        print(f"\n" + "=" * 50)
        print(f"🎉 Demo completed!")
        
        # Show next steps
        print(f"\n📋 Next Steps:")
        print(f"   1. To enable AI responses, set environment variables:")
        print(f"      • GOOGLE_API_KEY=your_gemini_api_key")
        print(f"      • ANTHROPIC_API_KEY=your_claude_api_key")
        print(f"   2. To add content, use the main Streamlit app to analyze websites")
        print(f"   3. Use the RAG Knowledge Repository for intelligent Q&A")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_rag_system()
    sys.exit(0 if success else 1)
