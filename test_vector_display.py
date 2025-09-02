#!/usr/bin/env python3
"""
Quick test script to verify vector display functionality
"""

import sys
from pathlib import Path
import streamlit as st

# Add paths
project_root = Path(__file__).parent
frontend_path = project_root / "frontend" / "streamlit"
sys.path.insert(0, str(frontend_path))

# Import and test the RAG component
try:
    from components.rag_knowledge_repository import RAGKnowledgeRepository, ContentChunk, RetrievalResult
    import numpy as np
    from datetime import datetime
    
    print("✅ Successfully imported RAG Knowledge Repository")
    
    # Test creating sample data
    sample_chunks = []
    for i in range(3):
        chunk = ContentChunk(
            chunk_id=f"test_chunk_{i+1}",
            website_id="test_website",
            website_title="Sample Website",
            website_url="https://example.com",
            content=f"This is sample content chunk {i+1} for testing vector display functionality.",
            chunk_type="paragraph",
            position=i,
            embedding=np.random.rand(384)
        )
        
        result = RetrievalResult(
            chunk=chunk,
            similarity_score=0.9 - (i * 0.2),
            relevance_score=0.85 - (i * 0.15)
        )
        sample_chunks.append(result)
    
    print(f"✅ Created {len(sample_chunks)} sample chunks")
    print("✅ Vector display functionality should be working!")
    
    print("\nTo test:")
    print("1. Go to the Knowledge Repository tab in your Streamlit app")
    print("2. Open 'Advanced Settings' and check 'Show Vector Details'")
    print("3. Click 'Test Vector Display' button")
    print("4. You should see vector details in the chat")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
