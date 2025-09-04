#!/usr/bin/env python3
"""
Debug script to check ChromaDB migration status
"""

import sqlite3
import os

try:
    import chromadb  # type: ignore[import-untyped]
    CHROMADB_AVAILABLE = True
except ImportError:
    print("❌ ChromaDB not available. Install with: pip install chromadb")
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("❌ SentenceTransformers not available. Install with: pip install sentence-transformers")
    EMBEDDINGS_AVAILABLE = False

def check_databases():
    print("=== Database Status Check ===")
    
    # Check dependencies first
    if not CHROMADB_AVAILABLE:
        print("⚠️ ChromaDB not available - ChromaDB checks will be skipped")
    
    if not EMBEDDINGS_AVAILABLE:
        print("⚠️ SentenceTransformers not available - embedding checks will be skipped")
    
    # Check data directory
    data_dir = "data"
    if os.path.exists(data_dir):
        print(f"✅ Data directory exists: {data_dir}")
        files = os.listdir(data_dir)
        print(f"📁 Files in data: {files}")
    else:
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Check SQLite database
    db_path = "data/rag_knowledge_repository.db"
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_chunks'")
            table_exists = cursor.fetchone()
            
            if table_exists:
                print("✅ SQLite content_chunks table exists")
                
                # Count total chunks
                cursor.execute('SELECT COUNT(*) FROM content_chunks')
                total_count = cursor.fetchone()[0]
                print(f"📊 Total chunks in SQLite: {total_count}")
                
                # Count chunks with embeddings
                cursor.execute('SELECT COUNT(*) FROM content_chunks WHERE embedding IS NOT NULL')
                embedding_count = cursor.fetchone()[0]
                print(f"🔢 Chunks with embeddings: {embedding_count}")
                
                # Show sample chunks
                cursor.execute('SELECT website_id, content, chunk_type FROM content_chunks LIMIT 3')
                examples = cursor.fetchall()
                print(f"📝 Sample chunks ({len(examples)}):")
                for i, (website_id, content, chunk_type) in enumerate(examples):
                    print(f"  {i+1}. Website {website_id}: {content[:80]}... [{chunk_type}]")
            else:
                print("❌ SQLite content_chunks table not found")
    
    except Exception as e:
        print(f"❌ SQLite check failed: {e}")
    
    # Check ChromaDB
    try:
        chroma_path = "data/chroma_db"
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(
            name="website_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        count = collection.count()
        print(f"✅ ChromaDB collection count: {count}")
        
        if count > 0:
            # Show sample data
            results = collection.get(limit=3, include=["documents", "metadatas"])
            print(f"📝 Sample ChromaDB entries ({len(results['ids'])}):")
            for i, doc_id in enumerate(results['ids']):
                content = results['documents'][i]
                metadata = results['metadatas'][i]
                print(f"  {i+1}. ID {doc_id}: {content[:80]}... [Website: {metadata.get('website_title', 'N/A')}]")
        else:
            print("⚠️  ChromaDB collection is empty")
            
    except Exception as e:
        print(f"❌ ChromaDB check failed: {e}")
    
    # Check embedding model
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model loaded successfully")
        # Test embedding
        test_embedding = model.encode("test sentence")
        print(f"🧮 Test embedding shape: {test_embedding.shape}")
    except Exception as e:
        print(f"❌ Sentence transformer check failed: {e}")

if __name__ == "__main__":
    check_databases()
