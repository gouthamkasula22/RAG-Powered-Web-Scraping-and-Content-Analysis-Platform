#!/usr/bin/env python3
"""
Manual ChromaDB migration script
"""

import sqlite3
import json

try:
    import chromadb  # type: ignore[import-untyped]
    import numpy as np  # type: ignore[import-untyped]
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

def migrate_to_chromadb():
    print("🔄 Starting manual migration to ChromaDB...")
    
    # Check dependencies
    if not CHROMADB_AVAILABLE:
        print("❌ ChromaDB is not available. Please install it with: pip install chromadb")
        return False
        
    if not EMBEDDINGS_AVAILABLE:
        print("❌ SentenceTransformers is not available. Please install it with: pip install sentence-transformers")
        return False
    
    # Initialize components
    db_path = "data/rag_knowledge_repository.db"
    chroma_path = "data/chroma_db"
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name="website_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Components initialized")
    
    # Get data from SQLite
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        SELECT c.id, c.website_id, c.content, c.chunk_type, c.position, 
               c.embedding, w.title, w.url 
        FROM content_chunks c 
        JOIN websites w ON c.website_id = w.id 
        WHERE c.embedding IS NOT NULL
        ''')
        
        rows = cursor.fetchall()
        print(f"📊 Found {len(rows)} chunks with embeddings to migrate")
        
        if not rows:
            print("⚠️  No chunks to migrate")
            return
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for chunk_id, website_id, content, chunk_type, position, embedding_blob, website_title, website_url in rows:
            # Convert embedding from blob to numpy array
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            
            ids.append(str(chunk_id))
            documents.append(content)
            metadatas.append({
                'website_id': str(website_id),
                'website_title': website_title,
                'website_url': website_url,
                'chunk_type': chunk_type,
                'position': position
            })
            embeddings.append(embedding.tolist())
        
        # Add to ChromaDB in batch
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        print(f"✅ Successfully migrated {len(ids)} chunks to ChromaDB")
        
        # Verify migration
        final_count = collection.count()
        print(f"🔍 ChromaDB final count: {final_count}")
        
        # Test a query
        test_results = collection.query(
            query_texts=["website content"],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )
        
        if test_results['ids'][0]:
            print(f"🧪 Test query found {len(test_results['ids'][0])} results")
            for i, doc_id in enumerate(test_results['ids'][0]):
                distance = test_results['distances'][0][i]
                similarity = 1.0 - distance
                metadata = test_results['metadatas'][0][i]
                print(f"  - ID {doc_id}: similarity={similarity:.3f} [{metadata.get('website_title', 'N/A')}]")
        else:
            print("⚠️  Test query returned no results")

if __name__ == "__main__":
    migrate_to_chromadb()
