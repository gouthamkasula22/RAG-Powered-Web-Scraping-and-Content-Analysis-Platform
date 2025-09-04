#!/usr/bin/env python3
"""
Comprehensive ChromaDB migration from frontend database
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
    import numpy as np  # type: ignore[import-untyped]
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("❌ SentenceTransformers not available. Install with: pip install sentence-transformers")
    EMBEDDINGS_AVAILABLE = False

def migrate_frontend_data_to_chromadb():
    print("🔄 Starting comprehensive migration from frontend database to ChromaDB...")
    
    # Check dependencies
    if not CHROMADB_AVAILABLE:
        print("❌ ChromaDB is not available. Please install it with: pip install chromadb")
        return False
        
    if not EMBEDDINGS_AVAILABLE:
        print("❌ SentenceTransformers is not available. Please install it with: pip install sentence-transformers")
        return False
    
    # Database paths
    frontend_db = "frontend/streamlit/data/rag_knowledge_repository.db"
    chroma_path = "data/chroma_db"
    
    # Check if frontend database exists
    if not os.path.exists(frontend_db):
        print(f"❌ Frontend database not found at {frontend_db}")
        return False
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Delete existing collection and recreate to avoid conflicts
    try:
        client.delete_collection("website_chunks")
        print("🗑️  Deleted existing collection")
    except:
        pass  # Collection might not exist
    
    collection = client.get_or_create_collection(
        name="website_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize embedding model
    print("🧮 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get data from frontend SQLite database
    with sqlite3.connect(frontend_db) as conn:
        cursor = conn.cursor()
        
        # First, check what we have
        cursor.execute('SELECT COUNT(*) FROM content_chunks')
        total_chunks = cursor.fetchone()[0]
        print(f"📊 Found {total_chunks} total chunks in frontend database")
        
        cursor.execute('SELECT COUNT(*) FROM content_chunks WHERE embedding IS NOT NULL')
        existing_embeddings = cursor.fetchone()[0]
        print(f"📊 Existing embeddings: {existing_embeddings}")
        
        # Get all chunks with their website information
        cursor.execute('''
        SELECT c.id, c.website_id, c.content, c.chunk_type, c.position, 
               c.embedding, w.title, w.url 
        FROM content_chunks c 
        JOIN websites w ON c.website_id = w.id 
        WHERE LENGTH(TRIM(c.content)) > 20
        ORDER BY c.website_id, c.position
        ''')
        
        rows = cursor.fetchall()
        print(f"📊 Processing {len(rows)} valid chunks...")
        
        if not rows:
            print("⚠️  No chunks to migrate")
            return False
        
        # Process chunks in batches
        batch_size = 50
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        processed = 0
        for chunk_id, website_id, content, chunk_type, position, embedding_blob, website_title, website_url in rows:
            try:
                # Generate embedding if not exists or use existing one
                if embedding_blob:
                    # Use existing embedding
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                else:
                    # Generate new embedding
                    embedding = model.encode(content)
                
                # Prepare data for ChromaDB
                ids.append(str(chunk_id))
                documents.append(content)
                metadatas.append({
                    'website_id': str(website_id),
                    'website_title': website_title or 'Unknown',
                    'website_url': website_url or 'Unknown',
                    'chunk_type': chunk_type or 'paragraph',
                    'position': int(position) if position else 0
                })
                embeddings.append(embedding.tolist())
                
                processed += 1
                
                # Process in batches
                if len(ids) >= batch_size:
                    collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )
                    print(f"✅ Processed batch: {processed}/{len(rows)} chunks")
                    
                    # Clear batch
                    ids = []
                    documents = []
                    metadatas = []
                    embeddings = []
                
            except Exception as e:
                print(f"⚠️  Error processing chunk {chunk_id}: {e}")
                continue
        
        # Process remaining batch
        if ids:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            print(f"✅ Processed final batch: {processed}/{len(rows)} chunks")
        
        print(f"✅ Successfully migrated {processed} chunks to ChromaDB")
        
        # Verify migration
        final_count = collection.count()
        print(f"🔍 ChromaDB final count: {final_count}")
        
        # Test queries
        print("🧪 Testing vector search...")
        test_queries = ["website content", "information about", "company details"]
        
        for query in test_queries:
            results = collection.query(
                query_texts=[query],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            
            if results['ids'][0]:
                print(f"✅ Query '{query}' found {len(results['ids'][0])} results")
                best_result = results['metadatas'][0][0] if results['metadatas'][0] else {}
                distance = results['distances'][0][0] if results['distances'][0] else 1.0
                similarity = 1.0 - distance
                print(f"   Best match: similarity={similarity:.3f} from '{best_result.get('website_title', 'N/A')}'")
            else:
                print(f"⚠️  Query '{query}' returned no results")
        
        print("✅ Migration completed successfully!")
        return True

if __name__ == "__main__":
    migrate_frontend_data_to_chromadb()
