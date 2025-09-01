"""
RAG-based Knowledge Repository Implementation
Retrieval-Augmented Generation for intelligent Q&A about analyzed websites
"""

import streamlit as st
import asyncio
import sqlite3
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import re
import sys
import os
from dataclasses import dataclass
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add backend to Python path for LLM service imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
backend_path = project_root / "backend"
src_path = project_root / "src"

# Add paths to sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(src_path))

# Debug: Print paths to help troubleshoot
print(f"Project root: {project_root}")
print(f"Backend path: {backend_path}")
print(f"Backend exists: {backend_path.exists()}")

# For embeddings and vector search
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    # Store warning to display later, don't call st.warning during import

# For ChromaDB vector database
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    # Store warning to display later, don't call st.warning during import

# For LLM integration
try:
    from backend.src.infrastructure.llm.service import ProductionLLMService, LLMServiceConfig
    from backend.src.application.interfaces.llm import LLMRequest, AnalysisRequest
    LLM_SERVICE_AVAILABLE = True
    print("✅ LLM service imports successful")
except ImportError as e:
    LLM_SERVICE_AVAILABLE = False
    print(f"❌ LLM service import failed: {e}")

@dataclass
class ContentChunk:
    """Represents a chunk of content with metadata"""
    chunk_id: str
    website_id: str
    website_title: str
    website_url: str
    content: str
    chunk_type: str  # 'paragraph', 'heading', 'list', 'table'
    position: int
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    """Result from content retrieval"""
    chunk: ContentChunk
    similarity_score: float
    relevance_score: float

class RAGKnowledgeRepository:
    """RAG-based Knowledge Repository with ChromaDB vector search and intelligent retrieval"""
    
    def __init__(self):
        # Use the correct database path where the actual data is stored
        self.db_path = "/app/data/rag_knowledge_repository.db"
        self.chroma_path = "/app/data/chroma_db"
        self.embedding_model = None
        self.chroma_client = None
        self.chroma_collection = None
        self.chunk_size = 500  # characters per chunk
        self.chunk_overlap = 100  # overlap between chunks
        
        # Store initialization messages to display later
        self.initialization_messages = []
        
        # Initialize embedding model if available
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Store success message instead of displaying immediately
                self.initialization_messages.append(("success", "✅ Embedding model loaded successfully"))
            except Exception as e:
                # Store error message instead of displaying immediately
                self.embedding_model = None
                self.initialization_messages.append(("error", f"❌ Failed to load embedding model: {e}"))
        else:
            self.initialization_messages.append(("warning", "⚠️ sentence-transformers not available. Install with: pip install sentence-transformers"))
        
        # Initialize ChromaDB if available
        if CHROMADB_AVAILABLE and self.embedding_model:
            try:
                self._init_chromadb()
            except Exception as e:
                print(f"ChromaDB initialization failed: {e}")
                self.chroma_client = None
                self.initialization_messages.append(("error", f"❌ ChromaDB initialization failed: {e}"))
        elif not CHROMADB_AVAILABLE:
            self.initialization_messages.append(("warning", "⚠️ ChromaDB not available. Install with: pip install chromadb"))
        
        # Initialize LLM service for intelligent responses
        self.llm_service = None
        if LLM_SERVICE_AVAILABLE:
            try:
                # Import the config class
                config = LLMServiceConfig(
                    primary_provider="gemini",
                    premium_provider="claude",
                    max_cost_per_request=0.05
                )
                
                # Initialize the service with config
                self.llm_service = ProductionLLMService(config)
                if self.llm_service.providers:
                    provider_names = list(self.llm_service.providers.keys())
                    self.initialization_messages.append(("success", f"✅ LLM service available with providers: {', '.join(provider_names)}"))
                else:
                    # Store messages to display later instead of immediate Streamlit calls
                    self.initialization_messages.extend([
                        ("warning", "⚠️ LLM service initialized but no providers available"),
                        ("info", "💡 **To enable AI responses, set up API keys:**"),
                        ("code", "# For Google Gemini (Free):\nGOOGLE_API_KEY=your_google_gemini_api_key_here\n\n# For Anthropic Claude (Premium):\nANTHROPIC_API_KEY=your_anthropic_claude_api_key_here"),
                        ("info", "🔧 **Meanwhile, using enhanced rule-based responses**")
                    ])
            except Exception as e:
                self.initialization_messages.extend([
                    ("error", f"❌ Failed to initialize LLM service: {e}"),
                    ("info", "🔧 **Using enhanced rule-based responses as fallback**")
                ])
        else:
            self.initialization_messages.append(("info", "ℹ️ Using rule-based responses (LLM service not available)"))
        
        self._init_database()
        self._migrate_to_chromadb_if_needed()
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            
            # Create or get collection for website chunks
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="website_chunks",
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            print(f"✅ ChromaDB initialized successfully at {self.chroma_path}")
            return True
            
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            self.chroma_client = None
            self.chroma_collection = None
            return False
    
    def _migrate_to_chromadb_if_needed(self):
        """Migrate existing SQLite embeddings to ChromaDB if needed"""
        if not self.chroma_collection or not self.embedding_model:
            return
        
        try:
            # Check if ChromaDB is empty but SQLite has embeddings
            chroma_count = self.chroma_collection.count()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM content_chunks WHERE embedding IS NOT NULL')
                sqlite_count = cursor.fetchone()[0]
            
            if chroma_count == 0 and sqlite_count > 0:
                print(f"🔄 Migrating {sqlite_count} embeddings from SQLite to ChromaDB...")
                self._perform_migration()
                
        except Exception as e:
            print(f"Migration check failed: {e}")
    
    def _perform_migration(self):
        """Perform the actual migration from SQLite to ChromaDB"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT c.id, c.website_id, c.content, c.chunk_type, c.position, 
                       c.metadata, c.created_at, w.title, w.url
                FROM content_chunks c
                JOIN websites w ON c.website_id = w.id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.website_id, c.position
                ''')
                
                chunks_data = cursor.fetchall()
                
            if not chunks_data:
                return
            
            # Prepare data for ChromaDB batch insert
            chunk_ids = []
            documents = []
            metadatas = []
            
            for chunk_data in chunks_data:
                chunk_id, website_id, content, chunk_type, position, metadata, created_at, title, url = chunk_data
                
                chunk_ids.append(chunk_id)
                documents.append(content)
                metadatas.append({
                    "website_id": website_id,
                    "website_title": title,
                    "website_url": url,
                    "chunk_type": chunk_type,
                    "position": position,
                    "created_at": created_at,
                    "metadata": metadata or "{}"
                })
            
            # Generate embeddings for all chunks
            print("🧠 Generating embeddings for migration...")
            embeddings = self.embedding_model.encode(documents)
            
            # Batch insert into ChromaDB
            self.chroma_collection.add(
                ids=chunk_ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings.tolist()
            )
            
            print(f"✅ Successfully migrated {len(chunk_ids)} chunks to ChromaDB")
            
            # Remove embeddings from SQLite to save space (keep the chunks for fallback)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE content_chunks SET embedding = NULL')
                conn.commit()
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            raise
    
    def _init_database(self):
        """Initialize the SQLite database for website metadata (ChromaDB handles vectors)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Websites table (unchanged)
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS websites (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    summary TEXT,
                    content_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0
                )
                ''')
                
                # Content chunks table (no embedding column - ChromaDB handles vectors)
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS content_chunks (
                    id TEXT PRIMARY KEY,
                    website_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_type TEXT DEFAULT 'paragraph',
                    position INTEGER NOT NULL,
                    embedding BLOB,  -- Kept for migration compatibility
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (website_id) REFERENCES websites (id) ON DELETE CASCADE
                )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_website_id ON content_chunks(website_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_position ON content_chunks(position)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_websites_url ON websites(url)')
                
                conn.commit()
                
        except Exception as e:
            # Store error message instead of immediate Streamlit display
            self.initialization_messages.append(("error", f"❌ Database initialization failed: {e}"))
    
    def _display_initialization_messages(self):
        """Display any initialization messages that were stored during __init__"""
        if not hasattr(self, 'initialization_messages'):
            return
            
        # Only show messages if not already shown (prevent popup on every navigation)
        if hasattr(st.session_state, 'rag_repo_initialized') and st.session_state.rag_repo_initialized:
            return
        
        for message_type, message in self.initialization_messages:
            if message_type == "success":
                st.success(message)
            elif message_type == "error":
                st.error(message)
            elif message_type == "warning":
                st.warning(message)
            elif message_type == "info":
                st.info(message)
            elif message_type == "code":
                st.code(message)
        
        # Clear messages after displaying so they don't show again
        self.initialization_messages = []

    def render(self):
        """Render the professional RAG Knowledge Repository interface"""
        
        # Display any initialization messages first
        self._display_initialization_messages()
        
        # Inject professional CSS
        self._inject_professional_css()
        
        # Professional header
        st.markdown("""
        <div class="rag-header">
            <h1>Knowledge Repository</h1>
            <p>Intelligent Q&A powered by Retrieval-Augmented Generation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load available websites
        websites = self._get_available_websites()
        
        # Status and statistics bar
        self._render_status_bar(websites)
        
        # Chat input at top level (outside any containers) for proper functionality
        question = st.chat_input(
            placeholder="Ask anything about your analyzed websites...",
            key="rag_chat_input"
        )
        
        # Handle query submission (chat_input automatically clears on submit)
        if question and question.strip():
            # Get selected websites for the query based on multiselect widget
            if "website_dropdown_select" in st.session_state and st.session_state.website_dropdown_select:
                selected_options = st.session_state.website_dropdown_select
                
                if "All websites" in selected_options:
                    selected_websites = websites
                    st.info("🌐 Querying **all websites**")
                else:
                    # Get website titles
                    website_titles = [f"{website['title'][:60]}..." if len(website['title']) > 60 else website['title'] 
                                     for website in websites]
                    dropdown_options = ["All websites"] + website_titles
                    
                    # Get indices of selected websites (subtract 1 because "All websites" is at index 0)
                    selected_indices = []
                    for option in selected_options:
                        if option in dropdown_options and option != "All websites":
                            idx = dropdown_options.index(option) - 1
                            if 0 <= idx < len(websites):
                                selected_indices.append(idx)
                    
                    selected_websites = [websites[i] for i in selected_indices] if selected_indices else websites
                    st.info(f"🎯 Querying **{len(selected_websites)} selected website(s)**: {', '.join([w['title'][:30] + '...' if len(w['title']) > 30 else w['title'] for w in selected_websites])}")
            else:
                selected_websites = websites
                st.info("🌐 Querying **all websites** (no selection found)")
            
            self._handle_rag_query(question, selected_websites)
        
        if not websites:
            self._render_empty_state()
        else:
            self._render_main_interface(websites)
    
    def render_main_interface(self):
        """Alias for render() method to maintain compatibility"""
        self.render()
    
    def _inject_professional_css(self):
        """Inject professional CSS styling"""
        st.markdown("""
        <style>
        .rag-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }
        
        .rag-header h1 {
            color: white;
            margin: 0;
            font-size: 1.8rem;
            font-weight: 600;
            letter-spacing: -0.025em;
        }
        
        .rag-header p {
            color: rgba(255,255,255,0.9);
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
            font-weight: 400;
        }
        
        .rag-status-bar {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .rag-stat {
            display: inline-block;
            margin-right: 2rem;
        }
        
        .rag-stat-label {
            font-size: 0.875rem;
            color: #6c757d;
            font-weight: 500;
            display: block;
        }
        
        .rag-stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin-top: 0.25rem;
        }
        
        .rag-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }
        
        .rag-card-header {
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            padding: 1rem 1.5rem;
        }
        
        .rag-card-header h3 {
            margin: 0;
            font-size: 1.125rem;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .rag-card-header p {
            margin: 0.5rem 0 0 0;
            color: #6c757d;
            font-size: 0.875rem;
        }
        
        .rag-card-body {
            padding: 1.5rem;
        }
        
        .rag-chat-message {
            margin: 1rem 0;
            display: flex;
        }
        
        .rag-chat-user {
            justify-content: flex-end;
        }
        
        .rag-chat-assistant {
            justify-content: flex-start;
        }
        
        .rag-message-bubble {
            padding: 0.75rem 1rem;
            border-radius: 18px;
            max-width: 85%;
            font-size: 0.875rem;
            line-height: 1.4;
        }
        
        .rag-message-user {
            background: #007bff;
            color: white;
            border-radius: 18px 18px 4px 18px;
            max-width: 70%;
        }
        
        .rag-message-assistant {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            color: #2c3e50;
            border-radius: 18px 18px 18px 4px;
        }
        
        .rag-message-timestamp {
            font-size: 0.75rem;
            opacity: 0.8;
            margin-top: 0.25rem;
        }
        
        .rag-message-sources {
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px solid #e9ecef;
            font-size: 0.75rem;
            color: #6c757d;
        }
        
        .rag-empty-state {
            text-align: center;
            padding: 4rem 2rem;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin: 2rem 0;
        }
        
        .rag-empty-icon {
            width: 80px;
            height: 80px;
            margin: 0 auto 1.5rem auto;
            background: #e9ecef;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            color: #6c757d;
        }
        
        .rag-website-item {
            border-top: 1px solid #e9ecef;
            padding: 1rem 1.5rem;
            transition: background-color 0.2s;
        }
        
        .rag-website-item:first-child {
            border-top: none;
        }
        
        .rag-website-item:hover {
            background-color: #f8f9fa;
        }
        
        .rag-sources {
            margin-top: 1rem;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 1rem;
        }
        
        .rag-source-item {
            margin-bottom: 0.75rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #dee2e6;
        }
        
        .rag-source-item:last-child {
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }
        
        @media (max-width: 768px) {
            .rag-header {
                padding: 1.5rem 1rem;
            }
            
            .rag-header h1 {
                font-size: 1.5rem;
            }
            
            .rag-stat {
                display: block;
                margin-right: 0;
                margin-bottom: 1rem;
            }
            
            .rag-message-bubble {
                max-width: 95%;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_status_bar(self, websites: List[Dict]):
        """Render professional status and statistics bar with ChromaDB info"""
        total_chunks = sum(w.get('chunk_count', 0) for w in websites)
        
        # Check ChromaDB status
        chroma_count = 0
        if self.chroma_collection:
            try:
                chroma_count = self.chroma_collection.count()
            except Exception:
                chroma_count = 0
        
        st.markdown(f"""
        <div class="rag-status-bar">
            <div class="rag-stat">
                <span class="rag-stat-label">INDEXED WEBSITES</span>
                <div class="rag-stat-value">{len(websites)}</div>
            </div>
            <div class="rag-stat">
                <span class="rag-stat-label">CONTENT CHUNKS</span>
                <div class="rag-stat-value">{total_chunks:,}</div>
            </div>
            <div class="rag-stat">
                <span class="rag-stat-label">VECTOR DATABASE</span>
                <div class="rag-stat-value" style="color: {'#28a745' if self.chroma_collection else '#dc3545'};">
                    {'CHROMADB' if self.chroma_collection else 'SQLITE'} ({chroma_count:,} vectors)
                </div>
            </div>
            <div class="rag-stat">
                <span class="rag-stat-label">EMBEDDINGS</span>
                <div class="rag-stat-value" style="color: {'#28a745' if self.embedding_model else '#dc3545'};">
                    {'ACTIVE' if self.embedding_model else 'DISABLED'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_empty_state(self):
        """Render professional empty state"""
        st.markdown("""
        <div class="rag-empty-state">
            <div class="rag-empty-icon">📚</div>
            <h3 style="
                color: #2c3e50;
                margin: 0 0 1rem 0;
                font-weight: 600;
                font-size: 1.25rem;
            ">Knowledge Base Empty</h3>
            <p style="
                color: #6c757d;
                margin: 0 0 2rem 0;
                line-height: 1.6;
                max-width: 500px;
                margin-left: auto;
                margin-right: auto;
            ">
                Your knowledge repository is ready to be populated with analyzed website content. 
                Start by analyzing websites using the main analyzer, then return here for intelligent Q&A.
            </p>
            <div class="rag-card" style="max-width: 600px; margin: 0 auto;">
                <div class="rag-card-header">
                    <h3>Getting Started</h3>
                </div>
                <div class="rag-card-body">
                    <ol style="margin: 0; padding-left: 1.5rem; color: #495057; line-height: 1.8;">
                        <li>Navigate to the <strong>Analysis</strong> tab</li>
                        <li>Analyze websites using comprehensive analysis</li>
                        <li>Return here to ask questions about the content</li>
                        <li>Get intelligent answers powered by RAG technology</li>
                    </ol>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if we can load from session state
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(
                "Load from Analysis Results", 
                type="primary",
                use_container_width=True,
                help="Load websites from current session analysis results"
            ):
                if self._load_from_session_state():
                    st.rerun()
                else:
                    st.warning("No analysis results found in current session.")
    
    def _render_main_interface(self, websites: List[Dict]):
        """Render the main professional interface with intelligent Q&A"""
        
        # Main content area with two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chat interface with proper input positioning
            self._render_chat_interface(websites)
        
        with col2:
            # Sidebar with website management
            self._render_sidebar_controls(websites)
    
    def _render_chat_interface(self, websites: List[Dict]):
        """Render professional chat interface with proper input positioning"""
        
        st.markdown("""
        <div class="rag-card">
            <div class="rag-card-header">
                <h3>Intelligent Q&A</h3>
                <p>Ask questions about your analyzed websites</p>
            </div>
            <div class="rag-card-body">
        """, unsafe_allow_html=True)
        
        # Website selection with dropdown
        if len(websites) > 1:
            st.markdown("**Select websites to query:**")
            
            # Create options: "All websites" + individual website titles
            website_titles = [f"{website['title'][:60]}..." if len(website['title']) > 60 else website['title'] 
                             for website in websites]
            
            dropdown_options = ["All websites"] + website_titles
            
            selected_options = st.multiselect(
                "Choose websites to query",
                options=dropdown_options,
                default=["All websites"],  # Default to "All websites"
                key="website_dropdown_select"
            )
            
            if not selected_options:
                st.warning("Please select at least one option.")
                return
            
            # Handle selection logic
            if "All websites" in selected_options:
                selected_websites = websites
                if len(selected_options) > 1:
                    st.info("🌐 Querying **all websites** (other selections ignored)")
            else:
                # Get indices of selected websites
                selected_indices = [dropdown_options.index(option) - 1 for option in selected_options]
                selected_websites = [websites[i] for i in selected_indices]
                st.info(f"🎯 Querying **{len(selected_websites)} selected website(s)**")
        else:
            selected_websites = websites
        
        # Display chat history
        self._display_professional_chat_history()
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    def _render_sidebar_controls(self, websites: List[Dict]):
        """Render professional sidebar with website management"""
        
        st.markdown("""
        <div class="rag-card">
            <div class="rag-card-header">
                <h3>Website Management</h3>
            </div>
            <div class="rag-card-body">
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh", use_container_width=True):
                self._load_from_session_state()
                st.rerun()
        
        with col2:
            if st.button("Clear All", use_container_width=True, type="secondary"):
                if st.button("Confirm Clear", type="secondary", use_container_width=True):
                    self._clear_all_data()
                    st.rerun()
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Website list
        self._render_website_list(websites)
    
    def _render_website_list(self, websites: List[Dict]):
        """Render professional website list"""
        
        st.markdown("""
        <div class="rag-card">
            <div class="rag-card-header">
                <h3>Indexed Websites</h3>
            </div>
        """, unsafe_allow_html=True)
        
        for website in websites:
            st.markdown(f"""
            <div class="rag-website-item">
                <h4 style="
                    margin: 0 0 0.5rem 0;
                    font-size: 0.875rem;
                    font-weight: 600;
                    color: #2c3e50;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                ">{website['title']}</h4>
                <p style="
                    margin: 0 0 0.5rem 0;
                    font-size: 0.75rem;
                    color: #6c757d;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                ">{website['url']}</p>
                <div style="display: flex; gap: 1rem; font-size: 0.75rem; color: #6c757d;">
                    <span>{website.get('chunk_count', 0)} chunks</span>
                    <span>{website.get('created_at', 'Unknown')[:10] if website.get('created_at') else 'Unknown'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _handle_rag_query(self, question: str, websites: List[Dict]):
        """Handle RAG-based query processing with professional interface"""
        
        # Initialize chat history
        if "rag_chat_history" not in st.session_state:
            st.session_state.rag_chat_history = []
        
        # Add user message
        st.session_state.rag_chat_history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Process query with professional loading indicator
        with st.spinner("🧠 Processing with AI..."):
            
            # Step 1: Retrieve relevant chunks with website filtering
            relevant_chunks = self._retrieve_relevant_chunks(question, websites, top_k=5)
            
            if not relevant_chunks:
                response = self._generate_no_results_response(question, websites)
                sources = []
                method_used = "No relevant content found"
            else:
                # Step 2: Generate response using RAG
                result = self._generate_rag_response(question, relevant_chunks)
                response = result["response"]
                method_used = result["method"]
                sources = [f"{chunk.chunk.website_url}" for chunk in relevant_chunks[:3]]
        
        # Add assistant response to chat history
        st.session_state.rag_chat_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now().strftime("%H:%M"),
            "sources": sources,
            "method_used": method_used,
            "relevant_chunks": relevant_chunks if relevant_chunks else []
        })
        
        # Rerun to update display
        st.rerun()
    
    def _on_input_change(self):
        """Handle Enter key press in the input field"""
        if "rag_query_input" in st.session_state and st.session_state.rag_query_input.strip():
            # Get all websites for query (or handle website selection)
            websites = self._get_available_websites()
            
            # Get selected websites if in session state
            if "website_dropdown_select" in st.session_state:
                selected_options = st.session_state.website_dropdown_select
                
                if "All websites" in selected_options:
                    selected_websites = websites
                else:
                    # Get website titles
                    website_titles = [f"{website['title'][:60]}..." if len(website['title']) > 60 else website['title'] 
                                     for website in websites]
                    dropdown_options = ["All websites"] + website_titles
                    
                    # Get indices of selected websites
                    selected_indices = [dropdown_options.index(option) - 1 for option in selected_options]
                    selected_websites = [websites[i] for i in selected_indices]
            else:
                selected_websites = websites
            
            # Process the query
            question = st.session_state.rag_query_input
            self._handle_rag_query(question, selected_websites)
            
            # Clear input
            st.session_state.rag_query_input = ""

    def _display_professional_chat_history(self):
        """Display chat history with professional styling"""
        if "rag_chat_history" not in st.session_state or not st.session_state.rag_chat_history:
            st.markdown("""
            <div style="
                text-align: center;
                padding: 2rem;
                color: #6c757d;
                font-style: italic;
            ">
                Start a conversation by asking a question above
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Display last 10 messages
        recent_messages = st.session_state.rag_chat_history[-10:]
        
        for message in recent_messages:
            if message["role"] == "user":
                # User message - use pure Streamlit components
                st.markdown("**👤 You:**")
                st.write(message['content'])
                st.caption(f"🕒 {message['timestamp']}")
                
            else:
                # Assistant message - use pure Streamlit components, no HTML
                st.markdown("**🤖 Assistant:**")
                st.write(message['content'])
                
                # Show method used
                if message.get("method_used"):
                    st.caption(f"📡 {message['method_used']}")
                
                # Display sources separately if available
                if message.get("sources"):
                    sources_list = list(set(message["sources"]))  # Remove duplicates
                    with st.expander("📄 Sources", expanded=False):
                        for i, source in enumerate(sources_list, 1):
                            st.write(f"{i}. {source}")
                
                # Display timestamp
                st.caption(f"🕒 {message['timestamp']}")
                st.markdown("---")  # Separator between messages
    
    def _retrieve_relevant_chunks(self, question: str, selected_websites: List[Dict], top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve most relevant content chunks using ChromaDB or SQLite fallback with website filtering"""
        try:
            if not self.embedding_model:
                # Fallback to keyword-based search with website filtering
                return self._keyword_based_retrieval(question, selected_websites, top_k)
            
            # Create list of website URLs for filtering
            selected_urls = [website['url'] for website in selected_websites] if selected_websites else []
            
            # Try ChromaDB first (preferred method)
            if self.chroma_collection:
                try:
                    # Query ChromaDB for similar chunks
                    results = self.chroma_collection.query(
                        query_texts=[question],
                        n_results=top_k * 3,  # Get more results to filter by website
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    retrieval_results = []
                    
                    # Process ChromaDB results with website filtering
                    if results['ids'][0]:  # Check if we got any results
                        for i, chunk_id in enumerate(results['ids'][0]):
                            content = results['documents'][0][i]
                            metadata = results['metadatas'][0][i]
                            website_url = metadata.get('website_url', '')
                            
                            # Filter by selected websites
                            if selected_urls and website_url not in selected_urls:
                                continue
                                
                            # ChromaDB returns distances, convert to similarity (1 - distance for cosine)
                            distance = results['distances'][0][i]
                            similarity_score = 1.0 - distance
                            
                            chunk = ContentChunk(
                                chunk_id=chunk_id,
                                website_id=metadata.get('website_id', ''),
                                website_title=metadata.get('website_title', ''),
                                website_url=metadata.get('website_url', ''),
                                content=content,
                                chunk_type=metadata.get('chunk_type', 'paragraph'),
                                position=metadata.get('position', 0)
                            )
                            
                            retrieval_results.append(RetrievalResult(
                                chunk=chunk,
                                similarity_score=float(similarity_score),
                                relevance_score=float(similarity_score) * self._calculate_content_quality_score(content)
                            ))
                    
                    # Sort by relevance score and return top_k results
                    retrieval_results.sort(key=lambda x: x.relevance_score, reverse=True)
                    return retrieval_results[:top_k]
                    
                except Exception as e:
                    print(f"ChromaDB query failed: {e}, falling back to SQLite")
                    # Fall through to SQLite fallback
            
            # SQLite fallback method with website filtering
            return self._sqlite_retrieval(question, selected_websites, top_k)
                
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            return self._keyword_based_retrieval(question, top_k)
    
    def _sqlite_retrieval(self, question: str, selected_websites: List[Dict], top_k: int) -> List[RetrievalResult]:
        """Fallback SQLite-based retrieval with embeddings and website filtering"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([question])[0]
            
            # Create website URL filter
            selected_urls = [website['url'] for website in selected_websites] if selected_websites else []
            
            # Build query with website filtering
            if selected_urls:
                placeholders = ','.join(['?' for _ in selected_urls])
                where_clause = f"AND w.url IN ({placeholders})"
                query_params = selected_urls
            else:
                where_clause = ""
                query_params = []
            
            # Retrieve chunks with embeddings from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                SELECT c.*, w.title, w.url 
                FROM content_chunks c 
                JOIN websites w ON c.website_id = w.id 
                WHERE c.embedding IS NOT NULL {where_clause}
                ORDER BY c.position
                ''', query_params)
                
                results = []
                for row in cursor.fetchall():
                    chunk_id, website_id, content, chunk_type, position, embedding_blob, metadata, created_at, title, url = row
                    
                    # Deserialize embedding
                    if embedding_blob:
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        
                        # Calculate similarity
                        similarity = np.dot(query_embedding, embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                        )
                        
                        chunk = ContentChunk(
                            chunk_id=chunk_id,
                            website_id=website_id,
                            website_title=title,
                            website_url=url,
                            content=content,
                            chunk_type=chunk_type,
                            position=position,
                            embedding=embedding
                        )
                        
                        results.append(RetrievalResult(
                            chunk=chunk,
                            similarity_score=float(similarity),
                            relevance_score=float(similarity) * self._calculate_content_quality_score(content)
                        ))
                
                # Sort by relevance and return top k
                results.sort(key=lambda x: x.relevance_score, reverse=True)
                return results[:top_k]
                
        except Exception as e:
            print(f"SQLite retrieval failed: {e}")
            return self._keyword_based_retrieval(question, selected_websites, top_k)
    
    def _keyword_based_retrieval(self, question: str, selected_websites: List[Dict], top_k: int) -> List[RetrievalResult]:
        """Fallback keyword-based retrieval when embeddings are not available with website filtering"""
        try:
            # Extract keywords from question
            keywords = self._extract_keywords(question)
            
            # Create website URL filter
            selected_urls = [website['url'] for website in selected_websites] if selected_websites else []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build search query
                search_conditions = []
                params = []
                
                for keyword in keywords:
                    search_conditions.append("(c.content LIKE ? OR w.title LIKE ?)")
                    params.extend([f"%{keyword}%", f"%{keyword}%"])
                
                # Add website filtering
                website_conditions = []
                if selected_urls:
                    for url in selected_urls:
                        website_conditions.append("w.url = ?")
                        params.append(url)
                
                if search_conditions:
                    # Build the complete WHERE clause
                    where_clauses = [f"({' OR '.join(search_conditions)})"]
                    if website_conditions:
                        where_clauses.append(f"({' OR '.join(website_conditions)})")
                    
                    query = f'''
                    SELECT c.*, w.title, w.url 
                    FROM content_chunks c 
                    JOIN websites w ON c.website_id = w.id 
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY c.position
                    '''
                    
                    cursor.execute(query, params)
                    results = []
                    
                    for row in cursor.fetchall():
                        chunk_id, website_id, content, chunk_type, position, embedding_blob, metadata, created_at, title, url = row
                        
                        chunk = ContentChunk(
                            chunk_id=chunk_id,
                            website_id=website_id,
                            website_title=title,
                            website_url=url,
                            content=content,
                            chunk_type=chunk_type,
                            position=position
                        )
                        
                        # Calculate keyword-based score
                        score = self._calculate_keyword_score(question, content)
                        
                        results.append(RetrievalResult(
                            chunk=chunk,
                            similarity_score=score,
                            relevance_score=score
                        ))
                    
                    # Sort by relevance
                    results.sort(key=lambda x: x.relevance_score, reverse=True)
                    return results[:top_k]
            
            return []
            
        except Exception as e:
            st.error(f"Keyword retrieval failed: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common words and extract meaningful terms
        stop_words = {'is', 'are', 'was', 'were', 'what', 'who', 'where', 'when', 'how', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_keyword_score(self, question: str, content: str) -> float:
        """Calculate relevance score based on keyword matching"""
        question_words = set(self._extract_keywords(question))
        content_words = set(self._extract_keywords(content))
        
        if not question_words:
            return 0.0
        
        intersection = question_words.intersection(content_words)
        return len(intersection) / len(question_words)
    
    def _calculate_content_quality_score(self, content: str) -> float:
        """Calculate content quality score based on various factors"""
        base_score = 1.0
        
        # Length bonus (not too short, not too long)
        length = len(content)
        if 100 <= length <= 1000:
            base_score += 0.2
        elif length > 50:
            base_score += 0.1
        
        # Structure bonus (paragraphs, lists)
        if '\n' in content or '•' in content or '-' in content:
            base_score += 0.1
        
        # Information density (avoid repetitive content)
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        if total_words > 0:
            diversity = unique_words / total_words
            base_score += diversity * 0.2
        
        return min(base_score, 2.0)  # Cap at 2.0
    
    def _generate_rag_response(self, question: str, relevant_chunks: List[RetrievalResult]) -> dict:
        """Generate response using retrieved chunks and intelligent processing"""
        
        # Prepare clean context from retrieved chunks
        context_parts = []
        sources = []
        
        for result in relevant_chunks:
            chunk = result.chunk
            # Add content without source formatting for cleaner processing
            context_parts.append(chunk.content.strip())
            sources.append(f"{chunk.website_title} ({chunk.website_url})")
        
        # Join context with clear separators
        context = "\n\n".join(context_parts)
        
        # Use intelligent response generation (simulates LLM behavior)
        # TODO: For production, integrate with actual LLM APIs like:
        # - OpenAI GPT-4
        # - Anthropic Claude
        # - Google Gemini
        # - Local models via Ollama
        result = self._generate_contextual_response(question, context, sources)
        
        return result
    
    def _generate_contextual_response(self, question: str, context: str, sources: List[str]) -> dict:
        """Generate contextual response using LLM or fallback to rule-based processing"""
        
        if not context.strip():
            return {
                "response": "I couldn't find relevant information to answer your question.",
                "method": "No Context"
            }
        
        # Try LLM first if available
        if self.llm_service and self.llm_service.providers:
            try:
                # Handle async call properly in different contexts
                import asyncio
                try:
                    # Try to get the current event loop
                    loop = asyncio.get_running_loop()
                    # If we're in a running loop, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._generate_llm_response(question, context, sources))
                        response_text = future.result(timeout=30)
                except RuntimeError:
                    # No running loop, use asyncio.run
                    response_text = asyncio.run(self._generate_llm_response(question, context, sources))
                
                return {
                    "response": response_text,
                    "method": "AI Response (Gemini/Claude)"
                }
            except Exception as e:
                # Only show warning in debug mode, don't spam user
                print(f"LLM response failed, using fallback: {str(e)}")
        
        # Fallback to rule-based processing
        fallback_response = self._generate_intelligent_response_fallback(question, context)
        return {
            "response": fallback_response,
            "method": "Rule-based Response"
        }
    
    async def _generate_llm_response(self, question: str, context: str, sources: List[str]) -> str:
        """Generate response using actual LLM service"""
        
        # Create RAG prompt for the LLM
        rag_prompt = self._create_rag_prompt(question, context)
        
        # Create LLM request for direct provider use
        llm_request = LLMRequest(
            prompt=rag_prompt,
            max_tokens=512,  # Short response
            temperature=0.3
        )
        
        # Get first available provider and use it directly
        if not self.llm_service or not self.llm_service.providers:
            raise Exception("No LLM providers available")
        
        # Use Gemini first (free tier)
        provider = None
        if "gemini" in self.llm_service.providers:
            provider = self.llm_service.providers["gemini"]
        elif "claude" in self.llm_service.providers:
            provider = self.llm_service.providers["claude"]
        else:
            provider = list(self.llm_service.providers.values())[0]
        
        # Get response directly from provider
        response = await provider.generate_response(llm_request)
        
        if response.success and response.content:
            return response.content.strip()
        else:
            error_msg = response.error_message or "Unknown error"
            raise Exception(f"LLM request failed: {error_msg}")
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a well-structured RAG prompt for the LLM"""
        
        prompt = f"""You are a helpful assistant answering questions based on website content. Answer the user's question directly and concisely using ONLY the provided context.

Context from analyzed websites:
{context}

Question: {question}

Instructions:
- Answer directly and concisely (1-2 sentences maximum)
- Use ONLY information from the context above
- If the context doesn't contain the answer, say "I couldn't find that information in the analyzed content."
- Don't provide analysis or explanations, just answer the question
- For "who is" questions, provide the person's name and role/title
- For "what is" questions, provide a brief definition or description

Answer:"""
        
        return prompt
    
    def _generate_intelligent_response_fallback(self, question: str, context: str) -> str:
        """Fallback intelligent response when LLM is not available"""
        
        question_lower = question.lower()
        
        # Extract relevant sentences from context that relate to the question
        relevant_sentences = self._find_relevant_sentences(question, context)
        
        if not relevant_sentences:
            return "I couldn't find specific information to answer your question in the provided content."
        
        # Generate response based on question type and relevant content
        if any(word in question_lower for word in ['who is', 'who are']):
            return self._answer_who_question(question, relevant_sentences)
        elif any(word in question_lower for word in ['what is', 'what are', 'what does']):
            return self._answer_what_question(question, relevant_sentences)
        elif any(word in question_lower for word in ['how', 'how to', 'how can']):
            return self._answer_how_question(question, relevant_sentences)
        elif any(word in question_lower for word in ['where', 'where is']):
            return self._answer_where_question(question, relevant_sentences)
        elif any(word in question_lower for word in ['when', 'when is', 'when did']):
            return self._answer_when_question(question, relevant_sentences)
        elif any(word in question_lower for word in ['why', 'why is', 'why does']):
            return self._answer_why_question(question, relevant_sentences)
        else:
            return self._answer_general_question(question, relevant_sentences)
    
    def _find_relevant_sentences(self, question: str, context: str) -> List[str]:
        """Find sentences in context that are most relevant to the question"""
        
        # Extract keywords from question
        question_keywords = self._extract_meaningful_keywords(question)
        
        # Split context into sentences
        sentences = []
        for paragraph in context.split('\n'):
            if paragraph.strip():
                # Split by periods but be careful with abbreviations
                sent_parts = re.split(r'\.(?=\s+[A-Z])', paragraph)
                sentences.extend([s.strip() for s in sent_parts if len(s.strip()) > 20])
        
        # Score sentences based on keyword overlap
        scored_sentences = []
        for sentence in sentences:
            score = self._calculate_relevance_score(question_keywords, sentence)
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # Sort by relevance and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent[0] for sent in scored_sentences[:5]]
    
    def _extract_meaningful_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text for relevance scoring"""
        # Remove common question words and stop words
        stop_words = {
            'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'can', 'could', 'should', 'would', 'will', 'shall'
        }
        
        # Extract words and filter
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_relevance_score(self, keywords: List[str], sentence: str) -> float:
        """Calculate how relevant a sentence is to the question keywords"""
        sentence_lower = sentence.lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in sentence_lower)
        
        # Calculate score (percentage of keywords found)
        if not keywords:
            return 0.0
        
        base_score = matches / len(keywords)
        
        # Bonus for exact phrase matches or multiple word proximity
        bonus = 0.0
        for i, keyword in enumerate(keywords):
            for j, other_keyword in enumerate(keywords[i+1:], i+1):
                # Check if keywords appear close to each other
                k1_pos = sentence_lower.find(keyword)
                k2_pos = sentence_lower.find(other_keyword)
                if k1_pos != -1 and k2_pos != -1 and abs(k1_pos - k2_pos) < 50:
                    bonus += 0.1
        
        return min(base_score + bonus, 1.0)
    
    def _answer_who_question(self, question: str, relevant_sentences: List[str]) -> str:
        """Answer 'who' questions using relevant sentences"""
        if not relevant_sentences:
            return "I couldn't find information about the person you're asking about."
        
        # Extract the name from the question
        question_words = question.lower().split()
        name_start_idx = -1
        for i, word in enumerate(question_words):
            if word in ['is', 'are', 'was', 'were']:
                name_start_idx = i + 1
                break
        
        if name_start_idx > 0 and name_start_idx < len(question_words):
            # Get the name from the question
            name_part = ' '.join(question_words[name_start_idx:]).strip('?').strip()
            
            # Look for the person in the relevant sentences
            for sentence in relevant_sentences:
                if name_part.lower() in sentence.lower():
                    # Try to extract clean information about this person
                    person_info = self._extract_clean_person_info(sentence, name_part)
                    if person_info:
                        return f"**{name_part.title()}** is {person_info}."
            
            # If no clean extraction, provide the most relevant sentence
            return f"Based on the content, {relevant_sentences[0].strip()}."
        
        # Fallback to general response
        return f"According to the information: {relevant_sentences[0].strip()}."
    
    def _extract_clean_person_info(self, sentence: str, name: str) -> str:
        """Extract clean information about a specific person from a sentence"""
        name_lower = name.lower()
        sentence_lower = sentence.lower()
        
        # Find where the name appears
        name_pos = sentence_lower.find(name_lower)
        if name_pos == -1:
            return ""
        
        # Get text after the name
        name_end = name_pos + len(name)
        after_name = sentence[name_end:].strip()
        
        # Remove common separators and clean up
        after_name = re.sub(r'^[-–|,:\s]+', '', after_name)
        
        # Look for role/title information
        # Split at common boundaries that indicate end of this person's info
        role_end_patterns = [
            r'\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:President|CEO|Director|Manager|Head|Chief)',  # Next person's name + title
            r'\s+[A-Z][a-z]+\s+[A-Z]\s+[A-Z][a-z]+',  # Next person with middle initial
        ]
        
        role_info = after_name
        for pattern in role_end_patterns:
            match = re.search(pattern, role_info)
            if match:
                role_info = role_info[:match.start()].strip()
                break
        
        # Clean up the role information
        # Look for common title patterns
        title_match = re.match(r'([^,\n]*(?:President|CEO|CTO|CFO|Director|Manager|Head|Chief|Officer|Chairman|Founder|Executive)[^,\n]*)', role_info, re.IGNORECASE)
        if title_match:
            clean_title = title_match.group(1).strip()
            # Handle compound titles like "President – Group CEO"
            clean_title = clean_title.replace('–', '-').replace('—', '-')
            return clean_title
        
        # Fallback: take first meaningful part
        parts = re.split(r'[,\n]', role_info)
        if parts:
            first_part = parts[0].strip()
            if len(first_part) > 3:
                return first_part
        
        return ""
    
    def _answer_what_question(self, question: str, relevant_sentences: List[str]) -> str:
        """Answer 'what' questions using relevant sentences"""
        if not relevant_sentences:
            return "I couldn't find information to answer your question."
        
        # Take the most relevant sentence and clean it up
        best_sentence = relevant_sentences[0].strip()
        
        # If multiple sentences are very relevant, combine them intelligently
        if len(relevant_sentences) > 1:
            combined_info = '. '.join([s.strip() for s in relevant_sentences[:2]])
            if len(combined_info) < 200:  # Keep it concise
                return combined_info + "."
        
        return best_sentence + ("." if not best_sentence.endswith('.') else "")
    
    def _answer_how_question(self, question: str, relevant_sentences: List[str]) -> str:
        """Answer 'how' questions using relevant sentences"""
        if not relevant_sentences:
            return "I couldn't find information about the process or method you're asking about."
        
        # For "how" questions, often multiple sentences provide better context
        if len(relevant_sentences) > 1:
            combined_info = '. '.join([s.strip() for s in relevant_sentences[:3]])
            return combined_info + "."
        
        return relevant_sentences[0].strip() + ("." if not relevant_sentences[0].strip().endswith('.') else "")
    
    def _answer_where_question(self, question: str, relevant_sentences: List[str]) -> str:
        """Answer 'where' questions using relevant sentences"""
        if not relevant_sentences:
            return "I couldn't find location information to answer your question."
        
        best_sentence = relevant_sentences[0].strip()
        return best_sentence + ("." if not best_sentence.endswith('.') else "")
    
    def _answer_when_question(self, question: str, relevant_sentences: List[str]) -> str:
        """Answer 'when' questions using relevant sentences"""
        if not relevant_sentences:
            return "I couldn't find timing or date information to answer your question."
        
        best_sentence = relevant_sentences[0].strip()
        return best_sentence + ("." if not best_sentence.endswith('.') else "")
    
    def _answer_why_question(self, question: str, relevant_sentences: List[str]) -> str:
        """Answer 'why' questions using relevant sentences"""
        if not relevant_sentences:
            return "I couldn't find explanatory information to answer your 'why' question."
        
        # "Why" questions often benefit from multiple explanatory sentences
        if len(relevant_sentences) > 1:
            combined_info = '. '.join([s.strip() for s in relevant_sentences[:2]])
            return combined_info + "."
        
        return relevant_sentences[0].strip() + ("." if not relevant_sentences[0].strip().endswith('.') else "")
    
    def _answer_general_question(self, question: str, relevant_sentences: List[str]) -> str:
        """Answer general questions using relevant sentences"""
        if not relevant_sentences:
            return "I couldn't find relevant information to answer your question."
        
        # For general questions, provide the most relevant information
        if len(relevant_sentences) == 1:
            return relevant_sentences[0].strip() + ("." if not relevant_sentences[0].strip().endswith('.') else "")
        else:
            # Combine up to 2 most relevant sentences
            combined_info = '. '.join([s.strip() for s in relevant_sentences[:2]])
            return combined_info + "."
    
    def _generate_clean_fallback_response(self, context: str, question: str) -> str:
        """Generate a clean fallback response without showing raw content"""
        question_lower = question.lower()
        
        # Try to provide a helpful response based on question type
        if any(word in question_lower for word in ['who', 'person', 'people']):
            return "I found information about people in the content, but I couldn't extract specific details to answer your question. Try asking about a specific person's name or role."
        
        elif any(word in question_lower for word in ['what', 'describe', 'about']):
            return "I found relevant information but couldn't provide a specific answer to your question. Try rephrasing your question or being more specific about what you're looking for."
        
        elif any(word in question_lower for word in ['plan', 'model', 'subscription', 'pricing']):
            return "I found information about plans or services, but couldn't extract the specific details you're looking for. Try asking about specific plan names or features."
        
        else:
            return "I found relevant content but couldn't extract a specific answer to your question. Please try rephrasing your question or asking about specific details."
    
    def _generate_structured_fallback_response(self, context: str, question: str) -> str:
        """Generate a structured fallback response"""
        meaningful_sentences = self._extract_meaningful_sentences(context, 4)
        
        if meaningful_sentences:
            return f"Based on the available information:\n\n" + "\n".join([f"• {sentence}." for sentence in meaningful_sentences])
        
        return f"I found relevant content but couldn't extract specific information to answer your question. The content may be structured in a way that requires rephrasing your question or asking about specific aspects."
    
    def _extract_names_and_roles(self, context: str) -> List[str]:
        """Extract names and roles from context"""
        results = []
        lines = context.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('**Source'):
                continue
                
            # Look for patterns like "Name - Role" or "Name, Role"
            role_patterns = [
                r'([A-Z][a-z]+ [A-Z][a-z]+)\s*[-–]\s*([A-Z][^,\n]+)',
                r'([A-Z][a-z]+ [A-Z][a-z]+),\s*([A-Z][^,\n]+)',
                r'([A-Z][a-z]+ [A-Z][a-z]+)\s*\|\s*([A-Z][^,\n]+)',
                r'([A-Z][a-z]+ [A-Z][a-z]+)\s+([A-Z][A-Z][A-Z])',  # Name CEO
            ]
            
            for pattern in role_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    name, role = match
                    if len(name.split()) >= 2:  # At least first and last name
                        results.append(f"{name} - {role.strip()}")
        
        return results[:5]  # Return top 5 matches
    
    def _extract_plans_and_models(self, context: str) -> List[str]:
        """Extract plans and models from context"""
        results = []
        lines = context.split('\n')
        
        plan_keywords = ['plan', 'plans', 'model', 'models', 'subscription', 'tier', 'pro', 'plus', 'premium', 'basic', 'free', 'enterprise']
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('**Source'):
                continue
            
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in plan_keywords):
                # Clean up the line and add it
                clean_line = re.sub(r'[•\-\*]\s*', '', line).strip()
                if len(clean_line) > 10:
                    results.append(clean_line)
        
        return results[:8]  # Return top 8 matches
    
    def _extract_contact_info(self, context: str) -> List[str]:
        """Extract contact information from context"""
        results = []
        lines = context.split('\n')
        
        contact_patterns = [
            r'email:?\s*([^\s,\n]+@[^\s,\n]+)',
            r'phone:?\s*([\+\d\s\-\(\)]{10,})',
            r'address:?\s*([^\n]+)',
            r'([^\s,\n]+@[^\s,\n]+)',  # General email pattern
        ]
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('**Source'):
                continue
            
            for pattern in contact_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    if match.strip():
                        results.append(f"{match.strip()}")
        
        return results[:5]  # Return top 5 matches
    
    def _generate_no_results_response(self, question: str, websites: List[Dict]) -> str:
        """Generate professional response when no relevant chunks are found"""
        website_names = [w.get('title', 'Unknown') for w in websites[:3]]
        
        response = f"""I couldn't find specific information to answer your question in the current knowledge base.

**Available Sources:** {', '.join(website_names)}{"..." if len(websites) > 3 else ""}

**Suggestions:**
• Try rephrasing your question with different keywords
• Ask about specific people, roles, or company information
• Check if the information was captured during website analysis
• Consider asking about products, services, or contact information

**Example Questions:**
• "Who is the CEO of [company name]?"
• "What services does [company] provide?"
• "What are the available subscription plans?"
"""
        return response
    
    def _display_sources(self, relevant_chunks: List[RetrievalResult]):
        """Display sources used for the response with professional styling"""
        if not relevant_chunks:
            return
        
        st.markdown("""
        <div class="rag-sources">
            <h4 style="
                margin: 0 0 0.75rem 0;
                font-size: 0.875rem;
                font-weight: 600;
                color: #2c3e50;
            ">Reference Sources</h4>
        """, unsafe_allow_html=True)
        
        for i, result in enumerate(relevant_chunks):
            chunk = result.chunk
            relevance_color = "#28a745" if result.relevance_score > 0.7 else "#ffc107" if result.relevance_score > 0.4 else "#6c757d"
            
            st.markdown(f"""
            <div class="rag-source-item">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h5 style="
                        margin: 0;
                        font-size: 0.8125rem;
                        font-weight: 600;
                        color: #2c3e50;
                    ">{chunk.website_title}</h5>
                    <span style="
                        background: {relevance_color};
                        color: white;
                        padding: 0.125rem 0.5rem;
                        border-radius: 12px;
                        font-size: 0.6875rem;
                        font-weight: 500;
                    ">
                        {result.relevance_score:.1%}
                    </span>
                </div>
                <p style="
                    margin: 0 0 0.25rem 0;
                    font-size: 0.75rem;
                    color: #6c757d;
                    line-height: 1.4;
                ">{chunk.content[:150]}{'...' if len(chunk.content) > 150 else ''}</p>
                <p style="
                    margin: 0;
                    font-size: 0.6875rem;
                    color: #6c757d;
                ">{chunk.website_url}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _get_available_websites(self) -> List[Dict]:
        """Get all available websites from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT id, title, url, summary, created_at, chunk_count 
                FROM websites 
                ORDER BY updated_at DESC
                ''')
                
                websites = []
                for row in cursor.fetchall():
                    websites.append({
                        'id': row[0],
                        'title': row[1],
                        'url': row[2],
                        'summary': row[3],
                        'created_at': row[4],
                        'chunk_count': row[5] or 0
                    })
                
                return websites
                
        except Exception as e:
            st.error(f"Failed to load websites: {e}")
            return []
    
    def _load_from_session_state(self) -> bool:
        """Load websites from session state analysis results"""
        if 'analysis_results' not in st.session_state:
            return False
        
        try:
            count = 0
            for result in st.session_state.analysis_results:
                if hasattr(result, 'url') and hasattr(result, 'scraped_content'):
                    success = self.add_website_from_analysis(result)
                    if success:
                        count += 1
            
            if count > 0:
                st.success(f"✅ Loaded {count} websites into knowledge base")
                return True
                
        except Exception as e:
            st.warning(f"Could not load from analysis results: {e}")
        
        return False
    
    def add_website_from_analysis(self, analysis_result) -> bool:
        """Add a website to the RAG knowledge base from analysis result with ChromaDB storage"""
        try:
            # Extract information from analysis result
            url = analysis_result.url
            title = getattr(analysis_result.scraped_content, 'title', url)
            summary = getattr(analysis_result, 'executive_summary', '')
            
            # Get content
            content = ""
            if hasattr(analysis_result, 'scraped_content'):
                scraped = analysis_result.scraped_content
                if hasattr(scraped, 'main_content'):
                    content = scraped.main_content or ""
                elif isinstance(scraped, dict):
                    content = scraped.get('main_content', '')
            
            if not content:
                return False
            
            # Check if already exists
            website_id = hashlib.md5(url.encode()).hexdigest()
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if exists
                cursor.execute('SELECT content_hash FROM websites WHERE id = ?', (website_id,))
                existing = cursor.fetchone()
                
                if existing and existing[0] == content_hash:
                    return False  # Already up to date
                
                # Insert or update website metadata in SQLite
                cursor.execute('''
                INSERT OR REPLACE INTO websites 
                (id, title, url, summary, content_hash, updated_at) 
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (website_id, title, url, summary, content_hash, datetime.now()))
                
                # Clear existing chunks from SQLite
                cursor.execute('DELETE FROM content_chunks WHERE website_id = ?', (website_id,))
                
                conn.commit()
            
            # Chunk the content
            chunks = self._chunk_content(content)
            
            if not chunks:
                return False
            
            # Store chunks in ChromaDB if available
            if self.chroma_collection and self.embedding_model:
                try:
                    # Remove existing chunks from ChromaDB
                    try:
                        # Get existing chunk IDs for this website
                        existing_results = self.chroma_collection.get(
                            where={"website_id": website_id}
                        )
                        if existing_results['ids']:
                            self.chroma_collection.delete(ids=existing_results['ids'])
                    except Exception as e:
                        print(f"Could not delete existing ChromaDB entries: {e}")
                    
                    # Prepare data for ChromaDB batch insert
                    chunk_ids = []
                    documents = []
                    metadatas = []
                    
                    for i, chunk_text in enumerate(chunks):
                        chunk_id = f"{website_id}_{i}"
                        chunk_ids.append(chunk_id)
                        documents.append(chunk_text)
                        metadatas.append({
                            "website_id": website_id,
                            "website_title": title,
                            "website_url": url,
                            "chunk_type": "paragraph",
                            "position": i,
                            "created_at": datetime.now().isoformat()
                        })
                    
                    # Generate embeddings for all chunks at once (more efficient)
                    embeddings = self.embedding_model.encode(documents)
                    
                    # Batch insert into ChromaDB
                    self.chroma_collection.add(
                        ids=chunk_ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings.tolist()
                    )
                    
                    # Update chunk count
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('UPDATE websites SET chunk_count = ? WHERE id = ?', (len(chunks), website_id))
                        conn.commit()
                    
                    print(f"✅ Added {len(chunks)} chunks to ChromaDB for {title}")
                    return True
                    
                except Exception as e:
                    print(f"ChromaDB storage failed: {e}")
                    # Fall back to SQLite storage
                    return self._add_chunks_to_sqlite(website_id, chunks)
            else:
                # Fall back to SQLite storage
                return self._add_chunks_to_sqlite(website_id, chunks)
                
        except Exception as e:
            st.error(f"Failed to add website to RAG knowledge base: {e}")
            return False
    
    def _add_chunks_to_sqlite(self, website_id: str, chunks: List[str]) -> bool:
        """Fallback method to store chunks in SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for i, chunk_text in enumerate(chunks):
                    chunk_id = f"{website_id}_{i}"
                    
                    # Generate embedding if model available
                    embedding_blob = None
                    if self.embedding_model:
                        try:
                            embedding = self.embedding_model.encode([chunk_text])[0]
                            embedding_blob = embedding.astype(np.float32).tobytes()
                        except Exception as e:
                            print(f"Could not generate embedding for chunk {i}: {e}")
                    
                    # Store chunk in SQLite
                    cursor.execute('''
                    INSERT INTO content_chunks 
                    (id, website_id, content, chunk_type, position, embedding) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (chunk_id, website_id, chunk_text, 'paragraph', i, embedding_blob))
                
                # Update chunk count
                cursor.execute('UPDATE websites SET chunk_count = ? WHERE id = ?', (len(chunks), website_id))
                conn.commit()
                
                print(f"✅ Stored {len(chunks)} chunks in SQLite (fallback)")
                return True
                
        except Exception as e:
            print(f"SQLite fallback storage failed: {e}")
            return False
    
    def _chunk_content(self, content: str) -> List[str]:
        """Split content into overlapping chunks"""
        if not content:
            return []
        
        chunks = []
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk
                if len(paragraph) > self.chunk_size:
                    # Split long paragraphs
                    sentences = paragraph.split('. ')
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) > self.chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                        else:
                            temp_chunk += (". " if temp_chunk else "") + sentence
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = paragraph
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _remove_website(self, website_id: str):
        """Remove a website and all its chunks"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM websites WHERE id = ?', (website_id,))
                conn.commit()
                st.success("✅ Website removed from knowledge base")
        except Exception as e:
            st.error(f"Failed to remove website: {e}")
    
    def _clear_all_data(self):
        """Clear all data from the knowledge base"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM content_chunks')
                cursor.execute('DELETE FROM websites')
                conn.commit()
                st.success("✅ Knowledge base cleared")
        except Exception as e:
            st.error(f"Failed to clear knowledge base: {e}")
