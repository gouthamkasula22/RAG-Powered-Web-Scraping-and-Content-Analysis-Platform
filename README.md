# Web Content Analysis Platform

An intelligent web content analysis platform that provides comprehensive insights into website quality, SEO performance, user experience, and competitive intelligence through advanced AI-powered analysis.

## Overview

The Web Content Analysis Platform combines sophisticated web scraping capabilities with cutting-edge AI language models to deliver professional-grade website analysis reports. Built with enterprise-level security and scalability in mind, the platform serves developers, marketers, and businesses who need deep insights into web content performance.

### Core Capabilities

**Intelligent Content Analysis**
- Comprehensive website content evaluation using Google Gemini and Anthropic Claude
- SEO optimization analysis with actionable recommendations  
- User experience assessment and improvement suggestions
- Content quality scoring across multiple dimensions

**RAG Knowledge Repository**
- Advanced question-answering system for analyzed websites
- Vector-based semantic search using sentence transformers
- Website-specific queries with source attribution
- Intelligent content retrieval and summarization

**Professional Analytics**
- Detailed performance metrics and scoring
- Bulk analysis for multiple websites simultaneously
- Comparative analysis between competing sites
- Export capabilities in CSV, JSON, and PDF formats

**Enterprise Security**
- Server-side request forgery (SSRF) protection
- Comprehensive input sanitization and validation
- API rate limiting and abuse prevention
- JWT-based authentication for secure access

## Technical Architecture

The platform follows a clean, layered architecture that ensures maintainability, testability, and scalability:

```
┌─────────────────────────────────────────────────────────┐
│                 Presentation Layer                       │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │   Streamlit UI  │    │      FastAPI REST API      │ │
│  │                 │    │                             │ │
│  │ • Interactive   │    │ • RESTful endpoints         │ │
│  │   Dashboard     │    │ • Request validation        │ │
│  │ • Bulk Analysis │    │ • Error handling           │ │
│  │ • RAG Interface │    │ • Documentation           │ │
│  └─────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────┐
│                Application Layer                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Business Services                       │ │
│  │                                                     │ │
│  │ • Content Analysis Service                          │ │
│  │ • RAG Knowledge Service                             │ │
│  │ • Report Generation Service                         │ │
│  │ • Security Validation Service                       │ │
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
│  │ • Content     │  │ • Fallback   │  │ • Rate       │ │
│  │   Extraction  │  │   Handling   │  │   Limiting   │ │
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

The platform is built following established software engineering principles:

- **Clean Architecture**: Separation of concerns with clear layer boundaries
- **SOLID Principles**: Maintainable, extensible, and testable code structure  
- **Domain-Driven Design**: Business logic encapsulated in the domain layer
- **Dependency Inversion**: Abstract interfaces enable flexible implementations
- **Security by Design**: Security considerations integrated at every layer

## Quick Start Guide

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Internet connection for API services

### Installation

1. **Clone and Setup**
   ```bash
   git clone https://github.com/yourusername/web-content-analysis.git
   cd web-content-analysis
   
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file with your API credentials:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ANTHROPIC_API_KEY=your_claude_api_key_here
   ```

3. **Launch Application**
   
   **Option A: Streamlit Interface (Recommended)**
   ```bash
   streamlit run frontend/streamlit/app.py
   ```
   Access at: http://localhost:8501

   **Option B: API Server**
   ```bash
   uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
   ```
   API documentation at: http://localhost:8000/api/docs

### Basic Usage

1. **Single Website Analysis**
   - Enter a website URL in the Streamlit interface
   - Select analysis type (Basic, Comprehensive, or Detailed)
   - Review the generated insights and recommendations

2. **Bulk Analysis**
   - Upload a CSV file with website URLs or enter multiple URLs
   - Configure parallel processing settings
   - Download results in your preferred format

3. **RAG Knowledge Queries**
   - Add analyzed websites to the knowledge repository
   - Ask questions about website content
   - Receive AI-generated answers with source attribution

## Development and Testing

### Running Tests

The platform includes a comprehensive test suite with 145 passing tests:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests  
pytest tests/security/      # Security tests
pytest tests/performance/   # Performance tests
```

### Code Quality

```bash
# Code formatting
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Development Environment

The project supports development with:
- Hot reloading for both frontend and backend
- Comprehensive logging and debugging
- Docker containerization for consistent environments
- Pre-commit hooks for code quality

## Configuration

### Environment Variables

```env
# LLM Service Configuration
GOOGLE_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_claude_key
GEMINI_MODEL=gemini-1.5-flash
CLAUDE_MODEL=claude-3-haiku-20240307

# Application Settings
MAX_CONCURRENT_ANALYSES=5
DEFAULT_TIMEOUT_SECONDS=30
RATE_LIMIT_PER_MINUTE=100

# Database Configuration
DATABASE_URL=sqlite:///./data/app.db
RAG_DATABASE_URL=sqlite:///./data/rag_knowledge_repository.db

# Security Settings
ALLOWED_HOSTS=localhost,127.0.0.1
SECRET_KEY=your-secret-key-here
```

### Feature Configuration

Features can be enabled or disabled through configuration:

```python
FEATURES = {
    "enable_rag_system": True,
    "enable_bulk_analysis": True,
    "enable_export_formats": ["csv", "json", "pdf"],
    "enable_comparative_analysis": True,
    "max_websites_per_bulk": 50
}
```

## Technology Stack

### Backend Technologies
- **FastAPI**: Modern, fast web framework for APIs
- **SQLite**: Lightweight, serverless database
- **Pydantic**: Data validation and serialization
- **AsyncIO**: Asynchronous programming support

### AI and Machine Learning
- **Google Gemini**: Advanced language model for content analysis
- **Anthropic Claude**: Alternative LLM provider for redundancy
- **Sentence Transformers**: Vector embeddings for semantic search
- **BeautifulSoup**: Robust HTML parsing and content extraction

### Frontend and User Interface
- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced data visualization
- **Pandas**: Data manipulation and analysis

### DevOps and Quality Assurance
- **Docker**: Containerization for consistent deployments
- **Pytest**: Comprehensive testing framework
- **GitHub Actions**: Continuous integration and deployment
- **Pre-commit**: Automated code quality checks

## Security Features

The platform implements multiple layers of security:

- **SSRF Prevention**: Comprehensive server-side request forgery protection
- **Input Validation**: Rigorous sanitization of all user inputs
- **Rate Limiting**: Protection against API abuse and excessive usage
- **Domain Filtering**: Configurable blacklist for restricted websites
- **Error Handling**: Secure error messages that don't leak sensitive information

## Performance and Scalability

- **Concurrent Processing**: Parallel analysis of multiple websites
- **Efficient Memory Management**: Optimized for large-scale operations
- **Caching Strategies**: Intelligent caching of analysis results
- **Resource Monitoring**: Built-in performance metrics and monitoring

## Contributing

We welcome contributions from the community. Please read our contributing guidelines and follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with appropriate tests
4. Ensure all tests pass (`pytest`)
5. Submit a pull request with a detailed description

## Support and Documentation

- **API Documentation**: Complete REST API reference available at `/api/docs`
- **User Guide**: Step-by-step instructions for all features
- **Developer Guide**: Technical implementation details and architecture
- **Security Guide**: Security features and best practices

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Professional web content analysis powered by artificial intelligence**
