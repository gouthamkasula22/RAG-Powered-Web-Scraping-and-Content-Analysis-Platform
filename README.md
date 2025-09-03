# Web Content Analysis Tool

A comprehensive AI-powered web content analyzer that extracts, processes, and analyzes web content with advanced performance optimizations and intelligent image processing capabilities.

## Overview

This tool provides deep insights into web content by combining intelligent web scraping, AI-powered analysis, and advanced image processing. Built with modern async architecture, it delivers high-performance content analysis while maintaining accuracy and reliability.

## Key Features

### Content Analysis
- **Intelligent Web Scraping**: Extracts structured content from any website
- **AI-Powered Insights**: Leverages OpenAI GPT-4, Groq, and Anthropic Claude models
- **Quality Assessment**: Advanced content quality scoring and metrics
- **Executive Summaries**: Generates concise, actionable summaries

### Performance Optimizations
- **Parallel Processing**: Optimized async operations for 70-95% faster analysis
- **Smart Caching**: Intelligent caching system with TTL-based invalidation
- **Lazy Loading**: On-demand resource loading to minimize initial load times
- **Batch Operations**: Efficient bulk analysis capabilities

### Image Processing
- **Intelligent Extraction**: Context-aware image discovery and classification
- **Parallel Downloads**: Optimized concurrent image downloading with fallback mechanisms
- **Lazy Thumbnails**: On-demand thumbnail generation with quality optimization
- **Visual Context**: Maintains relationship between images and content

### User Interface
- **Modern Interface**: Clean, responsive Streamlit-based UI
- **Analysis History**: Persistent storage and retrieval of past analyses
- **Interactive Components**: Real-time feedback and progress indicators
- **Bulk Processing**: Support for multiple URL analysis

## Architecture

### Backend (FastAPI)
- **RESTful API**: Clean, well-documented endpoints
- **Async Architecture**: High-performance concurrent processing
- **Database Layer**: SQLite for development, easily scalable to PostgreSQL
- **Modular Design**: Clean separation of concerns with domain-driven architecture

### Frontend (Streamlit)
- **Component-Based**: Reusable UI components for consistency
- **Responsive Design**: Adaptive layouts for different screen sizes
- **Real-time Updates**: Live progress tracking and status updates
- **Error Handling**: Graceful degradation with user-friendly error messages

### Data Processing
- **Vector Database**: ChromaDB integration for semantic search capabilities
- **Content Extraction**: Advanced HTML parsing with noise reduction
- **Image Processing**: PIL-based image manipulation with format support
- **Caching Layer**: Multi-level caching for optimal performance

## Quick Start

### Prerequisites
- Python 3.10 or higher
- 4GB RAM minimum (8GB recommended for bulk operations)
- Internet connection for AI model access

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gouthamkasula22/Web-Content-Analysis.git
   cd Web-Content-Analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Initialize database**
   ```bash
   python -c "from backend.src.infrastructure.database.setup import setup_database; setup_database()"
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   cd backend
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Launch the frontend interface**
   ```bash
   cd frontend/streamlit
   streamlit run app.py
   ```

3. **Access the application**
   - Frontend UI: http://localhost:8501
   - API Documentation: http://localhost:8000/api/docs

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# AI Provider API Keys
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=true

# Database
DATABASE_URL=sqlite:///./data/analysis_history.db

# Image Processing
MAX_IMAGE_SIZE=10485760  # 10MB
THUMBNAIL_SIZE=200
IMAGE_QUALITY=85

# Performance
CACHE_TTL=300  # 5 minutes
MAX_CONCURRENT_REQUESTS=5
```

### Image Processing Options

The application provides granular control over image processing:

- **Extract Images**: Toggle image discovery and extraction
- **Download Images**: Control whether images are downloaded locally
- **Generate Thumbnails**: Enable/disable thumbnail creation
- **Max Images**: Limit the number of images processed per analysis

## API Reference

### Core Endpoints

#### Analyze Content
```http
POST /api/analyze
Content-Type: application/json

{
  "url": "https://example.com",
  "extract_images": true,
  "download_images": true,
  "max_images": 10,
  "generate_thumbnails": true
}
```

#### Retrieve Analysis
```http
GET /api/analysis/{analysis_id}
```

#### Get Images
```http
GET /api/images/content/{content_id}
```

#### Health Check
```http
GET /api/health
```

## Performance Characteristics

### Benchmarks
- **Text Analysis**: 2-5 seconds for standard web pages
- **With Images**: 10-15 seconds for image-heavy content
- **Cache Hits**: 95% performance improvement (sub-second responses)
- **Parallel Processing**: 70% faster than sequential operations

### Optimization Features
- **Lazy Loading**: Resources loaded only when needed
- **Smart Caching**: Intelligent cache invalidation and refresh
- **Batch Processing**: Efficient handling of multiple URLs
- **Resource Management**: Automatic cleanup of temporary files

## Project Structure

```
Web Content Analysis/
├── backend/                 # FastAPI backend application
│   ├── api/                # API endpoints and routing
│   └── src/                # Core business logic
│       ├── application/    # Service layer
│       ├── domain/         # Business models
│       └── infrastructure/ # External integrations
├── frontend/               # Streamlit frontend application
│   └── streamlit/          # UI components and pages
├── data/                   # Application data (git-ignored)
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── tests/                  # Test suites
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 backend/ frontend/
black backend/ frontend/

# Type checking
mypy backend/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, bug reports, or feature requests, please open an issue on GitHub.

## Acknowledgments

- Built with FastAPI, Streamlit, and modern Python async architecture
- AI capabilities powered by OpenAI, Groq, and Anthropic
- Image processing using Pillow and advanced caching strategies
- Vector storage and retrieval using ChromaDB

---
