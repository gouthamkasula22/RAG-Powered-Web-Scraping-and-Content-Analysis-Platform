# Web Content Analyzer

A comprehensive web content analysis tool that scrapes websites and generates detailed analysis reports using LLM integration.

## Project Structure

```
web-content-analyzer/
├── src/
│   ├── domain/                 # Business logic & entities (Domain Layer)
│   │   ├── __init__.py
│   │   ├── models.py          # Core business models
│   │   ├── enums.py           # Business enumerations
│   │   └── exceptions.py      # Domain-specific exceptions
│   ├── application/           # Application services (Application Layer)
│   │   ├── __init__.py
│   │   ├── interfaces/        # Abstract interfaces (SOLID - DIP)
│   │   │   ├── __init__.py
│   │   │   ├── scraping.py    # Scraping interfaces
│   │   │   ├── validation.py  # Validation interfaces
│   │   │   └── extraction.py  # Content extraction interfaces
│   │   ├── services/          # Application services
│   │   │   ├── __init__.py
│   │   │   ├── scraping_service.py
│   │   │   ├── analysis_service.py
│   │   │   └── validation_service.py
│   │   └── dtos/              # Data Transfer Objects
│   │       ├── __init__.py
│   │       ├── requests.py
│   │       └── responses.py
│   ├── infrastructure/        # External concerns (Infrastructure Layer)
│   │   ├── __init__.py
│   │   ├── web/               # Web scraping implementations
│   │   │   ├── __init__.py
│   │   │   ├── scrapers/      # Different scraping strategies
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_scraper.py
│   │   │   │   ├── general_scraper.py
│   │   │   │   └── news_scraper.py
│   │   │   ├── extractors/    # Content extraction strategies
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_extractor.py
│   │   │   │   ├── article_extractor.py
│   │   │   │   └── homepage_extractor.py
│   │   │   └── proxies/       # Proxy pattern implementations
│   │   │       ├── __init__.py
│   │   │       └── secure_scraping_proxy.py
│   │   ├── security/          # Security implementations
│   │   │   ├── __init__.py
│   │   │   ├── url_validator.py
│   │   │   └── rate_limiter.py
│   │   ├── http/              # HTTP client implementations
│   │   │   ├── __init__.py
│   │   │   └── http_client.py
│   │   └── config/            # Configuration management
│   │       ├── __init__.py
│   │       └── settings.py
│   └── presentation/          # UI Layer (Presentation Layer)
│       ├── __init__.py
│       ├── api/               # FastAPI controllers
│       │   ├── __init__.py
│       │   ├── main.py        # FastAPI app setup
│       │   ├── routes/        # API route handlers
│       │   │   ├── __init__.py
│       │   │   ├── analysis.py
│       │   │   └── health.py
│       │   └── middleware/    # API middleware
│       │       ├── __init__.py
│       │       ├── cors.py
│       │       └── error_handler.py
│       └── ui/                # Streamlit interface
│           ├── __init__.py
│           ├── main.py        # Streamlit app
│           ├── components/    # UI components
│           │   ├── __init__.py
│           │   ├── url_input.py
│           │   ├── analysis_display.py
│           │   └── progress_tracker.py
│           └── utils/         # UI utilities
│               ├── __init__.py
│               └── formatting.py
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── unit/                  # Unit tests
│   │   ├── __init__.py
│   │   ├── test_domain/
│   │   ├── test_application/
│   │   └── test_infrastructure/
│   ├── integration/           # Integration tests
│   │   ├── __init__.py
│   │   └── test_scraping_flow.py
│   ├── fixtures/              # Test fixtures
│   │   ├── __init__.py
│   │   └── sample_html.py
│   └── conftest.py           # Pytest configuration
├── docs/                      # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   └── deployment.md
├── config/                    # Configuration files
│   ├── development.env
│   ├── production.env
│   └── test.env
├── scripts/                   # Utility scripts
│   ├── setup.py
│   ├── run_tests.py
│   └── deploy.py
├── requirements/              # Requirements management
│   ├── base.txt              # Base requirements
│   ├── development.txt       # Development requirements
│   ├── production.txt        # Production requirements
│   └── testing.txt           # Testing requirements
├── .env.example              # Environment variables example
├── .gitignore               # Git ignore rules
├── pyproject.toml           # Python project configuration
├── pytest.ini              # Pytest configuration
├── docker-compose.yml       # Docker configuration
├── Dockerfile              # Docker image
└── requirements.txt        # Main requirements file
```

## Architecture Principles

### N-Layer Architecture
- **Domain Layer**: Core business logic, entities, and rules
- **Application Layer**: Use cases, services, and application logic
- **Infrastructure Layer**: External dependencies (web, database, HTTP)
- **Presentation Layer**: User interfaces (API, Web UI)

### SOLID Principles
- **S**ingle Responsibility: Each class has one reason to change
- **O**pen/Closed: Open for extension, closed for modification
- **L**iskov Substitution: Subtypes must be substitutable for base types
- **I**nterface Segregation: Many client-specific interfaces
- **D**ependency Inversion: Depend on abstractions, not concretions

### Design Patterns
- **Proxy Pattern**: Secure scraping with validation
- **Strategy Pattern**: Different extraction strategies
- **Factory Pattern**: Create appropriate scrapers
- **Repository Pattern**: Abstract data access

## Getting Started

1. **Setup Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run Development Server**
   ```bash
   # FastAPI Backend
   uvicorn src.presentation.api.main:app --reload --port 8000
   
   # Streamlit Frontend
   streamlit run src/presentation/ui/main.py --server.port 8501
   ```

4. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

## Milestone 1 Goals

- [x] Project structure with N-layer architecture
- [ ] Domain models and business entities
- [ ] Security layer with URL validation
- [ ] Web scraping foundation with proxy pattern
- [ ] Basic Streamlit interface
- [ ] Comprehensive error handling
- [ ] Unit tests for core components

## Security Features

- SSRF prevention with URL validation
- Private IP range blocking
- Input sanitization and validation
- Rate limiting and timeout handling
- Secure content extraction

## Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **Frontend**: Streamlit
- **Web Scraping**: requests, BeautifulSoup4
- **Validation**: Pydantic, validators
- **Testing**: pytest, pytest-asyncio
- **Security**: Custom URL validation, rate limiting
