# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Setup Environment
```bash
# Run the setup script
python scripts/setup.py

# OR manually:
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
cp .env.example .env
```

### 2. Start Development Servers

**Terminal 1 - Backend (FastAPI):**
```bash
venv\Scripts\activate
uvicorn src.presentation.api.main:app --reload --port 8000
```

**Terminal 2 - Frontend (Streamlit):**
```bash
venv\Scripts\activate
streamlit run src/presentation/ui/main.py --server.port 8501
```

### 3. Access Applications
- **Streamlit UI**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 4. Run Tests
```bash
pytest tests/ -v
```

## 📁 Project Structure Overview

```
src/
├── domain/          # Business logic & entities
├── application/     # Use cases & interfaces
├── infrastructure/ # External dependencies
└── presentation/   # UI & API layers
```

## 🔧 Development Commands

```bash
# Code formatting
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🛡️ Security Features

- SSRF prevention with URL validation
- Private IP range blocking
- Input sanitization
- Rate limiting
- Timeout handling

## 📝 Configuration

Edit `.env` file for:
- API ports and hosts
- Scraping timeouts and limits
- Security settings
- Logging configuration

## 🎯 Milestone 1 Tasks

- [x] Project structure setup
- [x] Domain models implementation
- [x] Domain services (classification, quality, URL analysis)
- [x] Comprehensive error handling with structured exceptions
- [x] Logging integration throughout domain layer
- [x] Unit tests for domain layer
- [ ] Security layer with URL validation
- [ ] Web scraping foundation
- [ ] Basic Streamlit interface
- [ ] Error handling implementation
- [ ] Integration tests

## 🔗 Useful Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
