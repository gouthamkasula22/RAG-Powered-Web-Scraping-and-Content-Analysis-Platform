# Docker Setup for Web Content Analyzer

This guide explains how to run the Web Content Analyzer using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Quick Start

1. Clone the repository
2. Navigate to the project directory
3. Run the application with Docker Compose:

```bash
docker-compose up -d
```

4. Access the applications:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs

## Configuration

Environment variables can be configured in the `.env` file. Key configurations:

- API settings (host, port, workers)
- Streamlit settings
- Database paths
- Security settings
- API keys for LLM services

## Services

The application consists of two main services:

1. **Backend API** (FastAPI)
   - Port: 8000
   - Provides RESTful API for web content analysis
   - Handles scraping, content processing, and analysis

2. **Frontend** (Streamlit)
   - Port: 8501
   - User interface for interacting with the system
   - Visualizes analysis results and manages knowledge repository

## Data Persistence

Data is stored in SQLite databases located in the `/app/data` directory, which is mapped to `./data` on your host system.

## Development Mode

For development with hot-reloading:

```bash
docker-compose up
```

This will mount the local directories into the containers, allowing code changes to be reflected immediately.

## Stopping the Application

```bash
docker-compose down
```

## Rebuilding After Changes to Dependencies

If you update requirements.txt or make other changes that require a rebuild:

```bash
docker-compose build --no-cache
docker-compose up -d
```

## Troubleshooting

- **Container not starting**: Check logs with `docker-compose logs -f [service_name]`
- **API connection issues**: Ensure the BACKEND_URL in .env points to the correct service
- **Embedding model issues**: May require additional memory allocation to Docker
