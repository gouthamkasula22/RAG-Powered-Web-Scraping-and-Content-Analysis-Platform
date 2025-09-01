#!/bin/bash

# Install ChromaDB if not present
if ! python -c "import chromadb" 2>/dev/null; then
    echo "Installing ChromaDB..."
    pip install --no-cache-dir chromadb>=0.4.15
    echo "ChromaDB installed successfully!"
else
    echo "ChromaDB already installed"
fi

# Original entrypoint logic
if [ "$SERVICE" = "api" ]; then
    cd /app/backend/api
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
elif [ "$SERVICE" = "frontend" ]; then
    cd /app/frontend/streamlit
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0
else
    echo "Unknown service: $SERVICE"
    exit 1
fi
