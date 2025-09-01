@echo off
echo ========================================
echo Web Content Analyzer - ChromaDB Setup
echo ========================================

echo Starting containers...
docker-compose up -d

echo.
echo Waiting for containers to be ready...
timeout /t 10

echo.
echo Checking ChromaDB status...
docker-compose exec frontend python -c "import chromadb; client = chromadb.PersistentClient(path='data/chroma_db'); collection = client.get_or_create_collection(name='website_chunks'); print(f'ChromaDB count: {collection.count()}')"

echo.
echo ========================================
echo ChromaDB setup complete!
echo.
echo You can now access the application at:
echo - Frontend: http://localhost:8501
echo - Backend API: http://localhost:8000
echo.
echo The RAG Knowledge Repository is ready with ChromaDB vector search.
echo ========================================
pause
