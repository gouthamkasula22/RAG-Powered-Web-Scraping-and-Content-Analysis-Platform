#!/bin/bash
# Development mode script with auto-reload

echo "🚀 Starting Web Content Analyzer in Development Mode"
echo "📁 Auto-reload enabled for both frontend and backend"

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start in development mode with auto-reload
echo "🔄 Starting with auto-reload..."
docker-compose -f docker-compose.dev.yml up --build

echo "✅ Development mode started!"
echo "🌐 Frontend: http://localhost:8501"
echo "🔧 Backend API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "💡 Changes to code will automatically reload the services!"
echo "🛑 Press Ctrl+C to stop"
