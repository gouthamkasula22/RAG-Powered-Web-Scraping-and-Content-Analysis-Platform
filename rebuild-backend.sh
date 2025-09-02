#!/bin/bash
echo "🔄 Rebuilding backend with latest image extraction code..."

# Stop current containers
docker-compose -f docker-compose.dev.yml down

# Build new backend image with current code
docker build -t web-content-analyzer-backend:latest .

# Update docker-compose to use the new image
sed -i 's/web-content-analyzer-backend:chromadb/web-content-analyzer-backend:latest/g' docker-compose.yml

# Start containers
docker-compose up -d

echo "✅ Backend rebuilt and started with latest code!"
