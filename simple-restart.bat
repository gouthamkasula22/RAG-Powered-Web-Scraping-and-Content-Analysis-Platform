@echo off
echo Restarting containers with latest code...
docker-compose down
docker-compose up -d
echo Done! Access the app at http://localhost:8501
