#!/bin/bash

# Setup script for Docker deployment
# This script helps with common Docker operations for the Web Content Analyzer

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Web Content Analyzer - Docker Setup${NC}\n"

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first:${NC}"
        echo "https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed. Please install Docker Compose:${NC}"
        echo "https://docs.docker.com/compose/install/"
        exit 1
    fi
}

# Function to build and start the application
start_app() {
    echo -e "${YELLOW}Building and starting the application...${NC}"
    docker-compose up -d --build
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}Application started successfully!${NC}"
        echo -e "Frontend: ${YELLOW}http://localhost:8501${NC}"
        echo -e "Backend API: ${YELLOW}http://localhost:8000${NC}"
        echo -e "API Documentation: ${YELLOW}http://localhost:8000/api/docs${NC}"
    else
        echo -e "\n${RED}Failed to start the application.${NC}"
        echo "Check the logs with: docker-compose logs"
    fi
}

# Function to stop the application
stop_app() {
    echo -e "${YELLOW}Stopping the application...${NC}"
    docker-compose down
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}Application stopped successfully!${NC}"
    else
        echo -e "\n${RED}Failed to stop the application.${NC}"
    fi
}

# Function to view logs
view_logs() {
    if [ -z "$1" ]; then
        echo -e "${YELLOW}Viewing logs for all services...${NC}"
        docker-compose logs -f
    else
        echo -e "${YELLOW}Viewing logs for $1...${NC}"
        docker-compose logs -f "$1"
    fi
}

# Function to rebuild the application
rebuild_app() {
    echo -e "${YELLOW}Rebuilding the application...${NC}"
    docker-compose build --no-cache
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}Application rebuilt successfully!${NC}"
        echo -e "${YELLOW}Starting the application...${NC}"
        docker-compose up -d
    else
        echo -e "\n${RED}Failed to rebuild the application.${NC}"
    fi
}

# Main script
check_docker

# Parse command line arguments
case "$1" in
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        stop_app
        start_app
        ;;
    logs)
        view_logs "$2"
        ;;
    rebuild)
        rebuild_app
        ;;
    *)
        echo -e "${GREEN}Usage:${NC}"
        echo -e "  ${YELLOW}./docker-setup.sh start${NC} - Build and start the application"
        echo -e "  ${YELLOW}./docker-setup.sh stop${NC} - Stop the application"
        echo -e "  ${YELLOW}./docker-setup.sh restart${NC} - Restart the application"
        echo -e "  ${YELLOW}./docker-setup.sh logs [service]${NC} - View logs (optional: specify service)"
        echo -e "  ${YELLOW}./docker-setup.sh rebuild${NC} - Rebuild the application"
        ;;
esac

exit 0
