@echo off
REM Setup script for Docker deployment on Windows
REM This script helps with common Docker operations for the Web Content Analyzer

setlocal enabledelayedexpansion

echo Web Content Analyzer - Docker Setup
echo.

REM Function to check if Docker is installed
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Docker is not installed. Please install Docker first:
    echo https://docs.docker.com/get-docker/
    exit /b 1
)

where docker-compose >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Docker Compose is not installed. Please install Docker Compose:
    echo https://docs.docker.com/compose/install/
    exit /b 1
)

REM Parse command line arguments
if "%1"=="" goto help
if "%1"=="start" goto start_app
if "%1"=="stop" goto stop_app
if "%1"=="restart" goto restart_app
if "%1"=="logs" goto view_logs
if "%1"=="rebuild" goto rebuild_app
goto help

:start_app
echo Building and starting the application...
docker-compose up -d --build
if %ERRORLEVEL% equ 0 (
    echo.
    echo Application started successfully!
    echo Frontend: http://localhost:8501
    echo Backend API: http://localhost:8000
    echo API Documentation: http://localhost:8000/api/docs
) else (
    echo.
    echo Failed to start the application.
    echo Check the logs with: docker-compose logs
)
goto end

:stop_app
echo Stopping the application...
docker-compose down
if %ERRORLEVEL% equ 0 (
    echo.
    echo Application stopped successfully!
) else (
    echo.
    echo Failed to stop the application.
)
goto end

:restart_app
call :stop_app
call :start_app
goto end

:view_logs
if "%2"=="" (
    echo Viewing logs for all services...
    docker-compose logs -f
) else (
    echo Viewing logs for %2...
    docker-compose logs -f %2
)
goto end

:rebuild_app
echo Rebuilding the application...
docker-compose build --no-cache
if %ERRORLEVEL% equ 0 (
    echo.
    echo Application rebuilt successfully!
    echo Starting the application...
    docker-compose up -d
) else (
    echo.
    echo Failed to rebuild the application.
)
goto end

:help
echo Usage:
echo   docker-setup.bat start - Build and start the application
echo   docker-setup.bat stop - Stop the application
echo   docker-setup.bat restart - Restart the application
echo   docker-setup.bat logs [service] - View logs (optional: specify service)
echo   docker-setup.bat rebuild - Rebuild the application
goto end

:end
endlocal
exit /b 0
