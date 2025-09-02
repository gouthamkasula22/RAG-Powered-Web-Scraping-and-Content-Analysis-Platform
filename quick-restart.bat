@echo off
REM Quick restart script for development

echo 🔄 Quick restarting services...

if "%1"=="backend" (
    echo 🔧 Restarting backend only...
    docker-compose restart backend
    echo ✅ Backend restarted!
) else if "%1"=="frontend" (
    echo 🎨 Restarting frontend only...
    docker-compose restart frontend  
    echo ✅ Frontend restarted!
) else (
    echo 🔄 Restarting both services...
    docker-compose restart
    echo ✅ Both services restarted!
)

echo 🌐 Frontend: http://localhost:8501
echo 🔧 Backend: http://localhost:8000
