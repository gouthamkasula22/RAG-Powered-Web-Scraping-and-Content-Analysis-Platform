"""
API Backend Launch Script
WBS 2.4: FastAPI backend launcher with environment configuration
"""

import sys
import os
import subprocess
import signal
import time
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if required dependencies are installed"""
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-multipart"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            sys.exit(1)


def setup_environment():
    """Setup environment variables and configuration"""
    
    # Set default environment variables if not already set
    env_defaults = {
        "ENVIRONMENT": "development",
        "API_HOST": "127.0.0.1",
        "API_PORT": "8000",
        "DEBUG": "true",
        "CORS_ORIGINS": "http://localhost:8501,http://127.0.0.1:8501"
    }
    
    for key, value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = value
    
    print("Environment configuration:")
    print(f"  Host: {os.environ.get('API_HOST')}:{os.environ.get('API_PORT')}")
    print(f"  Environment: {os.environ.get('ENVIRONMENT')}")
    print(f"  Debug: {os.environ.get('DEBUG')}")


def launch_api_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = True,
    log_level: str = "info"
) -> None:
    """Launch the FastAPI server using Uvicorn"""
    
    try:
        import uvicorn
        from src.api.main import app
        from src.api.config.settings import get_settings, configure_logging
        
        # Get settings and configure logging
        settings = get_settings()
        configure_logging(settings)
        
        print(f"\n🚀 Starting Web Content Analyzer API Server...")
        print(f"📍 Server URL: http://{host}:{port}")
        print(f"📖 API Docs: http://{host}:{port}/docs")
        print(f"🔄 Auto-reload: {reload}")
        print(f"🌐 Environment: {settings.environment}")
        print(f"\n💡 Use Ctrl+C to stop the server\n")
        
        # Get Uvicorn configuration from settings
        uvicorn_config = settings.get_uvicorn_config()
        
        # Start the server
        uvicorn.run(
            "src.api.main:app",
            **uvicorn_config
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)


def check_port_availability(host: str, port: int) -> bool:
    """Check if port is available"""
    
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(host: str, start_port: int = 8000) -> int:
    """Find next available port"""
    
    port = start_port
    while port < start_port + 100:  # Try 100 ports
        if check_port_availability(host, port):
            return port
        port += 1
    
    raise RuntimeError(f"No available ports found starting from {start_port}")


def show_health_check(host: str, port: int) -> None:
    """Show API health check status"""
    
    try:
        import requests
        
        health_url = f"http://{host}:{port}/health"
        
        print(f"\n🔍 Checking API health at {health_url}")
        
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is healthy!")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Version: {health_data.get('version', 'unknown')}")
        else:
            print(f"⚠️  API health check failed: {response.status_code}")
            
    except ImportError:
        print("📦 Install 'requests' package to enable health checks")
    except Exception as e:
        print(f"❌ Health check failed: {e}")


def main():
    """Main launcher function"""
    
    print("🔧 Web Content Analyzer API Backend Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("1. Checking dependencies...")
    check_dependencies()
    
    # Setup environment
    print("2. Setting up environment...")
    setup_environment()
    
    # Get configuration
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    environment = os.environ.get("ENVIRONMENT", "development")
    
    # Check port availability
    print("3. Checking port availability...")
    if not check_port_availability(host, port):
        print(f"⚠️  Port {port} is already in use")
        try:
            new_port = find_available_port(host, port + 1)
            print(f"🔄 Using port {new_port} instead")
            port = new_port
            os.environ["API_PORT"] = str(port)
        except RuntimeError as e:
            print(f"❌ {e}")
            sys.exit(1)
    
    # Launch server
    print("4. Launching API server...")
    
    reload = environment == "development"
    
    try:
        launch_api_server(
            host=host,
            port=port,
            reload=reload,
            log_level="debug" if environment == "development" else "info"
        )
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
