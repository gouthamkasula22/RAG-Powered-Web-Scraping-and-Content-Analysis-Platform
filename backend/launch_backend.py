"""
Launch script for the FastAPI Backend
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the FastAPI backend"""
    
    backend_path = Path(__file__).parent
    api_path = backend_path / "main.py"
    
    print("🚀 Starting Web Content Analyzer Backend...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/api/docs")
    print("🔍 Health Check: http://localhost:8000/api/health")
    
    try:
        # Launch FastAPI with uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], cwd=str(backend_path / "api"), check=True)
    except KeyboardInterrupt:
        print("\n✋ Backend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching backend: {e}")
        print("💡 Make sure FastAPI and uvicorn are installed:")
        print("   pip install fastapi uvicorn")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
