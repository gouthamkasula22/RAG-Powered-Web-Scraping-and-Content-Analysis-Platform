"""
Main launch script for the complete Web Content Analyzer
Starts both frontend and backend services
"""
import subprocess
import sys
import time
import threading
from pathlib import Path

def launch_backend():
    """Launch the FastAPI backend in a separate process"""
    backend_path = Path(__file__).parent / "backend"
    launcher_path = backend_path / "launch_backend.py"
    
    try:
        subprocess.run([sys.executable, str(launcher_path)], check=True)
    except Exception as e:
        print(f"❌ Backend launch failed: {e}")

def launch_frontend():
    """Launch the Streamlit frontend in a separate process"""
    frontend_path = Path(__file__).parent / "frontend" / "streamlit"
    launcher_path = frontend_path / "launch_frontend.py"
    
    try:
        subprocess.run([sys.executable, str(launcher_path)], check=True)
    except Exception as e:
        print(f"❌ Frontend launch failed: {e}")

def main():
    """Launch both frontend and backend"""
    
    print("🚀 Starting Web Content Analyzer - Full Stack")
    print("=" * 60)
    print("📡 Backend API: http://localhost:8000")
    print("🖥️  Frontend UI: http://localhost:8501")
    print("📚 API Docs: http://localhost:8000/api/docs")
    print("=" * 60)
    
    choice = input("Choose launch mode:\n1. Full Stack (Backend + Frontend)\n2. Backend Only\n3. Frontend Only\nEnter choice (1-3): ")
    
    if choice == "1":
        print("\n🚀 Starting Full Stack...")
        
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=launch_backend, daemon=True)
        backend_thread.start()
        
        # Wait a moment for backend to start
        print("⏳ Starting backend...")
        time.sleep(3)
        
        # Start frontend in main thread
        print("🎨 Starting frontend...")
        launch_frontend()
        
    elif choice == "2":
        print("\n🚀 Starting Backend Only...")
        launch_backend()
        
    elif choice == "3":
        print("\n🚀 Starting Frontend Only...")
        launch_frontend()
        
    else:
        print("❌ Invalid choice. Please run again and select 1, 2, or 3.")

if __name__ == "__main__":
    main()
