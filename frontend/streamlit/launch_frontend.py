"""
Enhanced Launch Script for WBS 2.4 Streamlit Frontend
"""
import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check and install required dependencies for enhanced UI"""
    
    required_packages = [
        "streamlit",
        "plotly",
        "pandas",
        "requests",
        "sqlite3"  # Built-in, no need to install
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if package == "sqlite3":  # Skip built-in module
            continue
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            return False
    
    return True

def setup_data_directory():
    """Setup data directory for persistent storage"""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    print(f"Data directory ready: {data_dir.absolute()}")

def main():
    """Launch the enhanced Streamlit frontend"""
    
    print("🚀 Starting Enhanced Web Content Analyzer Frontend (WBS 2.4)")
    print("=" * 60)
    
    # Check dependencies
    print("1. Checking dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed")
        return
    
    # Setup directories
    print("2. Setting up data directories...")
    setup_data_directory()
    
    frontend_path = Path(__file__).parent
    app_path = frontend_path / "app.py"
    
    print("3. Starting enhanced interface...")
    print("📍 Interface will be available at: http://localhost:8501")
    print("\n🎯 Enhanced Features:")
    print("   • Real-time progress tracking")
    print("   • Interactive report navigation")
    print("   • Persistent analysis history")
    print("   • Advanced search functionality")
    print("   • Responsive design")
    print("   • Report comparison")
    
    # Set environment variables for enhanced features
    os.environ['STREAMLIT_THEME_PRIMARY_COLOR'] = '#007bff'
    os.environ['STREAMLIT_THEME_BACKGROUND_COLOR'] = '#ffffff'
    os.environ['STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR'] = '#f8f9fa'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'true'
    
    try:
        # Launch Streamlit with enhanced configuration
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "false",
            "--server.enableCORS", "true",
            "--server.enableXsrfProtection", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n✋ Frontend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching frontend: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
