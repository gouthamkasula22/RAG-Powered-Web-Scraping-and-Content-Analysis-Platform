"""
Launch script for the Web Content Analyzer
"""
import subprocess
import sys
import os
import webbrowser
import time

def main():
    """Launch the Streamlit application"""
    
    print("🚀 Starting Web Content Analyzer...")
    print("📍 Interface will be available at: http://localhost:8501")
    
    # Set environment variables for better performance
    os.environ['STREAMLIT_THEME_PRIMARY_COLOR'] = '#007bff'
    os.environ['STREAMLIT_THEME_BACKGROUND_COLOR'] = '#ffffff'
    os.environ['STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR'] = '#f8f9fa'
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n✋ Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
