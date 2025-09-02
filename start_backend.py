#!/usr/bin/env python3
"""
Simple backend launch script
"""
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Set up environment
os.environ['DATABASE_PATH'] = './data/analysis_history.db'

# Start the backend
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=[str(backend_dir)]
    )
