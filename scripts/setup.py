#!/usr/bin/env python3
"""
Setup script for Web Content Analyzer project.
Run this script to initialize the development environment.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.11 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor} is compatible")
    return True


def create_virtual_environment():
    """Create and activate virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command(
        "python -m venv venv",
        "Creating virtual environment"
    )


def activate_and_install_dependencies():
    """Install project dependencies."""
    if os.name == 'nt':  # Windows
        activate_script = r"venv\Scripts\activate"
        pip_command = r"venv\Scripts\pip"
    else:  # Unix/Linux/macOS
        activate_script = "source venv/bin/activate"
        pip_command = "venv/bin/pip"
    
    commands = [
        f"{pip_command} install --upgrade pip",
        f"{pip_command} install -r requirements.txt",
    ]
    
    for command in commands:
        if not run_command(command, f"Running: {command}"):
            return False
    
    return True


def setup_environment_file():
    """Copy .env.example to .env if it doesn't exist."""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from .env.example")
        return True
    else:
        print("❌ .env.example file not found")
        return False


def run_initial_tests():
    """Run initial tests to verify setup."""
    if os.name == 'nt':  # Windows
        pytest_command = r"venv\Scripts\pytest"
    else:  # Unix/Linux/macOS
        pytest_command = "venv/bin/pytest"
    
    # Only run if tests directory has actual test files
    test_files = list(Path("tests").rglob("test_*.py"))
    if not test_files:
        print("ℹ️ No test files found, skipping test run")
        return True
    
    return run_command(
        f"{pytest_command} tests/ -v",
        "Running initial tests"
    )


def display_next_steps():
    """Display next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("\n2. Start the FastAPI backend:")
    print("   uvicorn src.presentation.api.main:app --reload --port 8000")
    
    print("\n3. Start the Streamlit frontend (in another terminal):")
    print("   streamlit run src/presentation/ui/main.py --server.port 8501")
    
    print("\n4. Open your browser:")
    print("   - API: http://localhost:8000")
    print("   - UI: http://localhost:8501")
    print("   - API Docs: http://localhost:8000/docs")
    
    print("\n5. Run tests:")
    print("   pytest tests/ -v")
    
    print("\n📝 Configuration:")
    print("   - Edit .env file for custom settings")
    print("   - Check src/infrastructure/config/settings.py")
    
    print("\n🛠️ Development Commands:")
    print("   - Format code: black src/ tests/")
    print("   - Sort imports: isort src/ tests/")
    print("   - Type checking: mypy src/")
    print("   - Linting: flake8 src/")


def main():
    """Main setup function."""
    print("🚀 Web Content Analyzer - Project Setup")
    print("="*50)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    # Setup steps
    setup_steps = [
        create_virtual_environment,
        activate_and_install_dependencies,
        setup_environment_file,
        # run_initial_tests,  # Commented out since we don't have tests yet
    ]
    
    for step in setup_steps:
        if not step():
            print(f"\n❌ Setup failed at step: {step.__name__}")
            return False
    
    display_next_steps()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
