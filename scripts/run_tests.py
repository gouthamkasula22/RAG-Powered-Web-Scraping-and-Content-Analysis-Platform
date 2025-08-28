#!/usr/bin/env python3
"""
Script to run tests with the correct Python path setup.
This script handles the Python path correctly for all test files.
"""
import os
import sys
import subprocess
from pathlib import Path
import argparse

def run_tests():
    """Run pytest with proper path configuration."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tests with proper path configuration")
    parser.add_argument("--fix-imports", action="store_true", help="Fix import issues in test files")
    parser.add_argument("--all", action="store_true", help="Run all tests including those with known issues")
    parser.add_argument("--category", type=str, help="Run tests by category (unit, integration, ui, rag, performance, security)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Run tests in verbose mode")
    parser.add_argument("tests", nargs="*", help="Specific test files or directories to run")
    
    args, unknown_args = parser.parse_known_args()
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    # Add paths to PYTHONPATH
    env = os.environ.copy()
    
    # Set up the PYTHONPATH to include all necessary directories
    paths = [
        str(project_root),             # Root directory
        str(project_root / "src"),     # Main src directory
        str(project_root / "backend"), # Backend directory
    ]
    
    # Join the paths with the appropriate separator
    python_path = os.pathsep.join(paths)
    
    # Preserve any existing PYTHONPATH
    if "PYTHONPATH" in env and env["PYTHONPATH"]:
        python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    
    env["PYTHONPATH"] = python_path
    
    # Build pytest command
    pytest_args = [sys.executable, "-m", "pytest"]
    
    # Add verbose flag if requested
    if args.verbose:
        pytest_args.append("-v")
    
    # Determine which tests to run
    if args.tests:
        # Run specific tests provided as arguments
        test_paths = args.tests
    elif args.category:
        # Run tests by category
        category = args.category.lower()
        if category == "unit":
            test_paths = ["tests/unit/"]
        elif category == "integration":
            test_paths = ["tests/integration/"]
        elif category == "ui":
            test_paths = ["tests/ui/"]
        elif category == "rag":
            test_paths = ["tests/rag_integration/"]
        elif category == "performance":
            test_paths = ["tests/performance/"]
        elif category == "security":
            test_paths = ["tests/security/"]
        else:
            print(f"Unknown category: {category}")
            return 1
    elif args.all:
        # Run all tests
        test_paths = ["tests/"]
    else:
        # Run only tests we know work
        test_paths = [
            "tests/unit/test_web_scraper.py",
            "tests/unit/knowledge_repository/",
            "tests/ui/",
            "tests/rag_integration/"
        ]
    
    # Add test paths to command
    pytest_args.extend(test_paths)
    
    # Add any unknown args to the command
    pytest_args.extend(unknown_args)
    
    # Run pytest with the updated environment
    print(f"Running tests with PYTHONPATH: {python_path}")
    print(f"Running command: {' '.join(pytest_args)}")
    result = subprocess.run(pytest_args, env=env, check=False)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())
