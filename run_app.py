#!/usr/bin/env python3
"""
Launcher script for the Vehicle Registration AI Streamlit App
============================================================

This script ensures the virtual environment is activated and runs the Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app"""
    
    # Get the project directory
    project_dir = Path(__file__).parent
    
    # Check if virtual environment exists
    venv_path = project_dir / ".venv"
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run setup first.")
        print("   Run: python setup.py")
        sys.exit(1)
    
    # Check if .env file exists
    env_path = project_dir / ".env"
    if not env_path.exists():
        print("❌ .env file not found. Please create it with your OpenAI API key.")
        print("   Create .env file with: OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    # Activate virtual environment and run streamlit
    venv_python = venv_path / "bin" / "python"
    streamlit_script = project_dir / "streamlit_app.py"
    
    print("🚀 Starting Vehicle Registration AI Streamlit App...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            str(venv_python), "-m", "streamlit", "run", 
            str(streamlit_script),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], cwd=project_dir)
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
