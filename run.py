"""Quick start script"""
import os
import sys
import subprocess

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         🔍 IMAGE FORGERY DETECTION SYSTEM                    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Check dependencies
    print("Checking dependencies...")
    try:
        import torch
        import fastapi
        import cv2
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return
    
    # Initialize database
    print("\nInitializing database...")
    from backend.database import init_db
    init_db()
    print("✓ Database ready")
    
    # Start server
    print("\n" + "="*50)
    print("Starting server...")
    print("API: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("Frontend: Open frontend/index.html in browser")
    print("="*50 + "\n")
    
    subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

if __name__ == "__main__":
    main()
