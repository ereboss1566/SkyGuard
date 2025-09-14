"""
SkyGuard System Orchestrator
Orchestrates the complete real-time weather prediction system
"""
import sys
import os
import subprocess
import time
import threading
from pathlib import Path

def start_backend():
    """Start the backend API server"""
    print("Starting backend server...")
    try:
        backend_dir = Path("web/backend")
        if not backend_dir.exists():
            print("Backend directory not found!")
            return None
            
        # Change to backend directory
        os.chdir(backend_dir)
        
        # Start the Flask server
        backend_process = subprocess.Popen([
            sys.executable, "server.py"
        ])
        
        print("Backend server started successfully!")
        return backend_process
        
    except Exception as e:
        print(f"Error starting backend: {e}")
        return None

def start_frontend():
    """Start the frontend development server"""
    print("Starting frontend development server...")
    try:
        frontend_dir = Path("../frontend")
        if not frontend_dir.exists():
            print("Frontend directory not found!")
            return None
            
        # Change to frontend directory
        os.chdir(frontend_dir)
        
        # Start the React development server
        frontend_process = subprocess.Popen([
            "npm", "start"
        ])
        
        print("Frontend development server started successfully!")
        return frontend_process
        
    except Exception as e:
        print(f"Error starting frontend: {e}")
        return None

def main():
    """Main orchestrator function"""
    print("=" * 60)
    print("SkyGuard Real-time Weather Prediction System")
    print("=" * 60)
    print()
    
    # Store original directory
    original_dir = os.getcwd()
    
    try:
        # Start backend server
        print("1. Starting Backend Services...")
        backend_process = start_backend()
        if not backend_process:
            print("Failed to start backend services!")
            return
            
        # Wait a moment for backend to initialize
        time.sleep(5)
        
        # Start frontend development server
        print("\n2. Starting Frontend Services...")
        os.chdir(original_dir)  # Return to original directory
        frontend_process = start_frontend()
        if not frontend_process:
            print("Failed to start frontend services!")
            # Terminate backend if frontend fails
            if backend_process:
                backend_process.terminate()
            return
            
        # Wait a moment for frontend to initialize
        time.sleep(10)
        
        print("\n" + "=" * 60)
        print("SYSTEM STARTUP COMPLETE!")
        print("=" * 60)
        print("Frontend: http://localhost:3000")
        print("Backend API: http://localhost:8000")
        print("Backend Health Check: http://localhost:8000/health")
        print()
        print("Press Ctrl+C to stop both services...")
        print("=" * 60)
        
        # Wait for both processes
        try:
            backend_process.wait()
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\n\nShutting down services...")
            if backend_process:
                backend_process.terminate()
            if frontend_process:
                frontend_process.terminate()
            print("Services stopped.")
            
    except Exception as e:
        print(f"Orchestrator error: {e}")
    finally:
        # Return to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main()