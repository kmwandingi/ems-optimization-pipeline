#!/usr/bin/env python
"""
MLflow UI Launcher for EMS Optimization

Simple script to launch the MLflow UI for viewing EMS optimization experiments.

Usage:
    python scripts/launch_mlflow.py
    python scripts/launch_mlflow.py --port 5001
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def launch_mlflow_ui(port: int = 5000, host: str = "localhost"):
    """
    Launch MLflow UI.
    
    Args:
        port: Port to run MLflow UI on
        host: Host to bind to
    """
    
    # Set MLflow tracking URI to local directory
    tracking_uri = "file:./mlflow_runs"
    
    print("="*60)
    print("LAUNCHING MLFLOW UI FOR EMS OPTIMIZATION")
    print("="*60)
    print(f"Tracking URI: {tracking_uri}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"URL: http://{host}:{port}")
    print("="*60)
    
    # Check if mlflow_runs directory exists
    if not os.path.exists("mlflow_runs"):
        print("⚠ mlflow_runs directory not found")
        print("Run some experiments first using:")
        print("  python scripts/01_run.py --building DE_KN_residential1 --mode centralised")
        print("  python scripts/02_integrated_pipeline.py --building DE_KN_residential1")
        return False
    
    # Check if MLflow is installed
    try:
        import mlflow
        print(f"✓ MLflow version: {mlflow.__version__}")
    except ImportError:
        print("❌ MLflow not installed")
        print("Install with: pip install mlflow")
        return False
    
    # Set environment variables
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    
    # Launch MLflow UI
    try:
        print("\n🚀 Starting MLflow UI...")
        print("Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Run mlflow ui command with explicit tracking URI
        cmd = ["mlflow", "ui", "--backend-store-uri", tracking_uri, "--host", host, "--port", str(port)]
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n✅ MLflow UI stopped")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting MLflow UI: {e}")
        return False
    except FileNotFoundError:
        print("❌ MLflow command not found")
        print("Make sure MLflow is properly installed and in your PATH")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Launch MLflow UI for EMS optimization")
    parser.add_argument("--port", type=int, default=5000, help="Port to run MLflow UI on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    
    args = parser.parse_args()
    
    success = launch_mlflow_ui(args.port, args.host)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()