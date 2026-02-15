#!/usr/bin/env python
"""
Docker deployment automation script for Walmart Retail Forecasting System.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, check=True):
    """Run shell command and return output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout


def check_docker():
    """Check if Docker is installed and running."""
    try:
        run_command("docker --version")
        run_command("docker-compose --version")
        print("[OK] Docker and Docker Compose are installed")
        return True
    except:
        print("[ERROR] Docker or Docker Compose not found. Please install Docker Desktop.")
        return False


def check_env_file():
    """Check if .env file exists."""
    if not Path(".env").exists():
        print("[WARNING] .env file not found. Creating from .env.example...")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("[OK] Created .env file. Please edit it with your credentials.")
            return False
        else:
            print("[ERROR] .env.example not found")
            return False
    print("[OK] .env file exists")
    return True


def build_image(tag="latest"):
    """Build Docker image."""
    print("\n[BUILD] Building Docker image...")
    run_command(f"docker build -t walmart-forecasting:{tag} .")
    print(f"[OK] Image built: walmart-forecasting:{tag}")


def start_services():
    """Start Docker Compose services."""
    print("\n[START] Starting services with Docker Compose...")
    run_command("docker-compose up -d")
    print("[OK] Services started")
    
    print("\n[STATUS] Service status:")
    run_command("docker-compose ps", check=False)


def stop_services():
    """Stop Docker Compose services."""
    print("\n[STOP] Stopping services...")
    run_command("docker-compose down")
    print("[OK] Services stopped")


def view_logs(service="app", follow=False):
    """View service logs."""
    cmd = f"docker-compose logs {service}"
    if follow:
        cmd += " -f"
    run_command(cmd, check=False)


def push_to_dockerhub(username, tag="latest"):
    """Push image to Docker Hub."""
    print(f"\n[PUSH] Pushing to Docker Hub as {username}/walmart-forecasting:{tag}")
    
    # Login
    print("Logging in to Docker Hub...")
    run_command("docker login")
    
    # Tag image
    print(f"Tagging image...")
    run_command(f"docker tag walmart-forecasting:{tag} {username}/walmart-forecasting:{tag}")
    
    # Push
    print(f"Pushing image...")
    run_command(f"docker push {username}/walmart-forecasting:{tag}")
    
    print(f"[OK] Image pushed: {username}/walmart-forecasting:{tag}")
    print(f"\n[INFO] Pull with: docker pull {username}/walmart-forecasting:{tag}")


def run_tests():
    """Run tests in Docker container."""
    print("\n[TEST] Running tests...")
    run_command("docker-compose exec app pytest tests/ -v", check=False)


def initialize_data():
    """Initialize database and load data."""
    print("\n[INIT] Initializing database...")
    run_command("docker-compose exec app python scripts/setup_database.py")
    
    print("\n[DATA] Loading data...")
    run_command("docker-compose exec app python data_pipeline/data_loader.py")
    
    print("[OK] Data initialized")


def main():
    parser = argparse.ArgumentParser(description="Docker deployment automation")
    parser.add_argument("command", choices=[
        "check", "build", "start", "stop", "restart",
        "logs", "push", "test", "init", "full-deploy"
    ], help="Command to execute")
    parser.add_argument("--tag", default="latest", help="Docker image tag")
    parser.add_argument("--username", help="Docker Hub username (for push)")
    parser.add_argument("--follow", action="store_true", help="Follow logs")
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    if args.command == "check":
        print("[CHECK] Checking prerequisites...")
        docker_ok = check_docker()
        env_ok = check_env_file()
        if docker_ok and env_ok:
            print("\n[OK] All checks passed! Ready to deploy.")
        else:
            print("\n[WARNING] Please fix the issues above before deploying.")
    
    elif args.command == "build":
        if not check_docker():
            return
        build_image(args.tag)
    
    elif args.command == "start":
        if not check_docker() or not check_env_file():
            return
        start_services()
        print("\n[INFO] Access dashboard at: http://localhost:8501")
        print("[INFO] Access MLflow at: http://localhost:5000")
    
    elif args.command == "stop":
        stop_services()
    
    elif args.command == "restart":
        stop_services()
        start_services()
    
    elif args.command == "logs":
        view_logs(follow=args.follow)
    
    elif args.command == "push":
        if not args.username:
            print("[ERROR] Please provide --username for Docker Hub")
            return
        if not check_docker():
            return
        push_to_dockerhub(args.username, args.tag)
    
    elif args.command == "test":
        run_tests()
    
    elif args.command == "init":
        initialize_data()
    
    elif args.command == "full-deploy":
        print("[DEPLOY] Starting full deployment...")
        if not check_docker() or not check_env_file():
            return
        
        build_image(args.tag)
        start_services()
        
        print("\n[WAIT] Waiting for services to be ready...")
        import time
        time.sleep(10)
        
        initialize_data()
        
        print("\n[OK] Full deployment complete!")
        print("\n[INFO] Access dashboard at: http://localhost:8501")
        print("[INFO] Access MLflow at: http://localhost:5000")
        
        if args.username:
            push_to_dockerhub(args.username, args.tag)


if __name__ == "__main__":
    main()
