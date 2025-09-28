#!/usr/bin/env python3
"""
AI Trading System Dashboard Runner
Replaces run.py - Launches the complete dashboard system with backend and frontend
Production-ready launcher with comprehensive system checks and monitoring
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import time
import signal
import logging
logging.getLogger("gymnasium.envs.registration").setLevel(logging.ERROR)
import argparse
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional, List
import psutil
import threading
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# Configuration and Constants
# ═══════════════════════════════════════════════════════════════════

DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 3000
BACKEND_STARTUP_TIMEOUT = 120
BACKEND_MODULE = "backend.main:app"

MESSAGES = {
    'title': '🤖 AI Trading System Dashboard',
    'subtitle': 'Production-ready PPO-Lagrangian Trading System',
    'starting': 'Starting AI Trading Dashboard...',
    'backend_starting': 'Starting backend server...',
    'frontend_starting': 'Starting frontend development server...',
    'system_ready': 'System ready! Dashboard available at:',
    'backend_ready': 'Backend API available at:',
    'opening_browser': 'Opening browser...',
    'shutdown': 'Shutting down dashboard...',
    'error': 'Error occurred:',
    'checking_deps': 'Checking dependencies...',
    'deps_available': 'All dependencies available',
    'missing_deps': 'Missing dependencies. Run: pip install -r requirements.txt',
    'port_in_use': 'Port {port} is already in use',
    'checking_ports': 'Checking port availability...',
    'ports_available': 'Ports available',
    'backend_health': 'Backend health check: {status}',
    'frontend_build_check': 'Checking frontend build...',
    'frontend_build_missing': 'Frontend build not found. Building now...',
    'frontend_build_complete': 'Frontend build complete',
    'production_mode': 'Running in production mode (frontend served by backend)',
    'dev_mode': 'Running in development mode (separate frontend server)',
}

# ═══════════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════════

def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'dashboard.log', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger("DashboardRunner")
    return logger

# ═══════════════════════════════════════════════════════════════════
# Dashboard Manager Class
# ═══════════════════════════════════════════════════════════════════

class DashboardManager:
    """Manages the complete AI trading dashboard system"""
    
    def __init__(self, backend_port: int = DEFAULT_BACKEND_PORT, 
                 frontend_port: int = DEFAULT_FRONTEND_PORT, 
                 dev_mode: bool = False, debug: bool = False):
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.dev_mode = dev_mode
        self.debug = debug
        
        self.logger = setup_logging(debug)
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.is_running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Received shutdown signal. Stopping dashboard...")
        self.stop()
        sys.exit(0)
    
    def display_header(self):
        """Display professional header"""
        width = 80
        print("=" * width)
        print(f"{MESSAGES['title']:^{width}}")
        print(f"{MESSAGES['subtitle']:^{width}}")
        print("=" * width)
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Mode: {'Development' if self.dev_mode else 'Production'}")
        print(f"🌐 Backend Port: {self.backend_port}")
        if self.dev_mode:
            print(f"🖥️ Frontend Port: {self.frontend_port}")
        print("=" * width)
        print()
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        self.logger.info(MESSAGES['checking_deps'])
        
        # Check Python packages
        required_packages = [
            'fastapi', 'uvicorn', 'pandas', 'numpy', 
            'MetaTrader5', 'torch', 'stable_baselines3'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing packages: {missing_packages}")
            self.logger.error(MESSAGES['missing_deps'])
            return False
            
        # Check Node.js if in dev mode
        if self.dev_mode:
            try:
                result = subprocess.run(['node', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    self.logger.error("Node.js not found")
                    return False
                self.logger.info(f"Node.js version: {result.stdout.strip()}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.error("Node.js not found or not responding")
                return False
                
            # Check if node_modules exists
            if not Path('frontend/node_modules').exists():
                self.logger.error("frontend/node_modules not found. Run: cd frontend && npm install")
                return False
        
        self.logger.info(MESSAGES['deps_available'])
        return True

    def check_ports(self) -> bool:
        """Check if required ports are available"""
        self.logger.info(MESSAGES['checking_ports'])
        
        ports_to_check = [self.backend_port]
        if self.dev_mode:
            ports_to_check.append(self.frontend_port)
            
        for port in ports_to_check:
            if self.is_port_in_use(port):
                self.logger.error(MESSAGES['port_in_use'].format(port=port))
                return False
                
        self.logger.info(MESSAGES['ports_available'])
        return True

    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is currently in use"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return True
        return False
    
    def check_frontend_build(self) -> bool:
        """Check and build frontend if necessary"""
        frontend_dist = Path("frontend/dist")
        
        if not self.dev_mode:
            self.logger.info(MESSAGES['frontend_build_check'])
            
            if not frontend_dist.exists() or not any(frontend_dist.iterdir()):
                self.logger.info(MESSAGES['frontend_build_missing'])
                
                # Build frontend
                try:
                    os.chdir("frontend")
                    
                    # Install dependencies if needed
                    if not Path("node_modules").exists():
                        self.logger.info("Installing frontend dependencies...")
                        subprocess.run(["npm", "install"], check=True, timeout=300)
                    
                    # Build frontend
                    subprocess.run(["npm", "run", "build"], check=True, timeout=180)
                    os.chdir("..")
                    
                    self.logger.info(MESSAGES['frontend_build_complete'])
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Frontend build failed: {e}")
                    os.chdir("..")  # Ensure we're back in root directory
                    return False
            else:
                self.logger.info("Frontend build found")
                return True
        
        return True
    
    def start_backend(self) -> bool:
        """Start the FastAPI backend server and wait for /health to respond."""
        self.logger.info(MESSAGES['backend_starting'])
        
        try:
            # Build Uvicorn command
            backend_cmd = [
                sys.executable, "-m", "uvicorn",
                BACKEND_MODULE,
                "--host", "0.0.0.0",
                "--port", str(self.backend_port),
                "--log-level", "debug" if self.debug else "info"
            ]
            if self.debug:
                # Insert --reload just after the port argument
                backend_cmd.insert(6, "--reload")
            
            # Inherit environment + custom vars
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": ".",
                "PYTHONUNBUFFERED": "1",
                "TRADING_ENV": "development" if self.debug else "production"
            })
            
            # Launch the process
            self.backend_process = subprocess.Popen(
                backend_cmd,
                env=env,
                stdout=(None if self.debug else subprocess.PIPE),
                stderr=(None if self.debug else subprocess.PIPE),
                universal_newlines=True
            )
            
            # Poll /health once per second until it returns 200 or we timeout
            for elapsed in range(BACKEND_STARTUP_TIMEOUT):
                # If process died early, capture stderr and fail immediately
                if self.backend_process.poll() is not None:
                    error_output = (self.backend_process.stderr.read()
                                    if self.backend_process.stderr else "Unknown error")
                    self.logger.error(f"Backend failed to start: {error_output}")
                    return False
                
                if self.check_backend_health():
                    self.logger.info(
                        f"Backend started successfully on port {self.backend_port} "
                        f"(waited {elapsed+1}s)")
                    return True
                
                time.sleep(1)
            
            # If we get here, we never saw a healthy response in time
            self.logger.error(
                f"Backend failed to start within {BACKEND_STARTUP_TIMEOUT}s timeout")
            return False
        
        except Exception as e:
            self.logger.error(f"Failed to start backend: {e}")
            return False

    
    def start_frontend(self) -> bool:
        """Start the frontend development server (dev mode only)"""
        if not self.dev_mode:
            return True
            
        self.logger.info(MESSAGES['frontend_starting'])
        
        try:
            frontend_cmd = [
                "npm", "run", "dev",
                "--", "--port", str(self.frontend_port), "--host", "0.0.0.0"
            ]
            
            self.frontend_process = subprocess.Popen(
                frontend_cmd,
                cwd="frontend",
                stdout=subprocess.PIPE if not self.debug else None,
                stderr=subprocess.PIPE if not self.debug else None,
                universal_newlines=True
            )
            
            # Wait for frontend to start
            for i in range(20):  # Wait up to 20 seconds
                if self.frontend_process.poll() is not None:
                    error_output = self.frontend_process.stderr.read() if self.frontend_process.stderr else "Unknown error"
                    self.logger.error(f"Frontend failed to start: {error_output}")
                    return False
                
                if self.is_port_in_use(self.frontend_port):
                    self.logger.info(f"Frontend started successfully on port {self.frontend_port}")
                    return True
                
                time.sleep(1)
            
            self.logger.error("Frontend failed to start within timeout period")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start frontend: {e}")
            return False
    
    def check_backend_health(self) -> bool:
        """Check if backend is responding"""
        try:
            import requests
            response = requests.get(f"http://localhost:{self.backend_port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def open_browser(self):
        """Open the dashboard in the default browser"""
        if self.dev_mode:
            url = f"http://localhost:{self.frontend_port}"
        else:
            url = f"http://localhost:{self.backend_port}"
        
        self.logger.info(MESSAGES['opening_browser'])
        
        def delayed_open():
            time.sleep(2)  # Give servers time to fully start
            try:
                webbrowser.open(url)
            except Exception as e:
                self.logger.warning(f"Could not open browser automatically: {e}")
        
        thread = threading.Thread(target=delayed_open, daemon=True)
        thread.start()
    
    def display_ready_message(self):
        """Display system ready message with URLs"""
        print("\n" + "="*60)
        print(f"🎉 {MESSAGES['system_ready']}")
        print("="*60)
        
        if self.dev_mode:
            print(f"🖥️  Frontend Dashboard:  http://localhost:{self.frontend_port}")
            print(f"🔧 Backend API:         http://localhost:{self.backend_port}")
            print(f"📚 API Documentation:   http://localhost:{self.backend_port}/docs")
        else:
            print(f"🖥️  Dashboard:           http://localhost:{self.backend_port}")
            print(f"📚 API Documentation:   http://localhost:{self.backend_port}/docs")
            print(f"❤️  Health Check:       http://localhost:{self.backend_port}/health")
        
        print("="*60)
        print("🛑 Press Ctrl+C to stop the dashboard")
        print("="*60)
    
    def start(self) -> bool:
        """Start the complete dashboard system"""
        self.display_header()
        
        # Pre-flight checks
        if not self.check_dependencies():
            return False
            
        if not self.check_ports():
            return False
            
        if not self.check_frontend_build():
            return False
        
        # Start backend
        if not self.start_backend():
            self.stop()
            return False
        
        # Start frontend (dev mode only)
        if not self.start_frontend():
            self.stop()
            return False
        
        self.is_running = True
        
        # Display ready message and open browser
        self.display_ready_message()
        self.open_browser()
        
        return True
    
    def stop(self):
        """Stop all dashboard processes"""
        if not self.is_running:
            return
        
        self.logger.info(MESSAGES['shutdown'])
        self.is_running = False
        
        # Stop frontend process
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping frontend: {e}")
        
        # Stop backend process
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping backend: {e}")
        
        self.logger.info("Dashboard stopped successfully")
    
    def wait(self):
        """Wait for processes to complete"""
        try:
            while self.is_running:
                # Check if processes are still running
                if self.backend_process and self.backend_process.poll() is not None:
                    self.logger.error("Backend process died unexpectedly")
                    break
                    
                if self.frontend_process and self.frontend_process.poll() is not None:
                    self.logger.error("Frontend process died unexpectedly")
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            self.stop()

# ═══════════════════════════════════════════════════════════════════
# Command Line Interface
# ═══════════════════════════════════════════════════════════════════

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Trading System Dashboard Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start in production mode
  %(prog)s --dev                    # Start in development mode
  %(prog)s --port 8080              # Use custom backend port
  %(prog)s --dev --frontend-port 3001  # Custom frontend port
  %(prog)s --debug                  # Enable debug mode
        """
    )
    
    parser.add_argument(
        '--port', '--backend-port',
        type=int,
        default=DEFAULT_BACKEND_PORT,
        help=f'Backend server port (default: {DEFAULT_BACKEND_PORT})'
    )
    
    parser.add_argument(
        '--frontend-port',
        type=int,
        default=DEFAULT_FRONTEND_PORT,
        help=f'Frontend server port for dev mode (default: {DEFAULT_FRONTEND_PORT})'
    )
    
    parser.add_argument(
        '--dev', '--development',
        action='store_true',
        help='Run in development mode with separate frontend server'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check system requirements and exit'
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create dashboard manager
    dashboard = DashboardManager(
        backend_port=args.port,
        frontend_port=args.frontend_port,
        dev_mode=args.dev,
        debug=args.debug
    )
    
    # Check mode only
    if args.check:
        dashboard.display_header()
        success = (
            dashboard.check_dependencies() and
            dashboard.check_ports() and
            dashboard.check_frontend_build()
        )
        
        if success:
            print("\n✅ All system checks passed! Ready to run dashboard.")
            return 0
        else:
            print("\n❌ System checks failed. Please fix the issues above.")
            return 1
    
    # Disable browser opening if requested
    if args.no_browser:
        dashboard.open_browser = lambda: None
    
    # Start dashboard
    try:
        if dashboard.start():
            dashboard.wait()
            return 0
        else:
            print("\n❌ Failed to start dashboard. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        dashboard.stop()
        return 1

if __name__ == "__main__":
    sys.exit(main())