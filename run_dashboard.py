#!/usr/bin/env python3
"""
AI Trading Dashboard Launcher
Starts both backend and frontend services with proper coordination
"""

import os
import sys
import time
import signal
import subprocess
import threading
import logging
from pathlib import Path
import psutil
import argparse

# Configure logging for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Use stdout explicitly
        logging.FileHandler('logs/dashboard_launcher.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("DashboardLauncher")

# Windows-compatible messages (no emojis)
MESSAGES = {
    'directory_created': '[OK] Directory structure created',
    'starting_dashboard': '[STARTUP] Starting AI Trading Dashboard...',
    'checking_deps': '[CHECK] Checking dependencies...',
    'deps_available': '[OK] All dependencies available',
    'checking_ports': '[CHECK] Checking port availability...',
    'ports_available': '[OK] All ports available',
    'starting_backend': '[STARTUP] Starting backend server...',
    'backend_started': '[OK] Backend started on port {}',
    'production_mode': '[INFO] Production mode - frontend served by backend',
    'starting_tensorboard': '[STARTUP] Starting TensorBoard...',
    'tensorboard_failed': '[WARNING] TensorBoard failed to start (may start later)',
    'starting_monitor': '[MONITOR] Starting process monitor...',
    'process_stopped': '[WARNING] Process {} stopped unexpectedly',
    'restarting_process': '[RESTART] Restarting {}...',
    'restart_failed': '[ERROR] Failed to restart {}',
    'shutdown_signal': '[SIGNAL] Received signal {}, shutting down...',
    'stopping_service': '[SHUTDOWN] Stopping {}...',
    'force_killing': '[WARNING] Force killing {}...',
    'shutdown_complete': '[OK] Dashboard shutdown complete'
}

class DashboardLauncher:
    def __init__(self, dev_mode=False, port_backend=8000, port_frontend=3000):
        self.dev_mode = dev_mode
        self.port_backend = port_backend
        self.port_frontend = port_frontend
        self.processes = {}
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Ensure required directories exist
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'logs', 'logs/training', 'logs/risk', 'logs/simulation',
            'logs/strategy', 'logs/position', 'logs/tensorboard',
            'checkpoints', 'models', 'data'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        logger.info(MESSAGES['directory_created'])

    def check_dependencies(self):
        """Check if all required dependencies are available"""
        logger.info(MESSAGES['checking_deps'])
        
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
            logger.error(f"[ERROR] Missing Python packages: {missing_packages}")
            logger.error("Run: pip install -r requirements.txt")
            return False
            
        # Check Node.js if in dev mode
        if self.dev_mode:
            try:
                result = subprocess.run(['node', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    logger.error("[ERROR] Node.js not found")
                    return False
                logger.info(f"[OK] Node.js version: {result.stdout.strip()}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.error("[ERROR] Node.js not found or not responding")
                return False
                
            # Check if node_modules exists
            if not Path('node_modules').exists():
                logger.error("[ERROR] node_modules not found. Run: npm install")
                return False
        
        logger.info(MESSAGES['deps_available'])
        return True

    def check_ports(self):
        """Check if required ports are available"""
        logger.info(MESSAGES['checking_ports'])
        
        ports_to_check = [self.port_backend]
        if self.dev_mode:
            ports_to_check.append(self.port_frontend)
            
        for port in ports_to_check:
            if self.is_port_in_use(port):
                logger.error(f"[ERROR] Port {port} is already in use")
                return False
                
        logger.info(MESSAGES['ports_available'])
        return True

    def is_port_in_use(self, port):
        """Check if a port is currently in use"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return True
        return False

    def start_backend(self):
        """Start the FastAPI backend"""
        logger.info(MESSAGES['starting_backend'])
        
        try:
            backend_cmd = [
                sys.executable, 'backend/main.py'
            ]
            
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()
            env['PYTHONIOENCODING'] = 'utf-8'  # Force UTF-8 encoding
            
            self.processes['backend'] = subprocess.Popen(
                backend_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=os.getcwd()
            )
            
            # Give backend time to start
            time.sleep(3)
            
            if self.processes['backend'].poll() is None:
                logger.info(MESSAGES['backend_started'].format(self.port_backend))
                return True
            else:
                stderr_output = self.processes['backend'].stderr.read().decode()
                logger.error(f"[ERROR] Backend failed to start: {stderr_output}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to start backend: {e}")
            return False

    def start_frontend(self):
        """Start the frontend development server"""
        if not self.dev_mode:
            logger.info(MESSAGES['production_mode'])
            return True
            
        logger.info("[STARTUP] Starting frontend development server...")
        
        try:
            frontend_cmd = ['npm', 'run', 'dev']
            
            self.processes['frontend'] = subprocess.Popen(
                frontend_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.join(os.getcwd(), "frontend")  # <-- fixed path
            )
            
            # Give frontend time to start
            time.sleep(5)
            
            if self.processes['frontend'].poll() is None:
                logger.info(f"[OK] Frontend started on port {self.port_frontend}")
                return True
            else:
                stderr_output = self.processes['frontend'].stderr.read().decode()
                logger.error(f"[ERROR] Frontend failed to start: {stderr_output}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to start frontend: {e}")
            return False


    def start_tensorboard(self):
        """Start TensorBoard server"""
        logger.info(MESSAGES['starting_tensorboard'])
        
        try:
            # Ensure tensorboard directory exists
            tb_dir = 'logs/tensorboard'
            os.makedirs(tb_dir, exist_ok=True)
            
            # Create a dummy event file if none exist
            import glob
            event_files = glob.glob(os.path.join(tb_dir, "**/events.out.*"), recursive=True)
            if not event_files:
                # Create dummy event file to prevent TensorBoard from failing
                dummy_dir = os.path.join(tb_dir, 'dummy')
                os.makedirs(dummy_dir, exist_ok=True)
                logger.info("[INFO] Created dummy TensorBoard directory")
            
            tensorboard_cmd = [
                sys.executable, '-m', 'tensorboard',
                '--logdir', tb_dir,
                '--port', '6006',
                '--host', '0.0.0.0',
                '--reload_interval', '30'
            ]
            
            # Set environment for Windows compatibility
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            self.processes['tensorboard'] = subprocess.Popen(
                tensorboard_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            # Give TensorBoard more time to start
            time.sleep(5)
            
            if self.processes['tensorboard'].poll() is None:
                logger.info("[OK] TensorBoard started on port 6006")
                return True
            else:
                # Get error output for debugging
                try:
                    stdout, stderr = self.processes['tensorboard'].communicate(timeout=1)
                    logger.warning(f"[WARNING] TensorBoard startup issue: {stderr.decode()}")
                except:
                    pass
                logger.warning(MESSAGES['tensorboard_failed'])
                return True  # Not critical for system operation
                
        except Exception as e:
            logger.warning(f"[WARNING] TensorBoard start failed: {e}")
            return True  # Not critical

    def monitor_processes(self):
        """Monitor running processes and restart if needed"""
        logger.info(MESSAGES['starting_monitor'])
        
        while self.running:
            try:
                for name, process in list(self.processes.items()):
                    if process and process.poll() is not None:
                        # Don't restart TensorBoard automatically to avoid spam
                        if name == 'tensorboard':
                            continue
                            
                        logger.warning(MESSAGES['process_stopped'].format(name))
                        
                        # Try to restart critical processes
                        if name == 'backend':
                            logger.info(MESSAGES['restarting_process'].format(name))
                            if not self.start_backend():
                                logger.error(MESSAGES['restart_failed'].format(name))
                                self.running = False
                                break
                        elif name == 'frontend' and self.dev_mode:
                            logger.info(MESSAGES['restarting_process'].format(name))
                            if not self.start_frontend():
                                logger.warning(MESSAGES['restart_failed'].format(name))
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"[ERROR] Process monitor error: {e}")
                time.sleep(5)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(MESSAGES['shutdown_signal'].format(signum))
        self.shutdown()

    def shutdown(self):
        """Gracefully shutdown all processes"""
        logger.info("[SHUTDOWN] Shutting down dashboard...")
        self.running = False
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info(MESSAGES['stopping_service'].format(name))
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(MESSAGES['force_killing'].format(name))
                    process.kill()
                except Exception as e:
                    logger.error(f"[ERROR] Error stopping {name}: {e}")
        
        logger.info(MESSAGES['shutdown_complete'])

    def display_info(self):
        """Display connection information"""
        print("\n" + "="*60)
        print("AI Trading Dashboard")
        print("="*60)
        print(f"Backend API:     http://localhost:{self.port_backend}")
        
        if self.dev_mode:
            print(f"Frontend:        http://localhost:{self.port_frontend}")
        else:
            print(f"Frontend:        http://localhost:{self.port_backend}")
            
        print(f"TensorBoard:     http://localhost:6006")
        print(f"API Docs:        http://localhost:{self.port_backend}/docs")
        print("="*60)
        print("Tips:")
        print("   - Use Ctrl+C to stop all services")
        print("   - Check logs/ directory for detailed logs")
        print("   - Visit /docs for API documentation")
        print("="*60)

    def run(self):
        """Main run method"""
        logger.info(MESSAGES['starting_dashboard'])
        
        # Pre-flight checks
        if not self.check_dependencies():
            sys.exit(1)
            
        if not self.check_ports():
            sys.exit(1)
        
        # Start services
        if not self.start_backend():
            sys.exit(1)
            
        if not self.start_frontend():
            sys.exit(1)
            
        self.start_tensorboard()  # Non-critical
        
        # Display connection info
        self.display_info()
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

def main():
    parser = argparse.ArgumentParser(description="AI Trading Dashboard Launcher")
    parser.add_argument(
        '--dev', 
        action='store_true', 
        help='Run in development mode (separate frontend server)'
    )
    parser.add_argument(
        '--backend-port', 
        type=int, 
        default=8000, 
        help='Backend port (default: 8000)'
    )
    parser.add_argument(
        '--frontend-port', 
        type=int, 
        default=3000, 
        help='Frontend port for dev mode (default: 3000)'
    )
    
    args = parser.parse_args()
    
    launcher = DashboardLauncher(
        dev_mode=args.dev,
        port_backend=args.backend_port,
        port_frontend=args.frontend_port
    )
    
    launcher.run()

if __name__ == "__main__":
    main()