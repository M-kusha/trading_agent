#!/usr/bin/env python3
"""
AI Trading System Dashboard Runner (Hardened)
- Launches FastAPI backend + (optionally) frontend dev server
- Robust dependency gating, process supervision, graceful shutdown
- Clean cross-platform process groups; real-time log streaming
"""

import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import time
import signal
import logging
import argparse
import subprocess
import webbrowser
import socket
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
import threading
import json

# ───────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 3000
BACKEND_STARTUP_TIMEOUT = 120          # seconds
FRONTEND_STARTUP_TIMEOUT = 90          # seconds
BACKEND_HEALTH_PATH = "/health"
BACKEND_MODULE = "backend.main:app"
DEFAULT_DEV_MODE = True                # dev by default
OPEN_BROWSER_DELAY = 1.0               # seconds

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
    'deps_available': 'All core dependencies available',
    'missing_deps': 'Missing dependencies. See the advice above.',
    'port_in_use': 'Port {port} is already in use',
    'checking_ports': 'Checking port availability...',
    'ports_available': 'Ports available',
    'frontend_build_check': 'Checking frontend build...',
    'frontend_build_missing': 'Frontend build not found. Building now...',
    'frontend_build_complete': 'Frontend build complete',
    'production_mode': 'Running in production mode (frontend served by backend)',
    'dev_mode': 'Running in development mode (separate frontend server)',
}

# ───────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────
def setup_logging(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'dashboard.log', encoding='utf-8')
        ]
    )
    return logging.getLogger("DashboardRunner")

# ───────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────
def _import_optional(module: str):
    try:
        return __import__(module)
    except Exception:
        return None

def is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0

def who_owns_port(logger: logging.Logger, port: int) -> Optional[Tuple[int, str]]:
    """Best-effort: return (pid, cmdline) for the process bound to port."""
    psutil = _import_optional("psutil")
    if not psutil:
        return None
    try:
        for p in psutil.process_iter(attrs=["pid", "name", "cmdline", "connections"]):
            for c in p.info.get("connections", []):
                laddr = getattr(c, "laddr", None)
                if laddr and getattr(laddr, "port", None) == port:
                    cmd = " ".join(p.info.get("cmdline") or [p.info.get("name") or ""])
                    return (p.info["pid"], cmd.strip())
    except Exception as e:
        logger.debug(f"psutil port ownership scan failed: {e}")
    return None

def detect_pkg_manager(frontend_dir: Path) -> List[str]:
    if (frontend_dir / "pnpm-lock.yaml").exists():
        return ["pnpm"]
    if (frontend_dir / "yarn.lock").exists():
        return ["yarn"]
    return ["npm"]  # default

def stream_output_to_logger(proc: subprocess.Popen, logger: logging.Logger, prefix: str):
    def _reader(stream, log_fn):
        if not stream:
            return
        for line in iter(stream.readline, ''):
            line = line.rstrip("\n")
            if line:
                log_fn(f"[{prefix}] {line}")
        try:
            stream.close()
        except Exception:
            pass
    if proc.stdout:
        threading.Thread(target=_reader, args=(proc.stdout, logger.info), daemon=True).start()
    if proc.stderr:
        threading.Thread(target=_reader, args=(proc.stderr, logger.error), daemon=True).start()

def build_env(base: dict, extra: dict) -> dict:
    env = os.environ.copy()
    env.update(base or {})
    env.update(extra or {})
    return env

def http_get_ok(url: str, timeout: float = 2.5) -> bool:
    requests = _import_optional("requests")
    if not requests:
        # Fallback: simple socket connect only (less precise)
        try:
            host, port = url.split("//", 1)[1].split("/", 1)[0].split(":")
            return is_port_open(int(port), host="127.0.0.1")
        except Exception:
            return False
    try:
        r = requests.get(url, timeout=timeout)
        return 200 <= r.status_code < 400
    except Exception:
        return False

# ───────────────────────────────────────────────────────────────────
# Manager
# ───────────────────────────────────────────────────────────────────
class DashboardManager:
    def __init__(self, backend_port: int, frontend_port: int, dev_mode: bool, debug: bool):
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.dev_mode = dev_mode
        self.debug = debug

        self.logger = setup_logging(debug)
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.is_running = False

        # Signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ── Lifecycle ──────────────────────────────────────────────────
    def _signal_handler(self, signum, frame):
        self.logger.info("Received shutdown signal. Stopping dashboard...")
        self.stop()
        sys.exit(0)

    def display_header(self):
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

    # ── Checks ─────────────────────────────────────────────────────
    def check_dependencies(self) -> bool:
        self.logger.info(MESSAGES['checking_deps'])
        core = ["fastapi", "uvicorn", "requests"]  # minimal runtime
        # optional libs (warn only)
        optional = ["pandas", "numpy", "torch", "stable_baselines3", "MetaTrader5", "psutil"]

        missing_core = [m for m in core if _import_optional(m) is None]
        if missing_core:
            self.logger.error(f"Missing core packages: {missing_core}")
            self.logger.error("Install with: pip install fastapi uvicorn requests")
            return False

        missing_opt = [m for m in optional if _import_optional(m) is None]
        if missing_opt:
            self.logger.warning(f"Optional packages not found (not required for the dashboard to run): {missing_opt}")

        if self.dev_mode:
            # Ensure Node is present
            try:
                result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    self.logger.error("Node.js not found in PATH.")
                    return False
                self.logger.info(f"Node.js version: {result.stdout.strip()}")
            except Exception:
                self.logger.error("Node.js not found or not responding.")
                return False

        self.logger.info(MESSAGES['deps_available'])
        return True

    def check_ports(self) -> bool:
        self.logger.info(MESSAGES['checking_ports'])
        ports = [self.backend_port] + ([self.frontend_port] if self.dev_mode else [])
        for p in ports:
            if is_port_open(p):
                owner = who_owns_port(self.logger, p)
                if owner:
                    pid, cmd = owner
                    self.logger.error(MESSAGES['port_in_use'].format(port=p) + f" (pid {pid}: {cmd})")
                else:
                    self.logger.error(MESSAGES['port_in_use'].format(port=p))
                return False
        self.logger.info(MESSAGES['ports_available'])
        return True

    def check_frontend_build(self) -> bool:
        if self.dev_mode:
            return True
        self.logger.info(MESSAGES['frontend_build_check'])
        dist = Path("frontend/dist")
        if dist.exists() and any(dist.iterdir()):
            self.logger.info("Frontend build found")
            return True

        self.logger.info(MESSAGES['frontend_build_missing'])
        frontend_dir = Path("frontend")
        if not frontend_dir.exists():
            self.logger.error("frontend/ folder missing.")
            return False

        pm = detect_pkg_manager(frontend_dir)[0]
        try:
            if not (frontend_dir / "node_modules").exists():
                self.logger.info("Installing frontend dependencies...")
                subprocess.run([pm, "install"], cwd=str(frontend_dir), check=True, timeout=600)

            self.logger.info("Building frontend...")
            build_cmd = [pm, "run", "build"]
            subprocess.run(build_cmd, cwd=str(frontend_dir), check=True, timeout=600)
            self.logger.info(MESSAGES['frontend_build_complete'])
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Frontend build failed (exit {e.returncode}).")
            return False
        except Exception as e:
            self.logger.error(f"Frontend build failed: {e}")
            return False

    # ── Start/Stop ─────────────────────────────────────────────────
    def start_backend(self) -> bool:
        self.logger.info(MESSAGES['backend_starting'])
        # Compose uvicorn command (order of options is flexible)
        cmd = [
            sys.executable, "-m", "uvicorn",
            BACKEND_MODULE,
            "--host", "0.0.0.0",
            "--port", str(self.backend_port),
            "--log-level", "debug" if self.debug else "info"
        ]
        if self.debug:
            cmd.append("--reload")
            # Exclude directories that change during training to prevent unwanted restarts
            # These directories are modified by the training subprocess and InfoBus persistence
            cmd.extend([
                "--reload-exclude", "state",
                "--reload-exclude", "logs",
                "--reload-exclude", "data",
                "--reload-exclude", "checkpoints",
                "--reload-exclude", "models",
                "--reload-exclude", "metrics",
                "--reload-exclude", "*.json",
                "--reload-exclude", "*.log",
                "--reload-exclude", "*.jsonl",
            ])

        env = build_env(
            base={"PYTHONPATH": ".", "PYTHONUNBUFFERED": "1"},
            extra={
                # Common auth/CORS knobs your FastAPI app can read
                "TRADING_ENV": "development" if self.debug else "production",
                "DASHBOARD_FRONTEND_URL": f"http://localhost:{self.frontend_port}",
                "DASHBOARD_BACKEND_URL": f"http://localhost:{self.backend_port}",
                # Example flags your backend can use to relax cookies/CORS in dev:
                "ALLOW_DEV_ORIGINS": "true",
                "AUTH_COOKIE_SECURE": "false",  # localhost dev
            }
        )

        # Cross-platform: create a new process group so we can kill children
        popen_kwargs = {
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "universal_newlines": True
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["preexec_fn"] = os.setsid

        try:
            self.backend_process = subprocess.Popen(cmd, **popen_kwargs)
            stream_output_to_logger(self.backend_process, self.logger, "backend")

            # Wait for health
            health_url = f"http://localhost:{self.backend_port}{BACKEND_HEALTH_PATH}"
            start = time.time()
            while time.time() - start < BACKEND_STARTUP_TIMEOUT:
                if self.backend_process.poll() is not None:
                    self.logger.error("Backend process exited early.")
                    return False
                if http_get_ok(health_url, timeout=2.0):
                    self.logger.info(f"Backend healthy at {health_url}")
                    return True
                time.sleep(1.0)

            self.logger.error(f"Backend failed health check within {BACKEND_STARTUP_TIMEOUT}s.")
            return False

        except Exception as e:
            self.logger.error(f"Failed to start backend: {e}")
            return False

    def start_frontend(self) -> bool:
        if not self.dev_mode:
            return True

        self.logger.info(MESSAGES['frontend_starting'])
        front_dir = Path("frontend")
        if not front_dir.exists():
            self.logger.error("frontend/ folder missing.")
            return False

        pm = detect_pkg_manager(front_dir)[0]

        # Ensure deps
        try:
            if not (front_dir / "node_modules").exists():
                self.logger.info("Installing frontend dependencies...")
                subprocess.run([pm, "install"], cwd=str(front_dir), check=True, timeout=1200)
        except Exception as e:
            self.logger.error(f"Failed to install frontend deps: {e}")
            return False

        # Start dev server
        dev_cmd = {
            "npm": ["npm", "run", "dev", "--", "--port", str(self.frontend_port), "--host", "0.0.0.0"],
            "yarn": ["yarn", "dev", "--port", str(self.frontend_port), "--host", "0.0.0.0"],
            "pnpm": ["pnpm", "dev", "--port", str(self.frontend_port), "--host", "0.0.0.0"],
        }[pm]

        env = build_env({}, {"PORT": str(self.frontend_port)})

        popen_kwargs = {
            "cwd": str(front_dir),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "universal_newlines": True,
            "shell": os.name == "nt"  # Use shell=True on Windows for npm/yarn/pnpm
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["preexec_fn"] = os.setsid

        try:
            self.frontend_process = subprocess.Popen(dev_cmd, **popen_kwargs)
            stream_output_to_logger(self.frontend_process, self.logger, "frontend")

            # Wait for HTTP readiness (not just port open)
            url = f"http://localhost:{self.frontend_port}"
            start = time.time()
            while time.time() - start < FRONTEND_STARTUP_TIMEOUT:
                if self.frontend_process.poll() is not None:
                    self.logger.error("Frontend process exited early.")
                    return False
                if http_get_ok(url, timeout=2.0):
                    self.logger.info(f"Frontend reachable at {url}")
                    return True
                time.sleep(1.0)

            self.logger.error("Frontend failed to start within timeout window.")
            return False

        except Exception as e:
            self.logger.error(f"Failed to start frontend: {e}")
            return False

    def open_browser(self):
        url = f"http://localhost:{self.frontend_port}" if self.dev_mode else f"http://localhost:{self.backend_port}"
        self.logger.info(MESSAGES['opening_browser'])

        def delayed_open():
            time.sleep(OPEN_BROWSER_DELAY)
            try:
                webbrowser.open(url)
            except Exception as e:
                self.logger.warning(f"Could not open browser automatically: {e}")

        threading.Thread(target=delayed_open, daemon=True).start()

    def display_ready_message(self):
        print("\n" + "=" * 60)
        print(f"🎉 {MESSAGES['system_ready']}")
        print("=" * 60)
        if self.dev_mode:
            print(f"🖥️  Frontend Dashboard:  http://localhost:{self.frontend_port}")
            print(f"🔧 Backend API:         http://localhost:{self.backend_port}")
            print(f"📚 API Documentation:   http://localhost:{self.backend_port}/docs")
        else:
            print(f"🖥️  Dashboard:           http://localhost:{self.backend_port}")
            print(f"📚 API Documentation:   http://localhost:{self.backend_port}/docs")
            print(f"❤️  Health Check:       http://localhost:{self.backend_port}{BACKEND_HEALTH_PATH}")
        print("=" * 60)
        print("🛑 Press Ctrl+C to stop the dashboard")
        print("=" * 60)

    def start(self) -> bool:
        self.display_header()
        if not self.check_dependencies():
            return False
        if not self.check_ports():
            return False
        if not self.check_frontend_build():
            return False

        if not self.start_backend():
            self.stop()
            return False

        if not self.start_frontend():
            self.stop()
            return False

        self.is_running = True
        self.display_ready_message()
        self.open_browser()
        return True

    def _terminate_group(self, proc: subprocess.Popen, name: str):
        try:
            if proc.poll() is None:
                if os.name == "nt":
                    # Try to send CTRL_BREAK to the whole group, then terminate
                    try:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                        proc.wait(timeout=5)
                    except Exception:
                        pass
                    proc.terminate()
                else:
                    # Kill process group
                    import os as _os
                    _os.killpg(_os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.logger.warning(f"{name} did not stop gracefully; killing.")
            try:
                proc.kill()
            except Exception:
                pass
        except Exception as e:
            self.logger.error(f"Error stopping {name}: {e}")

    def stop(self):
        if not self.is_running and not (self.backend_process or self.frontend_process):
            return
        self.logger.info(MESSAGES['shutdown'])
        self.is_running = False

        if self.frontend_process:
            self._terminate_group(self.frontend_process, "frontend")
            self.frontend_process = None

        if self.backend_process:
            self._terminate_group(self.backend_process, "backend")
            self.backend_process = None

        self.logger.info("Dashboard stopped successfully")

    def wait(self):
        try:
            while self.is_running:
                if self.backend_process and self.backend_process.poll() is not None:
                    self.logger.error("Backend process died unexpectedly")
                    break
                if self.frontend_process and self.frontend_process.poll() is not None:
                    self.logger.error("Frontend process died unexpectedly")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Interrupt received")
        finally:
            self.stop()

# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI Trading System Dashboard Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start in development mode (default)
  %(prog)s --prod                   # Start in production mode (serve built frontend)
  %(prog)s --port 8080              # Custom backend port
  %(prog)s --dev --frontend-port 3001  # Custom frontend port (dev)
  %(prog)s --debug                  # Verbose logging
  %(prog)s --check                  # Only run pre-flight checks
        """
    )
    parser.add_argument('--port', '--backend-port', type=int, default=DEFAULT_BACKEND_PORT,
                        help=f'Backend server port (default: {DEFAULT_BACKEND_PORT})')
    parser.add_argument('--frontend-port', type=int, default=DEFAULT_FRONTEND_PORT,
                        help=f'Frontend port for dev mode (default: {DEFAULT_FRONTEND_PORT})')
    parser.add_argument('--dev', '--development', action='store_true', default=DEFAULT_DEV_MODE,
                        help='Run in development mode (separate frontend server)')
    parser.add_argument('--prod', '--production', action='store_true',
                        help='Run in production mode (frontend served by backend)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--no-browser', action='store_true', help='Do not auto-open browser')
    parser.add_argument('--check', action='store_true', help='Check system requirements and exit')
    return parser.parse_args()

def main():
    args = parse_arguments()
    dev_mode = args.dev and not args.prod

    dashboard = DashboardManager(
        backend_port=args.port,
        frontend_port=args.frontend_port,
        dev_mode=dev_mode,
        debug=args.debug
    )

    if args.check:
        dashboard.display_header()
        ok = dashboard.check_dependencies() and dashboard.check_ports() and dashboard.check_frontend_build()
        print("\n✅ All system checks passed!" if ok else "\n❌ System checks failed. See log above.")
        return 0 if ok else 1

    if args.no_browser:
        dashboard.open_browser = lambda: None

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
