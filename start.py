#!/usr/bin/env python3
"""
Helper script to launch backend (FastAPI/uvicorn) and frontend (Vite) together.

Usage (from repo root):
    python start.py
Stop with Ctrl+C.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional
import shutil


ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
UVICORN_EXE = ROOT / ".venv" / "Scripts" / "uvicorn.exe"


def _find_python() -> str:
    """Prefer the project venv python; fall back to current interpreter."""
    if VENV_PY.exists():
        return str(VENV_PY)
    return sys.executable


def _find_uvicorn_cmd() -> List[str]:
    """Build the uvicorn command, using the venv binary when present."""
    # Common reload exclusions to prevent restarts when training modifies data files
    reload_excludes = [
        "--reload-exclude", "state",
        "--reload-exclude", "logs",
        "--reload-exclude", "data",
        "--reload-exclude", "checkpoints",
        "--reload-exclude", "models",
        "--reload-exclude", "metrics",
        "--reload-exclude", "*.json",
        "--reload-exclude", "*.log",
        "--reload-exclude", "*.jsonl",
    ]
    
    if UVICORN_EXE.exists():
        return [
            str(UVICORN_EXE),
            "main:app",
            "--app-dir",
            "backend",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ] + reload_excludes
    return [
        _find_python(),
        "-m",
        "uvicorn",
        "main:app",
        "--app-dir",
        "backend",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--reload",
    ] + reload_excludes

def _find_npm() -> str:
    """Locate npm (works when PATH misses it in some shells)."""
    for candidate in ("npm", "npm.cmd", "npm.exe"):
        path = shutil.which(candidate)
        if path:
            return path
    fallback = Path(r"C:\Program Files\nodejs\npm.cmd")
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("npm not found; install Node.js or add npm to PATH")


def _frontend_cmd() -> List[str]:
    return [_find_npm(), "run", "dev", "--", "--host", "--port", "5173"]


def start_process(cmd: List[str], cwd: Optional[Path] = None, extra_env: Optional[dict] = None) -> subprocess.Popen:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"[START] {' '.join(cmd)} (cwd={cwd or ROOT})")
    return subprocess.Popen(cmd, cwd=str(cwd or ROOT), env=env)


def main() -> int:
    procs: List[subprocess.Popen] = []
    try:
        # Backend
        backend_env = {"PYTHONPATH": str(ROOT)}
        procs.append(start_process(_find_uvicorn_cmd(), cwd=ROOT, extra_env=backend_env))

        time.sleep(1)  # small stagger

        # Frontend
        procs.append(start_process(_frontend_cmd(), cwd=ROOT / "frontend"))

        print("[INFO] Backend: http://localhost:8000  |  Frontend: http://localhost:5173")
        print("[INFO] Press Ctrl+C to stop both.")

        # Wait for any process to exit
        while True:
            for p in list(procs):
                ret = p.poll()
                if ret is not None:
                    cmd_str = " ".join(map(str, p.args if isinstance(p.args, (list, tuple)) else [p.args]))
                    print(f"[EXIT] {cmd_str} (code={ret})")
                    raise KeyboardInterrupt
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down processes...")
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        return 0
    except FileNotFoundError as e:
        print(f"[ERROR] Command not found: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
