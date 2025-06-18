#!/usr/bin/env python3

import os
import sys
import signal
import threading
import subprocess
import argparse
import logging

# Ensure Windows console can print Unicode (e.g. “”)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ───── Your core imports ────────────────────────────────────────────
from live.api import start_api
from live.live_trading import LiveTradingSystem, Config   # adjust import path if needed

# ───── Logging setup ────────────────────────────────────────────────
logger = logging.getLogger("run_all")
logger.setLevel(logging.INFO)
if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh.setFormatter(fmt)
    logger.addHandler(sh)

# ───── CLI args ────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["1","2","3"], default="2",
                   help="Which RL agent to load: 1=PPO, 2=SAC, 3=TD3")
    p.add_argument("--sleep", type=int, default=5,
                   help="Seconds between live trading steps")
    return p.parse_args()

# ───── Entrypoint ───────────────────────────────────────────────────
def main():
    args = parse_args()
    # override the Config used by LiveTradingSystem
    Config.AGENT_ID   = args.agent
    Config.SLEEP_SECS = args.sleep

    # 1) build your live-trading system
    system = LiveTradingSystem()

    # 2) start the FastAPI server in a daemon thread
    threading.Thread(
        target=lambda: start_api(system),
        daemon=True
    ).start()
    logger.info("✅ Status API launching at http://0.0.0.0:8000")

    # 3) start the Streamlit UI in a subprocess
    ui_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "ui", "app.py"))
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", ui_script,
        "--server.port", "8501",
    ], cwd=os.path.dirname(ui_script))
    logger.info("✅ Streamlit UI launching at http://localhost:8501")

    # 4) graceful shutdown hooks
    def shutdown(signum, frame):
        logger.info("🛑 Shutting down…")
        try:
            system.connector.disconnect()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 5) hand off to your live loop
    system.run()


if __name__ == "__main__":
    main()
