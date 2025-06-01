#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import signal
import numpy as np
import pandas as pd

# Set TensorFlow logging level (suppress warnings)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ==== Imports ====
from live_connector import LiveDataConnector
from envs.env import EnhancedTradingEnv
from stable_baselines3 import PPO, SAC, TD3
from modules.strategy.voting import StrategyArbiter
from modules.position.position import PositionManager
from modules.risk.risk_controller import DynamicRiskController
from modules.memory.memory import MistakeMemory

# ==== Logging Config (no duplicates) ====
logger = logging.getLogger("live_trading")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("live_trading.log", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

# ==== Config ====
class Config:
    INSTRUMENTS = ["EURUSD", "XAUUSD"]
    TIMEFRAMES  = ["H1", "H4", "D1"]
    HIST_BARS   = 1000
    AGENT_MAP   = {
        "1": ("PPO", "models/ppo_final_model.zip"),
        "2": ("SAC", "models/sac_final_model.zip"),
        "3": ("TD3", "models/td3_final_model.zip"),
    }
    AGENT_ID    = "2"  # Default to PPO
    SLEEP_SECS  = 5

# ==== Agent Factory ====
class AgentFactory:
    @staticmethod
    def load(agent_name, model_path):
        if agent_name == "PPO":
            return PPO.load(model_path, device="cpu")
        elif agent_name == "SAC":
            return SAC.load(model_path, device="cpu")
        elif agent_name == "TD3":
            return TD3.load(model_path, device="cpu")
        else:
            raise ValueError(f"Unsupported agent: {agent_name}")

# ==== Live Trading System ====
class LiveTradingSystem:
    def __init__(self):
        # 1) MT5 & data
        self.connector = LiveDataConnector(
            instruments=Config.INSTRUMENTS,
            timeframes=Config.TIMEFRAMES
        )
        self.connector.connect()

        # 2) history + env
        hist = self.connector.get_historical_data(n_bars=Config.HIST_BARS)
        self.env = EnhancedTradingEnv(
            data_dict=hist,
            initial_balance=10_000.0,
            max_steps=10_000_000
        )
        setattr(self.env, "live_mode", True)

        # 3) load agent
        agent_name, model_path = Config.AGENT_MAP[Config.AGENT_ID]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model at {model_path}")
        self.model = AgentFactory.load(agent_name, model_path)
        logger.info(f"{agent_name} agent loaded from {model_path}")

        # 4) ensemble
        self.arbiter = StrategyArbiter(
            members=[
                PositionManager(self.env.initial_balance),
                DynamicRiskController({"vol_history_len":100,"dd_threshold":0.2,"vol_ratio_threshold":1.5}),
                MistakeMemory(10, 3)
            ],
            init_weights=np.array([0.4,0.4,0.2]),
            action_dim=self.env.action_space.shape[0]
        )

    def append_new_bar(self):
        new = self.connector.get_historical_data(n_bars=1)
        for sym_raw in Config.INSTRUMENTS:
            sym_internal = sym_raw[:3] + "/" + sym_raw[3:]  # "EURUSD" → "EUR/USD"
            for tf in Config.TIMEFRAMES:
                try:
                    old = self.env.data[sym_internal][tf]
                    bar = new[sym_internal][tf]
                    df  = pd.concat([old, bar]).iloc[-Config.HIST_BARS:]
                    self.env.data[sym_internal][tf] = df
                    latest = bar.iloc[-1].to_dict()
                    logger.info(f"Appended bar [{sym_internal} {tf}]: {latest}")
                except KeyError as e:
                    logger.error(f"append_new_bar failed for {sym_internal}/{tf}: {e}")

    def run(self):
        logger.info("Starting live loop")
        obs, _ = self.env.reset()

        # 🩹 Pad observation if needed
        if obs.shape[0] < self.model.observation_space.shape[0]:
            pad = self.model.observation_space.shape[0] - obs.shape[0]
            obs = np.pad(obs, (0, pad), constant_values=0)

        while True:
            try:
                self.append_new_bar()

                if obs.shape[0] < self.model.observation_space.shape[0]:
                    pad = self.model.observation_space.shape[0] - obs.shape[0]
                    obs = np.pad(obs, (0, pad), constant_values=0)

                a, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(a)

                logger.info(
                    f"Live Step --> Balance={info['balance']} "
                    f"| PnL={info['pnl']} | DD={info['drawdown']:.2%}"
                )

                if done:
                    logger.warning("Episode ended, resetting...")
                    obs, _ = self.env.reset()
                    if obs.shape[0] < self.model.observation_space.shape[0]:
                        pad = self.model.observation_space.shape[0] - obs.shape[0]
                        obs = np.pad(obs, (0, pad), constant_values=0)

                time.sleep(Config.SLEEP_SECS)

            except KeyboardInterrupt:
                logger.info("Live trading interrupted by user")
                break
            except Exception as e:
                logger.exception(f"update failed: {e}")
                time.sleep(Config.SLEEP_SECS)
        self.connector.disconnect()

# ==== Argparse CLI ====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["1", "2", "3"], default="1", help="1=PPO, 2=SAC, 3=TD3")
    parser.add_argument("--sleep", type=int, default=5, help="Seconds between steps")
    return parser.parse_args()

# ==== Main Entrypoint ====
def main():
    args = parse_args()
    Config.AGENT_ID = args.agent
    Config.SLEEP_SECS = args.sleep

    # 1) Create the trading system
    system = LiveTradingSystem()

    # 2) Launch the FastAPI status server in the background
    from live.api import start_api
    import threading
    threading.Thread(target=lambda: start_api(system), daemon=True).start()

    # 3) Graceful shutdown support
    def graceful_exit(signum, frame):
        logger.info("Received exit signal. Cleaning up...")
        system.connector.disconnect()
        exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    # 4) Run trading loop
    system.run()

if __name__ == "__main__":
    main()
