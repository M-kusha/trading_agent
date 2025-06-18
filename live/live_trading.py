#!/usr/bin/env python3

import os
import sys
# Ensure Windows console can print Unicode (e.g. “”)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import time
import logging
import argparse
import signal

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# ==== Imports ====
from live.live_connector import LiveDataConnector
from envs.env import EnhancedTradingEnv
from stable_baselines3 import PPO, SAC, TD3
from modules.strategy.voting import StrategyArbiter
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
    AGENT_ID    = "2"  # Default to SAC
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
        # 1) Initialize and authenticate MT5
        if not mt5.initialize():
            raise RuntimeError("MetaTrader5 initialize() failed")
        account_info = mt5.account_info()
        if account_info is None or not hasattr(account_info, "balance"):
            raise RuntimeError("Could not retrieve account_info from MT5")
        live_balance = float(account_info.balance)
        logger.info(f"Connected to MT5 – live balance={live_balance:.2f}")

        # 2) Connect data feed and fetch history
        self.connector = LiveDataConnector(
            instruments=Config.INSTRUMENTS,
            timeframes=Config.TIMEFRAMES
        )
        self.connector.connect()
        hist = self.connector.get_historical_data(n_bars=Config.HIST_BARS)

        # 3) Create environment with real balance
        self.env = EnhancedTradingEnv(
            data_dict=hist,
            initial_balance=live_balance,
            max_steps=10_000_000
        )
        self.env.live_mode = True

        # 4) Load chosen RL agent
        agent_name, model_path = Config.AGENT_MAP[Config.AGENT_ID]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model at {model_path}")
        self.model = AgentFactory.load(agent_name, model_path)
        logger.info(f"{agent_name} agent loaded from {model_path}")

        # 5) Ensemble arbiter – reuse the SAME instances from the env
        self.arbiter = StrategyArbiter(
            members=[
                self.env.position_manager,
                self.env.risk_controller,
                self.env.mistake_memory
            ],
            init_weights=np.array([0.4, 0.4, 0.2], np.float32),
            action_dim=self.env.action_space.shape[0]
        )

    def append_new_bar(self):
        new = self.connector.get_historical_data(n_bars=1)
        for sym_raw in Config.INSTRUMENTS:
            sym_internal = sym_raw[:3] + "/" + sym_raw[3:]
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

        while True:
            try:
                self.append_new_bar()

                # ── ENSURE obs matches exactly what the model expects ────────
                exp_dim = self.model.observation_space.shape[0]
                obs = np.asarray(obs, dtype=np.float32)

                if obs.shape[0] > exp_dim:
                    obs = obs[:exp_dim]
                elif obs.shape[0] < exp_dim:
                    pad_width = exp_dim - obs.shape[0]
                    obs = np.pad(obs, (0, pad_width), mode="constant", constant_values=0.0)
                # ────────────────────────────────────────────────────────────────

                # 1) get actions
                action, _     = self.model.predict(obs, deterministic=True)
                raw_action, _ = self.model.predict(obs, deterministic=True)
                arbiter_action = self.arbiter.propose(obs)

                # 2) blend & clip
                action = np.clip(
                    raw_action + arbiter_action,
                    self.env.action_space.low,
                    self.env.action_space.high
                )

                # 3) step the environment
                obs, reward, done, truncated, info = self.env.step(action)

                logger.info(
                    f"Live Step  Balance={info['balance']:.2f} | "
                    f"PnL={info['pnl']} | DD={info['drawdown']:.2%}"
                )

                if done:
                    logger.warning("Episode ended; resetting environment")
                    obs, _ = self.env.reset()

                time.sleep(Config.SLEEP_SECS)

            except KeyboardInterrupt:
                logger.info("Live trading interrupted by user")
                break
            except Exception as e:
                logger.exception(f"Live loop error: {e}")
                time.sleep(Config.SLEEP_SECS)

        self.connector.disconnect()

# ==== CLI & Entrypoint ====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["1","2","3"], default="2",
                        help="1=PPO, 2=SAC, 3=TD3")
    parser.add_argument("--sleep", type=int, default=5,
                        help="Seconds between steps")
    return parser.parse_args()

def main():
    args = parse_args()
    Config.AGENT_ID   = args.agent
    Config.SLEEP_SECS = args.sleep

    system = LiveTradingSystem()

    # Optional: start status API
    from live.api import start_api
    import threading
    threading.Thread(target=lambda: start_api(system), daemon=True).start()

    # Graceful shutdown
    def graceful_exit(signum, frame):
        logger.info("Received exit signal; cleaning up")
        system.connector.disconnect()
        sys.exit(0)
    signal.signal(signal.SIGINT,  graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    system.run()

if __name__ == "__main__":
    main()
