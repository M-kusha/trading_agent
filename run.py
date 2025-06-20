#!/usr/bin/env python3

import os
import sys
# Ensure Windows console can print Unicode
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
from envs.env import EnhancedTradingEnv, TradingConfig
from stable_baselines3 import PPO, SAC, TD3
from modules.strategy.voting import StrategyArbiter
from modules.risk.risk_controller import DynamicRiskController
from modules.memory.memory import MistakeMemory

# ==== Logging Config ====
logger = logging.getLogger("live_trading")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("live_trading.log", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
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
    AGENT_ID    = "2"  # default
    SLEEP_SECS  = 5

# ==== Agent Factory ====
class AgentFactory:
    @staticmethod
    def load(agent_name, path):
        if agent_name == "PPO":
            return PPO.load(path, device="cpu")
        if agent_name == "SAC":
            return SAC.load(path, device="cpu")
        if agent_name == "TD3":
            return TD3.load(path, device="cpu")
        raise ValueError(f"Unsupported agent {agent_name}")

# ==== Live Trading System ====
class LiveTradingSystem:
    def __init__(self):
        # 1) MT5 init
        if not mt5.initialize():
            raise RuntimeError("MT5 initialize() failed")
        ai = mt5.account_info()
        if not ai or not hasattr(ai, "balance"):
            raise RuntimeError("Could not fetch MT5 account_info")
        balance = float(ai.balance)
        logger.info(f"Connected to MT5 – live balance={balance:.2f}")

        # 2) Data feed
        self.connector = LiveDataConnector(
            instruments=Config.INSTRUMENTS,
            timeframes=Config.TIMEFRAMES
        )
        self.connector.connect()
        hist = self.connector.get_historical_data(n_bars=Config.HIST_BARS)

        # 3) Env with live balance
        cfg = TradingConfig(
            initial_balance=balance,
            max_steps=10_000_000,
            live_mode=True
        )
        self.env = EnhancedTradingEnv(data_dict=hist, config=cfg)

        # 4) Load RL model
        name, model_path = Config.AGENT_MAP[Config.AGENT_ID]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model at {model_path}")
        self.model = AgentFactory.load(name, model_path)
        logger.info(f"{name} agent loaded from {model_path}")

        # 5) Arbiter
        self.arbiter = StrategyArbiter(
            members=[
                self.env.position_manager,
                self.env.risk_controller,
                self.env.mistake_memory
            ],
            init_weights=np.array([0.4, 0.4, 0.2], np.float32),
            action_dim=self.env.action_space.shape[0]
        )

    @staticmethod
    def format_symbol(sym: str) -> str:
        if "/" in sym:
            return sym
        return sym[:3] + "/" + sym[3:] if len(sym) == 6 else sym

    def append_new_bar(self):
        new = self.connector.get_historical_data(n_bars=1)
        for raw in Config.INSTRUMENTS:
            inst = self.format_symbol(raw)
            for tf in Config.TIMEFRAMES:
                try:
                    old = self.env.data[inst][tf]
                    bar = new[inst][tf]
                    df  = pd.concat([old, bar]).iloc[-Config.HIST_BARS:]
                    self.env.data[inst][tf] = df
                    logger.info(f"Appended bar [{inst} {tf}]: {bar.iloc[-1].to_dict()}")
                except KeyError as e:
                    logger.error(f"append_new_bar failed for {inst}/{tf}: {e}")

    def run(self):
        logger.info("Starting live loop")
        obs, _ = self.env.reset()

        while True:
            try:
                self.append_new_bar()

                # ── prepare obs ────────────────────────────────────────────────
                exp_dim = self.model.observation_space.shape[0]
                obs = np.asarray(obs, dtype=np.float32)
                if obs.shape[0] > exp_dim:
                    obs = obs[:exp_dim]
                elif obs.shape[0] < exp_dim:
                    obs = np.pad(obs, (0, exp_dim - obs.shape[0]), mode="constant")
                # ────────────────────────────────────────────────────────────────

                # 1) model + arbiter actions
                model_action, _ = self.model.predict(obs, deterministic=True)
                arbiter_action = self.arbiter.propose(obs)

                # 2) blend & clip
                final_action = np.clip(
                    model_action + arbiter_action,
                    self.env.action_space.low,
                    self.env.action_space.high
                )

                # 3) step the env
                obs, _, done, _, info = self.env.step(final_action)

                # 4) safe logging
                bal      = info.get("balance", self.env.balance)
                pnl      = info.get("pnl",     0.0)
                drawdown = info.get("drawdown", self.env.current_drawdown)
                logger.info(
                    f"Live Step  Balance={bal:.2f} | "
                    f"PnL={pnl:.2f} | DD={drawdown:.2%}"
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


# ==== CLI & Entrypoint ====
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["1","2","3"], default="2")
    p.add_argument("--sleep", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    Config.AGENT_ID   = args.agent
    Config.SLEEP_SECS = args.sleep

    system = LiveTradingSystem()

    # optional API server
    from live.api import start_api
    import threading
    threading.Thread(target=lambda: start_api(system), daemon=True).start()

    def graceful_exit(sig, frame):
        logger.info("Shutting down…")
        system.connector.disconnect()
        if hasattr(system.env, "close"):
            system.env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    system.run()

if __name__ == "__main__":
    main()
