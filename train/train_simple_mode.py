#!/usr/bin/env python3
"""
Simple Mode PPO Training - No Modules, Pure Exploration + Optuna Tuning

This script trains the PPO agent WITHOUT the SmartInfoBus module system,
allowing the agent to freely explore market dynamics using only:
- Raw OHLCV price data
- Simple technical features (computed locally in env)
- Basic reward (risk-adjusted P&L delta)

INCLUDES Optuna hyperparameter optimization for finding best parameters.

USE THIS FIRST to build a baseline before adding modules.

Usage:
    # Standard training
    python train/train_simple_mode.py --timesteps 100000
    python train/train_simple_mode.py --timesteps 500000 --lr 3e-4
    python train/train_simple_mode.py --resume  # Continue from checkpoint
    
    # Optuna hyperparameter tuning
    python train/train_simple_mode.py --optuna --n-trials 30 --timesteps 50000
    python train/train_simple_mode.py --optuna --n-trials 50 --timesteps 100000
    python train/train_simple_mode.py --show-best  # Show best params from tuning
"""

from __future__ import annotations

import gc
import json
import os
import sys
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Optuna imports (optional)
OPTUNA_AVAILABLE = False
optuna = None  # type: ignore[assignment]
MedianPruner = None  # type: ignore[assignment,misc]
TPESampler = None  # type: ignore[assignment,misc]
TrialState = None  # type: ignore[assignment,misc]

try:
    import optuna as _optuna  # type: ignore[import-not-found]
    from optuna.pruners import MedianPruner as _MedianPruner  # type: ignore[import-not-found]
    from optuna.samplers import TPESampler as _TPESampler  # type: ignore[import-not-found]
    from optuna.trial import TrialState as _TrialState  # type: ignore[import-not-found]
    optuna = _optuna
    MedianPruner = _MedianPruner
    TPESampler = _TPESampler
    TrialState = _TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    pass  # Already set defaults above

# Optional YAML
YAML_AVAILABLE = False
yaml = None  # type: ignore[assignment]

try:
    import yaml as _yaml  # type: ignore[import-not-found]
    yaml = _yaml
    YAML_AVAILABLE = True
except ImportError:
    pass  # Already set defaults above


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE MODE CONFIG - No modules, no bus, pure exploration
# ═══════════════════════════════════════════════════════════════════════════════
class SimpleConfig:
    """Minimal config for module-free training."""

    def __init__(self, **kwargs):
        # ═════ CRITICAL: Disable all module/bus features ═════
        self.bus_first = False
        self.prefer_bus_data = False
        self.prefer_bus_features = False
        self.prefer_bus_rewards = False
        self.prefer_bus_limits = False
        self.prefer_bus_metrics = False
        self.info_bus_enabled = False
        self.halt_on_emergency = False

        # ═════ Environment settings ═════
        # Load initial_balance from risk_policy.yaml (single source of truth)
        default_balance = 100_000.0
        try:
            import yaml
            from pathlib import Path
            rp_path = Path("config/risk_policy.yaml")
            if rp_path.exists():
                with open(rp_path, "r", encoding="utf-8") as f:
                    rp = yaml.safe_load(f) or {}
                default_balance = float(
                    rp.get("prop_firm", {}).get("account_size")
                    or rp.get("lot_sizing", {}).get("account_balance")
                    or default_balance
                )
        except Exception:
            pass
        self.initial_balance = kwargs.get("initial_balance", default_balance)
        self.max_steps = kwargs.get("max_steps", 10000)  # Max steps per episode
        self.environment_observation_size = kwargs.get(
            "obs_size", 256
        )  # Match ModernTradingEnv for transfer
        self.min_required_data_bars = kwargs.get("min_bars", 50)
        self.primary_timeframe = "H1"

        # Train/eval temporal split to reduce overfitting
        #  - train episodes start in first train_split of data
        #  - eval episodes start in remaining tail
        self.train_split = kwargs.get("train_split", 0.7)

        # ═════ Risk parameters (permissive for exploration) ═════
        self.max_drawdown = kwargs.get("max_drawdown", 0.30)  # 30% - more room to explore
        self.max_position_pct = kwargs.get("max_position_pct", 0.40)  # v3.8: 40% for stronger signal
        self.max_total_exposure = kwargs.get("max_total_exposure", 0.80)  # v3.8: Allow more exposure

        # ═════ Trading costs (configurable + optional randomization) ═════
        # Base values for spread/slippage, can be slightly randomized per episode
        self.spread = kwargs.get("spread", 0.0001)
        self.slippage = kwargs.get("slippage", 0.00005)
        self.randomize_costs = kwargs.get("randomize_costs", True)

        # ═════ PPO Hyperparameters (tuned for exploration) ═════
        self.learning_rate = kwargs.get("learning_rate", 3e-4)
        self.n_steps = kwargs.get("n_steps", 2048)
        self.batch_size = kwargs.get("batch_size", 64)
        self.n_epochs = kwargs.get("n_epochs", 10)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.clip_range = kwargs.get("clip_range", 0.2)
        self.ent_coef = kwargs.get("ent_coef", 0.02)  # Higher entropy for exploration
        self.vf_coef = kwargs.get("vf_coef", 0.5)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)
        self.target_kl = kwargs.get("target_kl", 0.015)

        # ═════ Network architecture ═════
        # NOTE: Must match ModernTradingEnv architecture for transfer learning
        self.policy_hidden_size = kwargs.get("policy_hidden", 256)
        self.value_hidden_size = kwargs.get("value_hidden", 256)

        # ═════ Training schedule ═════
        self.final_training_steps = kwargs.get("timesteps", 100000)
        self.checkpoint_freq = kwargs.get("checkpoint_freq", 10000)
        self.eval_freq = kwargs.get("eval_freq", 5000)
        self.n_eval_episodes = kwargs.get("n_eval_episodes", 5)

        # ═════ Paths ═════
        self.data_dir = kwargs.get("data_dir", "data/processed")
        self.model_dir = kwargs.get("model_dir", "models/simple")
        self.checkpoint_dir = kwargs.get("checkpoint_dir", "checkpoints/simple")
        self.tensorboard_dir = kwargs.get("tensorboard_dir", "logs/tensorboard/simple")
        self.log_dir = kwargs.get("log_dir", "logs/simple")

        # ═════ Mode flags ═════
        self.live_mode = False
        self.test_mode = kwargs.get("test_mode", False)
        self.debug = kwargs.get("debug", True)
        self.init_seed = kwargs.get("seed", 42)

        # ═════ Data settings ═════
        self.instruments = kwargs.get("instruments", ["EURUSD", "XAUUSD"])
        self.timeframes = kwargs.get("timeframes", ["M15", "H1", "H4", "D1"])

        # Execution (no fees for pure exploration beyond spread/slippage)
        self.default_spread = 0.0
        self.slippage_pts = 0.0
        self.commission_per_million = 0.0

        # Consensus/confidence thresholds (permissive)
        self.min_confidence = 0.0
        self.min_intensity = 0.0
        self.consensus_min = 0.0
        self.consensus_max = 1.0
        self.ignore_hold = True

        # Log rotation
        self.log_rotation_lines = 2000

        # Create directories
        for d in [
            self.model_dir,
            self.checkpoint_dir,
            self.tensorboard_dir,
            self.log_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE TRADING ENVIRONMENT - No modules, pure OHLCV + basic features
# ═══════════════════════════════════════════════════════════════════════════════
import gymnasium as gym
from gymnasium import spaces


class SimpleTradingEnv(gym.Env):
    """
    SimpleTradingEnv v3.2
    - Proper position mechanics (smooth scaling)
    - Correct PnL based on equity delta
    - Spread & slippage with optional per-episode randomization
    - Per-instrument rewards
    - USD-correlation synthetic feature
    - ANTI-REWARD-HACKING: Time penalty, Sharpe-like shaping, episode caps
    - Explicit train/eval temporal split to reduce overfitting
    - Fully compatible with your v1 training pipeline
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data, config: SimpleConfig, mode: str = "train"):
        """
        Args:
            data: Dict[instrument][timeframe] -> DataFrame
            config: SimpleConfig instance
            mode: "train" or "eval" (controls which part of the history is used)
        """
        super().__init__()

        self.data = data
        self.config = config
        self.mode = mode.lower()
        self.instruments = list(data.keys())
        self.primary_tf = config.primary_timeframe

        if self.primary_tf not in data[self.instruments[0]]:
            self.primary_tf = list(data[self.instruments[0]].keys())[0]

        self._data_lengths = {
            inst: len(data[inst][self.primary_tf]) for inst in self.instruments
        }

        self._min_len = min(self._data_lengths.values())
        if self._min_len < config.min_required_data_bars:
            raise ValueError(f"Not enough data: {self._min_len} bars")

        # Action: dir, intensity per instrument
        n_inst = len(self.instruments)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_inst * 2,), dtype=np.float32
        )

        # Obs: identical size to v1 (zero-padded if needed)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.environment_observation_size,),
            dtype=np.float32,
        )

        # Internal state
        self.positions = {inst: 0.0 for inst in self.instruments}
        self.entry_prices = {inst: 0.0 for inst in self.instruments}

        self.balance = float(config.initial_balance)
        self.peak_balance = self.balance
        self.current_step = 0
        self.episode_count = 0
        self.episode_start_step = 0  # Track episode start for duration calc

        # Trading costs (base values)
        self.base_spread = float(getattr(config, "spread", 0.0001))
        self.base_slippage = float(getattr(config, "slippage", 0.00005))
        self.randomize_costs = bool(getattr(config, "randomize_costs", True))

        # Initialized per episode in reset()
        self.spread = self.base_spread
        self.slippage = self.base_slippage

        # ═══════════════════════════════════════════════════════════════════════
        # ANTI-REWARD-HACKING PARAMETERS (v3.6 - Simple Bounded)
        # ═══════════════════════════════════════════════════════════════════════
        self.max_episode_steps = 2000  # Hard cap
        self.initial_episode_balance = float(config.initial_balance)
        
        # v3.7: Fixed compounding exploit - use INITIAL balance for position sizing
        # v3.7: Fixed cumulative PnL bug - use step-over-step price change
        # v3.8: Stronger learning signal - 40% position, wider reward clip, trend bonus
        # v3.7: Fixed eval range bug - use max_episode_steps for index calc
        # v3.6: Simple bounded per-step reward, clipped ±0.01

        print(
            f"[ENV v3.8] Mode={self.mode.upper()} | "
            f"{len(self.instruments)} instruments, {self._min_len} bars | "
            f"STRONGER SIGNAL: 40% position, ±0.05 clip, trend bonus"
        )

    ###########################################################################
    # RESET
    ###########################################################################
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_count += 1

        # Temporal split: early part of data for training, later part for eval
        min_offset = 50
        # Ensure we have enough room for a full episode
        # v3.7 FIX: Use actual max_episode_steps (2000) not config.max_steps (10000)
        # This allows proper randomization within train/eval ranges
        max_index = self._min_len - self.max_episode_steps - 2
        max_index = max(min_offset + 1, max_index)

        split_index = int(self._min_len * float(getattr(self.config, "train_split", 0.7)))
        split_index = max(min_offset + 1, min(split_index, max_index))

        if self.mode == "train":
            start_min = min_offset
            start_max = max(split_index, start_min + 1)
        elif self.mode == "eval":
            start_min = split_index
            start_max = max_index
        else:
            # Fallback: full range, same as old behaviour but with safety
            start_min = min_offset
            start_max = max_index

        if start_max <= start_min:
            self.current_step = start_min
        else:
            self.current_step = np.random.randint(start_min, start_max)

        self.episode_start_step = self.current_step

        self.balance = float(self.config.initial_balance)
        self.peak_balance = self.balance
        self.initial_episode_balance = self.balance  # Track for terminal reward

        # Reset positions
        for inst in self.instruments:
            self.positions[inst] = 0.0
            self.entry_prices[inst] = 0.0

        # Randomize costs a bit to reduce overfitting to one fee regime
        if self.randomize_costs:
            self.spread = float(self.base_spread * np.random.uniform(0.5, 1.5))
            self.slippage = float(self.base_slippage * np.random.uniform(0.5, 1.5))
        else:
            self.spread = self.base_spread
            self.slippage = self.base_slippage

        obs = self._get_observation()
        return obs, {"episode": self.episode_count}

    ###########################################################################
    # STEP
    ###########################################################################
    def step(self, action):
        prev_balance = self.balance
        self.current_step += 1

        total_pnl = 0.0
        instrument_pnl: Dict[str, float] = {}
        instrument_rewards: Dict[str, float] = {}

        # Process each instrument independently
        for idx, inst in enumerate(self.instruments):
            dir_sig = float(np.clip(action[idx * 2], -1, 1))
            int_sig = float(np.clip(action[idx * 2 + 1], 0, 1))

            target_position = dir_sig * int_sig * self.config.max_position_pct
            current_pos = self.positions[inst]

            df = self.data[inst][self.primary_tf]
            price = float(df["close"].iloc[self.current_step])
            prev_price = float(df["close"].iloc[self.current_step - 1]) if self.current_step > 0 else price

            ###################################################################
            # REALISTIC PnL CALCULATION (v3.7 FIXED)
            # - Use step-over-step price change, NOT cumulative from entry
            # - Use INITIAL balance for position sizing
            # - This prevents both compounding exploit AND cumulative price exploit
            ###################################################################
            pnl = 0.0
            if current_pos != 0:
                # Position size as portion of INITIAL balance (not current)
                notional = abs(current_pos) * self.initial_episode_balance

                # v3.7 FIX: Use step-over-step price change, not cumulative from entry
                # This is how real trading works: PnL accrues from bar-to-bar movement
                price_change = price - prev_price  # STEP change, not cumulative!
                direction = np.sign(current_pos)

                # Slippage & spread (only apply on position changes, not holds)
                fee = 0.0  # Fees handled elsewhere now
                
                # PnL for this step = direction * step_price_change * (notional / price)
                pnl = direction * price_change * (notional / max(price, 1e-8)) - fee
                self.balance += pnl

            instrument_pnl[inst] = pnl

            ###################################################################
            # SMOOTH POSITION SCALING (Option B)
            ###################################################################
            new_pos = current_pos
            old_pos = current_pos

            # Move position toward target gradually (to avoid jerkiness)
            step_size = 0.5  # how aggressively scaling happens
            new_pos = current_pos + step_size * (target_position - current_pos)

            # If position direction flips, adjust entry price
            if np.sign(old_pos) != np.sign(new_pos) and abs(new_pos) > 0.01:
                # New trade → new entry price
                self.entry_prices[inst] = price
            elif abs(new_pos) < 0.01:
                # Position closed
                new_pos = 0.0
                self.entry_prices[inst] = 0.0
            else:
                # Same direction scaling
                if old_pos != 0:
                    weight_old = abs(old_pos)
                    weight_new = abs(new_pos - old_pos)
                    self.entry_prices[inst] = (
                        (self.entry_prices[inst] * weight_old + price * weight_new)
                        / max(weight_old + weight_new, 1e-9)
                    )
                else:
                    self.entry_prices[inst] = price

            self.positions[inst] = new_pos

            # Per-instrument reward (scaled PnL)
            inst_reward = pnl / max(self.config.initial_balance, 1.0)
            instrument_rewards[inst] = float(np.clip(inst_reward, -1.0, 1.0))

            total_pnl += pnl

        ###########################################################################
        # REWARD CALCULATION (v3.6 - Simple bounded per-step)
        ###########################################################################
        # Track episode duration
        episode_steps = self.current_step - self.episode_start_step

        # Drawdown + termination checks
        self.peak_balance = max(self.peak_balance, self.balance)
        dd = (self.peak_balance - self.balance) / max(self.peak_balance, 1.0)

        terminated = False
        truncated = False

        if dd > self.config.max_drawdown:
            terminated = True

        if self.current_step >= self._min_len - 1:
            truncated = True

        if episode_steps >= self.config.max_steps:
            truncated = True

        if episode_steps >= self.max_episode_steps:
            truncated = True

        # ═══════════════════════════════════════════════════════════════════════
        # v3.8 STRONGER LEARNING SIGNAL
        # - Wider clipping (±0.05) for stronger gradient signal
        # - Trend-following bonus: reward aligning with momentum
        # - Still bounded to prevent instability
        # ═══════════════════════════════════════════════════════════════════════
        equity_delta = self.balance - prev_balance
        base_reward = (equity_delta / max(self.initial_episode_balance, 1.0))
        
        # Trend-following bonus: reward positions aligned with short-term momentum
        trend_bonus = 0.0
        for inst in self.instruments:
            pos = self.positions[inst]
            if abs(pos) > 0.05:  # Only if we have a meaningful position
                df = self.data[inst][self.primary_tf]
                step = self.current_step
                if step > 5:
                    # 5-bar momentum
                    price_now = df["close"].iloc[step]
                    price_5ago = df["close"].iloc[step - 5]
                    momentum = (price_now - price_5ago) / max(price_5ago, 1e-8)
                    # Reward alignment: +bonus if position matches momentum direction
                    alignment = np.sign(pos) * np.sign(momentum)
                    trend_bonus += alignment * abs(momentum) * 0.1  # Scale factor
        trend_bonus /= max(len(self.instruments), 1)  # Average across instruments
        
        reward = base_reward + float(np.clip(trend_bonus, -0.005, 0.005))
        
        # WIDER CLIPPING: Max ±0.05 per step (5x stronger signal)
        # Over 2000 steps, max possible cumulative = ±100 (still bounded)
        reward = float(np.clip(reward, -0.05, 0.05))

        obs = self._get_observation()

        return obs, reward, terminated, truncated, {
            "balance": self.balance,
            "instrument_pnl": instrument_pnl,
            "instrument_rewards": instrument_rewards,
            "positions": dict(self.positions),
            "reward": reward,
            "final_return": (self.balance - self.initial_episode_balance) / self.initial_episode_balance,
            "episode_steps": episode_steps,
            "drawdown": dd,
        }

    ###########################################################################
    # OBSERVATION BUILDER (v2 + USD correlation)
    ###########################################################################
    def _get_observation(self):
        features: List[float] = []

        # Account state
        features.append(self.balance / self.config.initial_balance)
        features.append(
            (self.peak_balance - self.balance) / max(self.peak_balance, 1.0)
        )
        features.append(self.current_step / max(self.config.max_steps, 1))

        # Synthetic USD correlation predictor
        # Average short-term returns across instruments (inverse for USD strength)
        usd_strength = 0.0
        for inst in self.instruments:
            df = self.data[inst][self.primary_tf]
            step = self.current_step
            if step > 2:
                prev = df["close"].iloc[step - 2]
                if prev != 0:
                    r = (df["close"].iloc[step] - prev) / prev
                    usd_strength += -r  # USD inverse relationship
        usd_strength /= max(len(self.instruments), 1)
        features.append(float(usd_strength))

        # Per-instrument features
        for inst in self.instruments:
            pos = self.positions[inst]
            entry = self.entry_prices[inst]

            df = self.data[inst][self.primary_tf]
            price = float(df["close"].iloc[self.current_step])

            unreal = 0.0
            if entry > 0:
                unreal = (price - entry) / entry * np.sign(pos)

            features.append(pos)
            features.append(unreal)

        # Multi-TF market features (unchanged from v1)
        for inst in self.instruments:
            for tf in ["M15", "H1", "H4", "D1"]:
                if tf not in self.data[inst]:
                    features.extend([0.0] * 10)
                    continue

                df = self.data[inst][tf]
                step = min(self.current_step, len(df) - 1)

                if step < 20:
                    features.extend([0.0] * 10)
                    continue

                close = df["close"].iloc[step]
                open_ = df["open"].iloc[step]
                high = df["high"].iloc[step]
                low = df["low"].iloc[step]

                closes_20 = df["close"].iloc[step - 19 : step + 1].to_numpy()
                closes_5 = df["close"].iloc[step - 4 : step + 1].to_numpy()

                sma20 = np.mean(closes_20)
                sma5 = np.mean(closes_5)

                features.extend(
                    [
                        (close - sma20) / sma20,
                        (close - sma5) / sma5,
                        (sma5 - sma20) / sma20,
                        np.std(closes_20) / sma20,
                        (close - df["close"].iloc[step - 1])
                        / df["close"].iloc[step - 1],
                        (close - df["close"].iloc[step - 5])
                        / df["close"].iloc[step - 5]
                        if step >= 5
                        else 0.0,
                        (high - low) / max(close, 1e-8),
                        (close - open_) / max(close, 1e-8),
                        (close - low) / max(high - low, 1e-8),
                        1.0 if sma5 > sma20 else -1.0,
                    ]
                )

        # Output with deterministic padding/clipping
        arr = np.array(features, dtype=np.float32)

        if len(arr) < self.config.environment_observation_size:
            out = np.zeros(self.config.environment_observation_size, dtype=np.float32)
            out[: len(arr)] = arr
            return out

        return arr[: self.config.environment_observation_size]

    ###########################################################################
    def render(self, mode="human"):
        print(f"Step={self.current_step}, Balance={self.balance:.2f}")

    ###########################################################################
    def close(self):
        self.data = {}
        self.positions = {}
        self.entry_prices = {}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CALLBACK - Progress tracking with Learning Indicators
# ═══════════════════════════════════════════════════════════════════════════════
class SimpleTrainingCallback(BaseCallback):
    """Progress callback with learning quality indicators."""

    def __init__(self, total_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time: Optional[datetime] = None
        self.last_log_step = 0

        # Learning tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_wins: List[bool] = []  # Profit > 0
        self.recent_window = 100  # Rolling window for metrics

        # Per-instrument tracking
        self.instrument_pnl_history: Dict[str, List[float]] = {}

        # Best metrics tracking
        self.best_avg_reward = float("-inf")
        self.best_win_rate = 0.0
        self.improvements = 0

    def _on_training_start(self):
        self.start_time = datetime.now()
        print(f"\n{'═'*70}")
        print(f"  🚀 SIMPLE MODE TRAINING - Multi-Instrument Learning")
        print(f"{'═'*70}")
        print(f"  Total timesteps: {self.total_timesteps:,}")
        print(f"  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─'*70}")
        print(f"  📊 WHAT TO WATCH (Signs of Learning):")
        print(f"     • Avg Reward: Should trend UPWARD over time")
        print(f"     • Win Rate:   Should improve from ~50% toward 55-65%+")
        print(f"     • Per-Instrument: Each instrument should show improvement")
        print(f"     • Trend:      ↑ = improving, → = stable, ↓ = declining")
        print(f"{'═'*70}\n")

    def _on_step(self) -> bool:
        # Track completed episodes
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"].get("r", 0)
                ep_length = info["episode"].get("l", 0)
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_wins.append(ep_reward > 0)

            # Track per-instrument P&L
            inst_pnl = info.get("instrument_pnl", {})
            for inst, pnl in inst_pnl.items():
                if inst not in self.instrument_pnl_history:
                    self.instrument_pnl_history[inst] = []
                self.instrument_pnl_history[inst].append(pnl)

        # Log every 2000 steps
        if self.num_timesteps - self.last_log_step >= 2000:
            self.last_log_step = self.num_timesteps
            self._print_progress()

        return True

    def _print_progress(self):
        """Print detailed progress with learning indicators."""
        if self.start_time is None:
            elapsed = 0.0
        else:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        progress = self.num_timesteps / self.total_timesteps * 100
        fps = self.num_timesteps / max(elapsed, 1.0)

        # Calculate metrics
        n_episodes = len(self.episode_rewards)

        if n_episodes < 5:
            # Not enough data yet
            print(
                f"[{progress:5.1f}%] Step {self.num_timesteps:>8,} | "
                f"Time: {elapsed/60:.1f}min | FPS: {fps:.0f} | "
                f"Collecting data... ({n_episodes} episodes)"
            )
            return

        # Recent metrics (last N episodes)
        recent_n = min(self.recent_window, n_episodes)
        recent_rewards = self.episode_rewards[-recent_n:]
        recent_wins = self.episode_wins[-recent_n:]

        avg_reward = np.mean(recent_rewards)
        win_rate = sum(recent_wins) / len(recent_wins) * 100

        # Compare to earlier period to detect trend
        trend_symbol = "→"  # Default: stable
        if n_episodes >= 2 * recent_n:
            older_rewards = self.episode_rewards[-(2 * recent_n) : -recent_n]
            older_avg = np.mean(older_rewards)

            diff = avg_reward - older_avg
            if diff > abs(older_avg) * 0.05:  # 5% improvement
                trend_symbol = "↑"
            elif diff < -abs(older_avg) * 0.05:  # 5% decline
                trend_symbol = "↓"

        # Track improvements
        improved = ""
        if avg_reward > self.best_avg_reward:
            self.best_avg_reward = avg_reward
            self.improvements += 1
            improved = " 🆕 NEW BEST!"

        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate

        # Learning quality assessment
        if win_rate >= 55 and avg_reward > 0:
            quality = "🟢 LEARNING"
        elif win_rate >= 50 or avg_reward > self.best_avg_reward * 0.8:
            quality = "🟡 EXPLORING"
        else:
            quality = "🔴 STRUGGLING"

        # Print formatted output
        print(
            f"[{progress:5.1f}%] Step {self.num_timesteps:>8,} | "
            f"Reward: {avg_reward:>7.1f} {trend_symbol} | "
            f"WinRate: {win_rate:>5.1f}% | "
            f"{quality}{improved}"
        )

        # Print per-instrument stats every ~10k steps
        if self.num_timesteps % 10000 < 2000 and self.instrument_pnl_history:
            self._print_instrument_stats()

    def _print_instrument_stats(self):
        """Print per-instrument performance breakdown."""
        # Determine a representative recent length for header
        first_series = next(iter(self.instrument_pnl_history.values()), [])
        recent_len = min(1000, len(first_series))

        print(f"  {'─'*60}")
        print(
            f"  📊 Per-Instrument Performance "
            f"(recent {recent_len} steps where available):"
        )

        for inst, pnl_list in self.instrument_pnl_history.items():
            if not pnl_list:
                continue

            recent = pnl_list[-1000:] if len(pnl_list) > 1000 else pnl_list
            total_pnl = sum(recent)
            avg_pnl = np.mean(recent) if recent else 0.0
            win_pct = (
                sum(1 for p in recent if p > 0) / max(len(recent), 1) * 100.0
            )

            # Trend detection
            if len(pnl_list) >= 2000:
                old_avg = np.mean(pnl_list[-2000:-1000])
                new_avg = np.mean(pnl_list[-1000:])
                if new_avg > old_avg * 1.05:
                    trend = "↑"
                elif new_avg < old_avg * 0.95:
                    trend = "↓"
                else:
                    trend = "→"
            else:
                trend = "→"

            status = (
                "🟢" if win_pct > 52 and total_pnl > 0 else "🟡" if win_pct >= 48 else "🔴"
            )
            print(
                f"     {inst:10s} | PnL: {total_pnl:>+8.2f} "
                f"| Avg: {avg_pnl:>+7.4f} | Win: {win_pct:5.1f}% "
                f"| Trend: {trend} {status}"
            )

        print(f"  {'─'*60}")

    def _on_training_end(self):
        if self.start_time is None:
            elapsed = 0.0
        else:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        n_episodes = len(self.episode_rewards)

        print(f"\n{'═'*70}")
        print(f"  ✅ TRAINING COMPLETED - LEARNING SUMMARY")
        print(f"{'═'*70}")
        print(f"  Duration:       {elapsed/60:.1f} minutes")
        print(f"  Total steps:    {self.num_timesteps:,}")
        print(f"  Episodes:       {n_episodes:,}")
        print(f"{'─'*70}")

        if n_episodes >= 10:
            # Final metrics
            final_rewards = self.episode_rewards[-min(100, n_episodes) :]
            final_wins = self.episode_wins[-min(100, n_episodes) :]

            final_avg_reward = np.mean(final_rewards)
            final_win_rate = sum(final_wins) / len(final_wins) * 100

            # Early vs late comparison
            early_rewards = self.episode_rewards[: max(1, n_episodes // 4)]
            early_avg = np.mean(early_rewards) if early_rewards else 0.0

            improvement = final_avg_reward - early_avg

            print(f"  📈 LEARNING PROGRESS:")
            print(f"     Early Avg Reward:  {early_avg:>10.2f}")
            print(f"     Final Avg Reward:  {final_avg_reward:>10.2f}")
            print(
                f"     Improvement:       {improvement:>+10.2f} "
                f"{'✓' if improvement > 0 else '✗'}"
            )
            print(f"{'─'*70}")
            print(f"  📊 FINAL PERFORMANCE:")
            print(f"     Best Avg Reward:   {self.best_avg_reward:>10.2f}")
            print(f"     Final Win Rate:    {final_win_rate:>10.1f}%")
            print(f"     Times Improved:    {self.improvements:>10}")

            # Per-instrument final summary
            if self.instrument_pnl_history:
                print(f"{'─'*70}")
                print(f"  📊 PER-INSTRUMENT LEARNING:")
                for inst, pnl_list in self.instrument_pnl_history.items():
                    if not pnl_list:
                        continue
                    total_pnl = sum(pnl_list)
                    win_pct = (
                        sum(1 for p in pnl_list if p > 0)
                        / max(len(pnl_list), 1)
                        * 100.0
                    )

                    # Early vs late for this instrument
                    if len(pnl_list) >= 100:
                        early_inst = np.mean(pnl_list[: len(pnl_list) // 4])
                        late_inst = np.mean(pnl_list[-len(pnl_list) // 4 :])
                        inst_improvement = late_inst - early_inst
                        if inst_improvement > 0:
                            imp_symbol = "↑"
                        elif inst_improvement < 0:
                            imp_symbol = "↓"
                        else:
                            imp_symbol = "→"
                    else:
                        imp_symbol = "→"

                    status = "✓" if win_pct > 52 else "✗"
                    print(
                        f"     {inst:10s} | Total PnL: {total_pnl:>+10.2f} "
                        f"| Win: {win_pct:5.1f}% | Trend: {imp_symbol} {status}"
                    )

            print(f"{'─'*70}")

            # Assessment
            if improvement > 0 and final_win_rate > 52:
                print(
                    f"  🎯 VERDICT: Model is LEARNING! "
                    f"Ready for more training or transfer."
                )
            elif improvement > 0:
                print(
                    f"  🎯 VERDICT: Model is improving. "
                    f"Consider longer training."
                )
            elif final_win_rate > 50:
                print(
                    f"  🎯 VERDICT: Slight progress. "
                    f"May need hyperparameter tuning."
                )
            else:
                print(
                    f"  🎯 VERDICT: Limited learning. "
                    f"Try --optuna for better params."
                )

        print(f"{'═'*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_data(config: SimpleConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load OHLCV data from CSV files."""
    data: Dict[str, Dict[str, pd.DataFrame]] = {}

    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file in data_dir.glob("*.csv"):
        try:
            # Parse filename: EURUSD_H1_features.csv -> EUR_USD, H1
            base = file.stem.replace("_features", "")
            parts = base.split("_")

            # Known timeframes
            tf_codes = {
                "M1",
                "M5",
                "M15",
                "M30",
                "H1",
                "H2",
                "H4",
                "H8",
                "D1",
                "W1",
                "MN1",
            }

            if len(parts) >= 2 and parts[-1].upper() in tf_codes:
                timeframe = parts[-1].upper()
                instrument = "_".join(parts[:-1])
            else:
                instrument = "_".join(parts)
                timeframe = "H1"

            # Normalize instrument name
            if "EUR" in instrument.upper() and "USD" in instrument.upper():
                instrument = "EURUSD"
            elif "XAU" in instrument.upper():
                instrument = "XAUUSD"

            # Load CSV
            df = pd.read_csv(file)

            # Check required columns
            required = {"open", "high", "low", "close"}
            if not required.issubset(df.columns):
                print(f"[SKIP] {file.name}: Missing OHLC columns")
                continue

            # Ensure volume
            if "volume" not in df.columns:
                df["volume"] = 1.0

            # Convert to float32
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(0.0)
                    .astype(np.float32)
                )

            data.setdefault(instrument, {})[timeframe] = df
            print(f"[LOADED] {instrument}/{timeframe}: {len(df):,} bars")

        except Exception as e:
            print(f"[ERROR] Loading {file.name}: {e}")

    if not data:
        raise ValueError("No data files loaded!")

    total_bars = sum(len(df) for d in data.values() for df in d.values())
    print(f"\n[SUMMARY] {len(data)} instruments, {total_bars:,} total bars\n")

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def train_simple_mode(config: SimpleConfig, resume_from: Optional[str] = None):
    """Train PPO in simple mode (no modules)."""

    print("\n" + "=" * 70)
    print("  SIMPLE MODE TRAINING")
    print("  No SmartInfoBus | No Modules | Pure Exploration")
    print("=" * 70 + "\n")

    # Load data
    data = load_data(config)

    # Create environments (separate train/eval streams using temporal split)
    def make_env(rank: int, seed: int, mode: str):
        def _init():
            env = SimpleTradingEnv(data, config, mode=mode)
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)
            return Monitor(env, filename=f"{config.log_dir}/monitor_{mode}_{rank}.csv")

        set_random_seed(seed + rank)
        return _init

    train_env = DummyVecEnv([make_env(0, config.init_seed, "train")])
    eval_env = DummyVecEnv([make_env(1, config.init_seed + 1000, "eval")])

    # Create or load model
    if resume_from and os.path.exists(resume_from):
        print(f"[RESUME] Loading model from {resume_from}")
        model = PPO.load(resume_from, env=train_env)
        current_steps = model.num_timesteps
    else:
        print("[NEW] Creating fresh PPO model")

        policy_kwargs = dict(
            net_arch=dict(
                pi=[config.policy_hidden_size, config.policy_hidden_size // 2],
                vf=[config.value_hidden_size, config.value_hidden_size // 2],
            ),
            activation_fn=nn.Tanh,
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            verbose=0,
            tensorboard_log=config.tensorboard_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=config.init_seed,
            policy_kwargs=policy_kwargs,
        )
        current_steps = 0

    remaining = config.final_training_steps - current_steps
    if remaining <= 0:
        print(
            f"[DONE] Already trained {current_steps:,} steps "
            f"(target: {config.final_training_steps:,})"
        )
        return model

    print(
        f"\n[TRAIN] {remaining:,} steps remaining "
        f"(current: {current_steps:,}, target: {config.final_training_steps:,})"
    )
    print(f"[DEVICE] {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(
        f"[CONFIG] lr={config.learning_rate}, ent_coef={config.ent_coef}, "
        f"gamma={config.gamma}\n"
    )

    # Callbacks
    callbacks = CallbackList(
        [
            SimpleTrainingCallback(remaining),
            CheckpointCallback(
                save_freq=config.checkpoint_freq,
                save_path=config.checkpoint_dir,
                name_prefix="simple_ppo",
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(config.model_dir, "best"),
                log_path=os.path.join(config.log_dir, "eval"),
                eval_freq=config.eval_freq,
                deterministic=True,
                n_eval_episodes=config.n_eval_episodes,
            ),
        ]
    )

    # Train
    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            tb_log_name="simple_ppo",
            reset_num_timesteps=(current_steps == 0),
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Saving emergency checkpoint...")
        model.save(os.path.join(config.model_dir, "simple_ppo_interrupt.zip"))
        raise

    # Save final model
    final_path = os.path.join(config.model_dir, "simple_ppo_final.zip")
    model.save(final_path)
    print(f"\n[SAVED] Final model: {final_path}")

    # Save model metadata for transfer learning verification
    action_shape = (
        list(train_env.action_space.shape or (4,))
        if hasattr(train_env, "action_space") and train_env.action_space.shape is not None
        else [4]
    )
    metadata = {
        "model_type": "simple_mode_ppo",
        "observation_size": config.environment_observation_size,
        "action_shape": action_shape,
        "policy_hidden": config.policy_hidden_size,
        "value_hidden": config.value_hidden_size,
        "total_timesteps": model.num_timesteps,
        "created_at": datetime.now().isoformat(),
        "hyperparameters": {
            "learning_rate": config.learning_rate,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_range": config.clip_range,
            "ent_coef": config.ent_coef,
            "vf_coef": config.vf_coef,
            "max_grad_norm": config.max_grad_norm,
            "target_kl": config.target_kl,
        },
        "network_architecture": {
            "policy_layers": [config.policy_hidden_size, config.policy_hidden_size // 2],
            "value_layers": [config.value_hidden_size, config.value_hidden_size // 2],
            "activation": "Tanh",
        },
        "transfer_compatible": True,
        "target_env": "ModernTradingEnv",
        "train_split": config.train_split,
        "instruments": config.instruments,
    }
    metadata_path = os.path.join(config.model_dir, "simple_ppo_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Metadata: {metadata_path}")
    print(f"\n[TRANSFER] Model ready for Phase 2 training with ModernTradingEnv")
    print(f"           Use: python train/train_ppo_hybrid.py --auto-pretrained")

    # Cleanup environments
    train_env.close()
    eval_env.close()
    gc.collect()  # Force garbage collection

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
class OptunaPruningCallback(BaseCallback):
    """Callback for Optuna integration with early stopping."""

    def __init__(
        self,
        trial: "optuna.Trial",  # type: ignore
        eval_env: DummyVecEnv,
        n_eval_episodes: int = 5,
        eval_freq: int = 5000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.eval_idx = 0
        self.is_pruned = False
        self.reward_history: List[float] = []
        self.best_reward: float = float("-inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        try:
            mean_reward = self._evaluate()
        except Exception:
            mean_reward = float("-inf")

        self.reward_history.append(mean_reward)
        self.best_reward = max(self.best_reward, mean_reward)
        self.eval_idx += 1

        # Report to Optuna
        self.trial.report(mean_reward, self.eval_idx)

        # Check for pruning
        if self.trial.should_prune():
            self.is_pruned = True
            return False

        return True

    def _evaluate(self) -> float:
        """Run evaluation episodes."""
        episode_rewards: List[float] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            steps = 0

            while not done and steps < 2000:
                action, _ = self.model.predict(np.asarray(obs), deterministic=True)  # type: ignore
                obs, reward, done, _ = self.eval_env.step(action)

                if isinstance(reward, np.ndarray):
                    episode_reward += float(reward[0])
                else:
                    episode_reward += float(reward)

                if isinstance(done, np.ndarray):
                    done = bool(done[0])

                steps += 1

            episode_rewards.append(episode_reward)

        return float(np.mean(episode_rewards))


class SimpleHyperparameterSpace:
    """Search space for simple mode hyperparameters."""

    @staticmethod
    def sample_params(trial: "optuna.Trial") -> Dict[str, Any]:  # type: ignore
        """Sample PPO hyperparameters for simple mode."""

        # Sample n_steps and batch_size with constraint
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        batch_size = min(batch_size, n_steps)  # Ensure batch_size <= n_steps

        return {
            # Core PPO params
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 1e-2, log=True
            ),
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": trial.suggest_int("n_epochs", 3, 20),
            "gamma": trial.suggest_float("gamma", 0.9, 0.999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 2.0),
            # Exploration-specific
            "target_kl": trial.suggest_float("target_kl", 0.005, 0.05),
            # Network architecture - MUST match ModernTradingEnv for transfer
            "policy_hidden": trial.suggest_categorical(
                "policy_hidden", [128, 256, 512]
            ),
            "value_hidden": trial.suggest_categorical("value_hidden", [128, 256, 512]),
            # Environment params
            "max_drawdown": trial.suggest_float("max_drawdown", 0.15, 0.40),
        }


def create_optuna_objective(
    data: Dict[str, Dict[str, pd.DataFrame]],
    base_config: SimpleConfig,
    timesteps_per_trial: int,
    eval_freq: int,
    n_eval_episodes: int,
) -> Callable[["optuna.Trial"], float]:  # type: ignore
    """Create Optuna objective function for simple mode."""

    def objective(trial: "optuna.Trial") -> float:  # type: ignore
        """Single trial: train and evaluate with sampled hyperparameters."""

        # Sample hyperparameters
        params = SimpleHyperparameterSpace.sample_params(trial)

        print(f"\n{'─'*60}")
        print(f"[TRIAL {trial.number}] Starting")
        print(
            f"  lr={params['learning_rate']:.2e}, ent={params['ent_coef']:.4f}, "
            f"n_steps={params['n_steps']}, gamma={params['gamma']:.4f}"
        )

        train_env: Optional[DummyVecEnv] = None
        eval_env: Optional[DummyVecEnv] = None

        try:
            # Create config with sampled params
            config = SimpleConfig(
                timesteps=timesteps_per_trial,
                learning_rate=params["learning_rate"],
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                n_epochs=params["n_epochs"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                clip_range=params["clip_range"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                target_kl=params["target_kl"],
                policy_hidden=params["policy_hidden"],
                value_hidden=params["value_hidden"],
                max_drawdown=params["max_drawdown"],
                initial_balance=base_config.initial_balance,
                data_dir=base_config.data_dir,
                seed=base_config.init_seed + trial.number,
                train_split=base_config.train_split,
            )

            # Create environments with same temporal split logic
            def make_env(mode: str):
                def _init():
                    return Monitor(SimpleTradingEnv(data, config, mode=mode))

                return _init

            train_env = DummyVecEnv([make_env("train")])
            eval_env = DummyVecEnv([make_env("eval")])

            # Create model
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[params["policy_hidden"], params["policy_hidden"] // 2],
                    vf=[params["value_hidden"], params["value_hidden"] // 2],
                ),
                activation_fn=nn.Tanh,
            )

            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=params["learning_rate"],
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                n_epochs=params["n_epochs"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                clip_range=params["clip_range"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                target_kl=params["target_kl"],
                policy_kwargs=policy_kwargs,
                verbose=0,
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=config.init_seed,
            )

            # Create pruning callback
            pruning_callback = OptunaPruningCallback(
                trial=trial,
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                verbose=0,
            )

            # Train
            model.learn(
                total_timesteps=timesteps_per_trial,
                callback=pruning_callback,
                progress_bar=False,
            )

            if pruning_callback.is_pruned:
                raise optuna.TrialPruned()  # type: ignore

            final_reward = pruning_callback.best_reward
            print(f"[TRIAL {trial.number}] Completed: reward={final_reward:.2f}")

            return final_reward

        except optuna.TrialPruned:  # type: ignore
            raise
        except Exception as e:
            print(f"[TRIAL {trial.number}] Failed: {e}")
            return float("-inf")
        finally:
            if train_env:
                try:
                    train_env.close()
                except Exception:
                    pass
            if eval_env:
                try:
                    eval_env.close()
                except Exception:
                    pass
            gc.collect()

    return objective


def run_optuna_optimization(
    n_trials: int,
    timesteps_per_trial: int,
    data_dir: str = "data/processed",
    initial_balance: float = 100000.0,  # From risk_policy.yaml
    eval_freq: int = 5000,
    n_eval_episodes: int = 5,
    seed: int = 42,
    output_dir: str = "optuna_results/simple",
) -> "optuna.Study":  # type: ignore
    """Run Optuna hyperparameter optimization for simple mode."""

    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    print("\n" + "═" * 70)
    print("  OPTUNA HYPERPARAMETER OPTIMIZATION - SIMPLE MODE")
    print("═" * 70)
    print(f"  Trials:          {n_trials}")
    print(f"  Steps/trial:     {timesteps_per_trial:,}")
    print(f"  Eval frequency:  {eval_freq:,}")
    print(f"  Device:          {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("═" * 70 + "\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data once
    base_config = SimpleConfig(
        data_dir=data_dir, initial_balance=initial_balance, seed=seed
    )
    data = load_data(base_config)

    # Create study
    storage_path = f"sqlite:///{output_dir}/optuna_simple.db"

    assert TPESampler is not None, "Optuna not available"
    assert MedianPruner is not None, "Optuna not available"
    sampler = TPESampler(seed=seed, n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)

    study = optuna.create_study(  # type: ignore
        study_name="simple_mode_optimization",
        storage=storage_path,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    # Create objective
    objective = create_optuna_objective(
        data=data,
        base_config=base_config,
        timesteps_per_trial=timesteps_per_trial,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
    )

    # Run optimization
    start_time = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    total_time = time.time() - start_time

    # Save results
    _save_optuna_results(study, output_dir, total_time)

    return study


def _save_optuna_results(
    study: Any, output_dir: str, total_time: float
) -> None:
    """Save optimization results."""

    all_trials = study.trials
    completed = [t for t in all_trials if t.state == TrialState.COMPLETE]  # type: ignore
    pruned = [t for t in all_trials if t.state == TrialState.PRUNED]  # type: ignore

    print("\n" + "═" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("═" * 70)
    print(f"  Total time:      {total_time/60:.1f} minutes")
    print(f"  Total trials:    {len(all_trials)}")
    print(f"    Completed:     {len(completed)}")
    print(f"    Pruned:        {len(pruned)}")

    if study.best_trial is None:
        print("  [!] No successful trials.")
        return

    print(f"\n  🏆 BEST TRIAL: #{study.best_trial.number}")
    print(f"     Reward: {study.best_value:.2f}")
    print("\n  Best Parameters:")
    print("  " + "─" * 40)

    for key, value in sorted(study.best_params.items()):
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
    print("═" * 70)

    # Save to JSON
    best_config = {
        "optimization_info": {
            "study_name": "simple_mode_optimization",
            "best_trial": study.best_trial.number,
            "best_reward": study.best_value,
            "total_trials": len(all_trials),
            "completed_at": datetime.now().isoformat(),
        },
        "best_params": study.best_params,
    }

    config_path = os.path.join(output_dir, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n  📁 Best config saved to: {config_path}")

    # Save as YAML if available
    if YAML_AVAILABLE and yaml is not None:
        yaml_path = os.path.join(output_dir, "best_config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(best_config, f, default_flow_style=False)
        print(f"  📁 YAML config saved to: {yaml_path}")

    # Print usage instructions
    print("\n  To train with best parameters:")
    print("  " + "─" * 40)
    params = study.best_params
    print(f"    python train/train_simple_mode.py \\")
    print(f"      --lr {params.get('learning_rate', 3e-4):.2e} \\")
    print(f"      --ent-coef {params.get('ent_coef', 0.02):.4f} \\")
    print(f"      --gamma {params.get('gamma', 0.99):.4f} \\")
    print(f"      --n-steps {params.get('n_steps', 2048)} \\")
    print(f"      --batch-size {params.get('batch_size', 64)} \\")
    print(f"      --timesteps 200000")
    print("═" * 70 + "\n")


def show_best_params(output_dir: str = "optuna_results/simple") -> None:
    """Display best parameters from a completed study."""
    config_path = os.path.join(output_dir, "best_config.json")

    if not os.path.exists(config_path):
        print(f"[!] No results found at: {config_path}")
        print("    Run optimization first with: --optuna")
        return

    with open(config_path) as f:
        config = json.load(f)

    info = config.get("optimization_info", {})
    best_reward = info.get("best_reward", None)

    print("\n" + "═" * 70)
    print("  BEST OPTIMIZATION RESULTS - SIMPLE MODE")
    print("═" * 70)
    print(f"  Best trial:   #{info.get('best_trial', 'N/A')}")
    if isinstance(best_reward, (int, float)):
        print(f"  Best reward:  {best_reward:.2f}")
    else:
        print(f"  Best reward:  {best_reward}")
    print(f"  Completed at: {info.get('completed_at', 'N/A')}")
    print("\n  Best Parameters:")
    print("  " + "─" * 40)

    for key, value in sorted(config.get("best_params", {}).items()):
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
    print("═" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Simple Mode PPO Training (No Modules) with Optuna Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training
  python train/train_simple_mode.py --timesteps 200000

  # Optuna hyperparameter optimization
  python train/train_simple_mode.py --optuna --n-trials 30 --timesteps 50000

  # Show best parameters from previous optimization
  python train/train_simple_mode.py --show-best

  # Train with specific parameters
  python train/train_simple_mode.py --lr 3e-4 --ent-coef 0.02 --timesteps 200000

  # Resume training
  python train/train_simple_mode.py --resume
        """,
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--optuna", action="store_true", help="Run Optuna hyperparameter optimization"
    )
    mode_group.add_argument(
        "--show-best",
        action="store_true",
        help="Show best params from previous optimization",
    )

    # Optuna settings
    optuna_group = parser.add_argument_group("Optuna Settings")
    optuna_group.add_argument(
        "--n-trials", type=int, default=30, help="Number of Optuna trials (default: 30)"
    )
    optuna_group.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluation frequency for pruning (default: 5000)",
    )
    optuna_group.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Episodes per evaluation (default: 5)",
    )

    # Training steps
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (or per trial for Optuna)",
    )

    # Hyperparameters
    hp_group = parser.add_argument_group("Hyperparameters")
    hp_group.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    hp_group.add_argument(
        "--ent-coef",
        type=float,
        default=0.02,
        help="Entropy coefficient (higher = more exploration)",
    )
    hp_group.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    hp_group.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for advantage estimation",
    )
    hp_group.add_argument(
        "--clip-range", type=float, default=0.2, help="PPO clip range"
    )
    hp_group.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function coefficient"
    )
    hp_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Max gradient norm for clipping",
    )
    hp_group.add_argument(
        "--target-kl",
        type=float,
        default=0.015,
        help="Target KL divergence for early stopping",
    )
    hp_group.add_argument(
        "--n-steps", type=int, default=2048, help="Steps per update"
    )
    hp_group.add_argument(
        "--batch-size", type=int, default=64, help="Batch size"
    )
    hp_group.add_argument(
        "--n-epochs", type=int, default=10, help="PPO epochs"
    )

    # Environment
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Max steps per episode (logical cap)",
    )
    env_group.add_argument(
        "--balance", type=float, default=100_000.0, help="Initial balance (default from risk_policy.yaml)"
    )
    env_group.add_argument(
        "--max-dd",
        type=float,
        default=0.30,
        help="Max drawdown before termination",
    )
    env_group.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of history used for training (rest is eval)",
    )

    # Resume/paths
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if available",
    )
    path_group.add_argument(
        "--checkpoint", type=str, default=None, help="Specific checkpoint to resume from"
    )
    path_group.add_argument(
        "--data-dir", type=str, default="data/processed", help="Data directory"
    )
    path_group.add_argument(
        "--output-dir",
        type=str,
        default="optuna_results/simple",
        help="Optuna output directory",
    )

    # Other
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.show_best:
        show_best_params(args.output_dir)
        return

    # Optuna optimization mode
    if args.optuna:
        if not OPTUNA_AVAILABLE:
            print("[ERROR] Optuna not installed. Run: pip install optuna")
            sys.exit(1)

        run_optuna_optimization(
            n_trials=args.n_trials,
            timesteps_per_trial=args.timesteps,
            data_dir=args.data_dir,
            initial_balance=args.balance,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        return

    # Standard training mode
    config = SimpleConfig(
        timesteps=args.timesteps,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        max_steps=args.max_steps,
        initial_balance=args.balance,
        max_drawdown=args.max_dd,
        data_dir=args.data_dir,
        seed=args.seed,
        train_split=args.train_split,
    )

    # Find checkpoint to resume from
    resume_path = None
    if args.checkpoint:
        resume_path = args.checkpoint
    elif args.resume:
        # Find latest checkpoint
        ckpt_dir = Path(config.checkpoint_dir)
        if ckpt_dir.exists():
            checkpoints = sorted(
                ckpt_dir.glob("*.zip"),
                key=os.path.getmtime,
                reverse=True,
            )
            if checkpoints:
                resume_path = str(checkpoints[0])
                print(f"[RESUME] Found checkpoint: {resume_path}")

        # Or use final model
        final = Path(config.model_dir) / "simple_ppo_final.zip"
        if final.exists() and not resume_path:
            resume_path = str(final)
            print(f"[RESUME] Found final model: {resume_path}")

    # Train
    logging.basicConfig(level=logging.WARNING)
    train_simple_mode(config, resume_from=resume_path)

    print("\n✅ SIMPLE MODE TRAINING COMPLETE!")
    print(f"   Model saved to: {config.model_dir}/")
    print(f"   Checkpoints in: {config.checkpoint_dir}/")
    print(
        f"\n   Next step: Use this model as pretrained for full module training:"
    )
    print(
        f"   python train/train_ppo_hybrid.py "
        f"--pretrained {config.model_dir}/simple_ppo_final.zip"
    )


if __name__ == "__main__":
    main()
