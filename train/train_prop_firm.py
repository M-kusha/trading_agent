#!/usr/bin/env python3
"""
train/train_prop_firm.py

PropFirm PPO Training Script (10/10) with:
- Discrete action space + Action Masking (sb3-contrib MaskablePPO)
- Domain randomization (robustness)
- Walk-forward evaluation / Optuna objective scoring
- Frame-stacked memory (keeps masking compatible)
- Adversarial eval (worse execution params)
- VecEnv-correct episode tracking + cleanup

Requirements (recommended):
  pip install stable-baselines3 sb3-contrib optuna torch

This script assumes envs/prop_firm_env.py provides:
- PropFirmTradingEnv
- PropFirmConfig
- env.get_action_mask() -> np.ndarray[bool] for Discrete action space
"""

from __future__ import annotations

import os
import sys
import json
import copy
import argparse
import platform
import time
import logging
import random
import gc
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.base_class import BaseAlgorithm

# Environment
from envs.prop_firm_env import PropFirmTradingEnv, PropFirmConfig

# Curriculum system (optional)
try:
    from envs.curriculum_config import (
        CurriculumStage,
        CurriculumStageConfig,
        get_stage_config,
        get_stage_progression,
    )
    from envs.curriculum_manager import CurriculumManager
    from envs.curriculum_env_wrapper import CurriculumEnvWrapper
    CURRICULUM_AVAILABLE = True
except ImportError:
    CURRICULUM_AVAILABLE = False
    CurriculumStage = None  # type: ignore
    CurriculumStageConfig = None  # type: ignore
    CurriculumManager = None  # type: ignore
    CurriculumEnvWrapper = None  # type: ignore
    get_stage_config = None  # type: ignore
    get_stage_progression = None  # type: ignore

# Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    from optuna.exceptions import TrialPruned
    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None  # type: ignore
    MedianPruner = None  # type: ignore
    TPESampler = None  # type: ignore
    TrialPruned = Exception  # type: ignore
    OPTUNA_AVAILABLE = False

# sb3-contrib (MaskablePPO + ActionMasker)
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    MASKABLE_AVAILABLE = True
except Exception:
    MaskablePPO = None  # type: ignore
    ActionMasker = None  # type: ignore
    MASKABLE_AVAILABLE = False

# Dashboard server (optional)
try:
    from dashboard.server import start_dashboard_server, WEB_AVAILABLE as DASHBOARD_AVAILABLE
except ImportError:
    DASHBOARD_AVAILABLE = False
    start_dashboard_server = None  # type: ignore


# =============================================================================
# LOGGING
# =============================================================================

def configure_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("propfirm_train")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(level)

    fh = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


logger = configure_logging()


# =============================================================================
# SEEDING / START METHOD
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)


def pick_subproc_start_method() -> Optional[str]:
    """
    fork is fast but unsafe with CUDA and often problematic on macOS.
    """
    sysname = platform.system()
    cuda = torch.cuda.is_available()

    if sysname == "Windows":
        return None
    if sysname == "Darwin":
        return "spawn"
    if cuda:
        return "spawn"
    return "fork"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_market_data(
    data_dir: str = "data/processed",
    instruments: Optional[List[str]] = None,
    min_bars: int = 5000,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    if instruments is None:
        instruments = ["XAUUSD"]

    data: Dict[str, Dict[str, pd.DataFrame]] = {}

    if not os.path.exists(data_dir):
        logger.warning(f"Data directory not found: {data_dir}. Using synthetic data.")
        return _create_synthetic_data(instruments, min_bars)

    tf_candidates = {"M1", "M5", "M15", "M30", "H1", "H2", "H4", "H8", "D1", "W1"}

    for file in os.listdir(data_dir):
        if not file.endswith(".csv"):
            continue

        filepath = os.path.join(data_dir, file)
        try:
            base = file[:-4].replace("_features", "")
            parts = base.split("_")

            if len(parts) >= 2 and parts[-1].upper() in tf_candidates:
                timeframe = parts[-1].upper()
                instrument = "_".join(parts[:-1])
            else:
                instrument = "_".join(parts)
                timeframe = "M15"

            if instruments and instrument not in instruments:
                continue

            df = pd.read_csv(filepath)

            required = {"open", "high", "low", "close"}
            if not required.issubset(df.columns):
                logger.warning(f"Skipping {file}: missing OHLC columns")
                continue

            if "volume" not in df.columns:
                df["volume"] = 1.0

            # Parse time if present
            for tcol in ("time", "timestamp", "datetime", "date"):
                if tcol in df.columns:
                    ts = pd.to_datetime(df[tcol], errors="coerce")
                    df[tcol] = ts
                    if ts.notna().any():
                        df = df.sort_values(by=tcol).reset_index(drop=True)
                    break

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)

            if len(df) < min_bars:
                logger.info(f"Skipping {file}: only {len(df)} bars (need {min_bars}+)")
                continue

            data.setdefault(instrument, {})[timeframe] = df
            logger.info(f"Loaded {instrument}/{timeframe}: {len(df):,} bars")

        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not data:
        logger.warning("No data loaded from files; generating synthetic data")
        return _create_synthetic_data(instruments, min_bars)

    total_bars = sum(len(df) for tfs in data.values() for df in tfs.values())
    logger.info(f"Total: {len(data)} instruments, {total_bars:,} bars")
    return data


def _create_synthetic_data(instruments: List[str], n_bars: int = 50000) -> Dict[str, Dict[str, pd.DataFrame]]:
    logger.info(f"Generating synthetic data: {n_bars} bars for {instruments}")
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    rng = np.random.default_rng(42)

    for instrument in instruments:
        if "XAU" in instrument or "GOLD" in instrument:
            base_price = 1900.0
            volatility = 0.008
        elif "EUR" in instrument:
            base_price = 1.10
            volatility = 0.003
        elif "GBP" in instrument:
            base_price = 1.27
            volatility = 0.004
        else:
            base_price = 100.0
            volatility = 0.01

        returns = np.zeros(n_bars, dtype=np.float64)
        trend = 0.0

        for i in range(n_bars):
            if rng.random() < 0.001:
                trend = rng.uniform(-0.0005, 0.0005)
            noise = rng.normal(0, volatility)
            mean_rev = -0.01 * np.sign(returns[:i].sum()) if i > 100 else 0.0
            returns[i] = trend + noise + mean_rev * 0.1

        prices = base_price * np.exp(np.cumsum(returns))
        opens = prices[:-1]
        closes = prices[1:]
        spread = rng.uniform(0.0003, 0.001, len(closes)) * closes
        highs = np.maximum(opens, closes) + spread
        lows = np.minimum(opens, closes) - spread

        base_vol = rng.integers(500, 2000, len(closes)).astype(np.float32)
        volume = base_vol * (1 + 0.5 * np.abs(returns[1:]))

        df = pd.DataFrame({
            "open": opens.astype(np.float32),
            "high": highs.astype(np.float32),
            "low": lows.astype(np.float32),
            "close": closes.astype(np.float32),
            "volume": volume.astype(np.float32),
        })
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        data[instrument] = {"M15": df}

    return data


# =============================================================================
# DATA SPLITS (WALK-FORWARD)
# =============================================================================

def _min_len_across(data: Dict[str, Dict[str, pd.DataFrame]]) -> int:
    m = None
    for inst, tfs in data.items():
        for _, df in tfs.items():
            n = len(df)
            m = n if m is None else min(m, n)
    return int(m or 0)


def slice_data_by_index(
    data: Dict[str, Dict[str, pd.DataFrame]],
    start: int,
    end: int,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for inst, tfs in data.items():
        out[inst] = {}
        for tf, df in tfs.items():
            s = max(0, int(start))
            e = min(int(end), len(df))
            out[inst][tf] = df.iloc[s:e].reset_index(drop=True)
    return out


def build_walk_forward_folds(
    data: Dict[str, Dict[str, pd.DataFrame]],
    n_folds: int = 3,
    val_ratio: float = 0.12,
    min_train_ratio: float = 0.55,
) -> List[Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, pd.DataFrame]]]]:
    """
    Expanding-window walk-forward folds:
      train: [0 .. train_end)
      val:   [train_end .. train_end+val_len)
    """
    n = _min_len_across(data)
    if n <= 0:
        return []

    val_len = max(1000, int(n * val_ratio))
    min_train = max(2000, int(n * min_train_ratio))

    max_train_end = n - val_len
    if max_train_end <= min_train + 100:
        # Not enough length; single split
        train = slice_data_by_index(data, 0, max_train_end)
        val = slice_data_by_index(data, max_train_end, n)
        return [(train, val)]

    # Choose train_end positions
    step = max(500, int((max_train_end - min_train) / max(n_folds, 1)))
    folds: List[Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, pd.DataFrame]]]] = []
    train_end = min_train

    for _ in range(n_folds):
        if train_end + val_len > n:
            break
        train = slice_data_by_index(data, 0, train_end)
        val = slice_data_by_index(data, train_end, train_end + val_len)
        folds.append((train, val))
        train_end += step

    return folds or []


# =============================================================================
# ENV FACTORY (+ MASKING + FRAME STACK)
# =============================================================================

def create_prop_firm_env(data: Dict[str, Dict[str, pd.DataFrame]], config: PropFirmConfig) -> PropFirmTradingEnv:
    return PropFirmTradingEnv(data, config)


def create_eval_vec_env(
    data: Dict[str, Dict[str, pd.DataFrame]],
    config: PropFirmConfig,
    seed: int = 0,
    use_action_masking: bool = True,
    frame_stack: int = 1,
) -> VecEnv:
    """
    Create a single-env VecEnv for evaluation with the same wrappers as training.
    This ensures observation shape matches between train and eval.
    """
    def make_env() -> Callable[[], Any]:
        def _init():
            env = create_prop_firm_env(data, config)
            # Note: No Monitor wrapper for eval (we don't need logging)
            if use_action_masking and MASKABLE_AVAILABLE and ActionMasker is not None:
                env = ActionMasker(env, _mask_fn)
            try:
                env.reset(seed=seed)
            except Exception:
                pass
            return env
        return _init
    
    venv = DummyVecEnv([make_env()])
    
    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=int(frame_stack))
    
    return venv


def _mask_fn(env: Any) -> np.ndarray:
    """
    sb3-contrib ActionMasker callback.
    Must unwrap through Monitor/other wrappers to reach PropFirmTradingEnv.
    """
    # Unwrap to find the env that has action_masks()
    current = env
    while hasattr(current, 'env'):
        if hasattr(current, 'action_masks') and callable(getattr(current, 'action_masks')):
            return current.action_masks()
        current = current.env
    # Final unwrapped env should have action_masks
    if hasattr(current, 'action_masks') and callable(getattr(current, 'action_masks')):
        return current.action_masks()
    raise AttributeError(f"Could not find action_masks() on env or wrapped envs: {type(env)}")


def create_vec_envs(
    data: Dict[str, Dict[str, pd.DataFrame]],
    config: PropFirmConfig,
    n_envs: int,
    seed: int,
    monitor_dir: str = "logs/training",
    use_action_masking: bool = True,
    frame_stack: int = 1,
) -> VecEnv:
    Path(monitor_dir).mkdir(parents=True, exist_ok=True)

    def make_env(rank: int) -> Callable[[], Any]:
        def _init():
            env = create_prop_firm_env(data, config)
            env = Monitor(env, filename=str(Path(monitor_dir) / f"monitor_{rank}.csv"))

            # Masking wrapper (only if sb3-contrib is available)
            if use_action_masking and MASKABLE_AVAILABLE and ActionMasker is not None:
                env = ActionMasker(env, _mask_fn)

            # Seed
            try:
                env.reset(seed=seed + rank)
            except Exception:
                pass
            return env
        return _init

    if platform.system() != "Windows" and n_envs > 1:
        start_method = pick_subproc_start_method() or "spawn"
        logger.info(f"Creating {n_envs} PARALLEL envs (SubprocVecEnv, start_method={start_method})")
        venv = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method=start_method)
    else:
        note = " (Windows: DummyVecEnv is sequential; keep n_envs small)" if platform.system() == "Windows" else ""
        logger.info(f"Creating {n_envs} envs (DummyVecEnv){note}")
        venv = DummyVecEnv([make_env(i) for i in range(n_envs)])

    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=int(frame_stack))

    return venv


def test_environment(data: Dict[str, Dict[str, pd.DataFrame]], config: PropFirmConfig) -> None:
    logger.info("Testing PropFirmTradingEnv sanity...")
    env = create_prop_firm_env(data, config)
    obs, _ = env.reset(seed=42)

    if not isinstance(obs, np.ndarray):
        raise ValueError("Observation is not ndarray")
    if not np.all(np.isfinite(obs)):
        raise ValueError("Observation contains non-finite values")

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if not np.all(np.isfinite(obs)):
            raise ValueError("Step produced non-finite observation")
        if not np.isfinite(reward):
            raise ValueError("Step produced non-finite reward")
        if not isinstance(info, dict):
            raise ValueError("Info is not dict")

    env.close()
    logger.info("✓ Environment test passed")


# =============================================================================
# EVALUATION (CUSTOM METRICS)
# =============================================================================

def evaluate_agent_trading(
    model: BaseAlgorithm,
    env: VecEnv,
    n_episodes: int = 10,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    use_action_masks: bool = True,
) -> Dict[str, float]:
    """
    Mask-aware evaluation for MaskablePPO with VecEnv.
    
    When using MaskablePPO, action_masks are passed to predict() so the agent
    only considers legal actions during evaluation.
    
    NOTE: env must be a VecEnv (wrapped with same wrappers as training, including
    VecFrameStack if used during training) to ensure observation shapes match.
    """
    rewards: List[float] = []
    pnls: List[float] = []
    win_rates: List[float] = []
    drawdowns: List[float] = []
    trade_counts: List[int] = []
    dd_breaches = 0
    
    # Check if model is MaskablePPO
    is_maskable = MASKABLE_AVAILABLE and MaskablePPO is not None and isinstance(model, MaskablePPO)
    
    # Helper to get action masks from VecEnv
    def get_action_masks_from_vec_env(venv: VecEnv) -> Optional[np.ndarray]:
        """Extract action masks from a VecEnv by unwrapping to the base env."""
        # For DummyVecEnv, we can access the underlying envs
        try:
            # Unwrap VecFrameStack if present
            current: Any = venv
            while hasattr(current, 'venv'):
                current = getattr(current, 'venv')
            # Now current should be DummyVecEnv or SubprocVecEnv
            if hasattr(current, 'envs'):
                envs_list = getattr(current, 'envs')
                if envs_list and len(envs_list) > 0:
                    base_env: Any = envs_list[0]
                    # Unwrap Monitor/ActionMasker to find action_masks
                    while hasattr(base_env, 'env'):
                        if hasattr(base_env, 'action_masks') and callable(getattr(base_env, 'action_masks')):
                            return np.array([base_env.action_masks()])
                        base_env = base_env.env
                    if hasattr(base_env, 'action_masks') and callable(getattr(base_env, 'action_masks')):
                        return np.array([base_env.action_masks()])
        except Exception:
            pass
        return None

    for ep in range(n_episodes):
        obs = env.reset()  # VecEnv.reset() returns just obs (no info)
        done = False
        ep_reward = 0.0
        steps = 0
        info: Dict[str, Any] = {}

        while not done:
            # Get action masks for mask-aware prediction
            # Cast obs to np.ndarray for type checker
            obs_array: np.ndarray = np.asarray(obs)
            if is_maskable and use_action_masks:
                action_masks = get_action_masks_from_vec_env(env)
                if action_masks is not None:
                    predict_kwargs = {"deterministic": deterministic, "action_masks": action_masks}
                    action, _ = model.predict(obs_array, **predict_kwargs)
                else:
                    action, _ = model.predict(obs_array, deterministic=deterministic)
            else:
                action, _ = model.predict(obs_array, deterministic=deterministic)
            
            # VecEnv.step returns (obs, rewards, dones, infos)
            obs, r, dones, infos = env.step(action)
            ep_reward += float(r[0])
            done = bool(dones[0])
            info = infos[0] if infos else {}
            steps += 1
            if max_steps is not None and steps >= max_steps:
                done = True

        rewards.append(ep_reward)
        pnls.append(float(info.get("total_pnl", 0.0)))
        win_rates.append(float(info.get("win_rate", 0.0)))
        drawdowns.append(float(info.get("drawdown", 0.0)))
        trade_counts.append(int(info.get("trade_count", 0)))

        term_reason = str(info.get("termination_reason", ""))
        if "drawdown" in term_reason:
            dd_breaches += 1

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "pnl_std": float(np.std(pnls)) if pnls else 0.0,
        "mean_win_rate": float(np.mean(win_rates)) if win_rates else 0.0,  # Include ALL episodes (0-trade = 0% WR)
        "mean_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
        "max_drawdown": float(np.max(drawdowns)) if drawdowns else 0.0,
        "mean_trades": float(np.mean(trade_counts)) if trade_counts else 0.0,
        "dd_breaches": float(dd_breaches),
    }


def score_trading_metrics(m: Dict[str, float], eval_episodes: int) -> float:
    mean_pnl = m["mean_pnl"]
    mean_wr = m["mean_win_rate"]
    max_dd = m["max_drawdown"]
    mean_trades = m["mean_trades"]
    pnl_std = m["pnl_std"]
    rew_std = m["reward_std"]
    mean_reward = m["mean_reward"]
    dd_breaches = m["dd_breaches"]

    profit_score = float(np.clip(mean_pnl / 500.0, -1.5, 2.5))

    if mean_wr < 0.40:
        winrate_score = -1.0
    elif mean_wr < 0.50:
        winrate_score = (mean_wr - 0.40) * 5.0 - 0.5
    elif mean_wr <= 0.70:
        winrate_score = (mean_wr - 0.50) * 5.0
    else:
        winrate_score = 1.0 - (mean_wr - 0.70) * 2.5

    if max_dd < 0.03:
        dd_score = 1.0
    elif max_dd < 0.05:
        dd_score = 0.8
    elif max_dd < 0.08:
        dd_score = 0.5 - (max_dd - 0.05) * 10.0
    else:
        dd_score = -0.5 - (max_dd - 0.08) * 6.0

    if mean_trades < 1:
        sel_score = -0.6
    elif mean_trades < 3:
        sel_score = (mean_trades - 1) * 0.25
    elif mean_trades <= 15:
        sel_score = 0.5 + (1.0 - abs(mean_trades - 8.0) / 7.0) * 0.5
    elif mean_trades <= 25:
        sel_score = 0.5 - (mean_trades - 15.0) * 0.05
    else:
        sel_score = -0.6

    consistency_score = 1.0 - float(np.clip(pnl_std / max(abs(mean_pnl) + 50.0, 50.0), 0.0, 2.0))

    if pnl_std > 1e-6:
        risk_adj = float(np.clip((mean_pnl / pnl_std) / 2.0, -1.0, 1.0))
    else:
        risk_adj = 0.5 if mean_pnl > 0 else 0.0

    survival_penalty = -0.6 * (dd_breaches / max(float(eval_episodes), 1.0))
    reward_stability = 1.0 - float(np.clip(rew_std / max(abs(mean_reward) + 0.1, 0.1), 0.0, 2.0))

    score = (
        0.28 * profit_score +
        0.18 * winrate_score +
        0.22 * dd_score +
        0.14 * sel_score +
        0.08 * consistency_score +
        0.06 * risk_adj +
        0.04 * reward_stability +
        survival_penalty
    )

    if mean_pnl > 200 and mean_wr > 0.55 and max_dd < 0.05 and mean_trades >= 3:
        score += 0.25

    return float(score)


# =============================================================================
# CALLBACKS (VECENV-CORRECT)
# =============================================================================

class VecEpisodeTradingCallback(BaseCallback):
    """
    Correct per-env episode tracking for VecEnv.
    Logs trading metrics from info dict at episode end.
    
    Also integrates with UnifiedMemory for online learning during training.
    """

    def __init__(self, total_timesteps: int, log_interval_steps: int = 50_000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_interval_steps = log_interval_steps
        self._last_log = 0
        self._n_envs = 1

        self._ep_rewards: List[float] = []
        self._ep_lens: List[int] = []
        self._ep_pnls: List[float] = []
        self._ep_trades: List[int] = []
        self._ep_wrs: List[float] = []
        self._ep_dds: List[float] = []
        
        # New trading metrics from prop_firm_env
        self._ep_r_multiples: List[float] = []
        self._ep_profit_factors: List[float] = []
        self._ep_avg_maes: List[float] = []
        self._ep_avg_mfes: List[float] = []
        self._ep_avg_bars_held: List[float] = []
        self._ep_avg_entry_quality: List[float] = []
        self._ep_consecutive_wins: List[int] = []
        self._ep_consecutive_losses: List[int] = []
        self._exit_reason_counts: Dict[str, int] = {}  # Aggregate across all episodes

        self._cur_rewards: List[float] = []
        self._cur_lens: List[int] = []
        
        # PPO training diagnostics (updated after each rollout)
        self._diagnostics: Dict[str, float] = {}
        
        # Timing for FPS calculation
        self._start_time: Optional[float] = None
        
        # Memory system integration (lazy initialization)
        self._unified_memory: Optional[Any] = None
        self._memory_init_attempted: bool = False
        self._episode_trades_buffer: List[Dict[str, Any]] = []  # Accumulate trades within episode

    def _on_training_start(self) -> None:
        env = self.training_env
        self._n_envs = int(getattr(env, "num_envs", 1))
        self._cur_rewards = [0.0 for _ in range(self._n_envs)]
        self._cur_lens = [0 for _ in range(self._n_envs)]
        self._start_time = time.time()  # Initialize FPS tracking
    def _on_rollout_end(self) -> None:
        """Capture PPO diagnostics after each rollout (before update)."""
        # SB3 stores training stats in model.logger.name_to_value
        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logger_dict = getattr(self.model.logger, 'name_to_value', {})
                
                # Map SB3 logger keys to our diagnostics
                key_mapping = {
                    'train/approx_kl': 'approx_kl',
                    'train/clip_fraction': 'clip_fraction',
                    'train/entropy_loss': 'entropy',
                    'train/explained_variance': 'explained_variance',
                    'train/value_loss': 'value_loss',
                    'train/policy_gradient_loss': 'policy_loss',
                    'train/loss': 'total_loss',
                    'train/learning_rate': 'learning_rate',
                    'time/fps': 'fps',
                    'train/clip_range': 'clip_range',
                    'train/n_updates': 'n_updates',
                }
                
                for sb3_key, our_key in key_mapping.items():
                    if sb3_key in logger_dict:
                        self._diagnostics[our_key] = float(logger_dict[sb3_key])
                
                # Also try rollout keys
                rollout_keys = ['rollout/ep_rew_mean', 'rollout/ep_len_mean']
                for key in rollout_keys:
                    if key in logger_dict:
                        short_key = key.split('/')[-1]
                        self._diagnostics[short_key] = float(logger_dict[key])
            
            # Try to get FPS from model's num_timesteps and elapsed time
            if 'fps' not in self._diagnostics or self._diagnostics.get('fps', 0) == 0:
                if self._start_time is not None:
                    elapsed = time.time() - self._start_time
                    if elapsed > 0:
                        self._diagnostics['fps'] = self.num_timesteps / elapsed
                else:
                    self._start_time = time.time()
            
            # Get n_updates from model if not in logger
            if 'n_updates' not in self._diagnostics or self._diagnostics.get('n_updates', 0) == 0:
                if hasattr(self.model, '_n_updates'):
                    self._diagnostics['n_updates'] = int(self.model._n_updates)
            
            # Get learning rate from model
            if 'learning_rate' not in self._diagnostics or self._diagnostics.get('learning_rate', 0) == 0:
                if hasattr(self.model, 'learning_rate'):
                    lr = self.model.learning_rate
                    if callable(lr):
                        # Learning rate schedule
                        progress = self.num_timesteps / max(self.model._total_timesteps, 1) if hasattr(self.model, '_total_timesteps') else 1.0
                        lr = lr(1.0 - progress)
                    self._diagnostics['learning_rate'] = float(lr)
            
            # Get clip_range from model
            if 'clip_range' not in self._diagnostics:
                if hasattr(self.model, 'clip_range'):
                    cr: Any = getattr(self.model, 'clip_range', 0.2)
                    if callable(cr):
                        progress = self.num_timesteps / max(getattr(self.model, '_total_timesteps', 1), 1)
                        cr = cr(1.0 - progress)
                    self._diagnostics['clip_range'] = float(cr) if cr is not None else 0.2
                    
        except Exception as e:
            pass  # Don't crash training if diagnostics fail


    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if rewards is None or dones is None or infos is None:
            return True

        rewards_arr = np.array(rewards, dtype=np.float64).reshape(-1)
        dones_arr = np.array(dones, dtype=np.bool_).reshape(-1)

        # Accumulate per-env running reward/length
        for i in range(min(self._n_envs, len(rewards_arr))):
            self._cur_rewards[i] += float(rewards_arr[i])
            self._cur_lens[i] += 1

        # Handle episode termination per env
        for i, done in enumerate(dones_arr[:self._n_envs]):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            if not isinstance(info, dict):
                info = {}

            # Capture episode reward/length BEFORE resetting
            ep_reward = self._cur_rewards[i]
            ep_len = self._cur_lens[i]

            self._ep_rewards.append(ep_reward)
            self._ep_lens.append(ep_len)

            # SB3 DummyVecEnv stores terminal info under 'terminal_observation' and 'terminal_info'
            # For VecFrameStack, terminal info is preserved in the info dict
            finfo = info.get("terminal_info", info)

            # Extract metrics - try both direct and nested
            pnl = float(finfo.get("total_pnl", info.get("total_pnl", 0.0)))
            trades = int(finfo.get("trade_count", info.get("trade_count", 0)))
            wr = float(finfo.get("win_rate", info.get("win_rate", 0.0)))
            dd = float(finfo.get("drawdown", info.get("drawdown", 0.0)))

            self._ep_pnls.append(pnl)
            self._ep_trades.append(trades)
            self._ep_wrs.append(wr)
            self._ep_dds.append(dd)

            # Get episode_stats from env (contains aggregated trade metrics)
            ep_stats = finfo.get("episode_stats", info.get("episode_stats", {})) or {}

            # Use episode_stats for accurate metrics (falls back to per-step if not available)
            ep_r_mult = float(ep_stats.get("avg_r_multiple", 0.0))
            ep_mae = float(ep_stats.get("avg_mae", 0.0))
            ep_mfe = float(ep_stats.get("avg_mfe", 0.0))
            ep_bars = float(ep_stats.get("avg_bars_held", 0.0))
            ep_entry_q = float(ep_stats.get("avg_entry_quality", 0.5))
            ep_pf = float(ep_stats.get("profit_factor", 0.0))
            if ep_pf == float('inf'):
                ep_pf = 10.0  # Cap infinite profit factor

            self._ep_r_multiples.append(ep_r_mult)
            self._ep_avg_maes.append(ep_mae)
            self._ep_avg_mfes.append(ep_mfe)
            self._ep_avg_bars_held.append(ep_bars)
            self._ep_avg_entry_quality.append(ep_entry_q)
            self._ep_profit_factors.append(ep_pf)

            # Consecutive wins/losses (from step info)
            cons_wins = int(finfo.get("consecutive_wins", info.get("consecutive_wins", 0)))
            cons_losses = int(finfo.get("consecutive_losses", info.get("consecutive_losses", 0)))
            self._ep_consecutive_wins.append(cons_wins)
            self._ep_consecutive_losses.append(cons_losses)

            # Exit reason tracking from episode_stats (contains all trade close reasons)
            exit_dist = ep_stats.get("exit_quality_distribution", {}) or {}
            for reason, count in exit_dist.items():
                self._exit_reason_counts[reason] = self._exit_reason_counts.get(reason, 0) + count

            # Also track termination reason
            term_reason = str(finfo.get("termination_reason", info.get("termination_reason", "")))
            if term_reason:
                key = f"term:{term_reason}"
                self._exit_reason_counts[key] = self._exit_reason_counts.get(key, 0) + 1

            # Extract trades from episode info for memory learning
            episode_trades = finfo.get("trades", info.get("trades", []))
            if not episode_trades:
                # Construct minimal trade record from summary
                episode_trades = [{
                    "pnl": pnl,
                    "trade_count": trades,
                    "win_rate": wr,
                    "drawdown": dd,
                    "instrument": finfo.get("instrument", info.get("instrument", "UNKNOWN")),
                }]

            # Update memory system with episode data (now using correct ep_reward)
            self._update_memory_from_episode(
                episode_reward=ep_reward,
                episode_trades=episode_trades,
                market_context={
                    "regime": finfo.get("regime", info.get("regime", "unknown")),
                    "volatility": finfo.get("volatility", info.get("volatility", 0.5)),
                    "session": finfo.get("session", info.get("session", "unknown")),
                },
            )

            # Reset per-env running counters AFTER using them
            self._cur_rewards[i] = 0.0
            self._cur_lens[i] = 0

            # Save metrics every episode for dashboard
            self._save_live_metrics()

        # Periodic logging based on timesteps
        if self.num_timesteps - self._last_log >= self.log_interval_steps:
            self._log()
            self._last_log = self.num_timesteps

        return True


    def _log(self) -> None:
        if not self._ep_rewards:
            return

        n = min(50, len(self._ep_rewards))
        r = self._ep_rewards[-n:]
        pnls = self._ep_pnls[-n:]
        trades = self._ep_trades[-n:]
        wrs = self._ep_wrs[-n:]
        dds = self._ep_dds[-n:]
        lens = self._ep_lens[-n:]

        mean_reward = float(np.mean(r))
        mean_pnl = float(np.mean(pnls))
        total_trades = int(np.sum(trades))
        mean_wr = float(np.mean(wrs)) if wrs else 0.0  # Include ALL episodes
        max_dd = float(np.max(dds)) if dds else 0.0

        total_steps = int(np.sum(lens)) if lens else 1
        trades_per_1k = (total_trades / max(total_steps, 1)) * 1000.0
        progress = 100.0 * (self.num_timesteps / max(self.total_timesteps, 1))

        logger.info(
            f"Step {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.1f}%) | "
            f"Reward {mean_reward:+.3f} | PnL €{mean_pnl:+.0f} | "
            f"WR {mean_wr:.1%} | Trades/1k {trades_per_1k:.1f} | MaxDD {max_dd:.1%}"
        )
        
        # Save live metrics for dashboard
        self._save_live_metrics()
    
    def _save_live_metrics(self) -> None:
        """Save current training metrics to JSON for dashboard."""
        try:
            metrics_file = Path("logs/training/live_metrics.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Get recent episodes for charts
            n_recent = min(100, len(self._ep_rewards))
            
            # Ensure FPS is calculated if not yet available from SB3 logger
            if 'fps' not in self._diagnostics or self._diagnostics.get('fps', 0) <= 0:
                if self._start_time is not None and self.num_timesteps > 0:
                    elapsed = time.time() - self._start_time
                    if elapsed > 0:
                        self._diagnostics['fps'] = self.num_timesteps / elapsed
            
            # Calculate ETA
            eta_seconds = 0.0
            if self._start_time and self.num_timesteps > 0:
                elapsed = time.time() - self._start_time
                remaining = self.total_timesteps - self.num_timesteps
                rate = self.num_timesteps / max(elapsed, 1)
                eta_seconds = remaining / max(rate, 1)
            
            # Calculate summary stats
            mean_reward = float(np.mean(self._ep_rewards[-50:])) if self._ep_rewards else 0.0
            mean_pnl = float(np.mean(self._ep_pnls[-50:])) if self._ep_pnls else 0.0
            total_pnl = float(np.sum(self._ep_pnls)) if self._ep_pnls else 0.0
            mean_win_rate = float(np.mean(self._ep_wrs[-50:])) if self._ep_wrs else 0.0
            max_drawdown = float(np.max(self._ep_dds[-50:])) if self._ep_dds else 0.0
            mean_trades = float(np.mean(self._ep_trades[-50:])) if self._ep_trades else 0.0
            total_trades = int(np.sum(self._ep_trades)) if self._ep_trades else 0
            mean_r_multiple = float(np.mean(self._ep_r_multiples[-50:])) if self._ep_r_multiples else 0.0
            mean_profit_factor = float(np.mean(self._ep_profit_factors[-50:])) if self._ep_profit_factors else 0.0
            mean_entry_quality = float(np.mean(self._ep_avg_entry_quality[-50:])) if self._ep_avg_entry_quality else 0.5
            
            # Status calculations for dashboard
            def status_for_win_rate(wr):
                if wr >= 0.55: return "good"
                if wr >= 0.45: return "ok"
                return "bad"
            
            def status_for_drawdown(dd):
                if dd <= 0.03: return "good"
                if dd <= 0.06: return "ok"
                return "bad"
            
            def status_for_profit_factor(pf):
                if pf >= 1.5: return "good"
                if pf >= 1.0: return "ok"
                return "bad"
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                
                # Progress section (for dashboard progress display)
                "progress": {
                    "timesteps": self.num_timesteps,
                    "total_timesteps": self.total_timesteps,
                    "progress_pct": 100.0 * (self.num_timesteps / max(self.total_timesteps, 1)),
                    "total_episodes": len(self._ep_rewards),
                    "eta_seconds": eta_seconds,
                },
                
                # Learning section (PPO diagnostics)
                "learning": {
                    "fps": self._diagnostics.get('fps', 0),
                    "n_updates": self._diagnostics.get('n_updates', 0),
                    "mean_reward": mean_reward,
                    "mean_reward_status": "good" if mean_reward > 0 else "ok" if mean_reward > -5 else "bad",
                    "total_pnl": total_pnl,
                    "total_pnl_status": "good" if total_pnl > 0 else "ok" if total_pnl > -1000 else "bad",
                    "policy_loss": self._diagnostics.get('policy_loss', 0),
                    "value_loss": self._diagnostics.get('value_loss', 0),
                    "entropy": self._diagnostics.get('entropy', 0),
                    "kl_divergence": self._diagnostics.get('kl_divergence', 0),
                    "clip_fraction": self._diagnostics.get('clip_fraction', 0),
                    "explained_variance": self._diagnostics.get('explained_variance', 0),
                    "learning_rate": self._diagnostics.get('learning_rate', 0),
                },
                
                # Trading section
                "trading": {
                    "total_trades": total_trades,
                    "mean_trades": mean_trades,
                    "mean_trades_status": "good" if mean_trades >= 5 else "ok" if mean_trades >= 1 else "bad",
                    "mean_win_rate": mean_win_rate * 100,  # As percentage
                    "mean_win_rate_status": status_for_win_rate(mean_win_rate),
                    "max_drawdown": max_drawdown * 100,  # As percentage
                    "max_drawdown_status": status_for_drawdown(max_drawdown),
                },
                
                # Quality section
                "quality": {
                    "mean_profit_factor": mean_profit_factor,
                    "mean_profit_factor_status": status_for_profit_factor(mean_profit_factor),
                    "mean_r_multiple": mean_r_multiple,
                    "mean_r_multiple_status": "good" if mean_r_multiple > 1 else "ok" if mean_r_multiple > 0 else "bad",
                    "mean_entry_quality": mean_entry_quality,
                    "mean_entry_quality_status": "good" if mean_entry_quality > 0.6 else "ok" if mean_entry_quality > 0.4 else "bad",
                    "mean_mae": float(np.mean(self._ep_avg_maes[-50:])) if self._ep_avg_maes else 0.0,
                    "mean_mfe": float(np.mean(self._ep_avg_mfes[-50:])) if self._ep_avg_mfes else 0.0,
                    "mean_bars_held": float(np.mean(self._ep_avg_bars_held[-50:])) if self._ep_avg_bars_held else 0.0,
                    "max_consecutive_wins": int(max(self._ep_consecutive_wins[-50:])) if self._ep_consecutive_wins else 0,
                    "max_consecutive_losses": int(max(self._ep_consecutive_losses[-50:])) if self._ep_consecutive_losses else 0,
                },
                
                # Exit stats
                "exit_stats": {
                    "distribution": dict(self._exit_reason_counts),
                },
                
                # Recent history for charts
                "recent_rewards": [float(x) for x in self._ep_rewards[-n_recent:]],
                "recent_pnls": [float(x) for x in self._ep_pnls[-n_recent:]],
                "recent_win_rates": [float(x) * 100 for x in self._ep_wrs[-n_recent:]],  # As percentage
                "recent_drawdowns": [float(x) * 100 for x in self._ep_dds[-n_recent:]],  # As percentage
                "recent_trades": [int(x) for x in self._ep_trades[-n_recent:]],
                "recent_lengths": [int(x) for x in self._ep_lens[-n_recent:]],
                "recent_r_multiples": [float(x) for x in self._ep_r_multiples[-n_recent:]],
                "recent_entry_quality": [float(x) for x in self._ep_avg_entry_quality[-n_recent:]],
                "recent_bars_held": [float(x) for x in self._ep_avg_bars_held[-n_recent:]],
                
                # Legacy flat fields for backward compatibility
                "timesteps": self.num_timesteps,
                "total_timesteps": self.total_timesteps,
                "progress_pct": 100.0 * (self.num_timesteps / max(self.total_timesteps, 1)),
                "total_episodes": len(self._ep_rewards),
                "mean_reward": mean_reward,
                "mean_pnl": mean_pnl,
                "total_pnl": total_pnl,
                "mean_win_rate": mean_win_rate,
                "max_drawdown": max_drawdown,
                "mean_trades": mean_trades,
                "total_trades": total_trades,
                "mean_r_multiple": mean_r_multiple,
                "mean_profit_factor": mean_profit_factor,
                "mean_entry_quality": mean_entry_quality,
            }
            
            # Write directly - simpler and works on Windows
            # The dashboard server handles partial reads gracefully
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f)  # No indent for faster writes
                
            # Debug: confirm file was written
            if len(self._ep_rewards) <= 5:
                print(f"[Dashboard] Saved metrics: ep={len(self._ep_rewards)}, pnl={metrics['mean_pnl']:.4f}, wr={metrics['mean_win_rate']:.2%}, trades={metrics['mean_trades']:.1f}")
        except Exception as e:
            print(f"[Dashboard] Error saving metrics: {e}")
    
    def _get_unified_memory(self) -> Optional[Any]:
        """Lazy-load UnifiedMemory for training integration."""
        if self._memory_init_attempted:
            return self._unified_memory
        
        self._memory_init_attempted = True
        
        try:
            from modules.memory.unified_memory import UnifiedMemory, UnifiedMemoryConfig
            
            # Create memory with minimal config for training
            # UnifiedMemory.__init__ expects UnifiedMemoryConfig dataclass
            memory_config = UnifiedMemoryConfig(
                debug=False,  # Disable debug during training for performance
                enable_replay=False,  # Skip replay component (not needed for learning)
                enable_budget=False,  # Skip budget optimization
                enable_neural=True,  # Enable for encoder training
                enable_mistakes=True,  # Enable for danger zone learning
                enable_playbook=True,  # Enable for KNN learning
                enable_loss_risk_head=True,  # Enable for loss prediction
                enable_interventions=True,  # Enable for intervention learning
                enable_compression=True,  # Enable for intuition vector
            )
            
            self._unified_memory = UnifiedMemory(config=memory_config)  # type: ignore[arg-type]
            logger.info("[Memory] UnifiedMemory initialized for training integration")
            
        except Exception as e:
            logger.warning(f"[Memory] Could not initialize UnifiedMemory: {e}")
            self._unified_memory = None
        
        return self._unified_memory
    
    def _update_memory_from_episode(
        self,
        episode_reward: float,
        episode_trades: List[Dict[str, Any]],
        market_context: Dict[str, Any],
    ) -> None:
        """
        Update UnifiedMemory's learning components at episode end.
        
        This enables online learning of:
        - LossRiskHead: P(loss > τ) from trade outcomes
        - SharedEncoder: Contrastive learning (winners vs losers)
        - Interventions: Anti-relapse patterns
        """
        memory = self._get_unified_memory()
        if memory is None:
            return
        
        try:
            # Call the update_from_episode method we added to UnifiedMemory
            if hasattr(memory, 'update_from_episode'):
                stats = memory.update_from_episode(
                    episode_trades=episode_trades,
                    episode_reward=episode_reward,
                    market_context=market_context,
                )
                
                # Log occasionally (every 100 episodes)
                if len(self._ep_rewards) % 100 == 0 and stats.get("trades_processed", 0) > 0:
                    logger.info(
                        f"[Memory] Episode {len(self._ep_rewards)}: "
                        f"trades={stats.get('trades_processed', 0)}, "
                        f"loss_head={stats.get('loss_head_updates', 0)}, "
                        f"encoder={stats.get('encoder_updates', 0)}"
                    )
        except Exception as e:
            # Non-fatal: memory update failure shouldn't crash training
            if len(self._ep_rewards) <= 5:
                logger.debug(f"[Memory] Episode update failed (non-fatal): {e}")


# =============================================================================
# OPTUNA (WALK-FORWARD + EVAL-BASED PRUNING)
# =============================================================================

def sample_ppo_hyperparams(trial: Any) -> Dict[str, Any]:
    """
    Optimized search space based on Trial 12 success:
    - Higher ent_coef to prevent policy collapse
    - Narrower ranges around proven good values
    - Favor larger networks (512) that showed better capacity
    """
    return {
        # Trial 12: 1.16e-5 worked well, explore nearby
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
        # Larger n_steps showed better stability
        "n_steps": trial.suggest_categorical("n_steps", [2048, 4096, 8192]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "n_epochs": trial.suggest_int("n_epochs", 8, 20),
        # Trial 12: 0.945, explore higher gammas for longer-term credit
        "gamma": trial.suggest_float("gamma", 0.93, 0.995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.94, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.15, 0.30),
        # CRITICAL: Minimum 0.05 to prevent entropy collapse (0.03 still collapses sometimes)
        "ent_coef": trial.suggest_float("ent_coef", 0.05, 0.15, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.5, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.4, 0.8),
        "target_kl": trial.suggest_float("target_kl", 0.008, 0.05),
        # Favor larger networks - Trial 12's 512 worked best
        "policy_hidden": trial.suggest_categorical("policy_hidden", [256, 512]),
        "value_hidden": trial.suggest_categorical("value_hidden", [256, 512]),
    }


def sample_env_hyperparams(trial: Any) -> Dict[str, Any]:
    """
    Optimized env params based on Trial 12:
    - Moderate reward_scale (not too high to overwhelm ent_coef)
    - Higher entry_quality_threshold for selective trades
    """
    return {
        # Trial 12: 10.4, but lower can help entropy. Range 5-15
        "reward_scale": trial.suggest_float("reward_scale", 5.0, 15.0),
        # Higher risk penalty encourages prop-firm-safe behavior
        "risk_penalty_scale": trial.suggest_float("risk_penalty_scale", 1.5, 4.5),
        "quality_bonus_scale": trial.suggest_float("quality_bonus_scale", 0.2, 1.0),
        "blocked_action_penalty": trial.suggest_float("blocked_action_penalty", 0.02, 0.08),
        # Higher threshold = more selective entries (Trial 12: 0.533)
        "entry_quality_threshold": trial.suggest_float("entry_quality_threshold", 0.35, 0.65),
    }


def _sanity_adjust_ppo_params(n_envs: int, ppo_params: Dict[str, Any]) -> Dict[str, Any]:
    n_steps = int(ppo_params["n_steps"])
    batch_size = int(ppo_params["batch_size"])
    rollout = n_steps * n_envs
    if batch_size > rollout:
        ppo_params["batch_size"] = max(64, min(rollout, 256))
    return ppo_params


def _make_eval_config_adversarial(base: PropFirmConfig, adversity: float) -> PropFirmConfig:
    """
    adversity in [0..1]: increases execution harshness without touching prop limits.
    The env is written to safely ignore missing fields.
    """
    cfg = copy.deepcopy(base)  # deep copy preserves nested dataclasses

    # Tell env to randomize execution per episode; we also force worse ranges for eval
    cfg.domain_randomization_enabled = True
    cfg.spread_mult_range = (1.0 + 0.20 * adversity, 1.0 + 0.60 * adversity)
    cfg.slippage_mult_range = (1.0 + 0.25 * adversity, 1.0 + 0.80 * adversity)
    cfg.latency_bars_range = (int(1 + 1 * adversity), int(2 + 3 * adversity))
    cfg.volatility_scale_range = (1.0, 1.0 + 0.35 * adversity)
    return cfg


def run_optuna_optimization(
    data: Dict[str, Dict[str, pd.DataFrame]],
    n_trials: int = 40,
    timesteps_per_trial: int = 150_000,
    n_envs: int = 4,
    n_eval_episodes: int = 10,
    walk_forward_folds: int = 2,
    frame_stack: int = 4,
    study_name: str = "propfirm_ppo",
    storage: Optional[str] = None,
) -> Any:
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Run: pip install optuna")
    assert optuna is not None and TPESampler is not None and MedianPruner is not None

    folds = build_walk_forward_folds(data, n_folds=walk_forward_folds)
    if not folds:
        raise RuntimeError("Cannot build walk-forward folds (insufficient data).")

    sampler = TPESampler(seed=42, multivariate=True)
    # FIXED: More conservative pruning to avoid killing promising trials too early
    # The issue was: pruning after fold 1 compares incomplete trials against completed 3-fold robust scores
    # Solution: Only prune after fold 2 (2/3 of training), require more evidence
    per_fold_steps = timesteps_per_trial // walk_forward_folds
    pruner = MedianPruner(
        n_startup_trials=10,  # 10 trials run fully before ANY pruning kicks in
        n_warmup_steps=per_fold_steps * 2,  # Don't prune until AFTER fold 2 completes
        n_min_trials=8,  # Need 8 completed trials before comparing medians
        interval_steps=per_fold_steps,  # Only check at fold boundaries
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    results_dir = Path("logs/optuna")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "trial_results.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Optuna start: trials={n_trials} steps/trial={timesteps_per_trial:,} folds={len(folds)} "
        f"n_envs={n_envs} device={device} masking={MASKABLE_AVAILABLE}"
    )

    def objective(trial: Any) -> float:
        seed_everything(10_000 + trial.number)

        ppo_params = _sanity_adjust_ppo_params(n_envs, sample_ppo_hyperparams(trial))
        env_params = sample_env_hyperparams(trial)

        # Shared base config for this trial
        base_cfg = PropFirmConfig(
            max_steps_per_episode=2000,
            reward_scale=float(env_params["reward_scale"]),
            risk_penalty_scale=float(env_params["risk_penalty_scale"]),
            quality_bonus_scale=float(env_params["quality_bonus_scale"]),
            blocked_action_penalty=float(env_params["blocked_action_penalty"]),
            entry_quality_threshold=float(env_params["entry_quality_threshold"]),
            # keep meaningful brakes
            max_trades_per_day=12,
            max_trades_per_session=6,
            domain_randomization_enabled=True,  # type: ignore[attr-defined]
        )

        # Model builder (MaskablePPO preferred)
        def build_model(train_env: VecEnv) -> BaseAlgorithm:
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[int(ppo_params["policy_hidden"]), int(ppo_params["policy_hidden"]) // 2],
                    vf=[int(ppo_params["value_hidden"]), int(ppo_params["value_hidden"]) // 2],
                ),
                activation_fn=nn.Tanh,
            )

            if MASKABLE_AVAILABLE and MaskablePPO is not None:
                return MaskablePPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=float(ppo_params["learning_rate"]),
                    n_steps=int(ppo_params["n_steps"]),
                    batch_size=int(ppo_params["batch_size"]),
                    n_epochs=int(ppo_params["n_epochs"]),
                    gamma=float(ppo_params["gamma"]),
                    gae_lambda=float(ppo_params["gae_lambda"]),
                    clip_range=float(ppo_params["clip_range"]),
                    ent_coef=float(ppo_params["ent_coef"]),
                    vf_coef=float(ppo_params["vf_coef"]),
                    max_grad_norm=float(ppo_params["max_grad_norm"]),
                    target_kl=float(ppo_params["target_kl"]),
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    device=device,
                    seed=10_000 + trial.number,
                )

            # Fallback: PPO (no masks)
            return PPO(
                "MlpPolicy",
                train_env,
                learning_rate=float(ppo_params["learning_rate"]),
                n_steps=int(ppo_params["n_steps"]),
                batch_size=int(ppo_params["batch_size"]),
                n_epochs=int(ppo_params["n_epochs"]),
                gamma=float(ppo_params["gamma"]),
                gae_lambda=float(ppo_params["gae_lambda"]),
                clip_range=float(ppo_params["clip_range"]),
                ent_coef=float(ppo_params["ent_coef"]),
                vf_coef=float(ppo_params["vf_coef"]),
                max_grad_norm=float(ppo_params["max_grad_norm"]),
                target_kl=float(ppo_params["target_kl"]),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=device,
                seed=10_000 + trial.number,
            )

        # Walk-forward scoring
        fold_scores: List[float] = []
        per_fold_steps = max(25_000, int(timesteps_per_trial / max(len(folds), 1)))

        for fi, (train_data, val_data) in enumerate(folds):
            train_env: Optional[VecEnv] = None
            eval_env: Optional[VecEnv] = None
            try:
                # Train env uses masking + frame stack
                train_env = create_vec_envs(
                    train_data, base_cfg,
                    n_envs=n_envs, seed=trial.number + 100 * fi,
                    use_action_masking=True,
                    frame_stack=frame_stack,
                )
                try:
                    train_env.seed(1_000 + trial.number + 10 * fi)
                except Exception:
                    pass

                model = build_model(train_env)

                # Create callback for live metrics during Optuna
                optuna_callback = VecEpisodeTradingCallback(
                    total_timesteps=per_fold_steps,
                    log_interval_steps=per_fold_steps + 1,  # Don't spam logs, just save metrics
                )

                # Train (chunked) then evaluate on VAL with adversarial execution
                print(f"\n[Trial {trial.number}] Fold {fi+1}/{len(folds)} - Training {per_fold_steps:,} steps...")
                model.learn(total_timesteps=per_fold_steps, progress_bar=True, callback=optuna_callback)

                eval_cfg = _make_eval_config_adversarial(base_cfg, adversity=0.7)
                # IMPORTANT: eval_env must use same wrappers as train_env (including frame_stack)
                eval_env = create_eval_vec_env(
                    val_data, eval_cfg,
                    seed=20_000 + trial.number + fi,
                    use_action_masking=True,
                    frame_stack=frame_stack,
                )

                metrics = evaluate_agent_trading(model, eval_env, n_episodes=n_eval_episodes, deterministic=True)
                score = score_trading_metrics(metrics, eval_episodes=n_eval_episodes)
                fold_scores.append(float(score))

                # Report fold-progress to Optuna for pruning decisions
                # Use a conservative estimate: mean minus partial std penalty
                # This prevents pruning promising trials with one bad fold
                current_mean = float(np.mean(fold_scores))
                current_std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
                # Report a pessimistic estimate so only truly bad trials get pruned
                pessimistic_score = current_mean - 0.1 * current_std
                
                trial.report(pessimistic_score, (fi + 1) * per_fold_steps)
                
                # Only allow pruning after fold 2 (not fold 1) - gives trials a fair chance
                if fi >= 1 and trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned after fold {fi+1}: score={pessimistic_score:.3f}")
                    raise TrialPruned()

            finally:
                try:
                    if train_env is not None:
                        train_env.close()
                except Exception:
                    pass
                try:
                    if eval_env is not None:
                        eval_env.close()
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        mean_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
        score_std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0

        # Penalize high variance across folds (robustness requirement)
        robust_score = mean_score - 0.15 * score_std

        record = {
            "trial": trial.number,
            "robust_score": robust_score,
            "mean_score": mean_score,
            "score_std": score_std,
            "fold_scores": fold_scores,
            "params": trial.params,
            "timestamp": datetime.now().isoformat(),
        }
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(
            f"Trial {trial.number}: robust={robust_score:.3f} mean={mean_score:.3f} std={score_std:.3f} "
            f"folds={len(fold_scores)}"
        )
        return robust_score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("=" * 70)
    logger.info("OPTUNA COMPLETE")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    Path("logs/optuna").mkdir(parents=True, exist_ok=True)
    with open("logs/optuna/best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    return study


# =============================================================================
# MAIN TRAINING (PRODUCTION)
# =============================================================================

def train_prop_firm_agent(
    data: Dict[str, Dict[str, pd.DataFrame]],
    total_timesteps: int,
    n_envs: int,
    learning_rate: float,
    batch_size: int,
    n_steps: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    target_kl: float,
    policy_hidden: int,
    value_hidden: int,
    checkpoint_freq: int,
    eval_freq: int,
    pretrained_path: Optional[str],
    config_overrides: Dict[str, Any],
    frame_stack: int,
) -> BaseAlgorithm:
    seed_everything(42)

    config = PropFirmConfig(**config_overrides)

    test_environment(data, config)

    use_masking = bool(MASKABLE_AVAILABLE)
    train_env = create_vec_envs(
        data, config,
        n_envs=n_envs, seed=42,
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )
    try:
        train_env.seed(42)
    except Exception:
        pass

    # Single eval env for best-model saving + SB3 EvalCallback
    eval_env_vec = create_vec_envs(
        data, config,
        n_envs=1, seed=1337,
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )

    # Adversarial single-env eval for *real trading score logging*
    # MUST use same wrappers (including frame_stack) as training env
    eval_cfg_adv = _make_eval_config_adversarial(config, adversity=0.8)
    eval_env_single = create_eval_vec_env(
        data, eval_cfg_adv,
        seed=1337,
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )

    model_dir = Path("models/propfirm")
    checkpoint_dir = Path("checkpoints/propfirm")
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device.upper()} | n_envs={n_envs} | masking={use_masking} | frame_stack={frame_stack}")

    policy_kwargs = dict(
        net_arch=dict(
            pi=[policy_hidden, policy_hidden // 2],
            vf=[value_hidden, value_hidden // 2],
        ),
        activation_fn=nn.Tanh,
    )

    Algo = PPO
    if use_masking and MASKABLE_AVAILABLE and MaskablePPO is not None:
        Algo = MaskablePPO  # type: ignore[assignment]

    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained model: {pretrained_path}")
        model = Algo.load(pretrained_path, env=train_env, device=device)  # type: ignore[attr-defined]
        current_steps = int(getattr(model, "num_timesteps", 0))
    else:
        model = Algo(  # type: ignore[call-arg]
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log="runs/propfirm",
            device=device,
            seed=42,
        )
        current_steps = 0

    remaining = max(0, total_timesteps - current_steps)
    logger.info(f"Target timesteps: {total_timesteps:,} | current: {current_steps:,} | remaining: {remaining:,}")

    callbacks: List[BaseCallback] = [
        VecEpisodeTradingCallback(total_timesteps=total_timesteps, log_interval_steps=50_000),
        CheckpointCallback(save_freq=checkpoint_freq, save_path=str(checkpoint_dir), name_prefix="propfirm_ppo"),
        EvalCallback(
            eval_env_vec,
            best_model_save_path=str(model_dir / "best"),
            log_path="logs/eval",
            eval_freq=eval_freq,
            deterministic=True,
            n_eval_episodes=5,
        ),
    ]

    # Periodic adversarial score logging (no pruning here; just visibility)
    class AdversarialEvalLogger(BaseCallback):
        def __init__(self, eval_env: VecEnv, freq: int = 300_000):
            super().__init__(0)
            self.eval_env = eval_env
            self.freq = int(freq)
            self._last = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last < self.freq:
                return True
            m = evaluate_agent_trading(self.model, self.eval_env, n_episodes=8, deterministic=True)
            s = score_trading_metrics(m, eval_episodes=8)
            logger.info(
                f"[ADVERSARIAL EVAL] step={self.num_timesteps:,} score={s:.3f} "
                f"PnL=€{m['mean_pnl']:+.0f} WR={m['mean_win_rate']:.1%} MaxDD={m['max_drawdown']:.1%} Trades={m['mean_trades']:.1f}"
            )
            self._last = self.num_timesteps
            return True

    callbacks.append(AdversarialEvalLogger(eval_env_single, freq=300_000))

    start_time = datetime.now()
    try:
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=CallbackList(callbacks),
                tb_log_name="propfirm_ppo",
                reset_num_timesteps=(current_steps == 0),
                progress_bar=True,
            )
        else:
            logger.info("Already at target timesteps; skipping training.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        model.save(str(model_dir / "propfirm_ppo_interrupted.zip"))
    finally:
        duration = datetime.now() - start_time
        logger.info(f"Training duration: {duration}")

        final_path = model_dir / "propfirm_ppo_final.zip"
        model.save(str(final_path))
        logger.info(f"Saved final model: {final_path}")

        metadata = {
            "total_timesteps_target": int(total_timesteps),
            "actual_timesteps": int(getattr(model, "num_timesteps", 0)),
            "device": device,
            "n_envs": int(n_envs),
            "masking": bool(use_masking),
            "frame_stack": int(frame_stack),
            "hyperparameters": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_steps": n_steps,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "target_kl": target_kl,
                "policy_hidden": policy_hidden,
                "value_hidden": value_hidden,
            },
            "config_overrides": config_overrides,
            "created_at": datetime.now().isoformat(),
        }
        with open(model_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        try:
            train_env.close()
        except Exception:
            pass
        try:
            eval_env_vec.close()
        except Exception:
            pass
        try:
            eval_env_single.close()
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


# =============================================================================
# CURRICULUM TRAINING
# =============================================================================

class CurriculumTrainingCallback(BaseCallback):
    """
    Training callback with curriculum integration.
    
    Extends VecEpisodeTradingCallback with curriculum stage tracking,
    automatic progression, LR warmup, and automatic checkpointing.
    """
    
    def __init__(
        self,
        curriculum_manager: Any,  # CurriculumManager
        total_timesteps: int,
        log_interval_steps: int = 50_000,
        save_path: str = "logs/curriculum",
        verbose: int = 1,
        enable_lr_warmup: bool = True,
        enable_checkpoints: bool = True,
    ):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.total_timesteps = total_timesteps
        self.log_interval_steps = log_interval_steps
        self.save_path = Path(save_path)
        self.enable_lr_warmup = enable_lr_warmup
        self.enable_checkpoints = enable_checkpoints
        
        self._last_log = 0
        self._last_metrics_save: float = 0.0  # Time-based metrics saving
        self._n_envs = 1
        
        # Episode tracking
        self._ep_rewards: List[float] = []
        self._ep_pnls: List[float] = []
        self._ep_win_rates: List[float] = []
        self._ep_drawdowns: List[float] = []
        self._ep_trades: List[int] = []
        self._ep_lens: List[int] = []
        
        # NEW: Trading quality metrics per episode
        self._ep_profit_factors: List[float] = []
        self._ep_r_multiples: List[float] = []
        self._ep_entry_quality: List[float] = []
        self._exit_reason_counts: Dict[str, int] = {}
        
        # PPO diagnostics (updated from logger)
        self._ppo_diagnostics: Dict[str, float] = {}
        self._n_updates: int = 0
        
        self._cur_rewards: List[float] = []
        self._cur_lens: List[int] = []
        
        # Stage tracking
        self._stage_history: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None
        self._last_stage: Optional[str] = None
        self._base_lr: Optional[float] = None
        
        # Register transition callback
        if self.curriculum_manager is not None:
            self.curriculum_manager.on_transition_callback = self._on_stage_transition
    
    def _on_stage_transition(
        self,
        transition_type: str,
        old_stage: Any,
        new_stage: Any,
        info: Dict[str, Any],
    ) -> None:
        """Called when curriculum stage changes."""
        logger.info(f"🔄 Stage transition: {transition_type} ({old_stage.name} → {new_stage.name})")
        
        # Checkpoint on transition
        if self.enable_checkpoints and self.model is not None:
            try:
                checkpoint_dir = self.save_path / "stage_checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model
                model_path = checkpoint_dir / f"model_{old_stage.name}_to_{new_stage.name}_{self.num_timesteps}.zip"
                self.model.save(str(model_path))
                
                # Save curriculum state
                state_path = checkpoint_dir / f"curriculum_{old_stage.name}_to_{new_stage.name}_{self.num_timesteps}.json"
                self.curriculum_manager.save(state_path)
                
                logger.info(f"  📁 Checkpoint saved: {model_path.name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Checkpoint failed: {e}")
    
    def _on_training_start(self) -> None:
        env = self.training_env
        self._n_envs = int(getattr(env, "num_envs", 1))
        self._cur_rewards = [0.0] * self._n_envs
        self._cur_lens = [0] * self._n_envs
        self._start_time = time.time()
        
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Store base learning rate
        if self.model is not None:
            self._base_lr = float(self.model.learning_rate) if not callable(self.model.learning_rate) else None
        
        # Initialize stage tracking
        if self.curriculum_manager is not None:
            self._last_stage = self.curriculum_manager.current_stage.name
        
        # Save initial metrics file so dashboard sees data immediately
        self._save_live_metrics()
    
    def _apply_lr_warmup(self) -> None:
        """Apply learning rate warmup based on curriculum transition state."""
        if not self.enable_lr_warmup or self.curriculum_manager is None or self.model is None:
            return
        
        if self._base_lr is None:
            return
        
        # Get LR multiplier from curriculum manager
        lr_mult = self.curriculum_manager.get_lr_multiplier()
        
        if lr_mult < 1.0:
            # Apply reduced LR during warmup
            new_lr = self._base_lr * lr_mult
            if hasattr(self.model, 'lr_schedule'):
                # For SB3, we need to modify the lr_schedule
                self.model.lr_schedule = lambda _: new_lr
            elif hasattr(self.model, 'learning_rate'):
                self.model.learning_rate = new_lr
    
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if rewards is None or dones is None or infos is None:
            return True
        
        # Update transition state (LR warmup progress)
        if self.curriculum_manager is not None:
            self.curriculum_manager.step_transition_state(timesteps=self._n_envs)
            self._apply_lr_warmup()

        rewards_arr = np.array(rewards, dtype=np.float64).reshape(-1)
        dones_arr = np.array(dones, dtype=np.bool_).reshape(-1)

        for i in range(min(self._n_envs, len(rewards_arr))):
            self._cur_rewards[i] += float(rewards_arr[i])
            self._cur_lens[i] += 1

        for i, done in enumerate(dones_arr[:self._n_envs]):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            if not isinstance(info, dict):
                info = {}

            ep_reward = self._cur_rewards[i]
            ep_len = self._cur_lens[i]
            
            finfo = info.get("terminal_info", info)
            
            # Get episode_stats if available (contains detailed metrics)
            ep_stats = finfo.get("episode_stats", info.get("episode_stats", {}))
            
            # Record basic metrics
            self._ep_rewards.append(ep_reward)
            self._ep_lens.append(ep_len)
            self._ep_pnls.append(float(finfo.get("total_pnl", info.get("total_pnl", 0.0))))
            self._ep_win_rates.append(float(finfo.get("win_rate", info.get("win_rate", 0.0))))
            self._ep_drawdowns.append(float(finfo.get("drawdown", info.get("drawdown", 0.0))))
            self._ep_trades.append(int(finfo.get("trade_count", info.get("trade_count", 0))))
            
            # Record trading quality metrics
            self._ep_profit_factors.append(float(ep_stats.get("profit_factor", finfo.get("profit_factor", 0.0))))
            self._ep_r_multiples.append(float(ep_stats.get("avg_r_multiple", finfo.get("avg_r_multiple", 0.0))))
            self._ep_entry_quality.append(float(ep_stats.get("avg_entry_quality", finfo.get("avg_entry_quality", 0.5))))
            
            # Track exit reasons from episode_stats (key is exit_quality_distribution)
            exit_dist = ep_stats.get("exit_quality_distribution", ep_stats.get("exit_distribution", {}))
            for reason, count in exit_dist.items():
                self._exit_reason_counts[reason] = self._exit_reason_counts.get(reason, 0) + int(count)
            
            # Update curriculum episode transition tick
            if self.curriculum_manager is not None:
                self.curriculum_manager.episode_transition_tick()
            
            # Track stage transitions
            current_stage = finfo.get("curriculum_stage", info.get("curriculum_stage", "unknown"))
            if self._stage_history and self._stage_history[-1]["stage"] != current_stage:
                self._stage_history.append({
                    "stage": current_stage,
                    "timestep": self.num_timesteps,
                    "episode": len(self._ep_rewards),
                })
            elif not self._stage_history:
                self._stage_history.append({
                    "stage": current_stage,
                    "timestep": self.num_timesteps,
                    "episode": 1,
                })
            
            # Reset tracking
            self._cur_rewards[i] = 0.0
            self._cur_lens[i] = 0
            
            # Save live metrics
            self._save_live_metrics()

        # Collect PPO diagnostics from SB3 logger
        self._update_ppo_diagnostics()

        # Periodic logging
        if self.num_timesteps - self._last_log >= self.log_interval_steps:
            self._log_progress()
            self._last_log = self.num_timesteps
        
        # Save metrics every 1 second (time-based, not step-based)
        now = time.time()
        if now - self._last_metrics_save >= 1.0:
            self._save_live_metrics()
            self._last_metrics_save = now

        return True
    
    def _update_ppo_diagnostics(self) -> None:
        """Extract PPO training diagnostics from the model/logger."""
        if self.model is None:
            return
        
        # Try to get diagnostics from the model's logger
        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logger_obj = self.model.logger
                
                # SB3 stores values in name_to_value dict
                if hasattr(logger_obj, 'name_to_value'):
                    values = logger_obj.name_to_value
                    
                    # Extract common PPO metrics
                    self._ppo_diagnostics['policy_loss'] = float(values.get('train/policy_gradient_loss', values.get('train/policy_loss', 0)))
                    self._ppo_diagnostics['value_loss'] = float(values.get('train/value_loss', 0))
                    self._ppo_diagnostics['entropy'] = float(values.get('train/entropy_loss', values.get('train/entropy', 0)))
                    self._ppo_diagnostics['kl_divergence'] = float(values.get('train/approx_kl', 0))
                    self._ppo_diagnostics['clip_fraction'] = float(values.get('train/clip_fraction', 0))
                    self._ppo_diagnostics['explained_variance'] = float(values.get('train/explained_variance', 0))
                    self._ppo_diagnostics['learning_rate'] = float(values.get('train/learning_rate', 0))
                    self._ppo_diagnostics['clip_range'] = float(values.get('train/clip_range', 0.2))
                    
                    # Track n_updates
                    n_updates = values.get('train/n_updates', 0)
                    if n_updates > 0:
                        self._n_updates = int(n_updates)
            
            # Alternative: get from model attributes
            if self._n_updates == 0 and hasattr(self.model, '_n_updates'):
                self._n_updates = int(self.model._n_updates)
                
        except Exception:
            pass  # Silently fail - diagnostics are optional
    
    def _log_progress(self) -> None:
        """Log training progress."""
        if not self._ep_rewards:
            return


        n = min(50, len(self._ep_rewards))
        rewards = self._ep_rewards[-n:]
        pnls = self._ep_pnls[-n:]
        win_rates = self._ep_win_rates[-n:]
        drawdowns = self._ep_drawdowns[-n:]
        trades = self._ep_trades[-n:]

        mean_reward = float(np.mean(rewards))
        mean_pnl = float(np.mean(pnls))
        mean_wr = float(np.mean(win_rates))
        max_dd = float(np.max(drawdowns))
        mean_trades = float(np.mean(trades))

        progress = 100.0 * (self.num_timesteps / max(self.total_timesteps, 1))
        
        stage = self.curriculum_manager.current_stage.name if self.curriculum_manager else "N/A"

        logger.info(
            f"Step {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.1f}%) | "
            f"Stage: {stage} | "
            f"Reward {mean_reward:+.3f} | PnL €{mean_pnl:+.0f} | "
            f"WR {mean_wr:.1%} | Trades {mean_trades:.1f} | MaxDD {max_dd:.1%}"
        )
    
    def _save_live_metrics(self) -> None:
        """Save metrics for dashboard."""
        try:
            # Save to standard location that dashboard expects
            metrics_file = Path("logs/training/live_metrics.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Saving live metrics to {metrics_file.absolute()}")
            
            n_recent = min(100, len(self._ep_rewards))
            
            # Compute FPS
            fps = 0.0
            if self._start_time is not None and self.num_timesteps > 0:
                elapsed = time.time() - self._start_time
                if elapsed > 0:
                    fps = self.num_timesteps / elapsed
            
            # Calculate ETA
            eta_seconds = 0.0
            if self._start_time and self.num_timesteps > 0:
                elapsed = time.time() - self._start_time
                remaining = self.total_timesteps - self.num_timesteps
                rate = self.num_timesteps / max(elapsed, 1)
                eta_seconds = remaining / max(rate, 1)
            
            # Get curriculum progress
            curriculum_progress = {}
            if self.curriculum_manager:
                curriculum_progress = self.curriculum_manager.get_progress_report()
            
            # Build curriculum detail section
            curriculum_detail = {}
            if self.curriculum_manager:
                stage_cfg = self.curriculum_manager.stage_config
                competence = stage_cfg.competence  # It's 'competence' not 'promotion_criteria'
                
                # Extract current_metrics and criteria_met from promotion_checks
                promotion_checks = curriculum_progress.get("promotion_checks", {})
                current_metrics = {}
                criteria_met = {}
                
                # Map the nested check format to flat format expected by dashboard
                for check_name, check_data in promotion_checks.items():
                    if isinstance(check_data, dict):
                        current_metrics[check_name] = check_data.get("actual", 0)
                        criteria_met[check_name] = check_data.get("passed", False)
                
                # Also add rolling stats as current metrics
                rolling_stats_raw = curriculum_progress.get("rolling_stats", {})
                if rolling_stats_raw:
                    current_metrics["win_rate"] = rolling_stats_raw.get("mean_win_rate", 0)
                    current_metrics["profit_factor"] = rolling_stats_raw.get("mean_profit_factor", 0)
                    current_metrics["drawdown"] = rolling_stats_raw.get("mean_drawdown", 0)
                    current_metrics["trade_count"] = rolling_stats_raw.get("mean_trade_count", 0)
                    current_metrics["pnl"] = rolling_stats_raw.get("mean_pnl", 0)
                
                curriculum_detail = {
                    "stage_name": self.curriculum_manager.current_stage.name,
                    "stage_idx": int(self.curriculum_manager.current_stage.value),
                    "is_in_transition": self.curriculum_manager.is_in_transition,
                    "reward_blend_factor": self.curriculum_manager.reward_blend_factor,
                    "lr_multiplier": self.curriculum_manager.get_lr_multiplier(),
                    # Promotion criteria thresholds (from competence)
                    "promotion_thresholds": {
                        "min_episodes": competence.min_episodes,
                        "min_win_rate": competence.min_win_rate,
                        "min_profit_factor": competence.min_profit_factor,
                        "max_drawdown": competence.max_avg_drawdown,
                        "confidence_level": getattr(competence, 'confidence_level', 0.95),
                    },
                    # Current vs required metrics (extracted from promotion_checks)
                    "current_metrics": current_metrics,
                    "criteria_met": criteria_met,
                    # Data difficulty settings
                    "data_difficulty": {
                        "volatility_range": stage_cfg.data_difficulty.volatility_percentile_range if stage_cfg.data_difficulty else (0.0, 1.0),
                        "trend_clarity_min": stage_cfg.data_difficulty.min_trend_clarity if stage_cfg.data_difficulty else 0.0,
                        "include_asian": stage_cfg.data_difficulty.include_asian_session if stage_cfg.data_difficulty else True,
                        "include_london": stage_cfg.data_difficulty.include_london_session if stage_cfg.data_difficulty else True,
                        "include_ny": stage_cfg.data_difficulty.include_ny_session if stage_cfg.data_difficulty else True,
                        "exclude_high_impact_news": stage_cfg.data_difficulty.exclude_high_impact_news if stage_cfg.data_difficulty else False,
                    },
                    # Transition settings
                    "transition_settings": {
                        "lr_warmup_steps": stage_cfg.transition.lr_warmup_steps if stage_cfg.transition else 0,
                        "lr_warmup_factor": stage_cfg.transition.lr_warmup_factor if stage_cfg.transition else 1.0,
                        "reward_blend_episodes": stage_cfg.transition.reward_blend_episodes if stage_cfg.transition else 0,
                        "cooldown_episodes": stage_cfg.transition.transition_cooldown_episodes if stage_cfg.transition else 0,
                    },
                }
            
            # Calculate summary stats
            mean_reward = float(np.mean(self._ep_rewards[-50:])) if self._ep_rewards else 0.0
            mean_pnl = float(np.mean(self._ep_pnls[-50:])) if self._ep_pnls else 0.0
            total_pnl = float(np.sum(self._ep_pnls)) if self._ep_pnls else 0.0
            mean_win_rate = float(np.mean(self._ep_win_rates[-50:])) if self._ep_win_rates else 0.0
            max_drawdown = float(np.max(self._ep_drawdowns[-50:])) if self._ep_drawdowns else 0.0
            mean_trades = float(np.mean(self._ep_trades[-50:])) if self._ep_trades else 0.0
            total_trades = int(np.sum(self._ep_trades)) if self._ep_trades else 0
            
            # Get rolling stats from curriculum for more accurate metrics
            rolling_stats = curriculum_progress.get("rolling_stats", {})
            
            # Use curriculum rolling stats if available (more accurate)
            if rolling_stats.get("total_trades", 0) > 0:
                total_trades = rolling_stats.get("total_trades", total_trades)
                mean_win_rate = rolling_stats.get("mean_win_rate", mean_win_rate)
                mean_pnl = rolling_stats.get("mean_pnl", mean_pnl)
                max_drawdown = rolling_stats.get("max_drawdown_seen", max_drawdown)
                mean_trades = rolling_stats.get("mean_trade_count", mean_trades)
            
            # Calculate mean quality metrics from our tracked data
            # Use rolling_stats if available, otherwise our tracked values
            mean_profit_factor = rolling_stats.get("mean_profit_factor", 0.0)
            mean_r_multiple = rolling_stats.get("mean_r_multiple", 0.0)
            mean_entry_quality = rolling_stats.get("mean_entry_quality", 0.5)
            
            # Override with our tracked values if we have them (and they're non-zero)
            if self._ep_profit_factors:
                recent_pf = [x for x in self._ep_profit_factors[-50:] if x > 0]
                if recent_pf:
                    mean_profit_factor = float(np.mean(recent_pf))
            
            if self._ep_r_multiples:
                recent_rm = self._ep_r_multiples[-50:]
                if recent_rm:
                    mean_r_multiple = float(np.mean(recent_rm))
            
            if self._ep_entry_quality:
                recent_eq = self._ep_entry_quality[-50:]
                if recent_eq:
                    mean_entry_quality = float(np.mean(recent_eq))
            
            # Status calculations for dashboard
            def status_for_win_rate(wr):
                if wr >= 0.55: return "good"
                if wr >= 0.45: return "ok"
                return "bad"
            
            def status_for_drawdown(dd):
                if dd <= 0.03: return "good"
                if dd <= 0.06: return "ok"
                return "bad"
            
            def status_for_profit_factor(pf):
                if pf >= 1.5: return "good"
                if pf >= 1.0: return "ok"
                return "bad"
            
            # Get PPO diagnostics
            ppo_policy_loss = abs(self._ppo_diagnostics.get('policy_loss', 0))
            ppo_value_loss = self._ppo_diagnostics.get('value_loss', 0)
            ppo_entropy = abs(self._ppo_diagnostics.get('entropy', 0))
            ppo_kl = self._ppo_diagnostics.get('kl_divergence', 0)
            ppo_clip_fraction = self._ppo_diagnostics.get('clip_fraction', 0)
            ppo_explained_var = self._ppo_diagnostics.get('explained_variance', 0)
            ppo_learning_rate = self._ppo_diagnostics.get('learning_rate', 0)
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                
                # Progress section (for dashboard progress display)
                "progress": {
                    "timesteps": self.num_timesteps,
                    "total_timesteps": self.total_timesteps,
                    "progress_pct": 100.0 * (self.num_timesteps / max(self.total_timesteps, 1)),
                    "total_episodes": len(self._ep_rewards),
                    "eta_seconds": eta_seconds,
                },
                
                # Learning section (PPO diagnostics)
                "learning": {
                    "fps": fps,
                    "n_updates": self._n_updates,
                    "mean_reward": mean_reward,
                    "mean_reward_status": "good" if mean_reward > 0 else "ok" if mean_reward > -5 else "bad",
                    "total_pnl": total_pnl,
                    "total_pnl_status": "good" if total_pnl > 0 else "ok" if total_pnl > -1000 else "bad",
                    # PPO diagnostics
                    "policy_loss": ppo_policy_loss,
                    "value_loss": ppo_value_loss,
                    "entropy": ppo_entropy,
                    "kl_divergence": ppo_kl,
                    "clip_fraction": ppo_clip_fraction,
                    "explained_variance": ppo_explained_var,
                    "learning_rate": ppo_learning_rate,
                },
                
                # Trading section
                "trading": {
                    "total_trades": total_trades,
                    "mean_trades": mean_trades,
                    "mean_trades_status": "good" if mean_trades >= 5 else "ok" if mean_trades >= 1 else "bad",
                    "mean_win_rate": mean_win_rate * 100,  # As percentage
                    "mean_win_rate_status": status_for_win_rate(mean_win_rate),
                    "max_drawdown": max_drawdown * 100,  # As percentage
                    "max_drawdown_status": status_for_drawdown(max_drawdown),
                },
                
                # Quality section
                "quality": {
                    "mean_profit_factor": mean_profit_factor,
                    "mean_profit_factor_status": status_for_profit_factor(mean_profit_factor),
                    "mean_r_multiple": mean_r_multiple,
                    "mean_r_multiple_status": "good" if mean_r_multiple > 1 else "ok" if mean_r_multiple > 0 else "bad",
                    "mean_entry_quality": mean_entry_quality,
                    "mean_entry_quality_status": "good" if mean_entry_quality > 0.6 else "ok" if mean_entry_quality > 0.4 else "bad",
                },
                
                # Exit stats - use tracked exit reasons
                "exit_stats": {
                    "distribution": dict(self._exit_reason_counts),
                },
                
                # Curriculum data
                "curriculum_stage": self.curriculum_manager.current_stage.name if self.curriculum_manager else "N/A",
                "curriculum_stage_idx": int(self.curriculum_manager.current_stage.value) if self.curriculum_manager else 0,
                "curriculum_progress": curriculum_progress,
                "curriculum_detail": curriculum_detail,
                
                # Recent history for charts
                "recent_rewards": [float(x) for x in self._ep_rewards[-n_recent:]],
                "recent_pnls": [float(x) for x in self._ep_pnls[-n_recent:]],
                "recent_win_rates": [float(x) * 100 for x in self._ep_win_rates[-n_recent:]],  # As percentage
                "recent_drawdowns": [float(x) * 100 for x in self._ep_drawdowns[-n_recent:]],  # As percentage
                "recent_trades": [int(x) for x in self._ep_trades[-n_recent:]],
                
                # Stage history
                "stage_history": self._stage_history,
                
                # Legacy flat fields for backward compatibility
                "timesteps": self.num_timesteps,
                "total_timesteps": self.total_timesteps,
                "fps": fps,
                "progress_pct": 100.0 * (self.num_timesteps / max(self.total_timesteps, 1)),
                "total_episodes": len(self._ep_rewards),
                "mean_reward": mean_reward,
                "mean_pnl": mean_pnl,
                "total_pnl": total_pnl,
                "mean_win_rate": mean_win_rate,
                "max_drawdown": max_drawdown,
                "mean_trades": mean_trades,
                "total_trades": total_trades,
            }
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f)
                
        except Exception as e:
            import traceback
            logger.error(f"Failed to save live metrics: {e}")
            traceback.print_exc()


def create_curriculum_env(
    data: Dict[str, Dict[str, pd.DataFrame]],
    curriculum_manager: Any,  # CurriculumManager
    seed: int = 0,
    monitor_dir: Optional[str] = None,
    use_action_masking: bool = True,
) -> Any:
    """
    Create a single curriculum-wrapped environment.
    """
    if not CURRICULUM_AVAILABLE or CurriculumEnvWrapper is None:
        raise RuntimeError("Curriculum system not available")
    
    # Get current stage config
    stage_config = curriculum_manager.stage_config
    
    # Create base config with curriculum settings
    # CRITICAL: entry_quality_gate_enabled=False allows exploration in early stages
    base_config = PropFirmConfig(
        max_steps_per_episode=stage_config.max_steps_per_episode,
        entry_quality_gate_enabled=stage_config.constraints.entry_quality_gate_enabled,
        entry_quality_threshold=stage_config.constraints.entry_quality_threshold,
    )
    
    # Create base environment
    base_env = PropFirmTradingEnv(data, base_config)
    
    # Wrap with curriculum
    env = CurriculumEnvWrapper(
        env=base_env,
        curriculum_manager=curriculum_manager,
        auto_update_curriculum=True,
        verbose=True,
    )
    
    # Add monitor if requested
    if monitor_dir:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(Path(monitor_dir) / f"monitor_{seed}.csv"))
    
    # Add action masking if available
    if use_action_masking and MASKABLE_AVAILABLE and ActionMasker is not None:
        env = ActionMasker(env, _mask_fn)
    
    # Seed the environment
    try:
        env.reset(seed=seed)
    except Exception:
        pass
    
    return env


def create_curriculum_vec_envs(
    data: Dict[str, Dict[str, pd.DataFrame]],
    curriculum_manager: Any,  # CurriculumManager
    n_envs: int,
    seed: int,
    monitor_dir: str = "logs/curriculum/training",
    use_action_masking: bool = True,
    frame_stack: int = 1,
) -> VecEnv:
    """
    Create vectorized curriculum environments.
    
    All environments share the same CurriculumManager, so stage transitions
    are synchronized across all parallel environments.
    """
    Path(monitor_dir).mkdir(parents=True, exist_ok=True)

    def make_env(rank: int) -> Callable[[], Any]:
        def _init():
            env = create_curriculum_env(
                data=data,
                curriculum_manager=curriculum_manager,
                seed=seed + rank,
                monitor_dir=monitor_dir,
                use_action_masking=use_action_masking,
            )
            return env
        return _init

    if platform.system() != "Windows" and n_envs > 1:
        start_method = pick_subproc_start_method() or "spawn"
        logger.info(f"Creating {n_envs} PARALLEL curriculum envs (SubprocVecEnv)")
        venv = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method=start_method)
    else:
        logger.info(f"Creating {n_envs} curriculum envs (DummyVecEnv)")
        venv = DummyVecEnv([make_env(i) for i in range(n_envs)])

    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=int(frame_stack))

    return venv


def train_curriculum_agent(
    data: Dict[str, Dict[str, pd.DataFrame]],
    total_timesteps: int,
    n_envs: int,
    learning_rate: float,
    batch_size: int,
    n_steps: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    target_kl: float,
    policy_hidden: int,
    value_hidden: int,
    checkpoint_freq: int,
    start_stage: str = "FOUNDATION",
    resume_path: Optional[str] = None,
    frame_stack: int = 1,
) -> BaseAlgorithm:
    """
    Train with curriculum learning - progressive difficulty stages.
    
    The agent starts at FOUNDATION and earns progression to harder stages
    by demonstrating statistical competence (not just time or luck).
    """
    if not CURRICULUM_AVAILABLE or CurriculumStage is None or CurriculumManager is None:
        raise RuntimeError("Curriculum system not available. Check imports.")
    
    seed_everything(42)
    
    save_dir = Path("models/curriculum")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_masking = bool(MASKABLE_AVAILABLE)
    
    # Initialize or resume curriculum manager
    initial_stage = CurriculumStage[start_stage]
    
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming curriculum from {resume_path}")
        curriculum_manager = CurriculumManager.load(Path(resume_path))
    else:
        curriculum_manager = CurriculumManager(
            initial_stage=initial_stage,
            auto_promote=True,
            auto_demote=True,
            verbose=True,
        )
    
    logger.info(f"Curriculum training: starting at {curriculum_manager.current_stage.name}")
    if get_stage_progression is not None:
        logger.info(f"Stage progression: {' → '.join(s.name for s in get_stage_progression())}")
    
    # Create curriculum-wrapped environments
    train_env = create_curriculum_vec_envs(
        data=data,
        curriculum_manager=curriculum_manager,
        n_envs=n_envs,
        seed=42,
        monitor_dir=str(save_dir / "training"),
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )
    
    # Create model
    policy_kwargs = dict(
        net_arch=dict(
            pi=[policy_hidden, policy_hidden // 2],
            vf=[value_hidden, value_hidden // 2],
        ),
        activation_fn=nn.Tanh,
    )
    
    Algo = PPO
    if use_masking and MASKABLE_AVAILABLE and MaskablePPO is not None:
        Algo = MaskablePPO  # type: ignore[assignment]
    
    model = Algo(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log="runs/curriculum",
        device=device,
        seed=42,
    )
    
    # Create callbacks
    callbacks: List[BaseCallback] = [
        CurriculumTrainingCallback(
            curriculum_manager=curriculum_manager,
            total_timesteps=total_timesteps,
            log_interval_steps=50_000,
            save_path=str(save_dir),
        ),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="curriculum_ppo",
        ),
    ]
    
    start_time = datetime.now()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name="curriculum_ppo",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.warning("Curriculum training interrupted by user")
    finally:
        duration = datetime.now() - start_time
        logger.info(f"Training duration: {duration}")
        
        # Save final model and curriculum state
        model.save(str(save_dir / "curriculum_ppo_final.zip"))
        curriculum_manager.save(save_dir / "curriculum_state.json")
        
        # Save training summary
        summary = {
            "total_timesteps": int(getattr(model, "num_timesteps", 0)),
            "final_stage": curriculum_manager.current_stage.name,
            "curriculum_progress": curriculum_manager.get_progress_report(),
            "completed_at": datetime.now().isoformat(),
        }
        with open(save_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Final stage: {curriculum_manager.current_stage.name}")
        logger.info(f"Model saved to: {save_dir}")
        
        try:
            train_env.close()
        except Exception:
            pass
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="PropFirm PPO Training (10/10) with Walk-Forward Optuna")

    # Training mode selection
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--curriculum", action="store_true", help="Run curriculum learning (progressive difficulty)")
    
    # Optuna options
    parser.add_argument("--trials", type=int, default=25, help="Optuna trials")
    parser.add_argument("--trial-timesteps", type=int, default=750_000, help="Steps per Optuna trial")
    parser.add_argument("--optuna-storage", type=str, default=None, help="Optuna storage URL (optional)")
    parser.add_argument("--study-name", type=str, default="propfirm_ppo", help="Optuna study name")
    parser.add_argument("--walk-forward-folds", type=int, default=2, help="Walk-forward folds (Optuna)")
    
    # Curriculum options
    parser.add_argument(
        "--start-stage",
        type=str,
        default="FOUNDATION",
        help="Starting curriculum stage (FOUNDATION, DISCIPLINE, MARKET_STRUCTURE, etc.)",
    )
    parser.add_argument("--resume-curriculum", type=str, default=None, help="Resume curriculum from state file")

    parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total training timesteps")
    default_n_envs = 2 if platform.system() == "Windows" else 4
    parser.add_argument("--n-envs", type=int, default=default_n_envs, help="Number of envs")
    parser.add_argument("--test", action="store_true", help="Quick test mode")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.08)  # Safe default to prevent collapse
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)

    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--value-hidden", type=int, default=256)

    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--instruments", type=str, nargs="+", default=["XAUUSD"])

    # “memory without LSTM” (keeps masking compatibility)
    parser.add_argument("--frame-stack", type=int, default=1, help="Frame stack. 1=disabled (recommended).")

    # Config overrides
    parser.add_argument("--entry-quality-threshold", type=float, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--risk-penalty-scale", type=float, default=None)

    # Domain randomization toggle
    parser.add_argument("--no-domain-randomization", action="store_true", help="Disable domain randomization")

    # Dashboard
    parser.add_argument("--no-dashboard", action="store_true", help="Disable real-time dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8765, help="Dashboard server port")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PropFirm PPO Training System (10/10)")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"MaskablePPO available: {MASKABLE_AVAILABLE}")
    logger.info(f"Optuna available: {OPTUNA_AVAILABLE}")
    logger.info(f"Curriculum available: {CURRICULUM_AVAILABLE}")
    logger.info(f"Dashboard available: {DASHBOARD_AVAILABLE}")
    logger.info("=" * 70)

    # Start dashboard server if available and not disabled
    if DASHBOARD_AVAILABLE and not args.no_dashboard and start_dashboard_server is not None:
        try:
            start_dashboard_server(
                port=args.dashboard_port,
                metrics_file="logs/training/live_metrics.json",
                background=True,
            )
        except Exception as e:
            logger.warning(f"Could not start dashboard: {e}")
    elif not args.no_dashboard and not DASHBOARD_AVAILABLE:
        logger.info("Dashboard not available. Install: pip install fastapi uvicorn websockets")

    if args.optuna and not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not installed. Run: pip install optuna")

    logger.info("Loading market data...")
    data = load_market_data(
        data_dir=args.data_dir,
        instruments=args.instruments,
        min_bars=1000 if args.test else 5000,
    )
    if not data:
        raise RuntimeError("No data available for training")

    if args.test:
        args.timesteps = min(args.timesteps, 200_000)
        args.trial_timesteps = min(args.trial_timesteps, 80_000)
        args.checkpoint_freq = 10_000
        args.eval_freq = 10_000
        logger.info("TEST MODE enabled: reduced timesteps and frequencies")

    config_overrides: Dict[str, Any] = {}
    if args.entry_quality_threshold is not None:
        config_overrides["entry_quality_threshold"] = float(args.entry_quality_threshold)
    if args.reward_scale is not None:
        config_overrides["reward_scale"] = float(args.reward_scale)
    if args.risk_penalty_scale is not None:
        config_overrides["risk_penalty_scale"] = float(args.risk_penalty_scale)

    # Domain randomization default ON; allow disabling
    config_overrides["domain_randomization_enabled"] = (not bool(args.no_domain_randomization))

    if args.optuna:
        run_optuna_optimization(
            data=data,
            n_trials=args.trials,
            timesteps_per_trial=args.trial_timesteps,
            n_envs=args.n_envs,
            n_eval_episodes=10,
            walk_forward_folds=max(1, int(args.walk_forward_folds)),
            frame_stack=max(1, int(args.frame_stack)),
            study_name=args.study_name,
            storage=args.optuna_storage,
        )
    elif args.curriculum:
        # Curriculum learning mode
        if not CURRICULUM_AVAILABLE:
            raise RuntimeError("Curriculum system not available. Check curriculum imports.")
        
        logger.info("=" * 70)
        logger.info("CURRICULUM LEARNING MODE")
        logger.info("=" * 70)
        
        train_curriculum_agent(
            data=data,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_hidden=args.policy_hidden,
            value_hidden=args.value_hidden,
            checkpoint_freq=args.checkpoint_freq,
            start_stage=args.start_stage,
            resume_path=args.resume_curriculum,
            frame_stack=max(1, int(args.frame_stack)),
        )
    else:
        # Standard propfirm training (default)
        train_prop_firm_agent(
            data=data,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_hidden=args.policy_hidden,
            value_hidden=args.value_hidden,
            checkpoint_freq=args.checkpoint_freq,
            eval_freq=args.eval_freq,
            pretrained_path=args.pretrained,
            config_overrides=config_overrides,
            frame_stack=max(1, int(args.frame_stack)),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()

