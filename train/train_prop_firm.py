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
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import platform
import random
import sys
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack

# Environment
from envs.prop_firm_env import PropFirmConfig, PropFirmTradingEnv

# Curriculum system (optional)
try:
    from envs.curriculum import (
        CurriculumManager,
        CurriculumStage,
        CurriculumStageConfig,
        get_stage_config,
        get_stage_progression,
    )
    CURRICULUM_AVAILABLE = True
except ImportError:
    CURRICULUM_AVAILABLE = False
    CurriculumStage = None  # type: ignore
    CurriculumStageConfig = None  # type: ignore
    CurriculumManager = None  # type: ignore
    get_stage_config = None  # type: ignore
    get_stage_progression = None  # type: ignore

# Optuna
try:
    import optuna
    from optuna.exceptions import TrialPruned
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None  # type: ignore
    MedianPruner = None  # type: ignore
    TPESampler = None  # type: ignore
    TrialPruned = Exception  # type: ignore
    OPTUNA_AVAILABLE = False

# sb3-contrib (MaskablePPO + ActionMasker + MaskableEvalCallback)
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    MASKABLE_AVAILABLE = True
except Exception:
    MaskablePPO = None  # type: ignore
    ActionMasker = None  # type: ignore
    MASKABLE_AVAILABLE = False

try:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    MASKABLE_EVAL_AVAILABLE = True
except Exception:
    MaskableEvalCallback = None  # type: ignore
    MASKABLE_EVAL_AVAILABLE = False
    MASKABLE_AVAILABLE = False

# AUDIT FIX: Import official mask extraction helper (more robust than custom unwrapping)
try:
    from sb3_contrib.common.maskable.utils import get_action_masks as sb3_get_action_masks
    SB3_MASK_UTILS_AVAILABLE = True
except Exception:
    sb3_get_action_masks = None  # type: ignore
    SB3_MASK_UTILS_AVAILABLE = False

# Dashboard server (optional)
try:
    from dashboard.server import WEB_AVAILABLE as DASHBOARD_AVAILABLE
    from dashboard.server import start_dashboard_server
except ImportError:
    DASHBOARD_AVAILABLE = False
    start_dashboard_server = None  # type: ignore

# Extracted controllers and callbacks (Phase 1 modularization)
from modules.utils import simulation_time as simclock
from train.callbacks import CurriculumCheckpointCallback, CurriculumTrainingCallback, VecEpisodeTradingCallback

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

    # Determinism is not only about RNG. Any module that reads the wall clock
    # makes a run depend on when it was started - session logic differs between
    # a 02:00 and a 14:00 launch, and second-based cooldowns behave differently
    # under load. Declaring simulation mode here makes "now" a function of the
    # replayed bar instead, so a seeded run is genuinely reproducible.
    simclock.set_mode(simclock.TimeMode.SIMULATION)
    simclock.reset()
    logger.info("[CLOCK] simulation mode enabled - modules read bar time, not wall clock")


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
# CONFIG CONSTRUCTION (ROBUST OVERRIDES)
# =============================================================================

def _safe_propfirm_config_init_kwargs(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter overrides to only those accepted by PropFirmConfig.__init__.
    This prevents silent/late crashes when CLI includes legacy keys.
    """
    try:
        valid = {f.name for f in fields(PropFirmConfig)}
        return {k: v for k, v in overrides.items() if k in valid}
    except Exception:
        # PropFirmConfig may not be a dataclass in some versions; fall back conservatively
        # Keep only keys that are commonly present in the config.
        allow = {
            "initial_balance",
            "daily_drawdown_limit",
            "max_drawdown_limit",
            "max_steps_per_episode",
            "entry_quality_threshold",
            "entry_quality_gate_enabled",
            "max_trades_per_day",
            "max_trades_per_session",
            "domain_randomization_enabled",
            "spread_mult_range",
            "slippage_mult_range",
            "latency_bars_range",
            "volatility_scale_range",
        }
        return {k: v for k, v in overrides.items() if k in allow}


def build_propfirm_config(config_overrides: Dict[str, Any]) -> PropFirmConfig:
    """
    Build PropFirmConfig robustly:
    - pass only supported keys to the constructor
    - apply reward overrides explicitly to nested config.reward fields
    """
    init_kwargs = _safe_propfirm_config_init_kwargs(config_overrides)
    cfg = PropFirmConfig(**init_kwargs)

    # Apply reward overrides via nested RewardConfig when present
    # (These are frequently NOT constructor args and can be legacy CLI keys.)
    if "reward_scale" in config_overrides:
        try:
            cfg.reward.reward_scale = float(config_overrides["reward_scale"])
        except Exception:
            pass
    if "risk_penalty_scale" in config_overrides:
        try:
            cfg.reward.dd_penalty_scale = float(config_overrides["risk_penalty_scale"])
        except Exception:
            pass

    return cfg


# =============================================================================
# DATA LOADING
# =============================================================================

# Timeframe-aware minimum bars (based on ~1 year of data as baseline)
# M15: 96 bars/day * 252 trading days ≈ 24,000 → require 5000 (reasonable subset)
# H1:  24 bars/day * 252 ≈ 6,000 → require 2000
# H4:   6 bars/day * 252 ≈ 1,500 → require 500
# D1:   1 bar/day  * 252 ≈ 252   → require 200
TIMEFRAME_MIN_BARS = {
    "M1": 10000,
    "M5": 8000,
    "M15": 5000,
    "M30": 3000,
    "H1": 2000,
    "H2": 1000,
    "H4": 500,
    "H8": 300,
    "D1": 200,
    "W1": 50,
}

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

            # F-8 FIX: Normalize column names to lowercase for case-insensitive OHLC detection
            df.columns = df.columns.str.lower()

            required = {"open", "high", "low", "close"}
            if not required.issubset(df.columns):
                logger.warning(f"Skipping {file}: missing OHLC columns (have: {list(df.columns)[:10]})")
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

            # Use timeframe-specific minimum bars requirement
            tf_min_bars = TIMEFRAME_MIN_BARS.get(timeframe, min_bars)
            if len(df) < tf_min_bars:
                logger.info(f"Skipping {file}: only {len(df)} bars (need {tf_min_bars}+ for {timeframe})")
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
    for _, tfs in data.items():
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
            # IMPORTANT: Use .copy() to prevent data leakage between folds
            out[inst][tf] = df.iloc[s:e].copy().reset_index(drop=True)
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
            # Seed the environment
            try:
                env.reset(seed=seed)
            except Exception as e:
                logger.debug(f"Could not seed eval env with seed={seed}: {e}")
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
        if hasattr(current, 'action_masks') and callable(current.action_masks):
            return current.action_masks()
        current = current.env
    # Final unwrapped env should have action_masks
    if hasattr(current, 'action_masks') and callable(current.action_masks):
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
            except Exception as e:
                logger.debug(f"Could not seed env {rank}: {e}")
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

    # AUDIT FIX: Use official sb3_contrib helper when available (more robust across wrappers)
    # Falls back to custom extraction for older sb3_contrib versions
    def get_action_masks_from_vec_env(venv: VecEnv) -> Optional[np.ndarray]:
        """Extract action masks from a VecEnv."""
        # Try official helper first (works across Dummy/Subproc/FrameStack)
        if SB3_MASK_UTILS_AVAILABLE and sb3_get_action_masks is not None:
            try:
                return sb3_get_action_masks(venv)
            except Exception as e:
                logger.debug(f"sb3_get_action_masks failed, falling back: {e}")

        # Fallback: manual unwrapping (for DummyVecEnv only)
        try:
            current: Any = venv
            while hasattr(current, 'venv'):
                current = current.venv
            if hasattr(current, 'envs'):
                envs_list = current.envs
                if envs_list and len(envs_list) > 0:
                    base_env: Any = envs_list[0]
                    while hasattr(base_env, 'env'):
                        if hasattr(base_env, 'action_masks') and callable(base_env.action_masks):
                            return np.array([base_env.action_masks()])
                        base_env = base_env.env
                    if hasattr(base_env, 'action_masks') and callable(base_env.action_masks):
                        return np.array([base_env.action_masks()])
        except Exception as e:
            logger.debug(f"Could not extract action_masks from vec env: {e}")
        return None

    for _ in range(n_episodes):
        obs = env.reset()  # VecEnv.reset() returns just obs (no info)
        done = False
        ep_reward = 0.0
        steps = 0
        info: Dict[str, Any] = {}

        while not done:
            obs_array: np.ndarray = np.asarray(obs)

            # Build kwargs to keep Pylance happy (SB3 BaseAlgorithm.predict is not typed with action_masks)
            predict_kwargs: Dict[str, Any] = {"deterministic": deterministic}

            if is_maskable and use_action_masks:
                action_masks = get_action_masks_from_vec_env(env)
                if action_masks is not None:
                    predict_kwargs["action_masks"] = action_masks

            # Predict action (with safe fallback if some algo rejects unknown kwargs)
            try:
                action, _ = model.predict(obs_array, **predict_kwargs)
            except TypeError:
                action, _ = model.predict(obs_array, deterministic=deterministic)

            # VecEnv.step returns (obs, rewards, dones, infos)
            obs, r, dones, infos = env.step(action)
            ep_reward += float(r[0])
            done = bool(dones[0])
            info = infos[0] if infos else {}
            steps += 1
            if max_steps is not None and steps >= max_steps:
                done = True


        # IMPORTANT: many envs place episode-final stats into terminal_info
        finfo = info.get("terminal_info", info) if isinstance(info, dict) else {}

        pnl = float(finfo.get("total_pnl", 0.0))
        wr = float(finfo.get("win_rate", 0.0))
        dd = float(finfo.get("drawdown", 0.0))
        trades = int(finfo.get("trade_count", 0))

        # Normalize (some envs report percentages)
        if wr > 1.0:
            wr /= 100.0
        wr = float(np.clip(wr, 0.0, 1.0))

        if dd > 1.0:
            dd /= 100.0
        dd = float(np.clip(dd, 0.0, 1.0))

        rewards.append(ep_reward)
        pnls.append(pnl)
        win_rates.append(wr)
        drawdowns.append(dd)
        trade_counts.append(trades)

        term_reason = str(finfo.get("termination_reason", finfo.get("done_reason", "")))
        if "drawdown" in term_reason.lower():
            dd_breaches += 1

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "pnl_std": float(np.std(pnls)) if pnls else 0.0,
        "mean_win_rate": float(np.mean(win_rates)) if win_rates else 0.0,  # 0..1
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
# OPTUNA (WALK-FORWARD + EVAL-BASED PRUNING)
# =============================================================================
# NOTE: This Optuna setup is for finding PPO hyperparameters (not curriculum/stage params).
# It does NOT use curriculum stages - it runs on a fixed PropFirmConfig.
# For curriculum training, use --curriculum flag instead.
#
# IMPORTANT: ent_coef is FIXED at 0.10 (not tuned) because:
# 1. Curriculum training uses adaptive entropy that overrides ent_coef
# 2. Tuning ent_coef here would find values incompatible with adaptive entropy
# 3. 0.10 is a good baseline that curriculum can adjust from
# =============================================================================

def sample_ppo_hyperparams(trial: Any) -> Dict[str, Any]:
    """
    PPO hyperparameter search space for NON-CURRICULUM training.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [2048, 4096, 8192]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "n_epochs": trial.suggest_int("n_epochs", 8, 20),
        "gamma": trial.suggest_float("gamma", 0.93, 0.995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.94, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.15, 0.30),
        # Fixed: curriculum uses adaptive entropy
        "ent_coef": 0.10,
        "vf_coef": trial.suggest_float("vf_coef", 0.5, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.4, 0.8),
        "target_kl": trial.suggest_float("target_kl", 0.008, 0.05),
        "policy_hidden": trial.suggest_categorical("policy_hidden", [256, 512]),
        "value_hidden": trial.suggest_categorical("value_hidden", [256, 512]),
    }


def sample_env_hyperparams(trial: Any) -> Dict[str, Any]:
    """
    Optimized env params based on Trial 12.

    NOTE: Legacy fields have been removed. Use config.reward.* settings instead.
    """
    return {
        "reward_scale": trial.suggest_float("reward_scale", 5.0, 15.0),
        "entry_quality_threshold": trial.suggest_float("entry_quality_threshold", 0.35, 0.65),
        "dd_penalty_scale": trial.suggest_float("dd_penalty_scale", 1.5, 4.5),
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
    """
    cfg = copy.deepcopy(base)

    cfg.domain_randomization_enabled = True
    cfg.spread_mult_range = (1.0 + 0.20 * adversity, 1.0 + 0.60 * adversity)
    cfg.slippage_mult_range = (1.0 + 0.25 * adversity, 1.0 + 0.80 * adversity)
    cfg.latency_bars_range = (int(1 + 1 * adversity), int(2 + 3 * adversity))
    cfg.volatility_scale_range = (1.0, 1.0 + 0.35 * adversity)

    cfg.execution.spread_shock_enabled = True
    cfg.execution.spread_shock_probability = 0.02 + 0.03 * adversity
    cfg.execution.spread_shock_multiplier = 2.0 + 2.0 * adversity

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
    per_fold_steps = timesteps_per_trial // walk_forward_folds
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=per_fold_steps * 2,
        n_min_trials=8,
        interval_steps=per_fold_steps,
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

        base_cfg = PropFirmConfig(
            max_steps_per_episode=2000,
            entry_quality_threshold=float(env_params["entry_quality_threshold"]),
            max_trades_per_day=12,
            max_trades_per_session=6,
            domain_randomization_enabled=True,  # type: ignore[attr-defined]
        )

        base_cfg.reward.reward_scale = float(env_params["reward_scale"])
        base_cfg.reward.dd_penalty_scale = float(env_params["dd_penalty_scale"])

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

        fold_scores: List[float] = []
        per_fold_steps_local = max(25_000, int(timesteps_per_trial / max(len(folds), 1)))

        for fi, (train_data, val_data) in enumerate(folds):
            train_env: Optional[VecEnv] = None
            eval_env: Optional[VecEnv] = None
            try:
                train_env = create_vec_envs(
                    train_data, base_cfg,
                    n_envs=n_envs, seed=trial.number + 100 * fi,
                    monitor_dir=f"logs/optuna/trial_{trial.number}",
                    use_action_masking=True,
                    frame_stack=frame_stack,
                )
                try:
                    train_env.seed(1_000 + trial.number + 10 * fi)
                except Exception as e:
                    logger.debug(f"Could not seed optuna train_env: {e}")

                model = build_model(train_env)

                optuna_callback = VecEpisodeTradingCallback(
                    total_timesteps=per_fold_steps_local,
                    log_interval_steps=per_fold_steps_local + 1,
                    metrics_file=f"logs/optuna/trial_{trial.number}/live_metrics.json",
                )

                logger.info(f"[Trial {trial.number}] Fold {fi+1}/{len(folds)} - Training {per_fold_steps_local:,} steps...")
                model.learn(total_timesteps=per_fold_steps_local, progress_bar=True, callback=optuna_callback)

                eval_cfg = _make_eval_config_adversarial(base_cfg, adversity=0.7)
                eval_env = create_eval_vec_env(
                    val_data, eval_cfg,
                    seed=20_000 + trial.number + fi,
                    use_action_masking=True,
                    frame_stack=frame_stack,
                )

                metrics = evaluate_agent_trading(model, eval_env, n_episodes=n_eval_episodes, deterministic=True)
                score = score_trading_metrics(metrics, eval_episodes=n_eval_episodes)
                fold_scores.append(float(score))

                current_mean = float(np.mean(fold_scores))
                current_std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
                pessimistic_score = current_mean - 0.1 * current_std

                trial.report(pessimistic_score, (fi + 1) * per_fold_steps_local)

                if fi >= 1 and trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned after fold {fi+1}: score={pessimistic_score:.3f}")
                    raise TrialPruned()

            finally:
                try:
                    if train_env is not None:
                        train_env.close()
                except Exception as e:
                    logger.debug(f"Train env cleanup: {e}")
                try:
                    if eval_env is not None:
                        eval_env.close()
                except Exception as e:
                    logger.debug(f"Eval env cleanup: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        mean_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
        score_std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
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
    seed: int,
) -> BaseAlgorithm:
    seed_everything(seed)

    config = build_propfirm_config(config_overrides)

    test_environment(data, config)

    use_masking = bool(MASKABLE_AVAILABLE)

    # COUPLING FIX: Use separate monitor directories to avoid collisions
    train_monitor_dir = "logs/propfirm/training"

    train_env = create_vec_envs(
        data, config,
        n_envs=n_envs, seed=seed,
        monitor_dir=train_monitor_dir,
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )
    try:
        train_env.seed(seed)
    except Exception as e:
        logger.debug(f"Could not seed train_env: {e}")

    # Single eval env for best-model saving + SB3 EvalCallback
    eval_cfg = copy.deepcopy(config)
    try:
        eval_cfg.domain_randomization_enabled = False
        eval_cfg.spread_mult_range = (1.0, 1.0)
        eval_cfg.slippage_mult_range = (1.0, 1.0)
        eval_cfg.latency_bars_range = (0, 0)
        eval_cfg.volatility_scale_range = (1.0, 1.0)
    except Exception:
        pass

    eval_env_vec = create_vec_envs(
        data, eval_cfg,
        n_envs=1, seed=seed + 1337,
        monitor_dir="logs/propfirm/eval_vec",
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )

    # Adversarial single-env eval for *real trading score logging*
    eval_cfg_adv = _make_eval_config_adversarial(config, adversity=0.8)
    eval_env_single = create_eval_vec_env(
        data, eval_cfg_adv,
        seed=seed + 2337,
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )

    model_dir = Path("models/propfirm")
    checkpoint_dir = Path("checkpoints/propfirm")
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device.upper()} | n_envs={n_envs} | masking={use_masking} | frame_stack={frame_stack} | seed={seed}")

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

        # AUDIT FIX (CRIT-4): Validate observation version before continuing training
        try:
            from envs.prop_firm_env import PPO_OBS_VERSION, validate_observation_version
            model_obs_size: int = model.observation_space.shape[0]  # type: ignore[union-attr]
            saved_version = getattr(model, '_obs_version', PPO_OBS_VERSION)
            validate_observation_version(saved_version, model_obs_size)
            logger.info(f"Observation validation passed: size={model_obs_size}, version={saved_version}")
        except ValueError as e:
            logger.error(f"CRITICAL: {e}")
            raise
        except Exception as e:
            logger.warning(f"Could not validate observation version: {e}")

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
            seed=seed,
        )
        current_steps = 0

    remaining = max(0, total_timesteps - current_steps)
    logger.info(f"Target timesteps: {total_timesteps:,} | current: {current_steps:,} | remaining: {remaining:,}")

    eval_freq_calls = max(1, eval_freq // max(n_envs, 1))

    if use_masking and MASKABLE_EVAL_AVAILABLE and MaskableEvalCallback is not None:
        eval_cb = MaskableEvalCallback(
            eval_env_vec,
            best_model_save_path=str(model_dir / "best"),
            log_path="logs/propfirm/eval",
            eval_freq=eval_freq_calls,
            deterministic=True,
            n_eval_episodes=5,
        )
    else:
        eval_cb = EvalCallback(
            eval_env_vec,
            best_model_save_path=str(model_dir / "best"),
            log_path="logs/propfirm/eval",
            eval_freq=eval_freq_calls,
            deterministic=True,
            n_eval_episodes=5,
        )

    callbacks: List[BaseCallback] = [
        VecEpisodeTradingCallback(
            total_timesteps=total_timesteps,
            log_interval_steps=50_000,
            # Unify dashboard path to prevent “dashboard looks dead” surprises:
            metrics_file="logs/training/live_metrics.json",
        ),
        CheckpointCallback(
            save_freq=max(1, checkpoint_freq // max(n_envs, 1)),
            save_path=str(checkpoint_dir),
            name_prefix="propfirm_ppo",
        ),
        eval_cb,
    ]

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
            "seed": int(seed),
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
        except Exception as e:
            logger.debug(f"Train env cleanup: {e}")
        try:
            eval_env_vec.close()
        except Exception as e:
            logger.debug(f"Eval vec env cleanup: {e}")
        try:
            eval_env_single.close()
        except Exception as e:
            logger.debug(f"Eval single env cleanup: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


def create_curriculum_env(
    data: Dict[str, Dict[str, pd.DataFrame]],
    curriculum_manager: Any,  # CurriculumManager
    seed: int = 0,
    monitor_dir: Optional[str] = None,
    use_action_masking: bool = True,
) -> Any:
    """
    Create a curriculum environment using PropFirmTradingEnv's native curriculum support.
    """
    if not CURRICULUM_AVAILABLE:
        raise RuntimeError("Curriculum system not available")

    stage_config = curriculum_manager.stage_config

    base_config = PropFirmConfig(
        max_steps_per_episode=stage_config.max_steps_per_episode,
        entry_quality_gate_enabled=stage_config.constraints.entry_quality_gate_enabled,
        entry_quality_threshold=stage_config.constraints.entry_quality_threshold,
    )

    env = PropFirmTradingEnv(
        data,
        base_config,
        curriculum_manager=curriculum_manager,
        apply_curriculum_overrides=True,
    )

    if monitor_dir:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(Path(monitor_dir) / f"monitor_{seed}.csv"))

    if use_action_masking and MASKABLE_AVAILABLE and ActionMasker is not None:
        env = ActionMasker(env, _mask_fn)

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

    AUDIT FIX: Force DummyVecEnv to keep CurriculumManager truly shared.
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

    if n_envs > 1 and platform.system() != "Windows":
        logger.warning(
            "Curriculum mode: forcing DummyVecEnv to keep a truly shared CurriculumManager. "
            "SubprocVecEnv would desynchronize stage transitions."
        )

    logger.info(f"Creating {n_envs} curriculum envs (DummyVecEnv - shared CurriculumManager)")
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
    start_stage: str = "EXPLORER",
    resume_path: Optional[str] = None,
    load_model_path: Optional[str] = None,
    load_controllers_path: Optional[str] = None,
    load_metrics_path: Optional[str] = None,
    frame_stack: int = 1,
    goal_based_stopping: bool = False,
    max_hours: Optional[float] = None,
    plateau_threshold_episodes: int = 500,
    max_demotions_from_same_stage: int = 5,
    mastery_confirmation_episodes: int = 100,
    seed: int = 42,
    holdout_ratio: float = 0.15,
) -> BaseAlgorithm:
    """
    Train with curriculum learning - progressive difficulty stages.
    """
    if not CURRICULUM_AVAILABLE or CurriculumStage is None or CurriculumManager is None:
        raise RuntimeError("Curriculum system not available. Check imports.")

    seed_everything(seed)

    save_dir = Path("models/curriculum")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_masking = bool(MASKABLE_AVAILABLE)

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
            rng_seed=seed,
        )

    logger.info(f"Curriculum training: starting at {curriculum_manager.current_stage.name}")
    if get_stage_progression is not None:
        logger.info(f"Stage progression: {' → '.join(s.name for s in get_stage_progression())}")

    # ---------------------------------------------------------------------
    # Holdout split (unseen tail of data) for real validation/stress gating
    # ---------------------------------------------------------------------
    n_bars = _min_len_across(data)
    if n_bars <= 0:
        raise ValueError("No bars available for curriculum training")

    try:
        holdout_ratio = float(holdout_ratio)
    except Exception:
        holdout_ratio = 0.0
    if not np.isfinite(holdout_ratio):
        holdout_ratio = 0.0
    holdout_ratio = float(np.clip(holdout_ratio, 0.0, 0.50))

    # Use the maximum stage episode length so holdout remains valid all the way
    # to LIVE_READY (later stages have longer episodes).
    stage_max_steps = int(getattr(curriculum_manager.stage_config, "max_steps_per_episode", 0) or 0)
    max_steps_required = stage_max_steps
    if get_stage_progression is not None and get_stage_config is not None:
        try:
            max_steps_required = max(
                int(get_stage_config(s).max_steps_per_episode) for s in get_stage_progression()
            )
        except Exception:
            max_steps_required = stage_max_steps
    # Heuristic: ensure holdout can support buffer + varied start indices.
    min_holdout_len = max(1500, max_steps_required + 800)
    holdout_len = int(max(min_holdout_len, int(n_bars * holdout_ratio))) if holdout_ratio > 0 else 0

    min_train_len = max(3000, max_steps_required + 2000)
    if holdout_len <= 0 or (n_bars - holdout_len) < min_train_len:
        if holdout_len > 0:
            logger.warning(
                "Holdout disabled: insufficient bars for requested holdout "
                f"(n={n_bars}, holdout_len={holdout_len}, min_train_len={min_train_len}). "
                "Using full data for training/evaluation."
            )
        train_data = data
        holdout_data = data
        holdout_enabled = False
    else:
        train_end = int(n_bars - holdout_len)
        train_data = slice_data_by_index(data, 0, train_end)
        holdout_data = slice_data_by_index(data, train_end, n_bars)
        holdout_enabled = True
        logger.info(
            f"Holdout split: train=[0:{train_end}] ({train_end} bars), "
            f"holdout=[{train_end}:{n_bars}] ({holdout_len} bars, ratio={holdout_ratio:.0%})"
        )

    train_env = create_curriculum_vec_envs(
        data=train_data,
        curriculum_manager=curriculum_manager,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(save_dir / "training"),
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )

    # Evaluation environments on HOLDOUT data:
    # - eval_env: used by SB3 EvalCallback for model selection tracking
    # - holdout_validation_env: dedicated for curriculum validation/stress gates
    eval_env = create_curriculum_vec_envs(
        data=holdout_data,
        curriculum_manager=curriculum_manager,
        n_envs=1,
        seed=seed + 1337,
        monitor_dir=str(save_dir / "holdout_eval"),
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )
    holdout_validation_env = create_curriculum_vec_envs(
        data=holdout_data,
        curriculum_manager=curriculum_manager,
        n_envs=1,
        seed=seed + 4242,
        monitor_dir=str(save_dir / "holdout_validation"),
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )

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

    if load_model_path and Path(load_model_path).exists():
        logger.info(f"Loading model from: {load_model_path}")
        model = Algo.load(
            load_model_path,
            env=train_env,
            device=device,
        )

        loaded_n_steps = getattr(model, 'n_steps', n_steps)
        loaded_batch_size = getattr(model, 'batch_size', batch_size)
        if loaded_n_steps != n_steps:
            logger.warning(
                f"Loaded model n_steps={loaded_n_steps} differs from CLI n_steps={n_steps}. "
                f"Keeping loaded value to avoid rollout buffer mismatch."
            )
        if loaded_batch_size != batch_size:
            logger.warning(
                f"Loaded model batch_size={loaded_batch_size} differs from CLI batch_size={batch_size}. "
                f"Keeping loaded value."
            )

        # PRESERVE saved model hyperparameters instead of overwriting
        # The model's learned entropy/lr state should be continued, not reset
        saved_lr = getattr(model, 'learning_rate', learning_rate)
        if callable(saved_lr):
            saved_lr = saved_lr(1.0)  # Get current value from schedule
        saved_ent_coef = getattr(model, 'ent_coef', ent_coef)
        saved_clip_range = getattr(model, 'clip_range', clip_range)
        if callable(saved_clip_range):
            saved_clip_range = saved_clip_range(1.0)
        
        # Get optimizer lr for logging
        try:
            optimizer_lr = model.policy.optimizer.param_groups[0]["lr"]
        except Exception:
            optimizer_lr = saved_lr
        
        logger.info(
            f"PRESERVING saved model state: lr={saved_lr:.2e} (optimizer: {optimizer_lr:.2e}), "
            f"ent_coef={saved_ent_coef:.4f}, clip_range={saved_clip_range:.3f}"
        )
        logger.info(
            f"CLI values (NOT applied): lr={learning_rate:.2e}, ent_coef={ent_coef:.4f}, "
            f"clip_range={clip_range:.3f}"
        )

        try:
            from envs.prop_firm_env import PPO_OBS_VERSION, validate_observation_version
            model_obs_size: int = model.observation_space.shape[0]  # type: ignore[union-attr]
            saved_version = getattr(model, '_obs_version', PPO_OBS_VERSION)
            validate_observation_version(saved_version, model_obs_size)
            logger.info(f"Observation validation passed: size={model_obs_size}, version={saved_version}")
        except ValueError as e:
            logger.error(f"CRITICAL: {e}")
            raise
        except Exception as e:
            logger.warning(f"Could not validate observation version: {e}")

        logger.info(f"Model loaded successfully. Previous timesteps: {model.num_timesteps}")
    else:
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
            seed=seed,
        )

    save_freq_calls = max(1, checkpoint_freq // max(n_envs, 1))

    curriculum_training_callback = CurriculumTrainingCallback(
        curriculum_manager=curriculum_manager,
        total_timesteps=total_timesteps,
        log_interval_steps=50_000,
        save_path=str(save_dir),
        metrics_file="logs/training/live_metrics.json",
        base_ent_coef=ent_coef,
        base_clip_range=clip_range,
        goal_based_stopping=goal_based_stopping,
        max_hours=max_hours,
        plateau_stop=True,
        plateau_threshold_episodes=plateau_threshold_episodes,
        max_demotions_from_same_stage=max_demotions_from_same_stage,
        mastery_confirmation_episodes=mastery_confirmation_episodes,
    )
    
    # Load controller states if resuming
    if load_controllers_path and Path(load_controllers_path).exists():
        curriculum_training_callback.load_controller_states(Path(load_controllers_path))
    
    # Load metrics state (episode history) if resuming
    if load_metrics_path and Path(load_metrics_path).exists():
        curriculum_training_callback.load_metrics_state(Path(load_metrics_path))

    # Use a dedicated holdout env for promotion gates (validation + stress),
    # so scenario overrides do not interfere with EvalCallback's environment.
    curriculum_training_callback._eval_env = holdout_validation_env

    callbacks: List[BaseCallback] = [
        curriculum_training_callback,
        CheckpointCallback(
            save_freq=save_freq_calls,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="curriculum_ppo",
        ),
        CurriculumCheckpointCallback(
            curriculum_manager=curriculum_manager,
            save_freq=save_freq_calls,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="curriculum_state",
        ),
    ]

    if use_masking and MASKABLE_EVAL_AVAILABLE and MaskableEvalCallback is not None:
        eval_cb = MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best"),
            log_path=str(save_dir / "eval_logs"),
            eval_freq=max(50_000 // max(n_envs, 1), save_freq_calls),
            deterministic=True,
            n_eval_episodes=5,
        )
    else:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best"),
            log_path=str(save_dir / "eval_logs"),
            eval_freq=max(50_000 // max(n_envs, 1), save_freq_calls),
            deterministic=True,
            n_eval_episodes=5,
        )
    callbacks.append(eval_cb)

    current_steps = int(getattr(model, 'num_timesteps', 0))
    remaining_timesteps = max(0, total_timesteps - current_steps)
    should_reset_num_timesteps = (current_steps == 0)

    if current_steps > 0:
        logger.info(
            f"Resuming from {current_steps:,} timesteps. "
            f"Remaining: {remaining_timesteps:,} (total target: {total_timesteps:,})"
        )

    start_time = datetime.now()
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name="curriculum_ppo",
            progress_bar=True,
            reset_num_timesteps=should_reset_num_timesteps,
        )
    except KeyboardInterrupt:
        logger.warning("Curriculum training interrupted by user")
    finally:
        duration = datetime.now() - start_time
        logger.info(f"Training duration: {duration}")

        model.save(str(save_dir / "curriculum_ppo_final.zip"))
        curriculum_manager.save(save_dir / "curriculum_state.json")

        summary = {
            "total_timesteps": int(getattr(model, "num_timesteps", 0)),
            "final_stage": curriculum_manager.current_stage.name,
            "curriculum_progress": curriculum_manager.get_progress_report(),
            "completed_at": datetime.now().isoformat(),
            "seed": int(seed),
            "holdout": {
                "enabled": bool(holdout_enabled),
                "ratio": float(holdout_ratio),
                "min_bars": int(n_bars),
            },
        }
        with open(save_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Final stage: {curriculum_manager.current_stage.name}")
        logger.info(f"Model saved to: {save_dir}")

        try:
            train_env.close()
        except Exception:
            pass

        try:
            eval_env.close()
        except Exception:
            pass

        try:
            holdout_validation_env.close()
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
        default="EXPLORER",
        help="Starting curriculum stage (EXPLORER, EXPERIMENTER, TREND_STUDENT, SESSION_STUDENT, TIMING_STUDENT, INTEGRATOR, RISK_MANAGER, STRATEGIST, PROFESSIONAL, LIVE_READY)",
    )
    parser.add_argument("--resume-curriculum", type=str, default=None, help="Resume curriculum from state file")
    parser.add_argument("--load-model", type=str, default=None, help="Load model weights from checkpoint (.zip file)")
    parser.add_argument("--load-controllers", type=str, default=None, help="Load controller states from checkpoint (controllers_*.json file)")
    parser.add_argument("--load-metrics", type=str, default=None, help="Load episode history from checkpoint (metrics_*.json file)")
    parser.add_argument(
        "--goal-based",
        action="store_true",
        help="Train until LIVE_READY achieved (not fixed timesteps). Set --timesteps high as safety cap.",
    )
    parser.add_argument("--max-hours", type=float, default=None, help="Maximum training time in hours (goal-based stopping)")
    parser.add_argument("--plateau-episodes", type=int, default=500, help="Episodes without stage progress to trigger plateau stop (goal-based)")
    parser.add_argument("--mastery-episodes", type=int, default=100, help="Episodes at LIVE_READY to confirm completion (goal-based)")
    parser.add_argument(
        "--load-optuna-params",
        type=str,
        default=None,
        help="Load best hyperparams from Optuna JSON file (e.g., logs/optuna/best_params.json)",
    )

    parser.add_argument("--timesteps", type=int, default=100_000_000, help="Total training timesteps (safety cap - goal-based stops earlier)")
    default_n_envs = 2 if platform.system() == "Windows" else 4
    parser.add_argument("--n-envs", type=int, default=default_n_envs, help="Number of envs")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 = time-based random seed for exploration variance)")
    parser.add_argument("--test", action="store_true", help="Quick test mode")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.10)
    parser.add_argument("--vf-coef", type=float, default=0.7)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)

    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--value-hidden", type=int, default=256)

    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--instruments", type=str, nargs="+", default=["XAUUSD"])

    parser.add_argument("--frame-stack", type=int, default=1, help="Frame stack. 1=disabled (recommended).")
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.15,
        help="Fraction of latest bars reserved for holdout evaluation (curriculum validation/stress gates). 0 disables.",
    )

    parser.add_argument("--entry-quality-threshold", type=float, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--risk-penalty-scale", type=float, default=None)

    parser.add_argument("--no-domain-randomization", action="store_true", help="Disable domain randomization")

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

    # Resolve seed: -1 means time-based random seed for exploration variance
    if args.seed == -1:
        args.seed = int(time.time() * 1000) % (2**31)
        logger.info(f"Using time-based random seed: {args.seed}")
    else:
        logger.info(f"Using fixed seed: {args.seed} (reproducible)")

    # Start dashboard server if available and not disabled
    # Unify to one metrics path for both training modes.
    metrics_path_for_dashboard = "logs/training/live_metrics.json"
    if DASHBOARD_AVAILABLE and not args.no_dashboard and start_dashboard_server is not None:
        try:
            start_dashboard_server(
                port=args.dashboard_port,
                metrics_file=metrics_path_for_dashboard,
                background=True,
            )
        except Exception as e:
            logger.warning(f"Could not start dashboard: {e}")
    elif not args.no_dashboard and not DASHBOARD_AVAILABLE:
        logger.info("Dashboard not available. Install: pip install fastapi uvicorn websockets")

    config_overrides: Dict[str, Any] = {}
    if args.entry_quality_threshold is not None:
        config_overrides["entry_quality_threshold"] = float(args.entry_quality_threshold)
    if args.reward_scale is not None:
        config_overrides["reward_scale"] = float(args.reward_scale)
    if args.risk_penalty_scale is not None:
        config_overrides["risk_penalty_scale"] = float(args.risk_penalty_scale)

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
        if not CURRICULUM_AVAILABLE:
            raise RuntimeError("Curriculum system not available. Check curriculum imports.")

        if args.load_optuna_params:
            optuna_params_path = Path(args.load_optuna_params)
            if optuna_params_path.exists():
                with open(optuna_params_path, "r", encoding="utf-8") as f:
                    optuna_best = json.load(f)
                logger.info(f"Loaded Optuna best params from {optuna_params_path}")

                param_map = {
                    "learning_rate": "lr",
                    "batch_size": "batch_size",
                    "n_steps": "n_steps",
                    "n_epochs": "n_epochs",
                    "gamma": "gamma",
                    "gae_lambda": "gae_lambda",
                    "clip_range": "clip_range",
                    "vf_coef": "vf_coef",
                    "max_grad_norm": "max_grad_norm",
                    "target_kl": "target_kl",
                    "policy_hidden": "policy_hidden",
                    "value_hidden": "value_hidden",
                }
                for optuna_key, args_key in param_map.items():
                    if optuna_key in optuna_best:
                        old_val = getattr(args, args_key)
                        new_val = optuna_best[optuna_key]
                        setattr(args, args_key, new_val)
                        logger.info(f"  Optuna override: {args_key} {old_val} -> {new_val}")
                if "ent_coef" in optuna_best:
                    logger.info(f"  Optuna ent_coef={optuna_best['ent_coef']} ignored (curriculum uses adaptive entropy)")
            else:
                logger.warning(f"Optuna params file not found: {optuna_params_path}")

        logger.info("=" * 70)
        logger.info("CURRICULUM LEARNING MODE")
        if args.goal_based:
            logger.info("GOAL-BASED STOPPING ENABLED")
            logger.info(f"Training until mastery achieved ({args.mastery_episodes} episodes to confirm)")
            logger.info(f"Safety cap: {args.timesteps:,} timesteps")
            if args.max_hours:
                logger.info(f"Max hours: {args.max_hours}")
            logger.info(f"Plateau threshold: {args.plateau_episodes} episodes")
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
            load_model_path=args.load_model,
            load_controllers_path=args.load_controllers,
            load_metrics_path=args.load_metrics,
            frame_stack=max(1, int(args.frame_stack)),
            goal_based_stopping=args.goal_based,
            max_hours=args.max_hours,
            plateau_threshold_episodes=args.plateau_episodes,
            mastery_confirmation_episodes=args.mastery_episodes,
            seed=args.seed,
            holdout_ratio=args.holdout_ratio,
        )
    else:
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
            seed=args.seed,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
