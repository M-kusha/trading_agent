#!/usr/bin/env python3
"""
Complete Modern SmartInfoBus v4.0 Training Script
=================================================

Production-ready with data provider abstraction and richer telemetry.
Pylance-safe: no type identity clashes with external modules.

Architecture Notes (v4.0):
- This script trains using Stable Baselines3 (SB3) PPO
- ModernTradingEnv uses PPOObservationBuilder for 64-dim observations
- Trained models are saved in SB3 format (.zip files)
- For live trading, PPOAgentShell loads these models via SB3.load()
  or uses its own PPOCore for online learning

Observation Schema (PPO_OBS_SIZE = 64):
- Uses unified observation builder from modules.meta.ppo_observation_builder
- Same 64-dim schema in training (here) and live (PPOAgentShell)
- v4.0 adds world model predictions (8 dims) and trading mode state (8 dims)

Dashboard:
- Starts web dashboard on http://localhost:8765 automatically
- Shares same InfoBus instance for real-time data
"""

from __future__ import annotations

import os
import sys
import platform
import argparse
from datetime import datetime
from threading import Thread
from typing import Dict, Any, Optional, Protocol, Callable
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure repository root is on sys.path when running as a script
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
except Exception:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from config import get_logger, load_app_config, setup_logging
from envs.modern_env import ModernTradingEnv
from envs.exploration_env import ExplorationTradingEnv, ExplorationConfig
from envs.config import TradingConfig
from modules.core.module_system import ModuleOrchestrator

# Global flags to track env mode (set by main())
_EXPLORATION_MODE = False
_USE_PREBAKED = False

# ───────────────────────────────────────────────────────────────────
# Simulation Time Initialization (v6.0)
# ───────────────────────────────────────────────────────────────────

# Logging / InfoBus (treat external types as Any to avoid collisions)
# ───────────────────────────────────────────────────────────────────
from typing import Any
ENHANCED_LOGGING = False
RotatingLogger_Cls: Any = None
format_operator_message_func: Any = None

try:
    from modules.utils.audit_utils import RotatingLogger as _RotatingLogger, format_operator_message as _format_operator_message  # type: ignore
    RotatingLogger_Cls = _RotatingLogger
    format_operator_message_func = _format_operator_message
    ENHANCED_LOGGING = True
except Exception:
    class _FallbackRotatingLogger:
        def __init__(self, **_kw): pass
        def info(self, *_a, **_k): pass
        def warning(self, *_a, **_k): pass
        def error(self, *_a, **_k): print("ERROR:", *_a)
        def critical(self, *_a, **_k): print("CRITICAL:", *_a)

    def _fallback_format_operator_message(**kw): return f"[{kw.get('icon','')}] {kw.get('message','')}"
    RotatingLogger_Cls = _FallbackRotatingLogger
    format_operator_message_func = _fallback_format_operator_message

# Robust InfoBus fallback (training must run even without modules/)
# Declare as Any to avoid type identity clashes between local fallback class and external import
InfoBusManager: Any
try:
    from modules.utils.info_bus import InfoBusManager as _InfoBusManager  # type: ignore
    InfoBusManager = _InfoBusManager  # type: ignore[assignment]
except Exception:
    # Match the fallback used in the callback for consistency
    class _FallbackSmartBus:
        def __init__(self):
            self._d: Dict[str, Any] = {}
        def set(self, k, v, module=None, thesis=None): self._d[k] = v
        def get(self, k, module=None, default=None): return self._d.get(k, default)
        @property
        def _data_store(self): return dict(self._d)
        def export_session(self, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            import json
            with open(path, "w") as f: json.dump(self._d, f, indent=2)

    class _FallbackInfoBusManager:
        @staticmethod
        def get_instance(): return _FallbackSmartBus()
    InfoBusManager = _FallbackInfoBusManager  # type: ignore[assignment]

# Optional dependency inspector (typed Any to avoid arg-type issues)
DependencyInspector_Cls: Any = None
try:
    from modules.monitoring.system_integrity_suite import DependencyInspector as _DependencyInspector  # type: ignore
    DependencyInspector_Cls = _DependencyInspector
except Exception:
    DependencyInspector_Cls = None

# Enhanced callback
from train.enhanced_training_callback import ModernEnhancedTrainingCallback

# Dashboard server (runs in background thread, shares InfoBus instance)
StartDashboardServer = Callable[..., Optional[Thread]]
start_dashboard_server: StartDashboardServer
try:
    from traindashboard.server import start_dashboard_server
    import webbrowser
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    webbrowser = None
    def start_dashboard_server(*args: Any, **kwargs: Any) -> Optional[Thread]:
        print("[WARN] Dashboard not available - traindashboard package not found")
        return None

# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────
def _create_rotating_logger_safe(name: str, default_logger_name: str = "ModernPPOTraining"):
    """
    Create RotatingLogger with log_path if available; otherwise gracefully fall back.
    """
    if ENHANCED_LOGGING:
        try:
            os.makedirs("logs/rotate_logger/training", exist_ok=True)
            return RotatingLogger_Cls(
                name=name,
                log_path=os.path.join("logs/rotate_logger/training", f"{name}_{datetime.now():%Y%m%d}.log"),
                max_lines=2000,
                operator_mode=True,
            )
        except Exception:
            pass
    # Fallback: stdlib logger
    lg = logging.getLogger(default_logger_name)
    if not lg.handlers:
        h = logging.StreamHandler()
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


# ───────────────────────────────────────────────────────────────────
# DATA PROVIDERS
# ───────────────────────────────────────────────────────────────────
class DataProvider(Protocol):
    def load(self, config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
        ...

class FileDataProvider:
    def load(self, config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
        data_dir = getattr(config, "data_dir", "data/processed")
        if not os.path.exists(data_dir):
            print(f"[WARN] Data dir not found: {data_dir}; using dummy data")
            return create_dummy_data(config)
        data: Dict[str, Dict[str, pd.DataFrame]] = {}
        for file in os.listdir(data_dir):
            if not file.endswith(".csv"):
                continue
            try:
                p = os.path.join(data_dir, file)
                # Robust filename parsing: instrument = everything except last segment if last is a timeframe code
                base = file[:-4] if file.lower().endswith('.csv') else file
                base = base.replace("_features", "")
                parts = base.split("_")
                tf_candidates = {"M1","M5","M15","M30","H1","H2","H4","H8","D1","W1","MN1"}
                if len(parts) >= 2 and parts[-1].upper() in tf_candidates:
                    timeframe = parts[-1].upper()
                    instrument = "_".join(parts[:-1])
                else:
                    # No recognizable timeframe suffix; default to H1
                    instrument = "_".join(parts)
                    timeframe = "H1"
                df = pd.read_csv(p)
                req = {"open", "high", "low", "close"}
                if not req.issubset(df.columns):
                    print(f"[WARN] Skip {file}: missing OHLC")
                    continue
                if "volume" not in df.columns:
                    df["volume"] = 1.0
                # Coerce to float32 for env compatibility
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)
                data.setdefault(instrument, {})[timeframe] = df
            except Exception as e:
                print(f"[FAIL] Error loading {file}: {e}")

        if not data:
            print("[WARN] No usable CSVs; using dummy data")
            return create_dummy_data(config)

        # Filter out small timeframes that would limit episode length
        min_bars_for_training = getattr(config, "min_required_data_bars", 2000)
        filtered_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        for inst, tfs in data.items():
            filtered_tfs = {}
            for tf, df in tfs.items():
                if len(df) >= min_bars_for_training:
                    filtered_tfs[tf] = df
                else:
                    print(f"[FILTER] Excluding {inst}/{tf}: only {len(df)} bars (need {min_bars_for_training}+)")
            if filtered_tfs:
                filtered_data[inst] = filtered_tfs

        if not filtered_data:
            print("[WARN] No data survived filtering; using original data with warning")
            filtered_data = data
        else:
            data = filtered_data

        total_bars = sum(len(df) for d in data.values() for df in d.values())
        print(f"[SUMMARY] Loaded {len(data)} instruments, {total_bars:,} bars")
        return data

class OrchestratorDataProvider:
    """
    Pulls data via SmartInfoBus if upstream has preloaded frames.
    Expected keys: market_data_{instrument}_{timeframe} -> dict with arrays
    """
    def load(self, config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
        bus = InfoBusManager.get_instance()
        data: Dict[str, Dict[str, pd.DataFrame]] = {}
        instruments = getattr(config, "instruments", []) or []
        timeframes = getattr(config, "timeframes", None) or ["M15", "H1", "H4", "D1"]
        missing = []
        # Treat very short bus series as unusable for training; fall back to files for those
        min_bus_len = 100  # threshold to avoid 3-5 step episodes when bus publishes tiny windows

        for inst in instruments:
            for tf in timeframes:
                key = f"market_data_{inst}_{tf}"
                blob = None
                try:
                    blob = bus.get(key, module="TrainingScript")
                except Exception:
                    blob = None
                # Validate presence and minimal length
                if not blob:
                    missing.append((inst, tf))
                    continue
                try:
                    close_arr = np.asarray(blob.get("close", []))
                    if close_arr.size < min_bus_len:
                        missing.append((inst, tf))
                        continue
                    df = pd.DataFrame({
                        "open":  np.asarray(blob["open"], dtype=np.float32),
                        "high":  np.asarray(blob["high"], dtype=np.float32),
                        "low":   np.asarray(blob["low"],  dtype=np.float32),
                        "close": np.asarray(blob["close"], dtype=np.float32),
                        "volume": np.asarray(blob.get("volume", np.ones(close_arr.size)), dtype=np.float32),
                    })
                    data.setdefault(inst, {})[tf] = df
                except Exception:
                    missing.append((inst, tf))

        if missing:
            print(f"[WARN] Missing {len(missing)} pairs on bus; falling back to files for them")
            file_data = FileDataProvider().load(config)
            for inst, tf in missing:
                if inst in file_data and tf in file_data[inst]:
                    data.setdefault(inst, {})[tf] = file_data[inst][tf]

        if not data:
            print("[WARN] Orchestrator provided no data; using files")
            return FileDataProvider().load(config)

        total_bars = sum(len(df) for d in data.values() for df in d.values())
        print(f"[SUMMARY] Orchestrator data: {len(data)} instruments, {total_bars:,} bars")
        return data

def resolve_data_provider(source: str) -> DataProvider:
    if source == "files":
        return FileDataProvider()
    if source == "orchestrator":
        return OrchestratorDataProvider()
    # auto
    try:
        bus = InfoBusManager.get_instance()
        store = getattr(bus, "_data_store", {})
        any_key = any(str(k).startswith("market_data_") for k in store.keys())
        return OrchestratorDataProvider() if any_key else FileDataProvider()
    except Exception:
        return FileDataProvider()

# ───────────────────────────────────────────────────────────────────
# DUMMY DATA (for quick tests)
# ───────────────────────────────────────────────────────────────────
def create_dummy_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    dummy: Dict[str, Dict[str, pd.DataFrame]] = {}
    instruments = getattr(config, "instruments", None) or ["EURUSD"]
    for instrument in instruments:
        base = 1.10 if "EUR" in instrument else (1800.0 if "XAU" in instrument else 1.0)
        vol = 0.01 if "EUR" in instrument else (0.02 if "XAU" in instrument else 0.015)
        n = 2000
        rng = np.random.default_rng(42 + hash(instrument) % 1000)
        returns = rng.normal(0.0, vol, n)
        prices = base * np.exp(np.cumsum(returns))
        opens = prices[:-1]
        closes = prices[1:]
        spread = rng.uniform(0.0005, 0.002, len(closes)) * closes
        highs = np.maximum(opens, closes) + spread
        lows = np.minimum(opens, closes) - spread
        df = pd.DataFrame({
            "open": opens.astype(np.float32),
            "high": highs.astype(np.float32),
            "low": lows.astype(np.float32),
            "close": closes.astype(np.float32),
            "volume": rng.integers(100, 2000, len(closes)).astype(np.float32),
        })
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        dummy[instrument] = {"H1": df}
    return dummy

# ───────────────────────────────────────────────────────────────────
# ENV / MODEL BUILDERS
# ───────────────────────────────────────────────────────────────────

def _create_env_instance(data: Dict, config: TradingConfig):
    """Create the appropriate environment based on mode."""
    global _EXPLORATION_MODE, _USE_PREBAKED
    if _EXPLORATION_MODE:
        # Use lightweight exploration env (no modules)
        exploration_config = ExplorationConfig(
            initial_balance=float(getattr(config, "initial_balance", 100_000)),
            instruments=getattr(config, "instruments", None) or ["EURUSD", "XAUUSD"],
            primary_timeframe=getattr(config, "primary_timeframe", "M15"),
            gamma=float(getattr(config, "gamma", 0.95)),
            direction_long_threshold=float(getattr(config, "direction_long_threshold", 0.3)),
            direction_short_threshold=float(getattr(config, "direction_short_threshold", -0.3)),
            max_steps_per_episode=int(getattr(config, "max_steps", 2000)),
            use_prebaked_signals=_USE_PREBAKED,
            prebaked_dir="data/prebaked",
        )
        return ExplorationTradingEnv(data, exploration_config)
    else:
        # Use full ModernTradingEnv with modules
        return ModernTradingEnv(data, config)


def test_environment_creation(data: Dict, config: TradingConfig) -> bool:
    try:
        print("[TOOL] Testing environment creation...")
        env = _create_env_instance(data, config)
        obs, _ = env.reset(seed=getattr(config, "init_seed", 42))
        if not isinstance(obs, np.ndarray) or not np.all(np.isfinite(obs)):
            raise ValueError("Invalid observation")
        env.step(env.action_space.sample())
        env.close()
        env_type = "ExplorationTradingEnv" if _EXPLORATION_MODE else "ModernTradingEnv"
        print(f"[OK] Environment test passed ({env_type})")
        return True
    except Exception as e:
        print(f"[FAIL] Env test failed: {e}")
        return False

def create_environments(data: Dict, config: TradingConfig, n_envs: int = 1, seed: int = 42):
    global _EXPLORATION_MODE
    
    if not test_environment_creation(data, config):
        raise RuntimeError("Environment creation test failed")

    # Respect requested env count
    requested_envs = max(1, int(getattr(config, "num_envs", n_envs)))
    n_envs = requested_envs
    
    # Determine vectorization strategy
    # SubprocVecEnv requires 'fork' on Linux (not Windows) to avoid pickling issues
    # With fork, child processes inherit parent memory, so unpicklable objects work
    use_subproc = False
    
    if getattr(config, "live_mode", False):
        n_envs = 1
        print("[TOOL] Live mode: using single environment")
    elif n_envs > 1 and platform.system() != "Windows":
        # Use SubprocVecEnv with fork for true parallelism on Linux/Mac
        # fork() clones parent memory - no pickling needed for closure variables
        use_subproc = True
        env_type = "ExplorationEnv" if _EXPLORATION_MODE else "ModernEnv+Modules"
        print(f"[TOOL] {env_type} + SubprocVecEnv: {n_envs} PARALLEL environments (fast!)")
    elif n_envs > 1:
        print(f"[TOOL] DummyVecEnv: {n_envs} sequential environments (Windows)")
    else:
        print(f"[TOOL] Using {n_envs} environment(s)")

    def make(rank: int):
        def _init():
            env = _create_env_instance(data, config)
            Path("logs/training").mkdir(parents=True, exist_ok=True)
            return Monitor(env, filename=f"logs/training/monitor_{rank}.csv", info_keywords=())
        set_random_seed(seed + rank)
        return _init

    if use_subproc:
        # SubprocVecEnv with fork for true parallelism
        # fork() copies parent process memory - closures work without pickling
        # ModernTradingEnv has lazy init for bus/orchestrator (created in subprocess after fork)
        env = SubprocVecEnv([make(i) for i in range(n_envs)], start_method='fork')
    else:
        # DummyVecEnv for Windows or single env (sequential, same process)
        env = DummyVecEnv([make(i) for i in range(n_envs)])
    
    if hasattr(env, "seed"):
        env.seed(seed)
    return env

def create_ppo_model(env, config: TradingConfig):
    policy_kwargs = dict(
        net_arch=dict(
            pi=[config.policy_hidden_size, max(8, config.policy_hidden_size // 2)],
            vf=[config.value_hidden_size, max(8, config.value_hidden_size // 2)],
        ),
        activation_fn=nn.Tanh,
        normalize_images=False,
    )
    os.makedirs(getattr(config, "tensorboard_dir", "runs"), exist_ok=True)
    model = PPO(
        "MlpPolicy",
        env,
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
        target_kl=getattr(config, "target_kl", None),
        verbose=0,
        tensorboard_log=getattr(config, "tensorboard_dir", "runs"),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        seed=getattr(config, "init_seed", 42),
    )
    print(f"[BOT] PPO model on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    return model

# ───────────────────────────────────────────────────────────────────
# TRANSFER LEARNING VALIDATION
# ───────────────────────────────────────────────────────────────────
def _validate_pretrained_model(model_path: str, config: TradingConfig) -> bool:
    """Validate that a pretrained model is compatible with the current config."""
    import json
    
    # Check for metadata file from simple mode training
    metadata_path = model_path.replace(".zip", "_metadata.json").replace(
        "simple_ppo_final", "simple_ppo_metadata"
    )
    # Also try the direct metadata path
    model_dir = os.path.dirname(model_path)
    alt_metadata_path = os.path.join(model_dir, "simple_ppo_metadata.json")
    
    for meta_path in [metadata_path, alt_metadata_path]:
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
                
                # Check observation size
                model_obs = metadata.get("observation_size", 256)
                config_obs = getattr(config, "environment_observation_size", 256)
                if model_obs != config_obs:
                    print(f"[WARN] Obs size mismatch: model={model_obs}, config={config_obs}")
                    print("       Consider setting environment_observation_size to match.")
                
                # Check action shape - should be [n_instruments * 2]
                # Both SimpleTradingEnv and ModernTradingEnv use (2 * n_instruments,) action space
                model_action = metadata.get("action_shape", [4])
                model_instruments = metadata.get("instruments", ["EURUSD", "XAUUSD"])
                expected_action_dim = len(model_instruments) * 2
                
                if model_action != [expected_action_dim]:
                    print(f"[WARN] Action shape: {model_action} (expected [{expected_action_dim}] for {len(model_instruments)} instruments)")
                
                # Check policy architecture
                hyperparams = metadata.get("hyperparameters", {})
                policy_hidden = metadata.get("policy_hidden", 256)
                value_hidden = metadata.get("value_hidden", 256)
                
                print(f"\n[INFO] ✅ TRANSFER LEARNING - Pretrained model metadata:")
                print(f"       Type:         {metadata.get('model_type', 'unknown')}")
                print(f"       Timesteps:    {metadata.get('total_timesteps', 'N/A'):,}")
                print(f"       Created:      {metadata.get('created_at', 'N/A')}")
                print(f"       Obs Size:     {model_obs}")
                print(f"       Action Shape: {model_action}")
                print(f"       Instruments:  {model_instruments}")
                print(f"       Policy Net:   [{policy_hidden}, {policy_hidden // 2}]")
                print(f"       Value Net:    [{value_hidden}, {value_hidden // 2}]")
                if hyperparams:
                    print(f"       LR:           {hyperparams.get('learning_rate', 'N/A')}")
                    print(f"       Gamma:        {hyperparams.get('gamma', 'N/A')}")
                print(f"       Transfer OK:  {metadata.get('transfer_compatible', False)}")
                
                return True
                
            except Exception as e:
                print(f"[WARN] Could not read metadata: {e}")
    
    # No metadata - just warn and proceed
    print("[INFO] No metadata found for pretrained model - proceeding anyway")
    return True


# ───────────────────────────────────────────────────────────────────
# TRAINING
# ───────────────────────────────────────────────────────────────────
def train_modern_ppo(config: TradingConfig, data_source: str, pretrained_model_path: Optional[str] = None):
    mode_str = "LIVE" if getattr(config, "live_mode", False) else "OFFLINE"
    print(f"PPO Training - {mode_str} Mode")

    # Set voting mode to TRAINING - arbiter observes but doesn't block PPO actions
    try:
        from modules.voting.core.constants import set_voting_mode
        set_voting_mode("TRAINING")
        print("[VOTING] Mode set to TRAINING - arbiter is observer only")
    except ImportError:
        pass

    # Logger (tolerant to RotatingLogger arg names)
    training_log = _create_rotating_logger_safe("ModernPPOTraining")

    # Ensure output dirs exist
    os.makedirs(getattr(config, "model_dir", "models"), exist_ok=True)
    os.makedirs(getattr(config, "checkpoint_dir", "checkpoints"), exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)

    # Data
    provider = resolve_data_provider(data_source)
    data = provider.load(config)

    # Envs
    train_env = create_environments(data, config, n_envs=1, seed=getattr(config, "init_seed", 42))
    eval_env = create_environments(data, config, n_envs=1, seed=getattr(config, "init_seed", 42) + 1337)

    # Model (new vs checkpoint)
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"[LOAD] {pretrained_model_path}")
        _validate_pretrained_model(pretrained_model_path, config)
        model = PPO.load(pretrained_model_path, env=train_env)
    else:
        model = create_ppo_model(train_env, config)

    # Determine current vs target steps and remaining training
    current_steps = int(getattr(model, "num_timesteps", 0) or 0)
    target_steps = int(getattr(config, "final_training_steps", 100_000) or 0)

    if target_steps <= 0:
        # Safety: don't run an infinite or negative training
        print(f"[WARN] Non-positive target steps ({target_steps}); skipping training call.")
        remaining_steps = 0
    elif current_steps >= target_steps:
        # Already at or beyond target: no further training needed
        print(
            f"[RESUME] Checkpoint already at {current_steps:,} steps "
            f"(target {target_steps:,}); skipping additional training."
        )
        remaining_steps = 0
    else:
        remaining_steps = target_steps - current_steps
        print(
            f"[RESUME] Current steps: {current_steps:,} | "
            f"Target: {target_steps:,} | Remaining: {remaining_steps:,}"
        )

    # Decide whether to reset SB3's internal step counter
    # - From scratch (0 steps) -> reset_num_timesteps=True
    # - From checkpoint        -> reset_num_timesteps=False (continue)
    reset_timesteps = (current_steps == 0)

    # Callbacks
    callback = ModernEnhancedTrainingCallback(
        getattr(config, "final_training_steps", 100_000),
        config,
        verbose=1,
        enable_ws_broadcast=False,
    )
    callbacks = [
        callback,
        CheckpointCallback(
            save_freq=getattr(config, "checkpoint_freq", 10_000),
            save_path=getattr(config, "checkpoint_dir", "checkpoints"),
            name_prefix=f"modern_ppo_{'live' if getattr(config, 'live_mode', False) else 'offline'}"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(getattr(config, "model_dir", "models"), "best"),
            log_path=os.path.join("logs", "eval"),
            eval_freq=getattr(config, "eval_freq", 10_000),
            deterministic=True,
            render=False,
            n_eval_episodes=getattr(config, "n_eval_episodes", 5),
        ),
    ]

    # Learn
    start = datetime.now()
    try:
        if remaining_steps > 0:
            model.learn(
                total_timesteps=remaining_steps,
                callback=CallbackList(callbacks),
                tb_log_name=f"modern_ppo_{'live' if getattr(config, 'live_mode', False) else 'offline'}",
                reset_num_timesteps=reset_timesteps,
                progress_bar=False,
            )
        else:
            print("[INFO] No remaining steps to train; skipping model.learn().")
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user, saving emergency checkpoint…")
        try:
            model.save(os.path.join(getattr(config, "model_dir", "models"), "modern_ppo_interrupt.zip"))
        except Exception:
            pass
        raise
    finally:
        dur = datetime.now() - start
        print(f"Training finished in {dur}")

    # Save final
    final_path = os.path.join(getattr(config, "model_dir", "models"), "modern_ppo_final.zip")
    model.save(final_path)
    print(f"[OK] Model saved: {final_path}")

    # Cleanup
    try:
        train_env.close()
        eval_env.close()
    except Exception:
        pass


# ───────────────────────────────────────────────────────────────────
# EVALUATION
# ───────────────────────────────────────────────────────────────────
def evaluate_model(
    model_path: str,
    config: TradingConfig,
    data_source: str = "auto",
    n_episodes: int = 20,
    render: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to trained model (.zip)
        config: Trading configuration
        data_source: Where to load data from
        n_episodes: Number of evaluation episodes
        render: Whether to print per-episode details
        verbose: Print summary statistics
        
    Returns:
        Dictionary with evaluation metrics
    """
    import json
    from collections import defaultdict
    
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = PPO.load(model_path)
    print(f"[OK] Model loaded successfully")
    
    # Load data
    provider = resolve_data_provider(data_source)
    data = provider.load(config)
    
    # Create evaluation environment
    global _EXPLORATION_MODE
    if _EXPLORATION_MODE:
        exp_config = ExplorationConfig(
            initial_balance=getattr(config, "initial_balance", 100_000.0),
            max_steps_per_episode=getattr(config, "episode_length", 800),
            take_profit_eur=500.0,
            stop_loss_eur=200.0,
            max_drawdown_pct=0.20,
            max_position_age=100,
            direction_long_threshold=0.3,
            direction_short_threshold=-0.3,
        )
        env = ExplorationTradingEnv(data_dict=data, config=exp_config)
    else:
        env = ModernTradingEnv(data_dict=data, config=config)
    
    # Track metrics
    episode_rewards = []
    episode_lengths = []
    episode_balances = []
    episode_trades = []
    episode_wins = []
    episode_drawdowns = []
    actions_taken = defaultdict(int)  # Track action distribution
    
    print(f"\nRunning {n_episodes} evaluation episodes...")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0
        start_balance = env.balance if hasattr(env, 'balance') else 100_000.0
        max_balance = start_balance
        min_balance = start_balance
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            ep_reward += reward
            ep_steps += 1
            
            # Track balance for drawdown
            current_balance = env.balance if hasattr(env, 'balance') else start_balance
            max_balance = max(max_balance, current_balance)
            min_balance = min(min_balance, current_balance)
            
            # Track action distribution
            if hasattr(action, '__iter__'):
                dir_score = action[0] if len(action) > 0 else 0
                if dir_score > 0.3:
                    actions_taken['long'] += 1
                elif dir_score < -0.3:
                    actions_taken['short'] += 1
                else:
                    actions_taken['flat'] += 1
        
        # Episode metrics
        final_balance = env.balance if hasattr(env, 'balance') else start_balance
        drawdown = (max_balance - min_balance) / max_balance if max_balance > 0 else 0
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        episode_balances.append(final_balance)
        episode_drawdowns.append(drawdown)
        
        # Trade stats from env (use getattr to avoid errors)
        episode_trades.append(getattr(env, 'total_trades', getattr(env, 'trade_count', 0)))
        episode_wins.append(getattr(env, 'winning_trades', getattr(env, 'win_count', 0)))
        
        if render:
            pnl_pct = ((final_balance - start_balance) / start_balance) * 100
            print(f"  Episode {ep+1:3d}: Reward={ep_reward:+7.2f}, "
                  f"Balance=€{final_balance:,.0f} ({pnl_pct:+.1f}%), "
                  f"Steps={ep_steps}, Drawdown={drawdown:.1%}")
    
    env.close()
    
    # Compute statistics
    results = {
        "model_path": model_path,
        "n_episodes": n_episodes,
        "reward_mean": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "reward_min": float(np.min(episode_rewards)),
        "reward_max": float(np.max(episode_rewards)),
        "balance_mean": float(np.mean(episode_balances)),
        "balance_std": float(np.std(episode_balances)),
        "balance_min": float(np.min(episode_balances)),
        "balance_max": float(np.max(episode_balances)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "drawdown_mean": float(np.mean(episode_drawdowns)),
        "drawdown_max": float(np.max(episode_drawdowns)),
        "action_distribution": dict(actions_taken),
    }
    
    # Add trade stats if available
    if episode_trades:
        results["trades_per_episode"] = float(np.mean(episode_trades))
    if episode_wins and episode_trades:
        total_trades = sum(episode_trades)
        total_wins = sum(episode_wins)
        results["win_rate"] = total_wins / total_trades if total_trades > 0 else 0.0
    
    # Profit metrics
    initial_balance = getattr(config, "initial_balance", 100_000.0)
    results["profitable_episodes"] = sum(1 for b in episode_balances if b > initial_balance)
    results["profitable_pct"] = results["profitable_episodes"] / n_episodes
    results["avg_return_pct"] = ((results["balance_mean"] - initial_balance) / initial_balance) * 100
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Episodes:          {n_episodes}")
        print(f"")
        print(f"REWARDS:")
        print(f"  Mean:            {results['reward_mean']:+.3f}")
        print(f"  Std:             {results['reward_std']:.3f}")
        print(f"  Range:           [{results['reward_min']:+.2f}, {results['reward_max']:+.2f}]")
        print(f"")
        print(f"BALANCE:")
        print(f"  Mean:            €{results['balance_mean']:,.0f}")
        print(f"  Range:           [€{results['balance_min']:,.0f}, €{results['balance_max']:,.0f}]")
        print(f"  Avg Return:      {results['avg_return_pct']:+.2f}%")
        print(f"  Profitable:      {results['profitable_episodes']}/{n_episodes} ({results['profitable_pct']:.0%})")
        print(f"")
        print(f"RISK:")
        print(f"  Avg Drawdown:    {results['drawdown_mean']:.1%}")
        print(f"  Max Drawdown:    {results['drawdown_max']:.1%}")
        print(f"")
        print(f"ACTIONS:")
        total_actions = sum(actions_taken.values())
        if total_actions > 0:
            print(f"  Long:            {actions_taken['long']:,} ({actions_taken['long']/total_actions:.1%})")
            print(f"  Short:           {actions_taken['short']:,} ({actions_taken['short']/total_actions:.1%})")
            print(f"  Flat:            {actions_taken['flat']:,} ({actions_taken['flat']/total_actions:.1%})")
        if "win_rate" in results:
            print(f"")
            print(f"TRADING:")
            print(f"  Trades/Episode:  {results['trades_per_episode']:.0f}")
            print(f"  Win Rate:        {results['win_rate']:.1%}")
        print(f"{'='*60}")
        
        # Overall assessment
        print(f"\n📊 ASSESSMENT:")
        if results['reward_mean'] > 0.5 and results['profitable_pct'] > 0.6:
            print(f"   ✅ EXCELLENT - Model is profitable and consistent")
        elif results['reward_mean'] > 0 and results['profitable_pct'] > 0.5:
            print(f"   ✅ GOOD - Model is net profitable")
        elif results['reward_mean'] > -0.5:
            print(f"   ⚠️  MARGINAL - Model near breakeven, needs more training")
        else:
            print(f"   ❌ POOR - Model is losing money, investigate issues")
        
        if results['drawdown_max'] > 0.15:
            print(f"   ⚠️  HIGH RISK - Max drawdown {results['drawdown_max']:.1%} exceeds 15%")
    
    # Save results
    os.makedirs("logs/eval", exist_ok=True)
    eval_file = f"logs/eval/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(eval_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved: {eval_file}")
    
    return results


# ───────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────
def main():
    # Parse args first to check if exploration mode
    p = argparse.ArgumentParser(description="Modern PPO Training")
    p.add_argument("--mode", choices=["offline", "online", "test", "eval"], default="offline",
                   help="Mode: offline=train, online=live train, test=quick test, eval=evaluate model")
    p.add_argument("--preset", choices=["conservative", "aggressive", "research", "production", "exploration", "training_fast"],
                   help="Config preset: 'training_fast' for high-speed PPO training, 'exploration' disables modules")
    p.add_argument("--timesteps", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--n_epochs", type=int)
    p.add_argument("--gamma", type=float)
    p.add_argument("--n_steps", type=int)
    p.add_argument("--clip_range", type=float)
    p.add_argument("--ent_coef", type=float)
    p.add_argument("--vf_coef", type=float)
    p.add_argument("--max_grad_norm", type=float)
    p.add_argument("--target_kl", type=float)
    p.add_argument("--checkpoint_freq", type=int)
    p.add_argument("--eval_freq", type=int)
    p.add_argument("--num_envs", type=int)
    p.add_argument("--balance", type=float)
    p.add_argument("--data_dir", type=str, default=None, help="Override data directory from config")
    p.add_argument("--pretrained", type=str)
    p.add_argument("--auto-pretrained", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard")
    p.add_argument("--fast", action="store_true", 
                   help="Use ExplorationEnv (no modules) for fastest training")
    p.add_argument("--prebaked", action="store_true",
                   help="Use prebaked module signals from data/prebaked/ (run scripts/prebake_signals.py first)")
    p.add_argument("--model", type=str, help="Model path for evaluation (--mode eval)")
    p.add_argument("--eval-episodes", type=int, default=None, help="Number of evaluation episodes (default from config)")
    p.add_argument("--render", action="store_true", help="Print per-episode details during eval")
    p.add_argument(
        "--data-source",
        choices=["auto", "files", "orchestrator"],
        default=None,
        help="Where to load training data from (default from config)",
    )
    args = p.parse_args()

    app_mode = "live" if args.mode == "online" else "training"
    preset_name = args.preset if args.preset not in (None, "exploration") else None
    app_config = load_app_config(mode=app_mode, preset=preset_name)
    if args.data_dir:
        app_config.paths.data = args.data_dir
    if args.debug:
        app_config.logging.debug = True
    app_config.rl.n_eval_episodes = args.eval_episodes or app_config.rl.n_eval_episodes

    setup_logging(app_config.logging)
    logger = get_logger("train.train_ppo_hybrid")
    logger.info("Loaded app config (mode=%s%s)", app_mode, f", preset={preset_name}" if preset_name else "")

    # Determine if exploration mode (no modules) - for fast parallel training
    # --fast flag, --prebaked, or --preset exploration all enable this
    exploration_mode = (args.preset == "exploration") or getattr(args, "fast", False) or getattr(args, "prebaked", False)
    use_prebaked = getattr(args, "prebaked", False)
    
    # Set global flags for environment creation
    global _EXPLORATION_MODE, _USE_PREBAKED
    _EXPLORATION_MODE = exploration_mode
    _USE_PREBAKED = use_prebaked
    
    if use_prebaked:
        logger.info("Using PREBAKED module signals from data/prebaked/")
        print("[MODE] PREBAKED: Using real module signals (run scripts/prebake_signals.py to generate)")
    elif exploration_mode:
        logger.info("Using FAST exploration mode (no modules, inline signal approximation)")
        print("[MODE] FAST: Using inline signal approximation (no prebaked signals)")
    else:
        logger.info("Using FULL module mode (orchestrator + all modules)")
        print("[MODE] FULL: Using orchestrator with all modules")
    
    # Initialize the ModuleOrchestrator ONLY if not in exploration mode
    orchestrator = None
    bus_inspector = None
    if not exploration_mode:
        orchestrator = ModuleOrchestrator.get_instance()
        orchestrator.initialize()

        # Enable staleness checking for training mode (default config has it off for dashboard)
        try:
            bus = InfoBusManager.get_instance()
            if hasattr(bus, 'set_staleness_check_enabled'):
                bus.set_staleness_check_enabled(True)
            if hasattr(bus, 'set_live_mode'):
                bus.set_live_mode(False)  # Training mode = relaxed threshold (2hrs)
        except Exception:
            pass
    
    # ═══════════════════════════════════════════════════════════════
    # START WEB DASHBOARD (shares same InfoBus instance)
    # ═══════════════════════════════════════════════════════════════
    dashboard_thread = None
    if DASHBOARD_AVAILABLE and not args.no_dashboard:
        try:
            dashboard_thread = start_dashboard_server(port=8765)
            print("\n[DASHBOARD] Web dashboard running at http://localhost:8765")
            # Auto-open in browser
            if webbrowser:
                webbrowser.open("http://localhost:8765")
            print("")
        except Exception as e:
            print(f"[WARN] Could not start dashboard: {e}")
    
    if not exploration_mode:
        # Bus observability (optional; keep types as Any to avoid arg-type mismatches)
        try:
            if DependencyInspector_Cls is not None and ENHANCED_LOGGING:
                bus = InfoBusManager.get_instance()
                Path("logs/monitoring/dependency").mkdir(parents=True, exist_ok=True)
                session_logger = RotatingLogger_Cls(
                    name="DependencyInspector",
                    log_path=f"logs/monitoring/dependency/Training_{datetime.now():%Y%m%d_%H%M%S}.log",
                    max_lines=20000,
                    operator_mode=True,
                    plain_english=True,
                )
                try:
                    bus_inspector = DependencyInspector_Cls(bus=bus, log=session_logger)  # type: ignore[arg-type]
                    if hasattr(bus_inspector, "attach_live_taps"):
                        bus_inspector.attach_live_taps(show_values=True, max_preview=160)
                    if hasattr(bus_inspector, "emit_report"):
                        bus_inspector.emit_report("Training start")
                except Exception:
                    pass
        except Exception:
            pass
    else:
        print("\n" + "="*70)
        print("  EXPLORATION MODE - Modules disabled for free exploration")
        print("  Use 'python train/train_simple_mode.py' for a cleaner experience")
        print("="*70 + "\n")

    # Config
    config = app_config.to_trading_config()
    if args.mode == "test":
        config.test_mode = True

    # Overrides
    overrides: Dict[str, Any] = {}
    if args.timesteps: overrides["final_training_steps"] = args.timesteps
    if args.lr: overrides["learning_rate"] = args.lr
    if args.batch_size: overrides["batch_size"] = args.batch_size
    if args.n_epochs: overrides["n_epochs"] = args.n_epochs
    if args.gamma: overrides["gamma"] = args.gamma
    if args.n_steps: overrides["n_steps"] = args.n_steps
    if args.clip_range: overrides["clip_range"] = args.clip_range
    if args.ent_coef: overrides["ent_coef"] = args.ent_coef
    if args.vf_coef: overrides["vf_coef"] = args.vf_coef
    if args.max_grad_norm: overrides["max_grad_norm"] = args.max_grad_norm
    if args.target_kl: overrides["target_kl"] = args.target_kl
    if args.checkpoint_freq: overrides["checkpoint_freq"] = args.checkpoint_freq
    if args.eval_freq: overrides["eval_freq"] = args.eval_freq
    if args.num_envs: overrides["num_envs"] = args.num_envs
    if args.balance: overrides["initial_balance"] = args.balance
    if args.data_dir: overrides["data_dir"] = args.data_dir
    if args.debug: overrides["debug"] = True
    if overrides:
        config.apply_overrides(**overrides)

    # Sensible default when not specified
    if not args.timesteps and args.mode == "test":
        config.final_training_steps = 10_000
    elif not args.timesteps:
        config.final_training_steps = max(50_000, config.final_training_steps)

    data_source = args.data_source or app_config.environment.data_source

    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(config.model_dir) / "best").mkdir(parents=True, exist_ok=True)

    # Pretrained
    pretrained_path = None
    if args.pretrained:
        pretrained_path = args.pretrained
    elif args.auto_pretrained:
        # Search order:
        # 1) Final model in standard location
        # 2) Simple mode pretrained (exploration phase)
        # 3) Latest checkpoint
        
        auto_path = os.path.join(getattr(config, "model_dir", "models"), "modern_ppo_final.zip")
        simple_path = "models/simple/simple_ppo_final.zip"
        
        if os.path.exists(auto_path):
            pretrained_path = auto_path
            print(f"[AUTO] Using final model as pretrained: {pretrained_path}")
        elif os.path.exists(simple_path):
            pretrained_path = simple_path
            print(f"[AUTO] Using SIMPLE MODE pretrained model: {pretrained_path}")
            print("       (This model was trained without modules - good foundation!)")
        else:
            # Fall back to latest checkpoint in checkpoint_dir
            ckpt_dir = getattr(config, "checkpoint_dir", "checkpoints")
            if os.path.isdir(ckpt_dir):
                candidates = [
                    os.path.join(ckpt_dir, f)
                    for f in os.listdir(ckpt_dir)
                    if f.endswith(".zip")
                ]

                def _extract_steps(path: str) -> int:
                    # Expected name format: prefix_xxx_<steps>_steps.zip
                    name = os.path.basename(path)
                    core = name[:-4]  # remove ".zip"
                    parts = core.split("_")
                    try:
                        for i, p in enumerate(parts):
                            if p.isdigit() and i + 1 < len(parts) and parts[i + 1] == "steps":
                                return int(p)
                    except Exception:
                        pass
                    return 0

                if candidates:
                    candidates.sort(
                        key=lambda p: (_extract_steps(p), os.path.getmtime(p)),
                        reverse=True,
                    )
                    pretrained_path = candidates[0]
                    print(f"[AUTO] Using latest checkpoint as pretrained: {pretrained_path}")
            # If still nothing found, pretrained_path remains None

    # Save config (best-effort)
    try:
        os.makedirs("logs", exist_ok=True)
        config.save_config("logs/modern_training_config.json")
    except Exception:
        pass

    # Display summary
    logger.info("Mode: %s", args.mode.upper())
    if args.mode != "eval":
        logger.info("Training Steps: %s", f"{config.final_training_steps:,}")
        logger.info("Learning Rate: %s", config.learning_rate)
    logger.info("Initial Balance: $%s", f"{config.initial_balance:,.0f}")
    logger.info("Data Source: %s", data_source)

    # Bus visibility (non-fatal if fallback)
    # Debug prints removed - bus info available in logs if needed

    # ═══════════════════════════════════════════════════════════════
    # EVALUATION MODE
    # ═══════════════════════════════════════════════════════════════
    if args.mode == "eval":
        # Find model to evaluate
        model_path = args.model
        if not model_path:
            # Auto-find best model
            candidates = []
            for search_dir in ["checkpoints", "models", "models/best"]:
                if os.path.exists(search_dir):
                    for f in os.listdir(search_dir):
                        if f.endswith(".zip"):
                            candidates.append(os.path.join(search_dir, f))
            
            if not candidates:
                print("ERROR: No model found. Specify with --model PATH")
                sys.exit(1)
            
            # Prefer 'best' or 'final' models
            def model_priority(p):
                name = os.path.basename(p).lower()
                if "best" in name: return (0, os.path.getmtime(p))
                if "final" in name: return (1, os.path.getmtime(p))
                return (2, os.path.getmtime(p))
            
            candidates.sort(key=model_priority)
            model_path = candidates[0]
            print(f"[AUTO] Using model: {model_path}")
        
        try:
            results = evaluate_model(
                model_path=model_path,
                config=config,
                data_source=data_source,
                n_episodes=args.eval_episodes,
                render=args.render,
                verbose=True,
            )
            print("\nEVALUATION COMPLETED!")
            sys.exit(0)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    # TRAINING MODE
    # ═══════════════════════════════════════════════════════════════
    try:
        train_modern_ppo(config, data_source=data_source, pretrained_model_path=pretrained_path)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        if bus_inspector:
            try:
                if hasattr(bus_inspector, "emit_report"):
                    bus_inspector.emit_report("Training end")
                if hasattr(bus_inspector, "detach_live_taps"):
                    bus_inspector.detach_live_taps()
            except Exception:
                pass
        try:
            InfoBusManager.get_instance().export_session("logs/infobus_session.json")
            print("[OK] Exported SmartInfoBus session -> logs/infobus_session.json")
        except Exception:
            pass
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting Modern PPO Training System")
    if not torch.cuda.is_available():
        print("Note: Running on CPU")
    main()
