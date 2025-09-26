#!/usr/bin/env python3
"""
Complete Modern SmartInfoBus v4.0 Training Script
Production-ready with data provider abstraction and richer telemetry
Pylance-safe: no type identity clashes with external modules
"""

from __future__ import annotations

import os
import sys
import platform
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Protocol
from pathlib import Path

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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from envs.modern_env import ModernTradingEnv
from envs.config import TradingConfig, ConfigPresets, ConfigFactory
from modules.core.module_system import ModuleOrchestrator

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
    from modules.monitoring.dependency_inspector import DependencyInspector as _DependencyInspector  # type: ignore
    DependencyInspector_Cls = _DependencyInspector
except Exception:
    DependencyInspector_Cls = None

# Enhanced callback
from train.enhanced_training_callback import ModernEnhancedTrainingCallback

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
                base = file.replace(".csv", "").replace("_features", "")
                parts = base.split("_")
                if len(parts) >= 2:
                    instrument, timeframe = parts[0], parts[1]
                else:
                    instrument, timeframe = base.replace("_", "/"), "H1"
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
        timeframes = getattr(config, "timeframes", None) or ["H1", "H4", "D1"]
        missing = []

        for inst in instruments:
            for tf in timeframes:
                key = f"market_data_{inst}_{tf}"
                blob = None
                try:
                    blob = bus.get(key, module="TrainingScript")
                except Exception:
                    blob = None
                if not blob:
                    missing.append((inst, tf))
                    continue
                try:
                    df = pd.DataFrame({
                        "open":  np.asarray(blob["open"], dtype=np.float32),
                        "high":  np.asarray(blob["high"], dtype=np.float32),
                        "low":   np.asarray(blob["low"],  dtype=np.float32),
                        "close": np.asarray(blob["close"],dtype=np.float32),
                        "volume": np.asarray(blob.get("volume", np.ones(len(blob["close"]))), dtype=np.float32),
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
    instruments = getattr(config, "instruments", None) or ["EUR_USD"]
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
def test_environment_creation(data: Dict, config: TradingConfig) -> bool:
    try:
        print("[TOOL] Testing environment creation...")
        env = ModernTradingEnv(data, config)
        obs, _ = env.reset(seed=getattr(config, "init_seed", 42))
        if not isinstance(obs, np.ndarray) or not np.all(np.isfinite(obs)):
            raise ValueError("Invalid observation")
        env.step(env.action_space.sample())
        env.close()
        print(f"[OK] Environment test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Env test failed: {e}")
        return False

def create_environments(data: Dict, config: TradingConfig, n_envs: int = 1, seed: int = 42):
    if not test_environment_creation(data, config):
        raise RuntimeError("Environment creation test failed")

    if getattr(config, "live_mode", False) or platform.system() == "Windows":
        n_envs = 1
        print("[TOOL] Using single environment for stability")

    def make(rank: int):
        def _init():
            env = ModernTradingEnv(data, config)
            Path("logs/training").mkdir(parents=True, exist_ok=True)
            return Monitor(env, filename=f"logs/training/monitor_{rank}.csv", info_keywords=())
        set_random_seed(seed + rank)
        return _init

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
        policy_kwargs=policy_kwargs,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        seed=getattr(config, "init_seed", 42),
    )
    print(f"[BOT] PPO model on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    return model

# ───────────────────────────────────────────────────────────────────
# TRAINING
# ───────────────────────────────────────────────────────────────────
def train_modern_ppo(config: TradingConfig, data_source: str, pretrained_model_path: Optional[str] = None):
    mode_str = "LIVE" if getattr(config, "live_mode", False) else "OFFLINE"
    print(f"PPO Training - {mode_str} Mode")

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

    # Model
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"[LOAD] {pretrained_model_path}")
        model = PPO.load(pretrained_model_path, env=train_env)
    else:
        model = create_ppo_model(train_env, config)

    # Callbacks
    callback = ModernEnhancedTrainingCallback(getattr(config, "final_training_steps", 100_000), config, verbose=1)
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
        model.learn(
            total_timesteps=getattr(config, "final_training_steps", 100_000),
            callback=CallbackList(callbacks),
            tb_log_name=f"modern_ppo_{'live' if getattr(config, 'live_mode', False) else 'offline'}",
            reset_num_timesteps=True,
            progress_bar=False,
        )
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
# MAIN
# ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs("models/best", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Initialize the ModuleOrchestrator
    orchestrator = ModuleOrchestrator.get_instance()
    orchestrator.initialize()

    # Bus observability (optional; keep types as Any to avoid arg-type mismatches)
    bus_inspector = None
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
                # Type of "log" is external; we deliberately avoid annotations to please Pylance
                bus_inspector = DependencyInspector_Cls(bus=bus, log=session_logger)  # type: ignore[arg-type]
                if hasattr(bus_inspector, "attach_live_taps"):
                    bus_inspector.attach_live_taps(show_values=True, max_preview=160)
                if hasattr(bus_inspector, "emit_report"):
                    bus_inspector.emit_report("Training start")
            except Exception:
                pass
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Modern PPO Training")
    p.add_argument("--mode", choices=["offline", "online", "test"], default="offline")
    p.add_argument("--preset", choices=["conservative", "aggressive", "research", "production"])
    p.add_argument("--timesteps", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--balance", type=float)
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--pretrained", type=str)
    p.add_argument("--auto-pretrained", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--data-source",
        choices=["auto", "files", "orchestrator"],
        default="auto",
        help="Where to load training data from",
    )
    args = p.parse_args()

    # Config
    if args.mode == "online":
        config = ConfigPresets.conservative_live()
        config.live_mode = True
    elif args.preset == "conservative":
        config = ConfigPresets.conservative_live()
    elif args.preset == "aggressive":
        config = ConfigFactory.create_config("backtest", "aggressive")
    elif args.preset == "research":
        config = ConfigPresets.research_mode()
    elif args.preset == "production":
        config = ConfigPresets.production_backtest()
    else:
        config = TradingConfig(test_mode=(args.mode == "test"), live_mode=False)

    # Overrides
    if args.timesteps: config.final_training_steps = args.timesteps
    if args.lr: config.learning_rate = args.lr
    if args.batch_size: config.batch_size = args.batch_size
    if args.balance: config.initial_balance = args.balance
    if args.data_dir: config.data_dir = args.data_dir
    if args.debug: config.debug = True

    # Sensible default when not specified
    if not args.timesteps and args.mode == "test":
        config.final_training_steps = 10_000
    elif not args.timesteps:
        config.final_training_steps = max(50_000, config.final_training_steps)

    # Pretrained
    pretrained_path = None
    if args.pretrained:
        pretrained_path = args.pretrained
    elif args.auto_pretrained:
        auto_path = os.path.join(getattr(config, "model_dir", "models"), "modern_ppo_final.zip")
        if os.path.exists(auto_path):
            pretrained_path = auto_path

    # Save config (best-effort)
    try:
        os.makedirs("logs", exist_ok=True)
        config.save_config("logs/modern_training_config.json")
    except Exception:
        pass

    # Display summary
    print(f"Training Mode: {args.mode.upper()}")
    print(f"Training Steps: {config.final_training_steps:,}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Initial Balance: ${config.initial_balance:,.0f}")
    print(f"Data Source: {args.data_source}")

    # Bus visibility (non-fatal if fallback)
    try:
        bus = InfoBusManager.get_instance()
        print("[BUS] instance:", bus)
        store_keys = sorted(getattr(bus, "_data_store", {}).keys())
        print("[BUS] keys:", store_keys)
        for k in ("market_overview", "market_thesis", "time_risk_health", "regime_matrix_health"):
            try:
                print(f"[BUS] {k} =", bus.get(k, module="TrainingScript"))
            except Exception as e:
                print(f"[BUS] {k} <error: {e}>")
    except Exception:
        pass

    try:
        train_modern_ppo(config, data_source=args.data_source, pretrained_model_path=pretrained_path)
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
