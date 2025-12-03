#!/usr/bin/env python3
"""
Advanced Optuna Hyperparameter Tuner for PPO Trading Agent
===========================================================

A robust, production-grade hyperparameter optimization system with:
- Intelligent early stopping via MedianPruner
- Comprehensive logging and visualization
- Resumable studies with SQLite storage
- Best parameters export for manual integration

Author: AI Trading System
Version: 2.0.0

Usage Examples:
    # Quick test run (10 trials, 20K steps each)
    python train/optuna_tuner.py --n_trials 10 --timesteps 20000

    # Full optimization with both reward and PPO tuning
    python train/optuna_tuner.py --n_trials 50 --timesteps 100000 --tune_all

    # Tune only reward shaping parameters
    python train/optuna_tuner.py --n_trials 30 --tune_reward

    # Resume a previous study
    python train/optuna_tuner.py --n_trials 50 --study_name my_study --resume

    # Show best parameters from completed study
    python train/optuna_tuner.py --show_best

    # Export best parameters to YAML
    python train/optuna_tuner.py --export_best config/tuned_params.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Third-party imports - optuna
# pyright: reportMissingImports=false
import optuna  # type: ignore[import-untyped]
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner  # type: ignore[import-untyped]
from optuna.samplers import TPESampler  # type: ignore[import-untyped]
from optuna.trial import TrialState  # type: ignore[import-untyped]

# pyright: reportMissingImports=false
import yaml  # type: ignore[import-untyped]
YAML_AVAILABLE = True

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[ERROR] stable-baselines3 not installed.")
    sys.exit(1)

from envs.modern_env import ModernTradingEnv
from envs.config import TradingConfig
from modules.reward.shared.reward_config import RewardConfig

# Suppress noisy logs during optimization
logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


class TuningMode(Enum):
    """What to tune during optimization."""
    REWARD_ONLY = "reward"
    PPO_ONLY = "ppo"
    BOTH = "both"


class PrunerType(Enum):
    """Available pruner types."""
    MEDIAN = "median"
    HALVING = "halving"
    HYPERBAND = "hyperband"


def _load_optuna_initial_balance() -> float:
    """Load initial balance from risk_policy.yaml."""
    import yaml
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "risk_policy.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}
            return float(policy.get("prop_firm", {}).get("account_size", 100000.0))
    except Exception:
        pass
    return 100000.0


@dataclass
class OptunaTunerConfig:
    """Configuration for the Optuna hyperparameter tuner."""

    # Trial settings
    n_trials: int = 50
    timesteps_per_trial: int = 50000
    eval_freq: int = 10000
    n_eval_episodes: int = 5

    # Tuning scope
    tune_reward: bool = True
    tune_ppo: bool = True

    # Environment settings
    data_dir: str = "data/processed"
    initial_balance: float = 100000.0  # Loaded from risk_policy.yaml in __post_init__

    # Study settings
    study_name: str = "ppo_trading_optimization"
    storage_path: Optional[str] = None
    n_jobs: int = 1
    seed: int = 42

    # Pruner settings
    pruner_type: PrunerType = PrunerType.MEDIAN
    pruner_startup_trials: int = 5
    pruner_warmup_steps: int = 3

    # Output settings
    output_dir: str = "optuna_results"
    verbose: int = 1
    save_visualizations: bool = True

    # Advanced
    timeout_seconds: Optional[int] = None
    gc_after_trial: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Load initial_balance from risk_policy.yaml
        self.initial_balance = _load_optuna_initial_balance()
        
        self.n_trials = max(1, self.n_trials)
        self.timesteps_per_trial = max(1000, self.timesteps_per_trial)
        self.eval_freq = max(100, min(self.eval_freq, self.timesteps_per_trial // 2))
        self.n_eval_episodes = max(1, min(self.n_eval_episodes, 20))
        self.n_jobs = max(1, self.n_jobs)

        # Ensure storage path
        if self.storage_path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.storage_path = f"sqlite:///{self.output_dir}/optuna_study.db"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_training_data(data_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load processed CSV data for training.

    Args:
        data_dir: Directory containing processed CSV files

    Returns:
        Nested dict: {instrument: {timeframe: DataFrame}}
    """
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    tf_candidates = {"M1", "M5", "M15", "M30", "H1", "H2", "H4", "H8", "D1", "W1", "MN1"}

    for file_path in csv_files:
        try:
            # Parse filename: INSTRUMENT_TIMEFRAME_features.csv
            base = file_path.stem.replace("_features", "")
            parts = base.split("_")

            if len(parts) >= 2 and parts[-1].upper() in tf_candidates:
                timeframe = parts[-1].upper()
                instrument = "_".join(parts[:-1])
            else:
                instrument = base
                timeframe = "H1"

            # Load and validate
            df = pd.read_csv(file_path)
            required_cols = {"open", "high", "low", "close"}

            if not required_cols.issubset(set(df.columns)):
                continue

            # Ensure volume exists
            if "volume" not in df.columns:
                df["volume"] = 1.0

            # Coerce types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)

            # Store
            data.setdefault(instrument, {})[timeframe] = df

        except Exception as e:
            logging.warning(f"Failed to load {file_path}: {e}")

    if not data:
        raise ValueError(f"No valid data loaded from {data_dir}")

    total_bars = sum(len(df) for tfs in data.values() for df in tfs.values())
    logging.info(f"Loaded {len(data)} instruments, {total_bars:,} total bars")

    return data


def create_dummy_data(n_bars: int = 2000) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Create synthetic data for testing."""
    rng = np.random.default_rng(42)
    base_price = 1.10

    returns = rng.normal(0, 0.01, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": prices.astype(np.float32),
        "high": (prices * 1.001).astype(np.float32),
        "low": (prices * 0.999).astype(np.float32),
        "close": prices.astype(np.float32),
        "volume": rng.integers(100, 1000, n_bars).astype(np.float32),
    })

    return {"EURUSD": {"H1": df}}


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SEARCH SPACES
# ═══════════════════════════════════════════════════════════════════════════════


class HyperparameterSpace:
    """Defines the search space for hyperparameters."""

    # Names of PPO parameters for separation
    PPO_PARAM_NAMES = {
        "learning_rate", "n_steps", "batch_size", "n_epochs",
        "gamma", "gae_lambda", "clip_range", "ent_coef",
        "vf_coef", "max_grad_norm"
    }

    @staticmethod
    def sample_reward_params(trial: optuna.Trial) -> Dict[str, float]:
        """
        Sample reward shaping hyperparameters.

        These control how rewards are shaped to guide learning:
        - Penalties: Discourage risky/poor behavior
        - Bonuses: Encourage profitable/consistent behavior
        """
        return {
            # Penalty weights (lower = less punishment)
            "dd_pen_weight": trial.suggest_float(
                "dd_pen_weight", 0.1, 3.0, log=True,
            ),
            "risk_pen_weight": trial.suggest_float(
                "risk_pen_weight", 0.01, 0.5, log=True,
            ),
            "tail_pen_weight": trial.suggest_float(
                "tail_pen_weight", 0.1, 1.5,
            ),
            "mistake_pen_weight": trial.suggest_float(
                "mistake_pen_weight", 0.05, 0.8,
            ),
            "no_trade_penalty_weight": trial.suggest_float(
                "no_trade_penalty_weight", 0.01, 0.3,
            ),

            # Bonus weights (higher = more positive reinforcement)
            "win_bonus_weight": trial.suggest_float(
                "win_bonus_weight", 0.1, 3.0,
            ),
            "consistency_bonus_weight": trial.suggest_float(
                "consistency_bonus_weight", 0.1, 1.5,
            ),
            "sharpe_bonus_weight": trial.suggest_float(
                "sharpe_bonus_weight", 0.05, 1.0,
            ),
            "trade_frequency_bonus": trial.suggest_float(
                "trade_frequency_bonus", 0.02, 0.5,
            ),

            # Advanced tuning
            "volatility_adjustment": trial.suggest_float(
                "volatility_adjustment", 0.3, 2.0,
            ),
            "regime_bonus_weight": trial.suggest_float(
                "regime_bonus_weight", 0.02, 0.5,
            ),
            "momentum_bonus_weight": trial.suggest_float(
                "momentum_bonus_weight", 0.01, 0.4,
            ),
        }

    @staticmethod
    def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample PPO algorithm hyperparameters.

        These control the reinforcement learning algorithm behavior.
        """
        # Sample n_steps and batch_size with constraint
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        # Ensure batch_size <= n_steps
        batch_size = min(batch_size, n_steps)

        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 5e-3, log=True,
            ),
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": trial.suggest_int(
                "n_epochs", 3, 20,
            ),
            "gamma": trial.suggest_float(
                "gamma", 0.9, 0.9999, log=True,
            ),
            "gae_lambda": trial.suggest_float(
                "gae_lambda", 0.85, 0.99,
            ),
            "clip_range": trial.suggest_float(
                "clip_range", 0.1, 0.4,
            ),
            "ent_coef": trial.suggest_float(
                "ent_coef", 1e-4, 0.1, log=True,
            ),
            "vf_coef": trial.suggest_float(
                "vf_coef", 0.1, 1.0,
            ),
            "max_grad_norm": trial.suggest_float(
                "max_grad_norm", 0.3, 2.0,
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION CALLBACK WITH PRUNING
# ═══════════════════════════════════════════════════════════════════════════════


class OptunaPruningCallback(BaseCallback):
    """
    Callback for Optuna integration with early stopping.

    Reports intermediate evaluation results to Optuna for pruning decisions.
    Tracks multiple metrics for comprehensive evaluation.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        eval_env: DummyVecEnv,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.eval_idx = 0
        self.is_pruned = False

        # Track metrics history
        self.reward_history: List[float] = []
        self.episode_length_history: List[float] = []
        self.best_reward: float = float("-inf")

    def _on_step(self) -> bool:
        """Called at each step; performs evaluation at intervals."""
        if self.n_calls % self.eval_freq != 0:
            return True

        try:
            mean_reward, mean_ep_length = self._evaluate()
        except Exception as e:
            logging.warning(f"Evaluation failed: {e}")
            mean_reward = float("-inf")
            mean_ep_length = 0.0

        self.reward_history.append(mean_reward)
        self.episode_length_history.append(mean_ep_length)
        self.best_reward = max(self.best_reward, mean_reward)
        self.eval_idx += 1

        # Report to Optuna
        self.trial.report(mean_reward, self.eval_idx)

        # Check for pruning
        if self.trial.should_prune():
            self.is_pruned = True
            if self.verbose > 0:
                logging.info(
                    f"[PRUNE] Trial {self.trial.number} pruned at step {self.n_calls:,} "
                    f"(reward: {mean_reward:.2f})"
                )
            return False

        if self.verbose > 0:
            logging.info(
                f"[EVAL] Trial {self.trial.number} step {self.n_calls:,}: "
                f"reward={mean_reward:.2f}, best={self.best_reward:.2f}"
            )

        return True

    def _evaluate(self) -> Tuple[float, float]:
        """Run evaluation episodes and compute metrics."""
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            max_steps = 3000

            while not done and steps < max_steps:
                # Get observation as numpy array
                obs_array = np.asarray(obs)
                action, _ = self.model.predict(obs_array, deterministic=True)  # type: ignore[union-attr]
                obs, reward, done, info = self.eval_env.step(action)

                # Handle vectorized env output
                if isinstance(reward, np.ndarray):
                    episode_reward += float(reward[0])
                else:
                    episode_reward += float(reward)

                if isinstance(done, np.ndarray):
                    done = bool(done[0])

                steps += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

        return float(np.mean(episode_rewards)), float(np.mean(episode_lengths))

    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        return {
            "best_reward": self.best_reward,
            "final_reward": self.reward_history[-1] if self.reward_history else float("-inf"),
            "mean_reward": float(np.mean(self.reward_history)) if self.reward_history else float("-inf"),
            "n_evaluations": self.eval_idx,
            "pruned": self.is_pruned,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def create_objective(
    config: OptunaTunerConfig,
    data: Dict[str, Dict[str, pd.DataFrame]],
) -> Callable[[optuna.Trial], float]:
    """
    Factory function to create an Optuna objective.

    Args:
        config: Tuner configuration
        data: Training data

    Returns:
        Objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Single trial objective: create env, train, evaluate.

        Returns:
            Mean evaluation reward (higher is better)
        """
        trial_start = time.time()

        # Sample hyperparameters based on tuning scope
        reward_params: Dict[str, float] = {}
        ppo_params: Dict[str, Any] = {}

        if config.tune_reward:
            reward_params = HyperparameterSpace.sample_reward_params(trial)

        if config.tune_ppo:
            ppo_params = HyperparameterSpace.sample_ppo_params(trial)

        if config.verbose >= 1:
            _log_trial_start(trial, reward_params, ppo_params)

        train_env: Optional[DummyVecEnv] = None
        eval_env: Optional[DummyVecEnv] = None
        model: Optional[PPO] = None

        try:
            # Create trading config
            trading_config = TradingConfig(
                initial_balance=config.initial_balance,
                data_dir=config.data_dir,
                debug=False,
            )

            # Apply sampled PPO params to config
            if ppo_params:
                for key, value in ppo_params.items():
                    if hasattr(trading_config, key):
                        setattr(trading_config, key, value)

            # Create reward config
            reward_config = RewardConfig(initial_balance=config.initial_balance)
            if reward_params:
                reward_config.apply_genome(reward_params)

            # Create environments
            def make_env() -> Monitor:
                env = ModernTradingEnv(data, trading_config)
                return Monitor(env)

            train_env = DummyVecEnv([make_env])
            eval_env = DummyVecEnv([make_env])

            # Create PPO model
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=ppo_params.get("learning_rate", 3e-4),
                n_steps=ppo_params.get("n_steps", 2048),
                batch_size=ppo_params.get("batch_size", 64),
                n_epochs=ppo_params.get("n_epochs", 10),
                gamma=ppo_params.get("gamma", 0.99),
                gae_lambda=ppo_params.get("gae_lambda", 0.95),
                clip_range=ppo_params.get("clip_range", 0.2),
                ent_coef=ppo_params.get("ent_coef", 0.01),
                vf_coef=ppo_params.get("vf_coef", 0.5),
                max_grad_norm=ppo_params.get("max_grad_norm", 0.5),
                verbose=0,
                device="auto",
                seed=config.seed + trial.number,
            )

            # Create pruning callback
            pruning_callback = OptunaPruningCallback(
                trial=trial,
                eval_env=eval_env,
                n_eval_episodes=config.n_eval_episodes,
                eval_freq=config.eval_freq,
                verbose=config.verbose,
            )

            # Train
            model.learn(
                total_timesteps=config.timesteps_per_trial,
                callback=pruning_callback,
                progress_bar=False,
            )

            # Check if pruned
            if pruning_callback.is_pruned:
                raise optuna.TrialPruned()

            # Get final score
            summary = pruning_callback.get_summary()
            final_reward = summary["best_reward"]

            trial_time = time.time() - trial_start
            if config.verbose >= 1:
                logging.info(
                    f"[COMPLETE] Trial {trial.number}: reward={final_reward:.2f}, "
                    f"time={trial_time:.1f}s"
                )

            return final_reward

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logging.error(f"[FAIL] Trial {trial.number}: {e}")
            return float("-inf")
        finally:
            # Cleanup
            if train_env is not None:
                try:
                    train_env.close()
                except Exception:
                    pass
            if eval_env is not None:
                try:
                    eval_env.close()
                except Exception:
                    pass
            del model, train_env, eval_env
            if config.gc_after_trial:
                gc.collect()

    return objective


def _log_trial_start(
    trial: optuna.Trial,
    reward_params: Dict[str, float],
    ppo_params: Dict[str, Any],
) -> None:
    """Log trial start information."""
    parts = [f"\n{'─'*60}", f"[TRIAL {trial.number}] Starting"]

    if reward_params:
        parts.append(f"  Reward: dd_pen={reward_params.get('dd_pen_weight', 0):.3f}, "
                     f"win_bonus={reward_params.get('win_bonus_weight', 0):.3f}")

    if ppo_params:
        parts.append(f"  PPO: lr={ppo_params.get('learning_rate', 0):.2e}, "
                     f"ent_coef={ppo_params.get('ent_coef', 0):.4f}, "
                     f"n_steps={ppo_params.get('n_steps', 0)}")

    logging.info("\n".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════
# STUDY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════


def create_pruner(config: OptunaTunerConfig) -> optuna.pruners.BasePruner:
    """Create the appropriate pruner based on configuration."""
    if config.pruner_type == PrunerType.MEDIAN:
        return MedianPruner(
            n_startup_trials=config.pruner_startup_trials,
            n_warmup_steps=config.pruner_warmup_steps,
            interval_steps=1,
            n_min_trials=config.pruner_startup_trials,
        )
    elif config.pruner_type == PrunerType.HALVING:
        return SuccessiveHalvingPruner(
            min_resource=config.pruner_warmup_steps,
            reduction_factor=3,
            min_early_stopping_rate=0,
        )
    else:  # HYPERBAND
        return HyperbandPruner(
            min_resource=config.pruner_warmup_steps,
            reduction_factor=3,
        )


def run_optimization(config: OptunaTunerConfig) -> optuna.Study:
    """
    Run the full Optuna hyperparameter optimization.

    Args:
        config: Tuner configuration

    Returns:
        Completed Optuna study
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if config.verbose >= 1 else logging.WARNING,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(config.output_dir, exist_ok=True)

    # Print banner
    print("\n" + "═" * 70)
    print("  OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("═" * 70)
    print(f"  Trials:          {config.n_trials}")
    print(f"  Steps/trial:     {config.timesteps_per_trial:,}")
    print(f"  Eval frequency:  {config.eval_freq:,}")
    print(f"  Tune reward:     {config.tune_reward}")
    print(f"  Tune PPO:        {config.tune_ppo}")
    print(f"  Pruner:          {config.pruner_type.value}")
    print(f"  Storage:         {config.storage_path}")
    print("═" * 70 + "\n")

    # Load data
    try:
        data = load_training_data(config.data_dir)
    except FileNotFoundError as e:
        logging.warning(f"Data not found: {e}. Using synthetic data.")
        data = create_dummy_data()

    # Create study components
    sampler = TPESampler(seed=config.seed, n_startup_trials=10)
    pruner = create_pruner(config)

    # Create or load study
    study = optuna.create_study(
        study_name=config.study_name,
        storage=config.storage_path,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    # Check for existing trials
    existing_trials = len(study.trials)
    if existing_trials > 0:
        logging.info(f"Resuming study with {existing_trials} existing trials")

    # Create objective
    objective = create_objective(config, data)

    # Run optimization
    start_time = time.time()
    study.optimize(
        objective,
        n_trials=config.n_trials,
        n_jobs=config.n_jobs,
        timeout=config.timeout_seconds,
        show_progress_bar=True,
        gc_after_trial=config.gc_after_trial,
    )
    total_time = time.time() - start_time

    # Generate results
    _save_results(study, config, total_time)

    return study


def _save_results(
    study: optuna.Study,
    config: OptunaTunerConfig,
    total_time: float,
) -> None:
    """Save optimization results and generate reports."""

    # Compute statistics
    all_trials = study.trials
    completed = [t for t in all_trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in all_trials if t.state == TrialState.PRUNED]
    failed = [t for t in all_trials if t.state == TrialState.FAIL]

    print("\n" + "═" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("═" * 70)
    print(f"  Total time:      {total_time/60:.1f} minutes")
    print(f"  Total trials:    {len(all_trials)}")
    print(f"    Completed:     {len(completed)}")
    print(f"    Pruned:        {len(pruned)} (saved time!)")
    print(f"    Failed:        {len(failed)}")
    print("─" * 70)

    if study.best_trial is None:
        print("  [!] No successful trials completed.")
        return

    # Best trial info
    print(f"\n  🏆 BEST TRIAL: #{study.best_trial.number}")
    print(f"     Reward:    {study.best_value:.2f}")
    print("\n  Best Parameters:")
    print("  " + "─" * 40)

    for key, value in sorted(study.best_params.items()):
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

    print("═" * 70)

    # Separate reward and PPO params properly
    ppo_param_names = HyperparameterSpace.PPO_PARAM_NAMES

    ppo_params = {k: v for k, v in study.best_params.items() if k in ppo_param_names}
    reward_params = {k: v for k, v in study.best_params.items() if k not in ppo_param_names}

    # Save best config as JSON
    best_config = {
        "optimization_info": {
            "study_name": config.study_name,
            "best_trial": study.best_trial.number,
            "best_reward": study.best_value,
            "total_trials": len(all_trials),
            "completed_trials": len(completed),
            "pruned_trials": len(pruned),
            "optimization_time_minutes": total_time / 60,
            "completed_at": datetime.now().isoformat(),
        },
        "best_params": study.best_params,
        "ppo_params": ppo_params,
        "reward_params": reward_params,
    }

    best_config_path = os.path.join(config.output_dir, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n  📁 Best config saved to: {best_config_path}")

    # Save full trial history
    trials_data = []
    for t in all_trials:
        duration = None
        if t.datetime_complete and t.datetime_start:
            duration = (t.datetime_complete - t.datetime_start).total_seconds()
        trials_data.append({
            "number": t.number,
            "value": t.value,
            "state": t.state.name,
            "params": t.params,
            "duration_seconds": duration,
        })

    history_path = os.path.join(config.output_dir, "trial_history.json")
    with open(history_path, "w") as f:
        json.dump(trials_data, f, indent=2)
    print(f"  📁 Trial history saved to: {history_path}")

    # Save as YAML if available
    if YAML_AVAILABLE:
        yaml_path = os.path.join(config.output_dir, "best_config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
        print(f"  📁 YAML config saved to: {yaml_path}")

    # Generate visualizations if enabled
    if config.save_visualizations:
        _save_visualizations(study, config.output_dir)

    # Print application instructions
    _print_application_instructions(best_config)


def _save_visualizations(study: optuna.Study, output_dir: str) -> None:
    """Save Optuna visualization plots."""
    try:
        # Check if plotly is available
        from optuna.visualization import (  # type: ignore[import-untyped]
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )

        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(os.path.join(viz_dir, "optimization_history.html"))

        # Parameter importances
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(completed_trials) > 1:
            try:
                fig = plot_param_importances(study)
                fig.write_html(os.path.join(viz_dir, "param_importances.html"))
            except Exception:
                pass  # Not enough trials for importance

        # Parallel coordinate
        fig = plot_parallel_coordinate(study)
        fig.write_html(os.path.join(viz_dir, "parallel_coordinate.html"))

        print(f"  📊 Visualizations saved to: {viz_dir}/")

    except ImportError:
        logging.warning("Plotly not installed. Skipping visualizations.")
    except Exception as e:
        logging.warning(f"Failed to save visualizations: {e}")


def _print_application_instructions(best_config: Dict[str, Any]) -> None:
    """Print instructions for applying the best configuration."""
    print("\n" + "═" * 70)
    print("  HOW TO APPLY BEST PARAMETERS")
    print("═" * 70)

    ppo_params = best_config.get("ppo_params", {})
    reward_params = best_config.get("reward_params", {})

    if ppo_params:
        print("\n  1️⃣  Update envs/config.py (TradingConfig):")
        print("  " + "─" * 40)
        for key, value in ppo_params.items():
            if isinstance(value, float):
                print(f"      {key}: float = {value:.6f}")
            else:
                print(f"      {key}: int = {value}")

    if reward_params:
        print("\n  2️⃣  Update modules/reward/shared/reward_config.py (RewardConfig):")
        print("  " + "─" * 40)
        for key, value in reward_params.items():
            print(f"      {key}: float = {value:.6f}")

    print("\n  3️⃣  Run full training with tuned parameters:")
    print("  " + "─" * 40)
    print("      python train/train_ppo_hybrid.py --timesteps 1000000")
    print("\n" + "═" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def show_best_params(output_dir: str = "optuna_results") -> None:
    """Display best parameters from a completed study."""
    config_path = os.path.join(output_dir, "best_config.json")

    if not os.path.exists(config_path):
        print(f"[!] No results found at: {config_path}")
        print("    Run optimization first with: python train/optuna_tuner.py")
        return

    with open(config_path) as f:
        config = json.load(f)

    info = config.get("optimization_info", {})

    print("\n" + "═" * 70)
    print("  BEST OPTIMIZATION RESULTS")
    print("═" * 70)
    print(f"  Study:        {info.get('study_name', 'N/A')}")
    print(f"  Best trial:   #{info.get('best_trial', 'N/A')}")
    print(f"  Best reward:  {info.get('best_reward', 'N/A'):.2f}")
    print(f"  Total trials: {info.get('total_trials', 'N/A')}")
    print(f"  Completed at: {info.get('completed_at', 'N/A')}")
    print("─" * 70)

    _print_application_instructions(config)


def export_best_params(
    output_path: str,
    input_dir: str = "optuna_results",
) -> None:
    """Export best parameters to a config file."""
    config_path = os.path.join(input_dir, "best_config.json")

    if not os.path.exists(config_path):
        print(f"[!] No results found at: {config_path}")
        return

    with open(config_path) as f:
        config = json.load(f)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if output_path_obj.suffix == ".yaml" and YAML_AVAILABLE:
        with open(output_path_obj, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(output_path_obj, "w") as f:
            json.dump(config, f, indent=2)

    print(f"[✓] Best parameters exported to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced Optuna Hyperparameter Tuner for PPO Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python train/optuna_tuner.py --n_trials 10 --timesteps 20000

  # Full optimization
  python train/optuna_tuner.py --n_trials 50 --timesteps 100000 --tune_all

  # Tune only reward parameters
  python train/optuna_tuner.py --n_trials 30 --tune_reward

  # Tune only PPO parameters
  python train/optuna_tuner.py --n_trials 30 --tune_ppo

  # Resume a previous study
  python train/optuna_tuner.py --n_trials 50 --study_name my_study --resume

  # Show best parameters
  python train/optuna_tuner.py --show_best

  # Export best parameters
  python train/optuna_tuner.py --export_best config/tuned_params.yaml
        """,
    )

    # Trial settings
    trial_group = parser.add_argument_group("Trial Settings")
    trial_group.add_argument(
        "--n_trials", type=int, default=50,
        help="Number of optimization trials (default: 50)",
    )
    trial_group.add_argument(
        "--timesteps", type=int, default=50000, dest="timesteps_per_trial",
        help="Training timesteps per trial (default: 50000)",
    )
    trial_group.add_argument(
        "--eval_freq", type=int, default=10000,
        help="Evaluation frequency for pruning (default: 10000)",
    )
    trial_group.add_argument(
        "--n_eval_episodes", type=int, default=5,
        help="Episodes per evaluation (default: 5)",
    )

    # Tuning scope
    tune_group = parser.add_argument_group("Tuning Scope")
    tune_group.add_argument(
        "--tune_reward", action="store_true",
        help="Tune reward shaping parameters",
    )
    tune_group.add_argument(
        "--tune_ppo", action="store_true",
        help="Tune PPO algorithm parameters",
    )
    tune_group.add_argument(
        "--tune_all", action="store_true",
        help="Tune both reward and PPO parameters (default)",
    )

    # Environment
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument(
        "--data_dir", type=str, default="data/processed",
        help="Directory containing training data (default: data/processed)",
    )
    env_group.add_argument(
        "--balance", type=float, default=100000.0,
        help="Initial account balance (default: 100000 from risk_policy.yaml)",
    )

    # Study settings
    study_group = parser.add_argument_group("Study Settings")
    study_group.add_argument(
        "--study_name", type=str, default="ppo_trading_optimization",
        help="Optuna study name (default: ppo_trading_optimization)",
    )
    study_group.add_argument(
        "--resume", action="store_true",
        help="Resume existing study if found",
    )
    study_group.add_argument(
        "--n_jobs", type=int, default=1,
        help="Parallel jobs (default: 1, use -1 for all cores)",
    )
    study_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    # Pruner settings
    pruner_group = parser.add_argument_group("Pruner Settings")
    pruner_group.add_argument(
        "--pruner", type=str, default="median",
        choices=["median", "halving", "hyperband"],
        help="Pruner type (default: median)",
    )
    pruner_group.add_argument(
        "--startup_trials", type=int, default=5,
        help="Trials before pruning starts (default: 5)",
    )
    pruner_group.add_argument(
        "--warmup_steps", type=int, default=3,
        help="Evaluations before pruning (default: 3)",
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir", type=str, default="optuna_results",
        help="Output directory (default: optuna_results)",
    )
    output_group.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity level (default: 1)",
    )
    output_group.add_argument(
        "--no_viz", action="store_true",
        help="Disable visualization generation",
    )

    # Advanced
    advanced_group = parser.add_argument_group("Advanced")
    advanced_group.add_argument(
        "--timeout", type=int, default=None,
        help="Timeout in seconds (default: None)",
    )

    # Utility commands
    utility_group = parser.add_argument_group("Utility Commands")
    utility_group.add_argument(
        "--show_best", action="store_true",
        help="Show best parameters from previous run and exit",
    )
    utility_group.add_argument(
        "--export_best", type=str, metavar="PATH",
        help="Export best parameters to file and exit",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Handle utility commands
    if args.show_best:
        show_best_params(args.output_dir)
        return

    if args.export_best:
        export_best_params(args.export_best, args.output_dir)
        return

    # Determine tuning scope
    tune_reward = args.tune_reward or args.tune_all or (not args.tune_reward and not args.tune_ppo)
    tune_ppo = args.tune_ppo or args.tune_all or (not args.tune_reward and not args.tune_ppo)

    # Build configuration
    config = OptunaTunerConfig(
        n_trials=args.n_trials,
        timesteps_per_trial=args.timesteps_per_trial,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        tune_reward=tune_reward,
        tune_ppo=tune_ppo,
        data_dir=args.data_dir,
        initial_balance=args.balance,
        study_name=args.study_name,
        n_jobs=args.n_jobs,
        seed=args.seed,
        pruner_type=PrunerType(args.pruner),
        pruner_startup_trials=args.startup_trials,
        pruner_warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        verbose=args.verbose,
        save_visualizations=not args.no_viz,
        timeout_seconds=args.timeout,
    )

    # Run optimization
    try:
        study = run_optimization(config)

        if study.best_trial:
            print("\n[✓] Optimization completed successfully!")
            print(f"    Best reward: {study.best_value:.2f}")
            print(f"    See results in: {config.output_dir}/")
        else:
            print("\n[!] No successful trials. Check your configuration.")

    except KeyboardInterrupt:
        print("\n[!] Optimization interrupted by user.")
        print("    Progress saved. Resume with --resume flag.")
    except Exception as e:
        logging.exception(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
