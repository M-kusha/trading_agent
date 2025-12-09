#!/usr/bin/env python3
"""
Curriculum Training System for PPO Trading Agent
=================================================

Implements a structured training progression to prevent overfitting:

Phase 1: EXPLORATION (100K steps)
    - Learn basic market patterns
    - Random exploration with high entropy
    - Uses ExplorationTradingEnv (lightweight)
    
Phase 2: FOUNDATION (200K steps)
    - Train on older data (2023-07 to 2024-06)
    - Build core pattern recognition
    - Moderate entropy
    
Phase 3: REFINEMENT (200K steps)  
    - Fine-tune on recent data (2024-07 to 2025-06)
    - Lower entropy, exploit learned patterns
    
Phase 4: VALIDATION (50K steps)
    - Final polish on most recent data (2025-07+)
    - Evaluate generalization
    
Total: ~550K steps across 3-5 seeds = 1.5M-2.75M total steps

Training Timeline:
- 100K steps ≈ 30-60 minutes (depending on hardware)
- Full curriculum ≈ 3-6 hours per seed
- Multi-seed training ≈ 10-20 hours total
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from envs.exploration_env import ExplorationTradingEnv, ExplorationConfig


# ═══════════════════════════════════════════════════════════════════
# CURRICULUM PHASES
# ═══════════════════════════════════════════════════════════════════
@dataclass
class CurriculumPhase:
    """Configuration for a single training phase."""
    name: str
    timesteps: int
    data_start: str          # ISO date string "YYYY-MM-DD"
    data_end: str            # ISO date string "YYYY-MM-DD"
    ent_coef: float          # Entropy coefficient (exploration vs exploitation)
    learning_rate: float
    description: str = ""
    
    # Risk parameters (can adjust per phase)
    take_profit_eur: float = 500.0
    stop_loss_eur: float = 200.0
    max_drawdown_pct: float = 0.20


# Curriculum phases optimized for your data (2023-07 to 2025-12)
CURRICULUM_PHASES = [
    CurriculumPhase(
        name="exploration",
        timesteps=100_000,
        data_start="2023-07-01",
        data_end="2024-01-01",  # 6 months for exploration
        ent_coef=0.05,          # HIGH entropy - lots of exploration
        learning_rate=3e-4,     # Standard LR
        description="Learn basic market dynamics with high exploration",
        max_drawdown_pct=0.25,  # More lenient during exploration
    ),
    CurriculumPhase(
        name="foundation", 
        timesteps=200_000,
        data_start="2023-07-01",
        data_end="2024-06-30",  # Full year foundation
        ent_coef=0.02,          # Moderate entropy
        learning_rate=1e-4,     # Slower, more stable
        description="Build pattern recognition on historical data",
        max_drawdown_pct=0.20,
    ),
    CurriculumPhase(
        name="refinement",
        timesteps=200_000,
        data_start="2024-07-01",
        data_end="2025-06-30",  # Recent data for fine-tuning
        ent_coef=0.01,          # Low entropy - exploit patterns
        learning_rate=5e-5,     # Very slow, fine-tuning
        description="Fine-tune on recent market conditions",
        max_drawdown_pct=0.18,
    ),
    CurriculumPhase(
        name="validation",
        timesteps=50_000,
        data_start="2025-07-01",
        data_end="2025-12-31",  # Most recent for validation
        ent_coef=0.01,
        learning_rate=3e-5,
        description="Final polish and validation",
        max_drawdown_pct=0.15,  # Strict - real trading conditions
    ),
]

# Quick curriculum for testing (1/10th the steps)
QUICK_CURRICULUM = [
    CurriculumPhase(
        name="quick_explore",
        timesteps=10_000,
        data_start="2023-07-01",
        data_end="2024-06-30",
        ent_coef=0.03,
        learning_rate=3e-4,
        description="Quick exploration test",
    ),
    CurriculumPhase(
        name="quick_refine",
        timesteps=10_000,
        data_start="2024-07-01",
        data_end="2025-12-31",
        ent_coef=0.01,
        learning_rate=1e-4,
        description="Quick refinement test",
    ),
]


# ═══════════════════════════════════════════════════════════════════
# DATA SPLITTING
# ═══════════════════════════════════════════════════════════════════
def load_and_split_data(
    data_dir: str = "data/processed",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    instruments: Optional[List[str]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load data and filter by date range.
    
    Args:
        data_dir: Path to processed data
        start_date: Start date filter (inclusive) "YYYY-MM-DD"
        end_date: End date filter (inclusive) "YYYY-MM-DD"
        instruments: List of instruments to load
        
    Returns:
        Dict[instrument][timeframe] -> DataFrame
    """
    instruments = instruments or ["EURUSD", "XAUUSD"]
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    for file in os.listdir(data_dir):
        if not file.endswith(".csv"):
            continue
        
        # Parse filename
        parts = file.replace("_features.csv", "").replace(".csv", "").split("_")
        if len(parts) < 2:
            continue
            
        instrument = parts[0]
        timeframe = parts[1] if len(parts) > 1 else "H1"
        
        # Check if instrument is wanted
        if instrument not in instruments:
            continue
        
        # Load data
        filepath = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(filepath)
            
            # Find timestamp column
            time_col = None
            for col in ["timestamp", "time", "datetime", "date"]:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                print(f"[WARN] No time column in {file}, skipping date filter")
            else:
                # Convert to datetime
                df[time_col] = pd.to_datetime(df[time_col])
                
                # Apply date filters
                if start_date:
                    df = df[df[time_col] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df[time_col] <= pd.to_datetime(end_date)]
            
            if len(df) < 100:
                print(f"[WARN] {file}: only {len(df)} rows after filtering, skipping")
                continue
            
            # Ensure required columns
            for col in ["open", "high", "low", "close"]:
                if col not in df.columns:
                    print(f"[WARN] {file} missing {col}, skipping")
                    continue
            
            if "volume" not in df.columns:
                df["volume"] = 1.0
            
            data.setdefault(instrument, {})[timeframe] = df.reset_index(drop=True)
            print(f"[OK] {instrument}/{timeframe}: {len(df):,} bars ({start_date or 'start'} to {end_date or 'end'})")
            
        except Exception as e:
            print(f"[ERR] Failed to load {file}: {e}")
    
    return data


# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT FACTORY
# ═══════════════════════════════════════════════════════════════════
def create_curriculum_env(
    data: Dict[str, Dict[str, pd.DataFrame]],
    phase: CurriculumPhase,
    seed: int = 42,
) -> DummyVecEnv:
    """Create environment for a curriculum phase."""
    
    config = ExplorationConfig(
        instruments=list(data.keys()),
        take_profit_eur=phase.take_profit_eur,
        stop_loss_eur=phase.stop_loss_eur,
        max_drawdown_pct=phase.max_drawdown_pct,
        max_steps_per_episode=2000,
    )
    
    def make_env():
        env = ExplorationTradingEnv(data, config)
        Path("logs/curriculum").mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=f"logs/curriculum/monitor_{phase.name}.csv")
    
    set_random_seed(seed)
    return DummyVecEnv([make_env])


# ═══════════════════════════════════════════════════════════════════
# MULTI-SEED TRAINING
# ═══════════════════════════════════════════════════════════════════
@dataclass
class SeedResult:
    """Results from training with one seed."""
    seed: int
    final_reward: float
    best_reward: float
    total_trades: int
    win_rate: float
    final_balance: float
    model_path: str


def train_single_seed(
    seed: int,
    curriculum: List[CurriculumPhase],
    data_dir: str = "data/processed",
    output_dir: str = "models/curriculum",
) -> SeedResult:
    """Train full curriculum with one seed."""
    
    print(f"\n{'='*60}")
    print(f"TRAINING SEED {seed}")
    print(f"{'='*60}")
    
    model = None
    best_reward = -np.inf
    train_env = None  # Initialize to avoid unbound error
    
    for phase_idx, phase in enumerate(curriculum):
        print(f"\n--- Phase {phase_idx+1}/{len(curriculum)}: {phase.name} ---")
        print(f"    Timesteps: {phase.timesteps:,}")
        print(f"    Data: {phase.data_start} to {phase.data_end}")
        print(f"    Entropy: {phase.ent_coef}, LR: {phase.learning_rate}")
        
        # Load data for this phase
        data = load_and_split_data(
            data_dir=data_dir,
            start_date=phase.data_start,
            end_date=phase.data_end,
        )
        
        if not data:
            print(f"[ERR] No data for phase {phase.name}, skipping")
            continue
        
        # Create environments
        train_env = create_curriculum_env(data, phase, seed=seed)
        eval_env = create_curriculum_env(data, phase, seed=seed + 1000)
        
        # Create or update model
        if model is None:
            # First phase - create new model
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=phase.learning_rate,
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                gamma=0.95,
                gae_lambda=0.95,
                clip_range=0.15,
                ent_coef=phase.ent_coef,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.015,
                verbose=0,
                tensorboard_log=f"logs/tensorboard/curriculum_seed{seed}",
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=seed,
            )
        else:
            # Subsequent phases - update hyperparameters
            model.set_env(train_env)
            model.learning_rate = phase.learning_rate
            model.ent_coef = phase.ent_coef
        
        # Callbacks
        phase_dir = os.path.join(output_dir, f"seed{seed}", phase.name)
        os.makedirs(phase_dir, exist_ok=True)
        
        callbacks = [
            CheckpointCallback(
                save_freq=max(10000, phase.timesteps // 5),
                save_path=phase_dir,
                name_prefix=f"ppo_{phase.name}",
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=phase_dir,
                log_path=phase_dir,
                eval_freq=max(5000, phase.timesteps // 10),
                n_eval_episodes=5,
                deterministic=True,
            ),
        ]
        
        # Train
        try:
            model.learn(
                total_timesteps=phase.timesteps,
                callback=CallbackList(callbacks),
                tb_log_name=f"seed{seed}_{phase.name}",
                reset_num_timesteps=(phase_idx == 0),  # Only reset on first phase
                progress_bar=True,
            )
        except KeyboardInterrupt:
            print(f"[WARN] Phase {phase.name} interrupted")
            break
        
        # Track best reward from eval
        eval_log = os.path.join(phase_dir, "evaluations.npz")
        if os.path.exists(eval_log):
            eval_data = np.load(eval_log)
            if "results" in eval_data:
                phase_best = float(np.max(eval_data["results"]))
                if phase_best > best_reward:
                    best_reward = phase_best
        
        print(f"[OK] Phase {phase.name} complete")
    
    # Save final model
    final_path = os.path.join(output_dir, f"seed{seed}", "final_model.zip")
    if model:
        model.save(final_path)
        print(f"[SAVE] Final model: {final_path}")
    
    # Get final metrics from environment
    final_reward = 0.0
    final_balance = 100000.0
    total_trades = 0
    win_rate = 0.0
    
    if model and train_env is not None:
        # Quick evaluation
        env = train_env.envs[0]
        obs, _ = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            final_reward += float(reward)
            if done or truncated:
                final_balance = info.get("balance", 100000)
                total_trades = info.get("trade_count", 0)
                win_rate = info.get("win_rate", 0)
                break
    
    return SeedResult(
        seed=seed,
        final_reward=final_reward,
        best_reward=best_reward,
        total_trades=total_trades,
        win_rate=win_rate,
        final_balance=final_balance,
        model_path=final_path,
    )


def train_multi_seed(
    seeds: List[int],
    curriculum: List[CurriculumPhase],
    data_dir: str = "data/processed",
    output_dir: str = "models/curriculum",
) -> Dict[str, Any]:
    """Train curriculum with multiple seeds and aggregate results."""
    
    print("\n" + "="*70)
    print("MULTI-SEED CURRICULUM TRAINING")
    print("="*70)
    print(f"Seeds: {seeds}")
    print(f"Phases: {[p.name for p in curriculum]}")
    total_steps = sum(p.timesteps for p in curriculum)
    print(f"Total steps per seed: {total_steps:,}")
    print(f"Total training steps: {total_steps * len(seeds):,}")
    print("="*70)
    
    results: List[SeedResult] = []
    
    for seed in seeds:
        try:
            result = train_single_seed(
                seed=seed,
                curriculum=curriculum,
                data_dir=data_dir,
                output_dir=output_dir,
            )
            results.append(result)
        except Exception as e:
            print(f"[ERR] Seed {seed} failed: {e}")
    
    # Aggregate results
    if results:
        avg_reward = np.mean([r.best_reward for r in results])
        std_reward = np.std([r.best_reward for r in results])
        avg_winrate = np.mean([r.win_rate for r in results])
        best_seed = max(results, key=lambda r: r.best_reward)
        
        summary = {
            "seeds": seeds,
            "results": [asdict(r) for r in results],
            "summary": {
                "avg_best_reward": float(avg_reward),
                "std_best_reward": float(std_reward),
                "avg_win_rate": float(avg_winrate),
                "best_seed": best_seed.seed,
                "best_reward": float(best_seed.best_reward),
                "best_model_path": best_seed.model_path,
            },
            "curriculum": [asdict(p) if hasattr(p, '__dataclass_fields__') else {
                "name": p.name, "timesteps": p.timesteps
            } for p in curriculum],
            "trained_at": datetime.now().isoformat(),
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Average best reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average win rate: {avg_winrate*100:.1f}%")
        print(f"Best seed: {best_seed.seed} (reward: {best_seed.best_reward:.2f})")
        print(f"Best model: {best_seed.model_path}")
        print(f"Summary saved: {summary_path}")
        
        return summary
    
    return {"error": "No successful training runs"}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Curriculum Training for PPO Trading Agent")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds to use (default: 42 123 456)")
    parser.add_argument("--quick", action="store_true",
                        help="Use quick curriculum (1/10th steps) for testing")
    parser.add_argument("--phase", type=str, default=None,
                        help="Run only a specific phase (exploration, foundation, refinement, validation)")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Path to processed data")
    parser.add_argument("--output-dir", type=str, default="models/curriculum",
                        help="Output directory for models")
    parser.add_argument("--single-seed", type=int, default=None,
                        help="Train with only one seed")
    
    args = parser.parse_args()
    
    # Select curriculum
    if args.quick:
        curriculum = QUICK_CURRICULUM
        print("[MODE] Quick curriculum (testing)")
    else:
        curriculum = CURRICULUM_PHASES
        print("[MODE] Full curriculum")
    
    # Filter to single phase if specified
    if args.phase:
        curriculum = [p for p in curriculum if p.name == args.phase]
        if not curriculum:
            print(f"[ERR] Phase '{args.phase}' not found")
            return
    
    # Select seeds
    if args.single_seed is not None:
        seeds = [args.single_seed]
    else:
        seeds = args.seeds
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train
    train_multi_seed(
        seeds=seeds,
        curriculum=curriculum,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
