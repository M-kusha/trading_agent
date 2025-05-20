#!/usr/bin/env python3
"""
common.py – shared configuration, seeding, logging, metrics, pruner, and callbacks
"""
import os
# suppress TensorFlow logs early

import random
import numpy as np
import torch
# 2) Seed core libraries
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
# 3) Force CuDNN to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Logging helpers
import logging
from typing import List, Dict, Any

def setup_logger(name: str, file_name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(os.path.join("logs", file_name), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    fh.setLevel(level)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch.setLevel(logging.WARNING)
    logger.addHandler(ch)
    return logger

env_logger = setup_logger("env_logger", "training.log")
score_logger = setup_logger("score_logger", "training_score.log")

# Global seed for environment instantiation
GLOBAL_SEED = 42

# Utility: neat training-score summary
import numpy as _np

def compute_training_rating(logs: List[Dict[str, Any]], rw=0.4, bw=0.4, dw=0.2) -> float:
    if not logs:
        return 0.0
    rewards = _np.array([e["reward"] for e in logs], _np.float32)
    balances = _np.array([e["balance"] for e in logs], _np.float32)
    drawdowns = _np.array([e["drawdown"] for e in logs], _np.float32)
    stability = 1.0 - _np.std(rewards) / (_np.mean(rewards) + 1e-8)
    growth = (balances[-1] - balances[0]) / (abs(balances[0]) + 1e-8)
    max_dd = _np.max(drawdowns)
    score = rw * stability + bw * growth + dw * (1.0 - max_dd)
    return float(_np.clip(score, 0.0, 1.0) * 100.0)

def log_training_progress(logs: List[Dict[str, Any]], step: int) -> None:
    msg = f"[Overall @ Step {step}] Training Score: {compute_training_rating(logs):.2f}/100"
    print(f"\n➡️  {msg}\n")
    score_logger.info(msg)

# Domain-aware pruner
import optuna
from optuna.pruners import MedianPruner
class ForexGoldPruner(MedianPruner):
    """MedianPruner + extra kill-rules on DD / correlation / pattern diversity."""
    def __init__(self, max_dd_thresh=0.35, corr_thresh=0.65, **kwargs):
        super().__init__(**kwargs)
        self.max_dd_thresh = max_dd_thresh
        self.corr_thresh = corr_thresh

    def prune(self, study, trial) -> bool:
        if super().prune(study, trial):
            return True
        metrics = trial.user_attrs.get("metrics", {})
        if metrics.get("max_dd", 0) > self.max_dd_thresh:
            return True
        if abs(metrics.get("correlation", 0)) > self.corr_thresh:
            return True
        if metrics.get("exit_diversity", 2) < 2:
            return True
        return False

# Metric helpers for post-trial ranking
from typing import List as _List, Dict as _Dict

def calculate_trial_metrics(trial: optuna.trial.Trial, initial_balance: float) -> _Dict[str, float]:
    logs = trial.user_attrs.get("full_logs", [])
    all_balances, all_returns = [], []
    exit_reasons, correlations = [], []

    for ep in logs:
        balances = [step["balance"] for step in ep]
        all_balances.extend(balances)
        returns = _np.diff(balances) / (_np.array(balances[:-1]) + 1e-8)
        returns = _np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        all_returns.extend(returns)
        for step in ep:
            exit_reasons.extend(step["exit_reasons"])
            correlations.append(step["correlation"])

    correlations = _np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    sharpe = (_np.nanmean(all_returns) / (_np.nanstd(all_returns) + 1e-8)) * _np.sqrt(250)
    bals_arr = _np.array(all_balances, _np.float32)
    peak = _np.maximum.accumulate(bals_arr)
    max_dd = _np.max((peak - bals_arr) / (peak + 1e-8)) if bals_arr.size else 0.0
    profit = bals_arr[-1] - initial_balance if bals_arr.size else 0.0
    profit_factor = profit / max(1.0, initial_balance - _np.min(bals_arr)) if bals_arr.size else 0.0
    corr_mean = float(_np.mean(_np.abs(correlations))) if correlations else 0.0
    exit_diversity = len(set(exit_reasons))

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "profit_factor": profit_factor,
        "correlation": corr_mean,
        "exit_diversity": exit_diversity,
    }

# Trial-ranking by composite metric

def rank_trials(study: optuna.Study) -> _List[int]:
    ranked: _List[tuple[int, float]] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        m = t.user_attrs["metrics"]
        score = (
            0.4 * m["sharpe"]
            + 0.3 * (1.0 - m["max_dd"])
            + 0.2 * (m["exit_diversity"] / 5.0)
            + 0.1 * m["profit_factor"]
            - 0.2 * m["correlation"]
        )
        ranked.append((t.number, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [num for num, _ in ranked]

# Callbacks
from stable_baselines3.common.callbacks import BaseCallback
from optuna.exceptions import TrialPruned

class OptunaPruningCallback(BaseCallback):
    """Report mean-reward to Optuna & trigger pruning."""
    def __init__(
        self,
        trial: optuna.trial.Trial,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._last_eval = 0

    def _on_step(self) -> bool:
        from numpy import mean

        # only evaluate every self.eval_freq timesteps
        if (self.num_timesteps - self._last_eval) < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        rewards: list[float] = []
        depth = 5 if self.num_timesteps < 0.5 * globals().get("TIMESTEPS_PER_TRIAL", 0) else self.n_eval_episodes
        for _ in range(depth):
            # reset may return multiple values; extract obs only
            reset_result = self.eval_env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            done = False
            ep_r = 0.0
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            step = self.eval_env.step(action)
            if len(step) == 5:
                # Gymnasium style
                obs, r, term, trunc, _ = step
                done = term or trunc
            else:
                # VecEnv / classic Gym style
                obs, r, done, _ = step
            ep_r += float(r)

            rewards.append(ep_r)

        mean_reward = float(mean(rewards))
        self.trial.report(mean_reward, self.num_timesteps)
        if self.trial.should_prune():
            raise TrialPruned()
        return True

class HumanLoggerCallback(BaseCallback):
    """Log balance/reward/drawdown for human-readable summaries."""
    def __init__(self, print_freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.summary_logs: list[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            metrics = self.training_env.envs[0].get_metrics()
            self.summary_logs.append(
                {
                    "balance": metrics["balance"],
                    "reward": float(self.locals.get("rewards", [0])[0]),
                    "drawdown": metrics["drawdown"],
                }
            )
        return True

class DetailedTensorboardCallback(BaseCallback):
    """Rich TB logging every log_freq calls."""
    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.training_logs: list[Dict[str, Any]] = []

    def _on_training_start(self) -> None:
        from torch.utils.tensorboard import SummaryWriter
        logdir = getattr(self.model, "tensorboard_log", "logs/tensorboard")
        self.writer = SummaryWriter(log_dir=logdir)
        env_logger.info("►► TensorBoard logging started")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            ts = self.num_timesteps
            env = self.training_env.envs[0]
            m = env.get_metrics()

            self.writer.add_scalar("env/balance", m["balance"], ts)
            self.writer.add_scalar("env/win_rate", m["win_rate"], ts)
            self.writer.add_scalar("env_avg_pnl", m.get("avg_pnl", 0.0), ts)

            gm = env.get_genome_metrics()
            for k, v in gm.items():
                self.writer.add_scalar(f"genome/{k}", v, ts)

            # Accumulate for human summary
            dd = (env.peak_balance - env.balance) / (env.peak_balance + 1e-8)
            self.training_logs.append({"balance": env.balance, "reward": 0, "drawdown": dd})

            if ts % 1_000 == 0:
                log_training_progress(self.training_logs, ts)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()
        env_logger.info("►► TensorBoard logging ended")

# Config flags
TEST_MODE = True
OBJECTIVE = "composite"
if TEST_MODE:
    N_TRIALS = 5
    TIMESTEPS_PER_TRIAL = 5_000
    FINAL_TRAINING_STEPS = 10_000
    PRUNER_STARTUP_TRIALS = 1
    PRUNER_WARMUP_STEPS = 1_000
    PRUNER_INTERVAL_STEPS = 1_000
    TB_LOG_FREQ = 500
else:
    N_TRIALS = 50
    TIMESTEPS_PER_TRIAL = 500_000
    FINAL_TRAINING_STEPS = 5_000_000
    PRUNER_STARTUP_TRIALS = 5
    PRUNER_WARMUP_STEPS = 50_000
    PRUNER_INTERVAL_STEPS = 100_000
    TB_LOG_FREQ = 100

# VecEnv & episodes constants
NUM_ENVS = 4
N_EVAL_EPISODES = 10
LOG_DIR = "logs"
