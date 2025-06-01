#!/usr/bin/env python3
"""
train_ppo_fixed.py – SB3 + Optuna driver for EnhancedTradingEnv (FX/Gold)
with reproducibility, safer pruning, GPU-auto-switch and robustness patches.
"""

import os
# suppress TensorFlow logs early
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ──────────────────────────────────────────────────────────────────────────────
# Global deterministic seeding (before any other imports of torch, gym, etc.)
# ──────────────────────────────────────────────────────────────────────────────
# 1) Fix Python hash seed
os.environ["PYTHONHASHSEED"] = "42"
# 2) Seed core libraries
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 3) Force CuDNN to be deterministic (at cost of performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────────────
import warnings

# each entry is (module, message regex, warning category)
_warnings_to_silence = [
    ("gymnasium.envs.registration", r".*Overriding environment.*", UserWarning),
    ("numpy",                        r"invalid value encountered in divide", RuntimeWarning),
    ("keras.src.layers.rnn.rnn",     r"Do not pass an `input_shape`/`input_dim` argument to a layer.*", UserWarning),
]

for mod, msg, cat in _warnings_to_silence:
    warnings.filterwarnings(
        "ignore",
        message=msg,
        category=cat,
        module=mod,
    )

import logging
import json
import random
from typing import List, Dict, Any

import numpy as np
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned


from envs.env import EnhancedTradingEnv
from utils.data_utils import load_data
from utils.meta_learning import AdaptiveMetaLearner

# ──────────────────────────────────────────────────────────────────────────────
# ░█▀▄░█▀█░█░█░█▀█░█▀█░█▀▀░░░█▀█░█▀█░█▀▄
# ░█▀▄░█▀█░▄▀▄░█▀▀░█░█░█▀▀░░░█▀▀░█▀█░█░█
# ░▀░▀░▀░▀░▀░▀░▀░░░▀▀▀░▀▀▀░░░▀░░░▀░▀░▀▀░  – global deterministic seed
# ──────────────────────────────────────────────────────────────────────────────
N_EVAL_EPISODES   = 10
NUM_ENVS = 4
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Config flags
# ──────────────────────────────────────────────────────────────────────────────
TEST_MODE = True
OBJECTIVE = "composite"  # keep for future multi‑objective experiments

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

# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

def _setup_logger(name: str, file_name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:  # already configured (e.g. hot reload)
        return logger
    fh = logging.FileHandler(os.path.join(log_dir, file_name), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    fh.setLevel(level)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch.setLevel(logging.WARNING)
    logger.addHandler(ch)
    return logger

env_logger = _setup_logger("env_logger", "training.log")
score_logger = _setup_logger("score_logger", "training_score.log")

# ──────────────────────────────────────────────────────────────────────────────
# Utility: neat training‑score summary for console / log
# ──────────────────────────────────────────────────────────────────────────────

def compute_training_rating(logs: List[Dict[str, Any]], rw=0.4, bw=0.4, dw=0.2) -> float:
    if not logs:
        return 0.0
    rewards = np.array([e["reward"] for e in logs], np.float32)
    balances = np.array([e["balance"] for e in logs], np.float32)
    drawdowns = np.array([e["drawdown"] for e in logs], np.float32)
    stability = 1.0 - np.std(rewards) / (np.mean(rewards) + 1e-8)
    growth = (balances[-1] - balances[0]) / (abs(balances[0]) + 1e-8)
    max_dd = np.max(drawdowns)
    score = rw * stability + bw * growth + dw * (1.0 - max_dd)
    return float(np.clip(score, 0.0, 1.0) * 100.0)


def log_training_progress(logs: List[Dict[str, Any]], step: int) -> None:
    msg = f"[Overall @ Step {step}] Training Score: {compute_training_rating(logs):.2f}/100"
    print(f"\n➡️  {msg}\n")
    score_logger.info(msg)

# ──────────────────────────────────────────────────────────────────────────────
# Domain‑aware pruner (inherits Optuna MedianPruner)
# ──────────────────────────────────────────────────────────────────────────────

class ForexGoldPruner(MedianPruner):
    """MedianPruner + extra kill‑rules on DD / correlation / pattern diversity."""

    def __init__(self, max_dd_thresh=0.35, corr_thresh=0.65, **kwargs):
        super().__init__(**kwargs)
        self.max_dd_thresh = max_dd_thresh
        self.corr_thresh = corr_thresh

    def prune(self, study, trial) -> bool:
        # fall back to normal pruning first
        if super().prune(study, trial):
            return True
        metrics = trial.user_attrs.get("metrics", {})
        if metrics.get("max_dd", 0) > self.max_dd_thresh:
            return True
        if abs(metrics.get("correlation", 0)) > self.corr_thresh:
            return True
        if metrics.get("exit_diversity", 2) < 2:  # ensure at least 2 types of exits
            return True
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers for post‑trial ranking
# ──────────────────────────────────────────────────────────────────────────────

def calculate_trial_metrics(trial: optuna.trial.Trial, initial_balance: float) -> Dict[str, float]:
    logs = trial.user_attrs.get("full_logs", [])
    all_balances, all_returns = [], []
    exit_reasons, correlations = [], []

    for ep in logs:
        balances = [step["balance"] for step in ep]
        all_balances.extend(balances)
        returns = np.diff(balances) / (np.array(balances[:-1]) + 1e-8)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        all_returns.extend(returns)
        for step in ep:
            exit_reasons.extend(step["exit_reasons"])
            correlations.append(step["correlation"])

    correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)

    sharpe = (np.nanmean(all_returns) / (np.nanstd(all_returns) + 1e-8)) * np.sqrt(250)
    bals_arr = np.array(all_balances, np.float32)
    peak = np.maximum.accumulate(bals_arr)
    max_dd = np.max((peak - bals_arr) / (peak + 1e-8)) if bals_arr.size else 0.0
    profit = bals_arr[-1] - initial_balance if bals_arr.size else 0.0
    profit_factor = profit / max(1.0, initial_balance - np.min(bals_arr)) if bals_arr.size else 0.0
    corr_mean = float(np.mean(np.abs(correlations))) if correlations else 0.0
    exit_diversity = len(set(exit_reasons))

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "profit_factor": profit_factor,
        "correlation": corr_mean,
        "exit_diversity": exit_diversity,
    }


def rank_trials(study: optuna.Study) -> List[int]:
    """Return trial numbers sorted by composite metric (same as objective)."""
    ranked: List[tuple[int, float]] = []
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

# ──────────────────────────────────────────────────────────────────────────────
# Callbacks (unchanged except minor dtype / seed tweaks)
# ──────────────────────────────────────────────────────────────────────────────

class OptunaPruningCallback(BaseCallback):
    """Report mean‑reward to Optuna & trigger pruning."""

    def __init__(
        self,
        trial: optuna.trial.Trial,
        eval_env: DummyVecEnv,
        eval_freq: int,
        n_eval_episodes: int = N_EVAL_EPISODES,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._last_eval = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval) < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        rewards: list[float] = []
        depth = 5 if self.num_timesteps < 0.5 * TIMESTEPS_PER_TRIAL else self.n_eval_episodes
        for _ in range(depth):
            obs, _ = self.eval_env.reset()
            done = False
            ep_r = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = self.eval_env.step(action)
                done = term or trunc
                ep_r += float(r)
            rewards.append(ep_r)

        mean_reward = float(np.mean(rewards))
        self.trial.report(mean_reward, self.num_timesteps)
        if self.trial.should_prune():
            raise TrialPruned()
        return True



class HumanLoggerCallback(BaseCallback):
    def __init__(self, print_freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.summary_logs: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            metrics = self.training_env.envs[0].get_metrics()
            self.summary_logs.append(
                {
                    "balance": metrics["balance"],
                    "reward": float(self.locals["rewards"][0]),
                    "drawdown": metrics["drawdown"],
                }
            )
        return True


class DetailedTensorboardCallback(BaseCallback):
    """Rich TB logging every log_freq calls."""

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.training_logs: list[dict[str, Any]] = []

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
            self.writer.add_scalar("env/avg_pnl", m["avg_pnl"], ts)

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


class StableTradingPPOAgent(PPO):
    """PPO + meta‑learner wrapper."""

    def __init__(self, *args, meta_lr: float = 1e-4, adaptation_steps: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_learner = AdaptiveMetaLearner(self.policy, meta_lr=meta_lr, adaptation_steps=adaptation_steps)

# ──────────────────────────────────────────────────────────────────────────────
# Optuna objective – one trial
# ──────────────────────────────────────────────────────────────────────────────

def optimize_agent(trial: optuna.trial.Trial) -> float:
    print(f"\n🚀  Starting Trial #{trial.number + 1} …")
    env_logger.info(f"→ Trial #{trial.number + 1} starting")

    # Initialise empty metrics so custom pruner always has keys
    trial.set_user_attr(
        "metrics",
        {
            "sharpe": 0.0,
            "max_dd": 1.0,
            "profit_factor": 0.0,
            "correlation": 0.0,
            "exit_diversity": 0,
        },
    )

    # 1) Hyper‑param sampling
    hp = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [2_048, 4_096, 8_192]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
        "no_trade_penalty": trial.suggest_float("no_trade_penalty", 0.0, 1.0),
    }
    no_trade_penalty = hp.pop("no_trade_penalty")

    # 2) Environment factories
    data = load_data("data/processed")


    def make_env(rank: int):
        def _init():
            return EnhancedTradingEnv(
                data,
                initial_balance=3_000,
                max_steps=200,
                debug=False,
                checkpoint_dir="checkpoints",
                no_trade_penalty=no_trade_penalty,
                init_seed=GLOBAL_SEED + rank,
            )
        return _init

    # instead of DummyVecEnv([make_env])
    train_env = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])
    eval_env  = DummyVecEnv([make_env(i+NUM_ENVS) for i in range(NUM_ENVS)])

    # 3) Device selection – use GPU if available
    DEVICE = "cpu"

    # 4) Model
    model = StableTradingPPOAgent(
        "MlpPolicy",
        train_env,
        device=DEVICE,
        policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
        meta_lr=1e-4,
        adaptation_steps=3,
        **hp,
        verbose=0,
        tensorboard_log="logs/tensorboard",
    )

    # 5) Callbacks
    prune_cb = OptunaPruningCallback(
    trial,
    eval_env,
    eval_freq=PRUNER_INTERVAL_STEPS,
    n_eval_episodes=N_EVAL_EPISODES,
    )

    cb_list = [prune_cb, ProgressBarCallback(), HumanLoggerCallback(print_freq=PRUNER_INTERVAL_STEPS)]

    # 6) Train with try/except so pruned trial still stores metrics
    try:
        model.learn(total_timesteps=TIMESTEPS_PER_TRIAL, callback=cb_list)
    except TrialPruned:
        # compute metrics on partially trained model for pruner rules transparency
        metrics = calculate_trial_metrics(trial, 3_000)
        trial.set_user_attr("metrics", metrics)
        raise

    # 7) Post‑training evaluation (3 episodes)
    detailed_logs: list[list[dict[str, Any]]] = []
    for _ in range(N_EVAL_EPISODES):
        obs, _ = eval_env.reset()
        done = False
        ep_log: list[dict[str, Any]] = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(action)
            done = term or trunc
            ep_log.append(
                {
                    "balance": info["balance"],
                    "pnl": info["pnl"],
                    "drawdown": info["drawdown"],
                    "volatility": eval_env.envs[0].get_volatility_profile(),
                    "exit_reasons": eval_env.envs[0].get_trade_exit_reasons(),
                    "correlation": eval_env.envs[0].get_current_correlation(),
                }
            )
        detailed_logs.append(ep_log)

    # attach full logs for later analysis & compute metrics
    trial.set_user_attr("full_logs", detailed_logs)
    metrics = calculate_trial_metrics(trial, 3_000)
    trial.set_user_attr("metrics", metrics)

    comp_score = (
        0.4 * metrics["sharpe"]
        + 0.3 * (1.0 - metrics["max_dd"])
        + 0.2 * (metrics["exit_diversity"] / 5.0)
        + 0.1 * metrics["profit_factor"]
        - 0.2 * metrics["correlation"]
    )

    env_logger.info(
        "🎯  Trial #%d → Sharpe=%.3f DD=%.3f PF=%.3f Corr=%.3f Exits=%d Score=%.3f",
        trial.number + 1,
        metrics["sharpe"],
        metrics["max_dd"],
        metrics["profit_factor"],
        metrics["correlation"],
        metrics["exit_diversity"],
        comp_score,
    )

    return comp_score

# ──────────────────────────────────────────────────────────────────────────────
# Top‑level study orchestration + final training
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ──────────────────────────────────────────────────────────────────────────
    # 1) Optuna study setup
    # ──────────────────────────────────────────────────────────────────────────
    pruner = ForexGoldPruner(
        max_dd_thresh=0.30,
        corr_thresh=0.60,
        n_startup_trials=PRUNER_STARTUP_TRIALS,
        n_warmup_steps=PRUNER_WARMUP_STEPS,
        interval_steps=PRUNER_INTERVAL_STEPS,
    )

    study = optuna.create_study(
        study_name="ppo_trading_optimization",
        storage="sqlite:///optuna_ppo.db",
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 2) Hyperparameter optimization
    # ──────────────────────────────────────────────────────────────────────────
    study.optimize(optimize_agent, n_trials=N_TRIALS, show_progress_bar=False)

    ranked = rank_trials(study)
    if not ranked:
        raise RuntimeError("No successful trials – check logs for errors/pruning causes.")

    best_trial = study.trials[ranked[0]]
    best_params = best_trial.params.copy()
    no_trade_penalty = best_params.pop("no_trade_penalty")

    env_logger.info(
        "Best trial #%d → score=%.4f, params=%s",
        best_trial.number + 1,
        best_trial.value,
        best_params,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 3) Cache best hyperparameters
    # ──────────────────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    with open("models/best_params_ppo.json", "w", encoding="utf-8") as fp:
        json.dump(best_params | {"no_trade_penalty": no_trade_penalty}, fp, indent=4)

    # ──────────────────────────────────────────────────────────────────────────
    # 4) Final training with parallel envs & GPU auto-detect
    # ──────────────────────────────────────────────────────────────────────────
    from stable_baselines3.common.vec_env import SubprocVecEnv

    # Load your data once
    data = load_data("data/processed")

    # Create NUM_ENVS parallel environments
    NUM_ENVS = 4
    def make_env_final(rank: int):
        def _init():
            return EnhancedTradingEnv(
                data,
                initial_balance=3_000,
                max_steps=500,
                debug=False,
                checkpoint_dir="checkpoints",
                no_trade_penalty=no_trade_penalty,
                init_seed=GLOBAL_SEED + rank,
            )
        return _init

    env = DummyVecEnv([make_env_final(i) for i in range(NUM_ENVS)])

    # Auto-select device
    DEVICE = "cpu"
    env_logger.info(f"Using device for final training: {DEVICE}")

    # Instantiate and train the final model
    final_model = StableTradingPPOAgent(
        "MlpPolicy",
        env,
        device=DEVICE,
        policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
        meta_lr=1e-4,
        adaptation_steps=3,
        **best_params,
        verbose=1,
        tensorboard_log="logs/tensorboard",
    )

    final_model.learn(
        total_timesteps=FINAL_TRAINING_STEPS,
        callback=[
            DetailedTensorboardCallback(log_freq=TB_LOG_FREQ),
            HumanLoggerCallback(print_freq=20),
            ProgressBarCallback(),
        ],
    )

    final_model.save("models/ppo_final_model.zip")
    env_logger.info("✅ Final PPO model saved → models/ppo_final_model.zip")



if __name__ == "__main__":
    # prevent PyTorch from hogging all CPU threads on small VPS
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    main()

 