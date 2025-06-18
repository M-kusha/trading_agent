#!/usr/bin/env python3
"""
TD3 + Meta-Learner + Optuna for Forex Trading
Cleaned up to mirror the SAC training script: warning filters, deterministic seeding,
unified SubprocVecEnv for train & eval, one-time extraction of no_trade_penalty,
and no experimental Optuna flags.
"""
from __future__ import annotations
import warnings
# ─────────────────────────── Std-lib
_warns = [
    ("gymnasium.envs.registration", r".*Overriding environment.*", UserWarning),
    ("numpy", r"Mean of empty slice", RuntimeWarning),
    ("numpy", r"Degrees of freedom <= 0 for slice", RuntimeWarning),
    ("numpy", r"invalid value encountered in scalar divide", RuntimeWarning),
    ("numpy", r"invalid value encountered in subtract", RuntimeWarning),
    ("keras.src.layers.rnn.rnn", r"Do not pass an `input_shape`.*", UserWarning),
]
for mod_re, msg_re, cat in _warns:
    warnings.filterwarnings("ignore", message=msg_re, category=cat, module=mod_re)
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
import os
import sys
import io
import random
import json
import logging

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

# ───────────────────────── Environment flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # silence TF INFO
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # silence oneDNN banner
torch_threads = min(4, os.cpu_count() or 1)


# ───────────────────────── Third-party
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import optuna
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm

# ───────────────────────── Project modules
from envs.env import EnhancedTradingEnv
from utils.data_utils import load_data
from utils.meta_learning import AdaptiveMetaLearner, MetaLearningCallback

# ╔═════════════════════════════════════════════════════════════════════╗
# ║                        Global configuration                        ║
# ╚═════════════════════════════════════════════════════════════════════╝
@dataclass
class TrainingConfig:
    test_mode: bool = True
    num_envs: int = 1
    global_seed: int = 42

    # derived:
    n_trials: int = 0
    timesteps_per_trial: int = 0
    final_training_steps: int = 0
    pruner_startup_trials: int = 0
    pruner_warmup_steps: int = 0
    pruner_interval_steps: int = 0
    tb_log_freq: int = 0
    n_eval_episodes: int = 10

    def __post_init__(self):
        if self.test_mode:
            self.n_trials = 5
            self.timesteps_per_trial = 1_000
            self.final_training_steps = 1_000
            self.pruner_startup_trials = 1
            self.pruner_warmup_steps = 100
            self.pruner_interval_steps = 500
            self.tb_log_freq = 500
        else:
            self.n_trials = 50
            self.timesteps_per_trial = 500_000
            self.final_training_steps = 5_000_000
            self.pruner_startup_trials = 5
            self.pruner_warmup_steps = 50_000
            self.pruner_interval_steps = 100_000
            self.tb_log_freq = 100
            self.n_eval_episodes = 5

# CLI toggle
parser = argparse.ArgumentParser()
parser.add_argument("--prod", action="store_true", help="Run full-budget training.")
cfg = TrainingConfig(test_mode=not parser.parse_args().prod)

# ╔═════════════════════════════════════════════════════════════════════╗
# ║                  Deterministic seeding helper                      ║
# ╚═════════════════════════════════════════════════════════════════════╝
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(torch_threads)
    torch.set_float32_matmul_precision("high")

# ╔═════════════════════════════════════════════════════════════════════╗
# ║                             Loggers                                ║
# ╚═════════════════════════════════════════════════════════════════════╝
LOG_DIR = "logs"
MODEL_DIR = "models"
CKPT_DIR = "checkpoints"
for p in (LOG_DIR, MODEL_DIR, CKPT_DIR):
    os.makedirs(p, exist_ok=True)

# Optional UTF-8 console re-encode for Windows
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def _setup_logger(name: str, file_name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fh = logging.FileHandler(os.path.join(LOG_DIR, file_name), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    fh.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch.setLevel(logging.WARNING)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger

env_logger   = _setup_logger("env_logger",   "training.log")
score_logger = _setup_logger("score_logger", "training_score.log")





for _lg in (env_logger, score_logger):
    _lg.addFilter(lambda record: record.levelno >= logging.INFO)

# ╔═════════════════════════════════════════════════════════════════════╗
# ║                        Metrics helpers                             ║
# ╚═════════════════════════════════════════════════════════════════════╝
def compute_training_rating(logs: List[Dict[str, Any]], rw=0.4, bw=0.4, dw=0.2) -> float:
    if len(logs) < 2:
        return 0.0
    arr = pd.DataFrame(logs)
    rewards = arr["reward"].astype(np.float32).values
    balances = arr["balance"].astype(np.float32).values
    drawdowns = arr["drawdown"].astype(np.float32).values
    stability = 1.0 - rewards.std() / (rewards.mean() + 1e-8) if rewards.size>1 else 0.0
    growth = (balances[-1] - balances[0]) / (abs(balances[0]) + 1e-8) if balances.size>1 else 0.0
    max_dd = drawdowns.max() if drawdowns.size else 0.0
    score = rw*stability + bw*growth + dw*(1.0 - max_dd)
    return float(np.clip(score, 0.0, 1.0) * 100.0)

def log_training_progress(logs: List[Dict[str, Any]], step: int):
    msg = f"[Overall @ {step}] Training Score: {compute_training_rating(logs):.2f}/100"
    env_logger.info(msg)
    score_logger.info(msg)

# ╔═════════════════════════════════════════════════════════════════════╗
# ║                             Callbacks                              ║
# ╚═════════════════════════════════════════════════════════════════════╝
class SimpleTqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int, trial_id: int, print_freq=1_000):
        super().__init__(verbose=0)
        self.total = total_timesteps
        self.print_freq = print_freq
        self.pbar = tqdm(
            total=total_timesteps,
            desc=f"Trial {trial_id}" if trial_id>=0 else "Final",
            position=trial_id % 10,
            leave=False,
            ascii=True,
            colour="cyan",
        )
    def _on_step(self):
        self.pbar.update(1)
        if self.num_timesteps % self.print_freq == 0:
            self.pbar.set_postfix_str(f"{self.num_timesteps:>7}/{self.total}")
        return True
    def _on_training_end(self):
        self.pbar.close()

class WatchdogNaNCallback(BaseCallback):
    def _on_step(self):
        if np.isnan(self.locals["rewards"]).any() or np.isinf(self.locals["rewards"]).any():
            raise TrialPruned()
        return True

class OptunaPruningCallback(BaseCallback):
    def __init__(self, trial: optuna.trial.Trial, eval_env: SubprocVecEnv, eval_freq: int):
        super().__init__(verbose=0)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self._last_eval = 0
    def _on_step(self):
        if (self.num_timesteps - self._last_eval) < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        mean_reward, _ = evaluate_policy(self.model, self.eval_env,
                                         n_eval_episodes=cfg.n_eval_episodes, warn=False)
        if not np.isfinite(mean_reward):
            mean_reward = -np.inf
        self.trial.report(mean_reward, self.num_timesteps)
        if self.trial.should_prune():
            raise TrialPruned()
        return True

class HumanLoggerCallback(BaseCallback):
    def __init__(self, print_freq=1_000):
        super().__init__(verbose=0)
        self.print_freq = print_freq
        self.logs: List[Dict[str, Any]] = []
    def _on_step(self):
        if self.num_timesteps % self.print_freq == 0:
            m = self.training_env.envs[0].get_metrics()
            self.logs.append({
                "balance": m["balance"],
                "reward": float(self.locals["rewards"][0]),
                "drawdown": m["drawdown"],
            })
            log_training_progress(self.logs, self.num_timesteps)
        return True

class DetailedTensorboardCallback(BaseCallback):
    def __init__(self, log_freq: int):
        super().__init__(verbose=0)
        self.log_freq = log_freq
        self.writer: SummaryWriter | None = None
    def _on_training_start(self):
        path = os.path.join(LOG_DIR, "tensorboard")
        self.writer = SummaryWriter(log_dir=path)
        env_logger.info("TensorBoard logging started")
    def _on_step(self):
        if self.n_calls % self.log_freq != 0 or self.writer is None:
            return True
        ts = self.num_timesteps
        env = self.training_env.envs[0]
        m = env.get_metrics()
        self.writer.add_scalar("env/balance", m["balance"], ts)
        self.writer.add_scalar("env/win_rate", m["win_rate"], ts)
        self.writer.add_scalar("env/avg_pnl", m["avg_pnl"], ts)
        self.writer.add_scalar("env/drawdown", m["drawdown"], ts)
        for k, v in env.get_genome_metrics().items():
            self.writer.add_scalar(f"genome/{k}", v, ts)
        return True
    def _on_training_end(self):
        if self.writer:
            self.writer.close()
        env_logger.info("TensorBoard logging ended")

class StableTradingTD3(TD3):
    def __init__(self, *args, meta_lr=1e-4, adaptation_steps=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_learner = AdaptiveMetaLearner(self.policy,
                                                meta_lr=meta_lr,
                                                adaptation_steps=adaptation_steps)

# ╔═════════════════════════════════════════════════════════════════════╗
# ║                        Optuna helpers / metrics                     ║
# ╚═════════════════════════════════════════════════════════════════════╝
class ForexGoldPruner(MedianPruner):
    def __init__(self, max_dd_thresh=0.35, corr_thresh=0.65, **kwargs):
        super().__init__(**kwargs)
        self.max_dd_thresh = max_dd_thresh
        self.corr_thresh = corr_thresh
    def prune(self, study, trial) -> bool:
        if super().prune(study, trial):
            return True
        m = trial.user_attrs.get("metrics", {})
        if m.get("max_dd", 0) > self.max_dd_thresh:
            return True
        if abs(m.get("correlation", 0.0)) > self.corr_thresh:
            return True
        if m.get("exit_diversity", 2) < 2:
            return True
        return False

def calculate_trial_metrics(trial: optuna.trial.Trial,
                            initial_balance: float) -> Dict[str, float]:
    logs = trial.user_attrs.get("full_logs", [])
    if not logs:
        return dict(sharpe=0.0, max_dd=1.0,
                    profit_factor=0.0, correlation=0.0,
                    exit_diversity=0)
    flat = [step for ep in logs for step in ep]
    df = pd.DataFrame(flat)
    balances = df["balance"].astype(np.float32).values
    if balances.size < 2:
        return dict(sharpe=0.0, max_dd=1.0,
                    profit_factor=0.0, correlation=0.0,
                    exit_diversity=0)
    returns = np.diff(balances) / (balances[:-1] + 1e-8)
    returns = np.nan_to_num(returns)
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(250)
    peak = np.maximum.accumulate(balances)
    max_dd = np.max((peak - balances) / (peak + 1e-8))
    profit = balances[-1] - initial_balance
    profit_factor = profit / max(1.0, initial_balance - balances.min())
    corr_mean = float(np.abs(df["correlation"].fillna(0.0)).mean())
    exit_diversity = len({r for exits in df["exit_reasons"] for r in exits})
    return dict(sharpe=sharpe,
                max_dd=max_dd,
                profit_factor=profit_factor,
                correlation=corr_mean,
                exit_diversity=exit_diversity)

def rank_trials(study: optuna.Study) -> List[int]:
    scored: List[tuple[int, float]] = []
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        m = t.user_attrs["metrics"]
        score = (
            0.4 * m["sharpe"]
            + 0.3 * (1.0 - m["max_dd"])
            + 0.2 * (m["exit_diversity"] / 5.0)
            + 0.1 * m["profit_factor"]
            - 0.2 * m["correlation"]
        )
        scored.append((t.number, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored]

# ╔═════════════════════════════════════════════════════════════════════╗
# ║                       Optuna objective                             ║
# ╚═════════════════════════════════════════════════════════════════════╝
def optimise_agent(trial: optuna.trial.Trial) -> float:
    env_logger.info(" TD3 Trial %d starting", trial.number + 1)

    # 1) Sample hyperparameters (names matching TD3.__init__)
    hp = dict(
        learning_rate       = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        batch_size          = trial.suggest_categorical("batch_size", [64, 128, 256]),
        buffer_size         = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
        tau                 = trial.suggest_float("tau", 0.005, 0.02),
        gamma               = trial.suggest_float("gamma", 0.90, 0.9999),
        train_freq          = trial.suggest_categorical("train_freq", [1, 2, 4, 8]),
        gradient_steps      = trial.suggest_categorical("gradient_steps", [1, 2, 4, 8]),
        policy_delay        = trial.suggest_categorical("policy_delay", [1, 2]),
        target_policy_noise = trial.suggest_float("policy_noise", 0.1, 0.5),
        target_noise_clip   = trial.suggest_float("noise_clip", 0.1, 0.5),
        no_trade_penalty    = trial.suggest_float("no_trade_penalty", 0.0, 1.0),
    )
    # pull out penalty so it's not passed to TD3.__init__
    no_trade_penalty = hp.pop("no_trade_penalty")

    # 2) Load data
    data = load_data("data/processed")

    # 3) Env factory
    def make_env(rank: int):
        def _init():
            return EnhancedTradingEnv(
                data,
                initial_balance=3_000,
                max_steps=200,
                debug=False,
                checkpoint_dir=CKPT_DIR,
                no_trade_penalty=no_trade_penalty,
                init_seed=cfg.global_seed + rank,
            )
        return _init

    # 4) Build train & eval as SubprocVecEnv
    train_env = SubprocVecEnv([make_env(i) for i in range(cfg.num_envs)])
    eval_env  = SubprocVecEnv([make_env(i) for i in range(cfg.num_envs)])

    # 5) Instantiate agent
    model = StableTradingTD3(
        "MlpPolicy",
        train_env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
        meta_lr=1e-4,
        adaptation_steps=3,
        verbose=0,
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        **hp
    )

    # 6) Set up callbacks
    callbacks = CallbackList([
        MetaLearningCallback(model.meta_learner),
        EvalCallback(
            eval_env,
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=10_000,
            n_eval_episodes=cfg.n_eval_episodes,
            warn=False,
        ),
        WatchdogNaNCallback(),
        DetailedTensorboardCallback(cfg.tb_log_freq),
        OptunaPruningCallback(trial, eval_env, cfg.pruner_interval_steps),
        SimpleTqdmCallback(cfg.timesteps_per_trial, trial.number, 1_000),
    ])

    # 7) Training with pruning support
    try:
        model.learn(total_timesteps=cfg.timesteps_per_trial, callback=callbacks)
    except TrialPruned:
        trial.set_user_attr("metrics", calculate_trial_metrics(trial, 3_000))
        raise

    # 8) Detailed evaluation logs
    detailed_logs: List[List[Dict[str, Any]]] = []
    for _ in range(cfg.n_eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_log: List[Dict[str, Any]] = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = eval_env.step(action)
            done = term or trunc
            ep_log.append({
                "balance": info["balance"],
                "pnl": info["pnl"],
                "drawdown": info["drawdown"],
                "volatility": eval_env.envs[0].get_volatility_profile(),
                "exit_reasons": eval_env.envs[0].get_trade_exit_reasons(),
                "correlation": eval_env.envs[0].get_current_correlation(),
            })
        detailed_logs.append(ep_log)

    # 9) Compute & attach metrics
    trial.set_user_attr("full_logs", detailed_logs)
    metrics = calculate_trial_metrics(trial, 3_000)
    trial.set_user_attr("metrics", metrics)

    # 10) Composite score
    comp_score = (
        0.4 * metrics["sharpe"]
        + 0.3 * (1.0 - metrics["max_dd"])
        + 0.2 * (metrics["exit_diversity"] / 5.0)
        + 0.1 * metrics["profit_factor"]
        - 0.2 * metrics["correlation"]
    )
    env_logger.info(
        "🎯 TD3 Trial %d  Sharpe=%.3f DD=%.3f PF=%.3f Corr=%.3f Exits=%d Score=%.3f",
        trial.number + 1,
        metrics["sharpe"],
        metrics["max_dd"],
        metrics["profit_factor"],
        metrics["correlation"],
        metrics["exit_diversity"],
        comp_score,
    )
    return comp_score

# ╔═════════════════════════════════════════════════════════════════════╗
# ║                                Main                                ║
# ╚═════════════════════════════════════════════════════════════════════╝
def main():
    set_global_seed(cfg.global_seed)

    pruner = ForexGoldPruner(
        max_dd_thresh=0.30,
        corr_thresh=0.60,
        n_startup_trials=cfg.pruner_startup_trials,
        n_warmup_steps=cfg.pruner_warmup_steps,
        interval_steps=cfg.pruner_interval_steps,
    )

    study = optuna.create_study(
        study_name="td3_trading_optimisation",
        storage="sqlite:///optuna_td3.db",
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.global_seed),
    )

    study.optimize(
        optimise_agent,
        n_trials=cfg.n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )

    best = study.trials[rank_trials(study)[0]]
    best_hp = best.params.copy()
    best_no_trade_penalty = best_hp.pop("no_trade_penalty", 0.3)

    # ── Final training ────────────────────────────────────────────────
    data = load_data("data/processed")

    def make_env_final(rank: int):
        def _init():
            return EnhancedTradingEnv(
                data,
                initial_balance=3_000,
                max_steps=500,
                debug=False,
                checkpoint_dir=CKPT_DIR,
                no_trade_penalty=best_no_trade_penalty,
                init_seed=cfg.global_seed + rank,
            )
        return _init

    use_subproc = cfg.num_envs > 1 and sys.platform != "win32"
    env = DummyVecEnv([make_env_final(i) for i in range(cfg.num_envs)]) if not use_subproc else SubprocVecEnv([make_env_final(i) for i in range(cfg.num_envs)])
    final_model = TD3(
        "MlpPolicy",
        env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs={"net_arch":[256,256], "activation_fn":torch.nn.ReLU},
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR,"tensorboard"),
        **best_hp,
    )

    final_model.learn(
        total_timesteps=cfg.final_training_steps,
        callback=CallbackList([
            DetailedTensorboardCallback(cfg.tb_log_freq),
            HumanLoggerCallback(print_freq=1_000),
            SimpleTqdmCallback(cfg.final_training_steps, -1, 5_000),
        ]),
    )
    model_path = os.path.join(MODEL_DIR, "td3_final_model.zip")
    final_model.save(model_path)
    env_logger.info("✅ TD3 model saved  %s", model_path)

if __name__ == "__main__":
    env_logger.info("Starting TD3 training script")
    main()
    env_logger.info("Finished TD3 training script")
