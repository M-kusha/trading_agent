#!/usr/bin/env python3
"""
algorithms.py – agent wrappers, hyperparameter spaces, and Optuna objective builder
"""
import os
import torch
import optuna
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback
from utils.meta_learning import AdaptiveMetaLearner, MetaLearningCallback
from envs.ppo_env import EnhancedTradingEnv
from common import (
    NUM_ENVS,
    N_EVAL_EPISODES,
    LOG_DIR,
    GLOBAL_SEED,
    TIMESTEPS_PER_TRIAL,
    env_logger,
    ForexGoldPruner,
    calculate_trial_metrics,
    OptunaPruningCallback,
    HumanLoggerCallback,
    DetailedTensorboardCallback,
)

# ──────────────── Agent wrappers ─────────────────────────────────────────────

class StableTradingPPOAgent(PPO):
    """PPO + meta‑learner wrapper."""
    def __init__(self, *args, meta_lr: float = 1e-4, adaptation_steps: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_learner = AdaptiveMetaLearner(self.policy, meta_lr=meta_lr, adaptation_steps=adaptation_steps)

class StableTradingSACAgent(SAC):
    """SAC + meta‑learner wrapper."""
    def __init__(self, *args, meta_lr: float = 1e-4, adaptation_steps: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_learner = AdaptiveMetaLearner(self.policy, meta_lr=meta_lr, adaptation_steps=adaptation_steps)

class StableTradingTD3Agent(TD3):
    """TD3 + meta‑learner wrapper."""
    def __init__(self, *args, meta_lr: float = 1e-4, adaptation_steps: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_learner = AdaptiveMetaLearner(self.policy, meta_lr=meta_lr, adaptation_steps=adaptation_steps)

# ──────────────── Hyperparameter spaces ──────────────────────────────────────

def _ppo_hp(trial: optuna.trial.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps":       trial.suggest_categorical("n_steps", [2048, 4096, 8192]),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256]),
        "gamma":         trial.suggest_float("gamma", 0.90, 0.9999),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.8, 1.0),
        "clip_range":    trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef":      trial.suggest_float("ent_coef", 0.0, 0.05),
        "no_trade_penalty": trial.suggest_float("no_trade_penalty", 0.0, 1.0),
    }

def _sac_hp(trial: optuna.trial.Trial) -> dict:
    return {
        "learning_rate":   trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size":      trial.suggest_categorical("batch_size", [64, 128, 256]),
        "buffer_size":     trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
        "tau":             trial.suggest_float("tau", 0.005, 0.02),
        "gamma":           trial.suggest_float("gamma", 0.90, 0.9999),
        "ent_coef":        trial.suggest_float("ent_coef", 0.0, 0.05),
        "no_trade_penalty": trial.suggest_float("no_trade_penalty", 0.0, 1.0),
    }

def _td3_hp(trial: optuna.trial.Trial) -> dict:
    return {
        "learning_rate":   trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size":      trial.suggest_categorical("batch_size", [64, 128, 256]),
        "buffer_size":     trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
        "tau":             trial.suggest_float("tau", 0.005, 0.02),
        "gamma":           trial.suggest_float("gamma", 0.90, 0.9999),
        "train_freq":      trial.suggest_categorical("train_freq", [1, 2, 4, 8]),
        "gradient_steps":  trial.suggest_categorical("gradient_steps", [1, 2, 4, 8]),
        "policy_delay":    trial.suggest_categorical("policy_delay", [1, 2]),
        "noise_clip":      trial.suggest_float("noise_clip", 0.1, 0.5),
        "no_trade_penalty": trial.suggest_float("no_trade_penalty", 0.0, 1.0),
    }

ALG_CONFIG = {
    "ppo": {"class": StableTradingPPOAgent, "hp_space": _ppo_hp},
    "sac": {"class": StableTradingSACAgent, "hp_space": _sac_hp},
    "td3": {"class": StableTradingTD3Agent, "hp_space": _td3_hp},
}

# ──────────────── Single Optuna objective ──────────────────────────────────

def make_objective(algo_name: str, data) -> callable:
    AlgoCls = ALG_CONFIG[algo_name]["class"]
    hp_space = ALG_CONFIG[algo_name]["hp_space"]

    def objective(trial: optuna.trial.Trial) -> float:
        print(f"\n🚀  Starting {algo_name.upper()} Trial #{trial.number + 1} …")
        env_logger.info(f"→ {algo_name.upper()} Trial #{trial.number + 1} starting")

        # initialize metrics for pruner
        trial.set_user_attr("metrics", {"sharpe":0.0, "max_dd":1.0, "profit_factor":0.0, "correlation":0.0, "exit_diversity":0})

        # 1) Hyper‑param sampling
        hp = hp_space(trial)
        no_trade_penalty = hp.pop("no_trade_penalty")

        # 2) Environment factories
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

        train_env = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])
        eval_env = DummyVecEnv([make_env(i + NUM_ENVS) for i in range(NUM_ENVS)])

        # 3) Device
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # 4) Model
        model = AlgoCls(
            "MlpPolicy",
            train_env,
            device=DEVICE,
            policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
            meta_lr=1e-4,
            adaptation_steps=3,
            **hp,
            verbose=0,
            tensorboard_log=LOG_DIR,
        )

        # 5) Callbacks
        prune_cb = OptunaPruningCallback(
            trial,
            eval_env,
            eval_freq=globals().get("PRUNER_INTERVAL_STEPS", 0),
            n_eval_episodes=N_EVAL_EPISODES,
        )
        cb_list = [prune_cb, ProgressBarCallback(), HumanLoggerCallback(print_freq=globals().get("PRUNER_INTERVAL_STEPS", 0))]

        # 6) Train & prune
        try:
            model.learn(total_timesteps=globals().get("TIMESTEPS_PER_TRIAL", 0), callback=cb_list)
        except optuna.exceptions.TrialPruned:
            metrics = calculate_trial_metrics(trial, 3_000)
            trial.set_user_attr("metrics", metrics)
            raise

        # 7) Post‑training evaluation
        detailed_logs = []
        for _ in range(N_EVAL_EPISODES):
            obs, _ = eval_env.reset()
            done = False
            ep_log = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(action)
                done = term or trunc
                ep_log.append({
                    "balance": info["balance"],
                    "pnl": info.get("pnl", 0.0),
                    "drawdown": info["drawdown"],
                    "volatility": eval_env.envs[0].get_volatility_profile(),
                    "exit_reasons": eval_env.envs[0].get_trade_exit_reasons(),
                    "correlation": eval_env.envs[0].get_current_correlation(),
                })
            detailed_logs.append(ep_log)

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
            "🎯  %s Trial #%d → Sharpe=%.3f DD=%.3f PF=%.3f Corr=%.3f Exits=%d Score=%.3f",
            algo_name.upper(),
            trial.number + 1,
            metrics["sharpe"],
            metrics["max_dd"],
            metrics["profit_factor"],
            metrics["correlation"],
            metrics["exit_diversity"],
            comp_score,
        )

        return comp_score

    return objective
