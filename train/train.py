#!/usr/bin/env python3
"""
train_trading.py – entrypoint for training PPO, SAC, or TD3 with shared logic
"""
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 1) Fix Python hash seed
os.environ["PYTHONHASHSEED"] = "42"

import warnings
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

import json
import torch
import optuna
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback
from common import (
    TEST_MODE,
    N_TRIALS,
    FINAL_TRAINING_STEPS,
    PRUNER_STARTUP_TRIALS,
    PRUNER_WARMUP_STEPS,
    PRUNER_INTERVAL_STEPS,
    TB_LOG_FREQ,
    NUM_ENVS,
    LOG_DIR,
    DetailedTensorboardCallback,
    HumanLoggerCallback,
    env_logger,
    rank_trials,
    ForexGoldPruner,
)
from algorithms import make_objective, ALG_CONFIG
from utils.data_utils import load_data
from envs.ppo_env import EnhancedTradingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=list(ALG_CONFIG.keys()),
        required=True,
        help="Algorithm to train: ppo, sac, or td3",
    )
    args = parser.parse_args()

    # Load data once
    data = load_data("data/processed")

    # Optuna study setup
    study = optuna.create_study(
        study_name=f"{args.algo}_trading_optimization",
        storage=f"sqlite:///optuna_{args.algo}.db",
        direction="maximize",
        pruner=ForexGoldPruner(
            max_dd_thresh=0.30,
            corr_thresh=0.60,
            n_startup_trials=PRUNER_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_WARMUP_STEPS,
            interval_steps=PRUNER_INTERVAL_STEPS,
        ),
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Run optimization
    study.optimize(
        make_objective(args.algo, data),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    # Gather best trial
    ranked = rank_trials(study)
    if not ranked:
        raise RuntimeError("No successful trials – check logs for errors/pruning causes.")
    best_trial = study.trials[ranked[0]]
    best_params = best_trial.params.copy()

    # Cache best hyperparameters
    os.makedirs("models", exist_ok=True)
    with open(f"models/best_params_{args.algo}.json", "w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=4)

    # Final training with parallel envs & GPU auto-detect
    no_trade_penalty = best_params.pop("no_trade_penalty")

    def make_env_final(rank: int):
        def _init():
            return EnhancedTradingEnv(
                data,
                initial_balance=3_000,
                max_steps=500,
                debug=False,
                checkpoint_dir="checkpoints",
                no_trade_penalty=no_trade_penalty,
                init_seed=42 + rank,
            )
        return _init

    env = DummyVecEnv([make_env_final(i) for i in range(NUM_ENVS)])
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    env_logger.info(f"Using device for final training: {DEVICE}")

    # Instantiate and train the final model
    AlgoCls = ALG_CONFIG[args.algo]["class"]
    final_model = AlgoCls(
        "MlpPolicy",
        env,
        device=DEVICE,
        policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
        meta_lr=1e-4,
        adaptation_steps=3,
        **best_params,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    final_model.learn(
        total_timesteps=FINAL_TRAINING_STEPS,
        callback=[
            DetailedTensorboardCallback(log_freq=TB_LOG_FREQ),
            HumanLoggerCallback(print_freq=20),
            ProgressBarCallback(),
        ],
    )

    # Save
    final_model.save(f"models/{args.algo}_final_model.zip")
    env_logger.info(f"✅ {args.algo.upper()} model saved → models/{args.algo}_final_model.zip")


if __name__ == "__main__":
    # prevent PyTorch from hogging all CPU threads
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    main()
