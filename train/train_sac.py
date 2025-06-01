# Suppress as many warnings as possible before any imports
import os
import warnings

# Deep learning libraries (TensorFlow/Keras/PyTorch) backend spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # Suppress TensorFlow messages
os.environ["KMP_WARNINGS"] = "0"                   # Suppress OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"        # Suppress duplicate lib error
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"          # Suppress TF performance warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # Pretend no CUDA, removes CUDA warnings

# Ignore Python runtime warnings for numpy, pandas, etc.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["GLOG_minloglevel"] = "3"               # Silence glog-based logs from cuDNN/cuBLAS

# now suppress Abseil’s Python-side warnings
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
# prevent that “WARNING: All log messages before absl::InitializeLog()…” banner
absl_logging._warn_preinit_stderr = False

# === Set up file-based logging for ComplianceModule and root warnings ===
# 1) Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# 2) Create a common formatter
import logging
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# 3) ComplianceModule logger → logs/compliance.log (DEBUG+)
compliance_logger = logging.getLogger("ComplianceModule")
compliance_logger.setLevel(logging.DEBUG)
fh_compliance = logging.FileHandler("logs/compliance.log", mode="a", encoding="utf-8")
fh_compliance.setFormatter(formatter)
compliance_logger.addHandler(fh_compliance)

# 4) Root logger → logs/root_warnings.log (WARNING+)
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)
fh_root = logging.FileHandler("logs/root_warnings.log", mode="a", encoding="utf-8")
fh_root.setFormatter(formatter)
root_logger.addHandler(fh_root)
# ========================================================================

# Now do your standard imports
import random
from sys import platform
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import optuna
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy

from tqdm import tqdm

from envs.env import EnhancedTradingEnv
from utils.data_utils import load_data
from utils.meta_learning import AdaptiveMetaLearner, MetaLearningCallback
import logging.handlers
from tensorboardX import SummaryWriter

from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback



@dataclass
class TrainingConfig:
    test_mode: bool = True
    num_envs: int = 1
    global_seed: int = 42

    n_trials: int = 0
    timesteps_per_trial: int = 0
    final_training_steps: int = 0
    pruner_startup_trials: int = 0
    pruner_warmup_steps: int = 0
    pruner_interval_steps: int = 0
    tb_log_freq: int = 0
    n_eval_episodes: int = 0
    meta_eval_freq: int = 0

    def __post_init__(self):
        if self.test_mode:
            self.n_trials = 2
            self.timesteps_per_trial = 2000
            self.final_training_steps = 10000
            self.pruner_startup_trials = 1
            self.pruner_warmup_steps = 500
            self.pruner_interval_steps = 500
            self.tb_log_freq = 1000
            self.n_eval_episodes = 3
            self.meta_eval_freq = 500

        else:
            self.n_trials = 50
            self.timesteps_per_trial = 500_000
            self.final_training_steps = 5_000_000
            self.pruner_startup_trials = 5
            self.pruner_warmup_steps = 50_000
            self.pruner_interval_steps = 100_000
            self.tb_log_freq = 100
            self.n_eval_episodes = 5
            self.meta_eval_freq = 10_000

parser = argparse.ArgumentParser()
parser.add_argument("--prod", action="store_true", help="Run full-budget training.")
cfg = TrainingConfig(test_mode=not parser.parse_args().prod)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_float32_matmul_precision("high")

LOG_DIR = "logs"
MODEL_DIR = "models"
CHECKPOINT_DIR = "checkpoints"
for p in (LOG_DIR, MODEL_DIR, CHECKPOINT_DIR):
    os.makedirs(p, exist_ok=True)

def setup_logger(name: str, file_name: str, level=logging.DEBUG):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, file_name), maxBytes=20 * 1024 * 1024, backupCount=3
    )
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    fh.setLevel(level)
    ch = logging.StreamHandler()
    if hasattr(ch.stream, "reconfigure"):
        ch.stream.reconfigure(encoding="utf-8", errors="replace")
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch.setLevel(logging.DEBUG)  # Changed to DEBUG for more detailed logs
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


env_logger = setup_logger("env_logger", "training.log")
score_logger = setup_logger("score_logger", "training_score.log")

def _ascii(msg: str) -> str:
    return (
        msg.replace("→", "->")
           .replace("►►", ">>")
           .replace("🎯", "[*]")
           .replace("⚡️", "**")
    )

class AsciiFilter(logging.Filter):
    def filter(self, record):
        record.msg = _ascii(record.msg)
        return True

for _lg in (env_logger, score_logger):
    _lg.addFilter(AsciiFilter())

def compute_training_rating(logs: List[Dict[str, Any]], rw=0.4, bw=0.4, dw=0.2):
    if len(logs) < 2:
        return 0.0
    rewards   = np.array([x["reward"]   for x in logs], np.float32)
    balances  = np.array([x["balance"]  for x in logs], np.float32)
    drawdowns = np.array([x["drawdown"] for x in logs], np.float32)
    if rewards.size < 2:
        return 0.0
    stability = 1.0 - rewards.std() / (rewards.mean() + 1e-8)
    growth    = (balances[-1] - balances[0]) / (abs(balances[0]) + 1e-8)
    max_dd    = drawdowns.max()
    return float(np.clip(rw*stability + bw*growth + dw*(1-max_dd), 0, 1)*100)

def log_training_progress(logs: List[Dict[str, Any]], step: int):
    msg = f"[Overall @ Step {step}] Training Score: {compute_training_rating(logs):.2f}/100"
    env_logger.info(msg)
    score_logger.info(msg)


class LiveTqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int, trial_id: int, print_freq: int = 500):
        super().__init__(verbose=0)
        self.total = total_timesteps
        self.trial_id = trial_id
        self.print_freq = print_freq
        self.pbar = None
        self.last_update = 0

    def _on_training_start(self):
        desc = f"Trial {self.trial_id}" if self.trial_id >= 0 else "Final"
        self.pbar = tqdm(total=self.total, desc=desc, ncols=70, unit="step")

    def _on_step(self):
        steps_since = self.num_timesteps - self.last_update
        if steps_since >= self.print_freq or self.num_timesteps == self.total:
            self.pbar.n = self.num_timesteps
            self.pbar.refresh()
            self.last_update = self.num_timesteps
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.n = self.total
            self.pbar.refresh()
            self.pbar.close()


class WatchdogNaNCallback(BaseCallback):
    def _on_step(self) -> bool:
        if (
            np.isnan(self.locals["rewards"]).any()
            or np.isinf(self.locals["rewards"]).any()
        ):
            raise TrialPruned()
        return True

class OptunaPruningCallback(BaseCallback):
    def __init__(self, trial: optuna.trial.Trial, eval_env: DummyVecEnv, eval_freq: int):
        super().__init__(verbose=0)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self._last_eval = 0
    def _on_step(self):
        if (self.num_timesteps - self._last_eval) < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        mean_reward, _ = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=cfg.n_eval_episodes, warn=False
        )
        if not np.isfinite(mean_reward):
            mean_reward = -np.inf
        self.trial.report(mean_reward, self.num_timesteps)
        if self.trial.should_prune():
            raise TrialPruned()
        return True

class HumanLoggerCallback(BaseCallback):
    def __init__(self, print_freq: int = 1_000):
        super().__init__(verbose=0)
        self.print_freq = print_freq
        self.summary_logs: list[dict[str, Any]] = []
    def _on_step(self):
        if self.num_timesteps % self.print_freq == 0:
            metrics = self.training_env.envs[0].get_metrics()
            self.summary_logs.append(
                {
                    "balance": metrics["balance"],
                    "reward": float(self.locals["rewards"][0]),
                    "drawdown": metrics["drawdown"],
                    "volatility": metrics.get("volatility", 0.0),
                    "correlation": metrics.get("correlation", 0.0),
                    "exit_reasons": self.training_env.envs[0].get_trade_exit_reasons(),
                    "step": self.num_timesteps,
                    "reward_mean": float(np.mean(self.locals["rewards"])),
                    "reward_std": float(np.std(self.locals["rewards"])),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "avg_pnl": metrics.get("avg_pnl", 0.0),
                    "max_dd": metrics.get("max_drawdown", 0.0),
                    "sharpe": metrics.get("sharpe_ratio", 0.0),
                    "profit_factor": metrics.get("profit_factor", 0.0),
                    "exit_diversity": len(set(self.training_env.envs[0].get_trade_exit_reasons())),
                    "trial_id": self.model.meta_learner.trial.number if hasattr(self.model, "meta_learner") else -1,
                    "trades_executed": len(self.training_env.envs[0].trades)  # Add trade count
                }
            )
            log_training_progress(self.summary_logs, self.num_timesteps)
        return True

class DetailedTensorboardCallback(BaseCallback):
    def __init__(self, log_freq: int):
        super().__init__(verbose=0)
        self.log_freq = log_freq
        self.writer: SummaryWriter | None = None
    def _call_first(self, fn_name: str, *args, **kw):
        if hasattr(self.training_env, "envs"):
            env0 = self.training_env.envs[0]
            return getattr(env0, fn_name)(*args, **kw)
        return self.training_env.get_attr(fn_name, indices=0)[0](*args, **kw)
    def _on_training_start(self):
        path = getattr(self.model, "tensorboard_log", "logs/tensorboard")
        self.writer = SummaryWriter(log_dir=path)
        env_logger.info("TensorBoard logging started")
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq or self.writer is None:
            return True
        ts = self.num_timesteps
        m  = self._call_first("get_metrics")
        gm = self._call_first("get_genome_metrics")
        self.writer.add_scalar("env/balance",   m["balance"], ts)
        self.writer.add_scalar("env/win_rate",  m["win_rate"], ts)
        self.writer.add_scalar("env/avg_pnl",   m["avg_pnl"], ts)
        self.writer.add_scalar("env/drawdown",  m["drawdown"], ts)

        self.writer.add_scalar("env/trades", len(self.training_env.envs[0].trades), self.num_timesteps)
        self.writer.add_scalar("env/volatility", m.get("volatility", 0.0), ts)
        self.writer.add_scalar("env/correlation", m.get("correlation", 0.0), ts)
        self.writer.add_scalar("env/sharpe_ratio", m.get("sharpe_ratio", 0.0), ts)
        self.writer.add_scalar("env/profit_factor", m.get("profit_factor", 0.0), ts)
        self.writer.add_scalar("env/exit_diversity", len(set(self.training_env.envs[0].get_trade_exit_reasons())), ts)
       
        self.writer.add_scalar("env/trades_executed", len(self.training_env.envs[0].trades), ts)



        for k, v in gm.items():
            self.writer.add_scalar(f"genome/{k}", v, ts)
        return True
    def _on_training_end(self):
        if self.writer:
            self.writer.close()
        env_logger.info("TensorBoard logging closed")

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, env: EnhancedTradingEnv, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.env = env  # Reference to the environment
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            try:
                # Save the model weights (as usual)
                model_path = os.path.join(self.save_path, f"model_step_{self.n_calls}.zip")
                self.model.save(model_path)
                env_logger.info(f"Model checkpoint saved: {model_path}")
                
                # Check if the environment state file exists
                env_state_path = self.env._ckpt("env_state.pkl")
                
                if os.path.isfile(env_state_path):
                    # Load the environment state if it exists
                    self.env._maybe_load_checkpoints()
                    env_logger.info("Environment state loaded.")
                else:
                    # If no state exists, save the current environment state
                    self.env._save_checkpoints()
                    env_logger.info("New environment state saved.")
                    
            except Exception as e:
                env_logger.error(f"Checkpoint save failed: {e}")
                
        return True

class StableTradingSACAgent(SAC):
    def __init__(self, *args, meta_lr=1e-4, adaptation_steps=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_learner = AdaptiveMetaLearner(
            self.policy, meta_lr=meta_lr, adaptation_steps=adaptation_steps
        )

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

def calculate_trial_metrics(trial: optuna.trial.Trial, initial_balance: float) -> Dict[str, float]:
    logs = trial.user_attrs.get("full_logs", [])
    if not logs:
        return dict(sharpe=0.0, max_dd=1.0, profit_factor=0.0, correlation=0.0, exit_diversity=0)
    # flatten and turn into DataFrame
    flat = [step for ep in logs for step in ep]
    df = pd.DataFrame(flat)
    balances = df["balance"].astype(np.float32).values
    if balances.size < 2:
        return dict(sharpe=0.0, max_dd=1.0, profit_factor=0.0, correlation=0.0, exit_diversity=0)
    returns = np.diff(balances) / (balances[:-1] + 1e-8)
    returns = np.nan_to_num(returns)
    # compute raw metrics (these are numpy scalars)
    sharpe_raw = returns.mean() / (returns.std() + 1e-8) * np.sqrt(250)
    peak = np.maximum.accumulate(balances)
    max_dd_raw = np.max((peak - balances) / (peak + 1e-8))
    profit = balances[-1] - initial_balance                   # numpy.float32
    profit_factor_raw = profit / max(1.0, initial_balance - balances.min())
    corr_mean_raw = float(np.abs(df["correlation"].fillna(0.0)).mean())
    exit_diversity_raw = len({r for exits in df["exit_reasons"] for r in exits})

    # cast everything to native Python types
    return {
        "sharpe":        float(sharpe_raw),
        "max_dd":        float(max_dd_raw),
        "profit_factor": float(profit_factor_raw),
        "correlation":   float(corr_mean_raw),
        "exit_diversity": int(exit_diversity_raw),
    }

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

def make_vecenv(builders, n):
    return DummyVecEnv(builders) if n == 1 else SubprocVecEnv(builders)

def optimise_agent(trial: optuna.trial.Trial) -> float:
    env_logger.info("SAC Trial #%d starting", trial.number + 1)
    hp = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256]),
        "buffer_size":   trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
        "tau":           trial.suggest_float("tau", 0.005, 0.02),
        "gamma":         trial.suggest_float("gamma", 0.90, 0.9999),
        "ent_coef":      trial.suggest_float("ent_coef", 0.0, 0.05),
        "target_entropy":"auto",
        "no_trade_penalty": trial.suggest_float("no_trade_penalty", 0.0, 1.0),
    }
    no_trade_penalty = hp.pop("no_trade_penalty")
    data = load_data("data/processed")
    def make_env(rank: int):
        def _init():
            return EnhancedTradingEnv(
                data,
                initial_balance=3_000,
                max_steps=200,
                debug=False,
                checkpoint_dir=CHECKPOINT_DIR,
                no_trade_penalty=no_trade_penalty,
                init_seed=int(cfg.global_seed + rank),
            )
        return _init
    use_subproc = (cfg.num_envs > 1) and (platform != "win32")
    builders = [make_env(i) for i in range(cfg.num_envs)]
    if use_subproc:
        train_env = SubprocVecEnv(builders)
        eval_env  = SubprocVecEnv(builders)
    else:
        train_env = DummyVecEnv(builders)
        eval_env  = DummyVecEnv(builders)
    model = StableTradingSACAgent(
        policy="MlpPolicy",
        env=train_env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
        meta_lr=1e-4,
        adaptation_steps=3,
        verbose=0,
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        **hp,
    )
    callbacks = CallbackList([
        MetaLearningCallback(model.meta_learner),
        EvalCallback(
            eval_env,
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=cfg.meta_eval_freq,
            n_eval_episodes=cfg.n_eval_episodes,
            warn=False,
        ),
        WatchdogNaNCallback(),
        DetailedTensorboardCallback(cfg.tb_log_freq),
        OptunaPruningCallback(trial, eval_env, cfg.pruner_interval_steps),
        LiveTqdmCallback(cfg.timesteps_per_trial, trial.number, print_freq=1),
        CheckpointCallback(save_freq=5000, save_path=CHECKPOINT_DIR, env=train_env),  # Fix here
    ])

    
    try:
        model.learn(total_timesteps=cfg.timesteps_per_trial, callback=callbacks)
    except TrialPruned:
        trial.set_user_attr("metrics", calculate_trial_metrics(trial, 3_000))
        raise
    detailed_logs: List[List[Dict[str, Any]]] = []
    for _ in range(cfg.n_eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_log: List[Dict[str, Any]] = []
        while not done:
            
            action, _ = model.predict(obs, deterministic=True)
            next_obs, _, dones, infos = eval_env.step(action)
            done = dones[0] if isinstance(dones, (list, tuple)) else dones
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            vol_profile = eval_env.get_attr("get_volatility_profile", indices=0)[0]
            exits       = eval_env.get_attr("get_trade_exit_reasons", indices=0)[0]
            corr        = eval_env.get_attr("get_current_correlation", indices=0)[0]
            # Only JSON-serializable entries
            ep_log.append({
                "balance":      float(info["balance"]),
                "pnl":          float(info.get("pnl", 0.0)),
                "drawdown":     float(info["drawdown"]),
                "volatility":   float(vol_profile) if isinstance(vol_profile, (int, float, np.float32, np.float64)) else 0.0,
                "exit_reasons": list(exits) if isinstance(exits, (list, tuple)) else [],
                "correlation":  float(corr) if isinstance(corr, (int, float, np.float32, np.float64)) else 0.0,
                
            })
            
            obs = next_obs
        detailed_logs.append(ep_log)
    # Only JSON-serializable content!
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
        "SAC Trial #%d -> Sharpe=%.3f DD=%.3f PF=%.3f Corr=%.3f Exits=%d Score=%.3f",
        trial.number + 1,
        metrics["sharpe"],
        metrics["max_dd"],
        metrics["profit_factor"],
        metrics["correlation"],
        metrics["exit_diversity"],
        comp_score,
    )
    return comp_score

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
        study_name="sac_trading_optimisation",
        storage="sqlite:///optuna_sac.db",
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
    data = load_data("data/processed")
    def make_env_final(rank: int):
        def _init():
            return EnhancedTradingEnv(
                data,
                initial_balance=3_000,
                max_steps=500,
                debug=False,
                checkpoint_dir=CHECKPOINT_DIR,
                no_trade_penalty=best_no_trade_penalty,
                init_seed=cfg.global_seed + rank,
            )
        return _init
    use_subproc = cfg.num_envs > 1 and platform != "win32"
    env = make_vecenv([make_env_final(i) for i in range(cfg.num_envs)],
                      cfg.num_envs if use_subproc else 1)
    final_model = SAC(
        "MlpPolicy",
        env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        **best_hp,
    )
    final_model.learn(
        total_timesteps=cfg.final_training_steps,
        callback=CallbackList(
            [
                DetailedTensorboardCallback(cfg.tb_log_freq),
                HumanLoggerCallback(print_freq=1_000),
                LiveTqdmCallback(cfg.final_training_steps, 0, print_freq=10),


                CheckpointCallback(
                    save_freq=5000, 
                    save_path=CHECKPOINT_DIR, 
                    env=env,  # Pass the environment to the callback
                    verbose=0
                )

            ]
        ),
    )
    model_path = os.path.join(MODEL_DIR, "sac_final_model.zip")
    final_model.save(model_path)
    env_logger.info("SAC model saved -> %s", model_path)

if __name__ == "__main__":
    env_logger.info("Starting SAC training script")
    main()
    env_logger.info("Finished SAC training script")
