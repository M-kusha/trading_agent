#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack

from envs.prop_firm_env import PropFirmConfig, PropFirmTradingEnv

try:
    from envs.curriculum import (
        CurriculumManager,
        CurriculumStage,
        CurriculumStageConfig,
        get_stage_config,
        get_stage_progression,
    )
    CURRICULUM_AVAILABLE = True
except ImportError:
    CURRICULUM_AVAILABLE = False
    CurriculumStage = None  # type: ignore
    CurriculumStageConfig = None  # type: ignore
    CurriculumManager = None  # type: ignore
    get_stage_config = None  # type: ignore
    get_stage_progression = None  # type: ignore


try:
    import optuna
    from optuna.exceptions import TrialPruned
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None  # type: ignore
    MedianPruner = None  # type: ignore
    TPESampler = None  # type: ignore
    TrialPruned = Exception  # type: ignore
    OPTUNA_AVAILABLE = False


try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    MASKABLE_AVAILABLE = True
except Exception:
    MaskablePPO = None  # type: ignore
    ActionMasker = None  # type: ignore
    MASKABLE_AVAILABLE = False

try:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    MASKABLE_EVAL_AVAILABLE = True
except Exception:
    MaskableEvalCallback = None  # type: ignore
    MASKABLE_EVAL_AVAILABLE = False
    MASKABLE_AVAILABLE = False


try:
    from sb3_contrib.common.maskable.utils import get_action_masks as sb3_get_action_masks
    SB3_MASK_UTILS_AVAILABLE = True
except Exception:
    sb3_get_action_masks = None  # type: ignore
    SB3_MASK_UTILS_AVAILABLE = False


try:
    from dashboard.server import WEB_AVAILABLE as DASHBOARD_AVAILABLE
    from dashboard.server import start_dashboard_server
except ImportError:
    DASHBOARD_AVAILABLE = False
    start_dashboard_server = None  # type: ignore


from modules.utils import simulation_time as simclock
from train.callbacks import CurriculumCheckpointCallback, CurriculumTrainingCallback, VecEpisodeTradingCallback


def configure_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("propfirm_train")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(level)

    fh = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


logger = configure_logging()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)


    simclock.set_mode(simclock.TimeMode.SIMULATION)
    simclock.reset()
    logger.info("[CLOCK] simulation mode enabled - modules read bar time, not wall clock")


def pick_subproc_start_method() -> Optional[str]:
    sysname = platform.system()
    cuda = torch.cuda.is_available()

    if sysname == "Windows":
        return None
    if sysname == "Darwin":
        return "spawn"
    if cuda:
        return "spawn"
    return "fork"


def _safe_propfirm_config_init_kwargs(overrides: Dict[str, Any]) -> Dict[str, Any]:
    try:
        valid = {f.name for f in fields(PropFirmConfig)}
        return {k: v for k, v in overrides.items() if k in valid}
    except Exception:


        allow = {
            "initial_balance",
            "daily_drawdown_limit",
            "max_drawdown_limit",
            "max_steps_per_episode",
            "entry_quality_threshold",
            "entry_quality_gate_enabled",
            "max_trades_per_day",
            "max_trades_per_session",
            "domain_randomization_enabled",
            "spread_mult_range",
            "slippage_mult_range",
            "latency_bars_range",
            "volatility_scale_range",
        }
        return {k: v for k, v in overrides.items() if k in allow}


def build_propfirm_config(config_overrides: Dict[str, Any]) -> PropFirmConfig:
    init_kwargs = _safe_propfirm_config_init_kwargs(config_overrides)
    cfg = PropFirmConfig(**init_kwargs)


    if "reward_scale" in config_overrides:
        try:
            cfg.reward.reward_scale = float(config_overrides["reward_scale"])
        except Exception:
            pass
    if "risk_penalty_scale" in config_overrides:
        try:
            cfg.reward.dd_penalty_scale = float(config_overrides["risk_penalty_scale"])
        except Exception:
            pass

    return cfg


TIMEFRAME_MIN_BARS = {
    "M1": 10000,
    "M5": 8000,
    "M15": 5000,
    "M30": 3000,
    "H1": 2000,
    "H2": 1000,
    "H4": 500,
    "H8": 300,
    "D1": 200,
    "W1": 50,
}


def _normalize_market_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    require_spread: bool,
) -> pd.DataFrame:
    """Validate the executable market-data contract without repairing prices.

    Engineered feature columns may legitimately be sparse near indicator warmup
    boundaries, but the timestamp and OHLCV(+spread) substrate may not be.  A
    previous loader silently turned malformed numeric values into zero prices
    and continued after per-file failures, which made a run's actual dataset
    depend on which files happened to parse.
    """
    df = frame.copy()
    df.columns = df.columns.str.lower()

    if "time" not in df.columns:
        alias = next((c for c in ("timestamp", "datetime", "date") if c in df.columns), None)
        if alias is None:
            raise ValueError(f"{source}: missing timestamp column")
        df = df.rename(columns={alias: "time"})

    required = {"time", "open", "high", "low", "close", "volume"}
    if require_spread:
        required.add("spread")
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}")

    times = pd.to_datetime(df["time"], errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError(f"{source}: invalid timestamps")
    if times.duplicated().any():
        sample = times.loc[times.duplicated(keep=False)].iloc[0]
        raise ValueError(f"{source}: duplicate timestamp {sample}")
    if not times.is_monotonic_increasing:
        raise ValueError(f"{source}: timestamps are not strictly increasing")
    df["time"] = times

    numeric = ["open", "high", "low", "close", "volume"]
    if "spread" in df.columns:
        numeric.append("spread")
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce")
        array = values.to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(array).all():
            raise ValueError(f"{source}: {col} contains non-finite values")
        if col in ("open", "high", "low", "close") and (array <= 0.0).any():
            raise ValueError(f"{source}: {col} contains non-positive values")
        if col in ("volume", "spread") and (array < 0.0).any():
            raise ValueError(f"{source}: {col} contains negative values")
        df[col] = values.astype(np.float32)

    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    if (high < np.maximum(open_, close)).any() or (low > np.minimum(open_, close)).any():
        raise ValueError(f"{source}: OHLC envelope is internally inconsistent")
    if (high < low).any():
        raise ValueError(f"{source}: high is below low")
    return df.reset_index(drop=True)

def load_market_data(
    data_dir: str = "data/processed",
    instruments: Optional[List[str]] = None,
    min_bars: int = 5000,
    extra_dir: Optional[str] = None,
    allow_synthetic: bool = False,
    data_cutoff: Optional[Any] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    if instruments is None:
        instruments = ["XAUUSD"]

    data: Dict[str, Dict[str, pd.DataFrame]] = {}

    if not os.path.exists(data_dir):
        if allow_synthetic:
            logger.warning(f"Data directory not found: {data_dir}. Using explicitly enabled synthetic data.")
            return _create_synthetic_data(instruments, min_bars)
        raise FileNotFoundError(f"market data directory not found: {data_dir}")

    tf_candidates = {"M1", "M5", "M15", "M30", "H1", "H2", "H4", "H8", "D1", "W1"}

    for file in os.listdir(data_dir):
        if not file.endswith(".csv"):
            continue

        filepath = os.path.join(data_dir, file)
        try:
            base = file[:-4].replace("_features", "")
            parts = base.split("_")

            if len(parts) >= 2 and parts[-1].upper() in tf_candidates:
                timeframe = parts[-1].upper()
                instrument = "_".join(parts[:-1])
            else:
                instrument = "_".join(parts)
                timeframe = "M15"

            if instruments and instrument not in instruments:
                continue

            df = _normalize_market_frame(
                pd.read_csv(filepath),
                source=filepath,
                require_spread=False,
            )


            tf_min_bars = TIMEFRAME_MIN_BARS.get(timeframe, min_bars)
            if len(df) < tf_min_bars:
                raise ValueError(
                    f"{filepath}: only {len(df)} bars (need {tf_min_bars}+ for {timeframe})"
                )

            data.setdefault(instrument, {})[timeframe] = df
            logger.info(f"Loaded {instrument}/{timeframe}: {len(df):,} bars")

        except Exception as e:
            raise ValueError(f"failed to load market data file {filepath}: {e}") from e

    if not data:
        if allow_synthetic:
            logger.warning("No file data loaded; generating explicitly enabled synthetic data")
            return _create_synthetic_data(instruments, min_bars)
        raise RuntimeError(f"no valid market data loaded from {data_dir}")

    data = append_broker_bars(data, extra_dir=extra_dir)
    if data_cutoff is not None:
        cutoff = pd.Timestamp(data_cutoff)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        for instrument, frames in data.items():
            for timeframe, frame in list(frames.items()):
                times = pd.to_datetime(frame["time"], errors="coerce", utc=True)
                trimmed = frame.loc[times <= cutoff].copy().reset_index(drop=True)
                minimum = TIMEFRAME_MIN_BARS.get(timeframe, min_bars)
                if len(trimmed) < minimum:
                    raise ValueError(
                        f"data cutoff {cutoff.isoformat()} leaves {instrument}/{timeframe} "
                        f"with {len(trimmed)} bars (need {minimum})"
                    )
                trimmed.attrs["data_cutoff"] = cutoff.isoformat()
                frames[timeframe] = trimmed

    total_bars = sum(len(df) for tfs in data.values() for df in tfs.values())
    logger.info(f"Total: {len(data)} instruments, {total_bars:,} bars")
    return data


DATASET_MANIFEST_VERSION = 1
OPTUNA_OBJECTIVE_VERSION = "masked-ppo-walk-forward-v2"


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _training_code_fingerprint() -> str:
    """Hash the code that defines training, observations, rewards, and gates."""
    paths = [Path(__file__).resolve(), REPO_ROOT / "modules/meta/ppo_observation_builder.py"]
    paths.extend(sorted((REPO_ROOT / "envs").rglob("*.py")))
    paths.extend(sorted((REPO_ROOT / "train/callbacks").rglob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda p: p.as_posix()):
        if not path.is_file():
            continue
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_dataset_manifest(
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    role: str,
    boundary: Optional[Any] = None,
) -> Dict[str, Any]:
    """Fingerprint exactly the executable columns held in memory for a run."""
    frames: List[Dict[str, Any]] = []
    overall = hashlib.sha256()
    for instrument in sorted(data):
        for timeframe in sorted(data[instrument]):
            frame = data[instrument][timeframe]
            if frame is None or frame.empty:
                raise ValueError(f"cannot fingerprint empty frame {instrument}/{timeframe}")
            cols = [c for c in ("time", "open", "high", "low", "close", "volume", "spread") if c in frame]
            required = {"time", "open", "high", "low", "close", "volume"}
            if not required.issubset(cols):
                raise ValueError(f"cannot fingerprint incomplete frame {instrument}/{timeframe}")
            normalized = frame.loc[:, cols].copy()
            normalized["time"] = pd.to_datetime(normalized["time"], errors="raise", utc=True)
            row_hashes = pd.util.hash_pandas_object(normalized, index=False, categorize=False)
            frame_hash = hashlib.sha256(row_hashes.to_numpy(dtype=np.uint64).tobytes()).hexdigest()
            first = normalized["time"].iloc[0]
            last = normalized["time"].iloc[-1]
            record = {
                "instrument": instrument,
                "timeframe": timeframe,
                "rows": int(len(normalized)),
                "first_time": first.isoformat(),
                "last_time": last.isoformat(),
                "columns": cols,
                "content_sha256": frame_hash,
            }
            frames.append(record)
            overall.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    boundary_text = None
    if boundary is not None:
        ts = pd.Timestamp(boundary)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        boundary_text = ts.isoformat()
    return {
        "manifest_version": DATASET_MANIFEST_VERSION,
        "role": str(role),
        "dataset_fingerprint": overall.hexdigest(),
        "boundary": boundary_text,
        "git_head": _git_head(),
        "training_code_sha256": _training_code_fingerprint(),
        "frames": frames,
    }


def write_dataset_manifest(manifest: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


CURRICULUM_PROVENANCE_VERSION = 1
TERMINAL_CURRICULUM_STAGE = "LIVE_READY"


def _normalized_current_stage_weight(stage_config: Any) -> float:
    """Return only the declared probability mass assigned to the current stage.

    Recent/foundation branches can sometimes fall back to the current stage, but
    crediting that incidental behaviour would make a safety-cap preflight
    optimistic.  The budget therefore uses the explicit current-stage share.
    """
    sampling = stage_config.mixed_stage_sampling
    if not bool(getattr(sampling, "enabled", False)):
        return 1.0

    current = float(getattr(sampling, "current_stage_weight", 0.0))
    recent = float(getattr(sampling, "recent_stages_weight", 0.0))
    foundation = float(getattr(sampling, "foundation_weight", 0.0))
    weights = (current, recent, foundation)
    if not all(np.isfinite(weight) and weight >= 0.0 for weight in weights):
        raise ValueError(f"invalid mixed-stage weights for {stage_config.stage.name}: {weights}")
    total = sum(weights)
    if total <= 0.0:
        # This matches CurriculumManager.sample_training_stage(), which falls
        # back to the current stage when the declared total is zero.
        return 1.0
    return current / total


def build_curriculum_budget_preflight(
    *,
    start_stage: Any,
    safety_cap_timesteps: int,
    already_consumed_timesteps: int = 0,
    current_stage_timesteps: int = 0,
) -> Dict[str, Any]:
    """Build the necessary curriculum budget from ``start_stage`` to LIVE_READY.

    Promotion thresholds count timesteps recorded to the current stage, not
    global PPO timesteps.  When mixed-stage sampling is enabled, the declared
    current-stage probability is therefore used to inflate each stage-local
    floor.  This remains a necessary planning floor rather than a sufficiency
    guarantee: demotions, failed gates, recovery and review sessions can all
    require additional training.
    """
    if not CURRICULUM_AVAILABLE or CurriculumStage is None:
        raise RuntimeError("Curriculum system not available. Check imports.")
    if get_stage_progression is None or get_stage_config is None:
        raise RuntimeError("Curriculum stage configuration is not available")

    try:
        stage = start_stage if isinstance(start_stage, CurriculumStage) else CurriculumStage[str(start_stage)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unknown curriculum start stage: {start_stage!r}") from exc

    safety_cap = int(safety_cap_timesteps)
    consumed = int(already_consumed_timesteps)
    current_progress = int(current_stage_timesteps)
    if safety_cap < 0 or consumed < 0 or current_progress < 0:
        raise ValueError("curriculum timestep budgets cannot be negative")

    progression = list(get_stage_progression())
    try:
        start_index = progression.index(stage)
        terminal_index = next(i for i, candidate in enumerate(progression) if candidate.name == TERMINAL_CURRICULUM_STAGE)
    except (ValueError, StopIteration) as exc:
        raise RuntimeError("curriculum progression does not contain LIVE_READY") from exc
    if start_index > terminal_index:
        raise ValueError(f"start stage {stage.name} is after {TERMINAL_CURRICULUM_STAGE}")

    entries: List[Dict[str, Any]] = []
    minimum_remaining = 0
    for index, candidate in enumerate(progression[start_index : terminal_index + 1]):
        config = get_stage_config(candidate)
        local_minimum = int(config.competence.min_timesteps)
        local_recorded = min(current_progress, local_minimum) if index == 0 else 0
        local_remaining = max(0, local_minimum - local_recorded)
        current_probability = _normalized_current_stage_weight(config)
        if current_probability <= 0.0 and local_remaining > 0:
            global_required: Optional[int] = None
        else:
            global_required = (
                0
                if local_remaining == 0
                else int(math.ceil(local_remaining / max(current_probability, np.finfo(float).tiny)))
            )
            minimum_remaining += global_required
        entries.append(
            {
                "stage": candidate.name,
                "local_min_timesteps": local_minimum,
                "local_timesteps_already_recorded": local_recorded,
                "local_timesteps_remaining": local_remaining,
                "declared_current_stage_probability": current_probability,
                "global_timesteps_required": global_required,
            }
        )

    available = max(0, safety_cap - consumed)
    remaining_budget = available
    maximum_stage = stage
    terminal_mastery_possible = True
    impossible_stage: Optional[str] = None
    for index, entry in enumerate(entries):
        required = entry["global_timesteps_required"]
        if required is None or remaining_budget < required:
            terminal_mastery_possible = False
            impossible_stage = str(entry["stage"])
            break
        remaining_budget -= int(required)
        if index + 1 < len(entries):
            maximum_stage = progression[start_index + index + 1]

    shortfall = max(0, minimum_remaining - available)
    return {
        "budget_model": "stage_local_min_timesteps_divided_by_declared_current_stage_probability",
        "necessary_not_sufficient": True,
        "start_stage": stage.name,
        "terminal_stage": TERMINAL_CURRICULUM_STAGE,
        "safety_cap_timesteps": safety_cap,
        "already_consumed_timesteps": consumed,
        "available_timesteps": available,
        "minimum_remaining_timesteps": minimum_remaining,
        "minimum_total_target_timesteps": consumed + minimum_remaining,
        "shortfall_timesteps": shortfall,
        "maximum_theoretical_stage": maximum_stage.name,
        "terminal_mastery_possible": bool(terminal_mastery_possible and impossible_stage is None),
        "first_unfunded_stage": impossible_stage,
        "stages": entries,
        "excluded_overhead": [
            "failed_promotion_gates",
            "demotions",
            "recovery_protocols",
            "review_sessions",
        ],
    }


def enforce_goal_based_budget(preflight: Dict[str, Any], *, goal_based: bool) -> None:
    """Fail before learning when a goal-based cap cannot fund the stage floors."""
    if not goal_based or bool(preflight.get("terminal_mastery_possible", False)):
        return
    raise ValueError(
        "--goal-based safety cap cannot reach LIVE_READY under the declared curriculum budget: "
        f"available={int(preflight['available_timesteps']):,}, "
        f"minimum_remaining={int(preflight['minimum_remaining_timesteps']):,}, "
        f"shortfall={int(preflight['shortfall_timesteps']):,}, "
        f"first_unfunded_stage={preflight.get('first_unfunded_stage')}. "
        "This is only a necessary floor; gates, demotions and reviews can require more."
    )


def _manifest_provenance(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": manifest.get("role"),
        "fingerprint": manifest.get("dataset_fingerprint"),
        "boundary": manifest.get("boundary"),
        "git_head": manifest.get("git_head"),
        "training_code_sha256": manifest.get("training_code_sha256"),
        "frames": list(manifest.get("frames", [])),
    }


def current_observation_schema(model: Optional[BaseAlgorithm] = None, *, frame_stack: int = 1) -> Dict[str, Any]:
    """Describe the runtime observation contract without relying on a checkpoint tag."""
    from modules.meta.ppo_observation_builder import (
        PPO_OBS_FEATURE_NAMES,
        PPO_OBS_SIZE,
        PPO_OBS_VERSION,
    )

    names = [str(name) for name in PPO_OBS_FEATURE_NAMES]
    schema: Dict[str, Any] = {
        "version": str(PPO_OBS_VERSION),
        "base_width": int(PPO_OBS_SIZE),
        "feature_names": names,
        "feature_names_sha256": hashlib.sha256(
            json.dumps(names, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "frame_stack": int(frame_stack),
    }
    if model is not None:
        shape = getattr(getattr(model, "observation_space", None), "shape", None)
        schema["model_observation_shape"] = list(shape) if shape is not None else None
        schema["checkpoint_obs_version"] = str(getattr(model, "_obs_version", PPO_OBS_VERSION))
    return schema


def assess_terminal_mastery(
    curriculum_manager: Any,
    *,
    mastery_confirmation_episodes: int,
) -> Dict[str, Any]:
    """Require terminal floors plus explicit development gate evidence.

    Entering the terminal enum is not mastery.  The final stage must accumulate
    its own competence floors, and the transition into it must be backed by
    recorded, passing development validation and stress evaluations.  Missing
    histories fail closed; an internal/skipped gate cannot be reconstructed as
    external evidence after the run.
    """
    if get_stage_config is None or CurriculumStage is None:
        raise RuntimeError("Curriculum stage configuration is not available")
    config = get_stage_config(CurriculumStage.LIVE_READY)
    final_stage = curriculum_manager.current_stage.name
    terminal_reached = bool(getattr(config, "is_terminal", False)) and final_stage == TERMINAL_CURRICULUM_STAGE
    required_episodes = max(
        int(getattr(config.competence, "min_episodes", 0) or 0),
        int(mastery_confirmation_episodes),
    )
    required_timesteps = int(getattr(config.competence, "min_timesteps", 0) or 0)
    actual_episodes = int(getattr(curriculum_manager, "stage_episodes", 0) or 0)
    actual_timesteps = int(getattr(curriculum_manager, "stage_timesteps", 0) or 0)

    report = curriculum_manager.get_progress_report()
    terminal_transitions = [
        record
        for record in list(report.get("recent_transitions", []) or [])
        if record.get("type") == "promotion"
        and record.get("from_stage") == "PROFESSIONAL"
        and record.get("to_stage") == TERMINAL_CURRICULUM_STAGE
    ]
    terminal_transition = terminal_transitions[-1] if terminal_transitions else None

    # The PROFESSIONAL records justify entry into LIVE_READY.  They cannot also
    # prove that the policy remained valid after another 1.6m terminal-stage
    # timesteps.  Terminal mastery therefore requires a fresh LIVE_READY gate
    # produced by the non-transitioning readiness path.
    validation_history = [
        record
        for record in list(report.get("validation_gate_history", []) or [])
        if str(record.get("stage")) == TERMINAL_CURRICULUM_STAGE
    ]
    validation_record = validation_history[-1] if validation_history else None

    validation_passed = bool(
        validation_record
        and validation_record.get("passed") is True
        and int(validation_record.get("scenarios_total", 0) or 0) > 0
    )
    # The final validation suite contains the named wide-spread, high-slippage,
    # high-volatility and aggregate stress scenarios.  At PROFESSIONAL and
    # LIVE_READY the checker requires every declared scenario to pass, so this
    # is executable stress evidence.  A separate `stress_test` stage config was
    # never defined; demanding its empty history made mastery unreachable.
    stress_passed = bool(
        validation_record
        and validation_passed
        and int(validation_record.get("scenarios_passed", 0) or 0)
        == int(validation_record.get("scenarios_total", 0) or 0)
    )
    checks = {
        "terminal_stage_reached": terminal_reached,
        "professional_to_live_ready_promotion": {
            "passed": terminal_transition is not None,
            "record": terminal_transition,
        },
        "terminal_min_episodes": {
            "actual": actual_episodes,
            "required": required_episodes,
            "passed": actual_episodes >= required_episodes,
        },
        "terminal_min_timesteps": {
            "actual": actual_timesteps,
            "required": required_timesteps,
            "passed": actual_timesteps >= required_timesteps,
        },
        "development_validation_gate": {
            "passed": validation_passed,
            "record": validation_record,
        },
        "development_stress_gate": {
            "passed": stress_passed,
            "source": "all LIVE_READY validation scenarios passed",
            "record": validation_record,
        },
    }
    confirmed = bool(
        terminal_reached
        and terminal_transition is not None
        and checks["terminal_min_episodes"]["passed"]
        and checks["terminal_min_timesteps"]["passed"]
        and validation_passed
        and stress_passed
    )
    return {
        "confirmed": confirmed,
        "checks": checks,
        "evidence_policy": (
            "LIVE_READY plus terminal episode/timestep floors and passing recorded "
            "LIVE_READY validation including every declared stress scenario after "
            "the terminal training floor; "
            "the PROFESSIONAL promotion record separately proves the entry gate"
        ),
    }


def build_curriculum_run_provenance(
    *,
    loaded_manifest: Dict[str, Any],
    train_manifest: Dict[str, Any],
    holdout_manifest: Dict[str, Any],
    final_stage: str,
    run_outcome: str,
    actual_timesteps: int,
    start_stage: str,
    goal_based: bool,
    safety_cap_timesteps: int,
    holdout_enabled: bool,
    holdout_ratio: float,
    holdout_split_at: Optional[str],
    resolved_holdout_split_at: Optional[str],
    data_cutoff: Optional[str],
    regime_start_at: Optional[str],
    regime_target_share: float,
    observation_schema: Dict[str, Any],
    command: List[str],
    budget_preflight: Dict[str, Any],
    model_record: Dict[str, Any],
    terminal_mastery_confirmed: bool = False,
    terminal_mastery_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create fail-closed provenance for the model written by a curriculum run."""
    terminal_reached = str(final_stage) == TERMINAL_CURRICULUM_STAGE
    mastery_evidence_confirmed = bool(
        terminal_mastery_evidence
        and terminal_mastery_evidence.get("confirmed") is True
    )
    curriculum_complete = bool(
        terminal_reached
        and terminal_mastery_confirmed
        and mastery_evidence_confirmed
        and str(run_outcome) == "finished"
    )
    if str(run_outcome) == "interrupted":
        status = "INTERRUPTED_BEFORE_EXTERNAL_VALIDATION"
    elif str(run_outcome) == "failed":
        status = "FAILED_BEFORE_EXTERNAL_VALIDATION"
    elif curriculum_complete:
        status = "TERMINAL_MASTERY_CONFIRMED_PENDING_EXTERNAL_ACCEPTANCE"
    elif terminal_reached:
        status = "LIVE_READY_REACHED_MASTERY_UNCONFIRMED"
    else:
        status = "CURRICULUM_NOT_TERMINAL"

    return {
        "provenance_version": CURRICULUM_PROVENANCE_VERSION,
        "status": status,
        "run_outcome": str(run_outcome),
        "accepted": False,
        "curriculum_complete": curriculum_complete,
        "terminal_stage_reached": terminal_reached,
        "terminal_mastery": dict(terminal_mastery_evidence or {"confirmed": False}),
        "terminal_stage": TERMINAL_CURRICULUM_STAGE,
        "final_stage": str(final_stage),
        "promotion_eligible": False,
        "actual_timesteps": int(actual_timesteps),
        "model": dict(model_record),
        "datasets": {
            "loaded": _manifest_provenance(loaded_manifest),
            "train": _manifest_provenance(train_manifest),
            "holdout": _manifest_provenance(holdout_manifest),
        },
        "arguments": {
            "start_stage": str(start_stage),
            "goal_based": bool(goal_based),
            "safety_cap_timesteps": int(safety_cap_timesteps),
            "holdout_enabled": bool(holdout_enabled),
            "holdout_ratio": float(holdout_ratio),
            "holdout_split_at": holdout_split_at,
            "resolved_holdout_split_at": resolved_holdout_split_at,
            "data_cutoff": data_cutoff,
            "regime_start_at": regime_start_at,
            "regime_target_share": float(regime_target_share),
        },
        "observation_schema": dict(observation_schema),
        "budget_preflight": dict(budget_preflight),
        "command": list(command),
        "git_head": loaded_manifest.get("git_head", _git_head()),
        "training_code_sha256": loaded_manifest.get("training_code_sha256"),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }


def _file_record(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def write_json_atomic(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def validate_bound_optuna_params(
    payload: Any,
    expected_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Optuna parameter file must contain a JSON object")
    params = payload.get("params")
    metadata = payload.get("metadata")
    if not isinstance(params, dict) or not isinstance(metadata, dict):
        raise ValueError(
            "refusing unbound Optuna parameters: expected params plus metadata; "
            "legacy flat JSON does not identify its dataset, code, or objective"
        )
    required = {
        "dataset_fingerprint": expected_manifest["dataset_fingerprint"],
        "training_code_sha256": expected_manifest["training_code_sha256"],
        "objective_version": OPTUNA_OBJECTIVE_VERSION,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Optuna {key} mismatch: file has {metadata.get(key)!r}, expected {expected!r}"
            )
    return dict(params)


def append_broker_bars(
    data: Dict[str, Dict[str, pd.DataFrame]],
    extra_dir: Optional[str] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Extend each timeframe with broker bars newer than the processed set.

    The processed CSVs end 2025-12-18. Everything after that is the post-war
    regime: XAUUSD 20-day realised volatility ran 0.74-0.90% through 2025,
    spiked to 3.28% in February 2026 and settled at 1.37-1.68%. Only 5.5% of the
    processed bars resemble it, which is why a model trained on them alone does
    not transfer.

    Only OHLCV and spread are appended. The processed files carry 58 engineered
    columns, but the env reads none of them - _prepare_market_data passes
    open/high/low/close/volume and nothing else - so the join is sound.
    """
    if not extra_dir or not os.path.exists(extra_dir):
        return data

    for instrument, frames in data.items():
        for tf, df in list(frames.items()):
            path = os.path.join(extra_dir, f"{instrument}_{tf}.csv")
            if df is None or df.empty or "time" not in df.columns:
                raise ValueError(f"base frame {instrument}/{tf} is missing or untimestamped")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"explicit broker dataset is incomplete: missing {path} for loaded timeframe {instrument}/{tf}"
                )
            try:
                extra = _normalize_market_frame(
                    pd.read_csv(path),
                    source=path,
                    require_spread=True,
                )
                df = _normalize_market_frame(
                    df,
                    source=f"loaded base {instrument}/{tf}",
                    require_spread=False,
                )
                extra = extra.loc[extra["time"] > df["time"].max()]
                if extra.empty:
                    continue

                # MT5 can report a zero spread when no valid quote was captured
                # for a bar (six such rows exist across the supplied exports).
                # Keep the marker: execution treats it as unavailable and falls
                # back to the configured conservative spread, while session
                # features use a neutral ratio.  Negative spread is invalid.

                merged = pd.concat([df, extra], ignore_index=True, sort=False)
                merged = merged.sort_values(by="time", ignore_index=True)
                for col in ("open", "high", "low", "close", "volume"):
                    merged[col] = merged[col].astype(np.float32)
                frames[tf] = merged
                logger.info(
                    "Appended %d broker bars to %s/%s (now %d, through %s)",
                    len(extra), instrument, tf, len(merged), merged["time"].iloc[-1],
                )
            except Exception as e:
                raise ValueError(f"invalid broker bars for {instrument}/{tf} at {path}: {e}") from e

    return data


def _create_synthetic_data(instruments: List[str], n_bars: int = 50000) -> Dict[str, Dict[str, pd.DataFrame]]:
    logger.info(f"Generating synthetic data: {n_bars} bars for {instruments}")
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    rng = np.random.default_rng(42)

    for instrument in instruments:
        if "XAU" in instrument or "GOLD" in instrument:
            base_price = 1900.0
            volatility = 0.008
        elif "EUR" in instrument:
            base_price = 1.10
            volatility = 0.003
        elif "GBP" in instrument:
            base_price = 1.27
            volatility = 0.004
        else:
            base_price = 100.0
            volatility = 0.01

        returns = np.zeros(n_bars, dtype=np.float64)
        trend = 0.0

        for i in range(n_bars):
            if rng.random() < 0.001:
                trend = rng.uniform(-0.0005, 0.0005)
            noise = rng.normal(0, volatility)
            mean_rev = -0.01 * np.sign(returns[:i].sum()) if i > 100 else 0.0
            returns[i] = trend + noise + mean_rev * 0.1

        prices = base_price * np.exp(np.cumsum(returns))
        opens = prices[:-1]
        closes = prices[1:]
        spread = rng.uniform(0.0003, 0.001, len(closes)) * closes
        highs = np.maximum(opens, closes) + spread
        lows = np.minimum(opens, closes) - spread

        base_vol = rng.integers(500, 2000, len(closes)).astype(np.float32)
        volume = base_vol * (1 + 0.5 * np.abs(returns[1:]))

        df = pd.DataFrame({
            "open": opens.astype(np.float32),
            "high": highs.astype(np.float32),
            "low": lows.astype(np.float32),
            "close": closes.astype(np.float32),
            "volume": volume.astype(np.float32),
        })
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        data[instrument] = {"M15": df}

    return data


def _min_len_across(data: Dict[str, Dict[str, pd.DataFrame]]) -> int:
    m = None
    for _, tfs in data.items():
        for _, df in tfs.items():
            n = len(df)
            m = n if m is None else min(m, n)
    return int(m or 0)


def slice_data_by_index(
    data: Dict[str, Dict[str, pd.DataFrame]],
    start: int,
    end: int,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for inst, tfs in data.items():
        out[inst] = {}
        for tf, df in tfs.items():
            s = max(0, int(start))
            e = min(int(end), len(df))

            out[inst][tf] = df.iloc[s:e].copy().reset_index(drop=True)
    return out


def _primary_frame(tfs: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    for tf in ("M15", "M5", "M30", "H1", "H4", "D1"):
        df = tfs.get(tf)
        if df is not None and not df.empty:
            return df
    return next((df for df in tfs.values() if df is not None and not df.empty), None)


def split_data_by_time(
    data: Dict[str, Dict[str, pd.DataFrame]],
    ratio: float,
    split_at: Optional[Any] = None,
    include_holdout_context: bool = False,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, pd.DataFrame]], Optional[Any]]:
    """Chronological train/holdout split at a shared timestamp.

    The previous split was index-based off _min_len_across(data), which is the
    D1 row count (1,091) rather than M15's 99,908, and slice_data_by_index then
    applied that same index range to every timeframe - so a split at index 927
    would have cut M15 down to 927 of its 99,908 bars. The "insufficient bars"
    guard masked it by disabling the holdout entirely, which is why runs
    trained and evaluated on the same data with no out-of-sample check at all.

    Splitting on a timestamp keeps the timeframes aligned: each one is cut
    wherever that instant falls in its own index.
    """
    train: Dict[str, Dict[str, pd.DataFrame]] = {}
    holdout: Dict[str, Dict[str, pd.DataFrame]] = {}
    split_ts: Optional[Any] = None

    for inst, tfs in data.items():
        primary = _primary_frame(tfs)
        if primary is None or "time" not in primary.columns:
            raise ValueError(f"{inst} has no timestamped primary frame; chronological split is impossible")

        # An explicit date beats a ratio here. The dataset grows every time broker
        # bars are appended, so a fixed ratio silently walks the split backwards:
        # at 0.15 over 114,399 bars it lands on 2025-11-07, which would end
        # training BEFORE the February 2026 regime change and leave the agent
        # never having seen the market it now has to trade.
        if split_at is not None:
            inst_split = pd.Timestamp(split_at)
        else:
            cut = int(len(primary) * (1.0 - ratio))
            cut = int(np.clip(cut, 1, len(primary) - 1))
            inst_split = primary["time"].iloc[cut]
        inst_split = pd.Timestamp(inst_split)
        inst_split = inst_split.tz_localize("UTC") if inst_split.tzinfo is None else inst_split.tz_convert("UTC")
        if split_ts is None:
            split_ts = inst_split

        train[inst], holdout[inst] = {}, {}
        for tf, df in tfs.items():
            if df is None or df.empty or "time" not in df.columns:
                raise ValueError(f"{inst}/{tf} has no timestamp column; chronological split is impossible")
            times = pd.to_datetime(df["time"], errors="coerce", utc=True)
            if times.isna().any():
                raise ValueError(f"{inst}/{tf} contains invalid timestamps")
            frame = df.copy()
            frame["time"] = times
            split_utc = inst_split
            mask = frame["time"] < split_utc
            train[inst][tf] = frame.loc[mask].copy().reset_index(drop=True)
            if include_holdout_context:
                holdout_frame = frame.copy().reset_index(drop=True)
                holdout_frame.attrs["episode_start_min_time"] = split_utc.isoformat()
                holdout[inst][tf] = holdout_frame
            else:
                holdout[inst][tf] = frame.loc[~mask].copy().reset_index(drop=True)

    return train, holdout, split_ts


def slice_data_by_time_range(
    data: Dict[str, Dict[str, pd.DataFrame]],
    start: Any,
    end: Any,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Half-open [start, end) slice applied by timestamp on every timeframe."""
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for inst, tfs in data.items():
        out[inst] = {}
        for tf, df in tfs.items():
            if df is None or df.empty or "time" not in df.columns:
                out[inst][tf] = df
                continue
            mask = df["time"] >= start if start is not None else df["time"] == df["time"]
            if end is not None:
                mask = mask & (df["time"] < end)
            out[inst][tf] = df.loc[mask].copy().reset_index(drop=True)
    return out


def build_walk_forward_folds(
    data: Dict[str, Dict[str, pd.DataFrame]],
    n_folds: int = 3,
    val_ratio: float = 0.12,
    min_train_ratio: float = 0.55,
) -> List[Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, pd.DataFrame]]]]:
    """Expanding-window walk-forward folds, cut by timestamp.

    Previously sized off _min_len_across(data) - the D1 row count - and sliced
    every timeframe with that same index range, so on this dataset it produced a
    single fold with 91 M15 training bars out of 99,908. That is the function
    the Optuna search runs on, so any hyperparameters tuned through it were
    fitted to 91 bars.

    Cutting on timestamps keeps the timeframes aligned and sizes the folds off
    the primary timeframe, the way the holdout split does.
    """
    if not data:
        return []

    primary = _primary_frame(next(iter(data.values())))
    if primary is None or "time" not in primary.columns:
        return []

    times = primary["time"].reset_index(drop=True)
    n = len(times)
    if n <= 0:
        return []

    val_len = max(1000, int(n * val_ratio))
    min_train = max(2000, int(n * min_train_ratio))
    max_train_end = n - val_len

    if max_train_end <= min_train + 100:
        cut = times.iloc[max(1, max_train_end)]
        val_data = slice_data_by_time_range(data, None, None)
        for frames in val_data.values():
            for frame in frames.values():
                if frame is not None:
                    frame.attrs["episode_start_min_time"] = pd.Timestamp(cut).isoformat()
        return [(
            slice_data_by_time_range(data, None, cut),
            val_data,
        )]

    folds: List[Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, pd.DataFrame]]]] = []
    train_ends = np.linspace(min_train, max_train_end, num=max(1, int(n_folds)), dtype=int)

    for train_end in np.unique(train_ends):
        train_cut = times.iloc[train_end]
        val_end = min(int(train_end) + val_len, n)
        val_cut = times.iloc[val_end] if val_end < n else None
        val_data = slice_data_by_time_range(data, None, val_cut)
        for frames in val_data.values():
            for frame in frames.values():
                if frame is not None:
                    frame.attrs["episode_start_min_time"] = pd.Timestamp(train_cut).isoformat()
        folds.append((
            slice_data_by_time_range(data, None, train_cut),
            val_data,
        ))

    return folds


def create_prop_firm_env(data: Dict[str, Dict[str, pd.DataFrame]], config: PropFirmConfig) -> PropFirmTradingEnv:
    return PropFirmTradingEnv(data, config)


def create_eval_vec_env(
    data: Dict[str, Dict[str, pd.DataFrame]],
    config: PropFirmConfig,
    seed: int = 0,
    use_action_masking: bool = True,
    frame_stack: int = 1,
) -> VecEnv:
    def make_env() -> Callable[[], Any]:
        def _init():
            env = create_prop_firm_env(data, config)

            if use_action_masking and MASKABLE_AVAILABLE and ActionMasker is not None:
                env = ActionMasker(env, _mask_fn)

            # reset() is where the observation is built, so every contract
            # violation surfaces here. Swallowing it at debug level meant an env
            # that could never produce a valid observation was still handed to
            # the trainer. Let it raise.
            env.reset(seed=seed)
            return env
        return _init

    venv = DummyVecEnv([make_env()])

    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=int(frame_stack))

    return venv


def _mask_fn(env: Any) -> np.ndarray:
    # Walk the wrapper chain to the env that owns action_masks(). The result is
    # asserted to be a bool array: a malformed mask would silently un-gate
    # actions the environment considers illegal, which is worse than crashing.
    current = env
    while True:
        fn = getattr(current, "action_masks", None)
        if callable(fn):
            mask = np.asarray(fn(), dtype=np.bool_)
            if mask.ndim != 1 or mask.size == 0:
                raise ValueError(
                    f"action_masks() returned shape {mask.shape}; expected a "
                    f"non-empty 1-D boolean array"
                )
            if not mask.any():
                raise ValueError("action_masks() returned an all-False mask - no legal action")
            return mask
        current = getattr(current, "env", None)
        if current is None:
            raise AttributeError(
                f"Could not find action_masks() on env or wrapped envs: {type(env)}"
            )


def create_vec_envs(
    data: Dict[str, Dict[str, pd.DataFrame]],
    config: PropFirmConfig,
    n_envs: int,
    seed: int,
    monitor_dir: str = "logs/training",
    use_action_masking: bool = True,
    frame_stack: int = 1,
) -> VecEnv:
    Path(monitor_dir).mkdir(parents=True, exist_ok=True)

    def make_env(rank: int) -> Callable[[], Any]:
        def _init():
            env = create_prop_firm_env(data, config)
            env = Monitor(env, filename=str(Path(monitor_dir) / f"monitor_{rank}.csv"))


            if use_action_masking and MASKABLE_AVAILABLE and ActionMasker is not None:
                env = ActionMasker(env, _mask_fn)


            # See create_eval_vec_env: a failed reset means a broken observation
            # contract, not a seeding inconvenience.
            env.reset(seed=seed + rank)
            return env
        return _init

    if platform.system() != "Windows" and n_envs > 1:
        start_method = pick_subproc_start_method() or "spawn"
        logger.info(f"Creating {n_envs} PARALLEL envs (SubprocVecEnv, start_method={start_method})")
        venv = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method=start_method)
    else:
        note = " (Windows: DummyVecEnv is sequential; keep n_envs small)" if platform.system() == "Windows" else ""
        logger.info(f"Creating {n_envs} envs (DummyVecEnv){note}")
        venv = DummyVecEnv([make_env(i) for i in range(n_envs)])

    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=int(frame_stack))

    return venv


def test_environment(data: Dict[str, Dict[str, pd.DataFrame]], config: PropFirmConfig) -> None:
    logger.info("Testing PropFirmTradingEnv sanity...")
    env = create_prop_firm_env(data, config)
    obs, _ = env.reset(seed=42)

    if not isinstance(obs, np.ndarray):
        raise ValueError("Observation is not ndarray")
    if not np.all(np.isfinite(obs)):
        raise ValueError("Observation contains non-finite values")

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if not np.all(np.isfinite(obs)):
            raise ValueError("Step produced non-finite observation")
        if not np.isfinite(reward):
            raise ValueError("Step produced non-finite reward")
        if not isinstance(info, dict):
            raise ValueError("Info is not dict")

    env.close()
    logger.info("✓ Environment test passed")


def evaluate_agent_trading(
    model: BaseAlgorithm,
    env: VecEnv,
    n_episodes: int = 10,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    use_action_masks: bool = True,
) -> Dict[str, float]:
    rewards: List[float] = []
    pnls: List[float] = []
    win_rates: List[float] = []
    drawdowns: List[float] = []
    trade_counts: List[int] = []
    dd_breaches = 0


    is_maskable = MASKABLE_AVAILABLE and MaskablePPO is not None and isinstance(model, MaskablePPO)


    def get_action_masks_from_vec_env(venv: VecEnv) -> Optional[np.ndarray]:

        if SB3_MASK_UTILS_AVAILABLE and sb3_get_action_masks is not None:
            try:
                return sb3_get_action_masks(venv)
            except Exception as e:
                logger.debug(f"sb3_get_action_masks failed, falling back: {e}")


        try:
            current: Any = venv
            while hasattr(current, 'venv'):
                current = current.venv
            if hasattr(current, 'envs'):
                envs_list = current.envs
                if envs_list and len(envs_list) > 0:
                    base_env: Any = envs_list[0]
                    while hasattr(base_env, 'env'):
                        if hasattr(base_env, 'action_masks') and callable(base_env.action_masks):
                            return np.array([base_env.action_masks()])
                        base_env = base_env.env
                    if hasattr(base_env, 'action_masks') and callable(base_env.action_masks):
                        return np.array([base_env.action_masks()])
        except Exception as e:
            logger.debug(f"Could not extract action_masks from vec env: {e}")
        return None

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        info: Dict[str, Any] = {}

        while not done:
            obs_array: np.ndarray = np.asarray(obs)


            predict_kwargs: Dict[str, Any] = {"deterministic": deterministic}

            if is_maskable and use_action_masks:
                action_masks = get_action_masks_from_vec_env(env)
                if action_masks is None:
                    raise RuntimeError("maskable evaluation could not obtain action masks")
                predict_kwargs["action_masks"] = action_masks


            action, _ = model.predict(obs_array, **predict_kwargs)


            obs, r, dones, infos = env.step(action)
            ep_reward += float(r[0])
            done = bool(dones[0])
            info = infos[0] if infos else {}
            steps += 1
            if max_steps is not None and steps >= max_steps:
                done = True


        finfo = info.get("terminal_info", info) if isinstance(info, dict) else {}

        pnl = float(finfo.get("total_pnl", 0.0))
        wr = float(finfo.get("win_rate", 0.0))
        dd = float(finfo.get("drawdown", 0.0))
        trades = int(finfo.get("trade_count", 0))


        if wr > 1.0:
            wr /= 100.0
        wr = float(np.clip(wr, 0.0, 1.0))

        if dd > 1.0:
            dd /= 100.0
        dd = float(np.clip(dd, 0.0, 1.0))

        rewards.append(ep_reward)
        pnls.append(pnl)
        win_rates.append(wr)
        drawdowns.append(dd)
        trade_counts.append(trades)

        term_reason = str(finfo.get("termination_reason", finfo.get("done_reason", "")))
        if "drawdown" in term_reason.lower():
            dd_breaches += 1

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "pnl_std": float(np.std(pnls)) if pnls else 0.0,
        "mean_win_rate": float(np.mean(win_rates)) if win_rates else 0.0,
        "mean_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
        "max_drawdown": float(np.max(drawdowns)) if drawdowns else 0.0,
        "mean_trades": float(np.mean(trade_counts)) if trade_counts else 0.0,
        "dd_breaches": float(dd_breaches),
    }


def score_trading_metrics(m: Dict[str, float], eval_episodes: int) -> float:
    mean_pnl = m["mean_pnl"]
    mean_wr = m["mean_win_rate"]
    max_dd = m["max_drawdown"]
    mean_trades = m["mean_trades"]
    pnl_std = m["pnl_std"]
    rew_std = m["reward_std"]
    mean_reward = m["mean_reward"]
    dd_breaches = m["dd_breaches"]

    profit_score = float(np.clip(mean_pnl / 500.0, -1.5, 2.5))

    if mean_wr < 0.40:
        winrate_score = -1.0
    elif mean_wr < 0.50:
        winrate_score = (mean_wr - 0.40) * 5.0 - 0.5
    elif mean_wr <= 0.70:
        winrate_score = (mean_wr - 0.50) * 5.0
    else:
        winrate_score = 1.0 - (mean_wr - 0.70) * 2.5

    if max_dd < 0.03:
        dd_score = 1.0
    elif max_dd < 0.05:
        dd_score = 0.8
    elif max_dd < 0.08:
        dd_score = 0.5 - (max_dd - 0.05) * 10.0
    else:
        dd_score = -0.5 - (max_dd - 0.08) * 6.0

    if mean_trades < 1:
        sel_score = -0.6
    elif mean_trades < 3:
        sel_score = (mean_trades - 1) * 0.25
    elif mean_trades <= 15:
        sel_score = 0.5 + (1.0 - abs(mean_trades - 8.0) / 7.0) * 0.5
    elif mean_trades <= 25:
        sel_score = 0.5 - (mean_trades - 15.0) * 0.05
    else:
        sel_score = -0.6

    consistency_score = 1.0 - float(np.clip(pnl_std / max(abs(mean_pnl) + 50.0, 50.0), 0.0, 2.0))

    if pnl_std > 1e-6:
        risk_adj = float(np.clip((mean_pnl / pnl_std) / 2.0, -1.0, 1.0))
    else:
        risk_adj = 0.5 if mean_pnl > 0 else 0.0

    survival_penalty = -0.6 * (dd_breaches / max(float(eval_episodes), 1.0))
    reward_stability = 1.0 - float(np.clip(rew_std / max(abs(mean_reward) + 0.1, 0.1), 0.0, 2.0))

    score = (
        0.28 * profit_score +
        0.18 * winrate_score +
        0.22 * dd_score +
        0.14 * sel_score +
        0.08 * consistency_score +
        0.06 * risk_adj +
        0.04 * reward_stability +
        survival_penalty
    )

    if mean_pnl > 200 and mean_wr > 0.55 and max_dd < 0.05 and mean_trades >= 3:
        score += 0.25

    return float(score)


def sample_ppo_hyperparams(trial: Any) -> Dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [2048, 4096, 8192]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "n_epochs": trial.suggest_int("n_epochs", 8, 20),
        "gamma": trial.suggest_float("gamma", 0.93, 0.995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.94, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.15, 0.30),

        "ent_coef": 0.10,
        "vf_coef": trial.suggest_float("vf_coef", 0.5, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.4, 0.8),
        "target_kl": trial.suggest_float("target_kl", 0.008, 0.05),
        "policy_hidden": trial.suggest_categorical("policy_hidden", [256, 512]),
        "value_hidden": trial.suggest_categorical("value_hidden", [256, 512]),
    }


def sample_env_hyperparams(trial: Any) -> Dict[str, Any]:
    return {
        "reward_scale": trial.suggest_float("reward_scale", 5.0, 15.0),
        "entry_quality_threshold": trial.suggest_float("entry_quality_threshold", 0.35, 0.65),
        "dd_penalty_scale": trial.suggest_float("dd_penalty_scale", 1.5, 4.5),
    }


def _sanity_adjust_ppo_params(n_envs: int, ppo_params: Dict[str, Any]) -> Dict[str, Any]:
    n_steps = int(ppo_params["n_steps"])
    batch_size = int(ppo_params["batch_size"])
    rollout = n_steps * n_envs
    if batch_size > rollout:
        ppo_params["batch_size"] = max(64, min(rollout, 256))
    return ppo_params


def _make_eval_config_adversarial(base: PropFirmConfig, adversity: float) -> PropFirmConfig:
    cfg = copy.deepcopy(base)

    cfg.domain_randomization_enabled = True
    cfg.spread_mult_range = (1.0 + 0.20 * adversity, 1.0 + 0.60 * adversity)
    cfg.slippage_mult_range = (1.0 + 0.25 * adversity, 1.0 + 0.80 * adversity)
    cfg.latency_bars_range = (int(1 + 1 * adversity), int(2 + 3 * adversity))
    cfg.volatility_scale_range = (1.0, 1.0 + 0.35 * adversity)

    cfg.execution.spread_shock_enabled = True
    cfg.execution.spread_shock_probability = 0.02 + 0.03 * adversity
    cfg.execution.spread_shock_multiplier = 2.0 + 2.0 * adversity

    return cfg


def run_optuna_optimization(
    data: Dict[str, Dict[str, pd.DataFrame]],
    n_trials: int = 40,
    timesteps_per_trial: int = 150_000,
    n_envs: int = 4,
    n_eval_episodes: int = 10,
    walk_forward_folds: int = 2,
    frame_stack: int = 4,
    study_name: str = "propfirm_ppo",
    storage: Optional[str] = None,
    holdout_boundary: Optional[Any] = None,
) -> Any:
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Run: pip install optuna")
    assert optuna is not None and TPESampler is not None and MedianPruner is not None

    folds = build_walk_forward_folds(data, n_folds=walk_forward_folds)
    if not folds:
        raise RuntimeError("Cannot build walk-forward folds (insufficient data).")

    development_manifest = build_dataset_manifest(
        data,
        role="optuna_development",
        boundary=holdout_boundary,
    )
    fold_boundaries: List[Dict[str, Any]] = []
    for fold_index, (fold_train, fold_val) in enumerate(folds):
        train_primary = _primary_frame(next(iter(fold_train.values())))
        val_primary = _primary_frame(next(iter(fold_val.values())))
        if train_primary is None or val_primary is None:
            raise RuntimeError(f"walk-forward fold {fold_index} has no primary frame")
        val_start = val_primary.attrs.get("episode_start_min_time")
        if val_start is None:
            raise RuntimeError(f"walk-forward fold {fold_index} lacks validation boundary metadata")
        fold_boundaries.append({
            "fold": fold_index,
            "train_end_exclusive": pd.Timestamp(val_start).isoformat(),
            "train_last_bar": pd.Timestamp(train_primary["time"].iloc[-1]).isoformat(),
            "validation_last_bar": pd.Timestamp(val_primary["time"].iloc[-1]).isoformat(),
        })

    study_binding = {
        "dataset_fingerprint": development_manifest["dataset_fingerprint"],
        "training_code_sha256": development_manifest["training_code_sha256"],
        "objective_version": OPTUNA_OBJECTIVE_VERSION,
        "fold_boundaries": fold_boundaries,
        "git_head": development_manifest["git_head"],
    }
    bound_study_name = (
        f"{study_name}__{development_manifest['dataset_fingerprint'][:12]}__"
        f"{OPTUNA_OBJECTIVE_VERSION}"
    )

    sampler = TPESampler(seed=42, multivariate=True)
    per_fold_steps = timesteps_per_trial // walk_forward_folds
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=per_fold_steps * 2,
        n_min_trials=8,
        interval_steps=per_fold_steps,
    )

    study = optuna.create_study(
        study_name=bound_study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bool(storage),
    )
    existing_binding = dict(getattr(study, "user_attrs", {}) or {})
    if len(getattr(study, "trials", [])) > 0:
        for key, expected in study_binding.items():
            if existing_binding.get(key) != expected:
                raise RuntimeError(
                    f"refusing incompatible Optuna resume: {key}="
                    f"{existing_binding.get(key)!r}, expected {expected!r}"
                )
    for key, value in study_binding.items():
        study.set_user_attr(key, value)

    results_dir = Path("logs/optuna")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "trial_results.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Optuna start: trials={n_trials} steps/trial={timesteps_per_trial:,} folds={len(folds)} "
        f"n_envs={n_envs} device={device} masking={MASKABLE_AVAILABLE}"
    )

    def objective(trial: Any) -> float:
        seed_everything(10_000 + trial.number)

        ppo_params = _sanity_adjust_ppo_params(n_envs, sample_ppo_hyperparams(trial))
        env_params = sample_env_hyperparams(trial)

        base_cfg = PropFirmConfig(
            max_steps_per_episode=2000,
            entry_quality_threshold=float(env_params["entry_quality_threshold"]),
            max_trades_per_day=12,
            max_trades_per_session=6,
            domain_randomization_enabled=True,  # type: ignore[attr-defined]
        )

        base_cfg.reward.reward_scale = float(env_params["reward_scale"])
        base_cfg.reward.dd_penalty_scale = float(env_params["dd_penalty_scale"])

        def build_model(train_env: VecEnv) -> BaseAlgorithm:
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[int(ppo_params["policy_hidden"]), int(ppo_params["policy_hidden"]) // 2],
                    vf=[int(ppo_params["value_hidden"]), int(ppo_params["value_hidden"]) // 2],
                ),
                activation_fn=nn.Tanh,
            )

            if MASKABLE_AVAILABLE and MaskablePPO is not None:
                return MaskablePPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=float(ppo_params["learning_rate"]),
                    n_steps=int(ppo_params["n_steps"]),
                    batch_size=int(ppo_params["batch_size"]),
                    n_epochs=int(ppo_params["n_epochs"]),
                    gamma=float(ppo_params["gamma"]),
                    gae_lambda=float(ppo_params["gae_lambda"]),
                    clip_range=float(ppo_params["clip_range"]),
                    ent_coef=float(ppo_params["ent_coef"]),
                    vf_coef=float(ppo_params["vf_coef"]),
                    max_grad_norm=float(ppo_params["max_grad_norm"]),
                    target_kl=float(ppo_params["target_kl"]),
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    device=device,
                    seed=10_000 + trial.number,
                )

            return PPO(
                "MlpPolicy",
                train_env,
                learning_rate=float(ppo_params["learning_rate"]),
                n_steps=int(ppo_params["n_steps"]),
                batch_size=int(ppo_params["batch_size"]),
                n_epochs=int(ppo_params["n_epochs"]),
                gamma=float(ppo_params["gamma"]),
                gae_lambda=float(ppo_params["gae_lambda"]),
                clip_range=float(ppo_params["clip_range"]),
                ent_coef=float(ppo_params["ent_coef"]),
                vf_coef=float(ppo_params["vf_coef"]),
                max_grad_norm=float(ppo_params["max_grad_norm"]),
                target_kl=float(ppo_params["target_kl"]),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=device,
                seed=10_000 + trial.number,
            )

        fold_scores: List[float] = []
        per_fold_steps_local = max(25_000, int(timesteps_per_trial / max(len(folds), 1)))

        for fi, (train_data, val_data) in enumerate(folds):
            train_env: Optional[VecEnv] = None
            eval_env: Optional[VecEnv] = None
            try:
                train_env = create_vec_envs(
                    train_data, base_cfg,
                    n_envs=n_envs, seed=trial.number + 100 * fi,
                    monitor_dir=f"logs/optuna/trial_{trial.number}",
                    use_action_masking=True,
                    frame_stack=frame_stack,
                )
                try:
                    train_env.seed(1_000 + trial.number + 10 * fi)
                except Exception as e:
                    logger.debug(f"Could not seed optuna train_env: {e}")

                model = build_model(train_env)

                optuna_callback = VecEpisodeTradingCallback(
                    total_timesteps=per_fold_steps_local,
                    log_interval_steps=per_fold_steps_local + 1,
                    metrics_file=f"logs/optuna/trial_{trial.number}/live_metrics.json",
                )

                logger.info(f"[Trial {trial.number}] Fold {fi+1}/{len(folds)} - Training {per_fold_steps_local:,} steps...")
                model.learn(total_timesteps=per_fold_steps_local, progress_bar=True, callback=optuna_callback)

                eval_cfg = _make_eval_config_adversarial(base_cfg, adversity=0.7)
                val_primary = _primary_frame(next(iter(val_data.values())))
                val_start = val_primary.attrs.get("episode_start_min_time") if val_primary is not None else None
                if val_start is None:
                    raise RuntimeError("walk-forward validation fold lacks an eligible-start boundary")
                eval_cfg.episode_start_min_time = str(val_start)
                eval_cfg.mirror_augmentation_prob = 0.0
                eval_cfg.high_vol_oversample_prob = 0.0
                eval_env = create_eval_vec_env(
                    val_data, eval_cfg,
                    seed=20_000 + trial.number + fi,
                    use_action_masking=True,
                    frame_stack=frame_stack,
                )

                metrics = evaluate_agent_trading(model, eval_env, n_episodes=n_eval_episodes, deterministic=True)
                score = score_trading_metrics(metrics, eval_episodes=n_eval_episodes)
                fold_scores.append(float(score))

                current_mean = float(np.mean(fold_scores))
                current_std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
                pessimistic_score = current_mean - 0.1 * current_std

                trial.report(pessimistic_score, (fi + 1) * per_fold_steps_local)

                if fi >= 1 and trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned after fold {fi+1}: score={pessimistic_score:.3f}")
                    raise TrialPruned()

            finally:
                try:
                    if train_env is not None:
                        train_env.close()
                except Exception as e:
                    logger.debug(f"Train env cleanup: {e}")
                try:
                    if eval_env is not None:
                        eval_env.close()
                except Exception as e:
                    logger.debug(f"Eval env cleanup: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        mean_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
        score_std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
        robust_score = mean_score - 0.15 * score_std

        record = {
            "trial": trial.number,
            "robust_score": robust_score,
            "mean_score": mean_score,
            "score_std": score_std,
            "fold_scores": fold_scores,
            "params": trial.params,
            "timestamp": datetime.now().isoformat(),
            "dataset_fingerprint": development_manifest["dataset_fingerprint"],
            "objective_version": OPTUNA_OBJECTIVE_VERSION,
        }
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(
            f"Trial {trial.number}: robust={robust_score:.3f} mean={mean_score:.3f} std={score_std:.3f} "
            f"folds={len(fold_scores)}"
        )
        return robust_score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("=" * 70)
    logger.info("OPTUNA COMPLETE")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    Path("logs/optuna").mkdir(parents=True, exist_ok=True)
    with open("logs/optuna/best_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "params": study.best_params,
                "metadata": study_binding,
                "study_name": bound_study_name,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    write_dataset_manifest(development_manifest, Path("logs/optuna/dataset_manifest.json"))

    return study


def train_prop_firm_agent(
    data: Dict[str, Dict[str, pd.DataFrame]],
    total_timesteps: int,
    n_envs: int,
    learning_rate: float,
    batch_size: int,
    n_steps: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    target_kl: float,
    policy_hidden: int,
    value_hidden: int,
    checkpoint_freq: int,
    eval_freq: int,
    pretrained_path: Optional[str],
    config_overrides: Dict[str, Any],
    frame_stack: int,
    seed: int,
) -> BaseAlgorithm:
    seed_everything(seed)

    config = build_propfirm_config(config_overrides)

    test_environment(data, config)

    use_masking = bool(MASKABLE_AVAILABLE)


    train_monitor_dir = "logs/propfirm/training"

    train_env = create_vec_envs(
        data, config,
        n_envs=n_envs, seed=seed,
        monitor_dir=train_monitor_dir,
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )
    try:
        train_env.seed(seed)
    except Exception as e:
        logger.debug(f"Could not seed train_env: {e}")


    eval_cfg = copy.deepcopy(config)
    try:
        eval_cfg.domain_randomization_enabled = False
        eval_cfg.spread_mult_range = (1.0, 1.0)
        eval_cfg.slippage_mult_range = (1.0, 1.0)
        eval_cfg.latency_bars_range = (0, 0)
        eval_cfg.volatility_scale_range = (1.0, 1.0)
    except Exception:
        pass

    eval_env_vec = create_vec_envs(
        data, eval_cfg,
        n_envs=1, seed=seed + 1337,
        monitor_dir="logs/propfirm/eval_vec",
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )


    eval_cfg_adv = _make_eval_config_adversarial(config, adversity=0.8)
    eval_env_single = create_eval_vec_env(
        data, eval_cfg_adv,
        seed=seed + 2337,
        use_action_masking=use_masking,
        frame_stack=frame_stack,
    )

    model_dir = Path("models/propfirm")
    checkpoint_dir = Path("checkpoints/propfirm")
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device.upper()} | n_envs={n_envs} | masking={use_masking} | frame_stack={frame_stack} | seed={seed}")

    policy_kwargs = dict(
        net_arch=dict(
            pi=[policy_hidden, policy_hidden // 2],
            vf=[value_hidden, value_hidden // 2],
        ),
        activation_fn=nn.Tanh,
    )

    Algo = PPO
    if use_masking and MASKABLE_AVAILABLE and MaskablePPO is not None:
        Algo = MaskablePPO  # type: ignore[assignment]

    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained model: {pretrained_path}")
        model = Algo.load(pretrained_path, env=train_env, device=device)  # type: ignore[attr-defined]


        try:
            from envs.prop_firm_env import PPO_OBS_VERSION, validate_observation_version
            model_obs_size: int = model.observation_space.shape[0]  # type: ignore[union-attr]
            saved_version = getattr(model, '_obs_version', PPO_OBS_VERSION)
            validate_observation_version(saved_version, model_obs_size)
            logger.info(f"Observation validation passed: size={model_obs_size}, version={saved_version}")
        except ValueError as e:
            logger.error(f"CRITICAL: {e}")
            raise
        except Exception as e:
            logger.warning(f"Could not validate observation version: {e}")

        current_steps = int(getattr(model, "num_timesteps", 0))
    else:
        model = Algo(  # type: ignore[call-arg]
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log="runs/propfirm",
            device=device,
            seed=seed,
        )
        current_steps = 0

    remaining = max(0, total_timesteps - current_steps)
    logger.info(f"Target timesteps: {total_timesteps:,} | current: {current_steps:,} | remaining: {remaining:,}")

    eval_freq_calls = max(1, eval_freq // max(n_envs, 1))

    if use_masking and MASKABLE_EVAL_AVAILABLE and MaskableEvalCallback is not None:
        eval_cb = MaskableEvalCallback(
            eval_env_vec,
            best_model_save_path=str(model_dir / "best"),
            log_path="logs/propfirm/eval",
            eval_freq=eval_freq_calls,
            deterministic=True,
            n_eval_episodes=5,
        )
    else:
        eval_cb = EvalCallback(
            eval_env_vec,
            best_model_save_path=str(model_dir / "best"),
            log_path="logs/propfirm/eval",
            eval_freq=eval_freq_calls,
            deterministic=True,
            n_eval_episodes=5,
        )

    callbacks: List[BaseCallback] = [
        VecEpisodeTradingCallback(
            total_timesteps=total_timesteps,
            log_interval_steps=50_000,

            metrics_file="logs/training/live_metrics.json",
        ),
        CheckpointCallback(
            save_freq=max(1, checkpoint_freq // max(n_envs, 1)),
            save_path=str(checkpoint_dir),
            name_prefix="propfirm_ppo",
        ),
        eval_cb,
    ]

    class AdversarialEvalLogger(BaseCallback):
        def __init__(self, eval_env: VecEnv, freq: int = 300_000):
            super().__init__(0)
            self.eval_env = eval_env
            self.freq = int(freq)
            self._last = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last < self.freq:
                return True
            m = evaluate_agent_trading(self.model, self.eval_env, n_episodes=8, deterministic=True)
            s = score_trading_metrics(m, eval_episodes=8)
            logger.info(
                f"[ADVERSARIAL EVAL] step={self.num_timesteps:,} score={s:.3f} "
                f"PnL=€{m['mean_pnl']:+.0f} WR={m['mean_win_rate']:.1%} MaxDD={m['max_drawdown']:.1%} Trades={m['mean_trades']:.1f}"
            )
            self._last = self.num_timesteps
            return True

    callbacks.append(AdversarialEvalLogger(eval_env_single, freq=300_000))

    start_time = datetime.now()
    try:
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=CallbackList(callbacks),
                tb_log_name="propfirm_ppo",
                reset_num_timesteps=(current_steps == 0),
                progress_bar=True,
            )
        else:
            logger.info("Already at target timesteps; skipping training.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        model.save(str(model_dir / "propfirm_ppo_interrupted.zip"))
    finally:
        duration = datetime.now() - start_time
        logger.info(f"Training duration: {duration}")

        final_path = model_dir / "propfirm_ppo_final.zip"
        model.save(str(final_path))
        logger.info(f"Saved final model: {final_path}")

        metadata = {
            "total_timesteps_target": int(total_timesteps),
            "actual_timesteps": int(getattr(model, "num_timesteps", 0)),
            "device": device,
            "n_envs": int(n_envs),
            "masking": bool(use_masking),
            "frame_stack": int(frame_stack),
            "seed": int(seed),
            "hyperparameters": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_steps": n_steps,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "target_kl": target_kl,
                "policy_hidden": policy_hidden,
                "value_hidden": value_hidden,
            },
            "config_overrides": config_overrides,
            "created_at": datetime.now().isoformat(),
        }
        with open(model_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        try:
            train_env.close()
        except Exception as e:
            logger.debug(f"Train env cleanup: {e}")
        try:
            eval_env_vec.close()
        except Exception as e:
            logger.debug(f"Eval vec env cleanup: {e}")
        try:
            eval_env_single.close()
        except Exception as e:
            logger.debug(f"Eval single env cleanup: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


def create_curriculum_env(
    data: Dict[str, Dict[str, pd.DataFrame]],
    curriculum_manager: Any,
    seed: int = 0,
    monitor_dir: Optional[str] = None,
    use_action_masking: bool = True,
    episode_start_min_time: Optional[str] = None,
    raw_evaluation_mode: bool = False,
    recent_regime_start_time: Optional[str] = None,
    recent_regime_target_share: float = 0.0,
) -> Any:
    if not CURRICULUM_AVAILABLE:
        raise RuntimeError("Curriculum system not available")

    stage_config = curriculum_manager.stage_config

    base_config = PropFirmConfig(
        max_steps_per_episode=stage_config.max_steps_per_episode,
        entry_quality_gate_enabled=stage_config.constraints.entry_quality_gate_enabled,
        entry_quality_threshold=stage_config.constraints.entry_quality_threshold,
        episode_start_min_time=episode_start_min_time,
        raw_evaluation_mode=bool(raw_evaluation_mode),
        recent_regime_start_time=recent_regime_start_time,
        recent_regime_target_share=float(recent_regime_target_share),
    )

    env = PropFirmTradingEnv(
        data,
        base_config,
        curriculum_manager=curriculum_manager,
        apply_curriculum_overrides=True,
    )

    if monitor_dir:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(Path(monitor_dir) / f"monitor_{seed}.csv"))

    if use_action_masking and MASKABLE_AVAILABLE and ActionMasker is not None:
        env = ActionMasker(env, _mask_fn)

    # Was `except Exception: pass` - the most silent form. A curriculum env that
    # cannot reset cannot produce an observation, and training would have
    # started against it regardless.
    env.reset(seed=seed)

    return env


def create_curriculum_vec_envs(
    data: Dict[str, Dict[str, pd.DataFrame]],
    curriculum_manager: Any,
    n_envs: int,
    seed: int,
    monitor_dir: str = "logs/curriculum/training",
    use_action_masking: bool = True,
    frame_stack: int = 1,
    episode_start_min_time: Optional[str] = None,
    raw_evaluation_mode: bool = False,
    recent_regime_start_time: Optional[str] = None,
    recent_regime_target_share: float = 0.0,
) -> VecEnv:
    Path(monitor_dir).mkdir(parents=True, exist_ok=True)

    def make_env(rank: int) -> Callable[[], Any]:
        def _init():
            env = create_curriculum_env(
                data=data,
                curriculum_manager=curriculum_manager,
                seed=seed + rank,
                monitor_dir=monitor_dir,
                use_action_masking=use_action_masking,
                episode_start_min_time=episode_start_min_time,
                raw_evaluation_mode=raw_evaluation_mode,
                recent_regime_start_time=recent_regime_start_time,
                recent_regime_target_share=recent_regime_target_share,
            )
            return env
        return _init

    if n_envs > 1 and platform.system() != "Windows":
        logger.warning(
            "Curriculum mode: forcing DummyVecEnv to keep a truly shared CurriculumManager. "
            "SubprocVecEnv would desynchronize stage transitions."
        )

    logger.info(f"Creating {n_envs} curriculum envs (DummyVecEnv - shared CurriculumManager)")
    venv = DummyVecEnv([make_env(i) for i in range(n_envs)])

    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=int(frame_stack))

    return venv


def train_curriculum_agent(
    data: Dict[str, Dict[str, pd.DataFrame]],
    total_timesteps: int,
    n_envs: int,
    learning_rate: float,
    batch_size: int,
    n_steps: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    target_kl: float,
    policy_hidden: int,
    value_hidden: int,
    checkpoint_freq: int,
    start_stage: str = "EXPLORER",
    resume_path: Optional[str] = None,
    load_model_path: Optional[str] = None,
    load_controllers_path: Optional[str] = None,
    load_metrics_path: Optional[str] = None,
    frame_stack: int = 1,
    goal_based_stopping: bool = False,
    max_hours: Optional[float] = None,
    plateau_threshold_episodes: int = 500,
    max_demotions_from_same_stage: int = 5,
    mastery_confirmation_episodes: int = 100,
    seed: int = 42,
    holdout_ratio: float = 0.15,
    holdout_split_at: Optional[str] = None,
    recent_regime_start_time: Optional[str] = None,
    recent_regime_target_share: float = 0.0,
    data_cutoff: Optional[str] = None,
    loaded_dataset_manifest: Optional[Dict[str, Any]] = None,
    command: Optional[List[str]] = None,
) -> BaseAlgorithm:
    if not CURRICULUM_AVAILABLE or CurriculumStage is None or CurriculumManager is None:
        raise RuntimeError("Curriculum system not available. Check imports.")

    seed_everything(seed)

    save_dir = Path("models/curriculum")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_masking = bool(MASKABLE_AVAILABLE)

    initial_stage = CurriculumStage[start_stage]

    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming curriculum from {resume_path}")
        curriculum_manager = CurriculumManager.load(Path(resume_path))
    else:
        curriculum_manager = CurriculumManager(
            initial_stage=initial_stage,
            auto_promote=True,
            auto_demote=True,
            verbose=True,
            rng_seed=seed,
        )

    current_stage = curriculum_manager.current_stage
    if current_stage is None:
        raise RuntimeError("CurriculumManager has no current stage")
    run_start_stage = current_stage.name
    logger.info(f"Curriculum training: starting at {current_stage.name}")
    if get_stage_progression is not None:
        logger.info(f"Stage progression: {' → '.join(s.name for s in get_stage_progression())}")

    if goal_based_stopping:
        # Reject obviously impossible fresh/resumed plans before constructing
        # vector environments.  A second check after loading the model uses the
        # larger of checkpoint and manager timestep counters.
        early_preflight = build_curriculum_budget_preflight(
            start_stage=curriculum_manager.current_stage,
            safety_cap_timesteps=total_timesteps,
            already_consumed_timesteps=int(getattr(curriculum_manager, "total_timesteps", 0) or 0),
            current_stage_timesteps=int(getattr(curriculum_manager, "stage_timesteps", 0) or 0),
        )
        enforce_goal_based_budget(early_preflight, goal_based=True)


    n_bars = _min_len_across(data)
    if n_bars <= 0:
        raise ValueError("No bars available for curriculum training")

    try:
        holdout_ratio = float(holdout_ratio)
    except Exception:
        holdout_ratio = 0.0
    if not np.isfinite(holdout_ratio):
        holdout_ratio = 0.0
    holdout_ratio = float(np.clip(holdout_ratio, 0.0, 0.50))


    stage_max_steps = int(getattr(curriculum_manager.stage_config, "max_steps_per_episode", 0) or 0)
    max_steps_required = stage_max_steps
    if get_stage_progression is not None and get_stage_config is not None:
        try:
            max_steps_required = max(
                int(get_stage_config(s).max_steps_per_episode) for s in get_stage_progression()
            )
        except Exception:
            max_steps_required = stage_max_steps

    # Sized against the PRIMARY timeframe, not _min_len_across. The old length
    # checks used the D1 row count (1,091 vs M15's 99,908) and concluded there
    # were not enough bars for a 15% holdout, so every run silently trained and
    # evaluated on the same data.
    primary = _primary_frame(next(iter(data.values()))) if data else None
    n_primary = len(primary) if primary is not None else n_bars

    min_holdout_len = max(1500, max_steps_required + 2)
    min_train_len = max(3000, max_steps_required + 2000)
    holdout_requested = holdout_split_at is not None or holdout_ratio > 0.0

    if not holdout_requested:
        train_data = data
        holdout_data = data
        holdout_enabled = False
        split_ts = None
    else:
        train_data, holdout_data, split_ts = split_data_by_time(
            data,
            holdout_ratio,
            split_at=holdout_split_at,
            include_holdout_context=True,
        )
        tr_frame = _primary_frame(next(iter(train_data.values())))
        ho_frame = _primary_frame(next(iter(holdout_data.values())))
        if tr_frame is None or ho_frame is None or split_ts is None:
            raise ValueError("chronological holdout partition produced no primary frame")
        split_utc = pd.Timestamp(split_ts)
        split_utc = split_utc.tz_localize("UTC") if split_utc.tzinfo is None else split_utc.tz_convert("UTC")
        holdout_times = pd.to_datetime(ho_frame["time"], errors="coerce", utc=True)
        eligible_holdout_len = int((holdout_times >= split_utc).sum())
        train_len = len(tr_frame)
        if eligible_holdout_len < min_holdout_len or train_len < min_train_len:
            raise ValueError(
                "requested holdout is not executable: "
                f"train={train_len} (need {min_train_len}), post_cut={eligible_holdout_len} "
                f"(need {min_holdout_len}), split={split_utc.isoformat()}"
            )
        holdout_enabled = True
        logger.info(
            f"Holdout split at {split_utc}: train={train_len} primary bars, "
            f"holdout={eligible_holdout_len} eligible bars plus causal pre-cut context"
        )

    resolved_split_at = None
    if split_ts is not None:
        resolved_split = pd.Timestamp(split_ts)
        resolved_split = (
            resolved_split.tz_localize("UTC")
            if resolved_split.tzinfo is None
            else resolved_split.tz_convert("UTC")
        )
        resolved_split_at = resolved_split.isoformat()

    loaded_manifest = loaded_dataset_manifest or build_dataset_manifest(
        data,
        role="loaded",
        boundary=data_cutoff,
    )
    train_manifest = build_dataset_manifest(
        train_data,
        role="curriculum_train",
        boundary=split_ts,
    )
    holdout_manifest = build_dataset_manifest(
        holdout_data,
        role="curriculum_holdout_with_context" if holdout_enabled else "curriculum_holdout_disabled_train_reuse",
        boundary=split_ts,
    )
    write_dataset_manifest(loaded_manifest, save_dir / "loaded_dataset_manifest.json")
    write_dataset_manifest(train_manifest, save_dir / "train_dataset_manifest.json")
    write_dataset_manifest(holdout_manifest, save_dir / "holdout_dataset_manifest.json")

    train_env = create_curriculum_vec_envs(
        data=train_data,
        curriculum_manager=curriculum_manager,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(save_dir / "training"),
        use_action_masking=use_masking,
        frame_stack=frame_stack,
        recent_regime_start_time=recent_regime_start_time,
        recent_regime_target_share=recent_regime_target_share,
    )


    holdout_validation_env = create_curriculum_vec_envs(
        data=holdout_data,
        curriculum_manager=curriculum_manager,
        n_envs=1,
        seed=seed + 4242,
        monitor_dir=str(save_dir / "holdout_validation"),
        use_action_masking=use_masking,
        frame_stack=frame_stack,
        episode_start_min_time=str(split_ts) if holdout_enabled else None,
        raw_evaluation_mode=True,
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[policy_hidden, policy_hidden // 2],
            vf=[value_hidden, value_hidden // 2],
        ),
        activation_fn=nn.Tanh,
    )

    Algo = PPO
    if use_masking and MASKABLE_AVAILABLE and MaskablePPO is not None:
        Algo = MaskablePPO  # type: ignore[assignment]

    if load_model_path and Path(load_model_path).exists():
        logger.info(f"Loading model from: {load_model_path}")
        model = Algo.load(
            load_model_path,
            env=train_env,
            device=device,
        )

        loaded_n_steps = getattr(model, 'n_steps', n_steps)
        loaded_batch_size = getattr(model, 'batch_size', batch_size)
        if loaded_n_steps != n_steps:
            logger.warning(
                f"Loaded model n_steps={loaded_n_steps} differs from CLI n_steps={n_steps}. "
                f"Keeping loaded value to avoid rollout buffer mismatch."
            )
        if loaded_batch_size != batch_size:
            logger.warning(
                f"Loaded model batch_size={loaded_batch_size} differs from CLI batch_size={batch_size}. "
                f"Keeping loaded value."
            )


        saved_lr = getattr(model, 'learning_rate', learning_rate)
        if callable(saved_lr):
            saved_lr = saved_lr(1.0)
        saved_ent_coef = getattr(model, 'ent_coef', ent_coef)
        saved_clip_range = getattr(model, 'clip_range', clip_range)
        if callable(saved_clip_range):
            saved_clip_range = saved_clip_range(1.0)


        try:
            optimizer_lr = model.policy.optimizer.param_groups[0]["lr"]
        except Exception:
            optimizer_lr = saved_lr

        logger.info(
            f"PRESERVING saved model state: lr={saved_lr:.2e} (optimizer: {optimizer_lr:.2e}), "
            f"ent_coef={saved_ent_coef:.4f}, clip_range={saved_clip_range:.3f}"
        )
        logger.info(
            f"CLI values (NOT applied): lr={learning_rate:.2e}, ent_coef={ent_coef:.4f}, "
            f"clip_range={clip_range:.3f}"
        )

        try:
            from envs.prop_firm_env import PPO_OBS_VERSION, validate_observation_version
            model_obs_size: int = model.observation_space.shape[0]  # type: ignore[union-attr]
            saved_version = getattr(model, '_obs_version', PPO_OBS_VERSION)
            validate_observation_version(saved_version, model_obs_size)
            logger.info(f"Observation validation passed: size={model_obs_size}, version={saved_version}")
        except ValueError as e:
            logger.error(f"CRITICAL: {e}")
            raise
        except Exception as e:
            logger.warning(f"Could not validate observation version: {e}")

        logger.info(f"Model loaded successfully. Previous timesteps: {model.num_timesteps}")
    else:
        model = Algo(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log="runs/curriculum",
            device=device,
            seed=seed,
        )

    save_freq_calls = max(1, checkpoint_freq // max(n_envs, 1))

    terminal_min_episodes = 0
    if get_stage_config is not None:
        terminal_min_episodes = int(
            get_stage_config(CurriculumStage.LIVE_READY).competence.min_episodes
        )
    effective_mastery_episodes = max(
        int(mastery_confirmation_episodes),
        terminal_min_episodes,
    )
    if effective_mastery_episodes != int(mastery_confirmation_episodes):
        logger.warning(
            "Raising LIVE_READY confirmation from %s to the terminal competence floor of %s episodes",
            f"{int(mastery_confirmation_episodes):,}",
            f"{effective_mastery_episodes:,}",
        )

    curriculum_training_callback = CurriculumTrainingCallback(
        curriculum_manager=curriculum_manager,
        total_timesteps=total_timesteps,
        log_interval_steps=50_000,
        save_path=str(save_dir),
        metrics_file="logs/training/live_metrics.json",
        base_ent_coef=ent_coef,
        base_clip_range=clip_range,
        goal_based_stopping=goal_based_stopping,
        max_hours=max_hours,
        plateau_stop=True,
        plateau_threshold_episodes=plateau_threshold_episodes,
        max_demotions_from_same_stage=max_demotions_from_same_stage,
        mastery_confirmation_episodes=effective_mastery_episodes,
    )


    if load_controllers_path and Path(load_controllers_path).exists():
        curriculum_training_callback.load_controller_states(Path(load_controllers_path))


    if load_metrics_path and Path(load_metrics_path).exists():
        curriculum_training_callback.load_metrics_state(Path(load_metrics_path))


    curriculum_training_callback._eval_env = holdout_validation_env

    callbacks: List[BaseCallback] = [
        curriculum_training_callback,
        CheckpointCallback(
            save_freq=save_freq_calls,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="curriculum_ppo",
        ),
        CurriculumCheckpointCallback(
            curriculum_manager=curriculum_manager,
            save_freq=save_freq_calls,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="curriculum_state",
        ),
    ]

    # Do not select a "best" checkpoint by raw reward on this development
    # window.  Curriculum reward scales and episode lengths change by stage, so
    # those scores are not comparable; querying the same thin holdout every
    # 50k steps also turns it into training feedback.  The bounded promotion and
    # terminal-readiness gates above are the only in-run development queries.
    logger.info(
        "Periodic holdout best-model selection disabled; development holdout "
        "is reserved for spaced curriculum gates"
    )

    current_steps = int(getattr(model, 'num_timesteps', 0))
    remaining_timesteps = max(0, total_timesteps - current_steps)
    should_reset_num_timesteps = (current_steps == 0)

    consumed_for_budget = max(
        current_steps,
        int(getattr(curriculum_manager, "total_timesteps", 0) or 0),
    )
    budget_preflight = build_curriculum_budget_preflight(
        start_stage=curriculum_manager.current_stage,
        safety_cap_timesteps=total_timesteps,
        already_consumed_timesteps=consumed_for_budget,
        current_stage_timesteps=int(getattr(curriculum_manager, "stage_timesteps", 0) or 0),
    )
    logger.info(
        "Curriculum budget floor: start=%s available=%s required=%s max_theoretical_stage=%s",
        budget_preflight["start_stage"],
        f"{budget_preflight['available_timesteps']:,}",
        f"{budget_preflight['minimum_remaining_timesteps']:,}",
        budget_preflight["maximum_theoretical_stage"],
    )
    for stage_budget in budget_preflight["stages"]:
        required = stage_budget["global_timesteps_required"]
        required_text = "unfundable" if required is None else f"{required:,}"
        logger.info(
            "  %-17s local_remaining=%10s current_share=%6.2f%% global_floor=%s",
            stage_budget["stage"],
            f"{stage_budget['local_timesteps_remaining']:,}",
            100.0 * float(stage_budget["declared_current_stage_probability"]),
            required_text,
        )
    if goal_based_stopping:
        enforce_goal_based_budget(budget_preflight, goal_based=True)
    else:
        logger.info(
            "Fixed-run maximum theoretical stage at the configured cap: %s",
            budget_preflight["maximum_theoretical_stage"],
        )

    if current_steps > 0:
        logger.info(
            f"Resuming from {current_steps:,} timesteps. "
            f"Remaining: {remaining_timesteps:,} (total target: {total_timesteps:,})"
        )

    start_time = datetime.now()
    run_outcome = "finished"
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name="curriculum_ppo",
            progress_bar=True,
            reset_num_timesteps=should_reset_num_timesteps,
        )
    except KeyboardInterrupt:
        run_outcome = "interrupted"
        logger.warning("Curriculum training interrupted by user")
    except BaseException:
        run_outcome = "failed"
        raise
    finally:
        duration = datetime.now() - start_time
        logger.info(f"Training duration: {duration}")

        final_model_path = save_dir / "curriculum_ppo_final.zip"
        model.save(str(final_model_path))
        curriculum_manager.save(save_dir / "curriculum_state.json")

        current_stage = curriculum_manager.current_stage
        if current_stage is None:
            raise RuntimeError("CurriculumManager lost its current stage")
        final_stage = current_stage.name
        terminal_mastery = assess_terminal_mastery(
            curriculum_manager,
            mastery_confirmation_episodes=effective_mastery_episodes,
        )
        provenance = build_curriculum_run_provenance(
            loaded_manifest=loaded_manifest,
            train_manifest=train_manifest,
            holdout_manifest=holdout_manifest,
            final_stage=final_stage,
            run_outcome=run_outcome,
            actual_timesteps=int(getattr(model, "num_timesteps", 0)),
            start_stage=run_start_stage,
            goal_based=goal_based_stopping,
            safety_cap_timesteps=total_timesteps,
            holdout_enabled=holdout_enabled,
            holdout_ratio=holdout_ratio,
            holdout_split_at=holdout_split_at,
            resolved_holdout_split_at=resolved_split_at,
            data_cutoff=data_cutoff,
            regime_start_at=recent_regime_start_time,
            regime_target_share=recent_regime_target_share,
            observation_schema=current_observation_schema(model, frame_stack=frame_stack),
            command=list(command) if command is not None else [sys.executable, *sys.argv],
            budget_preflight=budget_preflight,
            model_record=_file_record(final_model_path),
            terminal_mastery_confirmed=bool(terminal_mastery["confirmed"]),
            terminal_mastery_evidence=terminal_mastery,
        )
        provenance_path = save_dir / "curriculum_ppo_final.provenance.json"
        write_json_atomic(provenance, provenance_path)

        summary = {
            "status": provenance["status"],
            "accepted": False,
            "curriculum_complete": provenance["curriculum_complete"],
            "terminal_mastery": terminal_mastery,
            "total_timesteps": int(getattr(model, "num_timesteps", 0)),
            "final_stage": final_stage,
            "curriculum_progress": curriculum_manager.get_progress_report(),
            "run_ended_at": datetime.now().astimezone().isoformat(),
            "run_outcome": run_outcome,
            "seed": int(seed),
            "provenance": provenance_path.name,
            "holdout": {
                "enabled": bool(holdout_enabled),
                "ratio": float(holdout_ratio),
                "split_at_argument": holdout_split_at,
                "resolved_split_at": resolved_split_at,
                "min_bars": int(n_bars),
            },
        }
        write_json_atomic(summary, save_dir / "training_summary.json")

        logger.info(f"Final stage: {final_stage} | status: {provenance['status']}")
        logger.info(f"Run provenance: {provenance_path}")
        logger.info(f"Model saved to: {save_dir}")

        try:
            train_env.close()
        except Exception:
            pass

        try:
            holdout_validation_env.close()
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="PropFirm PPO Training (10/10) with Walk-Forward Optuna")


    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--curriculum", action="store_true", help="Run curriculum learning (progressive difficulty)")


    parser.add_argument("--trials", type=int, default=25, help="Optuna trials")
    parser.add_argument("--trial-timesteps", type=int, default=750_000, help="Steps per Optuna trial")
    parser.add_argument("--optuna-storage", type=str, default=None, help="Optuna storage URL (optional)")
    parser.add_argument("--study-name", type=str, default="propfirm_ppo", help="Optuna study name")
    parser.add_argument("--walk-forward-folds", type=int, default=2, help="Walk-forward folds (Optuna)")


    parser.add_argument(
        "--start-stage",
        type=str,
        default="EXPLORER",
        help="Starting curriculum stage (EXPLORER, EXPERIMENTER, TREND_STUDENT, SESSION_STUDENT, TIMING_STUDENT, INTEGRATOR, RISK_MANAGER, STRATEGIST, PROFESSIONAL, LIVE_READY)",
    )
    parser.add_argument("--resume-curriculum", type=str, default=None, help="Resume curriculum from state file")
    parser.add_argument("--load-model", type=str, default=None, help="Load model weights from checkpoint (.zip file)")
    parser.add_argument("--load-controllers", type=str, default=None, help="Load controller states from checkpoint (controllers_*.json file)")
    parser.add_argument("--load-metrics", type=str, default=None, help="Load episode history from checkpoint (metrics_*.json file)")
    parser.add_argument(
        "--goal-based",
        action="store_true",
        help="Train until LIVE_READY achieved (not fixed timesteps). Set --timesteps high as safety cap.",
    )
    parser.add_argument("--max-hours", type=float, default=None, help="Maximum training time in hours (goal-based stopping)")
    parser.add_argument("--plateau-episodes", type=int, default=500, help="Episodes without stage progress to trigger plateau stop (goal-based)")
    parser.add_argument("--mastery-episodes", type=int, default=100, help="Episodes at LIVE_READY to confirm completion (goal-based)")
    parser.add_argument(
        "--load-optuna-params",
        type=str,
        default=None,
        help="Load best hyperparams from Optuna JSON file (e.g., logs/optuna/best_params.json)",
    )

    parser.add_argument("--timesteps", type=int, default=100_000_000, help="Total training timesteps (safety cap - goal-based stops earlier)")
    default_n_envs = 2 if platform.system() == "Windows" else 4
    parser.add_argument("--n-envs", type=int, default=default_n_envs, help="Number of envs")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Fixed by default so runs are reproducible - the simulation "
             "clock removes wall-clock nondeterminism, and a time-based default would "
             "put it straight back. Pass -1 for a time-based seed.",
    )
    parser.add_argument("--test", action="store_true", help="Quick test mode")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.10)
    parser.add_argument("--vf-coef", type=float, default=0.7)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)

    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--value-hidden", type=int, default=256)

    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument(
        "--extra-data-dir",
        type=str,
        default=None,
        help="Optional broker-bar directory to append. Omitted by default so evaluation data is never merged implicitly.",
    )
    parser.add_argument(
        "--allow-synthetic-data",
        action="store_true",
        help="Explicitly permit synthetic fallback when file data is absent (development only).",
    )
    parser.add_argument(
        "--data-cutoff",
        type=str,
        default=None,
        help=(
            "Inclusive UTC cutoff applied to every timeframe after loading, e.g. "
            "2026-08-03T08:15:00Z. Use it with growing broker files so a command "
            "reconstructs the same dataset later."
        ),
    )
    parser.add_argument("--instruments", type=str, nargs="+", default=["XAUUSD"])

    parser.add_argument("--frame-stack", type=int, default=1, help="Frame stack. 1=disabled (recommended).")
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.15,
        help="Fraction of latest bars reserved for holdout evaluation (curriculum validation/stress gates). 0 disables.",
    )
    parser.add_argument(
        "--holdout-split-at",
        type=str,
        default=None,
        help=(
            "Explicit holdout start date, e.g. 2026-06-15. Preferred over "
            "--holdout-ratio: the dataset grows as broker bars are appended, so a "
            "fixed ratio walks the split backwards - at 0.15 over 114,399 bars it "
            "lands on 2025-11-07, ending training before the February 2026 regime "
            "change."
        ),
    )
    parser.add_argument(
        "--regime-start-at",
        type=str,
        default=None,
        help=(
            "UTC start of a materially new training regime. Used only for "
            "curriculum episode sampling and must be paired with --regime-target-share."
        ),
    )
    parser.add_argument(
        "--regime-target-share",
        type=float,
        default=0.0,
        help=(
            "Exact probability mass assigned to eligible training starts on/after "
            "--regime-start-at, after each stage's filters (0 disables)."
        ),
    )

    parser.add_argument("--entry-quality-threshold", type=float, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--risk-penalty-scale", type=float, default=None)

    parser.add_argument("--no-domain-randomization", action="store_true", help="Disable domain randomization")

    parser.add_argument("--no-dashboard", action="store_true", help="Disable real-time dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8765, help="Dashboard server port")

    args = parser.parse_args()

    if not np.isfinite(float(args.regime_target_share)) or not (0.0 <= float(args.regime_target_share) <= 1.0):
        parser.error("--regime-target-share must be between 0 and 1")
    if float(args.regime_target_share) > 0.0 and not args.regime_start_at:
        parser.error("--regime-target-share requires --regime-start-at")

    logger.info("=" * 70)
    logger.info("PropFirm PPO Training System (10/10)")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"MaskablePPO available: {MASKABLE_AVAILABLE}")
    logger.info(f"Optuna available: {OPTUNA_AVAILABLE}")
    logger.info(f"Curriculum available: {CURRICULUM_AVAILABLE}")
    logger.info(f"Dashboard available: {DASHBOARD_AVAILABLE}")
    logger.info("=" * 70)

    if args.optuna and not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not installed. Run: pip install optuna")

    logger.info("Loading market data...")
    data = load_market_data(
        data_dir=args.data_dir,
        instruments=args.instruments,
        min_bars=1000 if args.test else 5000,
        extra_dir=args.extra_data_dir,
        allow_synthetic=bool(args.allow_synthetic_data),
        data_cutoff=args.data_cutoff,
    )
    if not data:
        raise RuntimeError("No data available for training")

    loaded_manifest = build_dataset_manifest(
        data,
        role="loaded",
        boundary=args.data_cutoff,
    )
    write_dataset_manifest(loaded_manifest, Path("logs/training/dataset_manifest.json"))
    logger.info(
        "Dataset fingerprint: %s (manifest logs/training/dataset_manifest.json)",
        loaded_manifest["dataset_fingerprint"],
    )

    if args.test:
        args.timesteps = min(args.timesteps, 200_000)
        args.trial_timesteps = min(args.trial_timesteps, 80_000)
        args.checkpoint_freq = 10_000
        args.eval_freq = 10_000
        logger.info("TEST MODE enabled: reduced timesteps and frequencies")


    if args.seed == -1:
        args.seed = int(time.time() * 1000) % (2**31)
        logger.info(f"Using time-based random seed: {args.seed}")
    else:
        logger.info(f"Using fixed seed: {args.seed} (reproducible)")


    metrics_path_for_dashboard = "logs/training/live_metrics.json"
    if DASHBOARD_AVAILABLE and not args.no_dashboard and start_dashboard_server is not None:
        try:
            start_dashboard_server(
                port=args.dashboard_port,
                metrics_file=metrics_path_for_dashboard,
                background=True,
            )
        except Exception as e:
            logger.warning(f"Could not start dashboard: {e}")
    elif not args.no_dashboard and not DASHBOARD_AVAILABLE:
        logger.info("Dashboard not available. Install: pip install fastapi uvicorn websockets")

    config_overrides: Dict[str, Any] = {}
    if args.entry_quality_threshold is not None:
        config_overrides["entry_quality_threshold"] = float(args.entry_quality_threshold)
    if args.reward_scale is not None:
        config_overrides["reward_scale"] = float(args.reward_scale)
    if args.risk_penalty_scale is not None:
        config_overrides["risk_penalty_scale"] = float(args.risk_penalty_scale)

    config_overrides["domain_randomization_enabled"] = (not bool(args.no_domain_randomization))

    if args.optuna:
        optuna_data = data
        optuna_cut = None
        if args.holdout_split_at is not None or float(args.holdout_ratio) > 0.0:
            optuna_data, _reserved, optuna_cut = split_data_by_time(
                data,
                float(args.holdout_ratio),
                split_at=args.holdout_split_at,
                include_holdout_context=False,
            )
            logger.info("Optuna restricted to data strictly before holdout boundary %s", optuna_cut)
        run_optuna_optimization(
            data=optuna_data,
            n_trials=args.trials,
            timesteps_per_trial=args.trial_timesteps,
            n_envs=args.n_envs,
            n_eval_episodes=10,
            walk_forward_folds=max(1, int(args.walk_forward_folds)),
            frame_stack=max(1, int(args.frame_stack)),
            study_name=args.study_name,
            storage=args.optuna_storage,
            holdout_boundary=optuna_cut,
        )
    elif args.curriculum:
        if not CURRICULUM_AVAILABLE:
            raise RuntimeError("Curriculum system not available. Check curriculum imports.")

        if args.load_optuna_params:
            optuna_params_path = Path(args.load_optuna_params)
            if optuna_params_path.exists():
                with open(optuna_params_path, "r", encoding="utf-8") as f:
                    optuna_payload = json.load(f)

                expected_optuna_data = data
                expected_cut = None
                if args.holdout_split_at is not None or float(args.holdout_ratio) > 0.0:
                    expected_optuna_data, _reserved, expected_cut = split_data_by_time(
                        data,
                        float(args.holdout_ratio),
                        split_at=args.holdout_split_at,
                        include_holdout_context=False,
                    )
                expected_manifest = build_dataset_manifest(
                    expected_optuna_data,
                    role="optuna_development",
                    boundary=expected_cut,
                )
                optuna_best = validate_bound_optuna_params(optuna_payload, expected_manifest)
                expected_fingerprint = expected_manifest["dataset_fingerprint"]
                logger.info(
                    "Loaded bound Optuna params from %s (dataset %s)",
                    optuna_params_path,
                    expected_fingerprint,
                )

                param_map = {
                    "learning_rate": "lr",
                    "batch_size": "batch_size",
                    "n_steps": "n_steps",
                    "n_epochs": "n_epochs",
                    "gamma": "gamma",
                    "gae_lambda": "gae_lambda",
                    "clip_range": "clip_range",
                    "vf_coef": "vf_coef",
                    "max_grad_norm": "max_grad_norm",
                    "target_kl": "target_kl",
                    "policy_hidden": "policy_hidden",
                    "value_hidden": "value_hidden",
                }
                for optuna_key, args_key in param_map.items():
                    if optuna_key in optuna_best:
                        old_val = getattr(args, args_key)
                        new_val = optuna_best[optuna_key]
                        setattr(args, args_key, new_val)
                        logger.info(f"  Optuna override: {args_key} {old_val} -> {new_val}")
                if "ent_coef" in optuna_best:
                    logger.info(f"  Optuna ent_coef={optuna_best['ent_coef']} ignored (curriculum uses adaptive entropy)")
            else:
                logger.warning(f"Optuna params file not found: {optuna_params_path}")

        logger.info("=" * 70)
        logger.info("CURRICULUM LEARNING MODE")
        if args.goal_based:
            logger.info("GOAL-BASED STOPPING ENABLED")
            terminal_episode_floor = 0
            if get_stage_config is not None and CurriculumStage is not None:
                terminal_episode_floor = int(
                    get_stage_config(CurriculumStage.LIVE_READY).competence.min_episodes
                )
            effective_mastery_episodes = max(
                int(args.mastery_episodes),
                terminal_episode_floor,
            )
            logger.info(
                f"Goal-stop LIVE_READY episode floor: {effective_mastery_episodes:,}"
            )
            logger.info(
                "Completion provenance additionally requires terminal timesteps plus recorded "
                "development validation and stress gates"
            )
            if effective_mastery_episodes != int(args.mastery_episodes):
                logger.info(
                    f"Requested --mastery-episodes={int(args.mastery_episodes):,} is below "
                    f"the terminal competence floor of {terminal_episode_floor:,}"
                )
            logger.info(f"Safety cap: {args.timesteps:,} timesteps")
            if args.max_hours:
                logger.info(f"Max hours: {args.max_hours}")
            logger.info(f"Plateau threshold: {args.plateau_episodes} episodes")
        logger.info("=" * 70)

        train_curriculum_agent(
            data=data,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_hidden=args.policy_hidden,
            value_hidden=args.value_hidden,
            checkpoint_freq=args.checkpoint_freq,
            start_stage=args.start_stage,
            resume_path=args.resume_curriculum,
            load_model_path=args.load_model,
            load_controllers_path=args.load_controllers,
            load_metrics_path=args.load_metrics,
            frame_stack=max(1, int(args.frame_stack)),
            goal_based_stopping=args.goal_based,
            max_hours=args.max_hours,
            plateau_threshold_episodes=args.plateau_episodes,
            mastery_confirmation_episodes=args.mastery_episodes,
            seed=args.seed,
            holdout_ratio=args.holdout_ratio,
            holdout_split_at=args.holdout_split_at,
            recent_regime_start_time=args.regime_start_at,
            recent_regime_target_share=args.regime_target_share,
            data_cutoff=args.data_cutoff,
            loaded_dataset_manifest=loaded_manifest,
            command=[sys.executable, *sys.argv],
        )
    else:
        train_prop_firm_agent(
            data=data,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_hidden=args.policy_hidden,
            value_hidden=args.value_hidden,
            checkpoint_freq=args.checkpoint_freq,
            eval_freq=args.eval_freq,
            pretrained_path=args.pretrained,
            config_overrides=config_overrides,
            frame_stack=max(1, int(args.frame_stack)),
            seed=args.seed,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
