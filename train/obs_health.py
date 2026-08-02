"""Observation health, shared by every training callback.

Both VecEpisodeTradingCallback and CurriculumTrainingCallback write
logs/training/live_metrics.json. In curriculum mode the curriculum callback
writes last and clobbers the episode callback's payload - which silently
dropped the observation-health block from exactly the runs it matters most for.

Keeping the computation here means one implementation and one place to change
it, rather than the duplication that caused the original train/live divergence.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

# Enough samples to distinguish a genuinely constant dimension from one that
# simply has not moved yet.
MIN_SAMPLES = 32
SAMPLE_CAPACITY = 512


class ObservationHealthTracker:
    """Rolling sample of raw observations plus the statistics that expose a
    dead one.

    The dashboard rendered normal reward curves through an outage where every
    observation was np.zeros(90). Nothing on screen distinguished a blind agent
    from a learning one, and the fault was found by reading source rather than
    by monitoring.
    """

    def __init__(self, capacity: int = SAMPLE_CAPACITY) -> None:
        self._samples: Deque[np.ndarray] = deque(maxlen=capacity)

    def observe(self, obs: Any) -> None:
        """Record one observation from a rollout. Accepts a batch or a vector."""
        if obs is None:
            return
        arr = np.asarray(obs, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > 0:
            self._samples.append(arr[0].copy())
        elif arr.ndim == 1 and arr.size:
            self._samples.append(arr.copy())

    def reset(self) -> None:
        self._samples.clear()

    def report(self) -> Dict[str, Any]:
        from modules.meta.ppo_observation_builder import (
            FEATURE_GROUPS,
            PPO_OBS_FEATURE_NAMES,
            PPO_OBS_SIZE,
            PPO_OBS_VERSION,
        )

        health: Dict[str, Any] = {
            "schema_version": PPO_OBS_VERSION,
            "schema_size": int(PPO_OBS_SIZE),
            # Identity, so a builder/checkpoint mismatch is visible rather than
            # inferred from downstream weirdness.
            "schema_hash": hashlib.sha256(
                "|".join(PPO_OBS_FEATURE_NAMES).encode("utf-8")
            ).hexdigest()[:12],
            "samples": len(self._samples),
        }

        if len(self._samples) < MIN_SAMPLES:
            health["status"] = "warming_up"
            health["dead_dims"] = 0
            health["nan_count"] = 0
            health["blocks"] = {}
            return health

        stacked = np.asarray(self._samples, dtype=np.float64)
        std = stacked.std(axis=0)
        dead_idx = np.flatnonzero(std < 1e-9)

        health["dead_dims"] = int(dead_idx.size)
        health["dead_dim_names"] = [
            PPO_OBS_FEATURE_NAMES[i]
            for i in dead_idx[:8]
            if i < len(PPO_OBS_FEATURE_NAMES)
        ]
        health["nan_count"] = int(np.isnan(stacked).sum())
        health["mean_abs"] = float(np.abs(stacked).mean())
        health["blocks"] = {
            name: {
                "dims": end - start,
                "dead": int((std[start:end] < 1e-9).sum()),
                "std": float(std[start:end].mean()),
            }
            for name, (start, end) in FEATURE_GROUPS.items()
        }

        if float(stacked.std()) < 1e-9:
            health["status"] = "blind"
            health["alert"] = "OBSERVATION IS CONSTANT - the agent cannot see the market"
        elif health["nan_count"] > 0:
            health["status"] = "bad"
            health["alert"] = f"{health['nan_count']} NaN values in observation"
        elif dead_idx.size > int(PPO_OBS_SIZE) // 2:
            health["status"] = "bad"
            health["alert"] = f"{dead_idx.size}/{PPO_OBS_SIZE} observation dims are constant"
        elif dead_idx.size > 0:
            health["status"] = "ok"
            health["alert"] = (
                f"{dead_idx.size} constant dim(s): "
                f"{', '.join(health['dead_dim_names'][:3])}"
            )
        else:
            health["status"] = "good"
        return health


def trading_frequency(
    mean_trades: float,
    mean_episode_len: float,
    stage_target_per_1k: Optional[float] = None,
    bars_per_day: float = 96.0,
) -> Dict[str, Any]:
    """Trade frequency in the unit a trader reasons in, against the stage target.

    The previous model "became a junkie which traded almost always" and it was
    only noticed after the run. 96 M15 bars is one 24h day. The gate comes from
    CURRICULUM_PLAN 12.5: reject above 3x the stage target.
    """
    per_day = (mean_trades / max(mean_episode_len, 1.0)) * bars_per_day
    target_per_day = (float(stage_target_per_1k or 0.0) / 1000.0) * bars_per_day

    if target_per_day > 0.0:
        ratio = per_day / target_per_day
        status = "bad" if ratio > 3.0 else "ok" if ratio > 1.5 else "good"
    else:
        ratio = 0.0
        status = "good" if 0.2 <= per_day <= 5.0 else "bad"

    return {
        "trades_per_day": per_day,
        "stage_target_trades_per_day": target_per_day,
        "overtrade_ratio": ratio,
        "status": status,
    }
