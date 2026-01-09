"""
train/controllers/health_watchdog.py

Training Health Watchdog - monitors training metrics and alerts on issues.
Upgraded: robust slopes via regression, warmup gating, optional normalized entropy,
and cleaner alert semantics.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional, Tuple, Deque, List

import math
import numpy as np

logger = logging.getLogger(__name__)


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _slope(values: List[float]) -> float:
    """
    Least-squares slope of values against index.
    Returns 0.0 if insufficient data or non-finite values.
    """
    if len(values) < 3:
        return 0.0
    y = np.array(values, dtype=float)
    if not np.all(np.isfinite(y)):
        return 0.0
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def _normalize_entropy(raw_entropy: float, n_valid_actions: int, min_valid_floor: int = 4) -> float:
    """
    Normalize entropy to [0,1] using log(K_eff). Defensive against bad K.
    """
    if not _is_finite(raw_entropy):
        return 0.5
    k = max(min_valid_floor, int(n_valid_actions))
    max_h = math.log(max(2, k))
    if max_h <= 0:
        return 0.5
    return max(0.0, min(1.0, float(raw_entropy) / max_h))


class TrainingHealthWatchdog:
    """
    Monitors training health metrics and alerts/stops if critical issues arise.

    Signals:
    - Explained variance collapse (critic not learning)
    - Sustained reward decline (policy degradation)
    - Win rate stuck near breakeven (no learning)
    - Entropy collapse/explosion (degenerate policy)
    """

    def __init__(
        self,
        ev_critical_threshold: float = -0.5,
        ev_warning_threshold: float = 0.05,
        ev_consecutive_failures: int = 5,
        reward_slope_threshold: float = -0.02,   # slope per check step (regression on history window)
        reward_consecutive_declines: int = 10,
        winrate_stuck_range: tuple = (0.45, 0.55),
        winrate_stuck_checks: int = 200,
        check_interval_steps: int = 20_000,
        auto_stop: bool = False,
        warmup_checks: int = 20,  # do not emit severe alerts before enough history
        # Entropy thresholds:
        entropy_low_raw: float = 0.10,
        entropy_high_raw: float = 2.00,
        # Optional normalized entropy thresholds (if mask info supplied)
        entropy_low_norm: float = 0.08,
        entropy_high_norm: float = 0.85,
    ):
        self.ev_critical = float(ev_critical_threshold)
        self.ev_warning = float(ev_warning_threshold)
        self.ev_failures_needed = int(max(1, ev_consecutive_failures))

        self.reward_slope_threshold = float(reward_slope_threshold)
        self.reward_declines_needed = int(max(1, reward_consecutive_declines))

        self.winrate_range = winrate_stuck_range
        self.winrate_stuck_needed = int(max(1, winrate_stuck_checks))

        self.check_interval = int(max(1, check_interval_steps))
        self.auto_stop = bool(auto_stop)
        self.warmup_checks = int(max(0, warmup_checks))

        self.entropy_low_raw = float(entropy_low_raw)
        self.entropy_high_raw = float(entropy_high_raw)
        self.entropy_low_norm = float(entropy_low_norm)
        self.entropy_high_norm = float(entropy_high_norm)

        # History
        self._ev_history: Deque[float] = deque(maxlen=30)
        self._reward_history: Deque[float] = deque(maxlen=80)
        self._winrate_history: Deque[float] = deque(maxlen=200)

        self._last_check_step = 0
        self._checks_done = 0

        self._consecutive_ev_failures = 0
        self._consecutive_reward_declines = 0
        self._winrate_stuck_count = 0

    def check(
        self,
        explained_variance: float,
        mean_reward: float,
        win_rate: float,
        entropy: float,
        timestep: int,
        # Optional mask-aware entropy normalization:
        n_valid_actions: Optional[int] = None,
        n_actions: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns:
            (should_stop, reason)
        """
        try:
            timestep = int(timestep)
        except Exception:
            timestep = 0

        if timestep - self._last_check_step < self.check_interval:
            return False, None
        self._last_check_step = timestep
        self._checks_done += 1

        # Update histories (store even if values are non-finite; slope funcs will guard)
        self._ev_history.append(float(explained_variance) if _is_finite(explained_variance) else float("nan"))
        self._reward_history.append(float(mean_reward) if _is_finite(mean_reward) else float("nan"))
        self._winrate_history.append(float(win_rate) if _is_finite(win_rate) else float("nan"))

        alerts: List[str] = []
        should_stop = False
        in_warmup = self._checks_done < self.warmup_checks

        # --- Check 1: Explained Variance ---
        if _is_finite(explained_variance):
            if explained_variance < self.ev_critical:
                self._consecutive_ev_failures += 1
                alerts.append(
                    f"CRITICAL: explained_variance={explained_variance:.3f} "
                    f"(failures {self._consecutive_ev_failures}/{self.ev_failures_needed})"
                )
                if self._consecutive_ev_failures >= self.ev_failures_needed and not in_warmup:
                    should_stop = self.auto_stop
                    alerts.append("CRITICAL: critic collapse suspected (consider LR, n_steps, reward scaling).")
            elif explained_variance < self.ev_warning:
                alerts.append(f"WARNING: explained_variance={explained_variance:.3f} (low)")
                self._consecutive_ev_failures = max(0, self._consecutive_ev_failures - 1)
            else:
                self._consecutive_ev_failures = 0
        else:
            alerts.append("WARNING: explained_variance is non-finite")

        # --- Check 2: Reward trend (regression slope) ---
        recent_rewards = [x for x in list(self._reward_history)[-40:] if _is_finite(x)]
        if len(recent_rewards) >= 10:
            s = _slope(recent_rewards)
            if s < self.reward_slope_threshold:
                self._consecutive_reward_declines += 1
                alerts.append(
                    f"WARNING: mean_reward trend declining (slope={s:+.4f}, "
                    f"count {self._consecutive_reward_declines}/{self.reward_declines_needed})"
                )
                if self._consecutive_reward_declines >= self.reward_declines_needed and not in_warmup:
                    alerts.append("WARNING: sustained reward degradation; policy may be regressing.")
            else:
                self._consecutive_reward_declines = 0

        # --- Check 3: Win rate stuck ---
        if _is_finite(win_rate) and self.winrate_range[0] <= win_rate <= self.winrate_range[1]:
            self._winrate_stuck_count += 1
            if self._winrate_stuck_count >= self.winrate_stuck_needed and not in_warmup:
                alerts.append(
                    f"WARNING: win_rate stagnant at {win_rate:.1%} "
                    f"for {self._winrate_stuck_count} checks"
                )
        else:
            self._winrate_stuck_count = 0

        # --- Check 4: Entropy ---
        # Prefer normalized entropy when mask info is available (more portable).
        if n_valid_actions is not None:
            k = int(max(1, n_valid_actions))
            norm_h = _normalize_entropy(entropy, k)
            if norm_h < self.entropy_low_norm:
                alerts.append(f"CRITICAL: entropy collapsed (H_norm={norm_h:.3f}, H_raw={entropy:.3f}, K={k})")
                if not in_warmup:
                    should_stop = should_stop or self.auto_stop
            elif norm_h > self.entropy_high_norm:
                alerts.append(f"WARNING: entropy very high (H_norm={norm_h:.3f}, H_raw={entropy:.3f}, K={k})")
        else:
            if _is_finite(entropy):
                if entropy < self.entropy_low_raw:
                    alerts.append(f"CRITICAL: entropy collapsed (H_raw={entropy:.3f})")
                    if not in_warmup:
                        should_stop = should_stop or self.auto_stop
                elif entropy > self.entropy_high_raw:
                    alerts.append(f"WARNING: entropy very high (H_raw={entropy:.3f})")
            else:
                alerts.append("WARNING: entropy is non-finite")

        # Logging
        for alert in alerts:
            if alert.startswith("CRITICAL"):
                logger.error(alert)
            else:
                logger.warning(alert)

        reason = "; ".join(alerts) if alerts else None
        return should_stop, reason
