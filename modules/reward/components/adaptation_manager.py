# modules/reward/components/adaptation_manager.py
"""
Adaptation Manager for Reward System (Hardened)
- Thread-safe internal mutations (local lock, no lock held across awaits)
- Bounded memory for parameter history (deque with maxlen)
- Defensive reads from shared state (tolerant to missing fields)
- Numeric clamps for all adapted params (stable ranges)
- Lightweight logging helpers (won’t break hot path)
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from collections import deque
import threading
import time
import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


class AdaptationManager:
    """
    Manages adaptive learning and parameter tuning

    Features:
    - Dynamic parameter adjustment
    - Performance-based adaptation
    - Regime-specific tuning
    - Learning rate management
    - Thread-safe state updates
    """

    # Stable operating ranges (can be widened if needed)
    PENALTY_SCALE_MIN = 0.5
    PENALTY_SCALE_MAX = 2.0
    ACTIVITY_TH_MIN = 0.5
    ACTIVITY_TH_MAX = 2.0
    REGIME_SENS_MIN = 0.7
    REGIME_SENS_MAX = 1.5
    RISK_TOL_MIN = 0.5
    RISK_TOL_MAX = 1.5
    CONFIDENCE_MIN = 0.1
    CONFIDENCE_MAX = 1.0

    PARAM_HISTORY_MAXLEN = 200  # bounded memory
    MOMENTUM_WINDOW = 5         # last-N snapshots to estimate momentum

    def __init__(
        self,
        config: Any,
        state: Any,
        logger: Any,
        debug_manager: Any
    ):
        """Initialize adaptation manager"""

        self.cfg = config
        self.state = state
        self.logger = logger
        self.debug_manager = debug_manager

        # Adaptation tracking (thread-safe via _lock)
        self._lock = threading.RLock()
        self.adaptation_count: int = 0
        self.parameter_history: deque = deque(maxlen=self.PARAM_HISTORY_MAXLEN)
        self.adaptation_effectiveness: float = 0.5

    # ─────────────────────────────────────────────────────────────
    # Public async API (called from the main async process)
    # ─────────────────────────────────────────────────────────────
    async def update_adaptive_learning(
        self,
        reward_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update adaptive learning parameters (async-safe, no locks across awaits)"""

        with self._lock:
            self.adaptation_count += 1

        try:
            # Each step computes a new value first, then applies it under lock
            await self._adapt_penalty_scaling()
            await self._adapt_activity_threshold()
            await self._adapt_regime_sensitivity()
            await self._adapt_risk_tolerance()
            await self._update_adaptation_confidence()

            learning_momentum = self._calculate_learning_momentum()
            with self._lock:
                self.state.adaptive_params['learning_momentum'] = learning_momentum
                self._record_adaptation_locked()

                snapshot_params = self.state.adaptive_params.copy()
                reward_quality = float(getattr(self.state, "reward_quality", 0.5))

            return {
                'adaptive_learning': {
                    'adaptive_params': snapshot_params,
                    'reward_quality': reward_quality,
                    'learning_effectiveness': snapshot_params.get('adaptation_confidence', 0.5),
                    'learning_momentum': learning_momentum,
                    'adaptation_count': self.adaptation_count
                }
            }

        except Exception as e:
            self._dbg_warn(f"Adaptive learning update failed: {e}", tag="ADAPTATION_ERROR")
            with self._lock:
                snapshot_params = self.state.adaptive_params.copy()
            return {
                'adaptive_learning': {
                    'adaptive_params': snapshot_params,
                    'error': str(e)
                }
            }

    # ─────────────────────────────────────────────────────────────
    # Parameter-specific adaptations (compute → apply under lock)
    # ─────────────────────────────────────────────────────────────
    async def _adapt_penalty_scaling(self) -> None:
        """Adapt dynamic penalty scaling based on recent reward performance"""
        rewards = list(getattr(self.state, "reward_history", []))[-20:]
        if len(rewards) < 20:
            return

        avg_reward = float(np.mean(rewards)) if len(rewards) else 0.0
        lr = float(getattr(self.cfg, "adaptive_learning_rate", 0.01))

        with self._lock:
            current = float(self.state.adaptive_params.get('dynamic_penalty_scaling', 1.0))

        if avg_reward < -0.5:
            new_value = current * (1 + lr)
        elif avg_reward > 0.5:
            new_value = current * (1 - lr * 0.5)
        else:
            # Gentle relaxation toward baseline 1.0
            new_value = current * 0.95 + 1.0 * 0.05

        new_value = _clamp(new_value, self.PENALTY_SCALE_MIN, self.PENALTY_SCALE_MAX)
        with self._lock:
            self.state.adaptive_params['dynamic_penalty_scaling'] = new_value
        if abs(new_value - current) > 0.01:
            self._dbg_note(f"Penalty scaling: {current:.3f} → {new_value:.3f}")

    async def _adapt_activity_threshold(self) -> None:
        """Adapt activity threshold based on trading frequency"""
        trade_counts = list(getattr(self.state, "trade_count_history", []))
        if len(trade_counts) < 10:
            return

        avg_activity = float(np.mean(trade_counts)) if trade_counts else 0.0
        with self._lock:
            current = float(self.state.adaptive_params.get('activity_threshold', 1.0))

        if avg_activity < 0.3:
            new_value = current * 1.05
        elif avg_activity > 3.0:
            new_value = current * 0.98
        else:
            new_value = current

        new_value = _clamp(new_value, self.ACTIVITY_TH_MIN, self.ACTIVITY_TH_MAX)
        with self._lock:
            self.state.adaptive_params['activity_threshold'] = new_value

    async def _adapt_regime_sensitivity(self) -> None:
        """Adapt regime sensitivity based on regime performance variance"""
        regime_perf = getattr(self.state, "regime_performance", {}) or {}
        if not isinstance(regime_perf, dict) or len(regime_perf) < 2:
            return

        regime_rewards = []
        for _, data in regime_perf.items():
            rewards = data.get('rewards', [])
            # rewards is backed by a deque in RewardState; convert to list
            # before slicing to avoid TypeError: sequence index must be integer, not 'slice'
            rewards_list = list(rewards) if rewards is not None else []
            if rewards_list:
                regime_rewards.append(float(np.mean(rewards_list[-10:])))

        if not regime_rewards:
            return

        variance = float(np.var(regime_rewards))
        with self._lock:
            current = float(self.state.adaptive_params.get('regime_sensitivity', 1.0))

        if variance > 0.1:
            new_value = current * 1.001
        else:
            new_value = current * 0.9999

        new_value = _clamp(new_value, self.REGIME_SENS_MIN, self.REGIME_SENS_MAX)
        with self._lock:
            self.state.adaptive_params['regime_sensitivity'] = new_value

    async def _adapt_risk_tolerance(self) -> None:
        """Adapt risk tolerance based on drawdown and win rate"""
        risk_metrics = getattr(self.state, "last_risk_metrics", None) or {}
        drawdown = float(risk_metrics.get('current_drawdown', risk_metrics.get('drawdown', 0.0)) or 0.0)
        win_rate = float(getattr(self.state, "win_rate", 0.0))

        with self._lock:
            current = float(self.state.adaptive_params.get('risk_tolerance', 1.0))

        if drawdown > 0.15:
            new_value = current * 0.95
        elif drawdown < 0.05 and win_rate > 0.6:
            new_value = current * 1.02
        else:
            new_value = current * 0.98 + 1.0 * 0.02

        new_value = _clamp(new_value, self.RISK_TOL_MIN, self.RISK_TOL_MAX)
        with self._lock:
            self.state.adaptive_params['risk_tolerance'] = new_value

    async def _update_adaptation_confidence(self) -> None:
        """Update confidence in adaptation effectiveness"""
        reward_quality = float(getattr(self.state, "reward_quality", 0.5))
        with self._lock:
            confidence = float(self.state.adaptive_params.get('adaptation_confidence', 0.5))

        if reward_quality > 0.7:
            new_conf = confidence * 1.01
        elif reward_quality < 0.3:
            new_conf = confidence * 0.99
        else:
            new_conf = confidence

        new_conf = _clamp(new_conf, self.CONFIDENCE_MIN, self.CONFIDENCE_MAX)
        with self._lock:
            self.state.adaptive_params['adaptation_confidence'] = new_conf

    # ─────────────────────────────────────────────────────────────
    # Momentum & Effectiveness
    # ─────────────────────────────────────────────────────────────
    def _calculate_learning_momentum(self) -> float:
        """Estimate learning momentum from recent param changes"""
        with self._lock:
            if len(self.parameter_history) < self.MOMENTUM_WINDOW:
                return 0.0
            recent = list(self.parameter_history)[-self.MOMENTUM_WINDOW:]

        # Change magnitudes between consecutive snapshots
        changes: list[float] = []
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            curr = recent[i]
            # Use intersection of keys; ignore timestamps
            keys = [k for k in prev.keys() if k in curr and k not in ("timestamp", "reward_quality")]
            change = sum(abs(float(curr[k]) - float(prev[k])) for k in keys)
            changes.append(change)

        if changes:
            stability = 1.0 / (1.0 + float(np.mean(changes)))
            with self._lock:
                conf = float(self.state.adaptive_params.get('adaptation_confidence', 0.5))
            momentum = stability * conf
            return float(np.clip(momentum, 0.0, 1.0))
        return 0.5

    def _record_adaptation_locked(self) -> None:
        """Record current adaptation state (caller must hold _lock)"""
        snapshot = self.state.adaptive_params.copy()
        snapshot['timestamp'] = time.monotonic()
        snapshot['reward_quality'] = float(getattr(self.state, "reward_quality", 0.5))
        self.parameter_history.append(snapshot)

    # ─────────────────────────────────────────────────────────────
    # Monitoring thread hook (sync, called under main state lock)
    # ─────────────────────────────────────────────────────────────
    def adapt_parameters(self) -> None:
        """Periodic analysis invoked by the monitoring thread"""
        try:
            rewards = getattr(self.state, "reward_history", [])
            if len(rewards) < 10:
                return
            self._analyze_parameter_effectiveness()
            # Light heartbeat every ~50 updates of async path; here we log
            with self._lock:
                count = self.adaptation_count
                eff = self.adaptation_effectiveness
                conf = float(self.state.adaptive_params.get('adaptation_confidence', 0.5))
                pscale = float(self.state.adaptive_params.get('dynamic_penalty_scaling', 1.0))
            if count and (count % 50 == 0):
                self.logger.info(
                    f"Adaptation Status: Count={count}, "
                    f"Effectiveness={eff:.2f}, "
                    f"Confidence={conf:.2f}, "
                    f"Penalty Scale={pscale:.2f}"
                )
        except Exception as e:
            self.logger.error(f"Parameter adaptation failed: {e}")

    def _analyze_parameter_effectiveness(self) -> None:
        """Analyze effectiveness of adaptations over time"""
        with self._lock:
            hist = list(self.parameter_history)
        if len(hist) < 10:
            return

        mid = len(hist) // 2
        early = hist[:mid]
        late = hist[mid:]

        early_q = np.mean([float(p.get('reward_quality', 0.5)) for p in early]) if early else 0.5
        late_q = np.mean([float(p.get('reward_quality', 0.5)) for p in late]) if late else early_q

        with self._lock:
            if late_q > early_q:
                self.adaptation_effectiveness = min(1.0, self.adaptation_effectiveness * 1.1)
            else:
                self.adaptation_effectiveness = max(0.0, self.adaptation_effectiveness * 0.9)

    # ─────────────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset adaptation state and return adaptive params to defaults"""
        with self._lock:
            self.adaptation_count = 0
            self.parameter_history.clear()
            self.adaptation_effectiveness = 0.5
            # Reset only known keys to avoid clobbering extended params
            defaults = {
                'dynamic_penalty_scaling': 1.0,
                'regime_sensitivity': 1.0,
                'activity_threshold': 1.0,
                'risk_tolerance': 1.0,
                'learning_momentum': 0.0,
                'adaptation_confidence': 0.5,
            }
            self.state.adaptive_params.update(defaults)

    def get_state(self) -> Dict[str, Any]:
        """Get adaptation state for persistence"""
        with self._lock:
            return {
                'adaptation_count': self.adaptation_count,
                'adaptation_effectiveness': self.adaptation_effectiveness,
                'adaptive_params': self.state.adaptive_params.copy(),
                'parameter_history_size': len(self.parameter_history)
            }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore adaptation state from persistence"""
        with self._lock:
            self.adaptation_count = int(state.get('adaptation_count', 0))
            self.adaptation_effectiveness = float(state.get('adaptation_effectiveness', 0.5))
            if 'adaptive_params' in state:
                self.state.adaptive_params.update(state['adaptive_params'])

    # ─────────────────────────────────────────────────────────────
    # Logging helpers (never raise)
    # ─────────────────────────────────────────────────────────────
    def _dbg_note(self, msg: str) -> None:
        try:
            if getattr(self.debug_manager, "enabled", False):
                # Try debug manager hook; fall back to logger
                if hasattr(self.debug_manager, "_log"):
                    self.debug_manager._log("DEBUG", msg, "ADAPTATION")  # type: ignore[attr-defined]
                else:
                    self.logger.debug(msg)
        except Exception:
            try:
                self.logger.debug(msg)
            except Exception:
                pass

    def _dbg_warn(self, msg: str, tag: str = "WARN") -> None:
        try:
            if getattr(self.debug_manager, "enabled", False) and hasattr(self.debug_manager, "log_error"):
                self.debug_manager.log_error(tag, RuntimeError(msg))
            else:
                self.logger.warning(msg)
        except Exception:
            try:
                self.logger.warning(msg)
            except Exception:
                pass
