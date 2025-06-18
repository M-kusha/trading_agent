import os
import numpy as np
import logging
from collections import deque
from typing import List, Dict, Any, Optional
from ..core.core import Module
import datetime

def utcnow() -> str:
    return datetime.datetime.utcnow().isoformat()

class RiskAdjustedReward(Module):
    """
    Evolutionary, audit-compliant risk and reward shaper:
      - Evolves reward weights for regime bonuses, risk, drawdown, tail, mistake memory.
      - Tracks recent rewards for diagnostics and obs.
      - Writes a full audit-trail of every shaping decision.
    """

    def __init__(self, initial_balance: float, env=None, debug: bool = False, history: int = 50):
        self.initial_balance = initial_balance
        self.env   = env
        self.debug = debug

        # Reward‐shaping weights (all evolvable)
        self.regime_weights = np.array([0.5, 0.3, 0.2], np.float32)
        self.dd_pen_weight = 10.0
        self.risk_pen_weight = 0.3
        self.tail_pen_weight = 1.0
        self.mistake_pen_weight = 0.5
        self.no_trade_penalty_weight = 0.1

        # Rolling history of shaped rewards
        self._history = deque(maxlen=history)
        self._last = 0.0
        self._last_reason = ""

        # Audit trail
        self.audit_trail: List[Dict[str, Any]] = []
        self._audit_log_size = history
                # ── ensure log directory exists ─────────────────────────────
        log_dir = os.path.join("logs", "reward")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "risk_adjusted_reward.log")

        # Logger setup
        self.logger = logging.getLogger("RiskAdjustedReward")
        if not self.logger.handlers:
            handler = logging.FileHandler("logs/reward/risk_adjusted_reward.log")
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def _record_audit(self, details: Dict[str, Any]) -> None:
        details["timestamp"] = utcnow()
        self.audit_trail.append(details)
        if len(self.audit_trail) > self._audit_log_size:
            self.audit_trail = self.audit_trail[-self._audit_log_size:]
        self.logger.debug(f"[AUDIT] {details}")

    def get_last_audit(self) -> Dict[str, Any]:
        return self.audit_trail[-1] if self.audit_trail else {}

    def get_audit_trail(self, n: int = 20) -> List[Dict[str, Any]]:
        return self.audit_trail[-n:]

    # ================= Evolution methods =================

    def mutate(self, std: float = 0.1) -> None:
        self.regime_weights += np.random.normal(0, std, size=3)
        self.regime_weights = np.clip(self.regime_weights, -1.0, 2.0)
        self.dd_pen_weight += np.random.normal(0, std * 10)
        self.dd_pen_weight = np.clip(self.dd_pen_weight, 0.0, 50.0)
        self.risk_pen_weight += np.random.normal(0, std)
        self.risk_pen_weight = np.clip(self.risk_pen_weight, 0.0, 5.0)
        self.tail_pen_weight += np.random.normal(0, std)
        self.tail_pen_weight = np.clip(self.tail_pen_weight, 0.0, 5.0)
        self.mistake_pen_weight += np.random.normal(0, std)
        self.mistake_pen_weight = np.clip(self.mistake_pen_weight, 0.0, 5.0)
        self.no_trade_penalty_weight += np.random.normal(0, std)
        self.no_trade_penalty_weight = np.clip(self.no_trade_penalty_weight, 0.0, 2.0)
        if self.debug:
            self.logger.debug(f"Mutated weights: {self.get_weights()}")

    def crossover(self, other: "RiskAdjustedReward") -> "RiskAdjustedReward":
        child = self.__class__(self.initial_balance, env=self.env, debug=self.debug, history=self._history.maxlen)
        for attr in [
            "regime_weights",
            "dd_pen_weight",
            "risk_pen_weight",
            "tail_pen_weight",
            "mistake_pen_weight",
            "no_trade_penalty_weight"
        ]:
            if np.random.rand() > 0.5:
                setattr(child, attr, getattr(other, attr).copy() if isinstance(getattr(other, attr), np.ndarray) else getattr(other, attr))
        if self.debug:
            self.logger.debug("Crossover complete.")
        return child

    def get_weights(self) -> Dict[str, Any]:
        return dict(
            regime_weights=self.regime_weights.copy(),
            dd_pen_weight=self.dd_pen_weight,
            risk_pen_weight=self.risk_pen_weight,
            tail_pen_weight=self.tail_pen_weight,
            mistake_pen_weight=self.mistake_pen_weight,
            no_trade_penalty_weight=self.no_trade_penalty_weight,
        )

    # ================= Interface =================

    def reset(self) -> None:
        self._history.clear()
        self._last = 0.0
        self._last_reason = ""

    def step(
        self,
        current_balance: float,
        trades: List[dict],
        current_drawdown: float,
        regime_onehot: np.ndarray,
        actions: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        # Prepare the audit record skeleton
        rec: Dict[str, Any] = {
            "reason": None,
            "pnl": None,
            "drawdown_penalty": None,
            "risk_penalty": None,
            "tail_penalty": None,
            "regime_bonus": None,
            "mistake_memory_penalty": None,
            "no_trade_penalty": None,
            "raw_actions_norm": float(np.sqrt((actions**2).mean())),
        }

        # No-trade penalty
        if not trades:
            penalty = -self.no_trade_penalty_weight * rec["raw_actions_norm"]
            rec.update(reason="no-trade", no_trade_penalty=penalty)
            reward = penalty

        else:
            pnl = sum(t["pnl"] for t in trades)
            rets = np.array([t["pnl"] for t in trades], np.float32)
            var = np.percentile(rets, 5)
            cvar = rets[rets <= var].mean() if np.any(rets <= var) else var
            tail_pen = -self.tail_pen_weight * (0.5 * var + 0.5 * cvar)

            dd_pen = current_drawdown * self.dd_pen_weight
            risk_pen = rec["raw_actions_norm"] * self.risk_pen_weight
            regime_bonus = float(regime_onehot.dot(self.regime_weights))

            mm_pen = 0.0
            if self.env and hasattr(self.env, "mistake_memory"):
                mm_pen = float(self.env.mistake_memory.get_observation_components()[0]) * self.mistake_pen_weight

            reward = pnl - dd_pen - risk_pen - tail_pen - mm_pen + regime_bonus

            rec.update(
                reason="trade",
                pnl=pnl,
                tail_penalty=tail_pen,
                drawdown_penalty=dd_pen,
                risk_penalty=risk_pen,
                regime_bonus=regime_bonus,
                mistake_memory_penalty=mm_pen
            )

        # Populate info dict
        if info is not None:
            info["shaped_reward"] = reward
            info["reward_components"] = rec.copy()

        # Finalize
        self._last = reward
        self._last_reason = rec["reason"]
        self._history.append(reward)
        self._record_audit(rec)

        if self.debug:
            self.logger.debug(f"Computed reward={reward:.3f} components={rec}")

        return reward

    def get_observation_components(self) -> np.ndarray:
        if not self._history:
            return np.zeros(3, np.float32)
        arr = np.array(self._history, np.float32)
        return np.array([self._last, arr.mean(), arr.std()], np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "history": list(self._history),
            "_last": float(self._last),
            "_last_reason": self._last_reason,
            "weights": self.get_weights(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._history = deque(state.get("history", []), maxlen=self._history.maxlen)
        self._last = float(state.get("_last", 0.0))
        self._last_reason = state.get("_last_reason", "")
        w = state.get("weights", {})
        if w:
            self.regime_weights = np.array(w.get("regime_weights", self.regime_weights), np.float32)
            self.dd_pen_weight = float(w.get("dd_pen_weight", self.dd_pen_weight))
            self.risk_pen_weight = float(w.get("risk_pen_weight", self.risk_pen_weight))
            self.tail_pen_weight = float(w.get("tail_pen_weight", self.tail_pen_weight))
            self.mistake_pen_weight = float(w.get("mistake_pen_weight", self.mistake_pen_weight))
            self.no_trade_penalty_weight = float(w.get("no_trade_penalty_weight", self.no_trade_penalty_weight))
