import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional
import copy
from ..core.core import Module

class RiskAdjustedReward(Module):
    """
    Evolutionary, audit-compliant risk and reward shaper:
    - Evolves reward weights for regime bonuses, risk, drawdown, tail, mistake memory.
    - Tracks recent rewards for diagnostics and obs.
    - Writes reason/source to info dict.
    """
    def __init__(self, initial_balance: float, env=None, debug: bool = False, history: int = 50):
        self.initial_balance = initial_balance
        self.env   = env
        self.debug = debug
        # All weights can evolve!
        self.regime_weights = np.array([0.5, 0.3, 0.2], np.float32)     # For regime onehot
        self.dd_pen_weight = 10.0
        self.risk_pen_weight = 0.3
        self.tail_pen_weight = 1.0
        self.mistake_pen_weight = 0.5
        self.no_trade_penalty_weight = 0.1
        self._history = deque(maxlen=history)
        self._last = 0.0
        self._last_reason = ""

    # Evolution methods
    def mutate(self, std=0.1):
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
            print("[RiskAdjustedReward] Mutated weights:", self.get_weights())

    def crossover(self, other: "RiskAdjustedReward"):
        child = copy.deepcopy(self)
        for attr in [
            "regime_weights",
            "dd_pen_weight",
            "risk_pen_weight",
            "tail_pen_weight",
            "mistake_pen_weight",
            "no_trade_penalty_weight"
        ]:
            if np.random.rand() > 0.5:
                setattr(child, attr, copy.deepcopy(getattr(other, attr)))
        if self.debug:
            print("[RiskAdjustedReward] Crossover complete.")
        return child

    def get_weights(self):
        return dict(
            regime_weights=self.regime_weights.copy(),
            dd_pen_weight=self.dd_pen_weight,
            risk_pen_weight=self.risk_pen_weight,
            tail_pen_weight=self.tail_pen_weight,
            mistake_pen_weight=self.mistake_pen_weight,
            no_trade_penalty_weight=self.no_trade_penalty_weight,
        )

    def reset(self):
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
        """
        Applies evolutionary reward shaping and populates info dict. Never double-penalizes.
        """
        # No-trade case
        if not trades:
            penalty = -self.no_trade_penalty_weight * float(np.sqrt((actions**2).mean()))
            self._last = penalty
            self._last_reason = "no-trade"
            if info is not None:
                info["shaped_reward"] = penalty
                info["reward_reason"] = "no-trade"
            self._history.append(penalty)
            if self.debug:
                print(f"[RiskAdjustedReward] penalty={penalty:.3f} reason=no-trade")
            return penalty

        # Basic PnL & tail risk
        pnl  = sum(t["pnl"] for t in trades)
        rets = np.array([t["pnl"] for t in trades], np.float32)
        var  = np.percentile(rets, 5)
        cvar = rets[rets <= var].mean() if np.any(rets <= var) else var
        tail_pen = -self.tail_pen_weight * (0.5 * var + 0.5 * cvar)

        # Drawdown & risk penalties
        dd_pen   = current_drawdown * self.dd_pen_weight
        risk_pen = float(np.sqrt((actions**2).mean())) * self.risk_pen_weight

        # Regime bonus
        regime_bonus = float(regime_onehot.dot(self.regime_weights))

        # MistakeMemory penalty (if present)
        mm_pen = 0.0
        if self.env is not None and hasattr(self.env, "mistake_memory"):
            mm_pen = float(self.env.mistake_memory.get_observation_components()[0]) * self.mistake_pen_weight

        reward = pnl - dd_pen - risk_pen - tail_pen - mm_pen + regime_bonus

        # State/obs tracking and info dict update
        self._last = reward
        self._last_reason = "trade"
        if info is not None:
            info["shaped_reward"] = reward
            info["reward_reason"] = "trade"
            info["pnl"] = pnl
            info["tail_penalty"] = tail_pen
            info["drawdown_penalty"] = dd_pen
            info["risk_penalty"] = risk_pen
            info["regime_bonus"] = regime_bonus
            info["mistake_memory_penalty"] = mm_pen

        self._history.append(reward)
        if self.debug:
            print(
                f"[RiskAdjustedReward] reward={reward:.3f} reason=trade pnl={pnl:.2f} "
                f"dd_pen={dd_pen:.2f} risk_pen={risk_pen:.2f} tail_pen={tail_pen:.2f} "
                f"regime_bonus={regime_bonus:.2f} mm_pen={mm_pen:.2f}"
            )

        return reward

    def get_observation_components(self) -> np.ndarray:
        if not self._history:
            return np.zeros(3, np.float32)
        arr = np.array(self._history, dtype=np.float32)
        return np.array([self._last, arr.mean(), arr.std()], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "history": list(self._history),
            "_last": float(self._last),
            "_last_reason": self._last_reason,
            "weights": self.get_weights(),
        }

    def set_state(self, state: Dict[str, Any]):
        self._history = deque(state.get("history", []), maxlen=self._history.maxlen)
        self._last = float(state.get("_last", 0.0))
        self._last_reason = state.get("_last_reason", "")
        # Restore weights (optional)
        w = state.get("weights", None)
        if w:
            self.regime_weights = np.array(w.get("regime_weights", self.regime_weights), np.float32)
            self.dd_pen_weight = float(w.get("dd_pen_weight", self.dd_pen_weight))
            self.risk_pen_weight = float(w.get("risk_pen_weight", self.risk_pen_weight))
            self.tail_pen_weight = float(w.get("tail_pen_weight", self.tail_pen_weight))
            self.mistake_pen_weight = float(w.get("mistake_pen_weight", self.mistake_pen_weight))
            self.no_trade_penalty_weight = float(w.get("no_trade_penalty_weight", self.no_trade_penalty_weight))
