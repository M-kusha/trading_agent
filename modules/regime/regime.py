import os
import numpy as np
from collections import deque
import copy
import logging
import json
from datetime import datetime, timezone
from ..core.core import Module

class MarketRegimeSwitcher(Module):
    """
    Evolutionary Market Regime Switcher with rich explainability and auditability.
    Exposes rationale for every regime switch for LLMs, UI, and quantitative audit.
    """
    REGIMES = ["trending_up", "trending_down", "mean_reverting", "high_vol", "low_vol", "neutral"]

    def __init__(
        self,
        window: int = 50,
        vol_window: int = 20,
        trend_factor: float = 0.5,
        mean_thr_factor: float = 0.1,
        vol_high_pct: float = 80.0,
        vol_low_pct: float = 20.0,
        debug: bool = True,
    ):
        self.window = window
        self.vol_window = vol_window
        self.trend_factor = trend_factor
        self.mean_thr_factor = mean_thr_factor
        self.vol_high_pct = vol_high_pct
        self.vol_low_pct = vol_low_pct
        self.debug = debug
        self.reset()
        self._action_dim = 1  # Initialize action dimension to a default value

        # Logger for regime changes (console+file)
        self.logger = logging.getLogger("MarketRegimeSwitcherLogger")
        # >>> NEW
        os.makedirs("logs/regime", exist_ok=True)
        # <<<
        # Prevent duplicate handlers by checking for existing file handlers for this logger
        handler_exists = any(
            isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath("logs/regime/market_regime.log")
            for h in self.logger.handlers
        )
        if not handler_exists:
            handler = logging.FileHandler("logs/regime/market_regime.log")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Regime rationale file (for LLM/analyst or forensic audit)
        self.rationale_log_path = "logs/regime/regime_rationale.jsonl"

        # Internal
        self.last_rationale = {}
        self.last_full_audit = {}

    # ---------------- Evolutionary mutation/crossover -------------------
    def mutate(self, std: float = 0.1):
        self.window += int(np.random.choice([-2, -1, 0, 1, 2]))
        self.window = int(np.clip(self.window, 10, 200))
        self.vol_window += int(np.random.choice([-2, -1, 0, 1, 2]))
        self.vol_window = int(np.clip(self.vol_window, 5, self.window))
        self.trend_factor += np.random.normal(0, std)
        self.trend_factor = float(np.clip(self.trend_factor, 0.1, 2.0))
        self.mean_thr_factor += np.random.normal(0, std / 2)
        self.mean_thr_factor = float(np.clip(self.mean_thr_factor, 0.01, 0.5))
        self.vol_high_pct += np.random.normal(0, 2.0)
        self.vol_high_pct = float(np.clip(self.vol_high_pct, 60.0, 99.0))
        self.vol_low_pct += np.random.normal(0, 2.0)
        self.vol_low_pct = float(np.clip(self.vol_low_pct, 1.0, 40.0))
        if self.debug:
            print("[MarketRegimeSwitcher] Mutated parameters.")

    def crossover(self, other: "MarketRegimeSwitcher"):
        child = copy.deepcopy(self)
        for attr in [
            "window", "vol_window", "trend_factor",
            "mean_thr_factor", "vol_high_pct", "vol_low_pct"
        ]:
            if np.random.rand() > 0.5:
                setattr(child, attr, getattr(other, attr))
        if self.debug:
            print("[MarketRegimeSwitcher] Crossover complete.")
        return child

    def reset(self):
        old_prices = list(getattr(self, "prices", []))
        self.regime = "neutral"
        self.volatility = 0.0
        self.last_rationale = {}
        self.last_full_audit = {}
        # Ensure deque maxlen matches current window
        self.prices = deque(old_prices[-self.window:], maxlen=self.window)

    def step(self, price: float = None):
        if price is not None:
            self.prices.append(price)

        if len(self.prices) < self.window:
            self.regime = "neutral"
            self.volatility = 0.0
            self.last_rationale = {"reason": "Insufficient data"}
            return

        prices = np.array(self.prices)
        returns = np.diff(prices)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        self.volatility = np.std(prices[-self.vol_window:]) if len(prices) >= self.vol_window else 0.0

        trend_thr = self.trend_factor * std_ret
        vol_high_thr = np.percentile(np.abs(returns), self.vol_high_pct)
        vol_low_thr = np.percentile(np.abs(returns), self.vol_low_pct)
        mean_thr = self.mean_thr_factor * std_ret

        # Regime logic w/ explanation hooks
        rationale = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "window": self.window,
            "vol_window": self.vol_window,
            "trend_factor": self.trend_factor,
            "mean_thr_factor": self.mean_thr_factor,
            "vol_high_pct": self.vol_high_pct,
            "vol_low_pct": self.vol_low_pct,
            "mean_ret": float(mean_ret),
            "std_ret": float(std_ret),
            "trend_thr": float(trend_thr),
            "volatility": float(self.volatility),
            "vol_high_thr": float(vol_high_thr),
            "vol_low_thr": float(vol_low_thr),
            "mean_thr": float(mean_thr),
        }

        if mean_ret > trend_thr:
            regime = "trending_up"
            rationale["trigger"] = "mean_ret > trend_thr"
        elif mean_ret < -trend_thr:
            regime = "trending_down"
            rationale["trigger"] = "mean_ret < -trend_thr"
        elif self.volatility > vol_high_thr:
            regime = "high_vol"
            rationale["trigger"] = "volatility > vol_high_thr"
        elif self.volatility < vol_low_thr:
            regime = "low_vol"
            rationale["trigger"] = "volatility < vol_low_thr"
        elif abs(mean_ret) < mean_thr:
            regime = "mean_reverting"
            rationale["trigger"] = "abs(mean_ret) < mean_thr"
        else:
            regime = "neutral"
            rationale["trigger"] = "none (neutral)"

        # Log regime changes only if changed
        if regime != self.regime or self.debug:
            self.logger.info(
                f"Market regime changed: {regime} | "
                f"Mean Return: {mean_ret:.4f}, Volatility: {self.volatility:.4f}, "
                f"Reason: {rationale.get('trigger')}"
            )
            # Save rationale JSON line (for forensic dashboard/LLM explanation)
            with open(self.rationale_log_path, "a") as f:
                f.write(json.dumps({
                    "old_regime": self.regime,
                    "new_regime": regime,
                    **rationale
                }) + "\n")

        self.regime = regime
        self.last_rationale = rationale
        self.last_full_audit = {
            "regime": regime,
            "rationale": rationale,
            "prices": list(self.prices),
        }

        if self.debug:
            print(f"[MarketRegimeSwitcher] regime={regime} ({rationale})")

    def get_regime(self) -> str:
        return self.regime

    def get_last_rationale(self) -> dict:
        return self.last_rationale.copy()

    def get_full_audit(self) -> dict:
        """All details for UI, LLM, REST, or analyst."""
        return copy.deepcopy(self.last_full_audit)

    def get_observation_components(self) -> np.ndarray:
        arr = np.zeros(len(self.REGIMES), dtype=np.float32)
        if self.regime in self.REGIMES:
            arr[self.REGIMES.index(self.regime)] = 1.0
        return arr
    
    # ─────────── voting-committee hooks (added) ───────────────────────────

    def set_action_dim(self, dim: int) -> None:
        """Called once by the environment so we know how long the
        action-vector should be."""
        self._action_dim = int(dim)

    def propose_action(self, obs=None) -> np.ndarray:
        """
        Convert the current regime into a direction & strength:

            trending_up   →  +strength
            trending_down →  -strength
            high_vol      →  -strength  (risk-off)
            low_vol       →  +strength
            everything else (mean-reverting / neutral) → 0
        """
        sign_map = {
            "trending_up":  +1.0,
            "trending_down": -1.0,
            "high_vol":     -1.0,
            "low_vol":      +1.0,
        }
        # Use returns-based volatility for strength calculation
        if len(self.prices) > 1:
            prices = np.array(self.prices)
            returns = np.diff(prices)
            returns_vol = np.std(returns)
            # Normalize by dividing by mean absolute return (avoid div by zero)
            mean_abs_ret = np.mean(np.abs(returns)) if np.mean(np.abs(returns)) > 0 else 1.0
            returns_vol = returns_vol / mean_abs_ret
            returns_vol = float(np.clip(returns_vol, 0.0, 1.0))  # 0…1
        else:
            returns_vol = 0.0
        strength = float(np.clip(returns_vol, 0.0, 1.0))  # 0…1
        sign = sign_map.get(self.regime, 0.0)
        return np.full(getattr(self, "_action_dim", 1),
                    sign * strength, dtype=np.float32)

    def confidence(self, obs=None) -> float:
        """Return confidence in current regime decision (0.0 to 1.0)."""
        if not self.last_rationale:
            return 0.0
        # Simple confidence based on how strong the signal was
        return 0.8  # placeholder - could be made more sophisticated
    
    def get_state(self):
        return {
            "params": {
                "window": self.window,
                "vol_window": self.vol_window,
                "trend_factor": self.trend_factor,
                "mean_thr_factor": self.mean_thr_factor,
                "vol_high_pct": self.vol_high_pct,
                "vol_low_pct": self.vol_low_pct,
            },
            "prices": list(self.prices),
            "regime": self.regime,
            "volatility": self.volatility,
            "last_rationale": self.last_rationale,
            "last_full_audit": self.last_full_audit,
        }

    def load_state(self, state):
        params = state.get("params", {})
        self.window = int(params.get("window", self.window))
        self.vol_window = int(params.get("vol_window", self.vol_window))
        self.trend_factor = float(params.get("trend_factor", self.trend_factor))
        self.mean_thr_factor = float(params.get("mean_thr_factor", self.mean_thr_factor))
        self.vol_high_pct = float(params.get("vol_high_pct", self.vol_high_pct))
        self.vol_low_pct = float(params.get("vol_low_pct", self.vol_low_pct))
        
        # Ensure deque maxlen matches updated window
        prices = state.get("prices", [])
        self.prices = deque(prices[-self.window:], maxlen=self.window)
        self.regime = state.get("regime", "neutral")
        self.volatility = float(state.get("volatility", 0.0))
        self.last_rationale = state.get("last_rationale", {})
        self.last_full_audit = state.get("last_full_audit", {})
