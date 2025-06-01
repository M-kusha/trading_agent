# modules/risk_monitor.py

import numpy as np
import logging
import copy
import random
from modules.core.core import Module

# ════════════════════════════════════════════════════════════════
class ActiveTradeMonitor(Module):
    """Watches open trades for anomalies (stuck, too long, underperforming, etc). Evolves max_duration."""

    def __init__(self, max_duration=50, enabled=True):
        self.max_duration = max_duration
        self.enabled = enabled
        self.alerted = False

    def reset(self):
        self.alerted = False

    def step(self, open_trades=None, **kwargs):
        if not self.enabled or not open_trades:
            return
        for t in open_trades:
            if t.get("duration", 0) > self.max_duration:
                self.alerted = True
                logging.warning(f"[TradeMonitor] Trade {t['instrument']} open too long: {t['duration']} steps")

    def mutate(self, std=5):
        """Randomly perturb max_duration."""
        self.max_duration = int(np.clip(self.max_duration + int(np.random.normal(0, std)), 10, 500))

    def crossover(self, other: "ActiveTradeMonitor") -> "ActiveTradeMonitor":
        md = self.max_duration if random.random() < 0.5 else other.max_duration
        return ActiveTradeMonitor(max_duration=md, enabled=self.enabled or other.enabled)

    def get_state(self):
        return {"max_duration": self.max_duration, "enabled": self.enabled}

    def set_state(self, state):
        self.max_duration = int(state.get("max_duration", self.max_duration))
        self.enabled = bool(state.get("enabled", self.enabled))

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(self.alerted)], dtype=np.float32)

# ════════════════════════════════════════════════════════════════
class CorrelatedRiskController(Module):
    """Limits new trades if asset correlations spike (reduces portfolio risk). Evolves max_corr."""

    def __init__(self, max_corr=0.8, enabled=True):
        self.max_corr = max_corr
        self.enabled = enabled
        self.high_corr_flag = False

    def reset(self):
        self.high_corr_flag = False

    def step(self, correlations=None, **kwargs):
        self.high_corr_flag = False
        if not self.enabled or correlations is None:
            return False
        for (i1, i2), corr in correlations.items():
            if abs(corr) > self.max_corr:
                self.high_corr_flag = True
                logging.warning(f"[RiskController] High correlation: {i1} & {i2} ({corr:.2f})")
                return True  # Should skip or reduce trading volume
        return False

    def mutate(self, std=0.05):
        """Randomly perturb max_corr."""
        self.max_corr = float(np.clip(self.max_corr + np.random.normal(0, std), 0.1, 0.99))

    def crossover(self, other: "CorrelatedRiskController") -> "CorrelatedRiskController":
        mc = self.max_corr if random.random() < 0.5 else other.max_corr
        return CorrelatedRiskController(max_corr=mc, enabled=self.enabled or other.enabled)

    def get_state(self):
        return {"max_corr": self.max_corr, "enabled": self.enabled}

    def set_state(self, state):
        self.max_corr = float(state.get("max_corr", self.max_corr))
        self.enabled = bool(state.get("enabled", self.enabled))

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(self.high_corr_flag)], dtype=np.float32)

# ════════════════════════════════════════════════════════════════
class DrawdownRescue(Module):
    """Triggers risk reduction or trading halt on severe drawdown. Evolves dd_limit."""

    def __init__(self, dd_limit=0.3, enabled=True):
        self.dd_limit = dd_limit
        self.enabled = enabled
        self.triggered = False

    def reset(self):
        self.triggered = False

    def step(self, current_drawdown=None, **kwargs):
        if not self.enabled or current_drawdown is None:
            return False
        if current_drawdown > self.dd_limit:
            self.triggered = True
            logging.warning(f"[DrawdownRescue] Drawdown exceeded: {current_drawdown:.2f}")
            return True
        return False

    def mutate(self, std=0.03):
        self.dd_limit = float(np.clip(self.dd_limit + np.random.normal(0, std), 0.05, 0.9))

    def crossover(self, other: "DrawdownRescue") -> "DrawdownRescue":
        dd = self.dd_limit if random.random() < 0.5 else other.dd_limit
        return DrawdownRescue(dd_limit=dd, enabled=self.enabled or other.enabled)

    def get_state(self):
        return {"dd_limit": self.dd_limit, "enabled": self.enabled}

    def set_state(self, state):
        self.dd_limit = float(state.get("dd_limit", self.dd_limit))
        self.enabled = bool(state.get("enabled", self.enabled))

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(self.triggered)], dtype=np.float32)

# ════════════════════════════════════════════════════════════════
class ExecutionQualityMonitor(Module):
    """Logs/flags slippage, failed trades, order anomalies. Evolves slip_limit."""

    def __init__(self, slip_limit=0.5, enabled=True):
        self.slip_limit = slip_limit
        self.enabled = enabled
        self.issues = []

    def reset(self):
        self.issues.clear()

    def step(self, trade_executions=None, **kwargs):
        if not self.enabled or not trade_executions:
            return
        for ex in trade_executions:
            slippage = abs(ex.get("slippage", 0))
            if slippage > self.slip_limit:
                msg = f"High slippage on {ex['instrument']}: {slippage:.2f}"
                self.issues.append(msg)
                logging.warning(f"[ExecQuality] {msg}")

    def mutate(self, std=0.05):
        self.slip_limit = float(np.clip(self.slip_limit + np.random.normal(0, std), 0.01, 2.0))

    def crossover(self, other: "ExecutionQualityMonitor") -> "ExecutionQualityMonitor":
        sl = self.slip_limit if random.random() < 0.5 else other.slip_limit
        return ExecutionQualityMonitor(slip_limit=sl, enabled=self.enabled or other.enabled)

    def get_state(self):
        return {"slip_limit": self.slip_limit, "enabled": self.enabled}

    def set_state(self, state):
        self.slip_limit = float(state.get("slip_limit", self.slip_limit))
        self.enabled = bool(state.get("enabled", self.enabled))

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(len(self.issues) > 0)], dtype=np.float32)

# ════════════════════════════════════════════════════════════════
class AnomalyDetector(Module):
    """General-purpose anomaly detector (PnL outliers, missing data, strange obs). Evolves alert thresholds."""

    def __init__(self, pnl_limit=10_000, enabled=True):
        self.enabled = enabled
        self.pnl_limit = pnl_limit
        self.last_alert = ""

    def reset(self):
        self.last_alert = ""

    def step(self, pnl=None, obs=None, **kwargs):
        if not self.enabled:
            return
        if pnl is not None and abs(pnl) > self.pnl_limit:
            self.last_alert = f"Abnormal PnL: {pnl}"
            logging.warning(f"[AnomalyDetector] {self.last_alert}")
        if obs is not None and (np.isnan(obs).any() or np.isinf(obs).any()):
            self.last_alert = "Observation contains NaN or Inf"
            logging.warning(f"[AnomalyDetector] {self.last_alert}")

    def mutate(self, std=2000):
        self.pnl_limit = float(np.clip(self.pnl_limit + np.random.normal(0, std), 500, 1e6))

    def crossover(self, other: "AnomalyDetector") -> "AnomalyDetector":
        pl = self.pnl_limit if random.random() < 0.5 else other.pnl_limit
        return AnomalyDetector(pnl_limit=pl, enabled=self.enabled or other.enabled)

    def get_state(self):
        return {"pnl_limit": self.pnl_limit, "enabled": self.enabled}

    def set_state(self, state):
        self.pnl_limit = float(state.get("pnl_limit", self.pnl_limit))
        self.enabled = bool(state.get("enabled", self.enabled))

    def get_observation_components(self) -> np.ndarray:
        return np.array([1.0 if self.last_alert else 0.0], dtype=np.float32)
