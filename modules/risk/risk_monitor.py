# modules/risk_monitor.py

import numpy as np
import logging
from modules.core.core import Module

class ActiveTradeMonitor(Module):
    """Watches open trades for anomalies (stuck, too long, underperforming, etc)."""
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

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(self.alerted)], dtype=np.float32)

class CorrelatedRiskController(Module):
    """Limits new trades if asset correlations spike (reduces portfolio risk)."""
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

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(self.high_corr_flag)], dtype=np.float32)

class DrawdownRescue(Module):
    """Triggers risk reduction or trading halt on severe drawdown."""
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

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(self.triggered)], dtype=np.float32)

class ExecutionQualityMonitor(Module):
    """Logs/flags slippage, failed trades, order anomalies."""
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

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(len(self.issues) > 0)], dtype=np.float32)

class AnomalyDetector(Module):
    """General-purpose anomaly detector (PnL outliers, missing data, strange obs)."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.last_alert = ""

    def reset(self):
        self.last_alert = ""

    def step(self, pnl=None, obs=None, **kwargs):
        if not self.enabled:
            return
        if pnl is not None and abs(pnl) > 10_000:
            self.last_alert = f"Abnormal PnL: {pnl}"
            logging.warning(f"[AnomalyDetector] {self.last_alert}")
        if obs is not None and (np.isnan(obs).any() or np.isinf(obs).any()):
            self.last_alert = "Observation contains NaN or Inf"
            logging.warning(f"[AnomalyDetector] {self.last_alert}")

    def get_observation_components(self) -> np.ndarray:
        return np.array([1.0 if self.last_alert else 0.0], dtype=np.float32)
