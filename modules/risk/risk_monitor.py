import numpy as np
import logging
import json
import os
import datetime
import random
import copy
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from modules.core.core import Module

def utcnow() -> str:
    return datetime.datetime.utcnow().isoformat()

# ─────────────────────────────────────────────────────────────────────────────#
# ActiveTradeMonitor
# ─────────────────────────────────────────────────────────────────────────────#
class ActiveTradeMonitor(Module):
    """
    Flags trades that exceed a maximum duration.
    Evolves max_duration; writes an audit for each alert.
    """
    AUDIT_PATH = "logs/risk/active_trade_monitor_audit.jsonl"
    LOG_PATH   = "logs/risk/active_trade_monitor.log"

    def __init__(self, max_duration: int = 50, enabled: bool = True, audit_log_size: int = 100):
        super().__init__()
        self.max_duration = max_duration
        self.enabled = enabled
        self.alerted = False
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)

        # Logger
        self.logger = logging.getLogger("ActiveTradeMonitor")
        if not self.logger.handlers:
            fh = logging.FileHandler(self.LOG_PATH)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def reset(self):
        self.alerted = False
        self._audit.clear()

    def step(self, open_trades: Optional[List[Dict[str, Any]]] = None, **__):
        if not self.enabled or not open_trades:
            self.alerted = False
            return

        # support dict-of-trades or list-of-trades
        trades = open_trades.values() if isinstance(open_trades, dict) else open_trades

        for t in trades:
            dur = t.get("duration", 0)
            if dur > self.max_duration:
                self.alerted = True
                msg = f"{t['instrument']} duration {dur} > max {self.max_duration}"
                self.logger.warning(msg)
                self._record_audit(t['instrument'], dur, msg)
                break
        else:
            self.alerted = False


    def _record_audit(self, instrument: str, duration: int, msg: str):
        entry = {
            "timestamp": utcnow(),
            "instrument": instrument,
            "duration": duration,
            "max_duration": self.max_duration,
            "message": msg,
        }
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit = self._audit[-self._max_audit:]
        with open(self.AUDIT_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_last_audit(self) -> Dict[str, Any]:
        return self._audit[-1] if self._audit else {}

    def get_audit_trail(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._audit[-n:]

    def mutate(self, std: float = 5):
        old = self.max_duration
        self.max_duration = int(np.clip(old + np.random.normal(0, std), 10, 500))
        self.logger.info(f"mutate: max_duration {old}→{self.max_duration}")

    def crossover(self, other: "ActiveTradeMonitor"):
        new_md = self.max_duration if random.random() < 0.5 else other.max_duration
        child = ActiveTradeMonitor(new_md, enabled=self.enabled or other.enabled)
        return child

    def get_state(self) -> Dict[str, Any]:
        return {
            "max_duration": self.max_duration,
            "enabled": self.enabled,
            "audit": copy.deepcopy(self._audit)
        }

    def set_state(self, state: Dict[str, Any]):
        self.max_duration = state.get("max_duration", self.max_duration)
        self.enabled = state.get("enabled", self.enabled)
        self._audit = copy.deepcopy(state.get("audit", []))

    def get_observation_components(self) -> np.ndarray:
        # [alert_flag, normalized_duration]
        return np.array([float(self.alerted), float(self.max_duration)], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────#
# CorrelatedRiskController
# ─────────────────────────────────────────────────────────────────────────────#
class CorrelatedRiskController(Module):
    """
    Flags if any pair’s absolute correlation exceeds max_corr.
    Evolves max_corr; audit per spike.
    """
    AUDIT_PATH = "logs/risk/correlated_risk_controller_audit.jsonl"
    LOG_PATH   = "logs/risk/correlated_risk_controller.log"

    def __init__(self, max_corr: float = 0.8, enabled: bool = True, audit_log_size: int = 100):
        super().__init__()
        self.max_corr = max_corr
        self.enabled = enabled
        self.high_corr = False
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)

        self.logger = logging.getLogger("CorrelatedRiskController")
        if not self.logger.handlers:
            fh = logging.FileHandler(self.LOG_PATH)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def reset(self):
        self.high_corr = False
        self._audit.clear()

    def step(self, correlations: Optional[Dict[Tuple[str, str], float]] = None, **__):
        self.high_corr = False
        if not self.enabled or not correlations:
            return False

        for (i1, i2), corr in correlations.items():
            if abs(corr) > self.max_corr:
                self.high_corr = True
                msg = f"{i1}&{i2} corr {corr:.2f} > max {self.max_corr}"
                self.logger.warning(msg)
                self._record_audit(i1, i2, corr, msg)
                return True
        return False

    def _record_audit(self, i1: str, i2: str, corr: float, msg: str):
        entry = {
            "timestamp": utcnow(),
            "pair": [i1, i2],
            "corr": corr,
            "max_corr": self.max_corr,
            "message": msg,
        }
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit = self._audit[-self._max_audit:]
        with open(self.AUDIT_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_last_audit(self) -> Dict[str, Any]:
        return self._audit[-1] if self._audit else {}

    def get_audit_trail(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._audit[-n:]

    def mutate(self, std: float = 0.05):
        old = self.max_corr
        self.max_corr = float(np.clip(old + np.random.normal(0, std), 0.1, 0.99))
        self.logger.info(f"mutate: max_corr {old:.2f}→{self.max_corr:.2f}")

    def crossover(self, other: "CorrelatedRiskController"):
        new_mc = self.max_corr if random.random() < 0.5 else other.max_corr
        child = CorrelatedRiskController(new_mc, enabled=self.enabled or other.enabled)
        return child

    def get_state(self) -> Dict[str, Any]:
        return {"max_corr": self.max_corr, "enabled": self.enabled, "audit": copy.deepcopy(self._audit)}

    def set_state(self, state: Dict[str, Any]):
        self.max_corr = state.get("max_corr", self.max_corr)
        self.enabled = state.get("enabled", self.enabled)
        self._audit = copy.deepcopy(state.get("audit", []))

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(self.high_corr), self.max_corr], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────#
# DrawdownRescue
# ─────────────────────────────────────────────────────────────────────────────#
class DrawdownRescue(Module):
    """
    Triggers if drawdown exceeds dd_limit.
    Evolves dd_limit; audit on exceed.
    """
    AUDIT_PATH = "logs/risk/drawdown_rescue_audit.jsonl"
    LOG_PATH   = "logs/risk/drawdown_rescue.log"

    def __init__(self, dd_limit: float = 0.3, enabled: bool = True, audit_log_size: int = 100):
        super().__init__()
        self.dd_limit = dd_limit
        self.enabled = enabled
        self.triggered = False
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)

        self.logger = logging.getLogger("DrawdownRescue")
        if not self.logger.handlers:
            fh = logging.FileHandler(self.LOG_PATH)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def reset(self):
        self.triggered = False
        self._audit.clear()

    def step(self, current_drawdown: Optional[float] = None, **__):
        if not self.enabled or current_drawdown is None:
            self.triggered = False
            return False
        if current_drawdown > self.dd_limit:
            self.triggered = True
            msg = f"drawdown {current_drawdown:.2f} > limit {self.dd_limit:.2f}"
            self.logger.warning(msg)
            self._record_audit(current_drawdown, msg)
            return True
        self.triggered = False
        return False

    def _record_audit(self, dd: float, msg: str):
        entry = {
            "timestamp": utcnow(),
            "drawdown": dd,
            "dd_limit": self.dd_limit,
            "message": msg,
        }
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit = self._audit[-self._max_audit:]
        with open(self.AUDIT_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_last_audit(self) -> Dict[str, Any]:
        return self._audit[-1] if self._audit else {}

    def get_audit_trail(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._audit[-n:]

    def mutate(self, std: float = 0.03):
        old = self.dd_limit
        self.dd_limit = float(np.clip(old + np.random.normal(0, std), 0.05, 0.9))
        self.logger.info(f"mutate: dd_limit {old:.2f}→{self.dd_limit:.2f}")

    def crossover(self, other: "DrawdownRescue"):
        new_dd = self.dd_limit if random.random() < 0.5 else other.dd_limit
        child = DrawdownRescue(new_dd, enabled=self.enabled or other.enabled)
        return child

    def get_state(self) -> Dict[str, Any]:
        return {"dd_limit": self.dd_limit, "enabled": self.enabled, "audit": copy.deepcopy(self._audit)}

    def set_state(self, state: Dict[str, Any]):
        self.dd_limit = state.get("dd_limit", self.dd_limit)
        self.enabled = state.get("enabled", self.enabled)
        self._audit = copy.deepcopy(state.get("audit", []))

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(self.triggered), self.dd_limit], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────#
# ExecutionQualityMonitor
# ─────────────────────────────────────────────────────────────────────────────#
class ExecutionQualityMonitor(Module):
    """
    Flags high slippage; evolves slip_limit.
    Audits each slippage event.
    """
    AUDIT_PATH = "logs/risk/execution_quality_monitor_audit.jsonl"
    LOG_PATH   = "logs/risk/execution_quality_monitor.log"

    def __init__(self, slip_limit: float = 0.5, enabled: bool = True, audit_log_size: int = 100):
        super().__init__()
        self.slip_limit = slip_limit
        self.enabled = enabled
        self.issues: List[str] = []
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)

        self.logger = logging.getLogger("ExecutionQualityMonitor")
        if not self.logger.handlers:
            fh = logging.FileHandler(self.LOG_PATH)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def reset(self):
        self.issues.clear()
        self._audit.clear()

    def step(self, trade_executions: Optional[List[Dict[str, Any]]] = None, **__):
        if not self.enabled or not trade_executions:
            return
        for ex in trade_executions:
            slip = abs(ex.get("slippage", 0.0))
            if slip > self.slip_limit:
                msg = f"{ex['instrument']} slip {slip:.2f} > limit {self.slip_limit}"
                self.issues.append(msg)
                self.logger.warning(msg)
                self._record_audit(ex['instrument'], slip, msg)

    def _record_audit(self, instrument: str, slip: float, msg: str):
        entry = {
            "timestamp": utcnow(),
            "instrument": instrument,
            "slippage": slip,
            "slip_limit": self.slip_limit,
            "message": msg,
        }
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit = self._audit[-self._max_audit:]
        with open(self.AUDIT_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_last_audit(self) -> Dict[str, Any]:
        return self._audit[-1] if self._audit else {}

    def get_audit_trail(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._audit[-n:]

    def mutate(self, std: float = 0.05):
        old = self.slip_limit
        self.slip_limit = float(np.clip(old + np.random.normal(0, std), 0.01, 2.0))
        self.logger.info(f"mutate: slip_limit {old:.2f}→{self.slip_limit:.2f}")

    def crossover(self, other: "ExecutionQualityMonitor"):
        new_sl = self.slip_limit if random.random() < 0.5 else other.slip_limit
        child = ExecutionQualityMonitor(new_sl, enabled=self.enabled or other.enabled)
        return child

    def get_state(self) -> Dict[str, Any]:
        return {"slip_limit": self.slip_limit, "enabled": self.enabled, "audit": copy.deepcopy(self._audit)}

    def set_state(self, state: Dict[str, Any]):
        self.slip_limit = state.get("slip_limit", self.slip_limit)
        self.enabled = state.get("enabled", self.enabled)
        self._audit = copy.deepcopy(state.get("audit", []))

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(bool(self.issues)), self.slip_limit], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────#
# AnomalyDetector
# ─────────────────────────────────────────────────────────────────────────────#
class AnomalyDetector(Module):
    """
    Flags PnL outliers or NaN/Inf in observations.
    Evolves pnl_limit; audits each anomaly.
    """
    AUDIT_PATH = "logs/risk/anomaly_detector_audit.jsonl"
    LOG_PATH   = "logs/risk/anomaly_detector.log"

    def __init__(self, pnl_limit: float = 10_000, enabled: bool = True, audit_log_size: int = 100):
        super().__init__()
        self.enabled = enabled
        self.pnl_limit = pnl_limit
        self.last_alert = ""
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)

        self.logger = logging.getLogger("AnomalyDetector")
        if not self.logger.handlers:
            fh = logging.FileHandler(self.LOG_PATH)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def reset(self):
        self.last_alert = ""
        self._audit.clear()

    def step(self, pnl: Optional[float] = None, obs: Optional[np.ndarray] = None, **__):
        if not self.enabled:
            return
        # PnL outlier
        if pnl is not None and abs(pnl) > self.pnl_limit:
            self.last_alert = f"PnL {pnl:.2f} > limit {self.pnl_limit}"
            self.logger.warning(self.last_alert)
            self._record_audit("pnl_outlier", pnl, self.pnl_limit, self.last_alert)
        # Observation NaN/Inf
        if obs is not None and (np.isnan(obs).any() or np.isinf(obs).any()):
            self.last_alert = "obs contained NaN/Inf"
            self.logger.warning(self.last_alert)
            self._record_audit("obs_invalid", obs.tolist(), None, self.last_alert)

    def _record_audit(self, event: str, value: Any, threshold: Optional[float], msg: str):
        entry = {
            "timestamp": utcnow(),
            "event": event,
            "value": value,
            "threshold": threshold,
            "message": msg,
        }
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit = self._audit[-self._max_audit:]
        with open(self.AUDIT_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_last_audit(self) -> Dict[str, Any]:
        return self._audit[-1] if self._audit else {}

    def get_audit_trail(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._audit[-n:]

    def mutate(self, std: float = 2000):
        old = self.pnl_limit
        self.pnl_limit = float(np.clip(old + np.random.normal(0, std), 500, 1e6))
        self.logger.info(f"mutate: pnl_limit {old:.0f}→{self.pnl_limit:.0f}")

    def crossover(self, other: "AnomalyDetector"):
        new_pl = self.pnl_limit if random.random() < 0.5 else other.pnl_limit
        child = AnomalyDetector(new_pl, enabled=self.enabled or other.enabled)
        return child

    def get_state(self) -> Dict[str, Any]:
        return {"pnl_limit": self.pnl_limit, "enabled": self.enabled, "audit": copy.deepcopy(self._audit)}

    def set_state(self, state: Dict[str, Any]):
        self.pnl_limit = state.get("pnl_limit", self.pnl_limit)
        self.enabled = state.get("enabled", self.enabled)
        self._audit = copy.deepcopy(state.get("audit", []))

    def get_observation_components(self) -> np.ndarray:
        alert_flag = 1.0 if self.last_alert else 0.0
        return np.array([alert_flag, self.pnl_limit], dtype=np.float32)
