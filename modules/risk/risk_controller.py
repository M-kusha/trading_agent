import numpy as np
import logging
import json
import os
import datetime
from collections import deque
from typing import Dict, Any, List, Optional
from modules.core.core import Module
import copy

class DynamicRiskController(Module):
    """
    Evolvable risk controller with full audit trail:
      - Throttles risk using drawdown & volatility, with freeze logic.
      - Logs every throttle/freeze/unfreeze with rationale for full explainability.
      - Persists audits to JSONL so you can review them later.
    """

    DEFAULTS = {
        "freeze_duration": 5,
        "vol_history_len": 100,
        "dd_threshold": 0.2,
        "vol_ratio_threshold": 1.5,
    }

    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        action_dim: int = 1,
        debug: bool = True,
        audit_log_size: int = 100,
        audit_log_path: str = "logs/risk/dynamic_risk_audit.jsonl",
    ):
        super().__init__()
        # Parameters
        p = copy.deepcopy(params) if params else dict(self.DEFAULTS)
        self.base_duration = int(p.get("freeze_duration", self.DEFAULTS["freeze_duration"]))
        self.vol_history = deque(maxlen=int(p.get("vol_history_len", self.DEFAULTS["vol_history_len"])))
        self.dd_th = float(p.get("dd_threshold", self.DEFAULTS["dd_threshold"]))
        self.vol_th = float(p.get("vol_ratio_threshold", self.DEFAULTS["vol_ratio_threshold"]))
        self.action_dim = int(action_dim)
        self.debug = debug

        # State
        self.freeze_counter = 0
        self._last_dd = 0.0
        self._last_vol = 0.0

        # Audit trail (in-memory and on-disk)
        self._audit_trail: List[Dict[str, Any]] = []
        self._audit_log_size = audit_log_size
        self.audit_log_path = audit_log_path
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)

        # Logger setup
        self.logger = logging.getLogger("DynamicRiskController")
        if not self.logger.handlers:
            fh = logging.FileHandler("logs/risk/dynamic_risk.log")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        self.logger.propagate = False

        # For serialization
        self._params = {
            "freeze_duration": self.base_duration,
            "vol_history_len": self.vol_history.maxlen,
            "dd_threshold": self.dd_th,
            "vol_ratio_threshold": self.vol_th,
        }

    # ------------------- Evolutionary logic (unchanged) ------------------- #
    def mutate(self, std: float = 0.1):
        self.base_duration = int(np.clip(
            self.base_duration + int(np.random.normal(0, std * 4)),
            1, 20
        ))
        new_vhl = int(np.clip(
            self.vol_history.maxlen + int(np.random.normal(0, std * 30)),
            10, 1000
        ))
        # preserve tail
        tail = list(self.vol_history)[-new_vhl:]
        self.vol_history = deque(tail, maxlen=new_vhl)
        self.dd_th = float(np.clip(self.dd_th + np.random.normal(0, std * 0.1), 0.05, 0.8))
        self.vol_th = float(np.clip(self.vol_th + np.random.normal(0, std * 0.2), 0.3, 5.0))
        self._params = {
            "freeze_duration": self.base_duration,
            "vol_history_len": self.vol_history.maxlen,
            "dd_threshold": self.dd_th,
            "vol_ratio_threshold": self.vol_th,
        }

    def crossover(self, other: "DynamicRiskController") -> "DynamicRiskController":
        def blend(a, b):
            return a if np.random.rand() > 0.5 else b
        params = {
            "freeze_duration": blend(self.base_duration, other.base_duration),
            "vol_history_len": blend(self.vol_history.maxlen, other.vol_history.maxlen),
            "dd_threshold": blend(self.dd_th, other.dd_th),
            "vol_ratio_threshold": blend(self.vol_th, other.vol_th),
        }
        return DynamicRiskController(
            params=params,
            action_dim=self.action_dim,
            debug=self.debug or other.debug,
            audit_log_size=self._audit_log_size,
            audit_log_path=self.audit_log_path,
        )

    def get_params(self) -> Dict[str, float]:
        return dict(self._params)

    # ---------------------- Audit/Explainability --------------------- #
    def _log_audit(self,
                   reason: str,
                   dd: float,
                   vol: float,
                   vr: float,
                   freeze_before: int,
                   freeze_after: int):
        audit = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "drawdown": dd,
            "volatility": vol,
            "vol_ratio": vr,
            "dd_threshold": self.dd_th,
            "vol_ratio_threshold": self.vol_th,
            "freeze_before": freeze_before,
            "freeze_after": freeze_after,
            "reason": reason,
        }
        # In-memory
        self._audit_trail.append(audit)
        if len(self._audit_trail) > self._audit_log_size:
            self._audit_trail = self._audit_trail[-self._audit_log_size:]
        # Persistent JSONL
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(audit) + "\n")
        # Logging
        self.logger.debug(f"[AUDIT] {json.dumps(audit)}")

    def get_last_audit(self) -> Dict[str, Any]:
        return self._audit_trail[-1] if self._audit_trail else {}

    def get_audit_trail(self, n: int = 20) -> List[Dict[str, Any]]:
        return self._audit_trail[-n:]

    # ---------------------- Core interface ---------------------- #
    def reset(self):
        self.freeze_counter = 0
        self.vol_history.clear()
        self._last_dd = 0.0
        self._last_vol = 0.0
        self._audit_trail.clear()

    def step(self, drawdown: float = 0.0, volatility: float = 0.0, **__):
        """
        Expects keywords 'drawdown' and 'volatility' from the pipeline.
        """
        self.adjust_risk({"drawdown": drawdown, "volatility": volatility})

    def adjust_risk(self, stats: Dict[str, float]) -> None:
        dd = float(stats.get("drawdown", 0.0))
        vol = float(stats.get("volatility", 0.0))
        self._last_dd = dd
        self._last_vol = vol

        # Update history
        self.vol_history.append(vol)

        # Compute volatility ratio
        if len(self.vol_history) > 1:
            avg_vol = float(np.mean(self.vol_history))
            vr = vol / (avg_vol + 1e-8)
        else:
            vr = 0.0  # no ratio on first reading

        # Determine new freeze_counter
        before = self.freeze_counter
        reason = ""
        # 1st reading: if no history
        if len(self.vol_history) == 1:
            if vol > self.vol_th:
                self.freeze_counter = self.base_duration
                reason = f"Initial vol {vol:.3f} > vol_th {self.vol_th:.3f}"
            else:
                self.freeze_counter = 0
                reason = "Initial vol OK"
        else:
            # Drawdown-triggered freeze
            if dd > self.dd_th:
                self.freeze_counter = int(self.base_duration * np.clip(vr, 0.5, 2.0))
                reason = f"DD {dd:.3f} > dd_th {self.dd_th:.3f}"
            # Volatility-ratio-triggered freeze
            elif vr > self.vol_th:
                self.freeze_counter = int(self.base_duration * np.clip(vr, 0.5, 2.0))
                reason = f"VR {vr:.2f} > vol_th {self.vol_th:.2f}"
            else:
                # decrement counter
                self.freeze_counter = max(0, self.freeze_counter - 1)
                reason = "No threshold exceeded; decrement"

        after = self.freeze_counter
        # Log it
        self._log_audit(reason, dd, vol, vr, before, after)

        if self.debug:
            self.logger.info(
                f"[DRC] dd={dd:.3f} vol={vol:.3f} vr={vr:.2f} "
                f"freeze {before}->{after} ({reason})"
            )

    def get_observation_components(self) -> np.ndarray:
        """
        Returns [scale, drawdown, volatility, vol_ratio]
        where scale=0 if frozen, else=1.
        """
        scale = 0.0 if self.freeze_counter > 0 else 1.0
        # recompute vr for observation
        if len(self.vol_history) > 1:
            avg_vol = float(np.mean(self.vol_history))
            vr = self._last_vol / (avg_vol + 1e-8)
        else:
            vr = 0.0
        obs = np.array([scale, self._last_dd, self._last_vol, vr], dtype=np.float32)
        if self.debug:
            self.logger.debug(f"[DRC] obs={obs}")
        return obs

    def propose_action(self, obs: Any = None) -> np.ndarray:
        """
        Returns a throttle vector: zeros if in freeze, ones otherwise.
        """
        scale = 0.0 if self.freeze_counter > 0 else 1.0
        return np.full(self.action_dim, scale, dtype=np.float32)

    def confidence(self, obs: Any = None) -> float:
        """
        Confidence degrades to 0.3 when frozen, 1.0 otherwise.
        """
        conf = 0.3 if self.freeze_counter > 0 else 1.0
        if self.debug:
            self.logger.debug(f"[DRC] confidence={conf:.2f}")
        return conf

    def get_state(self) -> Dict[str, Any]:
        return {
            "freeze_counter": int(self.freeze_counter),
            "vol_history": list(self.vol_history),
            "_last_dd": self._last_dd,
            "_last_vol": self._last_vol,
            "params": self.get_params(),
            "audit_trail": copy.deepcopy(self._audit_trail),
        }

    def set_state(self, state: Dict[str, Any]):
        self.freeze_counter = int(state.get("freeze_counter", 0))
        vhl = state.get("params", {}).get("vol_history_len", self.vol_history.maxlen)
        self.vol_history = deque(state.get("vol_history", []), maxlen=int(vhl))
        self._last_dd = float(state.get("_last_dd", 0.0))
        self._last_vol = float(state.get("_last_vol", 0.0))
        params = state.get("params", {})
        self.base_duration = int(params.get("freeze_duration", self.base_duration))
        self.dd_th = float(params.get("dd_threshold", self.dd_th))
        self.vol_th = float(params.get("vol_ratio_threshold", self.vol_th))
        self._params = params if params else self._params
        self._audit_trail = copy.deepcopy(state.get("audit_trail", []))
