# ──────────────────────────────────────────────────────────────
# File: modules/simulation/role_coach.py
# ──────────────────────────────────────────────────────────────

import os
import copy
import random
import logging
from typing import List, Dict, Any
import numpy as np

from utils.get_dir import utcnow
from ..core.core import Module

# ──────────────────────────────────────────────────────────────────────────────
class RoleCoach(Module):
    def __init__(
        self,
        max_trades: int = 2,
        penalty: float = 1.0,
        debug: bool = False,
        audit_log_size: int = 100,
    ):
        self.max_trades = int(max_trades)
        self.penalty    = float(penalty)
        self.debug      = debug

        # audit trail
        self.audit_trail: List[Dict[str, Any]] = []
        self._audit_log_size = audit_log_size

        # logger setup
        os.makedirs("logs", exist_ok=True)
        self.logger = logging.getLogger("RoleCoach")
        if not self.logger.handlers:
            fh = logging.FileHandler("logs/simulation/role_coach.log")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def reset(self):
        self.audit_trail.clear()

    def step(self, trades: List[dict] = None, **kwargs) -> float:
        n = len(trades) if trades else 0
        over = max(0, n - self.max_trades)
        score = over * self.penalty
        rationale = (
            f"{n} trades (max {self.max_trades})  penalty {score}"
            if over else
            f"{n} trades within limit ({self.max_trades})  no penalty"
        )
        entry = {
            "timestamp":       utcnow(),
            "num_trades":      n,
            "max_trades":      self.max_trades,
            "penalty_per":     self.penalty,
            "discipline_penalty": score,
            "rationale":       rationale,
        }
        self._record_audit(entry)
        self.logger.info(f"[RoleCoach] {rationale}")
        if self.debug:
            print(f"[RoleCoach][AUDIT] {entry}")
        return score

    def _record_audit(self, entry: Dict[str, Any]):
        self.audit_trail.append(entry)
        if len(self.audit_trail) > self._audit_log_size:
            self.audit_trail = self.audit_trail[-self._audit_log_size:]

    def get_last_audit(self) -> Dict[str, Any]:
        return self.audit_trail[-1] if self.audit_trail else {}

    def get_audit_trail(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.audit_trail[-n:]

    def mutate(self, std: float = 1.0) -> None:
        self.max_trades = max(1, self.max_trades + random.randint(-1, 1))
        self.penalty    = float(np.clip(self.penalty + np.random.normal(0, std), 0.1, 10.0))
        self.logger.info(f"Mutated: max_trades={self.max_trades}, penalty={self.penalty}")

    def crossover(self, other: "RoleCoach") -> "RoleCoach":
        new_mt = self.max_trades if random.random() < 0.5 else other.max_trades
        new_p  = self.penalty    if random.random() < 0.5 else other.penalty
        child = RoleCoach(max_trades=new_mt, penalty=new_p, debug=self.debug)
        self.logger.info(f"Crossover -> max_trades={new_mt}, penalty={new_p}")
        return child

    def get_observation_components(self) -> np.ndarray:
        return np.array([float(self.max_trades), float(self.penalty)], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "max_trades": self.max_trades,
            "penalty":    self.penalty,
            "audit_trail": copy.deepcopy(self.audit_trail),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.max_trades = int(state.get("max_trades", self.max_trades))
        self.penalty    = float(state.get("penalty", self.penalty))
        self.audit_trail = copy.deepcopy(state.get("audit_trail", []))
