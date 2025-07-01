# ──────────────────────────────────────────────────────────────
# File: modules/simulation/shadow_simulator.py
# ──────────────────────────────────────────────────────────────

import os
import copy
import random
import logging
from typing import List, Dict, Any
import numpy as np

from utils.get_dir import utcnow
from ..core.core import Module

class ShadowSimulator(Module):
    MODES = ["greedy", "random", "conservative"]

    def __init__(
        self,
        horizon: int = 5,
        mode: str = "greedy",
        debug: bool = True,
        audit_log_size: int = 100,
    ):
        self.horizon = int(horizon)
        self.mode = mode if mode in self.MODES else "greedy"
        self.debug = debug

        # audit trail
        self.audit_trail: List[Dict[str, Any]] = []
        self._audit_log_size = audit_log_size
        LOG_PATH = "logs/simulation/shadow_simulator.log"

        # 2) make sure the directory exists
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

        # 3) configure your logger (only adds the handler once)
        self.logger = logging.getLogger("ShadowSimulator")
        if not any(
            isinstance(h, logging.FileHandler)
            and os.path.abspath(h.baseFilename) == os.path.abspath(LOG_PATH)
            for h in self.logger.handlers
        ):
            fh = logging.FileHandler(LOG_PATH)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.propagate = False

    def reset(self):
        self.audit_trail.clear()

    def step(self, **kwargs):
        # Stateless in pipeline
        pass

    def simulate(self, env, actions: np.ndarray) -> List[dict]:
        """
        Simulate sequence of trades over the next `horizon` steps.
        Does not modify the real env.
        Returns list of simulated trade dicts.
        """
        if env is None or actions is None:
            return []
        # backup full env state
        try:
            state_backup = env.get_state()
        except Exception:
            state_backup = None

        trades: List[dict] = []
        for h in range(self.horizon):
            # stop if no more data
            first_tf = next(iter(env.data.values()))["D1"]
            if env.current_step >= len(first_tf):
                break

            step_audit = {
                "timestamp":     utcnow(),
                "horizon_step":  h,
                "mode":          self.mode,
                "planned_trades": [],
                "env_snapshot": {
                    "current_step": env.current_step,
                    "balance":      env.balance,
                }
            }

            for i, instr in enumerate(env.instruments):
                base_act = actions[2 * i : 2 * i + 2]
                if self.mode == "greedy":
                    act = base_act
                    rationale = "Greedy: use provided actions"
                elif self.mode == "random":
                    act = np.random.uniform(-1, 1, 2)
                    rationale = "Random exploration"
                else:  # conservative
                    act = base_act * 0.5
                    rationale = "Conservative: half intensity"

                tr = env._execute_trade(instr, float(act[0]), float(act[1]))
                if tr:
                    trades.append(tr)
                    step_audit["planned_trades"].append({
                        "instrument": instr,
                        "action":     act.tolist(),
                        "result":     tr,
                        "rationale":  rationale,
                    })

            self._record_audit(step_audit)
            if self.debug:
                print(f"[ShadowSimulator] step={h} mode={self.mode} trades={len(step_audit['planned_trades'])}")
            env.current_step += 1

        self.logger.info(f"Simulated {len(trades)} trades over horizon={self.horizon} mode={self.mode}")

        # restore env state
        if state_backup is not None:
            env.set_state(state_backup)

        return trades

    def _record_audit(self, entry: Dict[str, Any]):
        self.audit_trail.append(entry)
        if len(self.audit_trail) > self._audit_log_size:
            self.audit_trail = self.audit_trail[-self._audit_log_size:]
        if self.debug:
            logging.debug(f"[ShadowSimulator][AUDIT] {entry}")

    def get_last_audit(self) -> Dict[str, Any]:
        return self.audit_trail[-1] if self.audit_trail else {}

    def get_audit_trail(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.audit_trail[-n:]

    def mutate(self, std: float = 1.0) -> None:
        self.horizon = max(1, int(self.horizon + np.random.randint(-2, 3)))
        if random.random() < 0.3:
            self.mode = random.choice(self.MODES)
        self.logger.info(f"Mutated: horizon={self.horizon}, mode={self.mode}")

    def crossover(self, other: "ShadowSimulator") -> "ShadowSimulator":
        new_h = self.horizon if random.random() < 0.5 else other.horizon
        new_m = self.mode if random.random() < 0.5 else other.mode
        child = ShadowSimulator(horizon=new_h, mode=new_m, debug=self.debug)
        self.logger.info(f"Crossover -> horizon={new_h}, mode={new_m}")
        return child

    def get_observation_components(self) -> np.ndarray:
        mode_idx = float(self.MODES.index(self.mode))
        return np.array([float(self.horizon), mode_idx], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "horizon":     self.horizon,
            "mode":        self.mode,
            "audit_trail": copy.deepcopy(self.audit_trail),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.horizon     = int(state.get("horizon", self.horizon))
        self.mode        = state.get("mode", self.mode)
        self.audit_trail = copy.deepcopy(state.get("audit_trail", []))
