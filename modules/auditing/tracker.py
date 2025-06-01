# modules/tracker.py

import numpy as np
from typing import Any, Tuple, List, Dict, Optional
from modules.core.core import Module

class TradeThesisTracker(Module):
    """
    Tracks the outcome (PnL) of each unique trading thesis.
    Produces summary stats and supports persistence for checkpointing.

    Usage:
        - Call .record(thesis, pnl) after each trade to log result.
        - Optionally, call .step(trades=[...]) to auto-record all trades in a step.
        - Use .get_observation_components() for env state/observation.
        - .get_state() and .set_state() for save/load.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        self.records: List[Tuple[Any, float]] = []

    def step(self, trades: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        """
        Optionally auto-record thesis & pnl for each trade if provided.
        """
        if trades:
            for t in trades:
                thesis = t.get("thesis")
                pnl = t.get("pnl", 0.0)
                if thesis is not None:
                    self.record(thesis, pnl)

    def record(self, thesis: Any, pnl: float) -> None:
        """
        Add a (thesis, pnl) entry to the log.
        """
        self.records.append((thesis, pnl))
        if self.debug:
            print(f"[TradeThesisTracker] Recorded: {thesis} | PnL: {pnl:.2f}")

    def get_observation_components(self) -> np.ndarray:
        """
        Outputs:
            [uniq_theses, mean_pnl, sd_pnl, per-thesis mean (up to 4)]
            Pad with zeros if not enough unique theses.
        """
        if not self.records:
            return np.zeros(7, dtype=np.float32)

        theses, pnls = zip(*self.records)
        uniq = len(set(theses))
        mean_p = float(np.mean(pnls))
        sd_p = float(np.std(pnls))

        # per-thesis mean for up to 4 unique theses
        per: List[float] = []
        for t in list(dict.fromkeys(theses))[:4]:
            vs = [p for (th, p) in self.records if th == t]
            per.append(float(np.mean(vs)))
        per += [0.0] * (4 - len(per))

        return np.array([uniq, mean_p, sd_p, *per], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        """
        Return the internal state for checkpointing.
        """
        return {"records": self.records.copy()}

    def set_state(self, state: Dict[str, Any]):
        """
        Restore state from checkpoint.
        """
        self.records = state.get("records", []).copy()
        if self.debug:
            print(f"[TradeThesisTracker] State restored ({len(self.records)} records)")
