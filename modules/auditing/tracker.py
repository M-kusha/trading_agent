# # modules/tracker.py

# import numpy as np
# from typing import Any, Tuple, List, Dict, Optional
# from modules.core.core import Module

# class TradeThesisTracker(Module):
#     """
#     Tracks the outcome (PnL) of each unique trading thesis.
#     Produces summary stats and supports persistence for checkpointing.

#     Usage:
#         - Call .record(thesis, pnl) after each trade to log result.
#         - Optionally, call .step(trades=[...]) to auto-record all trades in a step.
#         - Use .get_observation_components() for env state/observation.
#         - .get_state() and .set_state() for save/load.
#     """

#     def __init__(self, debug: bool = False):
#         self.debug = debug
#         self.reset()

#     def reset(self) -> None:
#         self.records: List[Tuple[Any, float]] = []

#     def step(self, trades: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
#         """
#         Optionally auto-record thesis & pnl for each trade if provided.
#         """
#         if trades:
#             for t in trades:
#                 thesis = t.get("thesis")
#                 pnl = t.get("pnl", 0.0)
#                 if thesis is not None:
#                     self.record(thesis, pnl)

#     def record(self, thesis: Any, pnl: float) -> None:
#         """
#         Add a (thesis, pnl) entry to the log.
#         """
#         self.records.append((thesis, pnl))
#         if self.debug:
#             print(f"[TradeThesisTracker] Recorded: {thesis} | PnL: {pnl:.2f}")

#     def get_observation_components(self) -> np.ndarray:
#         """
#         Outputs:
#             [uniq_theses, mean_pnl, sd_pnl, per-thesis mean (up to 4)]
#             Pad with zeros if not enough unique theses.
#         """
#         if not self.records:
#             return np.zeros(7, dtype=np.float32)

#         theses, pnls = zip(*self.records)
#         uniq = len(set(theses))
#         mean_p = float(np.mean(pnls))
#         sd_p = float(np.std(pnls))

#         # per-thesis mean for up to 4 unique theses
#         per: List[float] = []
#         for t in list(dict.fromkeys(theses))[:4]:
#             vs = [p for (th, p) in self.records if th == t]
#             per.append(float(np.mean(vs)))
#         per += [0.0] * (4 - len(per))

#         return np.array([uniq, mean_p, sd_p, *per], dtype=np.float32)

#     def get_state(self) -> Dict[str, Any]:
#         """
#         Return the internal state for checkpointing.
#         """
#         return {"records": self.records.copy()}

#     def set_state(self, state: Dict[str, Any]):
#         """
#         Restore state from checkpoint.
#         """
#         self.records = state.get("records", []).copy()
#         if self.debug:
#             print(f"[TradeThesisTracker] State restored ({len(self.records)} records)")
import numpy as np
import datetime
from typing import Any, Tuple, List, Dict, Optional, Union

from modules.core.core import Module

class TradeThesisTracker(Module):
    """
    Tracks the outcome and statistics of each unique trading thesis.
    Includes analytics (win rate, avg, total, etc.) and supports tags/metadata.
    """

    def __init__(self, debug: bool = True):
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        self.records: List[Dict[str, Any]] = []   # Each record is a dict with thesis, pnl, meta, ts

    def step(self, trades: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        if trades:
            for t in trades:
                thesis = t.get("thesis")
                pnl = t.get("pnl", 0.0)
                meta = t.get("meta", {})
                timestamp = t.get("timestamp") or datetime.datetime.now().isoformat()
                if thesis is not None:
                    self.record(thesis, pnl, meta=meta, timestamp=timestamp)

    def record(self, thesis: Any, pnl: float, meta: Optional[Dict[str, Any]] = None, timestamp: Optional[str] = None) -> None:
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        self.records.append({
            "thesis": thesis,
            "pnl": pnl,
            "meta": meta or {},
            "timestamp": timestamp,
        })
        if self.debug:
            print(f"[TradeThesisTracker] Recorded: {thesis} | PnL: {pnl:.2f} | Meta: {meta} | @ {timestamp}")

    def get_observation_components(self, last_n: int = 50) -> np.ndarray:
        """
        Outputs:
            [uniq_theses, mean_pnl, sd_pnl, overall_winrate, per-thesis mean (up to 4)]
            Pads zeros if not enough unique theses.
        """
        if not self.records:
            return np.zeros(8, dtype=np.float32)

        # Use only last_n trades for rolling stats
        recs = self.records[-last_n:] if last_n else self.records
        theses, pnls = zip(*[(r["thesis"], r["pnl"]) for r in recs])
        uniq = len(set(theses))
        mean_p = float(np.mean(pnls))
        sd_p = float(np.std(pnls))
        winrate = float(np.mean([1 if p > 0 else 0 for p in pnls]))

        # per-thesis mean for up to 4 unique theses
        per: List[float] = []
        for t in list(dict.fromkeys(theses))[:4]:
            vs = [r["pnl"] for r in recs if r["thesis"] == t]
            per.append(float(np.mean(vs)))
        per += [0.0] * (4 - len(per))

        return np.array([uniq, mean_p, sd_p, winrate, *per], dtype=np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """Rich stats per thesis for analytics/UI/LLMs."""
        if not self.records:
            return {}
        theses = set(r["thesis"] for r in self.records)
        stats = {}
        for t in theses:
            vals = [r["pnl"] for r in self.records if r["thesis"] == t]
            stats[t] = {
                "count": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "total": float(np.sum(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "winrate": float(np.mean([1 if v > 0 else 0 for v in vals]))
            }
        return stats

    def get_state(self) -> Dict[str, Any]:
        return {"records": [r.copy() for r in self.records]}

    def set_state(self, state: Dict[str, Any]):
        self.records = [r.copy() for r in state.get("records", [])]
        if self.debug:
            print(f"[TradeThesisTracker] State restored ({len(self.records)} records)")

    def export_as_jsonl(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records)

    def get_last_n(self, n=10) -> List[Dict[str, Any]]:
        """Retrieve last n records for LLM/audit."""
        return self.records[-n:]

    # Example: Filter by meta tag (e.g., 'breakout')
    def filter_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        return [r for r in self.records if tag in (r["meta"].get("tags") or [])]
