import os
import json
import logging
import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from modules.core.core import Module

def utcnow() -> str:
    return datetime.datetime.utcnow().isoformat()

class OpponentSimulator(Module):
    """
    Simulates adversarial or randomized market opponents.
    - Modes: "random", "adversarial", "trend-follow"
    - Reproducible via a numpy RNG seed
    - Full JSONL audit trail and standard log file
    """
    MODES = ["random", "adversarial", "trend-follow"]

    def __init__(
        self,
        mode: str = "random",
        intensity: float = 1.0,
        debug: bool = True,
        audit_log_size: int = 100,
        seed: Optional[int] = None
    ):
        super().__init__()
        # Core parameters
        self.mode      = mode if mode in self.MODES else "random"
        self.intensity = float(intensity)
        self.debug     = bool(debug)

        # Reproducible RNG
        self.rng = np.random.RandomState(seed)

        # In-memory audit (capped)
        self.audit_trail: list[Dict[str, Any]] = []
        self._audit_log_size = int(audit_log_size)

        # Persistent JSONL audit file
        self.audit_log_path = "logs/simulation/opponent_simulator_audit.jsonl"
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)

        # Standard logger
        self.logger = logging.getLogger("OpponentSimulator")
        if not self.logger.handlers:
            os.makedirs("logs/simulation", exist_ok=True)
            handler = logging.FileHandler("logs/simulation/opponent_simulator.log")
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            ))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

    def reset(self):
        """Clear any internal state or audit history (if desired)."""
        self.audit_trail.clear()
        # You could also reset a simulation_state here if used.

    def step(self, **kwargs):
        """No-op by default; we expose `.apply()` to perturb data."""
        pass

    def apply(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Apply the chosen simulation mode to each instrument/timeframe.
        Returns a new dict of DataFrames with perturbed 'close' prices.
        """
        out: Dict[str, Dict[str, pd.DataFrame]] = {}
        for inst, tfs in data_dict.items():
            out[inst] = {}
            for tf, df in tfs.items():
                arr = df["close"].values.astype(np.float32)
                vol = df.get("volatility", pd.Series(1.0, index=df.index)).values.astype(np.float32)
                arr_sim = arr.copy()

                # Guard against too-short or flat arrays
                if arr.size < 2 or np.all(arr == arr[0]):
                    trend_dir = 0.0
                else:
                    diffs = np.diff(arr)
                    mean_diff = np.mean(diffs) if diffs.size > 0 else 0.0
                    trend_dir = -np.sign(mean_diff) if self.mode == "adversarial" else 0.0

                rationale = ""
                effect: Dict[str, Any] = {}

                if self.mode == "random":
                    noise = self.rng.randn(len(arr)) * vol * self.intensity
                    arr_sim = arr + noise
                    rationale = (
                        f"Injected Gaussian noise scaled by volatility × "
                        f"intensity={self.intensity:.2f}"
                    )
                    effect = {
                        "noise_std": float(np.std(noise)),
                        "noise_mean": float(np.mean(noise))
                    }

                elif self.mode == "adversarial":
                    sim_vec = trend_dir * vol * self.intensity * np.linspace(0.5, 1.5, len(arr))
                    arr_sim = arr + sim_vec
                    rationale = (
                        f"Adversarial: reversed trend direction "
                        f"amplified by volatility × intensity={self.intensity:.2f}"
                    )
                    effect = {
                        "trend_dir": float(trend_dir),
                        "sim_vec_mean": float(np.mean(sim_vec))
                    }

                elif self.mode == "trend-follow":
                    trend = np.cumsum(self.rng.randn(len(arr))) * vol * 0.2 * self.intensity
                    arr_sim = arr + trend
                    rationale = (
                        f"Trend-follow: added artificial trend scaled by volatility × "
                        f"intensity={self.intensity:.2f}"
                    )
                    effect = {
                        "trend_std": float(np.std(trend)),
                        "trend_mean": float(np.mean(trend))
                    }

                # Build the new DataFrame
                df2 = df.copy()
                df2["close"] = arr_sim
                out[inst][tf] = df2

                # Audit entry
                audit_entry = {
                    "timestamp":    utcnow(),
                    "mode":         self.mode,
                    "instrument":   inst,
                    "timeframe":    tf,
                    "intensity":    self.intensity,
                    "rationale":    rationale,
                    "effect_metrics": effect,
                    "snapshots": {
                        "original_first":   float(arr[0])   if arr.size else None,
                        "simulated_first":  float(arr_sim[0])if arr_sim.size else None,
                        "original_last":    float(arr[-1])  if arr.size else None,
                        "simulated_last":   float(arr_sim[-1])if arr_sim.size else None,
                    },
                }
                self._record_audit(audit_entry)
                if self.debug:
                    self.logger.debug(f"[OpponentSimulator] {inst}/{tf}: {rationale}")

        return out

    def _record_audit(self, entry: Dict[str, Any]):
        # In-memory cap
        self.audit_trail.append(entry)
        if len(self.audit_trail) > self._audit_log_size:
            self.audit_trail = self.audit_trail[-self._audit_log_size:]

        # Append to JSONL file
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_last_audit(self) -> Dict[str, Any]:
        return self.audit_trail[-1] if self.audit_trail else {}

    def get_audit_trail(self, n: int = 10) -> list[Dict[str, Any]]:
        return self.audit_trail[-n:]

    # ── Evolutionary interface ──────────────────────────────────────────
    def mutate(self, std: float = 0.2):
        self.intensity = float(np.clip(self.intensity + self.rng.normal(0, std), 0.05, 10.0))
        if self.rng.rand() < 0.2:
            self.mode = self.rng.choice(self.MODES)

    def crossover(self, other: "OpponentSimulator") -> "OpponentSimulator":
        mode      = self.mode if self.rng.rand() < 0.5 else other.mode
        intensity = self.intensity if self.rng.rand() < 0.5 else other.intensity
        seed      = None  # you could mix seeds if desired
        return OpponentSimulator(mode, intensity, self.debug, self._audit_log_size, seed)

    # ── Observation for RL / pipeline ──────────────────────────────────
    def get_observation_components(self) -> np.ndarray:
        idx = float(self.MODES.index(self.mode))
        return np.array([self.intensity, idx], dtype=np.float32)

    # ── State serialization ────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        return {
            "mode":           self.mode,
            "intensity":      self.intensity,
            "audit_trail":    self.audit_trail.copy(),
        }

    def set_state(self, state: Dict[str, Any]):
        self.mode        = state.get("mode", self.mode)
        self.intensity   = float(state.get("intensity", self.intensity))
        self.audit_trail = state.get("audit_trail", []).copy()
