# modules/position.py

import numpy as np
from typing import Any, Dict
from modules.core.core import Module

class PositionManager(Module):
    """
    Manages position sizing with volatility‐adjusted calculations,
    enhanced risk controls, and dynamic confidence scoring.
    """
    def __init__(self, initial_balance: float, max_pct: float = 0.10, debug: bool = False):
        self.initial_balance = initial_balance
        self.default_max_pct = max_pct
        self.max_pct = max_pct
        self.debug = debug

        # risk‐management enhancements
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        self.loss_reduction = 0.2              # shrink size after too many losses
        self.max_instrument_concentration = 0.25  # 25% of equity

        # track open positions as {instrument: {"size":…, "volatility":…, …}, …}
        self.open_positions: Dict[Any, Dict[str, float]] = {}
        self.min_volatility = 0.015  # 1.5% floor

        # will be assigned in step()
        self.env = None

    def reset(self):
        self.max_pct = self.default_max_pct
        self.consecutive_losses = 0
        self.open_positions.clear()

    def step(self, **kwargs):
        # capture env reference for sizing & confidence calls
        env = kwargs.get("env", None)
        if env:
            self.env = env
        if env and env.live_mode:
            self._sync_live_positions()

    def _sync_live_positions(self):
        if self.debug:
            print("[PositionManager] Syncing live positions…")
        # (Your logic to pull live orders into self.open_positions)

    def calculate_size(
        self,
        volatility: float,
        intensity: float,
        balance: float,
        drawdown: float
    ) -> float:
        # 1) floor volatility
        vol = max(float(volatility), self.min_volatility)

        # 2) clamp signal
        clamped_intensity = np.clip(intensity, -1.0, 1.0)

        # 3) basic risk capacity & vol adjustment
        risk_capacity = balance * self.max_pct
        vol_adjusted = risk_capacity / vol

        # 4) drawdown scaling (up to 80% reduction)
        drawdown_factor = 1.0 - np.clip(drawdown, 0.0, 0.8)

        # 5) raw position size
        raw_size = clamped_intensity * vol_adjusted * drawdown_factor

        # 6) correlation penalty
        corr = 0.0
        if self.env is not None:
            corr = self.env.get_current_correlation()
        corr_penalty = 1.0 - min(abs(corr) * 0.5, 1.0)
        raw_size *= corr_penalty

        # 7) clip to absolute max allowed
        max_allowable = balance * self.max_pct
        final_size = np.clip(raw_size, -max_allowable, max_allowable)

        # 8) circuit breaker on consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            final_size *= self.loss_reduction

        # 9) per‐instrument concentration check
        exposure = 0.0
        if self.open_positions and self.env is not None:
            # sum absolute sizes across this instrument
            exposure = sum(abs(p["size"]) for p in self.open_positions.values())
            exposure /= max(1.0, self.env.balance)
        if exposure > self.max_instrument_concentration:
            final_size = 0.0

        if self.debug:
            print(f"[PositionManager] vol={vol:.4f}, drawdown={drawdown:.2%}, corr={corr:.2f}, "
                  f"raw={raw_size:.2f}, final={final_size:.2f}, expo={exposure:.2%}")

        return float(final_size)

    def propose_action(self, obs: Any) -> np.ndarray:
        signals = []
        for inst in self.env.instruments:
            volatility = obs.get(f"{inst}_volatility", self.min_volatility)
            intensity  = obs.get(f"{inst}_intensity", 0.0)
            size = self.calculate_size(
                volatility=volatility,
                intensity=intensity,
                balance=self.env.balance,
                drawdown=self.env.current_drawdown
            )
            signals.extend([size, 0])  # size, duration
        return np.array(signals, dtype=np.float32)

    def confidence(self, obs: Any) -> float:
        """
        Real‐time confidence [0.1–1.0] based on:
         - current drawdown
         - position volatility
         - recent meta-agent performance
         - market liquidity
         - concentration
        """
        # drawdown penalty
        dd = getattr(self.env, "current_drawdown", 0.0)
        drawdown_penalty = max(0.0, 1.0 - dd * 1.5)

        # position volatility
        vols = [p.get("volatility", self.min_volatility) for p in self.open_positions.values()]
        avg_vol = float(np.mean(vols)) if vols else self.min_volatility
        vol_penalty = 1.0 - np.clip(avg_vol / 0.05, 0.0, 1.0)

        # concentration penalty
        sizes = [p.get("size", 0.0) for p in self.open_positions.values()]
        concentration = (sum(abs(s) for s in sizes) / max(1.0, self.env.balance)) if self.env else 0.0
        conc_penalty = 1.0 - np.clip(concentration / 0.5, 0.0, 1.0)

        # recent performance boost from meta-agent
        perf = 1.0
        if self.env and hasattr(self.env, "meta_agent"):
            perf = self.env.meta_agent.get_observation_components()[0]
        perf_boost = np.clip(perf * 2.0, 0.5, 1.5)

        # liquidity score
        liquidity = 1.0
        if self.env and hasattr(self.env, "liquidity_layer"):
            liquidity = getattr(self.env.liquidity_layer, "current_score", 1.0)

        score = drawdown_penalty * vol_penalty * perf_boost * liquidity * conc_penalty
        score = np.clip(score, 0.1, 1.0)
        return float(score)

    def get_observation_components(self) -> np.ndarray:
        # expose current drawdown and last confidence
        conf = self.confidence(None)
        return np.array([self.env.current_drawdown, conf], dtype=np.float32)
