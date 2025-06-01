import numpy as np
from typing import Any, Dict, List, Optional
import copy
from modules.trading_modes.trading_mode import TradingModeManager

class PositionManager:
    """
    Manages position sizing with volatility-adjusted calculations,
    enhanced risk controls, and dynamic confidence scoring.
    - Evolutionary ready: supports mutation and crossover of key parameters.
    - Audit/prod ready: hot-swap state, deterministic obs, NaN/Inf safety, monkeypatchable for tests.
    """

    def __init__(
        self,
        initial_balance: float,
        instruments: List[str],
        max_pct: float = 0.10,
        max_consecutive_losses: int = 5,
        loss_reduction: float = 0.2,
        max_instrument_concentration: float = 0.25,
        min_volatility: float = 0.015,
        debug: bool = False,
    ):
        # Core params
        self.mode_manager = TradingModeManager(initial_mode="safe", window=50)
        self.initial_balance = float(initial_balance)
        self.instruments = instruments
        self.default_max_pct = float(max_pct)
        self.max_pct = float(max_pct)
        self.debug = debug

        # Loss streak breaker
        self.consecutive_losses = 0
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.loss_reduction = float(loss_reduction)

        # Concentration & volatility floor
        self.max_instrument_concentration = float(max_instrument_concentration)
        self.min_volatility = float(min_volatility)

        # Track open positions: {instrument: {"size": float, "volatility": float}}
        self.open_positions: Dict[str, Dict[str, float]] = {}

        # Hook to the trading environment
        self.env = None  # set via set_env()

        # Optional monkeypatch/test overrides
        self._forced_action = None
        self._forced_conf = None

    # ──────────────────────────────────────────────
    # Evolutionary Logic
    # ──────────────────────────────────────────────

    def mutate(self, std: float = 0.05):
        """
        Mutate risk parameters slightly. Used by evolutionary algorithms.
        """
        self.max_pct += np.random.normal(0, std)
        self.max_pct = np.clip(self.max_pct, 0.01, 0.25)
        self.max_instrument_concentration += np.random.normal(0, std)
        self.max_instrument_concentration = np.clip(self.max_instrument_concentration, 0.05, 0.5)
        self.loss_reduction += np.random.normal(0, std)
        self.loss_reduction = np.clip(self.loss_reduction, 0.05, 1.0)
        self.min_volatility += np.random.normal(0, std)
        self.min_volatility = np.clip(self.min_volatility, 0.001, 0.10)
        self.max_consecutive_losses += int(np.random.choice([-1, 0, 1]))
        self.max_consecutive_losses = int(np.clip(self.max_consecutive_losses, 1, 20))
        if self.debug:
            print("[PositionManager] Mutated parameters.")

    def crossover(self, other: "PositionManager"):
        """
        Cross over parameters with another PositionManager.
        """
        child = copy.deepcopy(self)
        for attr in [
            "max_pct", "max_instrument_concentration", "loss_reduction",
            "min_volatility", "max_consecutive_losses"
        ]:
            if np.random.rand() > 0.5:
                setattr(child, attr, getattr(other, attr))
        if self.debug:
            print("[PositionManager] Crossover complete.")
        return child

    # ──────────────────────────────────────────────

    def set_env(self, env: Any):
        self.env = env

    def reset(self):
        self.max_pct = self.default_max_pct
        self.consecutive_losses = 0
        self.open_positions.clear()
        self._forced_action = None
        self._forced_conf = None

    def step(self, **kwargs):
        env = kwargs.get("env", None)
        if env:
            self.env = env
        if self.env and getattr(self.env, "live_mode", False):
            self._sync_live_positions()
        # Optional: Reset streaks at episode start, etc.

    def _sync_live_positions(self):
        if self.debug:
            print("[PositionManager] Syncing live positions…")
        # TODO: pull actual live positions into self.open_positions

    def calculate_size(
        self,
        volatility: float,
        intensity: float,
        balance: float,
        drawdown: float,
        correlation: Optional[float] = None,
        current_exposure: Optional[float] = None,
    ) -> float:
        """
        volatility: raw vol estimate (floored at self.min_volatility)
        intensity: trading signal in [-1,1]
        balance: current account equity
        drawdown: current drawdown [0..1]
        correlation: optional override for env.get_current_correlation()
        current_exposure: optional override for sum(abs(sizes))/balance
        """

        # 0) Sanitize NaNs before we start
        volatility = float(np.nan_to_num(volatility, nan=self.min_volatility))
        intensity = float(np.nan_to_num(intensity, nan=0.0))
        drawdown = float(np.nan_to_num(drawdown, nan=0.0))

        # 1) Volatility & signal floors
        vol = max(volatility, self.min_volatility)  # enforce vol floor
        inten = np.clip(intensity, -1.0, 1.0)  # clamp signal

        if self.debug:
            print(f"[PositionManager] inputs → raw_vol={volatility:.6f}, raw_inten={intensity:.3f}, raw_dd={drawdown:.3f}")
            print(f"[PositionManager] floored → vol={vol:.6f}, inten={inten:.3f}")

        # 2) Risk‐percent floor
        pct = max(self.max_pct, getattr(self, "min_risk", 0.05))
        risk_cap = balance * pct
        vol_adj_size = risk_cap / vol

        # 3) Drawdown scaling (only in non‐safe mode)
        cur_mode = getattr(self.mode_manager, "current_mode", "safe")
        if cur_mode == "safe":
            dd_factor = 1.0
        else:
            dd_factor = 1.0 - np.clip(drawdown, 0.0, 0.8)

        # 4) Raw lot‐size
        raw = inten * vol_adj_size * dd_factor

        # 5) Correlation penalty
        corr = (
            correlation
            if correlation is not None
            else (self.env.get_current_correlation() if self.env else 0.0)
        )
        corr_pen = 1.0 - min(abs(corr) * 0.5, 1.0)
        raw *= corr_pen

        # 6) Hard clip to pct of account
        max_allow = balance * pct
        size = float(np.clip(raw, -max_allow, max_allow))

        # 7) Enforce a minimum trade‐size on strong signals
        min_size = 0.01 * balance  # e.g. 1% of account
        if abs(size) < min_size and abs(inten) > 0.3:
            size = np.sign(size or inten) * min_size
            if self.debug:
                print(f"[PositionManager] applied min‐size floor → {size:.2f}")

        # 8) Loss‐streak reduction
        if self.consecutive_losses >= self.max_consecutive_losses:
            size *= self.loss_reduction
            if self.debug:
                print(f"[PositionManager] loss‐streak reduction → {size:.2f}")

        # 9) Instrument concentration check
        if current_exposure is None:
            total_expo = sum(abs(v["size"]) for v in self.open_positions.values())
            expo = total_expo / max(balance, 1.0)
        else:
            expo = current_exposure
        if expo > self.max_instrument_concentration:
            if self.debug:
                print(f"[PositionManager] expo {expo:.2%} > cap → zeroing size")
            size = 0.0

        # 10) Final NaN/Inf defense
        size = float(np.nan_to_num(size, nan=0.0, posinf=0.0, neginf=0.0))

        if self.debug:
            print(
                f"[PositionManager] final_size={size:.2f}, vol={vol:.4f}, pct={pct:.3f}, "
                f"corr={corr:.2f}, expo={expo:.2%}, drawdown={drawdown:.2%}"
            )

        return size

    def propose_action(self, obs: Any) -> np.ndarray:
        """
        Build an action vector of (intensity, duration) per instrument,
        where intensity ∈ [–1,1] and duration is normalized to 1.
        The actual lot-size conversion happens later in _execute_trade().
        """
        # Monkeypatch override for test/dev
        if self._forced_action is not None:
            return np.array([self._forced_action] * len(self.instruments) * 2, dtype=np.float32)

        signals: List[float] = []
        for inst in self.instruments:
            # 1) Pull raw signal and clamp to [–1,1]
            raw_inten = self.env.meta_agent.get_intensity(inst) if self.env and hasattr(self.env, "meta_agent") else 0.0
            inten = float(np.clip(raw_inten, -1.0, 1.0))
            duration = 1.0
            if self.debug:
                print(f"[PositionManager] propose_action → {inst}: inten={inten:.3f}, dur={duration}")
            signals.extend([inten, duration])
        return np.array(signals, dtype=np.float32)

    def confidence(self, obs: Any) -> float:
        # Monkeypatch override for test/dev
        if self._forced_conf is not None:
            return float(self._forced_conf)

        dd = getattr(self.env, "current_drawdown", 0.0) if self.env else 0.0
        dd_pen = max(0.0, 1.0 - dd * 1.5)

        vols = [p["volatility"] for p in self.open_positions.values()]
        avg_vol = float(np.mean(vols)) if vols else self.min_volatility
        vol_pen = 1.0 - np.clip(avg_vol / 0.05, 0.0, 1.0)

        tot_sz = sum(abs(p["size"]) for p in self.open_positions.values())
        conc = tot_sz / max(self.env.balance if self.env and hasattr(self.env, "balance") else 1.0, 1.0)
        conc_pen = 1.0 - np.clip(conc / 0.5, 0.0, 1.0)

        perf = (
            self.env.meta_agent.get_observation_components()[0]
            if self.env and hasattr(self.env, "meta_agent")
            else 1.0
        )
        perf_boost = np.clip(perf * 2.0, 0.5, 1.5)

        liq = (
            self.env.liquidity_layer.current_score()
            if self.env and hasattr(self.env, "liquidity_layer")
            else 1.0
        )

        score = dd_pen * vol_pen * perf_boost * liq * conc_pen
        return float(np.clip(score, 0.1, 1.0))

    def force_action(self, value: float):
        self._forced_action = float(value)

    def force_confidence(self, value: float):
        self._forced_conf = float(value)

    def clear_forced(self):
        self._forced_action = None
        self._forced_conf = None

    def get_observation_components(self) -> np.ndarray:
        dd = getattr(self.env, "current_drawdown", 0.0) if self.env else 0.0
        conf = self.confidence(None)
        return np.array([float(dd), float(conf)], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "positions": copy.deepcopy(self.open_positions),
            "max_pct": float(self.max_pct),
            "consecutive_losses": int(self.consecutive_losses),
        }

    def set_state(self, state: Dict[str, Any]):
        self.open_positions = copy.deepcopy(state.get("positions", {}))
        self.max_pct = float(state.get("max_pct", self.default_max_pct))
        self.consecutive_losses = int(state.get("consecutive_losses", 0))
        if self.debug:
            print(
                f"[PositionManager] Restored state: max_pct={self.max_pct}, "
                f"losses={self.consecutive_losses}, positions={self.open_positions}"
            )
