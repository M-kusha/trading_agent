#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
# File: modules/meta/ppo_observation_builder.py
# Unified PPO Observation Builder (v4.0)
#
# Single source of truth for PPO observation construction.
# Used identically in TRAINING (ModernTradingEnv) and LIVE (PPOAgent).
#
# Design:
# - M15 is the PRIMARY trading/decision timeframe
# - H1/H4/D1 are CONTEXT timeframes (filters/regime)
# - Voting/committee signals are included for informed decisions
# - World Model & Trading Mode features are integrated (v4.0)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

# Import canonical timeframe constants
try:
    from modules.voting.core.constants import (
        PRIMARY_TIMEFRAME,
        CONTEXT_TIMEFRAMES,
        SUPPORTED_TIMEFRAMES,
    )
except ImportError:
    PRIMARY_TIMEFRAME = "M15"
    CONTEXT_TIMEFRAMES = ("H1", "H4", "D1")
    SUPPORTED_TIMEFRAMES = ("M15", "H1", "H4", "D1")


# ═══════════════════════════════════════════════════════════════════
# OBSERVATION SCHEMA (v4.0) - World Model & Trading Mode Integration
# ═══════════════════════════════════════════════════════════════════
#
# Total: 64 dimensions (expanded for world model and trading mode)
#
# [0-9]   M15 Price Features (PRIMARY) - 10 dims
# [10-15] Higher TF Context (H1/H4/D1 aggregated) - 6 dims
# [16-23] Voting Expert Signals - 8 dims
# [24-31] Committee/Consensus Metrics - 8 dims
# [32-39] Risk/Memory Signals - 8 dims
# [40-47] Account/Position State - 8 dims
# [48-55] World Model Predictions - 8 dims (NEW v4.0)
# [56-63] Trading Mode State - 8 dims (NEW v4.0)
# ═══════════════════════════════════════════════════════════════════

# Observation dimension constants
PPO_OBS_VERSION = "4.0"
PPO_OBS_SIZE = 64

# Feature group indices for debugging/analysis
FEATURE_GROUPS: Dict[str, tuple[int, int]] = {
    "m15_price": (0, 10),        # M15 primary features
    "htf_context": (10, 16),     # H1/H4/D1 context
    "voting": (16, 24),          # Expert signals
    "committee": (24, 32),       # Committee metrics
    "risk": (32, 40),            # Risk/memory
    "account": (40, 48),         # Account state
    "world_model": (48, 56),     # World model predictions (NEW)
    "trading_mode": (56, 64),    # Trading mode state (NEW)
}


@dataclass
class PPOObservationConfig:
    """Configuration for PPO observation builder."""
    obs_size: int = PPO_OBS_SIZE
    version: str = PPO_OBS_VERSION

    # Normalization parameters
    price_lookback: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    trend_lookback: int = 20
    momentum_lookback: int = 10

    # Feature scaling
    max_drawdown_clip: float = 0.5
    max_danger_zones: int = 10
    max_trades_per_day: int = 20

    # World model parameters (v4.0)
    # If overall prediction confidence is below this threshold,
    # directional / volatility features are aggressively neutralized.
    prediction_confidence_threshold: float = 0.5
    scenario_confidence_threshold: float = 0.5  # reserved for future use


class PPOObservationBuilder:
    """
    Unified PPO Observation Builder.

    Constructs identical observation vectors for both TRAINING and LIVE.
    M15 is the PRIMARY timeframe; H1/H4/D1 are context only.
    """

    def __init__(self, config: Optional[PPOObservationConfig] = None) -> None:
        self.config = config or PPOObservationConfig()
        # Guard against accidental obs_size drift – the feature layout is hard-coded
        if self.config.obs_size != PPO_OBS_SIZE:
            # Force to canonical size; layout is fixed by design
            self.config.obs_size = PPO_OBS_SIZE
        self._eps: float = 1e-8

    @property
    def obs_size(self) -> int:
        return self.config.obs_size

    @property
    def version(self) -> str:
        return self.config.version

    def build(
        self,
        # Market data
        market_data: Optional[Dict[str, Any]] = None,
        # Voting state
        expert_signals: Optional[Dict[str, Any]] = None,
        committee_state: Optional[Dict[str, Any]] = None,
        # Risk state
        risk_state: Optional[Dict[str, Any]] = None,
        memory_state: Optional[Dict[str, Any]] = None,
        # Account state
        account_state: Optional[Dict[str, Any]] = None,
        # World Model state (v4.0)
        world_model_state: Optional[Dict[str, Any]] = None,
        # Trading Mode state (v4.0)
        trading_mode_state: Optional[Dict[str, Any]] = None,
        # SmartInfoBus (optional, for convenience)
        smart_bus: Optional[Any] = None,
        module_name: str = "PPOObservationBuilder",
    ) -> np.ndarray:
        """
        Build the unified PPO observation vector (v4.0).

        Can be called with explicit dicts OR with a SmartInfoBus reference.
        If smart_bus is provided, it will fetch missing data from the bus.

        Returns:
            np.ndarray of shape (obs_size,) with dtype float32
        """
        obs = np.zeros(self.config.obs_size, dtype=np.float32)

        # If smart_bus provided, fetch data from bus for any missing fields
        if smart_bus is not None:
            market_data = market_data or self._fetch_market_data(smart_bus, module_name)
            expert_signals = expert_signals or self._fetch_expert_signals(smart_bus, module_name)
            committee_state = committee_state or self._fetch_committee_state(smart_bus, module_name)
            risk_state = risk_state or self._fetch_risk_state(smart_bus, module_name)
            memory_state = memory_state or self._fetch_memory_state(smart_bus, module_name)
            account_state = account_state or self._fetch_account_state(smart_bus, module_name)
            world_model_state = world_model_state or self._fetch_world_model_state(smart_bus, module_name)
            trading_mode_state = trading_mode_state or self._fetch_trading_mode_state(smart_bus, module_name)

        # Build each feature group
        obs[0:10] = self._build_m15_features(market_data)
        obs[10:16] = self._build_htf_context(market_data)
        obs[16:24] = self._build_voting_features(expert_signals)
        obs[24:32] = self._build_committee_features(committee_state, expert_signals)
        obs[32:40] = self._build_risk_features(risk_state, memory_state, account_state)
        obs[40:48] = self._build_account_features(account_state)
        obs[48:56] = self._build_world_model_features(world_model_state)
        obs[56:64] = self._build_trading_mode_features(trading_mode_state)

        # Sanitize NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs

    def build_for_instrument(
        self,
        instrument: str,
        # Market data
        market_data: Optional[Dict[str, Any]] = None,
        # Voting state
        expert_signals: Optional[Dict[str, Any]] = None,
        committee_state: Optional[Dict[str, Any]] = None,
        # Risk state
        risk_state: Optional[Dict[str, Any]] = None,
        memory_state: Optional[Dict[str, Any]] = None,
        # Account state
        account_state: Optional[Dict[str, Any]] = None,
        # World Model state (v4.0)
        world_model_state: Optional[Dict[str, Any]] = None,
        # Trading Mode state (v4.0)
        trading_mode_state: Optional[Dict[str, Any]] = None,
        # SmartInfoBus (optional)
        smart_bus: Optional[Any] = None,
        module_name: str = "PPOObservationBuilder",
    ) -> np.ndarray:
        """
        Build observation vector for a SPECIFIC instrument (v4.0).

        This method extracts instrument-specific market/risk/account data while using
        the same observation structure as the global build() method.

        Args:
            instrument: The instrument symbol (e.g., "XAUUSD", "EURUSD")

        Returns:
            np.ndarray of shape (obs_size,) with dtype float32
        """
        obs = np.zeros(self.config.obs_size, dtype=np.float32)

        # If smart_bus provided, fetch data from bus
        if smart_bus is not None:
            market_data = market_data or self._fetch_market_data_for_instrument(
                smart_bus, module_name, instrument
            )
            expert_signals = expert_signals or self._fetch_expert_signals_for_instrument(
                smart_bus, module_name, instrument
            )
            committee_state = committee_state or self._fetch_committee_state(smart_bus, module_name)
            risk_state = risk_state or self._fetch_risk_state_for_instrument(
                smart_bus, module_name, instrument
            )
            memory_state = memory_state or self._fetch_memory_state(smart_bus, module_name)
            account_state = account_state or self._fetch_account_state_for_instrument(
                smart_bus, module_name, instrument
            )
            world_model_state = world_model_state or self._fetch_world_model_state(smart_bus, module_name)
            trading_mode_state = trading_mode_state or self._fetch_trading_mode_state(smart_bus, module_name)

        # Build each feature group (same structure as global)
        obs[0:10] = self._build_m15_features_for_instrument(market_data, instrument)
        obs[10:16] = self._build_htf_context_for_instrument(market_data, instrument)
        obs[16:24] = self._build_voting_features_for_instrument(expert_signals, instrument)
        obs[24:32] = self._build_committee_features(committee_state, expert_signals)
        obs[32:40] = self._build_risk_features(risk_state, memory_state, account_state)
        obs[40:48] = self._build_account_features(account_state)
        obs[48:56] = self._build_world_model_features(world_model_state)
        obs[56:64] = self._build_trading_mode_features(trading_mode_state)

        # Sanitize NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs

    # ─────────────────────────────────────────────────────────────
    # Instrument-aware fetchers / builders
    # ─────────────────────────────────────────────────────────────

    def _fetch_market_data_for_instrument(
        self, bus: Any, module: str, instrument: str
    ) -> Dict[str, Any]:
        """Fetch instrument-specific market data from SmartInfoBus."""
        try:
            # Try instrument-specific multi-timeframe data
            mtf = bus.get("multi_timeframe_data", module)
            if isinstance(mtf, dict):
                if instrument in mtf and isinstance(mtf[instrument], dict):
                    return mtf[instrument]
                return mtf

            # Try instrument-specific market_data
            md = bus.get("market_data", module)
            if isinstance(md, dict):
                if instrument in md and isinstance(md[instrument], dict):
                    return md[instrument]
                return md

            # Try historical_prices
            hp = bus.get("historical_prices", module)
            if isinstance(hp, dict):
                if instrument in hp and isinstance(hp[instrument], dict):
                    return hp[instrument]
                return hp
        except Exception:
            pass
        return {}

    def _fetch_expert_signals_for_instrument(
        self, bus: Any, module: str, instrument: str
    ) -> Dict[str, Any]:
        """Fetch instrument-specific expert signals."""
        global_signals = self._fetch_expert_signals(bus, module)
        experts = global_signals.get("experts", {})

        if not isinstance(experts, dict):
            return global_signals

        # Check if any expert has per-instrument data
        for expert_name, sig in list(experts.items()):
            if isinstance(sig, dict) and "instruments" in sig:
                instruments_map = sig.get("instruments", {})
                if isinstance(instruments_map, dict):
                    inst_sig = instruments_map.get(instrument, sig)
                    experts[expert_name] = inst_sig

        global_signals["experts"] = experts
        return global_signals

    def _fetch_risk_state_for_instrument(
        self, bus: Any, module: str, instrument: str
    ) -> Dict[str, Any]:
        """Fetch instrument-specific risk state."""
        risk = self._fetch_risk_state(bus, module)

        # Check for per-instrument risk
        portfolio_risk = risk.get("portfolio_risk", {})
        if isinstance(portfolio_risk, dict) and "instruments" in portfolio_risk:
            inst_risk = portfolio_risk["instruments"].get(instrument, {})
            # Attach a generic per-instrument view; downstream can decide how to use it
            risk["instrument_risk"] = inst_risk

        return risk

    def _fetch_account_state_for_instrument(
        self, bus: Any, module: str, instrument: str
    ) -> Dict[str, Any]:
        """Fetch instrument-specific account/position state."""
        account = self._fetch_account_state(bus, module)

        positions = bus.get("positions", module) or {}
        if isinstance(positions, dict):
            inst_pos = positions.get(instrument)
            if isinstance(inst_pos, dict):
                account["position_direction"] = inst_pos.get("direction", 0)
                account["position_size"] = inst_pos.get("size", 0.0)
                account["unrealized_pnl"] = inst_pos.get("unrealized_pnl", 0.0)

        return account

    def _build_m15_features_for_instrument(
        self, market_data: Optional[Dict[str, Any]], instrument: str
    ) -> np.ndarray:
        """Build M15 features for a specific instrument."""
        if market_data and instrument in market_data:
            inst_data = market_data[instrument]
            if isinstance(inst_data, dict):
                return self._build_m15_features(inst_data)
        return self._build_m15_features(market_data)

    def _build_htf_context_for_instrument(
        self, market_data: Optional[Dict[str, Any]], instrument: str
    ) -> np.ndarray:
        """Build HTF context for a specific instrument."""
        if market_data and instrument in market_data:
            inst_data = market_data[instrument]
            if isinstance(inst_data, dict):
                return self._build_htf_context(inst_data)
        return self._build_htf_context(market_data)

    def _build_voting_features_for_instrument(
        self, expert_signals: Optional[Dict[str, Any]], instrument: str  # instrument kept for symmetry
    ) -> np.ndarray:
        """Build voting features for a specific instrument."""
        # Same structure, but expert_signals may already be instrument-filtered.
        return self._build_voting_features(expert_signals)

    # ─────────────────────────────────────────────────────────────
    # M15 Price Features (PRIMARY) - 10 dims
    # ─────────────────────────────────────────────────────────────

    def _build_m15_features(self, market_data: Optional[Dict[str, Any]]) -> np.ndarray:
        """Build M15 primary price features."""
        feats = np.zeros(10, dtype=np.float32)

        if not market_data:
            return feats

        # Extract M15 OHLCV data
        m15 = self._extract_timeframe_data(market_data, PRIMARY_TIMEFRAME)
        if m15 is None or not isinstance(m15, dict):
            return feats

        close = m15.get("close")
        high = m15.get("high")
        low = m15.get("low")
        open_ = m15.get("open")
        volume = m15.get("volume")

        if close is None:
            return feats

        close_arr = np.asarray(close, dtype=np.float64)
        if close_arr.size < 2:
            return feats

        high_arr = np.asarray(high, dtype=np.float64) if high is not None else close_arr
        low_arr = np.asarray(low, dtype=np.float64) if low is not None else close_arr
        open_arr = np.asarray(open_, dtype=np.float64) if open_ is not None else close_arr
        vol_arr = np.asarray(volume, dtype=np.float64) if volume is not None else np.ones_like(close_arr)

        # Use last values
        c = float(close_arr[-1])
        h = float(high_arr[-1])
        l = float(low_arr[-1])
        o = float(open_arr[-1])
        v = float(vol_arr[-1])

        # Lookback means
        lb = min(self.config.price_lookback, close_arr.size)
        mean_close = float(np.mean(close_arr[-lb:])) if lb > 0 else c
        mean_vol = float(np.mean(vol_arr[-lb:])) if lb > 0 else max(v, 1.0)

        # [0] close_rel: close relative to mean
        feats[0] = float((c / max(mean_close, self._eps)) - 1.0)

        # [1] range_rel: bar range relative to close
        feats[1] = float((h - l) / max(abs(c), self._eps))

        # [2] change_rel: bar change
        feats[2] = float((c - o) / max(abs(o), self._eps))

        # [3] vol_norm: volume relative to mean
        feats[3] = float(np.clip((v / max(mean_vol, self._eps)) - 1.0, -1.0, 2.0))

        # [4] rsi_norm: RSI normalized to [-1, 1]
        rsi = self._compute_rsi(close_arr, self.config.rsi_period)
        feats[4] = float((rsi - 50.0) / 50.0)  # Maps [0,100] to [-1,1]

        # [5] macd_norm: MACD histogram (simplified)
        macd_hist = self._compute_macd_histogram(close_arr)
        denom = max(abs(c) * 0.01, self._eps)
        feats[5] = float(np.clip(macd_hist / denom, -1.0, 1.0))

        # [6] atr_norm: ATR relative to price
        atr = self._compute_atr(high_arr, low_arr, close_arr, self.config.atr_period)
        feats[6] = float(np.clip(atr / max(c, self._eps), 0.0, 0.1) * 10.0)  # [0,1]

        # [7] trend_slope: linear regression slope
        if close_arr.size >= self.config.trend_lookback:
            slope = self._compute_slope(close_arr[-self.config.trend_lookback:])
            denom_slope = max(abs(c) * 0.001, self._eps)
            feats[7] = float(np.clip(slope / denom_slope, -1.0, 1.0))

        # [8] momentum: rate of change
        if close_arr.size >= self.config.momentum_lookback:
            base = float(close_arr[-self.config.momentum_lookback])
            roc = (c - base) / max(abs(base), self._eps)
            feats[8] = float(np.clip(roc * 10.0, -1.0, 1.0))

        # [9] volatility: rolling std of returns
        if close_arr.size >= 20:
            prev = close_arr[-20:-1]
            curr = close_arr[-19:]
            returns = (curr - prev) / prev
            vol = float(np.std(returns))
            feats[9] = float(np.clip(vol * 100.0, 0.0, 1.0))

        return feats

    # ─────────────────────────────────────────────────────────────
    # Higher TF Context (H1/H4/D1) - 6 dims
    # ─────────────────────────────────────────────────────────────

    def _build_htf_context(self, market_data: Optional[Dict[str, Any]]) -> np.ndarray:
        """Build higher timeframe context features."""
        feats = np.zeros(6, dtype=np.float32)

        if not market_data:
            return feats

        trends: list[float] = []
        vols: list[float] = []
        moms: list[float] = []

        for i, tf in enumerate(CONTEXT_TIMEFRAMES):  # H1, H4, D1
            tf_data = self._extract_timeframe_data(market_data, tf)
            if tf_data is None or not isinstance(tf_data, dict):
                continue

            close = tf_data.get("close")
            high = tf_data.get("high")
            low = tf_data.get("low")

            if close is None:
                continue

            close_arr = np.asarray(close, dtype=np.float64)
            if close_arr.size < 5:
                continue

            # Trend direction
            trend = self._compute_trend_direction(close_arr)
            trends.append(trend)

            # Store individual HTF trend (first 3 dims)
            if i < 3:
                feats[i] = float(trend)

            # Volatility
            if high is not None and low is not None:
                high_arr = np.asarray(high, dtype=np.float64)
                low_arr = np.asarray(low, dtype=np.float64)
                atr = self._compute_atr(
                    high_arr,
                    low_arr,
                    close_arr,
                    min(14, close_arr.size - 1),
                )
                vols.append(float(atr / max(close_arr[-1], self._eps)))

            # Momentum
            if close_arr.size >= 5:
                base = float(close_arr[-5])
                mom = (float(close_arr[-1]) - base) / max(abs(base), self._eps)
                moms.append(mom)

        # [3] htf_alignment: do higher TFs agree?
        if len(trends) >= 2:
            signs = [np.sign(t) for t in trends if abs(t) > 0.1]
            if len(signs) >= 2:
                feats[3] = float(1.0 if len(set(signs)) == 1 else -abs(np.mean(trends)))

        # [4] htf_volatility: average volatility
        if vols:
            feats[4] = float(np.clip(np.mean(vols) * 100.0, 0.0, 1.0))

        # [5] htf_momentum: average momentum
        if moms:
            feats[5] = float(np.clip(np.mean(moms) * 10.0, -1.0, 1.0))

        return feats

    # ─────────────────────────────────────────────────────────────
    # Voting Expert Signals - 8 dims
    # ─────────────────────────────────────────────────────────────

    def _build_voting_features(self, expert_signals: Optional[Dict[str, Any]]) -> np.ndarray:
        """Build voting expert features."""
        feats = np.zeros(8, dtype=np.float32)

        if not expert_signals:
            return feats

        experts = expert_signals.get("experts", {})
        if not isinstance(experts, dict):
            return feats

        expert_names = ["trend", "momentum", "theme", "seasonality"]
        for i, name in enumerate(expert_names):
            sig = experts.get(name, {})
            if not isinstance(sig, dict):
                sig = {}

            # Direction
            direction = self._extract_direction(sig.get("proposal", "flat"))
            feats[i * 2] = float(direction)

            # Confidence
            conf_raw = sig.get("confidence", 0.0)
            try:
                conf_val = float(conf_raw)
            except (TypeError, ValueError):
                conf_val = 0.0
            feats[i * 2 + 1] = float(np.clip(conf_val, 0.0, 1.0))

        return feats

    # ─────────────────────────────────────────────────────────────
    # Committee/Consensus Metrics - 8 dims
    # ─────────────────────────────────────────────────────────────

    def _build_committee_features(
        self,
        committee_state: Optional[Dict[str, Any]],
        expert_signals: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Build committee/consensus features."""
        feats = np.zeros(8, dtype=np.float32)

        committee_state = committee_state or {}
        expert_signals = expert_signals or {}

        # [0] committee_dir
        action = committee_state.get("action", "hold")
        feats[0] = float(self._extract_direction(action))

        # [1] committee_conf
        feats[1] = float(np.clip(committee_state.get("confidence", 0.5), 0.0, 1.0))

        # [2] consensus_score
        feats[2] = float(np.clip(committee_state.get("consensus_score", 0.5), 0.0, 1.0))

        # [3] expert_agreement: do experts agree with committee?
        committee_dir = feats[0]
        expert_dirs: list[float] = []
        experts = expert_signals.get("experts", {}) if isinstance(expert_signals, dict) else {}
        if isinstance(experts, dict):
            for name in ["trend", "momentum", "theme", "seasonality"]:
                sig = experts.get(name, {})
                if not isinstance(sig, dict):
                    continue
                d = self._extract_direction(sig.get("proposal", "flat"))
                if abs(d) > 0.1:
                    expert_dirs.append(d)

        if expert_dirs and abs(committee_dir) > 0.1:
            agreement = sum(1 for d in expert_dirs if d * committee_dir > 0) / len(expert_dirs)
            feats[3] = float(agreement * 2.0 - 1.0)  # Map [0,1] to [-1,1]

        # [4] fragility
        feats[4] = float(np.clip(committee_state.get("fragility", 0.5), 0.0, 1.0))

        # [5] regime_encoded
        market = expert_signals.get("market", {}) if isinstance(expert_signals, dict) else {}
        regime = (market.get("regime", "unknown") if isinstance(market, dict) else "unknown")
        regime_map = {
            "trending": 0.8,
            "uptrend": 0.8,
            "downtrend": -0.8,
            "mean_reverting": 0.0,
            "ranging": 0.0,
            "volatile": -0.5,
            "unknown": 0.0,
        }
        feats[5] = float(regime_map.get(str(regime).lower(), 0.0))

        # [6] regime_strength
        regime_strength = 0.5
        if isinstance(market, dict):
            regime_strength = float(market.get("regime_strength", 0.5))
        feats[6] = float(np.clip(regime_strength, 0.0, 1.0))

        # [7] vote_spread: spread between long/short expert votes
        long_count = sum(1 for d in expert_dirs if d > 0.1)
        short_count = sum(1 for d in expert_dirs if d < -0.1)
        total = max(len(expert_dirs), 1)
        feats[7] = float(abs(long_count - short_count) / total)

        return feats

    # ─────────────────────────────────────────────────────────────
    # Risk/Memory Signals - 8 dims
    # ─────────────────────────────────────────────────────────────

    def _build_risk_features(
        self,
        risk_state: Optional[Dict[str, Any]],
        memory_state: Optional[Dict[str, Any]],
        account_state: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Build risk and memory features."""
        feats = np.zeros(8, dtype=np.float32)

        risk_state = risk_state or {}
        memory_state = memory_state or {}
        account_state = account_state or {}

        # [0] memory_gate
        memory_gate = memory_state.get("memory_gate", 1.0)
        if isinstance(memory_gate, dict):
            memory_gate = memory_gate.get("risk_multiplier", 1.0)
        try:
            mem_val = float(memory_gate)
        except (TypeError, ValueError):
            mem_val = 1.0
        feats[0] = float(np.clip(mem_val, 0.0, 1.0))

        # [1] danger_zone_count (normalized)
        danger_zones = memory_state.get("danger_zones", {})
        if isinstance(danger_zones, dict):
            count = int(danger_zones.get("zone_count", 0))
        elif isinstance(danger_zones, list):
            count = len(danger_zones)
        else:
            count = 0
        feats[1] = float(np.clip(count / max(self.config.max_danger_zones, 1), 0.0, 1.0))

        # [2] drawdown
        drawdown = float(account_state.get("current_drawdown", 0.0))
        feats[2] = float(np.clip(drawdown / max(self.config.max_drawdown_clip, self._eps), 0.0, 1.0))

        # [3] balance_ratio
        balance = float(account_state.get("balance", 100000.0))
        initial = float(account_state.get("initial_balance", 100000.0))
        feats[3] = float(np.clip(balance / max(initial, 1.0), 0.0, 2.0))

        # [4] position_heat (exposure)
        portfolio_risk = risk_state.get("portfolio_risk", {})
        exposure = 0.0
        if isinstance(portfolio_risk, dict):
            exposure = float(portfolio_risk.get("total_exposure", 0.0))
        feats[4] = float(np.clip(exposure, 0.0, 1.0))

        # [5] risk_budget
        risk_budget = risk_state.get("risk_budget", 1.0)
        try:
            rb_val = float(risk_budget)
        except (TypeError, ValueError):
            rb_val = 1.0
        feats[5] = float(np.clip(rb_val, 0.0, 1.0))

        # [6] win_rate_recent
        win_rate = account_state.get("win_rate", 0.5)
        try:
            win_val = float(win_rate)
        except (TypeError, ValueError):
            win_val = 0.5
        feats[6] = float(np.clip(win_val, 0.0, 1.0))

        # [7] pnl_momentum
        pnl_trend = account_state.get("pnl_trend", 0.0)
        try:
            pnl_val = float(pnl_trend)
        except (TypeError, ValueError):
            pnl_val = 0.0
        feats[7] = float(np.clip(pnl_val, -1.0, 1.0))

        return feats

    # ─────────────────────────────────────────────────────────────
    # Account/Position State - 8 dims
    # ─────────────────────────────────────────────────────────────

    def _build_account_features(self, account_state: Optional[Dict[str, Any]]) -> np.ndarray:
        """Build account and position state features."""
        feats = np.zeros(8, dtype=np.float32)

        account_state = account_state or {}

        # [0] step_progress
        step = int(account_state.get("current_step", 0))
        max_steps = int(account_state.get("max_steps", 10000))
        feats[0] = float(np.clip(step / max(max_steps, 1), 0.0, 1.0))

        # [1] episode_return (normalized)
        episode_return = account_state.get("episode_return", 0.0)
        try:
            ep_val = float(episode_return)
        except (TypeError, ValueError):
            ep_val = 0.0
        feats[1] = float(np.clip(ep_val / 100.0, -1.0, 1.0))

        # [2] position_dir
        position_dir = account_state.get("position_direction", 0)
        try:
            pos_dir_val = float(position_dir)
        except (TypeError, ValueError):
            pos_dir_val = 0.0
        feats[2] = float(np.clip(pos_dir_val, -1.0, 1.0))

        # [3] position_size
        position_size = account_state.get("position_size", 0.0)
        try:
            pos_size_val = float(position_size)
        except (TypeError, ValueError):
            pos_size_val = 0.0
        feats[3] = float(np.clip(pos_size_val, 0.0, 1.0))

        # [4] unrealized_pnl (normalized)
        unrealized = account_state.get("unrealized_pnl", 0.0)
        initial = float(account_state.get("initial_balance", 100000.0))
        try:
            unreal_val = float(unrealized)
        except (TypeError, ValueError):
            unreal_val = 0.0
        feats[4] = float(
            np.clip(unreal_val / max(initial * 0.01, 1.0), -1.0, 1.0)
        )

        # [5] time_in_position (normalized)
        time_in_pos = account_state.get("time_in_position", 0)
        try:
            tip_val = float(time_in_pos)
        except (TypeError, ValueError):
            tip_val = 0.0
        feats[5] = float(np.clip(tip_val / 100.0, 0.0, 1.0))

        # [6] trades_today (normalized)
        trades = account_state.get("trades_today", 0)
        try:
            trades_val = float(trades)
        except (TypeError, ValueError):
            trades_val = 0.0
        feats[6] = float(
            np.clip(trades_val / max(self.config.max_trades_per_day, 1), 0.0, 1.0)
        )

        # [7] last_action
        last_action = account_state.get("last_action", 0.0)
        try:
            la_val = float(last_action)
        except (TypeError, ValueError):
            la_val = 0.0
        feats[7] = float(np.clip(la_val, -1.0, 1.0))

        return feats

    # ─────────────────────────────────────────────────────────────
    # World Model Features (v4.0) - 8 dims
    # ─────────────────────────────────────────────────────────────

    def _build_world_model_features(self, world_model_state: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Build world model prediction features (v4.0).

        Features from EnhancedWorldModel:
        [0] prediction_confidence: Model confidence (0-1)
        [1] predicted_price_change_m15: M15 price direction (PRIMARY, -1 to 1)
        [2] predicted_price_change_weighted: Weighted avg across timeframes (-1 to 1)
        [3] predicted_volatility: Expected volatility level (0-1)
        [4] predicted_regime: Regime prediction (encoded)
        [5] model_trained: Is model trained (0 or 1)
        [6] scenario_bullish_prob: Bullish scenario probability
        [7] stability_score: Model stability/reliability
        """
        feats = np.zeros(8, dtype=np.float32)

        if not world_model_state or not isinstance(world_model_state, dict):
            return feats

        # Extract predictions
        predictions = world_model_state.get("market_predictions", world_model_state)
        if not isinstance(predictions, dict):
            predictions = {}

        # Get latest_predictions if nested
        latest = predictions.get("latest_predictions", predictions)

        # [0] prediction_confidence
        base_conf = predictions.get("model_confidence", latest.get("confidence", 0.0))
        # Allow an external prediction_confidence block to refine this
        extra_conf = world_model_state.get("prediction_confidence")
        if extra_conf is not None:
            try:
                if isinstance(extra_conf, dict):
                    # e.g. {"current_confidence": 0.7}
                    extra_conf_val = float(extra_conf.get("current_confidence", base_conf))
                else:
                    extra_conf_val = float(extra_conf)
                base_conf = (float(base_conf) + extra_conf_val) / 2.0
            except (TypeError, ValueError):
                pass

        try:
            feats[0] = float(np.clip(float(base_conf), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[0] = 0.0

        # [1] predicted_price_change_m15 (PRIMARY - from price_changes[0])
        # [2] predicted_price_change_weighted (weighted across M15/H1/H4/D1)
        price_changes = latest.get("price_changes", predictions.get("price_changes", []))
        if isinstance(price_changes, (list, np.ndarray)) and len(price_changes) > 0:
            try:
                # M15 is index 0 (primary decision timeframe)
                m15_change = float(price_changes[0]) if len(price_changes) > 0 else 0.0
                feats[1] = float(np.clip(m15_change * 100.0, -1.0, 1.0))

                # Weighted average: M15=0.5, H1=0.25, H4=0.15, D1=0.10
                if len(price_changes) >= 4:
                    weighted = (
                        float(price_changes[0]) * 0.5 +
                        float(price_changes[1]) * 0.25 +
                        float(price_changes[2]) * 0.15 +
                        float(price_changes[3]) * 0.10
                    )
                else:
                    weighted = m15_change
                feats[2] = float(np.clip(weighted * 100.0, -1.0, 1.0))
            except (TypeError, ValueError):
                feats[1] = 0.0
                feats[2] = 0.0

        # [3] predicted_volatility
        vol_preds = latest.get("volatility_predictions", predictions.get("volatility_predictions", []))
        if isinstance(vol_preds, (list, np.ndarray)) and len(vol_preds) > 0:
            try:
                # Use first volatility (M15)
                m15_vol = float(vol_preds[0])
                feats[3] = float(np.clip(m15_vol, 0.0, 1.0))
            except (TypeError, ValueError):
                feats[3] = 0.5
        else:
            feats[3] = 0.5  # Default medium volatility

        # [4] predicted_regime (encoded: trending_up=0.8, trending_down=-0.8, volatile=0, ranging=0.3)
        regime_probs = latest.get("regime_probabilities", predictions.get("regime_probabilities", []))
        predicted_regime = latest.get("predicted_regime", predictions.get("predicted_regime", -1))
        # Regime classes: 0=trending_up, 1=trending_down, 2=ranging, 3=volatile
        regime_map = {0: 0.8, 1: -0.8, 2: 0.3, 3: 0.0}
        if isinstance(predicted_regime, int) and predicted_regime in regime_map:
            feats[4] = regime_map[predicted_regime]
        elif isinstance(regime_probs, (list, np.ndarray)) and len(regime_probs) >= 4:
            try:
                regime_idx = int(np.argmax(regime_probs))
                feats[4] = regime_map.get(regime_idx, 0.0)
            except (TypeError, ValueError):
                feats[4] = 0.0

        # [5] model_trained (binary)
        is_trained = predictions.get("is_trained", latest.get("model_trained", False))
        feats[5] = 1.0 if is_trained else 0.0

        # [6] scenario_bullish_prob (from scenario_generation if available)
        scenarios = world_model_state.get("scenario_generation", world_model_state.get("scenarios", {}))
        if isinstance(scenarios, dict):
            # Extract scenarios list and compute bullish probability
            scenarios_list = scenarios.get("scenarios", [])
            bullish_prob = 0.5
            if isinstance(scenarios_list, list) and len(scenarios_list) > 0:
                bullish_total = 0.0
                for s in scenarios_list:
                    if not isinstance(s, dict):
                        continue
                    try:
                        prob = float(s.get("probability", 0.0) or 0.0)
                        outcome = float(s.get("outcome", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                    if outcome > 0:
                        bullish_total += prob
                bullish_prob = min(1.0, bullish_total)
            else:
                bullish_prob = scenarios.get("bullish_probability", 0.5)
            try:
                feats[6] = float(np.clip(float(bullish_prob), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[6] = 0.5
        else:
            feats[6] = 0.5

        # [7] stability_score
        stability = predictions.get("stability_score", predictions.get("prediction_quality", 0.5))
        try:
            feats[7] = float(np.clip(float(stability), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[7] = 0.5

        # ── CONFIDENCE-GATED WORLD MODEL CONTRIBUTION ─────────────
        # If overall confidence is low, aggressively neutralize directional signal
        # while still exposing "is_trained" and a small hint via stability.
        conf_val = float(feats[0])
        if conf_val < self.config.prediction_confidence_threshold or not bool(is_trained):
            # Keep confidence + trained flag + stability; flatten directional parts
            feats[1] = 0.0  # m15 direction
            feats[2] = 0.0  # weighted direction
            feats[3] = 0.5  # neutral volatility
            feats[4] = 0.0  # neutral regime
            # Scenario prob remains as soft prior, but not extreme
            feats[6] = float(np.clip(feats[6], 0.25, 0.75))

        return feats

    # ─────────────────────────────────────────────────────────────
    # Trading Mode Features (v4.0) - 8 dims
    # ─────────────────────────────────────────────────────────────

    def _build_trading_mode_features(self, trading_mode_state: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Build trading mode state features (v4.0).

        Features from TradingModeManager:
        [0] mode_encoded: Current mode (safe=0.25, normal=0.5, aggressive=0.75, extreme=1.0)
        [1] risk_multiplier: Mode-based risk multiplier (0.5-2.0 scaled to 0-1)
        [2] max_exposure: Maximum allowed exposure (0-1)
        [3] mode_effectiveness: How well current mode is performing (0-1)
        [4] performance_score: Decision factor score (0-1)
        [5] consensus_score: Decision factor consensus (0-1)
        [6] stability_score: System stability (0-1)
        [7] mode_confidence: Confidence in mode selection (0-1)
        """
        feats = np.zeros(8, dtype=np.float32)

        # Default to "normal" mode with standard values
        feats[0] = 0.5   # normal mode
        feats[1] = 0.5   # risk_multiplier 1.0 scaled
        feats[2] = 0.6   # max_exposure 60%
        feats[3] = 0.5   # mode_effectiveness neutral
        feats[4] = 0.5   # performance_score neutral
        feats[5] = 0.5   # consensus_score neutral
        feats[6] = 0.5   # stability_score neutral
        feats[7] = 0.5   # mode_confidence neutral

        if not trading_mode_state or not isinstance(trading_mode_state, dict):
            return feats

        # [0] mode_encoded
        mode = trading_mode_state.get("trading_mode", trading_mode_state.get("current_mode", "normal"))
        mode_map = {"safe": 0.25, "normal": 0.5, "aggressive": 0.75, "extreme": 1.0}
        if isinstance(mode, str):
            feats[0] = mode_map.get(mode.lower(), 0.5)

        # [1] risk_multiplier (scale 0.5-2.0 to 0-1)
        mode_config = trading_mode_state.get("mode_config", {})
        if isinstance(mode_config, dict):
            risk_mult = mode_config.get("risk_multiplier", 1.0)
            try:
                # Scale: 0.5 -> 0.0, 1.0 -> ~0.33, 2.0 -> 1.0
                feats[1] = float(np.clip((float(risk_mult) - 0.5) / 1.5, 0.0, 1.0))
            except (TypeError, ValueError):
                feats[1] = 0.5

            # [2] max_exposure
            max_exp = mode_config.get("max_exposure", 0.6)
            try:
                feats[2] = float(np.clip(float(max_exp), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[2] = 0.6

        # [3] mode_effectiveness
        effectiveness = trading_mode_state.get("mode_effectiveness", 0.5)
        try:
            feats[3] = float(np.clip(float(effectiveness), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[3] = 0.5

        # Decision factors
        decision_factors = trading_mode_state.get("decision_factors", {})
        if isinstance(decision_factors, dict):
            # [4] performance_score
            perf_score = decision_factors.get("performance_score", 0.5)
            try:
                feats[4] = float(np.clip(float(perf_score), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[4] = 0.5

            # [5] consensus_score
            cons_score = decision_factors.get("consensus_score", 0.5)
            try:
                feats[5] = float(np.clip(float(cons_score), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[5] = 0.5

            # [6] stability_score
            stab_score = decision_factors.get("stability_score", 0.5)
            try:
                feats[6] = float(np.clip(float(stab_score), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[6] = 0.5

        # [7] mode_confidence (from mode stats)
        mode_stats = trading_mode_state.get("mode_stats", {})
        if isinstance(mode_stats, dict):
            mode_eff = mode_stats.get("mode_effectiveness", 0.5)
            try:
                feats[7] = float(np.clip(float(mode_eff), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[7] = 0.5

        return feats

    # ─────────────────────────────────────────────────────────────
    # SmartInfoBus Data Fetchers
    # ─────────────────────────────────────────────────────────────

    def _fetch_market_data(self, bus: Any, module: str) -> Dict[str, Any]:
        """Fetch market data from SmartInfoBus."""
        try:
            mtf = bus.get("multi_timeframe_data", module)
            if isinstance(mtf, dict):
                return mtf

            md = bus.get("market_data", module)
            if isinstance(md, dict):
                return md

            hp = bus.get("historical_prices", module)
            if isinstance(hp, dict):
                return hp
        except Exception:
            pass
        return {}

    def _fetch_expert_signals(self, bus: Any, module: str) -> Dict[str, Any]:
        """Fetch expert voting signals from SmartInfoBus."""
        try:
            def _expert_block(vote_key: str, conf_key: str) -> Dict[str, Any]:
                raw = bus.get(vote_key, module) or "flat"
                conf_raw = bus.get(conf_key, module)
                try:
                    conf_val = float(conf_raw) if conf_raw is not None else 0.0
                except (TypeError, ValueError):
                    conf_val = 0.0
                return {"proposal": raw, "confidence": conf_val}

            experts = {
                "trend": _expert_block("TrendExpert_voting_proposal", "TrendExpert_confidence"),
                "momentum": _expert_block("MomentumExpert_voting_proposal", "MomentumExpert_confidence"),
                "theme": _expert_block("ThemeExpert_voting_proposal", "ThemeExpert_confidence"),
                "seasonality": _expert_block("SeasonalityRiskExpert_voting_proposal", "SeasonalityRiskExpert_confidence"),
            }

            market = {
                "regime": bus.get("market_regime", module) or "unknown",
                "regime_strength": float(bus.get("regime_strength", module) or 0.5),
            }

            return {"experts": experts, "market": market}
        except Exception:
            return {}

    def _fetch_committee_state(self, bus: Any, module: str) -> Dict[str, Any]:
        """Fetch committee state from SmartInfoBus."""
        try:
            decision = bus.get("committee_decision", module) or {}
            if isinstance(decision, str):
                decision = {"action": decision}

            return {
                "action": decision.get("action", "hold"),
                "confidence": float(bus.get("committee_confidence", module) or 0.5),
                "consensus_score": float(bus.get("consensus_score", module) or 0.5),
                "fragility": float(bus.get("fragility", module) or 0.5),
            }
        except Exception:
            return {}

    def _fetch_risk_state(self, bus: Any, module: str) -> Dict[str, Any]:
        """Fetch risk state from SmartInfoBus."""
        try:
            risk_data = bus.get("risk_data", module) or {}
            portfolio_risk = bus.get("portfolio_risk", module) or {}

            # Extract risk_budget from risk_data (PortfolioRiskSystem provides risk_budget_available)
            risk_budget = 1.0
            if isinstance(risk_data, dict):
                if "risk_budget_available" in risk_data:
                    risk_budget = float(risk_data.get("risk_budget_available", 1.0))
                elif "risk_budget_used" in risk_data:
                    risk_budget = max(0.0, 1.0 - float(risk_data.get("risk_budget_used", 0.0)))

            return {
                "risk_data": risk_data,
                "portfolio_risk": portfolio_risk,
                "risk_budget": risk_budget,
            }
        except Exception:
            return {}

    def _fetch_memory_state(self, bus: Any, module: str) -> Dict[str, Any]:
        """Fetch memory state from SmartInfoBus."""
        try:
            return {
                "memory_gate": bus.get("memory_gate", module) or 1.0,
                "danger_zones": bus.get("danger_zones", module) or {},
            }
        except Exception:
            return {}

    def _fetch_account_state(self, bus: Any, module: str) -> Dict[str, Any]:
        """Fetch account state from SmartInfoBus."""
        try:
            market_state = bus.get("market_state", module) or {}
            positions = bus.get("positions", module) or {}

            state: Dict[str, Any] = {
                "balance": float(market_state.get("balance", 100000.0)),
                "initial_balance": float(market_state.get("initial_balance", 100000.0))
                if "initial_balance" in market_state
                else 100000.0,
                "current_drawdown": float(market_state.get("drawdown", 0.0)),
                "current_step": int(market_state.get("step", 0)),
                "max_steps": int(market_state.get("max_steps", 10000)),
                # Optional performance stats (if available)
                "win_rate": float(market_state.get("win_rate", 0.5)),
                "pnl_trend": float(market_state.get("pnl_trend", 0.0)),
                "trades_today": int(market_state.get("trades_today", 0)),
            }

            # Optionally inject global position summary if present
            if isinstance(positions, dict) and "net" in positions and isinstance(positions["net"], dict):
                net = positions["net"]
                state.setdefault("position_direction", net.get("direction", 0))
                state.setdefault("position_size", net.get("size", 0.0))
                state.setdefault("unrealized_pnl", net.get("unrealized_pnl", 0.0))

            return state
        except Exception:
            return {}

    def _fetch_world_model_state(self, bus: Any, module: str) -> Dict[str, Any]:
        """Fetch world model state from SmartInfoBus (v4.0)."""
        try:
            market_predictions = bus.get("market_predictions", module) or {}
            prediction_confidence = bus.get("prediction_confidence", module) or {}
            scenario_generation = bus.get("scenario_generation", module) or {}
            world_model_analytics = bus.get("world_model_analytics", module) or {}

            # Merge all world model outputs
            state: Dict[str, Any] = {
                "market_predictions": market_predictions if isinstance(market_predictions, dict) else {},
                "prediction_confidence": prediction_confidence if isinstance(prediction_confidence, dict) else {},
                "scenario_generation": scenario_generation if isinstance(scenario_generation, dict) else {},
                "world_model_analytics": world_model_analytics if isinstance(world_model_analytics, dict) else {},
            }

            # Extract key values for convenience
            if isinstance(market_predictions, dict):
                state["is_trained"] = market_predictions.get("is_trained", False)
                state["model_confidence"] = market_predictions.get("model_confidence", 0.0)
                state["prediction_quality"] = market_predictions.get("prediction_quality", 0.0)
                state["stability_score"] = market_predictions.get("stability_score", 0.5)

                # Latest predictions if available
                latest = market_predictions.get("latest_predictions", {})
                if isinstance(latest, dict):
                    state["price_changes"] = latest.get("price_changes", [])
                    state["volatility_predictions"] = latest.get("volatility_predictions", [])
                    state["regime_probabilities"] = latest.get("regime_probabilities", [])
                    state["predicted_regime"] = latest.get("predicted_regime", -1)
                    state["confidence"] = latest.get("confidence", 0.0)

            return state
        except Exception:
            return {}

    def _fetch_trading_mode_state(self, bus: Any, module: str) -> Dict[str, Any]:
        """Fetch trading mode state from SmartInfoBus (v4.0)."""
        try:
            trading_mode = bus.get("trading_mode", module)
            mode_config = bus.get("mode_config", module) or {}
            mode_effectiveness = bus.get("mode_effectiveness", module)
            mode_stats = bus.get("mode_stats", module) or {}
            mode_thresholds = bus.get("mode_thresholds", module) or {}
            decision_factors = bus.get("decision_factors", module) or {}

            state: Dict[str, Any] = {
                "trading_mode": trading_mode if isinstance(trading_mode, str) else "normal",
                "mode_config": mode_config if isinstance(mode_config, dict) else {},
                "mode_effectiveness": float(mode_effectiveness) if mode_effectiveness is not None else 0.5,
                "mode_stats": mode_stats if isinstance(mode_stats, dict) else {},
                "mode_thresholds": mode_thresholds if isinstance(mode_thresholds, dict) else {},
                "decision_factors": decision_factors if isinstance(decision_factors, dict) else {},
            }

            return state
        except Exception:
            return {}

    # ─────────────────────────────────────────────────────────────
    # Helper Functions
    # ─────────────────────────────────────────────────────────────

    def _extract_timeframe_data(
        self, market_data: Dict[str, Any], timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Extract OHLCV data for a specific timeframe with forming bar integration."""
        if not market_data:
            return None

        # Try direct timeframe key
        if timeframe in market_data and isinstance(market_data[timeframe], dict):
            tf_data = market_data[timeframe]
            # REAL-TIME RESPONSIVENESS: Update last bar with forming bar data
            return self._apply_forming_bar(tf_data)

        # Try nested structure: market_data[symbol][timeframe]
        for sym_data in market_data.values():
            if isinstance(sym_data, dict) and timeframe in sym_data and isinstance(sym_data[timeframe], dict):
                tf_data = sym_data[timeframe]
                return self._apply_forming_bar(tf_data)

        return None

    def _apply_forming_bar(self, tf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply forming bar's current values to the last element of OHLCV arrays."""
        if not isinstance(tf_data, dict):
            return tf_data

        cur_bar = tf_data.get("current_bar")
        if not isinstance(cur_bar, dict):
            return tf_data

        forming_close = cur_bar.get("close")
        forming_high = cur_bar.get("high")
        forming_low = cur_bar.get("low")
        forming_volume = cur_bar.get("volume")

        # Only modify if we have forming bar data
        if forming_close is None:
            return tf_data

        # Create a copy to avoid mutating original
        result = dict(tf_data)

        try:
            close = result.get("close")
            if close is not None and len(close) > 0:
                # Replace last closed bar with forming bar's current value
                close_list = list(close)
                if abs(float(forming_close) - float(close_list[-1])) > 0.0001:
                    close_list[-1] = float(forming_close)
                    result["close"] = close_list

                    # Also update high/low/volume
                    if forming_high is not None:
                        high = result.get("high")
                        if high is not None and len(high) > 0:
                            high_list = list(high)
                            high_list[-1] = float(forming_high)
                            result["high"] = high_list

                    if forming_low is not None:
                        low = result.get("low")
                        if low is not None and len(low) > 0:
                            low_list = list(low)
                            low_list[-1] = float(forming_low)
                            result["low"] = low_list

                    if forming_volume is not None:
                        vol = result.get("volume")
                        if vol is not None and len(vol) > 0:
                            vol_list = list(vol)
                            vol_list[-1] = float(forming_volume)
                            result["volume"] = vol_list
        except Exception:
            # On any error, return original data
            return tf_data

        return result

    def _extract_direction(self, proposal: Any) -> float:
        """Extract direction from a proposal (string, dict, numeric, etc.)."""
        if isinstance(proposal, dict):
            direction = (
                proposal.get("direction")
                or proposal.get("action")
                or proposal.get("global_direction")
            )
        elif isinstance(proposal, (int, float)):
            return float(np.clip(proposal, -1.0, 1.0))
        else:
            direction = proposal

        direction_str = str(direction or "flat").lower()

        if direction_str in ("long", "buy", "bullish", "up"):
            return 1.0
        if direction_str in ("short", "sell", "bearish", "down"):
            return -1.0
        return 0.0

    def _compute_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Compute RSI."""
        if close.size < period + 1:
            return 50.0

        window = close[-(period + 1):]
        deltas = np.diff(window)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))

        if avg_loss < self._eps:
            if avg_gain > 0:
                return 100.0
            return 50.0

        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _compute_macd_histogram(self, close: np.ndarray) -> float:
        """Compute MACD histogram (fast EMA - slow EMA)."""
        if close.size < 26:
            return 0.0

        def ema(data: np.ndarray, period: int) -> float:
            if data.size < period:
                return float(data[-1]) if data.size > 0 else 0.0
            alpha = 2.0 / (period + 1)
            result = float(data[-period])
            for i in range(-period + 1, 0):
                result = alpha * float(data[i]) + (1.0 - alpha) * result
            return result

        fast_ema = ema(close, 12)
        slow_ema = ema(close, 26)

        return float(fast_ema - slow_ema)

    def _compute_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        """Compute Average True Range."""
        if close.size < 2:
            return 0.0

        period = min(period, close.size - 1)
        if period < 1:
            return float(high[-1] - low[-1]) if high.size > 0 else 0.0

        tr_list: list[float] = []
        for i in range(-period, 0):
            h = float(high[i])
            l = float(low[i])
            pc = float(close[i - 1])
            tr = max(h - l, abs(h - pc), abs(l - pc))
            tr_list.append(tr)

        return float(np.mean(tr_list)) if tr_list else 0.0

    def _compute_slope(self, data: np.ndarray) -> float:
        """Compute linear regression slope."""
        if data.size < 2:
            return 0.0

        x = np.arange(data.size, dtype=np.float64)
        coeffs = np.polyfit(x, data.astype(np.float64), 1)
        return float(coeffs[0])

    def _compute_trend_direction(self, close: np.ndarray) -> float:
        """Compute trend direction as a value in [-1, 1]."""
        if close.size < 5:
            return 0.0

        if close.size >= 20:
            sma = float(np.mean(close[-20:]))
        else:
            sma = float(np.mean(close))

        current = float(close[-1])
        diff = (current - sma) / max(abs(sma), self._eps)
        return float(np.clip(diff * 10.0, -1.0, 1.0))


# ═══════════════════════════════════════════════════════════════════
# Module-level convenience functions
# ═══════════════════════════════════════════════════════════════════

_default_builder: Optional[PPOObservationBuilder] = None


def get_ppo_observation_builder() -> PPOObservationBuilder:
    """Get the singleton PPO observation builder."""
    global _default_builder
    if _default_builder is None:
        _default_builder = PPOObservationBuilder()
    return _default_builder


def build_ppo_observation(
    market_data: Optional[Dict[str, Any]] = None,
    expert_signals: Optional[Dict[str, Any]] = None,
    committee_state: Optional[Dict[str, Any]] = None,
    risk_state: Optional[Dict[str, Any]] = None,
    memory_state: Optional[Dict[str, Any]] = None,
    account_state: Optional[Dict[str, Any]] = None,
    world_model_state: Optional[Dict[str, Any]] = None,
    trading_mode_state: Optional[Dict[str, Any]] = None,
    smart_bus: Optional[Any] = None,
    module_name: str = "PPOObservationBuilder",
) -> np.ndarray:
    """
    Convenience function to build PPO observation.

    Can be called from both training (ModernTradingEnv) and live (PPOAgent).
    """
    builder = get_ppo_observation_builder()
    return builder.build(
        market_data=market_data,
        expert_signals=expert_signals,
        committee_state=committee_state,
        risk_state=risk_state,
        memory_state=memory_state,
        account_state=account_state,
        world_model_state=world_model_state,
        trading_mode_state=trading_mode_state,
        smart_bus=smart_bus,
        module_name=module_name,
    )


def build_ppo_observation_for_instrument(
    instrument: str,
    market_data: Optional[Dict[str, Any]] = None,
    expert_signals: Optional[Dict[str, Any]] = None,
    committee_state: Optional[Dict[str, Any]] = None,
    risk_state: Optional[Dict[str, Any]] = None,
    memory_state: Optional[Dict[str, Any]] = None,
    account_state: Optional[Dict[str, Any]] = None,
    world_model_state: Optional[Dict[str, Any]] = None,
    trading_mode_state: Optional[Dict[str, Any]] = None,
    smart_bus: Optional[Any] = None,
    module_name: str = "PPOObservationBuilder",
) -> np.ndarray:
    """
    Convenience function to build PPO observation for a specific instrument (v4.0).

    Args:
        instrument: The instrument symbol (e.g., "XAUUSD", "EURUSD")

    Returns:
        np.ndarray of shape (obs_size,) with dtype float32
    """
    builder = get_ppo_observation_builder()
    return builder.build_for_instrument(
        instrument=instrument,
        market_data=market_data,
        expert_signals=expert_signals,
        committee_state=committee_state,
        risk_state=risk_state,
        memory_state=memory_state,
        account_state=account_state,
        world_model_state=world_model_state,
        trading_mode_state=trading_mode_state,
        smart_bus=smart_bus,
        module_name=module_name,
    )
