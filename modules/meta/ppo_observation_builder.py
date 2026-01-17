#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
# File: modules/meta/ppo_observation_builder.py
# Unified PPO Observation Builder (v5.3 - Master/Advisor + Advanced Signals)
#
# Single source of truth for PPO observation construction.
# Used identically in TRAINING (ModernTradingEnv) and LIVE (PPOAgent).
#
# SCHEMA V5.3 CHANGES:
# - Added advanced market structure signals (S/R, BOS, order blocks, liquidity)
# - Added momentum divergence signals (bullish/bearish RSI divergence)
# - Added overbought/oversold signals (for reversal detection)
# - Added theme regime signals (risk_on/off, volatility regime)
# - Observation builder now fetches per-instrument rich analysis from experts
# - Train/live parity ensured: training env and live experts compute same signals
#
# FIXES APPLIED (Dec 2025):
# - Prevent cross-symbol leakage when instrument key mismatches (XAUUSD vs XAU/USD, etc.)
# - Make expert signals schema match _build_voting_features() (direction + score/strength)
# - Normalize position direction parsing (string -> signed float)
# - FIX: forming-bar update tolerance uses relative/absolute epsilon (no XAUUSD 0.0001 mismatch)
# - FIX: MACD histogram is real histogram (MACD line - signal EMA9)
# - FIX: regime_accuracy fetch preserves float values (no accidental dict coercion)
# - FIX: Added use_forming_bar flag (default False) to ensure train/live parity
#        Training uses closed bars; live should too unless explicitly configured otherwise.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

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

# Import centralized trade limits
try:
    from config import get_trade_limits
    _TRADE_LIMITS = get_trade_limits()
except ImportError:
    _TRADE_LIMITS = {"max_trades_per_day": 20}


# ═══════════════════════════════════════════════════════════════════
# OBSERVATION SCHEMA (v5.4) - Expanded HTF Features
# ═══════════════════════════════════════════════════════════════════
#
# Total: 76 dimensions (+12 from v5.3)
#
# [0-9]   M15 Price Features (PRIMARY) - 10 dims
#         [3] = S/R Proximity Signal (SIGNED: +support/-resistance)
# [10-27] Higher TF Context (H1/H4/D1 EXPANDED) - 18 dims (6 per TF)
#         Per TF: [trend, momentum, RSI, ATR_norm, S/R_proximity, HH/HL_bias]
#         H1:  [10-15]
#         H4:  [16-21]
#         D1:  [22-27]
# [28-35] Expert ADVISOR Signals (SIGNED: +bull/-bear) - 8 dims
#         [2,3] = Momentum (direction boosted by divergence, conf + OB/OS)
#         [4,5] = Theme (direction, conf + regime risk)
# [36-43] Committee Consensus (SIGNED + structure) - 8 dims
#         [7] = Composite Market Structure (40% S/R + 30% HH/HL + 20% BOS + 10% OB)
# [44-51] Risk/Memory Signals - 8 dims
# [52-59] Account/Position State - 8 dims
# [60-67] World Model Predictions - 8 dims
# [68-75] Trading Mode State (incl. timing features) - 8 dims
# [76-83] Governor/Budget State (v5.5) - 8 dims
#         [0] loss_layer_ratio: consecutive_losses / loss_layer_stop [0,1]
#         [1] loss_layer_level: clamped loss layer / 5.0 [0,1]
#         [2] win_streak_ratio: consecutive_wins / 5.0 [0,1]
#         [3] session_pnl_headroom: remaining session loss headroom [0,2] -> normalized to [0,1]
#         [4] session_trade_budget: 1 - session_trades/max [0,1]
#         [5] session_consec_loss_ratio: session consec losses / limit [0,1]
#         [6] session_progress: bars into session / duration [0,1]
#         [7] pending_order_progress: bars until fill / max_latency [0,1]
#
# HTF FEATURE EXPANSION (v5.4):
# Each higher timeframe (H1, H4, D1) now gets 6 dedicated features:
# - [0] Trend direction: signed EMA slope (-1 to +1)
# - [1] Momentum: 5-bar momentum normalized
# - [2] RSI: normalized to -1 (oversold) to +1 (overbought)
# - [3] ATR_norm: volatility as % of price (0 to 1)
# - [4] S/R_proximity: signed distance to nearest S/R (-1=resistance, +1=support)
# - [5] HH/HL_bias: recent price structure bias (-1 lower lows, +1 higher highs)
#
# This gives the agent REAL higher timeframe signals instead of crushed averages!
# ═══════════════════════════════════════════════════════════════════

PPO_OBS_VERSION = "5.5"
PPO_OBS_SIZE = 84

FEATURE_GROUPS: Dict[str, tuple[int, int]] = {
    "m15_price": (0, 10),
    "htf_context": (10, 28),
    "voting": (28, 36),
    "committee": (36, 44),
    "risk": (44, 52),
    "account": (52, 60),
    "world_model": (60, 68),
    "trading_mode": (68, 76),
    "governor": (76, 84),
}


def _build_feature_names() -> List[str]:
    """
    Return the canonical feature name list for the 84-dim observation.

    This is primarily used for explain/audit tooling so you can map raw
    `observation[i]` values back to their semantic meaning.
    """
    names: List[str] = []

    # 0..9: M15 primary price features (10)
    names += [
        "m15_price_vs_mean",
        "m15_range_pct",
        "m15_close_open_pct",
        "m15_sr_proximity",
        "m15_rsi_norm",
        "m15_macd_hist_norm",
        "m15_atr_norm_x10",
        "m15_trend_slope_norm",
        "m15_roc_x10",
        "m15_volatility_std_x100",
    ]

    # 10..27: HTF context (H1/H4/D1, 6 each = 18)
    for tf in ("H1", "H4", "D1"):
        names += [
            f"htf_{tf}_trend",
            f"htf_{tf}_momentum",
            f"htf_{tf}_rsi_norm",
            f"htf_{tf}_atr_norm",
            f"htf_{tf}_sr_proximity",
            f"htf_{tf}_structure_bias",
        ]

    # 28..35: expert voting features (8)
    names += [
        "expert_trend_signed_strength",
        "expert_trend_confidence",
        "expert_momentum_signed_strength",
        "expert_momentum_confidence_adj",
        "expert_theme_signed_strength",
        "expert_theme_confidence_adj",
        "expert_seasonality_signed_strength",
        "expert_seasonality_confidence",
    ]

    # 36..43: committee/consensus features (8)
    names += [
        "committee_signed_consensus",
        "committee_confidence",
        "committee_expert_agreement",
        "committee_expert_conf_mean",
        "committee_fragility",
        "committee_market_regime",
        "committee_market_regime_strength",
        "committee_structure_composite",
    ]

    # 44..51: risk/memory features (8)
    names += [
        "risk_memory_gate",
        "risk_danger_zone_count",
        "risk_drawdown_norm",
        "risk_balance_ratio",
        "risk_portfolio_exposure",
        "risk_budget",
        "risk_win_rate",
        "risk_pnl_trend",
    ]

    # 52..59: account/position features (8)
    names += [
        "account_step_ratio",
        "account_episode_return_norm",
        "account_position_direction",
        "account_position_size_norm",
        "account_unrealized_pnl_norm",
        "account_time_in_position_norm",
        "account_trades_today_norm",
        "account_on_cooldown",
    ]

    # 60..67: world model features (8)
    names += [
        "wm_model_confidence",
        "wm_m15_price_change",
        "wm_weighted_price_change",
        "wm_volatility_pred",
        "wm_regime_signal",
        "wm_is_trained",
        "wm_bullish_probability",
        "wm_stability_score",
    ]

    # 68..75: trading mode features (8)
    names += [
        "mode_trading_mode",
        "mode_entry_allowed",
        "mode_entry_quality",
        "mode_theme_stability",
        "mode_zone_quality",
        "mode_vol_state",
        "mode_liquidity_score",
        "mode_effectiveness",
    ]

    # 76..83: governor/budget features (8)
    names += [
        "gov_loss_layer_ratio",
        "gov_loss_layer_level",
        "gov_win_streak_ratio",
        "gov_session_pnl_headroom",
        "gov_session_trade_budget",
        "gov_session_consec_loss_ratio",
        "gov_session_progress",
        "gov_pending_order_progress",
    ]

    if len(names) != PPO_OBS_SIZE:
        raise ValueError(f"Feature name list mismatch: {len(names)} != {PPO_OBS_SIZE}")

    return names


PPO_OBS_FEATURE_NAMES: List[str] = _build_feature_names()


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
    # Loaded from config/risk_policy.yaml -> trade_limits.max_trades_per_day
    max_trades_per_day: int = field(default_factory=lambda: _TRADE_LIMITS.get("max_trades_per_day", 20))

    # World model parameters (v4.0)
    prediction_confidence_threshold: float = 0.5
    scenario_confidence_threshold: float = 0.5  # reserved for future use

    # Forming bar tolerance (relative/absolute)
    # - relative: scaled by current price (works across instruments)
    # - absolute: protects near-zero / tiny prices
    forming_bar_rtol: float = 1e-7
    forming_bar_atol: float = 1e-6
    
    # CRITICAL: Train/Live parity flag
    # - False (default): Ignore forming bars, use only closed candles (RECOMMENDED)
    #   This ensures training and live see identical observation distributions.
    # - True: Merge forming bar into last candle (live-only use case)
    #   Only set True if you explicitly want live to see incomplete candles.
    use_forming_bar: bool = False


class PPOObservationBuilder:
    """
    Unified PPO Observation Builder.

    Constructs identical observation vectors for both TRAINING and LIVE.
    M15 is the PRIMARY timeframe; H1/H4/D1 are context only.
    """

    def __init__(self, config: Optional[PPOObservationConfig] = None) -> None:
        self.config = config or PPOObservationConfig()
        if self.config.obs_size != PPO_OBS_SIZE:
            self.config.obs_size = PPO_OBS_SIZE
        self._eps: float = 1e-8
        self._warned_missing_governor_state: bool = False

    @property
    def obs_size(self) -> int:
        return self.config.obs_size

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def feature_groups(self) -> Dict[str, tuple[int, int]]:
        """Named slices of the observation vector (start,end)."""
        return dict(FEATURE_GROUPS)

    @property
    def feature_names(self) -> List[str]:
        """Canonical feature names aligned to `build()` output indices."""
        return list(PPO_OBS_FEATURE_NAMES)

    def get_schema(self) -> Dict[str, Any]:
        """
        Return a JSON-friendly schema describing the observation layout.

        Used by explain/audit tooling to map raw observation vectors to named
        features without re-deriving the builder internals.
        """
        return {
            "version": self.version,
            "obs_size": int(self.obs_size),
            "feature_groups": {k: {"start": int(v[0]), "end": int(v[1])} for k, v in FEATURE_GROUPS.items()},
            "feature_names": self.feature_names,
        }

    # ======================================================================
    # Public Builders
    # ======================================================================

    def build(
        self,
        market_data: Optional[Dict[str, Any]] = None,
        expert_signals: Optional[Dict[str, Any]] = None,
        committee_state: Optional[Dict[str, Any]] = None,
        risk_state: Optional[Dict[str, Any]] = None,
        memory_state: Optional[Dict[str, Any]] = None,
        account_state: Optional[Dict[str, Any]] = None,
        world_model_state: Optional[Dict[str, Any]] = None,
        trading_mode_state: Optional[Dict[str, Any]] = None,
        governor_state: Optional[Dict[str, Any]] = None,  # v5.5: governor/budget observation
        smart_bus: Optional[Any] = None,
        module_name: str = "PPOObservationBuilder",
    ) -> np.ndarray:
        """Build the unified PPO observation vector."""
        obs = np.zeros(self.config.obs_size, dtype=np.float32)

        if smart_bus is not None:
            market_data = market_data or self._fetch_market_data(smart_bus, module_name)
            expert_signals = expert_signals or self._fetch_expert_signals(smart_bus, module_name)
            committee_state = committee_state or self._fetch_committee_state(smart_bus, module_name)
            risk_state = risk_state or self._fetch_risk_state(smart_bus, module_name)
            memory_state = memory_state or self._fetch_memory_state(smart_bus, module_name)
            account_state = account_state or self._fetch_account_state(smart_bus, module_name)
            world_model_state = world_model_state or self._fetch_world_model_state(smart_bus, module_name)
            trading_mode_state = trading_mode_state or self._fetch_trading_mode_state(smart_bus, module_name)
            governor_state = governor_state or self._fetch_governor_state(smart_bus, module_name)

        obs[0:10] = self._build_m15_features(market_data)
        obs[10:28] = self._build_htf_context(market_data, expert_signals)
        obs[28:36] = self._build_voting_features(expert_signals)
        obs[36:44] = self._build_committee_features(committee_state, expert_signals)
        obs[44:52] = self._build_risk_features(risk_state, memory_state, account_state)
        obs[52:60] = self._build_account_features(account_state)
        obs[60:68] = self._build_world_model_features(world_model_state)
        obs[68:76] = self._build_trading_mode_features(trading_mode_state)
        obs[76:84] = self._build_governor_features(governor_state)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def build_for_instrument(
        self,
        instrument: str,
        market_data: Optional[Dict[str, Any]] = None,
        expert_signals: Optional[Dict[str, Any]] = None,
        committee_state: Optional[Dict[str, Any]] = None,
        risk_state: Optional[Dict[str, Any]] = None,
        memory_state: Optional[Dict[str, Any]] = None,
        account_state: Optional[Dict[str, Any]] = None,
        world_model_state: Optional[Dict[str, Any]] = None,
        trading_mode_state: Optional[Dict[str, Any]] = None,
        governor_state: Optional[Dict[str, Any]] = None,  # v5.5: governor/budget observation
        smart_bus: Optional[Any] = None,
        module_name: str = "PPOObservationBuilder",
    ) -> np.ndarray:
        """Build observation vector for a SPECIFIC instrument."""
        obs = np.zeros(self.config.obs_size, dtype=np.float32)

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
            trading_mode_state = trading_mode_state or self._fetch_trading_mode_state_for_instrument(
                smart_bus, module_name, instrument
            )
            governor_state = governor_state or self._fetch_governor_state(smart_bus, module_name)

        obs[0:10] = self._build_m15_features_for_instrument(market_data, instrument)
        obs[10:28] = self._build_htf_context_for_instrument(market_data, instrument, expert_signals)
        obs[28:36] = self._build_voting_features_for_instrument(expert_signals, instrument)
        obs[36:44] = self._build_committee_features(committee_state, expert_signals)
        obs[44:52] = self._build_risk_features(risk_state, memory_state, account_state)
        obs[52:60] = self._build_account_features(account_state)
        obs[60:68] = self._build_world_model_features(world_model_state)
        obs[68:76] = self._build_trading_mode_features(trading_mode_state)
        obs[76:84] = self._build_governor_features(governor_state)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # ─────────────────────────────────────────────────────────────
    # Symbol-safe helpers (prevents cross-instrument leakage)
    # ─────────────────────────────────────────────────────────────

    def _norm_symbol(self, s: Any) -> str:
        """Normalize symbol keys (XAU/USD, XAUUSD, xau_usd, EURUSDm -> XAUUSD/EURUSD)."""
        if not isinstance(s, str):
            return ""
        return "".join(ch for ch in s.strip().upper() if ch.isalnum())

    def _lookup_symbol_block(self, mapping: Any, instrument: str) -> Optional[Dict[str, Any]]:
        """Return mapping[instrument] with normalization fallback; dict-only."""
        if not isinstance(mapping, dict):
            return None

        v = mapping.get(instrument)
        if isinstance(v, dict):
            return v

        target = self._norm_symbol(instrument)
        if not target:
            return None

        # 1) Exact normalized match
        for k, val in mapping.items():
            if not (isinstance(k, str) and isinstance(val, dict)):
                continue
            if self._norm_symbol(k) == target:
                return val

        # 2) Loose match for broker suffixes / aliasing
        candidates: list[tuple[int, Dict[str, Any]]] = []
        for k, val in mapping.items():
            if not (isinstance(k, str) and isinstance(val, dict)):
                continue
            k_norm = self._norm_symbol(k)
            if not k_norm:
                continue
            if k_norm.startswith(target) or target.startswith(k_norm):
                candidates.append((abs(len(k_norm) - len(target)), val))

        if candidates:
            candidates.sort(key=lambda t: t[0])
            return candidates[0][1]

        return None

    def _is_timeframe_dict(self, d: Any) -> bool:
        """True if dict looks like {M15:{...}, H1:{...}, ...}."""
        if not isinstance(d, dict):
            return False
        return any(tf in d and isinstance(d.get(tf), dict) for tf in SUPPORTED_TIMEFRAMES)

    # ─────────────────────────────────────────────────────────────
    # Instrument-aware fetchers / builders
    # ─────────────────────────────────────────────────────────────

    def _fetch_market_data_for_instrument(self, bus: Any, module: str, instrument: str) -> Dict[str, Any]:
        """Fetch instrument-specific market data from SmartInfoBus without cross-symbol leakage."""
        try:
            mtf = bus.get("multi_timeframe_data", module)
            block = self._lookup_symbol_block(mtf, instrument)
            if isinstance(block, dict):
                return block
            if self._is_timeframe_dict(mtf):
                return mtf

            md = bus.get("market_data", module)
            block = self._lookup_symbol_block(md, instrument)
            if isinstance(block, dict):
                return block
            if self._is_timeframe_dict(md):
                return md

            hp = bus.get("historical_prices", module)
            block = self._lookup_symbol_block(hp, instrument)
            if isinstance(block, dict):
                return block
            if self._is_timeframe_dict(hp):
                return hp
        except Exception:
            pass

        return {}

    def _fetch_expert_signals_for_instrument(self, bus: Any, module: str, instrument: str) -> Dict[str, Any]:
        """Fetch instrument-specific expert signals."""
        global_signals = self._fetch_expert_signals(bus, module)
        experts = global_signals.get("experts", {})

        if not isinstance(experts, dict):
            return global_signals

        target = self._norm_symbol(instrument)

        def _dir_label(raw: Any) -> str:
            if isinstance(raw, (int, float)):
                if float(raw) > 0:
                    return "bullish"
                if float(raw) < 0:
                    return "bearish"
                return "neutral"
            s = str(raw).lower().strip()
            if s in ("bullish", "long", "buy", "up", "uptrend"):
                return "bullish"
            if s in ("bearish", "short", "sell", "down", "downtrend"):
                return "bearish"
            # Non-directional / management actions
            if s in ("exit", "close", "tighten", "hold"):
                return "neutral"
            return "neutral"

        def _clip01(x: Any, default: float = 0.0) -> float:
            try:
                return float(np.clip(float(x), 0.0, 1.0))
            except Exception:
                return default

        # 1) If expert signals already include per-instrument maps, select those.
        for expert_name, sig in list(experts.items()):
            if isinstance(sig, dict) and "instruments" in sig:
                instruments_map = sig.get("instruments", {})
                if isinstance(instruments_map, dict):
                    inst_sig = instruments_map.get(instrument)
                    if not isinstance(inst_sig, dict) and target:
                        for k, v in instruments_map.items():
                            if (
                                isinstance(k, str)
                                and isinstance(v, dict)
                                and self._norm_symbol(k) == target
                            ):
                                inst_sig = v
                                break
                    if isinstance(inst_sig, dict):
                        experts[expert_name] = inst_sig

        # 2) Enrich from live expert analysis blobs (per-instrument) for parity
        # with training expert_signals schema.
        try:
            trend_analysis = bus.get("trend_analysis", module)
        except Exception:
            trend_analysis = None
        if isinstance(trend_analysis, dict):
            per_inst = trend_analysis.get("per_instrument", {})
            inst_trend = self._lookup_symbol_block(per_inst, instrument)
            if isinstance(inst_trend, dict):
                prev = experts.get("trend", {})
                sig = dict(prev) if isinstance(prev, dict) else {}
                sig["direction"] = _dir_label(
                    inst_trend.get("current_trend", inst_trend.get("action", "neutral"))
                )
                try:
                    ts = float(inst_trend.get("trend_strength", 0.0) or 0.0)
                except Exception:
                    ts = 0.0
                sig["score"] = float(np.clip(abs(ts), 0.0, 1.0))
                sig["confidence"] = _clip01(inst_trend.get("confidence", sig.get("confidence", 0.0)))

                proposal = sig.get("proposal", {})
                if not isinstance(proposal, dict):
                    proposal = {}

                # Map keys to training proposal schema (best-effort).
                key_map = {
                    "near_support": "near_support",
                    "near_resistance": "near_resistance",
                    "structure_trend": "structure_trend",
                    "structure_strength": "structure_strength",
                    "bos_signal": "bos_signal",
                    "liquidity_above": "liquidity_above",
                    "liquidity_below": "liquidity_below",
                    "order_block_bull": "order_block_bull",
                    "order_block_bear": "order_block_bear",
                    "plus_di": "plus_di",
                    "minus_di": "minus_di",
                    "sar_direction": "sar_direction",
                    "ma_alignment": "ma_alignment",
                    "trend_slope": "trend_slope",
                }
                for src, dst in key_map.items():
                    if src in inst_trend:
                        proposal[dst] = inst_trend[src]

                # Rename/alias common fields
                if "adx" in inst_trend and "adx_value" not in proposal:
                    proposal["adx_value"] = inst_trend.get("adx")

                if "bullish_confluence" in inst_trend or "bearish_confluence" in inst_trend:
                    try:
                        bc = float(inst_trend.get("bullish_confluence", 0.0) or 0.0)
                        bc2 = float(inst_trend.get("bearish_confluence", 0.0) or 0.0)
                        proposal.setdefault("confluence_score", float(max(abs(bc), abs(bc2))))
                    except Exception:
                        pass

                sig["proposal"] = proposal
                experts["trend"] = sig

        try:
            mom_analysis = bus.get("momentum_analysis", module)
        except Exception:
            mom_analysis = None
        if isinstance(mom_analysis, dict):
            per_inst = mom_analysis.get("per_instrument", {})
            inst_mom = self._lookup_symbol_block(per_inst, instrument)
            if isinstance(inst_mom, dict):
                prev = experts.get("momentum", {})
                sig = dict(prev) if isinstance(prev, dict) else {}
                sig["direction"] = _dir_label(inst_mom.get("direction", inst_mom.get("action", "neutral")))
                # Prefer confluence-based strength, fall back to composite momentum magnitude.
                try:
                    bc = float(inst_mom.get("bullish_confluence", 0.0) or 0.0)
                    sc = float(inst_mom.get("bearish_confluence", 0.0) or 0.0)
                    cm = float(inst_mom.get("composite_momentum", 0.0) or 0.0)
                    sig["score"] = float(np.clip(max(abs(bc), abs(sc), abs(cm)), 0.0, 1.0))
                except Exception:
                    sig["score"] = 0.0
                sig["confidence"] = _clip01(inst_mom.get("confidence", sig.get("confidence", 0.0)))

                proposal = sig.get("proposal", {})
                if not isinstance(proposal, dict):
                    proposal = {}
                # Merge full analysis for richer signals (safe dict-only).
                proposal.update(inst_mom)

                proposal.setdefault("divergence_signal", proposal.get("divergence"))
                rsi_val = proposal.get("rsi", 50.0)
                try:
                    rsi_val_f = float(rsi_val)
                except Exception:
                    rsi_val_f = 50.0
                proposal.setdefault(
                    "overbought",
                    max(0.0, (rsi_val_f - 70.0) / 30.0) if rsi_val_f > 70.0 else 0.0,
                )
                proposal.setdefault(
                    "oversold",
                    max(0.0, (30.0 - rsi_val_f) / 30.0) if rsi_val_f < 30.0 else 0.0,
                )
                proposal.setdefault("rsi_value", rsi_val_f)

                sig["proposal"] = proposal
                experts["momentum"] = sig

        try:
            theme_analysis = bus.get("theme_analysis", module)
        except Exception:
            theme_analysis = None
        if isinstance(theme_analysis, dict):
            per_inst = theme_analysis.get("per_instrument", {})
            inst_theme = self._lookup_symbol_block(per_inst, instrument)
            if isinstance(inst_theme, dict):
                prev = experts.get("theme", {})
                sig = dict(prev) if isinstance(prev, dict) else {}
                sig["direction"] = _dir_label(inst_theme.get("action", "neutral"))
                sig["score"] = _clip01(inst_theme.get("composite_score", inst_theme.get("vol_score", 0.0)))
                sig["confidence"] = _clip01(inst_theme.get("confidence", sig.get("confidence", 0.0)))

                proposal = sig.get("proposal", {})
                if not isinstance(proposal, dict):
                    proposal = {}
                proposal.update(inst_theme)
                sig["proposal"] = proposal
                experts["theme"] = sig

        try:
            seasonality_analysis = bus.get("seasonality_analysis", module)
        except Exception:
            seasonality_analysis = None
        if isinstance(seasonality_analysis, dict):
            per_inst = seasonality_analysis.get("per_instrument", {})
            inst_seas = self._lookup_symbol_block(per_inst, instrument)
            if isinstance(inst_seas, dict):
                prev = experts.get("seasonality", {})
                sig = dict(prev) if isinstance(prev, dict) else {}
                sig["direction"] = _dir_label(inst_seas.get("action", "neutral"))
                sig["score"] = _clip01(inst_seas.get("composite_score", 0.0))
                sig["confidence"] = _clip01(inst_seas.get("confidence", sig.get("confidence", 0.0)))

                proposal = sig.get("proposal", {})
                if not isinstance(proposal, dict):
                    proposal = {}
                proposal.update(inst_seas)
                sig["proposal"] = proposal
                experts["seasonality"] = sig

        # Ensure HTF expert signals map to the current instrument.
        global_signals["htf_experts"] = self._extract_htf_experts_from_bus(bus, module, instrument=instrument)

        global_signals["experts"] = experts
        return global_signals

    def _fetch_risk_state_for_instrument(self, bus: Any, module: str, instrument: str) -> Dict[str, Any]:
        """Fetch instrument-specific risk state."""
        risk = self._fetch_risk_state(bus, module)

        portfolio_risk = risk.get("portfolio_risk", {})
        if isinstance(portfolio_risk, dict) and "instruments" in portfolio_risk:
            inst_risk = portfolio_risk["instruments"].get(instrument, {})
            if not isinstance(inst_risk, dict):
                target = self._norm_symbol(instrument)
                for k, v in portfolio_risk["instruments"].items():
                    if isinstance(k, str) and isinstance(v, dict) and self._norm_symbol(k) == target:
                        inst_risk = v
                        break
            risk["instrument_risk"] = inst_risk if isinstance(inst_risk, dict) else {}

        return risk

    def _fetch_account_state_for_instrument(self, bus: Any, module: str, instrument: str) -> Dict[str, Any]:
        """Fetch instrument-specific account/position state."""
        account = self._fetch_account_state(bus, module)

        positions = bus.get("positions", module) or {}
        if isinstance(positions, dict):
            inst_pos = positions.get(instrument)
            if not isinstance(inst_pos, dict):
                target = self._norm_symbol(instrument)
                for k, v in positions.items():
                    if isinstance(k, str) and isinstance(v, dict) and self._norm_symbol(k) == target:
                        inst_pos = v
                        break

            if isinstance(inst_pos, dict):
                raw_dir = inst_pos.get("direction", 0)
                if isinstance(raw_dir, str):
                    raw_dir = self._extract_direction(raw_dir)
                account["position_direction"] = raw_dir
                account["position_size"] = inst_pos.get("size", 0.0)
                account["unrealized_pnl"] = inst_pos.get("unrealized_pnl", 0.0)

        # Enrich with cooldown information if available
        try:
            cooldown_state = bus.get("instrument_cooldown_state", module, default=None)
            if isinstance(cooldown_state, dict):
                inst_cd = cooldown_state.get(instrument)
                if not isinstance(inst_cd, dict):
                    target = self._norm_symbol(instrument)
                    for key, val in cooldown_state.items():
                        if isinstance(key, str) and isinstance(val, dict) and self._norm_symbol(key) == target:
                            inst_cd = val
                            break
                if isinstance(inst_cd, dict):
                    on_cd = bool(inst_cd.get("on_cooldown", False))
                    remaining = float(inst_cd.get("cooldown_remaining", 0.0) or 0.0)
                    account["on_cooldown"] = 1.0 if on_cd and remaining > 0.0 else 0.0
                    account["cooldown_remaining"] = remaining
        except Exception:
            pass

        return account

    # ======================================================================
    # Market Structure Helpers (Swing Points, S/R Proximity)
    # ======================================================================

    def _find_swing_points(self, highs: np.ndarray, lows: np.ndarray, lookback: int = 5) -> tuple[list[float], list[float]]:
        """
        Find swing high/low points for support/resistance detection.
        Uses 5-bar confirmation (2 bars each side).
        """
        support_levels: list[float] = []
        resistance_levels: list[float] = []
        
        if len(highs) < lookback or len(lows) < lookback:
            return support_levels, resistance_levels
        
        # Find swing highs (resistance)
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(float(highs[i]))
        
        # Find swing lows (support)
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(float(lows[i]))
        
        # Keep only the 3 most relevant levels
        resistance_levels = sorted(set(resistance_levels), reverse=True)[:3]
        support_levels = sorted(set(support_levels))[:3]
        
        return support_levels, resistance_levels

    def _compute_sr_proximity(self, current_price: float, support: list[float], resistance: list[float], threshold_pct: float = 0.005) -> tuple[float, float]:
        """
        Compute signed proximity to nearest support/resistance.
        
        Returns:
            near_support: 0.0 to 1.0 (how close to support, 1.0 = at support)
            near_resistance: 0.0 to 1.0 (how close to resistance, 1.0 = at resistance)
        """
        near_support = 0.0
        near_resistance = 0.0
        
        if current_price <= 0:
            return 0.0, 0.0
        
        threshold_abs = current_price * threshold_pct
        
        # Find closest support
        for s in support:
            dist = abs(current_price - s)
            if dist < threshold_abs:
                proximity = 1.0 - (dist / threshold_abs)
                near_support = max(near_support, proximity)
        
        # Find closest resistance
        for r in resistance:
            dist = abs(current_price - r)
            if dist < threshold_abs:
                proximity = 1.0 - (dist / threshold_abs)
                near_resistance = max(near_resistance, proximity)
        
        return float(near_support), float(near_resistance)

    def _compute_market_structure(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """
        Compute market structure signal: HH/HL = bullish, LL/LH = bearish.
        
        Returns:
            -1.0 to +1.0: positive = bullish structure, negative = bearish
        """
        if len(highs) < 10 or len(lows) < 10:
            return 0.0
        
        # Look at last 5 swing periods
        lookback = min(len(highs), 20)
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # Count higher highs vs lower highs
        hh_count = 0
        lh_count = 0
        for i in range(1, len(recent_highs)):
            if recent_highs[i] > recent_highs[i-1]:
                hh_count += 1
            elif recent_highs[i] < recent_highs[i-1]:
                lh_count += 1
        
        # Count higher lows vs lower lows
        hl_count = 0
        ll_count = 0
        for i in range(1, len(recent_lows)):
            if recent_lows[i] > recent_lows[i-1]:
                hl_count += 1
            elif recent_lows[i] < recent_lows[i-1]:
                ll_count += 1
        
        bullish_score = (hh_count + hl_count) / max(lookback - 1, 1)
        bearish_score = (lh_count + ll_count) / max(lookback - 1, 1)
        
        return float(np.clip(bullish_score - bearish_score, -1.0, 1.0))

    # ======================================================================
    # M15 Price Features (PRIMARY) - 10 dims
    # ======================================================================

    def _build_m15_features_for_instrument(self, market_data: Optional[Dict[str, Any]], instrument: str) -> np.ndarray:
        """Build M15 features for a specific instrument safely."""
        if self._is_timeframe_dict(market_data):
            return self._build_m15_features(market_data)

        block = self._lookup_symbol_block(market_data, instrument)
        if isinstance(block, dict):
            return self._build_m15_features(block)

        return np.zeros(10, dtype=np.float32)

    def _build_htf_context_for_instrument(self, market_data: Optional[Dict[str, Any]], instrument: str, expert_signals: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Build HTF context for a specific instrument safely."""
        if self._is_timeframe_dict(market_data):
            return self._build_htf_context(market_data, expert_signals)

        block = self._lookup_symbol_block(market_data, instrument)
        if isinstance(block, dict):
            return self._build_htf_context(block, expert_signals)

        return np.zeros(18, dtype=np.float32)

    def _build_voting_features_for_instrument(self, expert_signals: Optional[Dict[str, Any]], instrument: str) -> np.ndarray:
        """Build voting features for a specific instrument."""
        _ = instrument
        return self._build_voting_features(expert_signals)

    def _build_m15_features(self, market_data: Optional[Dict[str, Any]]) -> np.ndarray:
        """Build M15 primary price features."""
        feats = np.zeros(10, dtype=np.float32)
        if not market_data:
            return feats

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

        c = float(close_arr[-1])
        h = float(high_arr[-1])
        l = float(low_arr[-1])
        o = float(open_arr[-1])
        v = float(vol_arr[-1])

        lb = min(self.config.price_lookback, close_arr.size)
        mean_close = float(np.mean(close_arr[-lb:])) if lb > 0 else c
        mean_vol = float(np.mean(vol_arr[-lb:])) if lb > 0 else max(v, 1.0)

        feats[0] = float((c / max(mean_close, self._eps)) - 1.0)
        feats[1] = float((h - l) / max(abs(c), self._eps))
        feats[2] = float((c - o) / max(abs(o), self._eps))
        feats[3] = float(np.clip((v / max(mean_vol, self._eps)) - 1.0, -1.0, 2.0))

        rsi = self._compute_rsi(close_arr, self.config.rsi_period)
        feats[4] = float((rsi - 50.0) / 50.0)

        # FIX: real MACD histogram (MACD line - signal)
        macd_hist = self._compute_macd_histogram(close_arr)
        denom = max(abs(c) * 0.01, self._eps)
        feats[5] = float(np.clip(macd_hist / denom, -1.0, 1.0))

        atr = self._compute_atr(high_arr, low_arr, close_arr, self.config.atr_period)
        feats[6] = float(np.clip(atr / max(c, self._eps), 0.0, 0.1) * 10.0)

        if close_arr.size >= self.config.trend_lookback:
            slope = self._compute_slope(close_arr[-self.config.trend_lookback:])
            denom_slope = max(abs(c) * 0.001, self._eps)
            feats[7] = float(np.clip(slope / denom_slope, -1.0, 1.0))

        if close_arr.size >= self.config.momentum_lookback:
            base = float(close_arr[-self.config.momentum_lookback])
            roc = (c - base) / max(abs(base), self._eps)
            feats[8] = float(np.clip(roc * 10.0, -1.0, 1.0))

        if close_arr.size >= 20:
            prev = close_arr[-20:-1]
            curr = close_arr[-19:]
            denom_prev = np.maximum(np.abs(prev), self._eps)
            returns = (curr - prev) / denom_prev
            vol = float(np.std(returns))
            feats[9] = float(np.clip(vol * 100.0, 0.0, 1.0))
        
        # Override feats[3] (was volume ratio) with S/R proximity signal
        # This is more useful for learning WHERE to buy/sell
        if high_arr.size >= 10 and low_arr.size >= 10:
            support, resistance = self._find_swing_points(high_arr[-50:], low_arr[-50:])
            near_support, near_resistance = self._compute_sr_proximity(c, support, resistance)
            # SIGNED: positive = near support (good for longs), negative = near resistance (good for shorts)
            feats[3] = float(np.clip(near_support - near_resistance, -1.0, 1.0))

        return feats

    # ======================================================================
    # Higher TF Context (H1/H4/D1) - 18 dims (6 per timeframe)
    # ======================================================================

    def _build_htf_context(self, market_data: Optional[Dict[str, Any]], expert_signals: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Build EXPANDED higher timeframe context features.
        
        v5.4: Each HTF gets 6 dedicated features instead of crushed averages:
        - [0] Trend direction: signed EMA slope (-1 to +1), enhanced by HTF expert if available
        - [1] Momentum: 5-bar return normalized by ATR, enhanced by HTF expert if available
        - [2] RSI: normalized to -1 (oversold 30) to +1 (overbought 70), 0=neutral
        - [3] ATR_norm: volatility as % of price (0 to 1)
        - [4] S/R_proximity: signed distance to nearest S/R (-1=at resistance, +1=at support)
        - [5] HH/HL_bias: recent price structure (-1=lower lows, +1=higher highs)
        
        Total: 18 dims (H1: [0-5], H4: [6-11], D1: [12-17])
        
        When htf_experts are available in expert_signals, they provide more sophisticated
        indicators (ADX, MA alignment) that enhance the raw price-based features.
        """
        feats = np.zeros(18, dtype=np.float32)
        if not market_data:
            return feats

        # Get HTF expert signals if available
        htf_experts = {}
        if expert_signals and isinstance(expert_signals, dict):
            htf_experts = expert_signals.get("htf_experts", {})

        for tf_idx, tf in enumerate(CONTEXT_TIMEFRAMES):  # H1, H4, D1
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
                
            high_arr = np.asarray(high, dtype=np.float64) if high is not None else close_arr
            low_arr = np.asarray(low, dtype=np.float64) if low is not None else close_arr

            base_idx = tf_idx * 6  # 0, 6, 12 for H1, H4, D1
            
            # Get HTF expert signal for this timeframe if available
            htf_sig = htf_experts.get(tf, {}) if isinstance(htf_experts, dict) else {}

            # Feature 0: Trend direction (EMA slope normalized)
            # Enhanced by HTF expert trend signal if available
            trend = self._compute_trend_direction(close_arr)
            if htf_sig:
                # Blend raw trend with expert trend (weighted)
                expert_trend_str = htf_sig.get("trend_strength", 0.0)
                expert_trend_dir = htf_sig.get("trend_direction", "neutral")
                if expert_trend_dir == "bullish":
                    expert_trend = float(expert_trend_str)
                elif expert_trend_dir == "bearish":
                    expert_trend = -float(expert_trend_str)
                else:
                    expert_trend = 0.0
                # MA alignment provides confirmation
                ma_align = htf_sig.get("ma_alignment", 0)
                if ma_align != 0:
                    trend = 0.6 * trend + 0.4 * expert_trend  # Blend
                    if ma_align * np.sign(trend) > 0:  # Alignment confirms
                        trend = np.sign(trend) * min(abs(trend) * 1.2, 1.0)
            feats[base_idx + 0] = float(np.clip(trend, -1.0, 1.0))

            # Feature 1: Momentum (5-bar return normalized by ATR)
            # Enhanced by HTF expert momentum if available
            if close_arr.size >= 6:
                ret_5 = (close_arr[-1] - close_arr[-6]) / max(abs(close_arr[-6]), self._eps)
                atr = self._compute_atr(high_arr, low_arr, close_arr, min(14, close_arr.size - 1))
                atr_pct = atr / max(close_arr[-1], self._eps)
                mom_norm = ret_5 / max(atr_pct * 5, self._eps)  # Normalize by 5 ATRs
                
                if htf_sig:
                    expert_mom_str = htf_sig.get("momentum_strength", 0.0)
                    expert_mom_dir = htf_sig.get("momentum_direction", "neutral")
                    if expert_mom_dir == "bullish":
                        expert_mom = float(expert_mom_str)
                    elif expert_mom_dir == "bearish":
                        expert_mom = -float(expert_mom_str)
                    else:
                        expert_mom = 0.0
                    mom_norm = 0.6 * mom_norm + 0.4 * expert_mom  # Blend
                
                feats[base_idx + 1] = float(np.clip(mom_norm, -1.0, 1.0))

            # Feature 2: RSI (normalized: -1 = oversold, +1 = overbought)
            # Use expert RSI if available (more accurate with proper period)
            if htf_sig and "rsi" in htf_sig:
                rsi = float(htf_sig["rsi"])
            else:
                rsi = self._compute_rsi(close_arr, min(14, close_arr.size - 1))
            # Map RSI 30 -> -1, RSI 50 -> 0, RSI 70 -> +1
            rsi_norm = (rsi - 50.0) / 20.0  # Linear mapping
            feats[base_idx + 2] = float(np.clip(rsi_norm, -1.0, 1.0))

            # Feature 3: ATR normalized (volatility as % of price)
            atr = self._compute_atr(high_arr, low_arr, close_arr, min(14, close_arr.size - 1))
            atr_norm = atr / max(close_arr[-1], self._eps)
            # Scale so typical values are in 0-1 range (0.5% = 0.5, 2% = 1.0)
            feats[base_idx + 3] = float(np.clip(atr_norm * 50.0, 0.0, 1.0))

            # Feature 4: S/R proximity (signed: +1=at support, -1=at resistance)
            if high_arr.size >= 10 and low_arr.size >= 10:
                support, resistance = self._find_swing_points(high_arr[-30:], low_arr[-30:])
                near_support, near_resistance = self._compute_sr_proximity(
                    close_arr[-1], support, resistance
                )
                feats[base_idx + 4] = float(np.clip(near_support - near_resistance, -1.0, 1.0))

            # Feature 5: HH/HL bias (price structure)
            # Use expert structure_bias if available
            if htf_sig and "structure_bias" in htf_sig:
                hh_hl_bias = float(htf_sig["structure_bias"])
            elif close_arr.size >= 10 and high_arr.size >= 10 and low_arr.size >= 10:
                hh_hl_bias = self._compute_hh_hl_bias(high_arr[-20:], low_arr[-20:])
            else:
                hh_hl_bias = 0.0
            feats[base_idx + 5] = float(np.clip(hh_hl_bias, -1.0, 1.0))

        return feats

    def _compute_hh_hl_bias(self, high: np.ndarray, low: np.ndarray) -> float:
        """
        Compute higher-high / lower-low bias for price structure.
        
        Returns:
            +1.0: Clear uptrend (higher highs AND higher lows)
            -1.0: Clear downtrend (lower highs AND lower lows)
            0.0: Consolidation / mixed
        """
        if len(high) < 6 or len(low) < 6:
            return 0.0
            
        # Compare recent peak/trough to earlier peak/trough
        # Split into two halves
        mid = len(high) // 2
        
        # First half peaks/troughs
        first_high = float(np.max(high[:mid]))
        first_low = float(np.min(low[:mid]))
        
        # Second half peaks/troughs  
        second_high = float(np.max(high[mid:]))
        second_low = float(np.min(low[mid:]))
        
        # Count bullish/bearish signals
        score = 0.0
        
        # Higher high = bullish
        if second_high > first_high:
            score += 0.5
        elif second_high < first_high:
            score -= 0.5
            
        # Higher low = bullish
        if second_low > first_low:
            score += 0.5
        elif second_low < first_low:
            score -= 0.5
            
        return score

    # ======================================================================
    # Voting Expert Signals - 8 dims
    # ======================================================================

    def _build_voting_features(self, expert_signals: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Build expert signals as signed features for PPO to learn from.
        
        Schema v5.3 - Includes divergence and regime signals:
        [0] Trend: signed direction * strength
        [1] Trend: confidence + divergence bonus (0.2 if divergence aligns)
        [2] Momentum: signed direction * strength (boosted by divergence)
        [3] Momentum: confidence + overbought/oversold signal
        [4] Theme: signed direction * strength
        [5] Theme: confidence + regime risk signal
        [6] Seasonality: signed direction * strength
        [7] Seasonality: confidence
        """
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

            strength_raw = sig.get("score", sig.get("strength", sig.get("magnitude", 0.0)))
            try:
                strength_val = abs(float(strength_raw))
            except (TypeError, ValueError):
                strength_val = 0.0

            direction = str(sig.get("direction", sig.get("action", "neutral"))).lower()
            if direction in ("bullish", "long", "buy"):
                signed_strength = strength_val
            elif direction in ("bearish", "short", "sell"):
                signed_strength = -strength_val
            else:
                signed_strength = 0.0

            conf_raw = sig.get("confidence", 0.0)
            try:
                conf_val = float(conf_raw)
            except (TypeError, ValueError):
                conf_val = 0.0
            
            proposal = sig.get("proposal", {}) if isinstance(sig, dict) else {}
            
            # Enhanced feature encoding based on expert type
            if name == "momentum":
                # Boost momentum signal if divergence present
                divergence = proposal.get("divergence_signal") if isinstance(proposal, dict) else None
                if divergence == "bullish":
                    signed_strength = max(signed_strength, 0.3)  # Ensure bullish bias
                    signed_strength = min(signed_strength + 0.2, 1.0)  # Boost
                elif divergence == "bearish":
                    signed_strength = min(signed_strength, -0.3)  # Ensure bearish bias
                    signed_strength = max(signed_strength - 0.2, -1.0)  # Boost
                
                feats[i * 2] = float(np.clip(signed_strength, -1.0, 1.0))
                
                # Encode overbought/oversold into confidence slot
                overbought = float(proposal.get("overbought", 0)) if isinstance(proposal, dict) else 0.0
                oversold = float(proposal.get("oversold", 0)) if isinstance(proposal, dict) else 0.0
                # Overbought adds negative bias, oversold adds positive bias (reversal signal)
                ob_signal = oversold - overbought  # -1 to +1
                feats[i * 2 + 1] = float(np.clip(conf_val + ob_signal * 0.3, 0.0, 1.0))
                
            elif name == "theme":
                feats[i * 2] = float(np.clip(signed_strength, -1.0, 1.0))
                
                # Encode risk regime into confidence slot
                risk_regime = proposal.get("risk_regime", "neutral") if isinstance(proposal, dict) else "neutral"
                vol_regime = proposal.get("volatility_regime", "normal") if isinstance(proposal, dict) else "normal"
                
                regime_bonus = 0.0
                if risk_regime == "risk_on":
                    regime_bonus = 0.2  # Higher confidence in risk-on
                elif risk_regime == "risk_off":
                    regime_bonus = -0.1  # Lower confidence in risk-off
                
                if vol_regime == "high":
                    regime_bonus -= 0.1  # High vol = less confident
                
                feats[i * 2 + 1] = float(np.clip(conf_val + regime_bonus, 0.0, 1.0))
                
            else:
                # Standard encoding for trend and seasonality
                feats[i * 2] = float(np.clip(signed_strength, -1.0, 1.0))
                feats[i * 2 + 1] = float(np.clip(conf_val, 0.0, 1.0))

        return feats

    # ======================================================================
    # Committee / Consensus Metrics - 8 dims
    # ======================================================================

    def _build_committee_features(
        self,
        committee_state: Optional[Dict[str, Any]],
        expert_signals: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        feats = np.zeros(8, dtype=np.float32)

        committee_state = committee_state or {}
        expert_signals = expert_signals or {}

        consensus_score = committee_state.get("consensus_score", 0.5)
        consensus_action = str(committee_state.get("action", committee_state.get("direction", "flat"))).lower()

        if consensus_action in ("long", "bullish", "buy"):
            signed_consensus = abs(float(consensus_score))
        elif consensus_action in ("short", "bearish", "sell"):
            signed_consensus = -abs(float(consensus_score))
        else:
            signed_consensus = 0.0
        feats[0] = float(np.clip(signed_consensus, -1.0, 1.0))

        feats[1] = float(np.clip(committee_state.get("confidence", 0.5), 0.0, 1.0))

        experts = expert_signals.get("experts", {}) if isinstance(expert_signals, dict) else {}
        signed_expert_scores: list[float] = []
        if isinstance(experts, dict):
            for name in ["trend", "momentum", "theme", "seasonality"]:
                sig = experts.get(name, {})
                if not isinstance(sig, dict):
                    continue
                strength = sig.get("score", sig.get("strength", 0.0))
                direction = str(sig.get("direction", sig.get("action", "neutral"))).lower()
                try:
                    strength_val = abs(float(strength))
                    if direction in ("bullish", "long", "buy"):
                        signed_expert_scores.append(strength_val)
                    elif direction in ("bearish", "short", "sell"):
                        signed_expert_scores.append(-strength_val)
                    else:
                        signed_expert_scores.append(0.0)
                except (TypeError, ValueError):
                    pass

        if len(signed_expert_scores) >= 2:
            variance = float(np.var(signed_expert_scores))
            feats[2] = float(np.clip(1.0 / (1.0 + variance * 10), 0.0, 1.0))
        else:
            feats[2] = 0.5

        expert_confidences: list[float] = []
        if isinstance(experts, dict):
            for name in ["trend", "momentum", "theme", "seasonality"]:
                sig = experts.get(name, {})
                if isinstance(sig, dict):
                    conf = sig.get("confidence", 0.0)
                    try:
                        expert_confidences.append(float(conf))
                    except (TypeError, ValueError):
                        pass
        feats[3] = float(np.clip(np.mean(expert_confidences), 0.0, 1.0)) if expert_confidences else 0.5

        feats[4] = float(np.clip(committee_state.get("fragility", 0.5), 0.0, 1.0))

        market = expert_signals.get("market", {}) if isinstance(expert_signals, dict) else {}
        regime = (market.get("regime", "unknown") if isinstance(market, dict) else "unknown")
        regime_map = {
            "trending": 0.8,
            "uptrend": 0.8,
            "downtrend": 0.8,
            "mean_reverting": 0.3,
            "ranging": 0.2,
            "volatile": 0.5,
            "unknown": 0.5,
        }
        feats[5] = float(regime_map.get(str(regime).lower(), 0.5))

        regime_strength = 0.5
        if isinstance(market, dict):
            try:
                regime_strength = float(market.get("regime_strength", 0.5))
            except (TypeError, ValueError):
                regime_strength = 0.5
        feats[6] = float(np.clip(regime_strength, 0.0, 1.0))

        # Extract ADVANCED market structure from TrendExpert (if available)
        trend_sig = experts.get("trend", {}) if isinstance(experts, dict) else {}
        proposal = trend_sig.get("proposal", {}) if isinstance(trend_sig, dict) else {}
        if isinstance(proposal, dict):
            # Combine multiple structure signals into one composite:
            # - near_support (positive for longs)
            # - near_resistance (negative for shorts)  
            # - structure_trend (HH/HL vs LL/LH)
            # - bos_signal (break of structure)
            # - order_block proximity
            near_support = float(proposal.get("near_support", 0))
            near_resistance = float(proposal.get("near_resistance", 0))
            structure_trend = float(proposal.get("structure_trend", 0))  # -1 to +1
            bos_signal = float(proposal.get("bos_signal", 0))  # -1 to +1
            ob_bull = float(proposal.get("order_block_bull", 0))  # 0 to 1
            ob_bear = float(proposal.get("order_block_bear", 0))  # 0 to 1
            
            # Composite market structure signal:
            # Positive = bullish structure (near support, HH/HL, bullish BOS, bullish OB)
            # Negative = bearish structure (near resistance, LL/LH, bearish BOS, bearish OB)
            sr_signal = near_support - near_resistance  # -1 to +1
            ob_signal = ob_bull - ob_bear  # -1 to +1
            
            # Weight the signals: S/R proximity (40%), structure trend (30%), BOS (20%), OB (10%)
            composite = (sr_signal * 0.40 + structure_trend * 0.30 + bos_signal * 0.20 + ob_signal * 0.10)
            feats[7] = float(np.clip(composite, -1.0, 1.0))
        else:
            # Fallback: use conviction count if no S/R data
            strong_conviction_count = sum(1 for s in signed_expert_scores if abs(s) > 0.5)
            feats[7] = float(strong_conviction_count / max(len(signed_expert_scores), 1))

        return feats

    # ======================================================================
    # Risk / Memory Signals - 8 dims
    # ======================================================================

    def _build_risk_features(
        self,
        risk_state: Optional[Dict[str, Any]],
        memory_state: Optional[Dict[str, Any]],
        account_state: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        feats = np.zeros(8, dtype=np.float32)

        risk_state = risk_state or {}
        memory_state = memory_state or {}
        account_state = account_state or {}

        memory_gate = memory_state.get("memory_gate", 1.0)
        if isinstance(memory_gate, dict):
            memory_gate = memory_gate.get("risk_multiplier", 1.0)
        try:
            mem_val = float(memory_gate)
        except (TypeError, ValueError):
            mem_val = 1.0
        feats[0] = float(np.clip(mem_val, 0.0, 1.0))

        danger_zones = memory_state.get("danger_zones", {})
        if isinstance(danger_zones, dict):
            count = int(danger_zones.get("zone_count", 0))
        elif isinstance(danger_zones, list):
            count = len(danger_zones)
        else:
            count = 0
        feats[1] = float(np.clip(count / max(self.config.max_danger_zones, 1), 0.0, 1.0))

        try:
            drawdown = float(account_state.get("current_drawdown", 0.0))
        except (TypeError, ValueError):
            drawdown = 0.0
        feats[2] = float(np.clip(drawdown / max(self.config.max_drawdown_clip, self._eps), 0.0, 1.0))

        try:
            balance = float(account_state.get("balance", 100000.0))
        except (TypeError, ValueError):
            balance = 100000.0
        try:
            initial = float(account_state.get("initial_balance", 100000.0))
        except (TypeError, ValueError):
            initial = 100000.0
        feats[3] = float(np.clip(balance / max(initial, 1.0), 0.0, 2.0))

        portfolio_risk = risk_state.get("portfolio_risk", {})
        exposure = 0.0
        if isinstance(portfolio_risk, dict):
            try:
                exposure = float(portfolio_risk.get("total_exposure", 0.0))
            except (TypeError, ValueError):
                exposure = 0.0
        feats[4] = float(np.clip(exposure, 0.0, 1.0))

        risk_budget = risk_state.get("risk_budget", 1.0)
        try:
            rb_val = float(risk_budget)
        except (TypeError, ValueError):
            rb_val = 1.0
        feats[5] = float(np.clip(rb_val, 0.0, 1.0))

        win_rate = account_state.get("win_rate", 0.5)
        try:
            win_val = float(win_rate)
        except (TypeError, ValueError):
            win_val = 0.5
        feats[6] = float(np.clip(win_val, 0.0, 1.0))

        pnl_trend = account_state.get("pnl_trend", 0.0)
        try:
            pnl_val = float(pnl_trend)
        except (TypeError, ValueError):
            pnl_val = 0.0
        feats[7] = float(np.clip(pnl_val, -1.0, 1.0))

        return feats

    # ======================================================================
    # Account / Position State - 8 dims
    # ======================================================================

    def _build_account_features(self, account_state: Optional[Dict[str, Any]]) -> np.ndarray:
        feats = np.zeros(8, dtype=np.float32)
        account_state = account_state or {}

        step = int(account_state.get("current_step", 0))
        max_steps = int(account_state.get("max_steps", 10000))
        feats[0] = float(np.clip(step / max(max_steps, 1), 0.0, 1.0))

        episode_return = account_state.get("episode_return", 0.0)
        try:
            ep_val = float(episode_return)
        except (TypeError, ValueError):
            ep_val = 0.0
        feats[1] = float(np.clip(ep_val / 100.0, -1.0, 1.0))

        position_dir = account_state.get("position_direction", 0)
        if isinstance(position_dir, str):
            position_dir = self._extract_direction(position_dir)
        try:
            pos_dir_val = float(position_dir)
        except (TypeError, ValueError):
            pos_dir_val = 0.0
        feats[2] = float(np.clip(pos_dir_val, -1.0, 1.0))

        position_size = account_state.get("position_size", 0.0)
        try:
            pos_size_val = float(position_size)
        except (TypeError, ValueError):
            pos_size_val = 0.0
        feats[3] = float(np.clip(pos_size_val, 0.0, 1.0))

        unrealized = account_state.get("unrealized_pnl", 0.0)
        try:
            initial = float(account_state.get("initial_balance", 100000.0))
        except (TypeError, ValueError):
            initial = 100000.0
        try:
            unreal_val = float(unrealized)
        except (TypeError, ValueError):
            unreal_val = 0.0
        feats[4] = float(np.clip(unreal_val / max(initial * 0.01, 1.0), -1.0, 1.0))

        time_in_pos = account_state.get("time_in_position", 0)
        try:
            tip_val = float(time_in_pos)
        except (TypeError, ValueError):
            tip_val = 0.0
        feats[5] = float(np.clip(tip_val / 100.0, 0.0, 1.0))

        trades = account_state.get("trades_today", 0)
        try:
            trades_val = float(trades)
        except (TypeError, ValueError):
            trades_val = 0.0
        feats[6] = float(np.clip(trades_val / max(self.config.max_trades_per_day, 1), 0.0, 1.0))

        if "on_cooldown" in account_state:
            try:
                cd_val = float(account_state.get("on_cooldown") or 0.0)
            except (TypeError, ValueError):
                cd_val = 0.0
            feats[7] = float(np.clip(cd_val, 0.0, 1.0))
        else:
            last_action = account_state.get("last_action", 0.0)
            try:
                la_val = float(last_action)
            except (TypeError, ValueError):
                la_val = 0.0
            feats[7] = float(np.clip(la_val, -1.0, 1.0))

        return feats

    # ======================================================================
    # World Model Features (v4.0) - 8 dims
    # ======================================================================

    def _build_world_model_features(self, world_model_state: Optional[Dict[str, Any]]) -> np.ndarray:
        feats = np.zeros(8, dtype=np.float32)
        if not world_model_state or not isinstance(world_model_state, dict):
            return feats

        predictions = world_model_state.get("market_predictions", world_model_state)
        if not isinstance(predictions, dict):
            predictions = {}

        latest = predictions.get("latest_predictions", predictions)

        base_conf = predictions.get("model_confidence", latest.get("confidence", 0.0))
        extra_conf = world_model_state.get("prediction_confidence")
        if extra_conf is not None:
            try:
                if isinstance(extra_conf, dict):
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

        price_changes = latest.get("price_changes", predictions.get("price_changes", []))
        if isinstance(price_changes, (list, np.ndarray)) and len(price_changes) > 0:
            try:
                m15_change = float(price_changes[0]) if len(price_changes) > 0 else 0.0
                feats[1] = float(np.clip(m15_change * 100.0, -1.0, 1.0))

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

        vol_preds = latest.get("volatility_predictions", predictions.get("volatility_predictions", []))
        if isinstance(vol_preds, (list, np.ndarray)) and len(vol_preds) > 0:
            try:
                feats[3] = float(np.clip(float(vol_preds[0]), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[3] = 0.5
        else:
            feats[3] = 0.5

        regime_probs = latest.get("regime_probabilities", predictions.get("regime_probabilities", []))
        predicted_regime = latest.get("predicted_regime", predictions.get("predicted_regime", -1))
        regime_map = {0: 0.8, 1: -0.8, 2: 0.3, 3: 0.0}
        if isinstance(predicted_regime, int) and predicted_regime in regime_map:
            feats[4] = regime_map[predicted_regime]
        elif isinstance(regime_probs, (list, np.ndarray)) and len(regime_probs) >= 4:
            try:
                regime_idx = int(np.argmax(regime_probs))
                feats[4] = regime_map.get(regime_idx, 0.0)
            except (TypeError, ValueError):
                feats[4] = 0.0

        is_trained = predictions.get("is_trained", latest.get("model_trained", False))
        feats[5] = 1.0 if is_trained else 0.0

        scenarios = world_model_state.get("scenario_generation", world_model_state.get("scenarios", {}))
        if isinstance(scenarios, dict):
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

        stability = predictions.get("stability_score", predictions.get("prediction_quality", 0.5))
        try:
            feats[7] = float(np.clip(float(stability), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[7] = 0.5

        conf_val = float(feats[0])
        if conf_val < self.config.prediction_confidence_threshold or not bool(is_trained):
            feats[1] = 0.0
            feats[2] = 0.0
            feats[3] = 0.5
            feats[4] = 0.0
            feats[6] = float(np.clip(feats[6], 0.25, 0.75))

        return feats

    # ======================================================================
    # Trading Mode Features - 8 dims (with timing integration)
    # ======================================================================

    def _build_trading_mode_features(self, trading_mode_state: Optional[Dict[str, Any]]) -> np.ndarray:
        feats = np.zeros(8, dtype=np.float32)

        # Defaults
        feats[0] = 0.5
        feats[1] = 1.0
        feats[2] = 0.5
        feats[3] = 0.5
        feats[4] = 0.5
        feats[5] = 0.33
        feats[6] = 0.5
        feats[7] = 0.5

        if not trading_mode_state or not isinstance(trading_mode_state, dict):
            return feats

        mode = trading_mode_state.get("trading_mode", trading_mode_state.get("current_mode", "normal"))
        mode_map = {"safe": 0.25, "normal": 0.5, "aggressive": 0.75, "extreme": 1.0}
        if isinstance(mode, str):
            feats[0] = mode_map.get(mode.lower(), 0.5)

        regime_stability = trading_mode_state.get("regime_stability", 0.5)
        theme_transition = trading_mode_state.get("theme_transition", 0.0)
        theme_strength = trading_mode_state.get("theme_strength", 0.0)
        regime_accuracy = trading_mode_state.get("regime_accuracy", 0.5)
        risk_scaling_factor = trading_mode_state.get("risk_scaling_factor", 1.0)
        liquidity_score = trading_mode_state.get("liquidity_score", 0.5)

        timing = trading_mode_state.get("entry_timing", {})
        if isinstance(timing, dict) and timing:
            feats[1] = 1.0 if timing.get("entry_allowed", True) else 0.0

            eql = timing.get("entry_quality_long", 0.5)
            eqs = timing.get("entry_quality_short", 0.5)
            try:
                avg_quality = (float(eql) + float(eqs)) / 2.0
                stability_weight = float(regime_stability)
                feats[2] = float(np.clip(avg_quality * (0.5 + 0.5 * stability_weight), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[2] = 0.5

            try:
                ts = float(theme_strength)
                tt = float(theme_transition)
                theme_stability = ts * (1.0 - min(tt, 1.0))
                feats[3] = float(np.clip(theme_stability, 0.0, 1.0))
            except (TypeError, ValueError):
                feats[3] = 0.5

            zone_type = timing.get("zone_type", "good")
            zone_map = {"hot": 0.9, "good": 0.5, "bad": 0.1}
            base_zone = zone_map.get(zone_type, 0.5) if isinstance(zone_type, str) else 0.5
            try:
                acc = float(regime_accuracy)
                feats[4] = float(np.clip(base_zone * (0.5 + 0.5 * acc), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[4] = base_zone

            vol_state = timing.get("vol_state", "normal")
            vol_map = {"low": 0.0, "normal": 0.33, "high": 0.66, "extreme": 1.0}
            base_vol = vol_map.get(vol_state, 0.33) if isinstance(vol_state, str) else 0.33
            try:
                rsf = float(risk_scaling_factor)
                rsf_norm = float(np.clip((rsf - 0.5) / 1.5, 0.0, 1.0))
                feats[5] = float(np.clip((base_vol + rsf_norm) / 2.0, 0.0, 1.0))
            except (TypeError, ValueError):
                feats[5] = base_vol

            try:
                feats[6] = float(np.clip(float(liquidity_score), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[6] = 0.5

        else:
            mode_config = trading_mode_state.get("mode_config", {})
            if isinstance(mode_config, dict):
                max_exp = mode_config.get("max_exposure", 0.6)
                try:
                    feats[2] = float(np.clip(float(max_exp) * float(regime_stability), 0.0, 1.0))
                except (TypeError, ValueError):
                    feats[2] = 0.5

            try:
                ts = float(theme_strength)
                tt = float(theme_transition)
                feats[3] = float(np.clip(ts * (1.0 - min(tt, 1.0)), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[3] = 0.5

            decision_factors = trading_mode_state.get("decision_factors", {})
            if isinstance(decision_factors, dict):
                perf_score = decision_factors.get("performance_score", 0.5)
                try:
                    feats[4] = float(np.clip(float(perf_score) * (0.5 + 0.5 * float(regime_accuracy)), 0.0, 1.0))
                except (TypeError, ValueError):
                    feats[4] = 0.5

                stab_score = decision_factors.get("stability_score", 0.5)
                try:
                    rsf = float(risk_scaling_factor)
                    rsf_norm = float(np.clip((rsf - 0.5) / 1.5, 0.0, 1.0))
                    feats[5] = float(np.clip((float(stab_score) + rsf_norm) / 2.0, 0.0, 1.0))
                except (TypeError, ValueError):
                    feats[5] = 0.5

            try:
                feats[6] = float(np.clip(float(liquidity_score), 0.0, 1.0))
            except (TypeError, ValueError):
                feats[6] = 0.5

        mode_stats = trading_mode_state.get("mode_stats", {})
        if isinstance(mode_stats, dict):
            mode_eff = mode_stats.get("mode_effectiveness", 0.5)
            try:
                me = float(mode_eff)
                rs = float(regime_stability)
                # Add time-of-day awareness: use prime window + hour info
                timing = trading_mode_state.get("entry_timing", {})
                prime_bonus = 0.0 if not isinstance(timing, dict) else float(timing.get("in_prime_window", 0.0) or 0.0)
                hour_norm = 0.5 if not isinstance(timing, dict) else float(timing.get("hour_normalized", 0.5) or 0.5)
                # Combine: mode effectiveness, regime stability, and time quality
                time_quality = 0.5 + 0.3 * prime_bonus  # Prime window adds 0.3
                feats[7] = float(np.clip(me * 0.35 + rs * 0.25 + time_quality * 0.40, 0.0, 1.0))
            except (TypeError, ValueError):
                feats[7] = 0.5

        return feats

    # ======================================================================
    # Governor/Budget Features (v5.5) - 8 dims
    # ======================================================================

    def _build_governor_features(self, governor_state: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Build governor/budget observation features (v5.5).
        
        These features expose the agent's constraint state so it can learn to avoid
        hitting hard blocks. ALWAYS computed (even with loose limits in early stages)
        to prevent observation distribution shift during curriculum progression.
        
        [0] loss_layer_ratio: consecutive_losses / loss_layer_stop [0,1]
        [1] loss_layer_level: clamped loss layer / 5.0 [0,1]
        [2] win_streak_ratio: consecutive_wins / 5.0 [0,1]
        [3] session_pnl_headroom: remaining session loss headroom [0,1]
        [4] session_trade_budget: 1 - session_trades/max [0,1]
        [5] session_consec_loss_ratio: session consec losses / limit [0,1]
        [6] session_progress: bars into session / duration [0,1]
        [7] pending_order_progress: bars until fill / max_latency [0,1]
        """
        feats = np.zeros(8, dtype=np.float32)
        
        if not governor_state or not isinstance(governor_state, dict):
            # Default: all signals at neutral/safe values
            feats[3] = 1.0  # Full headroom
            feats[4] = 1.0  # Full trade budget
            return feats
        
        # [0] Loss layer ratio: how close to loss_layer_stop
        try:
            feats[0] = float(np.clip(governor_state.get("loss_layer_ratio", 0.0), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[0] = 0.0
        
        # [1] Loss layer level: current behavioral restriction level (0-5 normalized)
        try:
            feats[1] = float(np.clip(governor_state.get("loss_layer_level", 0.0), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[1] = 0.0
        
        # [2] Win streak ratio: winning momentum signal
        try:
            feats[2] = float(np.clip(governor_state.get("win_streak_ratio", 0.0), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[2] = 0.0
        
        # [3] Session PnL headroom: normalized to [0,1] from [0,2] raw
        try:
            raw_headroom = float(governor_state.get("session_pnl_headroom", 1.0))
            feats[3] = float(np.clip(raw_headroom / 2.0, 0.0, 1.0))
        except (TypeError, ValueError):
            feats[3] = 1.0  # Default: full headroom
        
        # [4] Session trade budget: remaining capacity
        try:
            feats[4] = float(np.clip(governor_state.get("session_trade_budget", 1.0), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[4] = 1.0  # Default: full budget
        
        # [5] Session consecutive loss ratio
        try:
            feats[5] = float(np.clip(governor_state.get("session_consec_loss_ratio", 0.0), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[5] = 0.0
        
        # [6] Session progress: how far into current session
        try:
            feats[6] = float(np.clip(governor_state.get("session_progress", 0.0), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[6] = 0.0
        
        # [7] Pending order progress: time until pending order fills
        try:
            feats[7] = float(np.clip(governor_state.get("pending_order_progress", 0.0), 0.0, 1.0))
        except (TypeError, ValueError):
            feats[7] = 0.0
        
        return feats

    def _fetch_governor_state(self, bus: Any, module: str) -> Dict[str, Any]:
        """
        Fetch governor state from InfoBus for live trading.
        
        In live, this must be published to InfoBus with key 'governor_state'
        by the live trading system, containing the same fields as
        PropFirmTradingEnv._get_governor_state().
        
        Fallback chain:
        1. Direct 'governor_state' key
        2. Build from 'trade_statistics' + 'market_state'
        3. Build from 'risk_assessment' + 'session_metrics'
        """
        try:
            # 1. Direct governor_state key (preferred)
            state = bus.get("governor_state", module)
            if isinstance(state, dict) and "loss_layer_ratio" in state:
                return state
            
            # 2. Fallback: build from trade_statistics + market_state
            trade_stats = bus.get("trade_statistics", module) or {}
            market_state = bus.get("market_state", module) or {}
            
            if isinstance(trade_stats, dict) and trade_stats:
                consecutive_losses = int(trade_stats.get("consecutive_losses", 0))
                consecutive_wins = int(trade_stats.get("consecutive_wins", 0))
                session_trades = int(trade_stats.get("session_trades", 0))
                session_pnl = float(trade_stats.get("session_pnl", 0.0))
                session_consecutive_losses = int(trade_stats.get("session_consecutive_losses", 0))
                
                # Get limits from config or use defaults
                loss_layer_stop = int(market_state.get("loss_layer_stop", 5))
                session_loss_limit = float(market_state.get("session_loss_limit_pct", 0.99))
                session_consec_limit = int(market_state.get("session_consecutive_loss_limit", 99))
                max_session_trades = int(market_state.get("max_trades_per_session", 99))
                session_start_balance = float(market_state.get("session_start_balance", 100000))
                
                # Compute headroom
                if session_start_balance > 0:
                    session_pnl_pct = session_pnl / session_start_balance
                else:
                    session_pnl_pct = 0.0
                headroom = (session_loss_limit + session_pnl_pct) / max(session_loss_limit, 0.001)
                
                return {
                    "loss_layer_ratio": float(np.clip(consecutive_losses / max(loss_layer_stop, 1), 0.0, 1.0)),
                    "loss_layer_level": float(np.clip(min(consecutive_losses, 5) / 5.0, 0.0, 1.0)),
                    "win_streak_ratio": float(np.clip(consecutive_wins / 5.0, 0.0, 1.0)),
                    "session_pnl_headroom": float(np.clip(headroom, 0.0, 2.0)),
                    "session_trade_budget": float(np.clip(1.0 - (session_trades / max(max_session_trades, 1)), 0.0, 1.0)),
                    "session_consec_loss_ratio": float(np.clip(session_consecutive_losses / max(session_consec_limit, 1), 0.0, 1.0)),
                    "session_progress": 0.5,  # Default mid-session
                    "pending_order_progress": 0.0,
                }
            
            # 3. Fallback: build from risk_assessment + session_metrics (DynamicRiskController publishes these)
            risk_assessment = bus.get("risk_assessment", module) or {}
            session_metrics = bus.get("session_metrics", module) or {}
            
            if isinstance(risk_assessment, dict) and risk_assessment:
                consecutive_losses = int(risk_assessment.get("consecutive_losses", 0))
                consecutive_wins = int(risk_assessment.get("consecutive_wins", 0))
                
                # Session data from session_metrics
                session_trades = int(session_metrics.get("session_trades", 0)) if isinstance(session_metrics, dict) else 0
                session_pnl = float(session_metrics.get("session_pnl", 0.0)) if isinstance(session_metrics, dict) else 0.0
                
                # Use defaults for session limits (not tracked by risk controller)
                loss_layer_stop = 5
                session_loss_limit = 0.99
                session_consec_limit = 99
                max_session_trades = 99
                session_start_balance = 100000
                
                # Compute headroom
                if session_start_balance > 0:
                    session_pnl_pct = session_pnl / session_start_balance
                else:
                    session_pnl_pct = 0.0
                headroom = (session_loss_limit + session_pnl_pct) / max(session_loss_limit, 0.001)
                
                return {
                    "loss_layer_ratio": float(np.clip(consecutive_losses / max(loss_layer_stop, 1), 0.0, 1.0)),
                    "loss_layer_level": float(np.clip(min(consecutive_losses, 5) / 5.0, 0.0, 1.0)),
                    "win_streak_ratio": float(np.clip(consecutive_wins / 5.0, 0.0, 1.0)),
                    "session_pnl_headroom": float(np.clip(headroom, 0.0, 2.0)),
                    "session_trade_budget": float(np.clip(1.0 - (session_trades / max(max_session_trades, 1)), 0.0, 1.0)),
                    "session_consec_loss_ratio": 0.0,  # Not tracked separately in live
                    "session_progress": 0.5,  # Default mid-session
                    "pending_order_progress": 0.0,
                }
                
        except Exception:
            pass
        
        # Return defaults - log once so operator notices missing data without spamming logs
        import logging
        logger = logging.getLogger("PPOObservationBuilder")
        if not self._warned_missing_governor_state:
            self._warned_missing_governor_state = True
            logger.warning(
                "[LIVE] Governor state not found on InfoBus - using defaults. "
                "Ensure LiveSessionTracker is running and publishing 'governor_state'. "
                "Observation dims [76-83] will be defaults until fixed."
            )
        return {
            "loss_layer_ratio": 0.0,
            "loss_layer_level": 0.0,
            "win_streak_ratio": 0.0,
            "session_pnl_headroom": 1.0,
            "session_trade_budget": 1.0,
            "session_consec_loss_ratio": 0.0,
            "session_progress": 0.0,
            "pending_order_progress": 0.0,
        }

    # ======================================================================
    # SmartInfoBus Data Fetchers
    # ======================================================================

    def _fetch_market_data(self, bus: Any, module: str) -> Dict[str, Any]:
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
        """
        Fetch expert voting signals in the schema expected by _build_voting_features():
        each expert has at least: direction, score, confidence, and proposal dict.
        
        Schema v5.3: Also fetches proposal sub-signals for:
        - TrendExpert: near_support, near_resistance, structure_trend, bos_signal, etc.
        - MomentumExpert: divergence_signal, overbought, oversold, rsi_value
        - ThemeExpert: volatility_regime, risk_regime, vol_score
        """
        try:
            def _expert_block(vote_key: str, conf_key: str, expert_name: str) -> Dict[str, Any]:
                proposal = bus.get(vote_key, module) or "flat"
                conf_raw = bus.get(conf_key, module)

                try:
                    conf = float(conf_raw) if conf_raw is not None else 0.0
                except (TypeError, ValueError):
                    conf = 0.0

                d = self._extract_direction(proposal)
                if d > 0:
                    direction = "bullish"
                elif d < 0:
                    direction = "bearish"
                else:
                    direction = "neutral"

                score = float(np.clip(conf, 0.0, 1.0))
                
                # Extract proposal dict with sub-signals (v5.3 schema)
                proposal_dict: Dict[str, Any] = {}
                if isinstance(proposal, dict):
                    # Raw proposal is a dict - extract sub-signals
                    proposal_dict = proposal
                else:
                    # Try to get per_instrument analysis for richer signals
                    per_inst = bus.get(f"{expert_name}_per_instrument", module)
                    if isinstance(per_inst, dict):
                        # Take first instrument's analysis as proposal
                        for inst_data in per_inst.values():
                            if isinstance(inst_data, dict):
                                proposal_dict = inst_data
                                break
                
                # Extract specific signals based on expert type
                if expert_name == "MomentumExpert":
                    # Try to get rich momentum analysis from bus
                    mom_analysis = bus.get("momentum_analysis", module)
                    if isinstance(mom_analysis, dict):
                        per_inst_mom = mom_analysis.get("per_instrument", {})
                        if isinstance(per_inst_mom, dict):
                            # Take first instrument's analysis
                            for inst_data in per_inst_mom.values():
                                if isinstance(inst_data, dict):
                                    proposal_dict.update(inst_data)
                                    break
                    
                    # Momentum signals: divergence, overbought/oversold
                    proposal_dict.setdefault("divergence_signal", proposal_dict.get("divergence"))
                    rsi_val = proposal_dict.get("rsi", 50)
                    try:
                        rsi_val = float(rsi_val)
                    except (TypeError, ValueError):
                        rsi_val = 50.0
                    proposal_dict.setdefault("overbought", max(0.0, (rsi_val - 70) / 30) if rsi_val > 70 else 0.0)
                    proposal_dict.setdefault("oversold", max(0.0, (30 - rsi_val) / 30) if rsi_val < 30 else 0.0)
                    proposal_dict.setdefault("rsi_value", rsi_val)
                    
                elif expert_name == "ThemeExpert":
                    # Try to get rich theme analysis from bus
                    theme_analysis = bus.get("theme_analysis", module)
                    if isinstance(theme_analysis, dict):
                        per_inst_theme = theme_analysis.get("per_instrument", {})
                        if isinstance(per_inst_theme, dict):
                            for inst_data in per_inst_theme.values():
                                if isinstance(inst_data, dict):
                                    proposal_dict.update(inst_data)
                                    break
                    
                    # Theme signals: risk_regime, volatility_regime
                    proposal_dict.setdefault("risk_regime", "neutral")
                    proposal_dict.setdefault("volatility_regime", "normal")
                    proposal_dict.setdefault("vol_score", 0.5)
                
                elif expert_name == "TrendExpert":
                    # Try to get rich trend analysis from bus  
                    trend_analysis = bus.get("trend_analysis", module)
                    if isinstance(trend_analysis, dict):
                        per_inst_trend = trend_analysis.get("per_instrument", {})
                        if isinstance(per_inst_trend, dict):
                            for inst_data in per_inst_trend.values():
                                if isinstance(inst_data, dict):
                                    proposal_dict.update(inst_data)
                                    break
                    
                    # Trend signals: S/R, structure, BOS, order blocks
                    proposal_dict.setdefault("near_support", 0.0)
                    proposal_dict.setdefault("near_resistance", 0.0)
                    proposal_dict.setdefault("structure_trend", 0.0)
                    proposal_dict.setdefault("bos_signal", 0.0)
                    proposal_dict.setdefault("order_block_bull", 0.0)
                    proposal_dict.setdefault("order_block_bear", 0.0)

                return {
                    "proposal": proposal_dict,
                    "direction": direction,
                    "score": score,
                    "confidence": score,
                }

            experts = {
                "trend": _expert_block("TrendExpert_voting_proposal", "TrendExpert_confidence", "TrendExpert"),
                "momentum": _expert_block("MomentumExpert_voting_proposal", "MomentumExpert_confidence", "MomentumExpert"),
                "theme": _expert_block("ThemeExpert_voting_proposal", "ThemeExpert_confidence", "ThemeExpert"),
                "seasonality": _expert_block("SeasonalityRiskExpert_voting_proposal", "SeasonalityRiskExpert_confidence", "SeasonalityRiskExpert"),
            }

            market = {
                "regime": bus.get("market_regime", module) or "unknown",
                "regime_strength": float(bus.get("regime_strength", module) or 0.5),
            }

            # Extract HTF expert signals for train/live parity (v5.4)
            htf_experts = self._extract_htf_experts_from_bus(bus, module, instrument=None)

            return {"experts": experts, "htf_experts": htf_experts, "market": market}
        except Exception:
            return {}

    def _extract_htf_experts_from_bus(
        self,
        bus: Any,
        module: str,
        instrument: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract HTF (higher timeframe) expert signals from live SmartBus data.
        
        Maps live TrendExpert mtf_analysis to the training env schema:
        - Live: trend_analysis.per_instrument[*].mtf_analysis.trends.{H1,H4,D1}
        - Training expects: htf_experts.{H1,H4,D1}.{trend_direction, trend_strength, ...}
        
        This ensures train/live parity for HTF context features (v5.4).
        """
        htf_experts: Dict[str, Dict[str, Any]] = {}
        
        # Default neutral values for each timeframe
        default_htf = {
            "trend_direction": "neutral",
            "trend_strength": 0.0,
            "momentum_direction": "neutral",
            "momentum_strength": 0.0,
            "rsi": 50.0,
            "rsi_signal": "neutral",
            "ma_alignment": 0,
            "adx": 0.0,
            "structure_bias": 0.0,
        }
        
        try:
            # Get trend_analysis which contains mtf_analysis per instrument
            trend_analysis = bus.get("trend_analysis", module)
            if not isinstance(trend_analysis, dict):
                return {}
            
            per_inst = trend_analysis.get("per_instrument", {})
            if not isinstance(per_inst, dict) or not per_inst:
                return {}
            
            # Prefer requested instrument to avoid cross-instrument leakage.
            inst_data = self._lookup_symbol_block(per_inst, instrument) if instrument else None
            if not isinstance(inst_data, dict):
                if instrument:
                    return {}
                for inst_analysis in per_inst.values():
                    if isinstance(inst_analysis, dict):
                        inst_data = inst_analysis
                        break
            
            if not inst_data:
                return {}
            
            # Extract mtf_analysis.trends which has per-TF trend data
            mtf_analysis = inst_data.get("mtf_analysis", {})
            if not isinstance(mtf_analysis, dict):
                return {}
            
            mtf_trends = mtf_analysis.get("trends", {})
            if not isinstance(mtf_trends, dict):
                return {}
            
            # Get momentum analysis for RSI (if available)
            mom_analysis = bus.get("momentum_analysis", module)
            per_inst_mom: Dict[str, Any] = {}
            if isinstance(mom_analysis, dict):
                per_inst_mom = mom_analysis.get("per_instrument", {}) or {}

            inst_mom = None
            if isinstance(per_inst_mom, dict) and per_inst_mom:
                inst_mom = self._lookup_symbol_block(per_inst_mom, instrument) if instrument else None
                if not isinstance(inst_mom, dict) and not instrument:
                    for v in per_inst_mom.values():
                        if isinstance(v, dict):
                            inst_mom = v
                            break
            
            # Map each HTF to training schema
            for tf in ["H1", "H4", "D1"]:
                tf_trend = mtf_trends.get(tf, {})
                if not isinstance(tf_trend, dict):
                    htf_experts[tf] = default_htf.copy()
                    continue
                
                # Map live schema to training schema
                direction = tf_trend.get("direction", "neutral")
                strength = float(tf_trend.get("strength", 0.0))
                ma_spread = float(tf_trend.get("ma_spread", 0.0))
                slope = float(tf_trend.get("slope", 0.0))
                
                # Determine MA alignment from ma_spread
                if ma_spread > 0.001:
                    ma_alignment = 1
                elif ma_spread < -0.001:
                    ma_alignment = -1
                else:
                    ma_alignment = 0
                
                # RSI from momentum analysis (if available for this TF)
                rsi = 50.0
                rsi_signal = "neutral"
                if isinstance(inst_mom, dict):
                    # Momentum may have per-TF RSI or just primary TF
                    rsi = float(inst_mom.get("rsi", 50.0))
                    if rsi > 70:
                        rsi_signal = "overbought"
                    elif rsi < 30:
                        rsi_signal = "oversold"
                
                # Structure bias from slope
                structure_bias = float(np.clip(slope * 50.0, -1.0, 1.0))
                
                # ADX from instrument data (may not be per-TF in live)
                adx = float(inst_data.get("adx", 0.0))
                
                htf_experts[tf] = {
                    "trend_direction": direction,
                    "trend_strength": strength,
                    "momentum_direction": direction,  # Use trend direction as proxy
                    "momentum_strength": strength * 0.8,  # Slightly lower for momentum
                    "rsi": rsi,
                    "rsi_signal": rsi_signal,
                    "ma_alignment": ma_alignment,
                    "adx": adx,
                    "structure_bias": structure_bias,
                }
            
            return htf_experts
            
        except Exception:
            return {}

    def _fetch_committee_state(self, bus: Any, module: str) -> Dict[str, Any]:
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
        try:
            risk_data = bus.get("risk_data", module) or {}
            portfolio_risk = bus.get("portfolio_risk", module) or {}

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
        try:
            return {
                "memory_gate": bus.get("memory_gate", module) or 1.0,
                "danger_zones": bus.get("danger_zones", module) or {},
            }
        except Exception:
            return {}

    def _fetch_account_state(self, bus: Any, module: str) -> Dict[str, Any]:
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
                "win_rate": float(market_state.get("win_rate", 0.5)),
                "pnl_trend": float(market_state.get("pnl_trend", 0.0)),
                "trades_today": int(market_state.get("trades_today", 0)),
            }

            if isinstance(positions, dict) and "net" in positions and isinstance(positions["net"], dict):
                net = positions["net"]
                raw_dir = net.get("direction", 0)
                if isinstance(raw_dir, str):
                    raw_dir = self._extract_direction(raw_dir)
                state.setdefault("position_direction", raw_dir)
                state.setdefault("position_size", net.get("size", 0.0))
                state.setdefault("unrealized_pnl", net.get("unrealized_pnl", 0.0))

            return state
        except Exception:
            return {}

    def _fetch_world_model_state(self, bus: Any, module: str) -> Dict[str, Any]:
        try:
            market_predictions = bus.get("market_predictions", module) or {}
            prediction_confidence = bus.get("prediction_confidence", module) or {}
            scenario_generation = bus.get("scenario_generation", module) or {}
            world_model_analytics = bus.get("world_model_analytics", module) or {}

            state: Dict[str, Any] = {
                "market_predictions": market_predictions if isinstance(market_predictions, dict) else {},
                "prediction_confidence": prediction_confidence if isinstance(prediction_confidence, dict) else {},
                "scenario_generation": scenario_generation if isinstance(scenario_generation, dict) else {},
                "world_model_analytics": world_model_analytics if isinstance(world_model_analytics, dict) else {},
            }

            if isinstance(market_predictions, dict):
                state["is_trained"] = market_predictions.get("is_trained", False)
                state["model_confidence"] = market_predictions.get("model_confidence", 0.0)
                state["prediction_quality"] = market_predictions.get("prediction_quality", 0.0)
                state["stability_score"] = market_predictions.get("stability_score", 0.5)

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
        try:
            trading_mode = bus.get("trading_mode", module)
            mode_config = bus.get("mode_config", module) or {}
            mode_effectiveness = bus.get("mode_effectiveness", module)
            mode_stats = bus.get("mode_stats", module) or {}
            mode_thresholds = bus.get("mode_thresholds", module) or {}
            decision_factors = bus.get("decision_factors", module) or {}
            entry_timing = bus.get("entry_timing", module) or {}

            market_context = bus.get("market_context", module) or {}

            regime_stability = bus.get("regime_stability", module)
            theme_transition = bus.get("theme_transition", module)

            # FIX: preserve float values; do not coerce with `or {}`
            regime_accuracy_raw = bus.get("regime_accuracy", module)
            risk_scaling_factor = bus.get("risk_scaling_factor", module)
            liquidity_score = bus.get("liquidity_score", module)
            theme_strength = bus.get("theme_strength", module)

            if isinstance(regime_accuracy_raw, dict):
                regime_accuracy = float(regime_accuracy_raw.get("value", 0.5))
            else:
                try:
                    regime_accuracy = float(regime_accuracy_raw) if regime_accuracy_raw is not None else 0.5
                except (TypeError, ValueError):
                    regime_accuracy = 0.5

            state: Dict[str, Any] = {
                "trading_mode": trading_mode if isinstance(trading_mode, str) else "normal",
                "mode_config": mode_config if isinstance(mode_config, dict) else {},
                "mode_effectiveness": float(mode_effectiveness) if mode_effectiveness is not None else 0.5,
                "mode_stats": mode_stats if isinstance(mode_stats, dict) else {},
                "mode_thresholds": mode_thresholds if isinstance(mode_thresholds, dict) else {},
                "decision_factors": decision_factors if isinstance(decision_factors, dict) else {},
                "entry_timing": entry_timing if isinstance(entry_timing, dict) else {},
                "market_context": market_context if isinstance(market_context, dict) else {},
                "regime_stability": float(regime_stability) if regime_stability is not None else 0.5,
                "theme_transition": float(theme_transition) if theme_transition is not None else 0.0,
                "regime_accuracy": regime_accuracy,
                "risk_scaling_factor": float(risk_scaling_factor) if risk_scaling_factor is not None else 1.0,
                "liquidity_score": float(liquidity_score) if liquidity_score is not None else 0.5,
                "theme_strength": float(theme_strength) if theme_strength is not None else 0.0,
            }

            return state
        except Exception:
            return {}

    def _fetch_trading_mode_state_for_instrument(self, bus: Any, module: str, instrument: str) -> Dict[str, Any]:
        base_state = self._fetch_trading_mode_state(bus, module)
        if not isinstance(base_state, dict):
            base_state = {}

        try:
            entry_timing_all = bus.get("entry_timing", module) or {}
        except Exception:
            entry_timing_all = {}

        inst_timing: Dict[str, Any] = {}
        if isinstance(entry_timing_all, dict):
            raw = entry_timing_all.get(instrument)
            if not isinstance(raw, dict):
                target = self._norm_symbol(instrument)
                for key, val in entry_timing_all.items():
                    if isinstance(key, str) and isinstance(val, dict) and self._norm_symbol(key) == target:
                        raw = val
                        break
            if isinstance(raw, dict):
                inst_timing = raw

        base_state["entry_timing"] = inst_timing
        return base_state

    # ======================================================================
    # Helper Functions
    # ======================================================================

    def _extract_timeframe_data(self, market_data: Dict[str, Any], timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Extract OHLCV data for a specific timeframe with forming bar integration.

        IMPORTANT SAFETY:
        - If market_data is a multi-symbol dict, do NOT silently pick the first symbol.
          Only pick a nested symbol if there is exactly one candidate.
        """
        if not market_data:
            return None

        if timeframe in market_data and isinstance(market_data[timeframe], dict):
            return self._apply_forming_bar(market_data[timeframe])

        candidates: list[Dict[str, Any]] = []
        for sym_data in market_data.values():
            if isinstance(sym_data, dict) and timeframe in sym_data and isinstance(sym_data[timeframe], dict):
                candidates.append(sym_data[timeframe])

        if len(candidates) == 1:
            return self._apply_forming_bar(candidates[0])

        return None

    def _apply_forming_bar(self, tf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge a forming/current bar into the last bar without float-noise issues.

        IMPORTANT: Controlled by config.use_forming_bar flag.
        - False (default): Return data unchanged (training parity - closed bars only)
        - True: Merge forming bar into last candle (live-only advanced use)
        
        The previous implementation used a hard-coded absolute tolerance (0.0001),
        which is not instrument-agnostic (especially wrong for XAUUSD).
        """
        if not isinstance(tf_data, dict):
            return tf_data

        # CRITICAL: Skip forming bar processing unless explicitly enabled
        if not self.config.use_forming_bar:
            return tf_data

        cur_bar = tf_data.get("current_bar")
        if not isinstance(cur_bar, dict):
            return tf_data

        forming_close = cur_bar.get("close")
        forming_high = cur_bar.get("high")
        forming_low = cur_bar.get("low")
        forming_volume = cur_bar.get("volume")

        if forming_close is None:
            return tf_data

        result = dict(tf_data)

        try:
            close = result.get("close")
            if close is None:
                return tf_data

            close_list = list(close)
            if len(close_list) == 0:
                return tf_data

            last_close = float(close_list[-1])
            f_close = float(forming_close)

            tol = max(
                self.config.forming_bar_atol,
                abs(last_close) * self.config.forming_bar_rtol,
            )

            # Update only if meaningfully different (not float noise)
            if abs(f_close - last_close) <= tol:
                return tf_data

            close_list[-1] = f_close
            result["close"] = close_list

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
            return tf_data

        return result

    def _extract_direction(self, proposal: Any) -> float:
        if isinstance(proposal, dict):
            direction = proposal.get("direction") or proposal.get("action") or proposal.get("global_direction")
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

    def _ema_series(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple EMA series (stable, fast enough for typical OHLC windows)."""
        if data.size == 0:
            return np.asarray([], dtype=np.float64)
        alpha = 2.0 / (period + 1.0)
        ema = np.empty_like(data, dtype=np.float64)
        ema[0] = float(data[0])
        for i in range(1, data.size):
            ema[i] = alpha * float(data[i]) + (1.0 - alpha) * float(ema[i - 1])
        return ema

    def _compute_macd_histogram(self, close: np.ndarray) -> float:
        """
        True MACD histogram:
          MACD line = EMA(12) - EMA(26)
          Signal    = EMA(9) of MACD line
          Hist      = MACD line - Signal

        If you want MACD LINE instead, replace the return with: float(macd_line[-1])
        """
        if close.size < 35:
            return 0.0

        ema12 = self._ema_series(close, 12)
        ema26 = self._ema_series(close, 26)
        macd_line = ema12 - ema26
        signal = self._ema_series(macd_line, 9)
        hist = macd_line - signal
        return float(hist[-1]) if hist.size > 0 else 0.0

    def _compute_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
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
        if data.size < 2:
            return 0.0
        x = np.arange(data.size, dtype=np.float64)
        coeffs = np.polyfit(x, data.astype(np.float64), 1)
        return float(coeffs[0])

    def _compute_trend_direction(self, close: np.ndarray) -> float:
        if close.size < 5:
            return 0.0

        sma = float(np.mean(close[-20:])) if close.size >= 20 else float(np.mean(close))
        current = float(close[-1])
        diff = (current - sma) / max(abs(sma), self._eps)
        return float(np.clip(diff * 10.0, -1.0, 1.0))


_default_builder: Optional[PPOObservationBuilder] = None


def get_ppo_observation_builder() -> PPOObservationBuilder:
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
    governor_state: Optional[Dict[str, Any]] = None,  # v5.5: governor/budget observation
    smart_bus: Optional[Any] = None,
    module_name: str = "PPOObservationBuilder",
) -> np.ndarray:
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
        governor_state=governor_state,
        smart_bus=smart_bus,
        module_name=module_name,
    )
