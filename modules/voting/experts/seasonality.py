"""
Advanced SeasonalityRiskExpert (v3.3) — Base-Aligned Time/Seasonality Overlay (XAUUSD only)

Purpose
-------
This expert is NOT a standalone trading bot.
It acts as a time/seasonality "overlay" and risk gatekeeper:

- Identifies favorable/unfavorable temporal regimes:
  - sessions (Asia/Europe/US + overlaps)
  - day-of-week / hour-of-day tendencies
  - month/quarter effects (lightweight, gold-aware)
  - rollover / weekend gap risk
  - holiday / low-liquidity windows
  - optional economic calendar awareness (fail-open + retry)
- Outputs a standard long/short/flat vote primarily driven by *signal quality*.
- Provides "trading_window" metadata so your Committee/Arbiter can gate entries.

Design Principles
-----------------
- Base-aligned: uses VotingExpertBase hooks, no duplicated core pipeline logic.
- Fail-open: missing data => neutral/flat, never crashes downstream.
- XAUUSD-only: prevents cross-symbol leakage and inconsistent behavior.
- Overlay-first: prefers veto/flat in bad conditions; directional bias is modest.

Key Upgrades vs v3.2
--------------------
- TTL caching (structure, volume, trading window, high-impact window)
- Throttled adaptive updates (per new closed candle)
- Config validation/clamping (stdlib-only, no heavy deps)
- LRU-bounded adaptive patterns (prevents unbounded memory growth)
- Seasonal-specific circuit breaker (critical-error aware)
- Economic calendar retry (async, fail-open)
- Debug integration: uses base _debug_log when available, otherwise buffered JSONL
- Improved type safety (_safe_float/_safe_dict) to satisfy Pylance and runtime safety
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from datetime import timezone as dt_timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.constants import (
    CONFIDENCE_THRESHOLD_F,
    HIGH_CONFIDENCE_THRESHOLD_F,
    MIN_SIGNAL_STRENGTH_F,
)
from modules.voting.core.per_instrument import normalize_instrument
from modules.voting.experts.base import VotingExpertBase

# ─────────────────────────────────────────────────────────────
# Typed context / state
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SessionContext:
    current_session: str
    active_sessions: List[str]
    session_quality: float
    liquidity_score: float


@dataclass(frozen=True)
class TimeContext:
    utc_time: datetime
    dow: int
    hour: int
    minute: int
    month: int
    day: int
    trading_quality: float
    is_pre_news: bool
    is_news_hour: bool


@dataclass
class _EmaStats:
    """Exponential moving stats for adaptive pattern learning."""
    count: int = 0
    ema_ret: float = 0.0
    ema_abs_ret: float = 0.0
    ema_win: float = 0.5
    last_ts_iso: str = ""


class AdaptivePatternLRU:
    """LRU wrapper for adaptive EMA stats to cap memory growth."""
    def __init__(self, max_size: int) -> None:
        self.max_size = max(64, int(max_size))
        self._od: "OrderedDict[str, _EmaStats]" = OrderedDict()
        self.evictions = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[_EmaStats]:
        st = self._od.get(key)
        if st is None:
            self.misses += 1
            return None
        self.hits += 1
        self._od.move_to_end(key)
        return st

    def set(self, key: str, value: _EmaStats) -> None:
        self._od[key] = value
        self._od.move_to_end(key)
        if len(self._od) > self.max_size:
            self._od.popitem(last=False)
            self.evictions += 1

    def __len__(self) -> int:
        return len(self._od)

    def snapshot_stats(self) -> Dict[str, int]:
        return {
            "size": len(self._od),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


@dataclass(frozen=True)
class SeasonalityRiskConfig:
    # Core
    trading_timezone: str = "Europe/Berlin"
    instruments: Tuple[str, ...] = ("XAUUSD",)

    # Local trading window (local time)
    local_trade_start_hour: int = 9
    local_trade_end_hour: int = 18
    local_hard_close_hour: int = 22
    no_trade_last_minutes: int = 60

    local_prime_start_hour: int = 14
    local_prime_end_hour: int = 17
    prime_hours_confidence_boost: float = 0.15
    prime_hours_lot_multiplier: float = 1.25

    allow_off_hours_trading: bool = True

    # Confidence shaping
    base_confidence: float = 0.35
    max_confidence: float = 0.85
    min_confidence: float = 0.12
    direction_min_threshold: float = 0.22

    # Filters
    use_chop_filter: bool = True
    chop_soft_threshold: float = 58.0
    chop_veto_threshold: float = 65.0

    use_volume_confirmation: bool = True
    volume_confirm_threshold: float = 0.60
    volume_lookback: int = 60

    use_structure_confirmation: bool = True
    structure_lookback: int = 80
    structure_edge_band: float = 0.15

    # Adaptive learning
    use_adaptive_patterns: bool = True
    adaptive_decay: float = 0.995
    max_adaptive_patterns: int = 1500
    adaptive_min_new_bars: int = 1  # throttle: update after N new bars

    # High impact hour heuristic (UTC)
    high_impact_hours_utc: Tuple[int, ...] = (8, 12, 13, 14, 18)
    high_impact_minutes_pre: int = 10
    high_impact_minutes_post: int = 30

    # Holidays / thin liquidity
    holiday_month_days: Tuple[Tuple[int, int], ...] = ((1, 1), (12, 24), (12, 25), (12, 26))
    late_dec_start_day: int = 20
    early_jan_end_day: int = 3

    # Rollover / weekend windows (UTC)
    rollover_start: dt_time = dt_time(21, 0)
    rollover_end: dt_time = dt_time(22, 0)
    weekend_risk_start: dt_time = dt_time(20, 0)

    # Persistence smoothing
    min_regime_persistence: int = 3

    # Debug
    debug_enabled: bool = True
    debug_path: str = "logs/seasonality_expert_debug.jsonl"
    debug_every_n: int = 5
    debug_buffer_size: int = 50
    debug_flush_seconds: float = 2.0

    # Caching
    cache_ttl_structure_s: float = 15.0
    cache_ttl_volume_s: float = 30.0
    cache_ttl_trading_window_s: float = 20.0
    cache_ttl_high_impact_s: float = 20.0

    # Seasonal circuit breaker
    seasonal_circuit_enabled: bool = True
    seasonal_error_trip_count: int = 5
    seasonal_error_window_s: float = 60.0
    seasonal_cooloff_s: float = 120.0

    @staticmethod
    def _clamp_float(x: Any, lo: float, hi: float, default: float) -> float:
        try:
            v = float(x)
        except Exception:
            v = float(default)
        return float(max(lo, min(hi, v)))

    @staticmethod
    def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
        try:
            v = int(x)
        except Exception:
            v = int(default)
        return int(max(lo, min(hi, v)))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SeasonalityRiskConfig":
        if not isinstance(raw, dict):
            return cls()

        def gb(name: str, default: Any) -> Any:
            return raw.get(name, default)

        cfg = cls(
            trading_timezone=str(gb("trading_timezone", cls.trading_timezone)),
            instruments=tuple(gb("instruments", cls.instruments)) if isinstance(gb("instruments", None), (list, tuple)) else cls.instruments,

            local_trade_start_hour=cls._clamp_int(gb("local_trade_start_hour", cls.local_trade_start_hour), 0, 23, cls.local_trade_start_hour),
            local_trade_end_hour=cls._clamp_int(gb("local_trade_end_hour", cls.local_trade_end_hour), 0, 23, cls.local_trade_end_hour),
            local_hard_close_hour=cls._clamp_int(gb("local_hard_close_hour", cls.local_hard_close_hour), 0, 23, cls.local_hard_close_hour),
            no_trade_last_minutes=cls._clamp_int(gb("no_trade_last_minutes", cls.no_trade_last_minutes), 0, 240, cls.no_trade_last_minutes),

            local_prime_start_hour=cls._clamp_int(gb("local_prime_start_hour", cls.local_prime_start_hour), 0, 23, cls.local_prime_start_hour),
            local_prime_end_hour=cls._clamp_int(gb("local_prime_end_hour", cls.local_prime_end_hour), 0, 23, cls.local_prime_end_hour),
            prime_hours_confidence_boost=cls._clamp_float(gb("prime_hours_confidence_boost", cls.prime_hours_confidence_boost), 0.0, 1.0, cls.prime_hours_confidence_boost),
            prime_hours_lot_multiplier=cls._clamp_float(gb("prime_hours_lot_multiplier", cls.prime_hours_lot_multiplier), 0.1, 5.0, cls.prime_hours_lot_multiplier),

            allow_off_hours_trading=bool(gb("allow_off_hours_trading", cls.allow_off_hours_trading)),

            base_confidence=cls._clamp_float(gb("base_confidence", cls.base_confidence), 0.0, 1.0, cls.base_confidence),
            max_confidence=cls._clamp_float(gb("max_confidence", cls.max_confidence), 0.0, 1.0, cls.max_confidence),
            min_confidence=cls._clamp_float(gb("min_confidence", cls.min_confidence), 0.0, 1.0, cls.min_confidence),
            direction_min_threshold=cls._clamp_float(gb("direction_min_threshold", cls.direction_min_threshold), 0.0, 1.0, cls.direction_min_threshold),

            use_chop_filter=bool(gb("use_chop_filter", cls.use_chop_filter)),
            chop_soft_threshold=cls._clamp_float(gb("chop_soft_threshold", cls.chop_soft_threshold), 0.0, 100.0, cls.chop_soft_threshold),
            chop_veto_threshold=cls._clamp_float(gb("chop_veto_threshold", cls.chop_veto_threshold), 0.0, 100.0, cls.chop_veto_threshold),

            use_volume_confirmation=bool(gb("use_volume_confirmation", cls.use_volume_confirmation)),
            volume_confirm_threshold=cls._clamp_float(gb("volume_confirm_threshold", cls.volume_confirm_threshold), 0.0, 3.0, cls.volume_confirm_threshold),
            volume_lookback=cls._clamp_int(gb("volume_lookback", cls.volume_lookback), 10, 5000, cls.volume_lookback),

            use_structure_confirmation=bool(gb("use_structure_confirmation", cls.use_structure_confirmation)),
            structure_lookback=cls._clamp_int(gb("structure_lookback", cls.structure_lookback), 20, 5000, cls.structure_lookback),
            structure_edge_band=cls._clamp_float(gb("structure_edge_band", cls.structure_edge_band), 0.01, 0.49, cls.structure_edge_band),

            use_adaptive_patterns=bool(gb("use_adaptive_patterns", cls.use_adaptive_patterns)),
            adaptive_decay=cls._clamp_float(gb("adaptive_decay", cls.adaptive_decay), 0.90, 0.9999, cls.adaptive_decay),
            max_adaptive_patterns=cls._clamp_int(gb("max_adaptive_patterns", cls.max_adaptive_patterns), 100, 100000, cls.max_adaptive_patterns),
            adaptive_min_new_bars=cls._clamp_int(gb("adaptive_min_new_bars", cls.adaptive_min_new_bars), 1, 10, cls.adaptive_min_new_bars),

            high_impact_hours_utc=tuple(gb("high_impact_hours_utc", cls.high_impact_hours_utc)) if isinstance(gb("high_impact_hours_utc", None), (list, tuple)) else cls.high_impact_hours_utc,
            high_impact_minutes_pre=cls._clamp_int(gb("high_impact_minutes_pre", cls.high_impact_minutes_pre), 0, 59, cls.high_impact_minutes_pre),
            high_impact_minutes_post=cls._clamp_int(gb("high_impact_minutes_post", cls.high_impact_minutes_post), 0, 59, cls.high_impact_minutes_post),

            holiday_month_days=tuple(tuple(x) for x in gb("holiday_month_days", cls.holiday_month_days)) if isinstance(gb("holiday_month_days", None), (list, tuple)) else cls.holiday_month_days,
            late_dec_start_day=cls._clamp_int(gb("late_dec_start_day", cls.late_dec_start_day), 1, 31, cls.late_dec_start_day),
            early_jan_end_day=cls._clamp_int(gb("early_jan_end_day", cls.early_jan_end_day), 1, 15, cls.early_jan_end_day),

            min_regime_persistence=cls._clamp_int(gb("min_regime_persistence", cls.min_regime_persistence), 1, 20, cls.min_regime_persistence),

            debug_enabled=bool(gb("debug_enabled", cls.debug_enabled)),
            debug_path=str(gb("debug_path", cls.debug_path)),
            debug_every_n=cls._clamp_int(gb("debug_every_n", cls.debug_every_n), 1, 1000, cls.debug_every_n),
            debug_buffer_size=cls._clamp_int(gb("debug_buffer_size", cls.debug_buffer_size), 1, 5000, cls.debug_buffer_size),
            debug_flush_seconds=cls._clamp_float(gb("debug_flush_seconds", cls.debug_flush_seconds), 0.1, 60.0, cls.debug_flush_seconds),

            cache_ttl_structure_s=cls._clamp_float(gb("cache_ttl_structure_s", cls.cache_ttl_structure_s), 0.1, 300.0, cls.cache_ttl_structure_s),
            cache_ttl_volume_s=cls._clamp_float(gb("cache_ttl_volume_s", cls.cache_ttl_volume_s), 0.1, 300.0, cls.cache_ttl_volume_s),
            cache_ttl_trading_window_s=cls._clamp_float(gb("cache_ttl_trading_window_s", cls.cache_ttl_trading_window_s), 0.1, 300.0, cls.cache_ttl_trading_window_s),
            cache_ttl_high_impact_s=cls._clamp_float(gb("cache_ttl_high_impact_s", cls.cache_ttl_high_impact_s), 0.1, 300.0, cls.cache_ttl_high_impact_s),

            seasonal_circuit_enabled=bool(gb("seasonal_circuit_enabled", cls.seasonal_circuit_enabled)),
            seasonal_error_trip_count=cls._clamp_int(gb("seasonal_error_trip_count", cls.seasonal_error_trip_count), 1, 100, cls.seasonal_error_trip_count),
            seasonal_error_window_s=cls._clamp_float(gb("seasonal_error_window_s", cls.seasonal_error_window_s), 5.0, 3600.0, cls.seasonal_error_window_s),
            seasonal_cooloff_s=cls._clamp_float(gb("seasonal_cooloff_s", cls.seasonal_cooloff_s), 5.0, 3600.0, cls.seasonal_cooloff_s),
        )

        # Invariants
        if cfg.chop_veto_threshold <= cfg.chop_soft_threshold:
            # enforce a minimal gap
            object.__setattr__(cfg, "chop_veto_threshold", min(100.0, cfg.chop_soft_threshold + 5.0))  # type: ignore

        if cfg.max_confidence < cfg.min_confidence:
            object.__setattr__(cfg, "max_confidence", cfg.min_confidence)  # type: ignore

        return cfg


@module(**module_args("SeasonalityRiskExpert"))
class SeasonalityRiskExpert(VotingExpertBase):
    """
    Base-aligned seasonality/time overlay expert.

    Produces:
      - SeasonalityRiskExpert_voting_proposal (standard)
      - SeasonalityRiskExpert_confidence
      - seasonality_voting_proposal / seasonal_voting_proposal (legacy aliases)
      - seasonality_risk_analysis (diagnostics)
      - trading_window (arbiter gating)
    """

    # ─────────────────────────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────────────────────────

    def _expert_specific_init(self) -> None:
        self.module_name = self.__class__.__name__

        # Parse/validate config (stdlib-only)
        self.cfg = SeasonalityRiskConfig.from_dict(self.config if isinstance(self.config, dict) else {})

        # Enforce XAUUSD-only (hard safety)
        cfg_instruments = self.cfg.instruments
        if any(normalize_instrument(x) != "XAUUSD" for x in cfg_instruments):
            self.log_warning(f"[SEASONALITY] Forcing XAUUSD-only. Ignoring instruments={cfg_instruments}")
        self.instruments: List[str] = ["XAUUSD"]

        # Sessions (UTC)
        self.sessions: Dict[str, Dict[str, dt_time]] = {
            "asian": {"start": dt_time(0, 0), "end": dt_time(9, 0)},
            "european": {"start": dt_time(7, 0), "end": dt_time(16, 0)},
            "american": {"start": dt_time(13, 0), "end": dt_time(22, 0)},
            "overlap_eu_us": {"start": dt_time(13, 0), "end": dt_time(16, 0)},
            "overlap_asia_eu": {"start": dt_time(7, 0), "end": dt_time(9, 0)},
        }
        self.session_weights: Dict[str, float] = {
            "overlap_eu_us": 1.30,
            "european": 1.10,
            "american": 1.00,
            "overlap_asia_eu": 0.90,
            "asian": 0.80,
            "off_hours": 0.40,
        }

        # DOW quality factors (overlay quality shaping, not direction oracle)
        self.dow_quality = {
            0: 0.80,  # Monday: re-open, repricing
            1: 1.00,
            2: 1.05,
            3: 1.00,
            4: 0.85,  # Friday: positioning + gap risk
        }

        # Gold-aware month bias (overlay, modest)
        self.gold_month_bias = {
            1: +0.35,
            2: +0.10,
            3: -0.20,
            4: -0.15,
            5: -0.10,
            6: -0.05,
            7: +0.05,
            8: +0.25,
            9: +0.25,
            10: +0.10,
            11: +0.10,
            12: +0.20,
        }

        # Adaptive learning store (LRU bounded)
        self._adaptive = AdaptivePatternLRU(max_size=self.cfg.max_adaptive_patterns)
        self._last_seen_len: Dict[str, int] = {}

        # Persistence smoothing
        self._action_history: Deque[str] = deque(maxlen=32)
        self._action_streak: int = 0

        # Debug
        self._debug_counter = 0
        self._debug_buf: List[Dict[str, Any]] = []
        self._debug_last_flush_s = self._now_s()

        # Internal TTL cache fallback: key -> (expires_at, value)
        self._ttl_cache: Dict[str, Tuple[float, Any]] = {}

        # Seasonal circuit breaker state
        self._seasonal_err_times: Deque[float] = deque()
        self._seasonal_circuit_until_s: float = 0.0

        self.log_info(
            f"[SEASONALITY] init v3.3 | instruments={self.instruments} | tz={self.cfg.trading_timezone} "
            f"| window={self.cfg.local_trade_start_hour}:00-{self.cfg.local_trade_end_hour}:00 "
            f"| prime={self.cfg.local_prime_start_hour}:00-{self.cfg.local_prime_end_hour}:00 "
            f"| allow_off_hours={self.cfg.allow_off_hours_trading} | debug={self.cfg.debug_enabled} "
            f"| adaptive_lru={self.cfg.max_adaptive_patterns}"
        )

        # Publish safe baseline keys (prevents stale consumers)
        self._publish_seasonality_baseline()

    # ─────────────────────────────────────────────────────────────
    # Base hooks
    # ─────────────────────────────────────────────────────────────

    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Base-aligned: produce proposal dict.
        Includes per-instrument proposals under 'proposals' even though we enforce XAUUSD-only.
        """

        # Seasonal circuit breaker (local)
        if self.cfg.seasonal_circuit_enabled and self._seasonal_circuit_open():
            return self._neutral_proposal("seasonal_circuit_open")

        now_utc = datetime.now(dt_timezone.utc)

        try:
            trading_window = self._analyze_trading_window_cached(now_utc)

            # Fail-open data retrieval: use passed market_data, fallback to bus if empty
            if not isinstance(market_data, dict) or not market_data:
                market_data = self._safe_bus_get("market_data", default={})
            features = self._safe_bus_get("features", default={})

            # Global time context
            session_ctx = self._analyze_session(now_utc)
            time_ctx = self._analyze_time_context(now_utc)

            rollover_risk = self._check_rollover_risk(now_utc)
            weekend_risk = self._check_weekend_risk(now_utc)
            holiday_risk, holiday_reason = self._check_holiday_risk(now_utc)

            high_impact_risk, high_impact_reason = await self._check_high_impact_window_async(now_utc)

            proposals: Dict[str, Dict[str, Any]] = {}
            per_inst_analysis: Dict[str, Any] = {}

            for inst in self.instruments:
                inst_norm = normalize_instrument(inst)
                inst_block = self._extract_instrument_block(market_data, inst_norm)

                closes = self._extract_series(inst_block, "close")
                highs = self._extract_series(inst_block, "high")
                lows = self._extract_series(inst_block, "low")
                vols = self._extract_series(inst_block, "volume")
                if vols.size == 0:
                    vols = self._extract_series(inst_block, "tick_volume")

                # Adaptive update (throttled to new closed candle)
                if self.cfg.use_adaptive_patterns:
                    self._update_adaptive_throttled(inst_norm, now_utc, closes, session_ctx.current_session)

                # Filter signals
                chop_val = self._get_chop_value(inst_norm, features)
                volume_ratio, volume_confirmed = self._volume_confirmation_cached(vols)
                struct = self._structure_alignment_cached(highs, lows, closes)

                # Direction bias (modest overlay)
                month_bias = float(self.gold_month_bias.get(now_utc.month, 0.0))
                adaptive_bias, adaptive_rel, adaptive_detail = self._adaptive_bias(inst_norm, now_utc, session_ctx)

                # Quality components
                dow_q = float(self.dow_quality.get(now_utc.weekday(), 1.0))
                hour_q = float(time_ctx.trading_quality)
                session_q = float(session_ctx.session_quality)

                # Hard vetoes for entries
                veto_reasons: List[str] = []

                if not self.cfg.allow_off_hours_trading and trading_window.get("no_new_trades", False):
                    veto_reasons.append("outside_primary_window")

                if rollover_risk:
                    veto_reasons.append("rollover_risk")

                if weekend_risk:
                    veto_reasons.append("weekend_gap_risk")

                if holiday_risk:
                    veto_reasons.append(f"holiday_low_liquidity:{holiday_reason}")

                if high_impact_risk:
                    veto_reasons.append(f"high_impact_window:{high_impact_reason}")

                # CHOP veto
                chop_veto = False
                if self.cfg.use_chop_filter and chop_val is not None:
                    chop_veto = float(chop_val) >= self.cfg.chop_veto_threshold
                    if chop_veto:
                        veto_reasons.append(f"chop_veto:{float(chop_val):.1f}")

                # Decide action/confidence
                if veto_reasons:
                    action = "flat"
                    confidence = min(0.80, max(0.65, self.cfg.base_confidence + 0.25))
                    thesis = f"{inst_norm}: VETO -> flat ({', '.join(veto_reasons)})"
                    direction_score = 0.0
                    quality_score = 0.0
                else:
                    # Direction score: month bias + adaptive bias, lightly structure-aware
                    direction_score = float(np.clip(month_bias + 0.80 * adaptive_bias, -1.0, 1.0))
                    direction_strength = abs(direction_score)

                    # Volume multiplier
                    vol_mult = 1.0
                    if self.cfg.use_volume_confirmation:
                        vol_mult = 1.0 if volume_confirmed else 0.60

                    # Quality score: session/hour/DOW * volume
                    quality_score = float(np.clip(session_q * hour_q * dow_q * vol_mult, 0.0, 1.25))
                    quality_score = float(np.clip(quality_score / 1.25, 0.0, 1.0))

                    # Structure multiplier: edge bonus => slightly better-defined risk
                    struct_mult = 1.0
                    if self.cfg.use_structure_confirmation and bool(struct.get("available", False)):
                        edge_bonus = self._safe_float(struct.get("edge_bonus"), 0.0)
                        struct_mult = 0.85 + 0.30 * edge_bonus  # 0.85..1.15

                    # Soft CHOP penalty (even if not veto)
                    chop_mult = 1.0
                    if self.cfg.use_chop_filter and chop_val is not None:
                        c = float(chop_val)
                        if c >= self.cfg.chop_soft_threshold:
                            span = max(1e-6, self.cfg.chop_veto_threshold - self.cfg.chop_soft_threshold)
                            t = float(np.clip((c - self.cfg.chop_soft_threshold) / span, 0.0, 1.0))
                            chop_mult = 1.0 - 0.35 * t

                    # Confidence composition (overlay, modest)
                    confidence = (
                        self.cfg.base_confidence
                        + 0.35 * quality_score
                        + 0.25 * direction_strength
                        + float(trading_window.get("prime_hours_confidence_boost", 0.0))
                    )
                    confidence *= struct_mult
                    confidence *= chop_mult
                    confidence = float(np.clip(confidence, self.cfg.min_confidence, self.cfg.max_confidence))

                    if direction_strength < self.cfg.direction_min_threshold:
                        action = "flat"
                        thesis = (
                            f"{inst_norm}: Neutral overlay "
                            f"(dir={direction_score:+.2f}, qual={quality_score:.2f}, rel={adaptive_rel:.2f})"
                        )
                    else:
                        action = "long" if direction_score > 0 else "short"
                        thesis = (
                            f"{inst_norm}: Seasonal overlay {action} "
                            f"(month_bias={month_bias:+.2f}, adapt={adaptive_bias:+.2f} rel={adaptive_rel:.2f}, "
                            f"qual={quality_score:.2f}, vol_ratio={volume_ratio:.2f})"
                        )

                # Magnitude shaping (overlay = modest)
                min_strength = MIN_SIGNAL_STRENGTH_F()
                if action == "flat":
                    magnitude = 0.0
                else:
                    magnitude = float(max(min_strength * 0.40, min(1.0, confidence * 0.60)))

                meta = {
                    "time_utc": now_utc.isoformat(),
                    "session": session_ctx.current_session,
                    "dow": time_ctx.dow,
                    "month": time_ctx.month,
                    "rollover_risk": rollover_risk,
                    "weekend_risk": weekend_risk,
                    "holiday_risk": holiday_risk,
                    "high_impact_risk": high_impact_risk,
                    "chop": float(chop_val) if chop_val is not None else None,
                    "volume_ratio": float(volume_ratio),
                    "volume_confirmed": bool(volume_confirmed),
                    "structure": struct,
                    "adaptive": adaptive_detail,
                    "trading_window": trading_window,
                    "direction_score": float(direction_score),
                    "quality_score": float(quality_score),
                    "cache_stats": {
                        "adaptive_lru": self._adaptive.snapshot_stats(),
                    },
                }

                proposals[inst_norm] = {
                    "action": action,
                    "confidence": confidence,
                    "magnitude": magnitude,
                    "rationale": thesis,
                    "meta": meta,
                }
                per_inst_analysis[inst_norm] = meta

            # Select global/top action (single inst anyway)
            if proposals:
                inst0 = next(iter(proposals.keys()))
                best = self._safe_dict(proposals.get(inst0))
                global_action = str(best.get("action", "flat"))
                global_conf = self._safe_float(best.get("confidence"), 0.10)
                global_thesis = str(best.get("rationale", ""))
            else:
                global_action = "flat"
                global_conf = 0.10
                global_thesis = "No proposals available"

            # Persistence smoothing (avoid rapid flip-flops)
            global_action, global_conf = self._apply_persistence_filter(global_action, global_conf)

            # Clip global confidence with shared thresholds
            conf_floor = CONFIDENCE_THRESHOLD_F()
            high_conf = HIGH_CONFIDENCE_THRESHOLD_F()
            global_conf = float(max(conf_floor * 0.50, min(high_conf, global_conf)))

            proposal = {
                "action": global_action,
                "signal_strength": global_conf,
                "confidence": global_conf,
                "reason": global_thesis,
                "proposals": proposals,
                "trading_window": trading_window,
                "analysis": {
                    "session": session_ctx.current_session,
                    "dow_bias": self._dow_bias_label(time_ctx.dow),
                    "monthly_pattern": self._month_pattern_label(time_ctx.month),
                    "rollover_risk": rollover_risk,
                    "weekend_risk": weekend_risk,
                    "holiday_risk": holiday_risk,
                    "high_impact_risk": high_impact_risk,
                    "per_instrument": per_inst_analysis,
                },
            }

            # Debug emit (buffered)
            self._debug_counter += 1
            if self.cfg.debug_enabled and (self._debug_counter % max(1, self.cfg.debug_every_n) == 0):
                self._debug_emit(event="seasonality_analysis", payload={
                    "ts_utc": now_utc.isoformat(),
                    "proposal": proposal,
                })

            return proposal

        except Exception as e:
            # Seasonal circuit breaker recording
            self._record_seasonal_error(e)
            self._debug_emit(event="seasonality_exception", payload={
                "ts_utc": now_utc.isoformat(),
                "error": str(e),
            })
            return self._neutral_proposal(f"exception:{type(e).__name__}")

    async def _calculate_expert_specific_confidence(self, proposal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Base hook: confidence comes from proposal payload."""
        return self._safe_float(proposal.get("signal_strength", proposal.get("confidence", 0.10)), 0.10)

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Minimal wrapper:
        - delegates to VotingExpertBase for the core pipeline
        - adds legacy aliases + analysis keys for compatibility (and type-safe Pylance behavior)
        """
        out = await super().process(**inputs)
        name = self.__class__.__name__

        raw_proposal = out.get(f"{name}_voting_proposal") or out.get("voting_proposal") or out.get("proposal")
        proposal = self._safe_dict(raw_proposal)

        if not proposal:
            neutral = self._neutral_output("missing_proposal_from_base")
            proposal = self._safe_dict(neutral.get(f"{name}_voting_proposal") or neutral.get("seasonality_voting_proposal"))
            out.update(neutral)

        confidence_raw = out.get(f"{name}_confidence")
        if confidence_raw is None:
            confidence_raw = out.get("confidence")
        if confidence_raw is None:
            confidence_raw = proposal.get("signal_strength", proposal.get("confidence", 0.10))

        confidence: float = self._safe_float(confidence_raw, 0.10)

        # Legacy aliases
        out.setdefault(f"{name}_voting_proposal", proposal)
        out.setdefault(f"{name}_confidence", float(confidence))

        out["seasonality_voting_proposal"] = proposal
        out["seasonality_confidence"] = float(confidence)
        out["seasonal_voting_proposal"] = proposal
        out["seasonal_confidence"] = float(confidence)

        analysis = proposal.get("analysis")
        analysis_dict = analysis if isinstance(analysis, dict) else {}

        out["seasonality_risk_analysis"] = {
            "action": proposal.get("action", "flat"),
            "confidence": float(confidence),
            "session": analysis_dict.get("session", "unknown"),
            "dow_bias": analysis_dict.get("dow_bias", "unknown"),
            "monthly_pattern": analysis_dict.get("monthly_pattern", "unknown"),
            "rollover_risk": bool(analysis_dict.get("rollover_risk", False)),
            "weekend_risk": bool(analysis_dict.get("weekend_risk", False)),
            "holiday_risk": bool(analysis_dict.get("holiday_risk", False)),
            "high_impact_risk": bool(analysis_dict.get("high_impact_risk", False)),
            "per_instrument": analysis_dict.get("per_instrument", {}),
            "trading_window": proposal.get("trading_window", {}),
        }

        # Required contract keys (ModuleRegistry + Orchestrator enforced)
        out["seasonal_session"] = str(out["seasonality_risk_analysis"].get("session", "unknown"))
        out["seasonal_dow_bias"] = str(out["seasonality_risk_analysis"].get("dow_bias", "unknown"))
        out["seasonal_monthly_pattern"] = str(out["seasonality_risk_analysis"].get("monthly_pattern", "unknown"))
        out["seasonal_composite_score"] = float(confidence)
        out["seasonal_rollover_risk"] = bool(out["seasonality_risk_analysis"].get("rollover_risk", False))
        out["seasonal_weekend_risk"] = bool(out["seasonality_risk_analysis"].get("weekend_risk", False))
        out["seasonality_analysis"] = out["seasonality_risk_analysis"]
        out["seasonality_expert_analysis"] = analysis_dict
        out["seasonality_expert_thesis"] = str(out.get("_thesis") or out.get("thesis") or "")

        # Publish legacy keys (best-effort)
        try:
            self.smart_bus.set("seasonality_voting_proposal", proposal, module=name, thesis=str(proposal.get("reason", "")))  # type: ignore
            self.smart_bus.set("seasonality_confidence", float(confidence), module=name, thesis="Seasonality confidence")  # type: ignore
            self.smart_bus.set("seasonal_voting_proposal", proposal, module=name, thesis=str(proposal.get("reason", "")))  # type: ignore
            self.smart_bus.set("seasonal_confidence", float(confidence), module=name, thesis="Seasonal confidence")  # type: ignore
            self.smart_bus.set("seasonality_risk_analysis", out["seasonality_risk_analysis"], module=name, thesis="Seasonality risk analysis")  # type: ignore
        except Exception:
            pass

        return out

    # ─────────────────────────────────────────────────────────────
    # Baseline / neutral
    # ─────────────────────────────────────────────────────────────

    def _publish_seasonality_baseline(self) -> None:
        thesis = "Seasonality baseline"
        confidence = 0.10
        now_utc = datetime.now(dt_timezone.utc)
        trading_window = self._analyze_trading_window_cached(now_utc)
        proposal = {
            "action": "flat",
            "signal_strength": confidence,
            "confidence": confidence,
            "reason": thesis,
            "proposals": {
                "XAUUSD": {
                    "action": "flat",
                    "confidence": confidence,
                    "magnitude": 0.0,
                    "rationale": thesis,
                    "meta": {"time_utc": now_utc.isoformat(), "trading_window": trading_window},
                }
            },
            "trading_window": trading_window,
            "analysis": {
                "session": "unknown",
                "dow_bias": "unknown",
                "monthly_pattern": "unknown",
                "rollover_risk": False,
                "weekend_risk": False,
                "holiday_risk": False,
                "high_impact_risk": False,
                "per_instrument": {},
            },
        }
        try:
            self.smart_bus.set("seasonality_voting_proposal", proposal, module=self.module_name, thesis=thesis)  # type: ignore
            self.smart_bus.set("seasonality_confidence", confidence, module=self.module_name, thesis="Seasonality baseline confidence")  # type: ignore
            self.smart_bus.set("seasonal_voting_proposal", proposal, module=self.module_name, thesis=thesis)  # type: ignore
            self.smart_bus.set("seasonal_confidence", confidence, module=self.module_name, thesis="Seasonal baseline confidence")  # type: ignore
            self.smart_bus.set("seasonality_risk_analysis", {
                "session": "unknown",
                "dow_bias": "unknown",
                "monthly_pattern": "unknown",
                "rollover_risk": False,
                "weekend_risk": False,
                "holiday_risk": False,
                "high_impact_risk": False,
                "action": "flat",
                "confidence": confidence,
                "per_instrument": {"XAUUSD": {"action": "flat", "confidence": confidence}},
                "trading_window": trading_window,
            }, module=self.module_name, thesis="Seasonality baseline analysis")  # type: ignore
        except Exception:
            pass

    def _neutral_proposal(self, reason: str) -> Dict[str, Any]:
        """Neutral proposal dict for base hook path."""
        thesis = f"Seasonal flat: {reason}"
        conf_floor = CONFIDENCE_THRESHOLD_F()
        confidence = float(max(0.10, conf_floor * 0.50))
        now_utc = datetime.now(dt_timezone.utc)
        trading_window = self._analyze_trading_window_cached(now_utc)
        per_inst_votes = {
            "XAUUSD": {
                "instrument": "XAUUSD",
                "action": "flat",
                "confidence": confidence,
                "signal_strength": 0.0,
                "magnitude": 0.0,
                "rationale": thesis,
            }
        }
        return {
            "action": "flat",
            "signal_strength": confidence,
            "confidence": confidence,
            "reason": thesis,
            "proposals": per_inst_votes,
            "trading_window": trading_window,
            "analysis": {
                "session": "unknown",
                "dow_bias": "unknown",
                "monthly_pattern": "unknown",
                "rollover_risk": False,
                "weekend_risk": False,
                "holiday_risk": False,
                "high_impact_risk": False,
                "per_instrument": {"XAUUSD": {"action": "flat", "confidence": confidence, "reason": thesis}},
            },
        }

    def _neutral_output(self, reason: str) -> Dict[str, Any]:
        thesis = f"Seasonal flat: {reason}"
        conf_floor = CONFIDENCE_THRESHOLD_F()
        confidence = float(max(0.10, conf_floor * 0.50))
        now_utc = datetime.now(dt_timezone.utc)
        trading_window = self._analyze_trading_window_cached(now_utc)
        proposal = self._neutral_proposal(reason)
        return {
            "SeasonalityRiskExpert_voting_proposal": proposal,
            "SeasonalityRiskExpert_confidence": confidence,
            "seasonality_voting_proposal": proposal,
            "seasonality_confidence": confidence,
            "seasonal_voting_proposal": proposal,
            "seasonal_confidence": confidence,
            "seasonality_risk_analysis": {
                "session": "unknown",
                "dow_bias": "unknown",
                "monthly_pattern": "unknown",
                "rollover_risk": False,
                "weekend_risk": False,
                "holiday_risk": False,
                "high_impact_risk": False,
                "action": "flat",
                "confidence": confidence,
                "per_instrument": {"XAUUSD": {"action": "flat", "confidence": confidence}},
                "trading_window": trading_window,
            },
            "_thesis": thesis,
        }

    # ─────────────────────────────────────────────────────────────
    # Time/session/window helpers
    # ─────────────────────────────────────────────────────────────

    def _time_in_range(self, current: dt_time, start: dt_time, end: dt_time) -> bool:
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    def _analyze_session(self, now_utc: datetime) -> SessionContext:
        current_t = now_utc.time()
        active: List[str] = []
        for sname, times in self.sessions.items():
            if self._time_in_range(current_t, times["start"], times["end"]):
                active.append(sname)

        if "overlap_eu_us" in active:
            primary, quality = "overlap_eu_us", 1.00
        elif "overlap_asia_eu" in active:
            primary, quality = "overlap_asia_eu", 0.85
        elif "european" in active:
            primary, quality = "european", 0.90
        elif "american" in active:
            primary, quality = "american", 0.85
        elif "asian" in active:
            primary, quality = "asian", 0.70
        else:
            primary, quality = "off_hours", 0.40

        liquidity = float(self.session_weights.get(primary, 0.50))
        return SessionContext(
            current_session=primary,
            active_sessions=active,
            session_quality=float(quality),
            liquidity_score=liquidity,
        )

    def _analyze_time_context(self, now_utc: datetime) -> TimeContext:
        hour = now_utc.hour
        minute = now_utc.minute
        dow = now_utc.weekday()
        month = now_utc.month
        day = now_utc.day

        # Quality bands (UTC)
        if 13 <= hour <= 16:
            trading_quality = 0.95
        elif 7 <= hour <= 11:
            trading_quality = 0.90
        elif 0 <= hour <= 6:
            trading_quality = 0.65
        else:
            trading_quality = 0.75

        is_pre_news = hour in self.cfg.high_impact_hours_utc and minute >= (60 - self.cfg.high_impact_minutes_pre)
        is_news_hour = hour in self.cfg.high_impact_hours_utc and minute <= self.cfg.high_impact_minutes_post

        return TimeContext(
            utc_time=now_utc,
            dow=dow,
            hour=hour,
            minute=minute,
            month=month,
            day=day,
            trading_quality=float(trading_quality),
            is_pre_news=bool(is_pre_news),
            is_news_hour=bool(is_news_hour),
        )

    def _check_rollover_risk(self, now_utc: datetime) -> bool:
        t = now_utc.time()
        return self._time_in_range(t, self.cfg.rollover_start, self.cfg.rollover_end)

    def _check_weekend_risk(self, now_utc: datetime) -> bool:
        if now_utc.weekday() == 4:
            return now_utc.time() >= self.cfg.weekend_risk_start
        return False

    def _check_holiday_risk(self, now_utc: datetime) -> Tuple[bool, str]:
        m, d = now_utc.month, now_utc.day
        if (m, d) in set(self.cfg.holiday_month_days):
            return True, "fixed_holiday"
        if m == 12 and d >= self.cfg.late_dec_start_day:
            return True, "late_dec_thin_liquidity"
        if m == 1 and d <= self.cfg.early_jan_end_day:
            return True, "early_jan_thin_liquidity"
        if m == 8:
            return True, "august_thin_liquidity"
        return False, ""

    async def _check_high_impact_window_async(self, now_utc: datetime) -> Tuple[bool, str]:
        # Cached by minute bucket
        minute_bucket = int(now_utc.timestamp() // 60)
        cache_key = f"hiwin:{minute_bucket}"
        cached = self._cache_get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            return bool(cached[0]), str(cached[1])

        # 1) Try economic events with retry (fail-open)
        events = await self._get_economic_calendar_with_retry(max_retries=2)
        if isinstance(events, list) and events:
            for e in events:
                if not isinstance(e, dict):
                    continue
                impact = str(e.get("impact", "")).lower()
                if impact not in ("high", "red"):
                    continue
                ts = e.get("time") or e.get("timestamp") or e.get("ts")
                dt = self._parse_event_time_utc(ts)
                if dt is None:
                    continue
                delta_min = abs((dt - now_utc).total_seconds()) / 60.0
                if delta_min <= max(self.cfg.high_impact_minutes_pre, self.cfg.high_impact_minutes_post):
                    res = (True, "economic_calendar_high_impact")
                    self._cache_set(cache_key, res, ttl_s=self.cfg.cache_ttl_high_impact_s)
                    return res

        # 2) Fallback heuristic window
        h = now_utc.hour
        m = now_utc.minute
        if h in self.cfg.high_impact_hours_utc and (m <= self.cfg.high_impact_minutes_post or m >= (60 - self.cfg.high_impact_minutes_pre)):
            res = (True, "heuristic_high_impact_hour")
            self._cache_set(cache_key, res, ttl_s=self.cfg.cache_ttl_high_impact_s)
            return res

        res = (False, "")
        self._cache_set(cache_key, res, ttl_s=self.cfg.cache_ttl_high_impact_s)
        return res

    async def _get_economic_calendar_with_retry(self, max_retries: int = 2) -> Optional[List[Dict[str, Any]]]:
        for attempt in range(max_retries + 1):
            try:
                events = self._safe_bus_get("economic_calendar_events", default=None)
                if isinstance(events, list):
                    return [e for e in events if isinstance(e, dict)]
            except Exception as e:
                self._debug_emit(event="economic_calendar_error", payload={
                    "attempt": attempt,
                    "error": str(e),
                })
            # small exponential backoff
            if attempt < max_retries:
                await asyncio.sleep(0.05 * (2 ** attempt))
        return None

    def _parse_event_time_utc(self, ts: Any) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            try:
                return datetime.fromtimestamp(float(ts), tz=dt_timezone.utc)
            except Exception:
                return None
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=dt_timezone.utc)
                return dt.astimezone(dt_timezone.utc)
            except Exception:
                return None
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=dt_timezone.utc)
            return ts.astimezone(dt_timezone.utc)
        return None

    def _analyze_trading_window(self, now_utc: datetime) -> Dict[str, Any]:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(self.cfg.trading_timezone)
            local_dt = now_utc.astimezone(tz)
            tz_name = self.cfg.trading_timezone
        except Exception:
            local_dt = now_utc
            tz_name = "UTC"

        lh, lm = local_dt.hour, local_dt.minute
        local_minutes = lh * 60 + lm

        start_m = self.cfg.local_trade_start_hour * 60
        end_m = self.cfg.local_trade_end_hour * 60
        hard_close_m = self.cfg.local_hard_close_hour * 60

        in_primary = start_m <= local_minutes < end_m
        after_cutoff = local_minutes >= end_m

        prime_start = self.cfg.local_prime_start_hour * 60
        prime_end = self.cfg.local_prime_end_hour * 60
        in_prime = prime_start <= local_minutes < prime_end

        minutes_to_close = max(0, hard_close_m - local_minutes)
        final_exit_window = 0 <= minutes_to_close <= self.cfg.no_trade_last_minutes

        if self.cfg.allow_off_hours_trading:
            no_new_trades = False
        else:
            no_new_trades = after_cutoff or final_exit_window or not in_primary

        return {
            "timezone": tz_name,
            "local_time": local_dt.isoformat(),
            "local_hour": lh,
            "local_minute": lm,
            "in_primary_window": in_primary,
            "in_prime_window": in_prime,
            "after_cutoff": after_cutoff,
            "minutes_to_close": minutes_to_close,
            "final_exit_window": final_exit_window,
            "no_new_trades": no_new_trades,
            "allow_off_hours_override": self.cfg.allow_off_hours_trading,
            "prime_hours_confidence_boost": self.cfg.prime_hours_confidence_boost if in_prime else 0.0,
            "prime_hours_lot_multiplier": self.cfg.prime_hours_lot_multiplier if in_prime else 1.0,
        }

    def _analyze_trading_window_cached(self, now_utc: datetime) -> Dict[str, Any]:
        # minute bucket caching
        minute_bucket = int(now_utc.timestamp() // 60)
        k = f"tw:{minute_bucket}"
        cached = self._cache_get(k)
        if isinstance(cached, dict):
            return cached
        tw = self._analyze_trading_window(now_utc)
        self._cache_set(k, tw, ttl_s=self.cfg.cache_ttl_trading_window_s)
        return tw

    # ─────────────────────────────────────────────────────────────
    # Data extraction / feature guards
    # ─────────────────────────────────────────────────────────────

    def _extract_instrument_block(self, market_data: Dict[str, Any], inst_norm: str) -> Dict[str, Any]:
        if not isinstance(market_data, dict):
            return {}
        candidates = [inst_norm, inst_norm.upper(), inst_norm.lower(), "XAU_USD", "XAUUSD", "GOLDUSD", "GOLD"]
        for k in candidates:
            if k in market_data and isinstance(market_data[k], dict):
                return market_data[k]
        return market_data

    def _extract_series(self, inst_block: Dict[str, Any], key: str) -> np.ndarray:
        if not isinstance(inst_block, dict):
            return np.array([], dtype=float)

        if key in inst_block and isinstance(inst_block[key], (list, np.ndarray)):
            return np.array(inst_block[key], dtype=float)

        for tf in ("M15", "H1", "H4", "D1"):
            if tf in inst_block and isinstance(inst_block[tf], dict):
                v = inst_block[tf].get(key)
                if isinstance(v, (list, np.ndarray)):
                    return np.array(v, dtype=float)

        return np.array([], dtype=float)

    def _get_chop_value(self, inst_norm: str, features: Dict[str, Any]) -> Optional[float]:
        # 1) features dict
        try:
            if isinstance(features, dict):
                blk = features.get(inst_norm) if isinstance(features.get(inst_norm), dict) else features
                for k in ("chop", "CHOP", "choppiness", "choppiness_index", "market_chop"):
                    if isinstance(blk, dict) and k in blk:
                        return float(blk[k])
        except Exception:
            pass

        # 2) direct bus keys
        for k in ("chop", "CHOP", "market_chop", "theme_chop", "ThemeExpert_chop"):
            try:
                v = self._safe_bus_get(k, default=None)
                if v is not None:
                    return float(v)
            except Exception:
                continue
        return None

    # ─────────────────────────────────────────────────────────────
    # Cached calculations
    # ─────────────────────────────────────────────────────────────

    def _volume_confirmation_cached(self, vols: np.ndarray) -> Tuple[float, bool]:
        if vols is None or vols.size < 10:
            return 1.0, True  # fail-open

        last = float(vols[-1])
        n = int(vols.size)
        k = f"vol:{n}:{int(last)}"
        cached = self._cache_get(k)
        if isinstance(cached, tuple) and len(cached) == 2:
            return float(cached[0]), bool(cached[1])

        lookback = min(int(self.cfg.volume_lookback), int(vols.size))
        window = vols[-lookback:]
        med = float(np.median(window)) if window.size else 0.0
        if med <= 0.0:
            res = (1.0, True)
            self._cache_set(k, res, ttl_s=self.cfg.cache_ttl_volume_s)
            return res

        ratio = float(np.clip(last / med, 0.0, 3.0))
        confirmed = ratio >= self.cfg.volume_confirm_threshold
        res = (ratio, confirmed)
        self._cache_set(k, res, ttl_s=self.cfg.cache_ttl_volume_s)
        return res

    def _structure_alignment_cached(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict[str, Any]:
        if closes is None or closes.size < 20 or highs.size < 20 or lows.size < 20:
            return {"available": False}

        n = int(closes.size)
        px = float(closes[-1])
        k = f"struct:{n}:{int(px)}"
        cached = self._cache_get(k)
        if isinstance(cached, dict):
            return cached

        lb = min(int(self.cfg.structure_lookback), int(closes.size), int(highs.size), int(lows.size))
        hh = float(np.max(highs[-lb:]))
        ll = float(np.min(lows[-lb:]))
        rng = max(1e-9, hh - ll)
        pos = float(np.clip((px - ll) / rng, 0.0, 1.0))

        near_low = pos <= self.cfg.structure_edge_band
        near_high = pos >= (1.0 - self.cfg.structure_edge_band)

        edge_bonus = 0.0
        if near_low or near_high:
            edge_bonus = 1.0
        else:
            dist_to_edge = min(abs(pos - 0.0), abs(pos - 1.0))
            edge_bonus = float(np.clip((self.cfg.structure_edge_band - dist_to_edge) / max(1e-6, self.cfg.structure_edge_band), 0.0, 1.0))

        out = {
            "available": True,
            "range_high": hh,
            "range_low": ll,
            "range_pos": pos,
            "near_support_zone": bool(near_low),
            "near_resistance_zone": bool(near_high),
            "edge_bonus": float(edge_bonus),
        }
        self._cache_set(k, out, ttl_s=self.cfg.cache_ttl_structure_s)
        return out

    # ─────────────────────────────────────────────────────────────
    # Adaptive patterns (EMA) — LRU bounded & throttled
    # ─────────────────────────────────────────────────────────────

    def _should_update_adaptive(self, inst_norm: str, closes_len: int) -> bool:
        prev_len = int(self._last_seen_len.get(inst_norm, 0))
        if closes_len <= prev_len:
            return False
        return (closes_len - prev_len) >= int(self.cfg.adaptive_min_new_bars)

    def _update_adaptive_throttled(self, inst_norm: str, now_utc: datetime, closes: np.ndarray, session: str) -> None:
        if closes is None or closes.size < 3:
            return
        n = int(closes.size)
        if not self._should_update_adaptive(inst_norm, n):
            return

        try:
            c1 = float(closes[-2])
            c2 = float(closes[-1])
            if c1 <= 0 or c2 <= 0:
                self._last_seen_len[inst_norm] = n
                return
            ret = float(np.log(c2 / c1))
        except Exception:
            self._last_seen_len[inst_norm] = n
            return

        dow = now_utc.weekday()
        hour = now_utc.hour
        month = now_utc.month

        keys = [
            f"{inst_norm}|dow:{dow}",
            f"{inst_norm}|hour:{hour}",
            f"{inst_norm}|month:{month}",
            f"{inst_norm}|session:{session}",
        ]

        win = 1.0 if ret > 0 else 0.0
        a = float(self.cfg.adaptive_decay)

        for k in keys:
            st = self._adaptive.get(k) or _EmaStats()
            st.count += 1
            st.ema_ret = a * st.ema_ret + (1.0 - a) * ret
            st.ema_abs_ret = a * st.ema_abs_ret + (1.0 - a) * abs(ret)
            st.ema_win = a * st.ema_win + (1.0 - a) * win
            st.last_ts_iso = now_utc.isoformat()
            self._adaptive.set(k, st)

        self._last_seen_len[inst_norm] = n

    def _adaptive_bias(self, inst_norm: str, now_utc: datetime, session_ctx: SessionContext) -> Tuple[float, float, Dict[str, Any]]:
        session = session_ctx.current_session
        dow = now_utc.weekday()
        hour = now_utc.hour
        month = now_utc.month

        keys = [
            f"{inst_norm}|dow:{dow}",
            f"{inst_norm}|hour:{hour}",
            f"{inst_norm}|month:{month}",
            f"{inst_norm}|session:{session}",
        ]

        scores: List[float] = []
        rels: List[float] = []
        detail: Dict[str, Any] = {}

        for k in keys:
            st = self._adaptive.get(k)
            if st is None or st.count < 3:
                continue

            denom = max(1e-9, float(st.ema_abs_ret))
            raw = float(st.ema_ret / denom)
            win_adj = float((st.ema_win - 0.5) * 2.0)  # -1..+1
            score = float(np.tanh(raw * 1.5) * win_adj)
            reliability = float(np.clip(st.count / 50.0, 0.0, 1.0))

            scores.append(score * reliability)
            rels.append(reliability)

            detail[k] = {
                "count": st.count,
                "ema_ret": float(st.ema_ret),
                "ema_abs_ret": float(st.ema_abs_ret),
                "ema_win": float(st.ema_win),
                "score": float(score),
                "reliability": float(reliability),
                "last_ts": st.last_ts_iso,
            }

        if not scores:
            return 0.0, 0.0, {"available": False}

        bias = float(np.clip(float(np.mean(scores)), -1.0, 1.0))
        rel = float(np.clip(float(np.mean(rels)), 0.0, 1.0))
        return bias, rel, {"available": True, "bias": bias, "reliability": rel, "detail": detail}

    # ─────────────────────────────────────────────────────────────
    # Persistence filter
    # ─────────────────────────────────────────────────────────────

    def _apply_persistence_filter(self, action: str, confidence: float) -> Tuple[str, float]:
        if self._action_history and self._action_history[-1] == action:
            self._action_streak += 1
            if self._action_streak >= self.cfg.min_regime_persistence:
                confidence = min(self.cfg.max_confidence, confidence + min(0.08, 0.02 * self._action_streak))
        else:
            self._action_streak = 1
            confidence = max(self.cfg.min_confidence, confidence * 0.85)

        self._action_history.append(action)
        return action, float(confidence)

    # ─────────────────────────────────────────────────────────────
    # Seasonal circuit breaker
    # ─────────────────────────────────────────────────────────────

    def _seasonal_circuit_open(self) -> bool:
        now = self._now_s()
        return now < float(self._seasonal_circuit_until_s)

    def _record_seasonal_error(self, error: Exception) -> None:
        if not self.cfg.seasonal_circuit_enabled:
            return

        # Ignore transient errors
        if not self._is_seasonal_error_critical(error):
            return

        now = self._now_s()
        self._seasonal_err_times.append(now)

        # Drop old
        window = float(self.cfg.seasonal_error_window_s)
        while self._seasonal_err_times and (now - self._seasonal_err_times[0]) > window:
            self._seasonal_err_times.popleft()

        if len(self._seasonal_err_times) >= int(self.cfg.seasonal_error_trip_count):
            self._seasonal_circuit_until_s = now + float(self.cfg.seasonal_cooloff_s)
            self._debug_emit(event="seasonal_circuit_tripped", payload={
                "ts": now,
                "cooloff_s": float(self.cfg.seasonal_cooloff_s),
                "error": str(error),
            })

    def _is_seasonal_error_critical(self, error: Exception) -> bool:
        s = str(error).lower()

        transient = [
            "holiday",
            "weekend",
            "outside_trading",
            "off_hours",
            "low_volume",
            "insufficient_data",
        ]
        if any(t in s for t in transient):
            return False

        critical = [
            "timezone",
            "zoneinfo",
            "config",
            "session",
            "adaptive",
            "corrupt",
            "nan",
        ]
        return any(c in s for c in critical)

    # ─────────────────────────────────────────────────────────────
    # Debug (base-integrated, buffered)
    # ─────────────────────────────────────────────────────────────

    def _debug_emit(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.cfg.debug_enabled:
            return

        # Prefer base debug logger if present
        base_dbg = getattr(self, "_debug_log", None)
        if callable(base_dbg):
            try:
                base_dbg(event=event, **payload)
                return
            except Exception:
                pass

        # Fallback: buffered JSONL writer
        rec = {"event": event, **payload}
        self._debug_buf.append(rec)

        now_s = self._now_s()
        if len(self._debug_buf) >= int(self.cfg.debug_buffer_size) or (now_s - self._debug_last_flush_s) >= float(self.cfg.debug_flush_seconds):
            self._debug_flush_buffer()

    def _debug_flush_buffer(self) -> None:
        if not self._debug_buf:
            return
        try:
            path = self.cfg.debug_path
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for rec in self._debug_buf:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._debug_buf.clear()
            self._debug_last_flush_s = self._now_s()
        except Exception:
            # drop buffer to avoid memory growth if disk is broken
            self._debug_buf.clear()
            self._debug_last_flush_s = self._now_s()

    # ─────────────────────────────────────────────────────────────
    # Caching wrappers (use base cache if available; else TTL cache)
    # ─────────────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Any:
        # Base cache integration if available
        get_fn = getattr(self, "_get_cached_indicators", None)
        if callable(get_fn):
            try:
                v = get_fn("XAUUSD", np.array([0.0], dtype=float), timeframe=key)  # shape-insensitive probe
                if v is not None:
                    return v
            except Exception:
                pass

        # TTL cache fallback
        entry = self._ttl_cache.get(key)
        if entry is None:
            return None
        exp, val = entry
        if self._now_s() >= exp:
            self._ttl_cache.pop(key, None)
            return None
        return val

    def _cache_set(self, key: str, value: Any, ttl_s: float) -> None:
        set_fn = getattr(self, "_set_cached_indicators", None)
        if callable(set_fn):
            try:
                set_fn("XAUUSD", np.array([0.0], dtype=float), value, timeframe=key, ttl_seconds=float(ttl_s))
                return
            except Exception:
                pass

        self._ttl_cache[key] = (self._now_s() + float(ttl_s), value)

    # ─────────────────────────────────────────────────────────────
    # Small helpers
    # ─────────────────────────────────────────────────────────────

    def _safe_bus_get(self, key: str, module: str = "", default: Any = None) -> Any:
        try:
            mod = module or getattr(self, "module_name", "SeasonalityExpert")
            return self.smart_bus.get(key, mod, default=default)  # type: ignore
        except Exception:
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            f = float(value)
            return f if np.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    def _now_s(self) -> float:
        return time.time()

    def _dow_bias_label(self, dow: int) -> str:
        if dow == 0:
            return "cautious"
        if dow == 4:
            return "closing_bias"
        if dow in (1, 2):
            return "trending"
        return "neutral"

    def _month_pattern_label(self, month: int) -> str:
        if month in (1, 8, 9, 12):
            return "gold_tailwind"
        if month in (3, 4, 5):
            return "gold_headwind"
        if month in (6, 7):
            return "summer_transition"
        return "neutral"
