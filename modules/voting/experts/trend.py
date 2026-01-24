#!/usr/bin/env python3
"""
File: modules/voting/experts/trend_expert.py

Advanced Trend Expert (v3.2) — Base-Aligned, Contract-Safe, Forensic-Debuggable
=============================================================================

Drop-in replacement for TrendExpert aligned with VotingExpertBase.process().

v3.2 additions (implements the audit upgrades you referenced):
1) Fast trigger line upgraded (HMA option) for faster slope/deceleration detection.
   - Keeps EMA alignment for the slow/medium regime backbone (stable).
   - Uses HMA(8) by default for slope/decel (reacts faster, less lag).

2) Choppiness Index (CHOP) veto
   - If CHOP > threshold (default 61.8), forces FLAT (trend engine should not trade chaos).

3) Volume-weighted structure (lightweight “volume profile” proxy)
   - Pivot/liquidity clusters are weighted by relative volume at the pivot candle.
   - Order-block proximity is also weighted by volume at displacement.

4) Reduced structure lag
   - Fractal pivot confirmation right-window reduced via config (default right=1).
   - You accept slightly more noise for materially faster BOS/structure detection.

Everything remains contract-safe:
- Missing data => graceful no-op / conservative outputs.
- Cross-asset confirmation stays confidence-only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.experts.base import VotingExpertBase
from modules.voting.core.per_instrument import normalize_instrument
from modules.voting.core.constants import (
    CONFIDENCE_THRESHOLD_F,
    MIN_SIGNAL_STRENGTH_F,
    HIGH_CONFIDENCE_THRESHOLD_F,
    PRIMARY_TIMEFRAME,
    CONTEXT_TIMEFRAMES,
)

# ────────────────────────────────────────────────────────────────
# FILE-LEVEL SWITCHES (requested: NOT config-driven)
# ────────────────────────────────────────────────────────────────

TREND_EXPERT_DEBUG_ENABLED: bool = True
USE_FORMING_BAR: bool = False  # parity-safe default
FORMING_BAR_MODE: str = "append"  # "append" (preferred) or "replace_last" (legacy-style)


@module(**module_args("TrendExpert"))
class TrendExpert(VotingExpertBase):
    """
    TrendExpert — base-aligned implementation.
    - M15 is the primary decision timeframe.
    - H1/H4/D1 only modify confidence (never direction).
    - Returns per-instrument blocks under proposal["proposals"].
    """

    # ────────────────────────────────────────────────────────────────
    # Debug logger override (file-level bool, not config-driven)
    # ────────────────────────────────────────────────────────────────

    def _init_debug_logger(self) -> None:
        """
        Override base debug logger init so enablement is controlled by
        TREND_EXPERT_DEBUG_ENABLED (file-level), not config.
        """
        self._debug_enabled = bool(TREND_EXPERT_DEBUG_ENABLED)
        self._debug_include_market_data = True
        self._debug_include_outputs = True
        self._debug_flush = True

        self._debug_logger: Optional[logging.Logger] = None
        if not self._debug_enabled:
            return

        debug_dir = str(self.config.get("debug_dir", "logs/voting_experts") or "logs/voting_experts")
        max_bytes = int(self.config.get("debug_max_bytes", 10 * 1024 * 1024))
        backup_count = int(self.config.get("debug_backup_count", 5))

        try:
            os.makedirs(debug_dir, exist_ok=True)
        except Exception:
            self._debug_enabled = False
            return

        name = self.__class__.__name__
        filename = os.path.join(debug_dir, f"{name}.debug.jsonl")

        logger = logging.getLogger(f"{__name__}.{name}.debug")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if not any(
            isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == filename
            for h in logger.handlers
        ):
            handler = RotatingFileHandler(filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)

        self._debug_logger = logger
        self._debug_log(event="debug_logger_initialized", debug_dir=debug_dir, filename=filename)

    # ────────────────────────────────────────────────────────────────
    # Init
    # ────────────────────────────────────────────────────────────────

    def _expert_specific_init(self) -> None:
        self.module_name = self.__class__.__name__

        # Instruments (default: XAUUSD only; still supports multiple if config supplies)
        instruments = self.config.get("instruments", ["XAUUSD"])
        if isinstance(instruments, str):
            instruments = [instruments]
        if not isinstance(instruments, list) or not instruments:
            instruments = ["XAUUSD"]
        self.instruments: List[str] = [normalize_instrument(x) for x in instruments if str(x).strip()]
        if not self.instruments:
            self.instruments = ["XAUUSD"]

        # Primary timeframe used to generate direction
        self.primary_tf = str(self.config.get("primary_timeframe", PRIMARY_TIMEFRAME) or PRIMARY_TIMEFRAME).upper()
        self.context_tfs = [
            str(tf).upper()
            for tf in (list(CONTEXT_TIMEFRAMES) if CONTEXT_TIMEFRAMES else ["H1", "H4", "D1"])
        ]

        # MA periods
        self.fast_period = int(self.config.get("fast_period", 8))
        self.medium_period = int(self.config.get("medium_period", 21))
        self.slow_period = int(self.config.get("slow_period", 90))

        # Fast trigger line mode (for slope/decel only)
        # - "hma" is recommended for faster cornering vs EMA
        self.fast_line_mode = str(self.config.get("fast_line_mode", "hma") or "hma").lower()
        if self.fast_line_mode not in ("ema", "hma"):
            self.fast_line_mode = "hma"

        # Minimum bars required
        self.min_price_history = int(self.config.get("min_price_history", self.slow_period + 10))

        # ADX / SAR / S/R
        self.adx_period = int(self.config.get("adx_period", 14))
        self.adx_trending_threshold = float(self.config.get("adx_trending", 25.0))
        self.adx_strong_threshold = float(self.config.get("adx_strong", 40.0))

        self.sar_af_start = float(self.config.get("sar_af_start", 0.02))
        self.sar_af_max = float(self.config.get("sar_af_max", 0.20))

        self.sr_lookback = int(self.config.get("sr_lookback", 60))
        self.sr_threshold = float(self.config.get("sr_threshold", 0.005))  # relative proximity threshold

        # Composite decision thresholds (normalized score space)
        self.score_deadzone = float(self.config.get("score_deadzone", 0.08))  # below this -> flat
        self.score_strong = float(self.config.get("score_strong", 0.22))  # above this -> “strong” trend

        # Confidence shaping from MTF
        self.use_mtf_confirmation = bool(self.config.get("use_mtf_confirmation", True))
        self.mtf_agreement_bonus = float(self.config.get("mtf_agreement_bonus", 0.12))
        self.mtf_disagreement_penalty = float(self.config.get("mtf_disagreement_penalty", 0.18))

        # Momentum deceleration (slope rounding)
        self.use_slope_deceleration = bool(self.config.get("use_slope_deceleration", True))
        self.decel_window = int(self.config.get("decel_window", 12))  # bars per window
        self.decel_k = float(self.config.get("decel_k", 60.0))  # tanh steepness
        self.decel_weight = float(self.config.get("decel_weight", 0.18))  # score penalty weight

        # Exhaustion / climax filters
        self.use_exhaustion_filter = bool(self.config.get("use_exhaustion_filter", True))
        self.exh_atr_period = int(self.config.get("exh_atr_period", 14))
        self.exh_z_window = int(self.config.get("exh_z_window", 50))
        self.exh_atr_z_thr = float(self.config.get("exh_atr_z_thr", 2.0))  # ATR zscore threshold
        self.exh_roc_z_thr = float(self.config.get("exh_roc_z_thr", 2.0))  # ATR-ROC zscore threshold
        self.exh_range_ratio_thr = float(self.config.get("exh_range_ratio_thr", 1.8))  # last3 vs prev20
        self.exh_conf_penalty = float(self.config.get("exh_conf_penalty", 0.22))
        self.exh_strength_penalty = float(self.config.get("exh_strength_penalty", 0.30))
        self.exh_flat_thr = float(self.config.get("exh_flat_thr", 0.85))  # if exhaustion >= this -> force flat

        # Liquidity grab veto (wick geometry)
        self.use_liquidity_grab_veto = bool(self.config.get("use_liquidity_grab_veto", True))
        self.wick_ratio_thr = float(self.config.get("wick_ratio_thr", 1.8))
        self.wick_min_range_atr = float(self.config.get("wick_min_range_atr", 0.6))  # candle range in ATRs
        self.liq_grab_penalty = float(self.config.get("liq_grab_penalty", 0.20))  # score pull toward 0
        self.liq_touch_eps_atr = float(self.config.get("liq_touch_eps_atr", 0.25))  # touch tolerance in ATRs

        # Cross-asset confirmation (confidence-only)
        self.use_cross_asset_confirmation = bool(self.config.get("use_cross_asset_confirmation", True))
        self.cross_asset_bonus = float(self.config.get("cross_asset_bonus", 0.10))
        self.cross_asset_penalty = float(self.config.get("cross_asset_penalty", 0.16))
        self.cross_asset_symbols = self.config.get(
            "cross_asset_symbols",
            {"dxy": ["DXY", "DX", "DOLLAR_INDEX"], "us10y": ["US10Y", "TNX", "US10", "10Y"]},
        )
        if not isinstance(self.cross_asset_symbols, dict):
            self.cross_asset_symbols = {"dxy": ["DXY", "DX", "DOLLAR_INDEX"], "us10y": ["US10Y", "TNX", "US10", "10Y"]}

        # CHOP veto (anti-chop)
        self.use_chop_filter = bool(self.config.get("use_chop_filter", True))
        self.chop_period = int(self.config.get("chop_period", 14))
        self.chop_thr = float(self.config.get("chop_thr", 61.8))
        self.chop_conf_penalty = float(self.config.get("chop_conf_penalty", 0.15))
        self.chop_strength_penalty = float(self.config.get("chop_strength_penalty", 0.25))

        # Structure lag control + volume weighting
        self.structure_left = int(self.config.get("structure_left", 3))
        self.structure_right = int(self.config.get("structure_right", 1))  # reduced lag default
        self.volume_weight_structure = bool(self.config.get("volume_weight_structure", True))

        # Lightweight per-instrument persistence (duration tracking)
        self._inst_state: Dict[str, Dict[str, Any]] = {inst: {"last_dir": "flat", "duration": 0} for inst in self.instruments}

        self.log_info(
            f"[TREND] TrendExpert(v3.2) init | instruments={self.instruments} | "
            f"tf={self.primary_tf} | MA={self.fast_period}/{self.medium_period}/{self.slow_period} | "
            f"fast_line_mode={self.fast_line_mode} | ADX={self.adx_period} | "
            f"decel={self.use_slope_deceleration} | exh={self.use_exhaustion_filter} | "
            f"chop={self.use_chop_filter} | liq_veto={self.use_liquidity_grab_veto} | "
            f"structR={self.structure_right} | volW={self.volume_weight_structure} | "
            f"xasset={self.use_cross_asset_confirmation}"
        )

    # ────────────────────────────────────────────────────────────────
    # Base hook: proposal generation
    # ────────────────────────────────────────────────────────────────

    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a proposal dict that contains per-instrument blocks under:
            proposal["proposals"][instrument] = {...}

        Base will:
        - compute confidence (via _calculate_expert_specific_confidence)
        - position-focus reframe (per-instrument aware)
        - postprocess/gate (top-level)
        - publish to bus
        """
        name = self.__class__.__name__

        historical = self._safe_bus_get("historical_prices", name, default=None)
        if not isinstance(historical, dict) or not historical:
            self._debug_log(event="no_historical_prices", keys=list(historical.keys()) if isinstance(historical, dict) else str(historical))
            return {"action": "flat", "signal_strength": 0.0, "reason": "No historical_prices on bus", "proposals": {}}

        primary_symbol = str(market_data.get("primary_symbol") or self.config.get("primary_symbol") or "XAUUSD")
        primary_norm = normalize_instrument(primary_symbol)

        per_inst: Dict[str, Dict[str, Any]] = {}
        analysis_brief: Dict[str, Any] = {}

        for inst in self.instruments:
            inst_block = self._analyze_instrument(historical, inst, market_data=market_data)
            per_inst[inst] = inst_block
            analysis_brief[inst] = {
                "action": inst_block.get("action"),
                "signal_strength": inst_block.get("signal_strength"),
                "confidence": inst_block.get("confidence"),
                "trend_strength": inst_block.get("trend_strength"),
                "reason": inst_block.get("reason"),
            }

        # Choose primary block for top-level defaults (committee still sees full per-instrument)
        top_inst = primary_norm if primary_norm in per_inst else (self.instruments[0] if self.instruments else primary_norm)
        top_block = per_inst.get(top_inst, {"action": "flat", "signal_strength": 0.0, "confidence": 0.1, "reason": "no_primary_block"})

        proposal: Dict[str, Any] = {
            "action": str(top_block.get("action", "flat")),
            "signal_strength": float(top_block.get("signal_strength", 0.0) or 0.0),
            "confidence": float(top_block.get("confidence", 0.1) or 0.1),
            "reason": str(top_block.get("reason", "Trend proposal")),
            "current_trend": str(top_block.get("current_trend", "neutral")),
            "trend_strength": float(top_block.get("trend_strength", 0.0) or 0.0),
            "proposals": per_inst,
            "analysis_brief": analysis_brief,
            "primary_instrument": top_inst,
            "forming_bar_used": bool(USE_FORMING_BAR),
        }

        self._debug_log(
            event="trend_proposal_built",
            primary=top_inst,
            top_action=proposal["action"],
            top_strength=proposal["signal_strength"],
            top_conf=proposal["confidence"],
            per_inst_actions={k: v.get("action") for k, v in per_inst.items()},
        )
        return proposal

    # ────────────────────────────────────────────────────────────────
    # Base hook: confidence calculation
    # ────────────────────────────────────────────────────────────────

    async def _calculate_expert_specific_confidence(self, proposal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """
        Confidence is taken from the primary instrument block if present.
        """
        try:
            if "confidence" in proposal:
                return float(proposal.get("confidence") or 0.1)

            primary = normalize_instrument(str(market_data.get("primary_symbol") or proposal.get("primary_instrument") or "XAUUSD"))
            per_inst = proposal.get("proposals", {})
            if isinstance(per_inst, dict) and primary in per_inst and isinstance(per_inst[primary], dict):
                return float(per_inst[primary].get("confidence", 0.1) or 0.1)

            sig = float(proposal.get("signal_strength", 0.0) or 0.0)
            return float(max(0.05, min(0.95, 0.25 + 0.65 * sig)))
        except Exception:
            return 0.1
        



    def _ensure_mtf_analysis_contract(self, proposal: Dict[str, Any]) -> None:
        """
        PPO downstream expects: proposal["proposals"][inst]["mtf_analysis"]["trends"][tf] to exist.
        v3.2 currently provides "mtf": {"details": {...}} but not always "mtf_analysis".
        This function normalizes per-instrument blocks to always include mtf_analysis.trends.
        """
        if not isinstance(proposal, dict):
            return

        per_inst_blocks = proposal.get("proposals")
        if not isinstance(per_inst_blocks, dict):
            return

        # Context timeframes the PPO builder expects
        tfs = list(getattr(self, "context_tfs", ["H1", "H4", "D1"]))

        for _inst, _blk in per_inst_blocks.items():
            if not isinstance(_blk, dict):
                continue

            mtf_analysis = _blk.get("mtf_analysis")
            trends = mtf_analysis.get("trends") if isinstance(mtf_analysis, dict) else None
            if isinstance(trends, dict) and trends:
                continue  # already good

            mtf = _blk.get("mtf")
            details = mtf.get("details") if isinstance(mtf, dict) else None

            built_trends: Dict[str, Any] = {}
            for tf in tfs:
                tfd = details.get(tf) if isinstance(details, dict) else None
                if isinstance(tfd, dict):
                    built_trends[str(tf)] = {
                        "direction": str(tfd.get("dir", "neutral")).lower(),
                        "strength": float(tfd.get("strength", 0.0) or 0.0),
                        "ma_spread": float(tfd.get("spread", 0.0) or 0.0),
                        "slope": float(tfd.get("slope", 0.0) or 0.0),
                    }
                else:
                    built_trends[str(tf)] = {"direction": "neutral", "strength": 0.0, "ma_spread": 0.0, "slope": 0.0}

            _blk["mtf_analysis"] = {"trends": built_trends}


    # ────────────────────────────────────────────────────────────────
    # OPTIONAL: publish legacy alias bus keys after base processing
    # ────────────────────────────────────────────────────────────────

    async def process(self, **inputs) -> Dict[str, Any]:
        out = await super().process(**inputs)

        name = self.__class__.__name__
        proposal = out.get(f"{name}_voting_proposal") or out.get("voting_proposal") or out.get("proposal")
        conf = out.get(f"{name}_confidence") or out.get("confidence") or 0.1

        if not isinstance(proposal, dict):
            primary_inst = normalize_instrument(
                str(
                    inputs.get("market_data", {}).get("primary_symbol", "XAUUSD")
                )
            ) or "XAUUSD"
            fallback_conf = float(conf) if conf is not None else 0.1
            fallback_conf = float(max(0.0, min(1.0, fallback_conf)))
            per_inst_block = {
                "instrument": primary_inst,
                "action": "flat",
                "confidence": fallback_conf,
                "signal_strength": 0.0,
                "magnitude": 0.0,
                "rationale": "Missing/invalid proposal from base",
                # Strict PPOObservationBuilder requirements (trend_analysis proposal contract)
                "near_support": False,
                "near_resistance": False,
                "structure_trend": 0.0,
                "bos_signal": 0.0,
                "order_block_bull": 0.0,
                "order_block_bear": 0.0,
                "mtf_analysis": {
                    "trends": {
                        "H1": {"direction": "neutral", "strength": 0.0, "ma_spread": 0.0, "slope": 0.0},
                        "H4": {"direction": "neutral", "strength": 0.0, "ma_spread": 0.0, "slope": 0.0},
                        "D1": {"direction": "neutral", "strength": 0.0, "ma_spread": 0.0, "slope": 0.0},
                    }
                },
            }
            proposal = {
                "action": "flat",
                "signal_strength": 0.0,
                "confidence": fallback_conf,
                "reason": "Missing/invalid proposal from base",
                "proposals": {primary_inst: per_inst_block},
                "analysis_brief": {},
                "primary_instrument": primary_inst,
                "forming_bar_used": bool(USE_FORMING_BAR),
            }

        # FIX: Ensure mtf_analysis.trends exists per instrument for PPO builder
        try:
            self._ensure_mtf_analysis_contract(proposal)
        except Exception:
            pass

        out["trend_voting_proposal"] = proposal
        out["trend_confidence"] = float(conf) if conf is not None else 0.1
        out["trend_analysis"] = {
            "top": proposal,
            "per_instrument": proposal.get("proposals", {}) if isinstance(proposal.get("proposals"), dict) else {},
        }

        try:
            self._safe_bus_set("trend_voting_proposal", out["trend_voting_proposal"], module=name, thesis="TrendExpert (alias)")
            self._safe_bus_set("trend_confidence", out["trend_confidence"], module=name, thesis="TrendExpert confidence (alias)")
            self._safe_bus_set("trend_analysis", out["trend_analysis"], module=name, thesis="TrendExpert analysis (alias)")
        except Exception:
            pass

        return out


    # ────────────────────────────────────────────────────────────────
    # Instrument analysis
    # ────────────────────────────────────────────────────────────────

    def _analyze_instrument(self, historical: Dict[str, Any], instrument_norm: str, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze one instrument on PRIMARY TF (direction) + context TFs (confidence).
        Returns a single per-instrument proposal block.
        """
        t0 = time.time()
        market_data = market_data or {}

        sym_key = self._match_symbol_key(historical, instrument_norm)
        if sym_key is None:
            return {"action": "flat", "signal_strength": 0.0, "confidence": 0.1, "reason": f"No historical block for {instrument_norm}", "current_trend": "unknown", "trend_strength": 0.0}

        sym_block = historical.get(sym_key)
        if not isinstance(sym_block, dict):
            return {"action": "flat", "signal_strength": 0.0, "confidence": 0.1, "reason": f"Invalid historical block for {instrument_norm}", "current_trend": "unknown", "trend_strength": 0.0}

        primary_rec = sym_block.get(self.primary_tf)
        if not isinstance(primary_rec, dict):
            return {"action": "flat", "signal_strength": 0.0, "confidence": 0.1, "reason": f"Missing {self.primary_tf} data for {instrument_norm}", "current_trend": "unknown", "trend_strength": 0.0}

        o, h, l, c = self._extract_ohlc(primary_rec)
        vol = self._extract_volume(primary_rec)

        if USE_FORMING_BAR:
            o, h, l, c = self._apply_forming_bar(primary_rec, o, h, l, c)
            vol = self._apply_forming_bar_volume(primary_rec, vol)

        n = min(len(o), len(h), len(l), len(c))
        if n < self.min_price_history:
            return {"action": "flat", "signal_strength": 0.0, "confidence": 0.1, "reason": f"Insufficient bars ({n} < {self.min_price_history})", "current_trend": "unknown", "trend_strength": 0.0, "bars": n}

        # Work on tail to reduce CPU
        lookback = min(260, n)
        o = o[-lookback:]
        h = h[-lookback:]
        l = l[-lookback:]
        c = c[-lookback:]
        vol = vol[-lookback:] if vol else []

        # Cache hash hardening:
        # - Always encode "volume availability" + last volume (or 0.0) so caches never go stale
        #   when volume appears/disappears or spikes.
        vol_present = 1.0 if (vol and len(vol) > 0) else 0.0
        vol_last = float(vol[-1]) if (vol and len(vol) > 0) else 0.0

        cache_hash_series: List[float] = list(c) + [h[-1], l[-1], vol_present, vol_last]


        cache_hash_series: List[float] = list(c) + [h[-1], l[-1], vol_present, vol_last]

        cached = self._get_cached_indicators(instrument_norm, cache_hash_series, timeframe=self.primary_tf)

        if cached is None:
            # Regime backbone: EMA alignment (stable)
            ema_fast = self._ema(c, self.fast_period)
            ema_med = self._ema(c, self.medium_period)
            ema_slow = self._ema(c, self.slow_period)
            ma_alignment = 1 if (ema_fast > ema_med > ema_slow) else (-1 if (ema_fast < ema_med < ema_slow) else 0)

            # Fast trigger line (for slope/decel): HMA recommended
            fast_line_value = ema_fast
            fast_line_series: List[float]
            if self.fast_line_mode == "hma":
                fast_line_series = self._hma_series(c, self.fast_period)
                fast_line_value = float(fast_line_series[-1]) if fast_line_series else float(ema_fast)
            else:
                fast_line_series = self._ema_series(c, self.fast_period)

            adx, plus_di, minus_di = self._adx_wilder(h, l, c, self.adx_period)
            sar, sar_dir = self._parabolic_sar(h, l, af_start=self.sar_af_start, af_max=self.sar_af_max)

            # Slope uses fast trigger series for faster turn detection
            slope = self._trend_slope_norm(fast_line_series, lookback=min(30, len(fast_line_series))) if fast_line_series else self._trend_slope_norm(c, lookback=min(30, len(c)))

            support, resistance = self._find_support_resistance(h, l, lookback=min(self.sr_lookback, len(h)))
            near_sup, near_res = self._check_sr_proximity(float(c[-1]), support, resistance)

            # CHOP (anti-chop veto)
            chop = self._choppiness_index(h, l, c, period=self.chop_period) if self.use_chop_filter else 0.0

            # Advanced structure (volume-weighted)
            struct = self._compute_advanced_market_structure(
                highs=h, lows=l, closes=c, current_price=float(c[-1]),
                opens=o, volume=vol if vol else None,
            )

            # Exhaustion metrics
            exh = self._compute_exhaustion_metrics(h, l, c, vol)

            # Momentum deceleration (uses fast trigger series)
            decel = self._compute_slope_decel(c, fast_series=fast_line_series)

            # Liquidity grab detection (needs ATR for eps)
            liq_grab = self._compute_liquidity_grab(
                o=o, h=h, l=l, c=c,
                support=support, resistance=resistance,
                struct=struct,
                atr=float(exh.get("atr_now", 0.0) or 0.0),
            )

            cached = {
                "ema_fast": float(ema_fast),
                "ema_med": float(ema_med),
                "ema_slow": float(ema_slow),
                "fast_line_value": float(fast_line_value),
                "ma_alignment": int(ma_alignment),
                "adx": float(adx),
                "plus_di": float(plus_di),
                "minus_di": float(minus_di),
                "sar": float(sar),
                "sar_dir": int(sar_dir),
                "slope": float(slope),
                "support": support,
                "resistance": resistance,
                "near_sup": bool(near_sup),
                "near_res": bool(near_res),
                "chop": float(chop),
                "struct": struct,
                "exhaustion": exh,
                "decel": decel,
                "liq_grab": liq_grab,
            }
            self._set_cached_indicators(instrument_norm, cache_hash_series, cached, timeframe=self.primary_tf)

        exh = cached.get("exhaustion", {}) if isinstance(cached.get("exhaustion"), dict) else {}
        decel = cached.get("decel", {}) if isinstance(cached.get("decel"), dict) else {}
        liq_grab = cached.get("liq_grab", {}) if isinstance(cached.get("liq_grab"), dict) else {}
        chop_val = float(cached.get("chop", 0.0) or 0.0)

        # Build normalized composite score
        score, components = self._composite_score(
            close=float(c[-1]),
            ema_fast=float(cached["ema_fast"]),
            ema_med=float(cached["ema_med"]),
            ema_slow=float(cached["ema_slow"]),
            ma_alignment=int(cached["ma_alignment"]),
            adx=float(cached["adx"]),
            plus_di=float(cached["plus_di"]),
            minus_di=float(cached["minus_di"]),
            sar_dir=int(cached["sar_dir"]),
            slope=float(cached["slope"]),
            near_support=bool(cached["near_sup"]),
            near_resistance=bool(cached["near_res"]),
            struct=cached.get("struct", {}) if isinstance(cached.get("struct"), dict) else {},
            slope_decel=float(decel.get("decel", 0.0) or 0.0),
        )

        # Decide action from score (direction), then compute strength/confidence
        action, trend_label = self._action_from_score(score)
        signal_strength = self._strength_from_score(score, adx=float(cached["adx"]))
        confidence = self._confidence_from(signal_strength, adx=float(cached["adx"]), ma_alignment=int(cached["ma_alignment"]))

        # 0) CHOP veto (before other filters)
        if self.use_chop_filter and chop_val > 0.0:
            components["chop"] = float(chop_val)
            if chop_val >= self.chop_thr:
                action = "flat"
                trend_label = "neutral"
                signal_strength = float(np.clip(signal_strength * (1.0 - self.chop_strength_penalty), 0.0, 1.0))
                confidence = float(np.clip(confidence - self.chop_conf_penalty, 0.05, 0.95))

        # 1) Exhaustion filter (flatten / reduce conviction in climax)
        exh_score = float(exh.get("exhaustion_score", 0.0) or 0.0)
        if self.use_exhaustion_filter and action in ("long", "short") and exh_score > 0.0:
            signal_strength = float(np.clip(signal_strength * (1.0 - self.exh_strength_penalty * exh_score), 0.0, 1.0))
            confidence = float(np.clip(confidence - self.exh_conf_penalty * exh_score, 0.05, 0.95))
            score = float(score * (1.0 - 0.65 * exh_score))
            components["exhaustion_score"] = float(exh_score)
            components["exhaustion_penalty_applied"] = float(exh_score)

            if exh_score >= self.exh_flat_thr:
                action = "flat"
                trend_label = "neutral"
                signal_strength = float(max(0.0, signal_strength * 0.35))
                confidence = float(max(0.05, confidence - 0.10))

        # 2) Liquidity grab veto (wick geometry at key levels)
        if self.use_liquidity_grab_veto and action in ("long", "short"):
            if action == "long" and bool(liq_grab.get("veto_long", False)):
                score = float(score - self.liq_grab_penalty * np.sign(score))
                action = "flat"
                trend_label = "neutral"
                signal_strength = float(max(0.0, signal_strength * 0.35))
                confidence = float(max(0.05, confidence - 0.12))
            elif action == "short" and bool(liq_grab.get("veto_short", False)):
                score = float(score - self.liq_grab_penalty * np.sign(score))
                action = "flat"
                trend_label = "neutral"
                signal_strength = float(max(0.0, signal_strength * 0.35))
                confidence = float(max(0.05, confidence - 0.12))

        # 3) MTF confirmation: confidence-only adjustment
        mtf = self._mtf_conf_adjust(sym_block, action)
        confidence = float(np.clip(confidence + float(mtf.get("delta_conf", 0.0) or 0.0), 0.05, 0.95))

        # 4) Cross-asset confirmation: confidence-only adjustment
        xasset = self._cross_asset_conf_adjust(primary_action=action, market_data=market_data)
        confidence = float(np.clip(confidence + float(xasset.get("delta_conf", 0.0) or 0.0), 0.05, 0.95))

        # Track duration (direction persistence)
        st = self._inst_state.setdefault(instrument_norm, {"last_dir": "flat", "duration": 0})
        last_dir = str(st.get("last_dir", "flat"))
        if action in ("long", "short") and action == last_dir:
            st["duration"] = int(st.get("duration", 0)) + 1
        elif action in ("long", "short"):
            st["duration"] = 1
        else:
            st["duration"] = 0
        st["last_dir"] = action
        duration = int(st.get("duration", 0))

        block: Dict[str, Any] = {
            "action": action,
            "signal_strength": float(signal_strength),
            "confidence": float(confidence),
            "reason": self._reason_string(
                instrument_norm=instrument_norm,
                trend_label=trend_label,
                score=float(score),
                adx=float(cached["adx"]),
                plus_di=float(cached["plus_di"]),
                minus_di=float(cached["minus_di"]),
                ma_alignment=int(cached["ma_alignment"]),
                mtf=mtf,
                exhaustion=exh,
                decel=decel,
                liq_grab=liq_grab,
                xasset=xasset,
                chop=float(chop_val),
            ),
            "current_trend": trend_label,
            "trend_strength": float(abs(score)),
            "trend_duration": duration,
            "ema_fast": float(cached["ema_fast"]),
            "ema_med": float(cached["ema_med"]),
            "ema_slow": float(cached["ema_slow"]),
            "fast_line_mode": str(self.fast_line_mode),
            "fast_line_value": float(cached.get("fast_line_value", cached["ema_fast"])),
            "ma_alignment": int(cached["ma_alignment"]),
            "adx": float(cached["adx"]),
            "plus_di": float(cached["plus_di"]),
            "minus_di": float(cached["minus_di"]),
            "sar_direction": int(cached["sar_dir"]),
            "trend_slope": float(cached["slope"]),
            "near_support": bool(cached["near_sup"]),
            "near_resistance": bool(cached["near_res"]),
            "chop": float(chop_val),
            "components": components,
            "mtf": mtf,
            "cross_asset": xasset,
            # Momentum decel debug
            "slope_decel": float(decel.get("decel", 0.0) or 0.0),
            "slope_now": float(decel.get("slope_now", 0.0) or 0.0),
            "slope_prev": float(decel.get("slope_prev", 0.0) or 0.0),
            # Exhaustion debug
            "exhaustion_score": float(exh.get("exhaustion_score", 0.0) or 0.0),
            "atr_now": float(exh.get("atr_now", 0.0) or 0.0),
            "atr_z": float(exh.get("atr_z", 0.0) or 0.0),
            "atr_roc_z": float(exh.get("atr_roc_z", 0.0) or 0.0),
            "range_ratio": float(exh.get("range_ratio", 0.0) or 0.0),
            # Liquidity grab debug
            "liquidity_grab": liq_grab,
            # Advanced structure (parity features)
            "structure_trend": float(cached["struct"].get("structure_trend", 0.0)),
            "structure_strength": float(cached["struct"].get("structure_strength", 0.0)),
            "bos_signal": float(cached["struct"].get("bos_signal", 0.0)),
            "liquidity_above": float(cached["struct"].get("liquidity_above", 0.0)),
            "liquidity_below": float(cached["struct"].get("liquidity_below", 0.0)),
            "order_block_bull": float(cached["struct"].get("order_block_bull", 0.0)),
            "order_block_bear": float(cached["struct"].get("order_block_bear", 0.0)),
        }

        # Apply the exact same gating semantics as the base (contract consistency)
        block_norm, conf_norm = self._postprocess_proposal_for_voting(dict(block), float(block["confidence"]))
        block_norm["confidence"] = float(conf_norm)

        self._debug_log(
            event="instrument_analyzed",
            instrument=instrument_norm,
            bars=int(n),
            action_raw=block["action"],
            strength_raw=block["signal_strength"],
            conf_raw=block["confidence"],
            action=block_norm.get("action"),
            strength=block_norm.get("signal_strength"),
            conf=block_norm.get("confidence"),
            score=float(score),
            chop=float(chop_val),
            exh=float(exh_score),
            decel=float(decel.get("decel", 0.0) or 0.0),
            liq_veto={"L": bool(liq_grab.get("veto_long", False)), "S": bool(liq_grab.get("veto_short", False))},
            xasset_delta=float(xasset.get("delta_conf", 0.0) or 0.0),
            elapsed_ms=(time.time() - t0) * 1000.0,
        )

        return block_norm

    # ────────────────────────────────────────────────────────────────
    # Composite scoring
    # ────────────────────────────────────────────────────────────────

    def _composite_score(
        self,
        close: float,
        ema_fast: float,
        ema_med: float,
        ema_slow: float,
        ma_alignment: int,
        adx: float,
        plus_di: float,
        minus_di: float,
        sar_dir: int,
        slope: float,
        near_support: bool,
        near_resistance: bool,
        struct: Dict[str, Any],
        slope_decel: float,
    ) -> Tuple[float, Dict[str, float]]:

        def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
            return float(np.clip(x, lo, hi))

        def tanh_scale(x: float, k: float) -> float:
            return float(np.tanh(k * x))

        # 1) MA structure + spread (slow backbone)
        ma_struct = float(ma_alignment) * 0.35
        ma_spread = 0.0
        if ema_slow > 0:
            ma_spread = (ema_fast - ema_slow) / ema_slow
        ma_spread_n = tanh_scale(ma_spread, 12.0) * 0.30

        # 2) ADX directionality
        di_sum = max(plus_di + minus_di, 1e-8)
        di_imb = (plus_di - minus_di) / di_sum
        adx_gate = clip((adx - 15.0) / 35.0, 0.0, 1.0)
        adx_dir = clip(di_imb) * (0.25 * adx_gate)

        # 3) SAR direction
        sar_comp = clip(float(sar_dir)) * 0.10

        # 4) Slope (fast trigger slope)
        slope_comp = clip(slope) * 0.20

        # 5) Advanced structure
        st_trend = float(struct.get("structure_trend", 0.0) or 0.0)
        st_strength = float(struct.get("structure_strength", 0.0) or 0.0)
        bos = float(struct.get("bos_signal", 0.0) or 0.0)
        struct_comp = clip(st_trend) * (0.12 * clip(st_strength, 0.0, 1.0)) + clip(bos) * 0.10

        # 6) S/R proximity penalty
        sr_pen = 0.0
        if near_resistance:
            sr_pen -= 0.10
        if near_support:
            sr_pen += 0.10

        raw = ma_struct + ma_spread_n + adx_dir + sar_comp + slope_comp + struct_comp + sr_pen

        # 7) Momentum deceleration penalty (rounding)
        decel_comp = 0.0
        if self.use_slope_deceleration:
            d = float(np.clip(slope_decel, -1.0, 1.0))
            if raw > 0.0:
                loss = max(0.0, -d)  # bullish losing momentum
                decel_comp = -float(self.decel_weight) * float(np.tanh(self.decel_k * loss))
            elif raw < 0.0:
                loss = max(0.0, d)  # bearish losing momentum
                decel_comp = +float(self.decel_weight) * float(np.tanh(self.decel_k * loss))

        raw2 = raw + decel_comp
        score = clip(raw2, -1.0, 1.0)

        components = {
            "ma_struct": float(ma_struct),
            "ma_spread_n": float(ma_spread_n),
            "adx_dir": float(adx_dir),
            "sar": float(sar_comp),
            "slope": float(slope_comp),
            "structure": float(struct_comp),
            "sr_bias": float(sr_pen),
            "raw": float(raw),
            "slope_decel": float(slope_decel),
            "decel_comp": float(decel_comp),
            "raw2": float(raw2),
            "score": float(score),
        }
        return score, components

    def _action_from_score(self, score: float) -> Tuple[str, str]:
        s = float(score)
        if abs(s) < self.score_deadzone:
            return "flat", "neutral"
        if s > 0:
            return "long", "uptrend" if s >= self.score_strong else "weak_uptrend"
        return "short", "downtrend" if abs(s) >= self.score_strong else "weak_downtrend"

    def _strength_from_score(self, score: float, adx: float) -> float:
        s = abs(float(score))
        adx_boost = float(
            np.clip(
                (adx - self.adx_trending_threshold)
                / max(1e-8, (self.adx_strong_threshold - self.adx_trending_threshold)),
                0.0,
                1.0,
            )
        )
        strength = float(np.clip(0.10 + 0.85 * s + 0.10 * adx_boost, 0.0, 1.0))
        return strength

    def _confidence_from(self, strength: float, adx: float, ma_alignment: int) -> float:
        conf_floor = float(CONFIDENCE_THRESHOLD_F())
        high_conf = float(HIGH_CONFIDENCE_THRESHOLD_F())

        base = 0.22 + 0.62 * float(np.clip(strength, 0.0, 1.0))

        if adx >= self.adx_strong_threshold:
            base *= 1.10
        elif adx >= self.adx_trending_threshold:
            base *= 1.05
        else:
            base *= 0.95

        if ma_alignment != 0:
            base *= 1.04

        base = float(np.clip(base, conf_floor * 0.5, high_conf))
        return base

    # ────────────────────────────────────────────────────────────────
    # Multi-timeframe confidence adjustment (context-only)
    # ────────────────────────────────────────────────────────────────

    def _mtf_conf_adjust(self, sym_block: Dict[str, Any], primary_action: str) -> Dict[str, Any]:
        if not self.use_mtf_confirmation or primary_action not in ("long", "short"):
            return {"available": False, "delta_conf": 0.0, "agreement": 0.5, "details": {}}

        details: Dict[str, Any] = {}
        votes: List[float] = []
        strengths: List[float] = []

        for tf in self.context_tfs:
            rec = sym_block.get(tf)
            if not isinstance(rec, dict):
                continue

            o, h, l, c = self._extract_ohlc(rec)
            if USE_FORMING_BAR:
                o, h, l, c = self._apply_forming_bar(rec, o, h, l, c)

            n = min(len(h), len(l), len(c))
            if n < max(60, self.slow_period + 5):
                continue

            c_tail = c[-min(240, n):]

            ema_f = self._ema(c_tail, min(self.fast_period, 20))
            ema_s = self._ema(c_tail, min(self.slow_period, 60))
            spread = (ema_f - ema_s) / ema_s if ema_s > 0 else 0.0
            slope = self._trend_slope_norm(c_tail, lookback=min(20, len(c_tail)))

            dirn = "neutral"
            if spread > 0.001 and slope > 0.0005:
                dirn = "bullish"
            elif spread < -0.001 and slope < -0.0005:
                dirn = "bearish"

            strength = float(np.clip(abs(spread) * 10.0 + abs(slope) * 25.0, 0.0, 1.0))
            details[tf] = {"dir": dirn, "strength": strength, "spread": float(spread), "slope": float(slope)}

            if dirn == "neutral":
                votes.append(0.0)
                strengths.append(strength)
                continue

            agrees = (primary_action == "long" and dirn == "bullish") or (primary_action == "short" and dirn == "bearish")
            votes.append(1.0 if agrees else -1.0)
            strengths.append(strength)

        if not votes:
            return {"available": False, "delta_conf": -0.03, "agreement": 0.5, "details": details}

        wsum = float(sum(max(s, 0.15) for s in strengths))
        signed = float(sum(v * max(s, 0.15) for v, s in zip(votes, strengths)) / max(wsum, 1e-8))
        agreement = float((signed + 1.0) / 2.0)

        if signed > 0.25:
            delta = float(self.mtf_agreement_bonus * agreement)
        elif signed < -0.25:
            delta = float(-self.mtf_disagreement_penalty * (1.0 - agreement + 0.25))
        else:
            delta = -0.03

        return {"available": True, "delta_conf": float(delta), "agreement": agreement, "details": details}

    # ────────────────────────────────────────────────────────────────
    # Cross-asset confirmation (confidence-only; optional)
    # ────────────────────────────────────────────────────────────────

    def _cross_asset_conf_adjust(self, primary_action: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.use_cross_asset_confirmation or primary_action not in ("long", "short"):
            return {"available": False, "delta_conf": 0.0, "details": {}}

        name = self.__class__.__name__
        bench = (
            market_data.get("benchmarks")
            or market_data.get("benchmark_prices")
            or self._safe_bus_get("benchmark_prices", name, default=None)
            or self._safe_bus_get("cross_asset_prices", name, default=None)
            or self._safe_bus_get("macro_prices", name, default=None)
        )
        if not isinstance(bench, dict) or not bench:
            return {"available": False, "delta_conf": 0.0, "details": {}}

        def find_symbol(sym_list: List[str]) -> Optional[str]:
            for k in bench.keys():
                nk = normalize_instrument(str(k))
                for s in sym_list:
                    if nk == normalize_instrument(str(s)):
                        return str(k)
            return None

        dxy_key = find_symbol(list(self.cross_asset_symbols.get("dxy", [])))
        us10y_key = find_symbol(list(self.cross_asset_symbols.get("us10y", [])))

        details: Dict[str, Any] = {}
        votes: List[Tuple[float, float]] = []  # (signed_vote, strength)

        for label, key in (("DXY", dxy_key), ("US10Y", us10y_key)):
            if key is None:
                continue
            rec = bench.get(key)
            if not isinstance(rec, dict):
                continue

            tf_rec = rec.get(self.primary_tf) or rec.get(str(PRIMARY_TIMEFRAME).upper()) or rec.get("M15") or rec.get("H1")
            if not isinstance(tf_rec, dict):
                continue

            _, _, _, close = self._extract_ohlc(tf_rec)
            if len(close) < 80:
                continue

            tail = close[-200:]
            ema_f = self._ema(tail, 10)
            ema_s = self._ema(tail, 40)
            spread = (ema_f - ema_s) / ema_s if ema_s > 0 else 0.0
            slp = self._trend_slope_norm(tail, lookback=20)

            dirn = "neutral"
            if spread > 0.0008 and slp > 0.0003:
                dirn = "up"
            elif spread < -0.0008 and slp < -0.0003:
                dirn = "down"

            strength = float(np.clip(abs(spread) * 12.0 + abs(slp) * 30.0, 0.0, 1.0))
            details[label] = {"dir": dirn, "strength": strength, "spread": float(spread), "slope": float(slp)}

            if dirn == "neutral":
                continue

            aligned = (primary_action == "long" and dirn == "down") or (primary_action == "short" and dirn == "up")
            votes.append((1.0 if aligned else -1.0, max(strength, 0.15)))

        if not votes:
            return {"available": False, "delta_conf": 0.0, "details": details}

        wsum = float(sum(w for _, w in votes))
        signed = float(sum(v * w for v, w in votes) / max(wsum, 1e-8))  # [-1, 1]
        agreement = float((signed + 1.0) / 2.0)

        if signed >= 0.25:
            delta = float(self.cross_asset_bonus * agreement)
        elif signed <= -0.25:
            delta = float(-self.cross_asset_penalty * (1.0 - agreement + 0.25))
        else:
            delta = 0.0

        return {"available": True, "delta_conf": float(delta), "agreement": agreement, "details": details}

    # ────────────────────────────────────────────────────────────────
    # Data extraction + indicators
    # ────────────────────────────────────────────────────────────────

    def _match_symbol_key(self, historical: Dict[str, Any], inst_norm: str) -> Optional[str]:
        for k in historical.keys():
            if normalize_instrument(str(k)) == inst_norm:
                return str(k)
        return None

    def _extract_ohlc(self, tf_rec: Dict[str, Any]) -> Tuple[List[float], List[float], List[float], List[float]]:
        def to_floats(x: Any) -> List[float]:
            if not isinstance(x, (list, tuple, np.ndarray)):
                return []
            out: List[float] = []
            for v in x:
                try:
                    out.append(float(v))
                except Exception:
                    continue
            return out

        o = to_floats(tf_rec.get("open"))
        h = to_floats(tf_rec.get("high"))
        l = to_floats(tf_rec.get("low"))
        c = to_floats(tf_rec.get("close"))

        n = min(len(o) if o else len(c), len(h) if h else len(c), len(l) if l else len(c), len(c))
        if n <= 0:
            return [], [], [], []
        if o and len(o) != n:
            o = o[-n:]
        if len(h) != n:
            h = h[-n:]
        if len(l) != n:
            l = l[-n:]
        if len(c) != n:
            c = c[-n:]
        if not o:
            o = [c[0]] + c[:-1]
            if len(o) != n:
                o = o[-n:]
        return o, h, l, c

    def _extract_volume(self, tf_rec: Dict[str, Any]) -> List[float]:
        def to_floats(x: Any) -> List[float]:
            if not isinstance(x, (list, tuple, np.ndarray)):
                return []
            out: List[float] = []
            for v in x:
                try:
                    out.append(float(v))
                except Exception:
                    continue
            return out

        v = to_floats(tf_rec.get("volume"))
        if not v:
            v = to_floats(tf_rec.get("tick_volume"))
        return v

    def _apply_forming_bar_volume(self, tf_rec: Dict[str, Any], vol: List[float]) -> List[float]:
        cur = tf_rec.get("current_bar")
        if not isinstance(cur, dict):
            return vol
        try:
            fv = cur.get("volume")
            if fv is None:
                fv = cur.get("tick_volume")
            if fv is None:
                return vol
            fv = float(fv)
            if not vol:
                return [fv]
            if FORMING_BAR_MODE == "replace_last":
                return vol[:-1] + [fv]
            return vol + [fv]
        except Exception:
            return vol

    def _apply_forming_bar(self, tf_rec: Dict[str, Any], o: List[float], h: List[float], l: List[float], c: List[float]) -> Tuple[List[float], List[float], List[float], List[float]]:
        cur = tf_rec.get("current_bar")
        if not isinstance(cur, dict):
            return o, h, l, c

        try:
            fc = cur.get("close")
            fh = cur.get("high")
            fl = cur.get("low")
            fo = cur.get("open")
            if fc is None:
                return o, h, l, c
            fc = float(fc)
            fh = float(fh) if fh is not None else fc
            fl = float(fl) if fl is not None else fc
            fo = float(fo) if fo is not None else (c[-1] if c else fc)

            if not c:
                return [fo], [fh], [fl], [fc]

            if FORMING_BAR_MODE == "replace_last":
                c2 = c[:-1] + [fc]
                h2 = h[:-1] + [max(h[-1], fh)]
                l2 = l[:-1] + [min(l[-1], fl)]
                o2 = o[:-1] + [o[-1] if o else fo]
                return o2, h2, l2, c2

            return o + [fo], h + [fh], l + [fl], c + [fc]
        except Exception:
            return o, h, l, c

    def _ema(self, prices: List[float], period: int) -> float:
        if not prices:
            return 0.0
        n = len(prices)
        p = int(max(1, period))
        if n < p + 1:
            return float(np.mean(prices))
        alpha = 2.0 / (p + 1.0)
        ema = float(prices[0])
        for x in prices[1:]:
            ema = (float(x) - ema) * alpha + ema
        return float(ema)

    def _ema_series(self, prices: List[float], period: int) -> List[float]:
        if not prices:
            return []
        p = int(max(1, period))
        alpha = 2.0 / (p + 1.0)
        out: List[float] = []
        ema = float(prices[0])
        out.append(ema)
        for x in prices[1:]:
            ema = (float(x) - ema) * alpha + ema
            out.append(float(ema))
        return out

    # ────────────────────────────────────────────────────────────────
    # HMA (Hull Moving Average) — fast trigger line
    # ────────────────────────────────────────────────────────────────

    def _wma_series(self, prices: List[float], period: int) -> List[float]:
        n = len(prices)
        p = int(max(1, period))
        if n == 0:
            return []
        if p == 1:
            return [float(x) for x in prices]

        w = np.arange(1, p + 1, dtype=np.float64)
        wsum = float(np.sum(w))
        out = [float(prices[0])] * min(p - 1, n)

        for i in range(p - 1, n):
            window = np.asarray(prices[i - p + 1 : i + 1], dtype=np.float64)
            out.append(float(np.dot(window, w) / max(wsum, 1e-12)))
        return out

    def _hma_series(self, prices: List[float], period: int) -> List[float]:
        n = len(prices)
        p = int(max(2, period))
        if n == 0:
            return []
        half = max(1, p // 2)
        root = max(1, int(np.sqrt(p)))

        wma_full = self._wma_series(prices, p)
        wma_half = self._wma_series(prices, half)

        # Align lengths
        m = min(len(wma_full), len(wma_half))
        wma_full = wma_full[-m:]
        wma_half = wma_half[-m:]

        diff = [float(2.0 * a - b) for a, b in zip(wma_half, wma_full)]
        return self._wma_series(diff, root)

    def _trend_slope_norm(self, prices: List[float], lookback: int = 20) -> float:
        n = len(prices)
        if n < max(5, lookback):
            return 0.0
        y = np.asarray(prices[-lookback:], dtype=np.float64)
        if not np.all(np.isfinite(y)) or float(np.mean(y)) == 0.0:
            return 0.0
        x = np.arange(len(y), dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        return float(np.clip(slope / max(1e-8, float(np.mean(y))), -1.0, 1.0))

    def _window_slope_norm(self, series: List[float]) -> float:
        if len(series) < 5:
            return 0.0
        y = np.asarray(series, dtype=np.float64)
        if not np.all(np.isfinite(y)):
            return 0.0
        m = float(np.mean(y))
        if not np.isfinite(m) or abs(m) < 1e-12:
            return 0.0
        x = np.arange(len(y), dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        return float(np.clip(slope / m, -1.0, 1.0))

    def _compute_slope_decel(self, closes: List[float], fast_series: Optional[List[float]] = None) -> Dict[str, float]:
        if not self.use_slope_deceleration:
            return {"available": 0.0, "slope_prev": 0.0, "slope_now": 0.0, "decel": 0.0}
        if len(closes) < (2 * max(6, self.decel_window) + 10):
            return {"available": 0.0, "slope_prev": 0.0, "slope_now": 0.0, "decel": 0.0}

        series = fast_series if (fast_series and len(fast_series) == len(closes)) else (
            self._hma_series(closes, self.fast_period) if self.fast_line_mode == "hma" else self._ema_series(closes, self.fast_period)
        )
        if len(series) < (2 * max(6, self.decel_window) + 2):
            return {"available": 0.0, "slope_prev": 0.0, "slope_now": 0.0, "decel": 0.0}

        w = int(max(6, self.decel_window))
        prev_win = series[-2 * w : -w]
        now_win = series[-w :]

        slope_prev = self._window_slope_norm(prev_win)
        slope_now = self._window_slope_norm(now_win)
        decel = float(np.clip(slope_now - slope_prev, -1.0, 1.0))

        return {"available": 1.0, "slope_prev": float(slope_prev), "slope_now": float(slope_now), "decel": float(decel)}

    # ────────────────────────────────────────────────────────────────
    # CHOP (Choppiness Index)
    # ────────────────────────────────────────────────────────────────

    def _choppiness_index(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        p = int(max(5, period))
        n = min(len(highs), len(lows), len(closes))
        if n < p + 2:
            return 0.0

        h = np.asarray(highs[-(p + 1):], dtype=np.float64)
        l = np.asarray(lows[-(p + 1):], dtype=np.float64)
        c = np.asarray(closes[-(p + 1):], dtype=np.float64)

        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        sum_tr = float(np.sum(tr))
        hi = float(np.max(h[1:]))
        lo = float(np.min(l[1:]))
        denom = max(1e-8, hi - lo)
        if sum_tr <= 1e-12:
            return 0.0

        chop = 100.0 * (np.log10(sum_tr / denom) / max(1e-12, np.log10(float(p))))
        if not np.isfinite(chop):
            return 0.0
        return float(np.clip(chop, 0.0, 100.0))

    # ────────────────────────────────────────────────────────────────
    # Exhaustion / Volatility Climax
    # ────────────────────────────────────────────────────────────────

    def _atr_wilder_series(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> List[float]:
        p = int(max(2, period))
        n = min(len(highs), len(lows), len(closes))
        if n < p + 2:
            return []

        h = np.asarray(highs, dtype=np.float64)
        l = np.asarray(lows, dtype=np.float64)
        c = np.asarray(closes, dtype=np.float64)

        tr1 = h[1:] - l[1:]
        tr2 = np.abs(h[1:] - c[:-1])
        tr3 = np.abs(l[1:] - c[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        atr = np.zeros_like(tr)
        atr[0] = float(np.mean(tr[:p]))
        for i in range(1, len(tr)):
            atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p
        return [float(x) for x in atr]

    def _zscore_last(self, series: List[float], window: int) -> float:
        if len(series) < max(10, window):
            return 0.0
        tail = np.asarray(series[-window:], dtype=np.float64)
        if not np.all(np.isfinite(tail)):
            return 0.0
        mu = float(np.mean(tail))
        sd = float(np.std(tail))
        if sd <= 1e-12:
            return 0.0
        return float((tail[-1] - mu) / sd)

    def _compute_exhaustion_metrics(self, highs: List[float], lows: List[float], closes: List[float], volume: List[float]) -> Dict[str, float]:
        if not self.use_exhaustion_filter:
            return {"exhaustion_score": 0.0, "atr_now": 0.0, "atr_z": 0.0, "atr_roc_z": 0.0, "range_ratio": 0.0}

        atr = self._atr_wilder_series(highs, lows, closes, self.exh_atr_period)
        if len(atr) < (self.exh_z_window + 5):
            return {"exhaustion_score": 0.0, "atr_now": float(atr[-1]) if atr else 0.0, "atr_z": 0.0, "atr_roc_z": 0.0, "range_ratio": 0.0}

        atr_now = float(atr[-1])
        atr_z = self._zscore_last(atr, self.exh_z_window)

        atr_roc = []
        for i in range(1, len(atr)):
            denom = max(1e-8, float(atr[i - 1]))
            atr_roc.append(float((atr[i] - atr[i - 1]) / denom))
        atr_roc_z = self._zscore_last(atr_roc, self.exh_z_window - 1) if len(atr_roc) >= self.exh_z_window else 0.0

        rng = [float(h - lo) for h, lo in zip(highs, lows)]
        range_ratio = 0.0
        if len(rng) >= 30:
            last3 = float(np.mean(rng[-3:]))
            prev20 = float(np.mean(rng[-23:-3])) if len(rng) >= 23 else float(np.mean(rng[:-3]))
            if prev20 > 1e-8:
                range_ratio = float(last3 / prev20)

        def sat(x: float, thr: float, k: float = 2.5) -> float:
            return float(np.clip(0.5 + 0.5 * np.tanh(k * (x - thr)), 0.0, 1.0))

        e1 = sat(atr_z, self.exh_atr_z_thr)
        e2 = sat(atr_roc_z, self.exh_roc_z_thr)
        e3 = sat(range_ratio, self.exh_range_ratio_thr, k=1.6)
        exhaustion_score = float(np.clip(max(e1, e2, e3), 0.0, 1.0))

        return {"exhaustion_score": exhaustion_score, "atr_now": atr_now, "atr_z": float(atr_z), "atr_roc_z": float(atr_roc_z), "range_ratio": float(range_ratio)}

    # ────────────────────────────────────────────────────────────────
    # Liquidity grab veto (wick-to-body geometry at key levels)
    # ────────────────────────────────────────────────────────────────

    def _compute_liquidity_grab(
        self,
        o: List[float],
        h: List[float],
        l: List[float],
        c: List[float],
        support: List[float],
        resistance: List[float],
        struct: Dict[str, Any],
        atr: float,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"veto_long": False, "veto_short": False, "details": {}}
        if not self.use_liquidity_grab_veto:
            return out
        if len(c) < 5 or len(h) != len(l) or len(o) != len(c):
            return out

        atr_now = float(atr) if atr and atr > 0 else 0.0
        if atr_now <= 0.0:
            atr_now = float(max(np.mean(np.asarray(h[-20:]) - np.asarray(l[-20:])), 1e-8))

        i = -1
        O, H, L, C = float(o[i]), float(h[i]), float(l[i]), float(c[i])
        rng = max(1e-8, H - L)
        body = max(1e-8, abs(C - O))
        upper_wick = max(0.0, H - max(O, C))
        lower_wick = max(0.0, min(O, C) - L)

        wick_ratio_up = float(upper_wick / body) if body > 0 else 0.0
        wick_ratio_dn = float(lower_wick / body) if body > 0 else 0.0

        if (rng / max(1e-8, atr_now)) < self.wick_min_range_atr:
            out["details"] = {"skipped": "range_too_small", "range_atr": float(rng / max(1e-8, atr_now))}
            return out

        eps = float(self.liq_touch_eps_atr) * atr_now

        px = float(c[i])
        nearest_res = min([r for r in resistance if r >= px - 5 * atr_now], default=None, key=lambda r: abs(r - px)) if resistance else None
        nearest_sup = min([s for s in support if s <= px + 5 * atr_now], default=None, key=lambda s: abs(s - px)) if support else None

        liq_above = float(struct.get("liquidity_above", 0.0) or 0.0)
        liq_below = float(struct.get("liquidity_below", 0.0) or 0.0)

        touched_res = False
        if nearest_res is not None and H >= float(nearest_res) - eps:
            touched_res = True
        implied_liq_above = liq_above >= 0.55

        long_reject = (wick_ratio_up >= self.wick_ratio_thr) and (C < (H - 0.55 * rng))
        if (touched_res or implied_liq_above) and long_reject:
            out["veto_long"] = True

        touched_sup = False
        if nearest_sup is not None and L <= float(nearest_sup) + eps:
            touched_sup = True
        implied_liq_below = liq_below >= 0.55

        short_reject = (wick_ratio_dn >= self.wick_ratio_thr) and (C > (L + 0.55 * rng))
        if (touched_sup or implied_liq_below) and short_reject:
            out["veto_short"] = True

        out["details"] = {
            "wick_ratio_up": float(wick_ratio_up),
            "wick_ratio_dn": float(wick_ratio_dn),
            "range_atr": float(rng / max(1e-8, atr_now)),
            "nearest_res": float(nearest_res) if nearest_res is not None else None,
            "nearest_sup": float(nearest_sup) if nearest_sup is not None else None,
            "touched_res": bool(touched_res),
            "touched_sup": bool(touched_sup),
            "liq_above": float(liq_above),
            "liq_below": float(liq_below),
        }
        return out

    # ────────────────────────────────────────────────────────────────
    # ADX / SAR / S/R
    # ────────────────────────────────────────────────────────────────

    def _adx_wilder(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> Tuple[float, float, float]:
        p = int(max(2, period))
        n = min(len(highs), len(lows), len(closes))
        if n < p + 2:
            return 20.0, 50.0, 50.0

        h = np.asarray(highs, dtype=np.float64)
        l = np.asarray(lows, dtype=np.float64)
        c = np.asarray(closes, dtype=np.float64)

        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr1 = h[1:] - l[1:]
        tr2 = np.abs(h[1:] - c[:-1])
        tr3 = np.abs(l[1:] - c[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        def wilder_smooth(x: np.ndarray, period_: int) -> np.ndarray:
            out = np.zeros_like(x)
            out[0] = np.sum(x[:period_])
            for i in range(1, len(x)):
                out[i] = out[i - 1] - (out[i - 1] / period_) + x[i]
            return out

        atr = wilder_smooth(tr, p)
        plus_sm = wilder_smooth(plus_dm, p)
        minus_sm = wilder_smooth(minus_dm, p)

        atr_safe = np.maximum(atr, 1e-8)
        plus_di = 100.0 * (plus_sm / atr_safe)
        minus_di = 100.0 * (minus_sm / atr_safe)

        di_sum = np.maximum(plus_di + minus_di, 1e-8)
        dx = 100.0 * (np.abs(plus_di - minus_di) / di_sum)

        adx = np.zeros_like(dx)
        adx[:p] = np.mean(dx[:p])
        for i in range(p, len(dx)):
            adx[i] = ((adx[i - 1] * (p - 1)) + dx[i]) / p

        return float(np.clip(adx[-1], 0.0, 100.0)), float(np.clip(plus_di[-1], 0.0, 100.0)), float(np.clip(minus_di[-1], 0.0, 100.0))

    def _parabolic_sar(self, highs: List[float], lows: List[float], af_start: float = 0.02, af_max: float = 0.2) -> Tuple[float, int]:
        n = min(len(highs), len(lows))
        if n < 10:
            return 0.0, 0

        h = np.asarray(highs[-min(80, n):], dtype=np.float64)
        l = np.asarray(lows[-min(80, n):], dtype=np.float64)

        direction = 1 if (h[3] + l[3]) > (h[0] + l[0]) else -1
        af = float(af_start)
        ep = float(h[0] if direction > 0 else l[0])
        sar = float(l[0] if direction > 0 else h[0])

        for i in range(1, len(h)):
            sar = sar + af * (ep - sar)

            if direction > 0:
                sar = min(sar, float(l[max(0, i - 1)]), float(l[max(0, i - 2)]))
                if h[i] > ep:
                    ep = float(h[i])
                    af = min(af + af_start, af_max)
                if l[i] < sar:
                    direction = -1
                    sar = ep
                    ep = float(l[i])
                    af = float(af_start)
            else:
                sar = max(sar, float(h[max(0, i - 1)]), float(h[max(0, i - 2)]))
                if l[i] < ep:
                    ep = float(l[i])
                    af = min(af + af_start, af_max)
                if h[i] > sar:
                    direction = 1
                    sar = ep
                    ep = float(h[i])
                    af = float(af_start)

        return float(sar), int(direction)

    def _find_support_resistance(self, highs: List[float], lows: List[float], lookback: int) -> Tuple[List[float], List[float]]:
        n = min(len(highs), len(lows))
        if n < max(10, lookback):
            return [], []
        hh = highs[-lookback:]
        ll = lows[-lookback:]

        resistance: List[float] = []
        support: List[float] = []

        for i in range(2, len(hh) - 2):
            if hh[i] > hh[i - 1] and hh[i] > hh[i - 2] and hh[i] > hh[i + 1] and hh[i] > hh[i + 2]:
                resistance.append(float(hh[i]))
            if ll[i] < ll[i - 1] and ll[i] < ll[i - 2] and ll[i] < ll[i + 1] and ll[i] < ll[i + 2]:
                support.append(float(ll[i]))

        resistance = sorted(set(resistance), reverse=True)[:3]
        support = sorted(set(support))[:3]
        return support, resistance

    def _check_sr_proximity(self, current_price: float, support: List[float], resistance: List[float]) -> Tuple[bool, bool]:
        if not np.isfinite(current_price) or current_price <= 0:
            return False, False
        thr = float(current_price) * float(self.sr_threshold)
        near_sup = any(abs(current_price - s) <= thr for s in support)
        near_res = any(abs(current_price - r) <= thr for r in resistance)
        return bool(near_sup), bool(near_res)

    # ────────────────────────────────────────────────────────────────
    # Reason string (forensic)
    # ────────────────────────────────────────────────────────────────

    def _reason_string(
        self,
        instrument_norm: str,
        trend_label: str,
        score: float,
        adx: float,
        plus_di: float,
        minus_di: float,
        ma_alignment: int,
        mtf: Dict[str, Any],
        exhaustion: Dict[str, Any],
        decel: Dict[str, Any],
        liq_grab: Dict[str, Any],
        xasset: Dict[str, Any],
        chop: float = 0.0,
    ) -> str:
        try:
            di_txt = "↑" if plus_di > minus_di else ("↓" if minus_di > plus_di else "•")
            parts: List[str] = []
            parts.append(f"{instrument_norm} {trend_label}")
            parts.append(f"score={float(score):+.2f}")
            parts.append(f"ADX={float(adx):.1f}{di_txt}")
            parts.append(f"MAalign={int(ma_alignment)}")

            if chop > 0.0:
                parts.append(f"CHOP={float(chop):.1f}")

            if isinstance(decel, dict) and float(decel.get("available", 0.0) or 0.0) > 0.0:
                d = float(decel.get("decel", 0.0) or 0.0)
                parts.append(f"decel={d:+.3f}")

            if isinstance(exhaustion, dict):
                e = float(exhaustion.get("exhaustion_score", 0.0) or 0.0)
                if e > 0.0:
                    az = float(exhaustion.get("atr_z", 0.0) or 0.0)
                    rz = float(exhaustion.get("atr_roc_z", 0.0) or 0.0)
                    rr = float(exhaustion.get("range_ratio", 0.0) or 0.0)
                    parts.append(f"exh={e:.2f}(atrZ={az:+.1f},rocZ={rz:+.1f},rr={rr:.2f})")

            if isinstance(liq_grab, dict):
                vL = bool(liq_grab.get("veto_long", False))
                vS = bool(liq_grab.get("veto_short", False))
                if vL or vS:
                    parts.append(f"liqVeto(L={int(vL)},S={int(vS)})")

            if isinstance(mtf, dict) and bool(mtf.get("available", False)):
                parts.append(f"MTFΔ={float(mtf.get('delta_conf', 0.0) or 0.0):+.2f}")
                parts.append(f"agree={float(mtf.get('agreement', 0.5) or 0.5):.2f}")

            if isinstance(xasset, dict) and bool(xasset.get("available", False)):
                parts.append(f"XΔ={float(xasset.get('delta_conf', 0.0) or 0.0):+.2f}")

            return " | ".join(parts)
        except Exception:
            return f"{instrument_norm} {trend_label} | score={float(score):+.2f} | ADX={float(adx):.1f}"

    # ────────────────────────────────────────────────────────────────
    # Advanced market structure (volume-weighted, reduced lag)
    # ────────────────────────────────────────────────────────────────

    def _structure_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        n = len(close)
        if n < period + 2:
            return float(max(np.mean(high - low), 1e-8))
        h = high[-(period + 1):]
        l = low[-(period + 1):]
        c = close[-(period + 1):]
        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        return float(max(np.mean(tr), 1e-8))

    def _find_fractal_pivots_live(
        self,
        high: np.ndarray,
        low: np.ndarray,
        left: int = 3,
        right: int = 1,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        n = len(high)
        if n < left + right + 3:
            return [], []
        piv_hi: List[Tuple[int, float]] = []
        piv_lo: List[Tuple[int, float]] = []
        for i in range(left, n - right):
            window_h = high[i - left : i + right + 1]
            window_l = low[i - left : i + right + 1]
            hi = high[i]
            lo = low[i]
            if hi == np.max(window_h) and np.sum(window_h == hi) == 1:
                piv_hi.append((i, float(hi)))
            if lo == np.min(window_l) and np.sum(window_l == lo) == 1:
                piv_lo.append((i, float(lo)))
        return piv_hi, piv_lo

    def _cluster_levels_weighted_live(self, levels: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
        """
        levels: list of (price, weight)
        returns clusters: (weighted_mean_price, weight_sum)
        """
        if not levels:
            return []
        xs = sorted(levels, key=lambda t: t[0])
        clusters: List[Tuple[float, float]] = []
        bucket_p: List[float] = [xs[0][0]]
        bucket_w: List[float] = [xs[0][1]]

        def flush() -> None:
            wsum = float(sum(bucket_w))
            if wsum <= 1e-12:
                clusters.append((float(np.mean(bucket_p)), 0.0))
            else:
                clusters.append((float(np.dot(np.asarray(bucket_p), np.asarray(bucket_w)) / wsum), wsum))

        for p, w in xs[1:]:
            if abs(p - bucket_p[-1]) <= eps:
                bucket_p.append(float(p))
                bucket_w.append(float(w))
            else:
                flush()
                bucket_p = [float(p)]
                bucket_w = [float(w)]
        flush()
        return clusters

    def _compute_advanced_market_structure(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        current_price: float,
        opens: Optional[List[float]] = None,
        volume: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        out = {
            "structure_trend": 0.0,
            "structure_strength": 0.0,
            "bos_signal": 0.0,
            "liquidity_above": 0.0,
            "liquidity_below": 0.0,
            "order_block_bull": 0.0,
            "order_block_bear": 0.0,
        }

        n = int(min(len(highs), len(lows), len(closes)))
        if n < 80 or current_price <= 0:
            return out

        try:
            lookback = min(220, n)
            high = np.asarray(highs[-lookback:], dtype=np.float64)
            low = np.asarray(lows[-lookback:], dtype=np.float64)
            close = np.asarray(closes[-lookback:], dtype=np.float64)
            px = float(current_price)
            if not np.isfinite(px) or px <= 0:
                return out

            if volume is not None and len(volume) >= lookback:
                vol = np.asarray(volume[-lookback:], dtype=np.float64)
                vol = np.where(np.isfinite(vol) & (vol > 0), vol, 1.0)
            else:
                vol = np.ones_like(close, dtype=np.float64)

            medv = float(np.median(vol))
            if not np.isfinite(medv) or medv <= 1e-12:
                medv = 1.0
            vol_rel = np.clip(vol / medv, 0.25, 3.0)

            atr = self._structure_atr(high, low, close, period=14)
            eps_pivot_break = max(0.25 * atr, px * 0.0012)
            eps_liq = max(0.15 * atr, px * 0.0010)
            eps_ob_prox = max(0.30 * atr, px * 0.0015)

            piv_hi, piv_lo = self._find_fractal_pivots_live(
                high, low, left=int(max(1, self.structure_left)), right=int(max(1, self.structure_right))
            )

            # 1) Structure trend/strength from last pivots
            if len(piv_hi) >= 2 and len(piv_lo) >= 2:
                (_, h1), (_, h2) = piv_hi[-2], piv_hi[-1]
                (_, l1), (_, l2) = piv_lo[-2], piv_lo[-1]

                hh = h2 > h1 + 0.05 * atr
                hl = l2 > l1 + 0.05 * atr
                ll = l2 < l1 - 0.05 * atr
                lh = h2 < h1 - 0.05 * atr

                if hh and hl:
                    trend = 1.0
                elif ll and lh:
                    trend = -1.0
                else:
                    trend = 0.0

                dh = abs(h2 - h1) / max(atr, 1e-8)
                dl = abs(l2 - l1) / max(atr, 1e-8)
                strength = float(np.clip(0.25 * (dh + dl), 0.0, 1.0))

                out["structure_trend"] = float(trend)
                out["structure_strength"] = float(strength)

            # 2) BOS
            if len(piv_hi) >= 1 and len(piv_lo) >= 1:
                last_hi = piv_hi[-1][1]
                last_lo = piv_lo[-1][1]
                if close[-1] > last_hi + eps_pivot_break:
                    mag = (close[-1] - (last_hi + eps_pivot_break)) / max(atr, 1e-8)
                    out["bos_signal"] = float(np.clip(mag, 0.0, 1.0))
                elif close[-1] < last_lo - eps_pivot_break:
                    mag = ((last_lo - eps_pivot_break) - close[-1]) / max(atr, 1e-8)
                    out["bos_signal"] = float(-np.clip(mag, 0.0, 1.0))

            # 3) Liquidity pools (volume-weighted pivot clusters)
            hi_levels_w: List[Tuple[float, float]] = []
            lo_levels_w: List[Tuple[float, float]] = []
            for idx, p in piv_hi:
                w = float(vol_rel[idx]) if (0 <= idx < len(vol_rel)) else 1.0
                hi_levels_w.append((float(p), w))
            for idx, p in piv_lo:
                w = float(vol_rel[idx]) if (0 <= idx < len(vol_rel)) else 1.0
                lo_levels_w.append((float(p), w))

            hi_clusters = self._cluster_levels_weighted_live(hi_levels_w, eps_liq) if self.volume_weight_structure else [(p, 1.0) for _, p in piv_hi]
            lo_clusters = self._cluster_levels_weighted_live(lo_levels_w, eps_liq) if self.volume_weight_structure else [(p, 1.0) for _, p in piv_lo]

            def liq_score(level: float, wsum: float) -> float:
                if wsum < 1.5:
                    return 0.0
                dist = abs(level - px)
                if dist > 3.0 * atr:
                    return 0.0
                prox = float(np.exp(-dist / max(1.5 * atr, 1e-8)))
                depth = float(np.clip(wsum / 6.0, 0.0, 1.0))  # volume-weighted “wall”
                return float(np.clip(prox * depth, 0.0, 1.0))

            out["liquidity_above"] = float(max((liq_score(lvl, w) for lvl, w in hi_clusters if lvl > px), default=0.0))
            out["liquidity_below"] = float(max((liq_score(lvl, w) for lvl, w in lo_clusters if lvl < px), default=0.0))

            # 4) Order blocks (displacement-based, unmitigated) — volume-weighted
            if len(close) >= 20:
                if opens is not None and len(opens) >= lookback:
                    approx_open = np.asarray(opens[-lookback:], dtype=np.float64)
                else:
                    approx_open = np.concatenate([[close[0]], close[:-1]])

                body = np.abs(close - approx_open)
                disp_thresh = 0.9 * atr
                bull_scores: List[float] = []
                bear_scores: List[float] = []

                for i in range(2, len(close) - 1):
                    vW = float(vol_rel[i]) if self.volume_weight_structure else 1.0

                    # Bullish displacement after bearish candle
                    if (close[i] > approx_open[i]) and (body[i] >= disp_thresh) and (close[i] > close[i - 1]):
                        if close[i - 1] < approx_open[i - 1]:
                            ob_low = float(low[i - 1])
                            ob_high = float(high[i - 1])
                            post_low = float(np.min(low[i + 1:])) if (i + 1) < len(low) else float(low[-1])
                            mitigated = post_low <= ob_high
                            if not mitigated:
                                dist = 0.0
                                if px < ob_low:
                                    dist = ob_low - px
                                elif px > ob_high:
                                    dist = px - ob_high
                                base = float(np.exp(-dist / max(eps_ob_prox, 1e-8)))
                                bull_scores.append(float(np.clip(base * (0.75 + 0.25 * vW), 0.0, 1.0)))

                    # Bearish displacement after bullish candle
                    if (close[i] < approx_open[i]) and (body[i] >= disp_thresh) and (close[i] < close[i - 1]):
                        if close[i - 1] > approx_open[i - 1]:
                            ob_low = float(low[i - 1])
                            ob_high = float(high[i - 1])
                            post_high = float(np.max(high[i + 1:])) if (i + 1) < len(high) else float(high[-1])
                            mitigated = post_high >= ob_low
                            if not mitigated:
                                dist = 0.0
                                if px < ob_low:
                                    dist = ob_low - px
                                elif px > ob_high:
                                    dist = px - ob_high
                                base = float(np.exp(-dist / max(eps_ob_prox, 1e-8)))
                                bear_scores.append(float(np.clip(base * (0.75 + 0.25 * vW), 0.0, 1.0)))

                out["order_block_bull"] = float(np.clip(max(bull_scores) if bull_scores else 0.0, 0.0, 1.0))
                out["order_block_bear"] = float(np.clip(max(bear_scores) if bear_scores else 0.0, 0.0, 1.0))

        except Exception:
            return out

        return out
