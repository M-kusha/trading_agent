#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# File: modules/voting/experts/base.py
# VotingExpertBase (Framework-Grade, Contract-Strict, Forensic-Debuggable)
#
# Goals of this base:
# - Enforce a SINGLE per-instrument output contract across Trend/Theme/Momentum/Seasonality
# - Centralize market-data canonicalization (OHLCV extraction, symbol normalization)
# - Provide unified debug/trace (JSONL, levels, optional forensic-on-fail)
# - Provide hybrid indicator caching (hash + TTL) without cross-TF collisions
# - Provide position-focus mode with an expert hook for custom position semantics
# - Provide safe bus interactions, circuit breaker + resilience policies
# - Provide performance breakdown + basic health metrics for committee weighting
#
# Notes:
# - This file intentionally avoids introducing new external dependencies.
# - It assumes your framework provides:
#   - VotingModuleBase with: self.config, self.logger, self.smart_bus
#   - Optional: self.performance_tracker, self.error_pinpointer
#   - VotingBusKeys helper and confidence threshold functions in constants
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import datetime
import json
import logging
import os
import time
import traceback
from abc import abstractmethod
from collections import deque
from dataclasses import asdict, is_dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple, Callable, cast


from modules.voting.core.base import VotingModuleBase
from modules.voting.core.types import VotingProposal
from modules.voting.core.constants import (
    VotingBusKeys,
    VotingAction,
    CONFIDENCE_THRESHOLD_F,
    MIN_SIGNAL_STRENGTH_F,
    HIGH_CONFIDENCE_THRESHOLD_F,
    PRIMARY_TIMEFRAME,
)

# Developer-level fallback if per-instance config lacks debug fields.
# Recommend leaving False for production; enable via config when needed.
BASE_EXPERT_DEBUG_DEFAULT: bool = True


class VotingExpertBase(VotingModuleBase):
    """
    Framework-grade base class for all voting experts.

    Subclasses must implement:
      - _generate_expert_specific_proposal(market_data) -> dict-like
      - _calculate_expert_specific_confidence(proposal, market_data) -> float in [0,1]

    The base guarantees:
      - strict per-instrument normalization under proposal["proposals"] and proposal["per_instrument"]
      - stable top-level action/signal_strength/confidence derived from primary instrument
      - robust gating, safe bus publishing, consistent debug trace
    """

    # Strict per-instrument contract (normalized by the base)
    REQUIRED_PER_INSTRUMENT_KEYS: Dict[str, type] = {
        "action": str,             # long/short/flat/abstain/exit/tighten
        "confidence": float,       # 0..1
        "signal_strength": float,  # 0..1
        "rationale": str,          # human-readable rationale
        "instrument": str,         # canonical symbol
    }

    # Minimal proposal scaffolding to avoid downstream KeyError
    DEFAULT_PROPOSAL_SCAFFOLD: Dict[str, Any] = {
        "meta": {},
        "components": {},
        "mtf_analysis": {},  # your PPO builders often expect this to exist
    }

    # Config schema (lightweight validation, no external libs)
    CONFIG_DEFAULTS: Dict[str, Any] = {
        "instruments": ["XAUUSD"],
        "primary_timeframe": PRIMARY_TIMEFRAME,
        "primary_symbol": "XAUUSD",
        "use_forming_bar": False,

        # gating
        "max_signal_strength": 1.0,
        "min_signal_strength": 0.10,
        "action_history_len": 100,

        # circuit breaker
        "max_consecutive_errors": 5,
        "circuit_reset_seconds": 60.0,

        # caching
        "indicator_cache_ttl": 5.0,          # seconds
        "indicator_cache_strategy": "hash",  # "hash" | "ttl" | "hybrid"

        # debug
        "debug_enabled": BASE_EXPERT_DEBUG_DEFAULT,
        "debug_level": "standard",           # "light" | "standard" | "forensic"
        "debug_forensic_on_fail": True,      # capture extra details only on failing ticks
        "debug_dir": "logs/voting_experts",
        "debug_max_bytes": 10 * 1024 * 1024,
        "debug_backup_count": 5,
        "debug_flush": True,
        "debug_include_market_data": True,
        "debug_include_outputs": True,

        # resilience policy (base-level defaults)
        "error_resilience": {
            "data_missing": "degrade_to_flat",       # degrade_to_flat | use_last_good | skip_tick
            "indicator_failure": "fallback_simple",  # fallback_simple | degrade_to_flat
            "external_api_failure": "use_cached",    # use_cached | degrade_to_flat
            "unknown": "degrade_to_flat",
        },

        # features (simple feature flags)
        "features": {
            "forming_bar": False,
            "volume_weighting": True,
            "adaptive_learning": True,
        },
    }

    # Defensive defaults (so a partially-initialized expert can't crash on attribute access)
    _process_seq: int = 0
    _debug_enabled: bool = False
    _debug_level: str = "standard"
    _debug_forensic_on_fail: bool = False
    _debug_include_market_data: bool = False
    _debug_include_outputs: bool = False
    _debug_flush: bool = False
    _debug_logger: Optional[logging.Logger] = None
    _debug_last_forensic_blob: Optional[Dict[str, Any]] = None

    # -------------------------------
    # Template methods for subclasses
    # -------------------------------

    @abstractmethod
    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def _calculate_expert_specific_confidence(
        self, proposal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> float:
        raise NotImplementedError

    # Optional subclass hooks
    def _expert_specific_init(self) -> None:
        """Override for expert-specific initialization."""
        pass

    def _expert_specific_position_evaluation(self, proposal: Dict[str, Any], position_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override if your expert has special position-management semantics.
        Default: passthrough.
        """
        return proposal

    def _fallback_simple_proposal(self, market_data: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Override if you want a deterministic/simple fallback when indicators fail.
        Default: abstain.
        """
        _ = market_data
        return {"action": "abstain", "signal_strength": 0.0, "reason": reason}

    # -------------------------------
    # Initialization
    # -------------------------------

    def _module_specific_init(self) -> None:
        # Process counter for traceability
        self._process_seq = 0

        # Config normalization (non-throwing)
        self._cfg_apply_defaults()

        # Action history
        self.action_history: deque = deque(maxlen=int(self._cfg_int("action_history_len", 100, 1, 10_000)))

        # Market context (base-level)
        self.market_context: Dict[str, Any] = {
            "regime": "unknown",
            "volatility_level": "medium",
            "trend_strength": 0.0,
            "session": "unknown",
        }

        # Analytics (base-level)
        self.expert_analytics: Dict[str, Any] = {
            "total_actions": 0,
            "successful_actions": 0,
            "avg_confidence": 0.5,
            "last_action": None,
            "last_action_time": None,
        }

        # Strength limits
        self.max_signal_strength = float(self._cfg_float("max_signal_strength", 1.0, 0.0, 10.0))
        self.min_signal_strength = float(self._cfg_float("min_signal_strength", 0.10, 0.0, 1.0))

        # Circuit breaker
        self._consecutive_errors = 0
        self._circuit_open = False
        self._circuit_open_until: Optional[datetime.datetime] = None
        self._circuit_open_reason: Optional[str] = None
        self._max_consecutive_errors = int(self._cfg_int("max_consecutive_errors", 5, 1, 10_000))
        self._circuit_reset_seconds = float(self._cfg_float("circuit_reset_seconds", 60.0, 1.0, 3600.0))

        # Last good output (for "use_last_good")
        self._last_good_proposal: Optional[Dict[str, Any]] = None
        self._last_good_confidence: Optional[float] = None

        # Indicator cache: key = "INST|TF"
        self._indicator_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_seconds = float(self._cfg_float("indicator_cache_ttl", 5.0, 0.0, 60.0))
        self._cache_strategy = str(self.config.get("indicator_cache_strategy", "hash") or "hash").lower().strip()

        # Performance breakdown
        self._perf: Dict[str, float] = {
            "data_fetch_ms": 0.0,
            "indicator_ms": 0.0,
            "scoring_ms": 0.0,
            "filtering_ms": 0.0,
            "proposal_build_ms": 0.0,
            "postprocess_ms": 0.0,
            "bus_publish_ms": 0.0,
            "total_ms": 0.0,
        }

        # Per-instrument state store (optional)
        self.state_store: Dict[str, Dict[str, Any]] = {}

        # Debug logging
        self._init_debug_logger()

        # Subclass init
        self._expert_specific_init()

        # Baseline publish
        self._publish_baseline_keys()

    # -------------------------------
    # Config helpers (safe, clamped)
    # -------------------------------

    def _cfg_apply_defaults(self) -> None:
        if not isinstance(getattr(self, "config", None), dict):
            self.config = {}

        for k, v in self.CONFIG_DEFAULTS.items():
            if k not in self.config:
                # deep-ish copy for dicts
                if isinstance(v, dict):
                    self.config[k] = dict(v)
                elif isinstance(v, list):
                    self.config[k] = list(v)
                else:
                    self.config[k] = v

        # normalize instruments list
        insts = self.config.get("instruments", ["XAUUSD"])
        if isinstance(insts, str):
            insts = [insts]
        if not isinstance(insts, list):
            insts = ["XAUUSD"]
        self.config["instruments"] = [self._canonicalize_instrument(x) for x in insts if str(x).strip()] or ["XAUUSD"]

        # normalize primary symbol
        ps = self.config.get("primary_symbol") or self.config.get("primary_instrument") or "XAUUSD"
        self.config["primary_symbol"] = self._canonicalize_instrument(str(ps))

        # normalize timeframe
        tf = self.config.get("primary_timeframe", PRIMARY_TIMEFRAME) or PRIMARY_TIMEFRAME
        self.config["primary_timeframe"] = str(tf).upper().strip()

        # features sanity
        feats = self.config.get("features", {})
        if not isinstance(feats, dict):
            feats = {}
        self.config["features"] = feats

        # resilience sanity
        er = self.config.get("error_resilience", {})
        if not isinstance(er, dict):
            er = dict(self.CONFIG_DEFAULTS["error_resilience"])
        self.config["error_resilience"] = er

    def feature_enabled(self, name: str) -> bool:
        feats = self.config.get("features", {})
        if not isinstance(feats, dict):
            return False
        return bool(feats.get(name, False))

    def _cfg_float(self, key: str, default: float, min_v: float, max_v: float) -> float:
        try:
            v = float(self.config.get(key, default))
        except Exception:
            v = float(default)
        return max(min_v, min(max_v, v))

    def _cfg_int(self, key: str, default: int, min_v: int, max_v: int) -> int:
        try:
            v = int(self.config.get(key, default))
        except Exception:
            v = int(default)
        return max(min_v, min(max_v, v))

    # -------------------------------
    # Debug logging (JSONL, levels)
    # -------------------------------

    def _init_debug_logger(self) -> None:
        self._debug_enabled = bool(self.config.get("debug_enabled", BASE_EXPERT_DEBUG_DEFAULT))
        self._debug_level = str(self.config.get("debug_level", "standard") or "standard").lower().strip()
        self._debug_forensic_on_fail = bool(self.config.get("debug_forensic_on_fail", True))

        self._debug_include_market_data = bool(self.config.get("debug_include_market_data", True))
        self._debug_include_outputs = bool(self.config.get("debug_include_outputs", True))
        self._debug_flush = bool(self.config.get("debug_flush", True))

        self._debug_logger: Optional[logging.Logger] = None
        self._debug_last_forensic_blob: Optional[Dict[str, Any]] = None  # captured only when needed

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

        self._debug_log(
            event="debug_logger_initialized",
            level="light",
            debug_dir=debug_dir,
            filename=filename,
            debug_level=self._debug_level,
            forensic_on_fail=self._debug_forensic_on_fail,
        )

    def _debug_level_allows(self, msg_level: str) -> bool:
        # light < standard < forensic
        order = {"light": 0, "standard": 1, "forensic": 2}
        cur = order.get(self._debug_level, 1)
        want = order.get(msg_level, 1)
        return want <= cur

    def _debug_log(self, event: str, level: str = "standard", **payload: Any) -> None:
        if not self._debug_enabled or self._debug_logger is None:
            return
        if not self._debug_level_allows(level):
            return

        record = {
            "ts": time.time(),
            "iso": datetime.datetime.now().isoformat(),
            "expert": self.__class__.__name__,
            "event": str(event),
            "level": str(level),
            "seq": int(getattr(self, "_process_seq", 0)),
            "payload": payload,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception:
            line = json.dumps(
                {
                    "ts": record.get("ts"),
                    "iso": record.get("iso"),
                    "expert": record.get("expert"),
                    "event": record.get("event"),
                    "level": record.get("level"),
                    "seq": record.get("seq"),
                    "payload": str(payload),
                },
                ensure_ascii=False,
                default=str,
            )

        self._debug_logger.debug(line)
        if self._debug_flush:
            for h in self._debug_logger.handlers:
                try:
                    h.flush()
                except Exception:
                    pass

    def _capture_forensic_blob(self, **blob: Any) -> None:
        if not self._debug_enabled or not self._debug_forensic_on_fail:
            return
        # keep bounded: only store the latest blob
        self._debug_last_forensic_blob = blob

    def _flush_forensic_blob(self, reason: str) -> None:
        if not self._debug_enabled or not self._debug_forensic_on_fail:
            return
        if not self._debug_last_forensic_blob:
            return
        self._debug_log(event="forensic_blob_flush", level="forensic", reason=reason, **self._debug_last_forensic_blob)
        self._debug_last_forensic_blob = None

    # -------------------------------
    # Safe bus utilities
    # -------------------------------

    def _safe_bus_get(self, key: str, module: str, default: Any = None) -> Any:
        _ = module  # kept for backward compatibility with legacy call sites
        return self.bus_get(key, default=default)

    def _safe_bus_set(self, key: str, value: Any, module: str, **kwargs: Any) -> bool:
        _ = module  # kept for backward compatibility with legacy call sites
        thesis = str(kwargs.pop("thesis", "") or "")
        return self.bus_set(key, value, thesis=thesis, **kwargs)

    # -------------------------------
    # Normalization helpers
    # -------------------------------

    def _canonicalize_instrument(self, instrument: str) -> str:
        # Single-source canonicalization lives in VotingModuleBase.canon() / voting.core.constants.normalize_instrument().
        return self.canon(instrument)

    def _normalize_action(self, action_like: Any, proposal: Optional[Dict[str, Any]] = None) -> str:
        raw = str(action_like if action_like is not None else "abstain").lower().strip()

        # Special-case: "scale_up" needs position context to resolve into direction.
        if raw == "scale_up":
            ctx = proposal or {}
            side = ctx.get("side", ctx.get("direction", ctx.get("position_side", 0)))
            try:
                side_i = int(side)
                if side_i > 0:
                    return "long"
                if side_i < 0:
                    return "short"
            except Exception:
                pass
            return "flat"

        # Canonical parsing is centralized in VotingAction. We keep "flat" as the
        # neutral label for expert outputs, but normalize everything else.
        action = VotingAction.from_string(raw)
        if action == VotingAction.HOLD:
            return "flat"
        return action.value

    def _coerce_proposal_to_dict(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return dict(obj)

        if isinstance(obj, VotingProposal):
            try:
                if is_dataclass(obj) and not isinstance(obj, type):
                    d = asdict(obj)
                    return dict(d) if isinstance(d, dict) else {"action": "abstain", "reason": "proposal_asdict_invalid"}
                if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
                    d = obj.to_dict()
                    return dict(d) if isinstance(d, dict) else {"action": "abstain", "reason": "proposal_to_dict_invalid"}
            except Exception:
                pass
            try:
                return dict(getattr(obj, "__dict__", {}))
            except Exception:
                return {"action": "abstain", "reason": "proposal_coerce_failed"}

        if is_dataclass(obj) and not isinstance(obj, type):
            try:
                d = asdict(obj)
                return dict(d) if isinstance(d, dict) else {"action": "abstain", "reason": "proposal_asdict_invalid"}
            except Exception:
                return {"action": "abstain", "reason": "proposal_asdict_failed"}

        try:
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict) and d:
                return dict(d)
        except Exception:
            pass

        return {"action": "abstain", "reason": "invalid_proposal_type"}

    # -------------------------------
    # Indicator caching (hash/ttl/hybrid)
    # -------------------------------

    def _compute_price_hash(self, prices: list) -> str:
        if not prices:
            return ""
        try:
            last_5 = prices[-5:] if len(prices) >= 5 else prices
            last_f = float(prices[-1])
            sum_f = 0.0
            for x in last_5:
                try:
                    sum_f += float(x)
                except Exception:
                    continue
            return f"{len(prices)}:{last_f:.6f}:{sum_f:.6f}"
        except Exception:
            return f"{len(prices)}:{str(prices[-1])}"

    def _cache_key(self, instrument: str, timeframe: Optional[str] = None) -> str:
        inst = self._canonicalize_instrument(instrument)
        tf = str(timeframe or "").upper().strip()
        return f"{inst}|{tf}" if tf else inst

    def _get_cached_blob(self, instrument: str, prices: list, timeframe: Optional[str]) -> Optional[Dict[str, Any]]:
        key = self._cache_key(instrument, timeframe)
        cache_entry = self._indicator_cache.get(key)
        if not cache_entry:
            return None

        now = time.time()
        cached_time = float(cache_entry.get("timestamp", 0.0) or 0.0)
        if self._cache_ttl_seconds > 0 and (now - cached_time) > self._cache_ttl_seconds:
            return None

        if self._cache_strategy in ("ttl",):
            return cache_entry.get("results")

        # hash or hybrid: require hash match
        price_hash = self._compute_price_hash(prices)
        if cache_entry.get("hash") != price_hash:
            return None
        return cache_entry.get("results")

    def _set_cached_blob(self, instrument: str, prices: list, results: Dict[str, Any], timeframe: Optional[str]) -> None:
        key = self._cache_key(instrument, timeframe)
        self._indicator_cache[key] = {
            "hash": self._compute_price_hash(prices),
            "results": results,
            "timestamp": time.time(),
        }

    # Backwards-compatible wrappers for expert implementations that expect
    # indicator-specific cache helpers. These adapt to the generic blob cache
    # used by the base class.
    def _get_cached_indicators(self, instrument: str, prices: list, timeframe: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            return self._get_cached_blob(instrument, prices, timeframe=timeframe)
        except Exception:
            return None

    def _set_cached_indicators(self, instrument: str, prices: list, results: Dict[str, Any], timeframe: Optional[str] = None) -> None:
        try:
            self._set_cached_blob(instrument, prices, results, timeframe=timeframe)
        except Exception:
            return

    # -------------------------------
    # Market data canonicalization
    # -------------------------------

    def _get_canonical_ohlcv(
        self,
        instrument: str,
        timeframe: str,
        include_forming_bar: Optional[bool] = None,
        market_data_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Centralized OHLCV retrieval. This method tries multiple bus keys that
        commonly exist in your system to avoid expert-by-expert duplication.

        Returns dict: {open:[...], high:[...], low:[...], close:[...], volume:[...]} (some keys may be missing)
        """
        name = self.__class__.__name__
        inst = self._canonicalize_instrument(instrument)
        tf = str(timeframe).upper().strip()

        if include_forming_bar is None:
            include_forming_bar = bool(self.config.get("use_forming_bar", False)) or self.feature_enabled("forming_bar")

        ohlcv: Dict[str, Any] = {}
        hint = market_data_hint if isinstance(market_data_hint, dict) else {}

        # 1) Prefer direct hint if provided
        hint_ohlcv = hint.get("ohlcv")
        if isinstance(hint_ohlcv, dict):
            for k in ("open", "high", "low", "close", "volume"):
                seq = hint_ohlcv.get(k)
                if isinstance(seq, (list, tuple)) and seq:
                    ohlcv[k] = list(seq)
            if ohlcv:
                return self._maybe_stitch_forming_bar(ohlcv, hint, include_forming_bar)

        # 2) Try multi_timeframe_data (common in your live/training plumbing)
        mtd = self._safe_bus_get("multi_timeframe_data", name, default=None)
        if isinstance(mtd, dict) and inst in mtd and isinstance(mtd.get(inst), dict):
            block = mtd.get(inst) or {}
            if isinstance(block.get(tf), dict):
                tf_block = block.get(tf) or {}
                for k in ("open", "high", "low", "close", "volume"):
                    seq = tf_block.get(k)
                    if isinstance(seq, (list, tuple)) and seq:
                        ohlcv[k] = list(seq)
                if ohlcv:
                    return self._maybe_stitch_forming_bar(ohlcv, hint, include_forming_bar)

        # 3) Try historical_prices
        hist = self._safe_bus_get("historical_prices", name, default=None)
        if isinstance(hist, dict) and inst in hist and isinstance(hist.get(inst), dict):
            inst_block = hist.get(inst) or {}
            tf_block = inst_block.get(tf)
            if not isinstance(tf_block, dict):
                # fallback TF search if requested TF missing
                for alt in ("M15", "H1", "H4", "D1"):
                    if isinstance(inst_block.get(alt), dict):
                        tf_block = inst_block.get(alt)
                        break
            if isinstance(tf_block, dict):
                for k in ("open", "high", "low", "close", "volume"):
                    seq = tf_block.get(k)
                    if isinstance(seq, (list, tuple)) and seq:
                        ohlcv[k] = list(seq)
                if ohlcv:
                    return self._maybe_stitch_forming_bar(ohlcv, hint, include_forming_bar)

        # 4) Nothing found
        return {}

    def _maybe_stitch_forming_bar(self, ohlcv: Dict[str, Any], hint: Dict[str, Any], include_forming_bar: bool) -> Dict[str, Any]:
        """
        If forming bars are enabled and a recognizable forming bar exists in market_data hint,
        append it as the last element (without duplicating if already present).
        """
        if not include_forming_bar:
            return ohlcv

        # We do not assume your forming-bar schema. We support a conservative, non-breaking pattern:
        # - hint["forming_bar"] = {"open":..,"high":..,"low":..,"close":..,"volume":..}
        fb = hint.get("forming_bar")
        if not isinstance(fb, dict):
            fb = hint.get("forming_ohlcv")

        if not isinstance(fb, dict):
            return ohlcv

        # append if types are sane and series exist
        stitched = dict(ohlcv)
        for k in ("open", "high", "low", "close", "volume"):
            if k not in stitched or not isinstance(stitched.get(k), list):
                continue
            if k in fb:
                try:
                    stitched[k].append(float(fb[k]))
                except Exception:
                    # ignore malformed forming data
                    pass
        return stitched

    def _build_market_data(self, raw_market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Canonical market_data snapshot for all experts.
        """
        md: Dict[str, Any] = dict(raw_market_data) if isinstance(raw_market_data, dict) else {}

        primary_symbol = self._canonicalize_instrument(
            str(md.get("primary_symbol") or self.config.get("primary_symbol") or "XAUUSD")
        )
        primary_tf = str(md.get("primary_timeframe") or self.config.get("primary_timeframe") or PRIMARY_TIMEFRAME).upper().strip()

        md["primary_symbol"] = primary_symbol
        md["primary_timeframe"] = primary_tf
        md.setdefault("instrument", primary_symbol)
        md.setdefault("timeframe", primary_tf)

        ohlcv = md.get("ohlcv")
        if not isinstance(ohlcv, dict) or not ohlcv:
            ohlcv = self._get_canonical_ohlcv(primary_symbol, primary_tf, include_forming_bar=None, market_data_hint=md)
            if ohlcv:
                md["ohlcv"] = ohlcv

        # canonical close_prices/prices
        if "close_prices" not in md:
            if isinstance(md.get("ohlcv"), dict) and isinstance(md["ohlcv"].get("close"), list):
                md["close_prices"] = md["ohlcv"]["close"]

        if "prices" not in md:
            closes = md.get("close_prices")
            if isinstance(closes, list) and closes:
                md["prices"] = closes
            else:
                # last-resort: bus "prices" might be scalar map
                prices_bus = self._safe_bus_get("prices", self.__class__.__name__, default=None)
                if isinstance(prices_bus, dict) and prices_bus:
                    val = prices_bus.get(primary_symbol, next(iter(prices_bus.values())))
                    try:
                        md["prices"] = [float(val)]
                        md["prices_is_scalar_series"] = True
                    except Exception:
                        md["prices"] = []

        # current_price
        if "current_price" not in md:
            price_data = self._safe_bus_get("price_data", self.__class__.__name__, default=None)
            value = None
            if isinstance(price_data, dict) and price_data:
                sym_block = price_data.get(primary_symbol)
                if not isinstance(sym_block, dict):
                    sym_block = next(iter(price_data.values()), None)
                if isinstance(sym_block, dict):
                    value = sym_block.get("last") or sym_block.get("close")
            if value is None:
                prices = md.get("prices")
                if isinstance(prices, list) and prices:
                    value = prices[-1]
            if value is not None:
                try:
                    md["current_price"] = float(value)
                except Exception:
                    pass

        # last OHLCV scalars for convenience
        if isinstance(md.get("ohlcv"), dict):
            for k in ("volume", "high", "low"):
                if k not in md:
                    seq = md["ohlcv"].get(k)
                    if isinstance(seq, list) and seq:
                        try:
                            md[k] = float(seq[-1])
                        except Exception:
                            pass

        return md

    # -------------------------------
    # Per-instrument contract enforcement
    # -------------------------------

    def _extract_per_instrument_map(self, proposal: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Accepts multiple legacy shapes and normalizes into:
          { "XAUUSD": { ...contract keys... }, ... }
        """
        candidates = [
            proposal.get("proposals"),
            proposal.get("per_instrument"),
            proposal.get("per_instrument_votes"),
        ]
        raw: Optional[Dict[str, Any]] = None
        for c in candidates:
            if isinstance(c, dict) and c:
                raw = c
                break
        if not isinstance(raw, dict):
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            inst = self._canonicalize_instrument(str(k))
            out[inst] = self._normalize_per_instrument_block(inst, v)

        return out

    def _normalize_per_instrument_block(self, instrument: str, block: Dict[str, Any]) -> Dict[str, Any]:
        inst = self._canonicalize_instrument(instrument)

        action = self._normalize_action(block.get("action", block.get("vote", block.get("direction", "abstain"))), proposal=block)
        conf = block.get("confidence", block.get("conf", block.get("probability", 0.0)))
        sig = block.get("signal_strength", block.get("strength", block.get("magnitude", 0.0)))
        rationale = block.get("rationale", block.get("reason", block.get("thesis", "")))

        try:
            conf_f = max(0.0, min(1.0, float(conf or 0.0)))
        except Exception:
            conf_f = 0.0
        try:
            sig_f = max(0.0, min(1.0, float(sig or 0.0)))
        except Exception:
            sig_f = 0.0

        normalized = dict(block)
        normalized["instrument"] = inst
        normalized["action"] = action
        normalized["confidence"] = conf_f
        normalized["signal_strength"] = sig_f
        normalized["rationale"] = str(rationale or "")

        # Ensure required keys exist (even if empty/default)
        for key, typ in self.REQUIRED_PER_INSTRUMENT_KEYS.items():
            if key not in normalized:
                normalized[key] = typ()  # type: ignore[misc]

        return normalized

    def _ensure_proposal_scaffold(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in self.DEFAULT_PROPOSAL_SCAFFOLD.items():
            if k not in proposal:
                proposal[k] = {} if isinstance(v, dict) else v
            elif isinstance(v, dict) and not isinstance(proposal.get(k), dict):
                proposal[k] = {}
        return proposal

    def _derive_top_level_from_primary(self, proposal: Dict[str, Any], per_inst: Dict[str, Dict[str, Any]], primary_symbol: str) -> Dict[str, Any]:
        if not per_inst:
            return proposal

        primary_norm = self._canonicalize_instrument(primary_symbol)
        block = per_inst.get(primary_norm)
        if block is None:
            # fallback to first instrument deterministically
            block = per_inst.get(next(iter(per_inst.keys())))

        if not isinstance(block, dict):
            return proposal

        # Only override if missing or weakly specified
        top_action = str(proposal.get("action", "abstain")).lower().strip()
        if top_action in ("", "abstain", "none", "skip"):
            proposal["action"] = block.get("action", proposal.get("action", "abstain"))

        if "signal_strength" not in proposal:
            proposal["signal_strength"] = block.get("signal_strength", 0.0)
        if "confidence" not in proposal:
            proposal["confidence"] = block.get("confidence", 0.0)

        proposal["top_level_source_instrument"] = block.get("instrument", primary_norm)
        return proposal

    def _validate_and_normalize_expert_output(self, proposal: Dict[str, Any], primary_symbol: str) -> Dict[str, Any]:
        proposal = dict(proposal)
        proposal = self._ensure_proposal_scaffold(proposal)

        # Normalize top-level action early
        proposal["action"] = self._normalize_action(proposal.get("action", "abstain"), proposal=proposal)

        per_inst = self._extract_per_instrument_map(proposal)
        if per_inst:
            # Write back canonical keys to end all downstream ambiguity
            proposal["proposals"] = per_inst
            proposal["per_instrument"] = per_inst
            proposal = self._derive_top_level_from_primary(proposal, per_inst, primary_symbol)

        # Ensure numeric fields exist
        if "signal_strength" not in proposal:
            proposal["signal_strength"] = 0.0
        if "confidence" not in proposal:
            proposal["confidence"] = 0.0

        # Clamp numeric fields
        try:
            proposal["signal_strength"] = max(0.0, min(1.0, float(proposal.get("signal_strength", 0.0) or 0.0)))
        except Exception:
            proposal["signal_strength"] = 0.0
        try:
            proposal["confidence"] = max(0.0, min(1.0, float(proposal.get("confidence", 0.0) or 0.0)))
        except Exception:
            proposal["confidence"] = 0.0

        return proposal

    # -------------------------------
    # Circuit breaker + resilience
    # -------------------------------

    def _check_circuit_breaker(self) -> bool:
        if not self._circuit_open:
            return False
        now = datetime.datetime.now()
        if self._circuit_open_until and now >= self._circuit_open_until:
            self._circuit_open = False
            self._consecutive_errors = 0
            self._circuit_open_until = None
            self._circuit_open_reason = None
            try:
                self.logger.info(f"[{self.__class__.__name__}] Circuit breaker reset")
            except Exception:
                pass
            self._debug_log(event="circuit_reset", level="light")
            return False
        return True

    def _classify_error(self, e: Exception) -> str:
        # Conservative classification only; avoids guessing domain specifics.
        if isinstance(e, (KeyError,)):
            return "data_missing"
        if isinstance(e, (ValueError, TypeError)):
            return "indicator_failure"
        msg = str(e).lower()
        if "timeout" in msg or "connection" in msg or "api" in msg:
            return "external_api_failure"
        return "unknown"

    def _policy(self, cls: str) -> str:
        er = self.config.get("error_resilience", {})
        if not isinstance(er, dict):
            return "degrade_to_flat"
        return str(er.get(cls, er.get("unknown", "degrade_to_flat"))).lower().strip()

    def _record_error(self, error: Exception) -> None:
        self._consecutive_errors += 1
        self._circuit_open_reason = str(error)

        self._debug_log(
            event="error_recorded",
            level="standard",
            consecutive_errors=self._consecutive_errors,
            max_consecutive_errors=self._max_consecutive_errors,
            error=str(error),
            traceback=traceback.format_exc(),
        )

        if self._consecutive_errors >= self._max_consecutive_errors:
            self._circuit_open = True
            self._circuit_open_until = datetime.datetime.now() + datetime.timedelta(seconds=self._circuit_reset_seconds)
            try:
                self.logger.warning(
                    f"[{self.__class__.__name__}] Circuit breaker OPEN until {self._circuit_open_until.isoformat()}"
                )
            except Exception:
                pass
            self._debug_log(
                event="circuit_opened",
                level="light",
                open_until=self._circuit_open_until.isoformat(),
                reason=self._circuit_open_reason,
            )

    def _record_success(self) -> None:
        if self._consecutive_errors != 0:
            self._debug_log(event="error_streak_cleared", level="light", previous_streak=self._consecutive_errors)
        self._consecutive_errors = 0

    # -------------------------------
    # Position focus mode
    # -------------------------------

    def _get_position_focus_context(self) -> Optional[Dict[str, Any]]:
        ctx = self._safe_bus_get("position_focus_context", self.__class__.__name__, default=None)
        if not isinstance(ctx, dict):
            return None
        if not bool(ctx.get("focus_mode_active", False)):
            return None
        return ctx

    def _get_position_for_instrument(self, instrument: str) -> Optional[Dict[str, Any]]:
        ctx = self._get_position_focus_context()
        if not ctx:
            return None
        positions = ctx.get("positions", {})
        if not isinstance(positions, dict) or not positions:
            return None

        inst_norm = self._canonicalize_instrument(instrument)
        if inst_norm in positions and isinstance(positions.get(inst_norm), dict):
            return cast(Dict[str, Any], positions.get(inst_norm))

        for k, v in positions.items():
            if isinstance(v, dict) and self._canonicalize_instrument(str(k)) == inst_norm:
                return cast(Dict[str, Any], v)

        return None

    def _evaluate_single_instrument_against_position(
        self,
        inst_proposal: Dict[str, Any],
        instrument: str,
        position: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, bool]:
        position_side = int(position.get("side", 0) or 0)
        position_pnl = float(position.get("unrealized_pnl", position.get("pnl", 0.0)) or 0.0)
        position_entry = float(position.get("entry_price", 0.0) or 0.0)

        action_norm = self._normalize_action(inst_proposal.get("action", "abstain"), proposal=inst_proposal)
        original_conf = float(inst_proposal.get("confidence", inst_proposal.get("signal_strength", 0.5)) or 0.5)
        original_conf = max(0.0, min(1.0, original_conf))

        supports = True
        evaluation = "neutral"

        if position_side > 0:
            if action_norm in ("short", "exit"):
                supports = False
                evaluation = "threatens_long"
            else:
                supports = True
                evaluation = "supports_long" if action_norm == "long" else "neutral_for_long"
        elif position_side < 0:
            if action_norm in ("long", "exit"):
                supports = False
                evaluation = "threatens_short"
            else:
                supports = True
                evaluation = "supports_short" if action_norm == "short" else "neutral_for_short"
        else:
            supports = True
            evaluation = "position_side_unknown"

        modified = dict(inst_proposal)
        modified["position_management_mode"] = True
        modified["position_instrument"] = instrument
        modified["supports_position"] = supports
        modified["position_evaluation"] = evaluation
        modified["position_side"] = position_side
        modified["original_action"] = str(inst_proposal.get("action", "abstain")).lower()
        modified["position_entry_price"] = position_entry
        modified["position_unrealized_pnl"] = position_pnl

        # Remap into position-management actions (committee can interpret)
        if supports:
            modified["action"] = "hold"
        else:
            if original_conf > 0.70:
                modified["action"] = "exit"
            elif original_conf > 0.50:
                modified["action"] = "tighten"
            else:
                modified["action"] = "hold"

        # PnL-aware confidence adjustment
        conf = original_conf
        if position_pnl > 0 and not supports:
            conf *= 0.80
        elif position_pnl < 0 and not supports:
            conf *= 1.20
        conf = max(0.0, min(1.0, conf))
        modified["confidence"] = conf

        return modified, conf, supports

    # -------------------------------
    # Market context (lightweight)
    # -------------------------------

    def _update_market_context(self, market_data: Dict[str, Any]) -> None:
        if not isinstance(market_data, dict):
            return

        name = self.__class__.__name__
        self.market_context["regime"] = (
            self._safe_bus_get("market_regime", name, default="unknown") or market_data.get("market_regime", "unknown")
        )
        self.market_context["volatility_level"] = (
            market_data.get("volatility_level") or market_data.get("volatility", "medium")
        )
        try:
            self.market_context["trend_strength"] = float(market_data.get("trend_strength", 0.0) or 0.0)
        except Exception:
            self.market_context["trend_strength"] = 0.0

        session = (
            self._safe_bus_get("session_canonical", name, default=None)
            or market_data.get("current_session")
            or market_data.get("session_type", "unknown")
        )
        self.market_context["session"] = str(session).lower()

    # -------------------------------
    # Analytics / baseline publish
    # -------------------------------

    def _record_action(self, action: str, confidence: float, proposal: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "confidence": float(confidence),
            "regime": self.market_context.get("regime", "unknown"),
            "session": self.market_context.get("session", "unknown"),
            "proposal_summary": {
                "action": proposal.get("action"),
                "signal_strength": proposal.get("signal_strength", 0.0),
            },
        }
        self.action_history.append(record)

        self.expert_analytics["total_actions"] += 1
        self.expert_analytics["last_action"] = action
        self.expert_analytics["last_action_time"] = record["timestamp"]

        n = int(self.expert_analytics["total_actions"])
        old_avg = float(self.expert_analytics.get("avg_confidence", 0.5) or 0.5)
        self.expert_analytics["avg_confidence"] = (old_avg * (n - 1) + float(confidence)) / max(1, n)

    def _publish_baseline_keys(self) -> None:
        name = self.__class__.__name__
        proposal_key = VotingBusKeys.expert_proposal(name)
        confidence_key = VotingBusKeys.expert_confidence(name)
        per_instrument_key = f"{name}_per_instrument_votes"

        baseline_proposal = {
            "action": "abstain",
            "signal_strength": 0.0,
            "position_size": 0.0,
            "reason": "baseline",
        }
        baseline_proposal = self._ensure_proposal_scaffold(baseline_proposal)

        self._safe_bus_set(proposal_key, baseline_proposal, module=name, thesis=f"Baseline proposal for {name}")
        self._safe_bus_set(confidence_key, 0.1, module=name, thesis=f"{name} baseline confidence: 10%")
        self._safe_bus_set(per_instrument_key, {}, module=name, thesis=f"{name} baseline per-instrument votes")
        self._safe_bus_set(f"{name}_market_context", dict(self.market_context), module=name, thesis=f"{name} market context")
        self._safe_bus_set(f"{name}_analytics", dict(self.expert_analytics), module=name, thesis=f"{name} analytics")

        self._debug_log(event="baseline_published", level="light", proposal_key=proposal_key, confidence_key=confidence_key)

    # -------------------------------
    # Post-processing / gating (pipeline)
    # -------------------------------

    def _postprocess_proposal_for_voting(self, proposal: Dict[str, Any], confidence: float) -> Tuple[Dict[str, Any], float]:
        action_norm = self._normalize_action(proposal.get("action", "abstain"), proposal=proposal)

        sig = proposal.get("signal_strength", proposal.get("magnitude", 0.0))
        try:
            sig_f = float(sig or 0.0)
        except Exception:
            sig_f = 0.0
        sig_f = max(0.0, min(1.0, sig_f))
        proposal["signal_strength"] = sig_f

        try:
            ps = float(proposal.get("position_size", sig_f) or 0.0)
        except Exception:
            ps = 0.0
        ps = max(0.0, min(self.max_signal_strength, ps))
        proposal["position_size"] = ps

        min_strength = float(MIN_SIGNAL_STRENGTH_F())
        conf_floor = float(CONFIDENCE_THRESHOLD_F())
        high_conf = float(HIGH_CONFIDENCE_THRESHOLD_F())

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        is_directional = action_norm in ("long", "short")
        is_exit = action_norm == "exit"
        is_tighten = action_norm == "tighten"
        is_flat = action_norm in ("flat", "hold")
        is_abstain = action_norm in ("abstain", "none", "skip")

        if is_directional:
            if sig_f < min_strength or confidence < conf_floor:
                proposal.setdefault("raw_action", proposal.get("action"))
                proposal.setdefault("raw_signal_strength", sig_f)
                proposal.setdefault("raw_confidence", confidence)

                proposal["action"] = "flat"
                proposal["signal_strength"] = min(sig_f, min_strength * 0.5)
                proposal["position_size"] = min(proposal["signal_strength"], self.max_signal_strength)
                confidence = max(0.15, min(confidence, conf_floor * 0.9))
            else:
                proposal["action"] = action_norm
                proposal["signal_strength"] = max(sig_f, min_strength)
                proposal["position_size"] = min(proposal["signal_strength"], self.max_signal_strength)
                confidence = max(conf_floor, min(confidence, high_conf))

        elif is_exit or is_tighten:
            proposal["action"] = action_norm
            effective_min = min_strength * 0.75
            proposal["signal_strength"] = max(sig_f, effective_min)
            proposal["position_size"] = min(proposal["signal_strength"], self.max_signal_strength)
            confidence = max(conf_floor * 0.5, min(confidence, 1.0))

        elif is_flat:
            proposal["action"] = "flat"
            proposal["signal_strength"] = min(sig_f, min_strength * 0.5)
            proposal["position_size"] = min(proposal["signal_strength"], self.max_signal_strength)
            confidence = min(confidence, conf_floor * 0.9)

        elif is_abstain:
            proposal["action"] = "abstain"
            proposal["signal_strength"] = 0.0
            proposal["position_size"] = 0.0
            confidence = min(confidence, conf_floor * 0.8)

        else:
            proposal.setdefault("raw_action", proposal.get("action"))
            proposal["action"] = "abstain"
            proposal["signal_strength"] = 0.0
            proposal["position_size"] = 0.0
            confidence = max(0.05, min(confidence, conf_floor * 0.7))

        return proposal, confidence

    # -------------------------------
    # Health metrics (lightweight)
    # -------------------------------

    def _compute_health_metrics(self, market_data: Dict[str, Any], proposal: Dict[str, Any], elapsed_ms: float) -> Dict[str, Any]:
        data_quality = 0.0
        try:
            ohlcv = market_data.get("ohlcv")
            if isinstance(ohlcv, dict) and isinstance(ohlcv.get("close"), list):
                n = len(ohlcv["close"])
                data_quality = 1.0 if n >= 50 else max(0.0, min(1.0, n / 50.0))
        except Exception:
            pass

        # simple stability: fraction of last N actions that match latest action
        stability = 0.0
        try:
            latest = str(proposal.get("action", "abstain"))
            hist = list(self.action_history)[-20:]
            if hist:
                same = sum(1 for r in hist if str(r.get("action")) == latest)
                stability = same / max(1, len(hist))
        except Exception:
            pass

        return {
            "data_quality": data_quality,
            "signal_stability": stability,
            "latency_ms": float(elapsed_ms),
        }

    # -------------------------------
    # Main process
    # -------------------------------

    async def process(self, **inputs) -> Dict[str, Any]:
        t0 = time.time()
        self._process_seq += 1

        name = self.__class__.__name__
        proposal_key = VotingBusKeys.expert_proposal(name)
        confidence_key = VotingBusKeys.expert_confidence(name)
        per_instrument_key = f"{name}_per_instrument_votes"

        try:
            if self._check_circuit_breaker():
                self._debug_log(
                    event="skipped_circuit_open",
                    level="light",
                    open_until=str(self._circuit_open_until),
                    reason=self._circuit_open_reason,
                )
                return self._degraded_output("circuit_breaker_open")

            # Data fetch + canonicalization
            tfetch0 = time.time()
            raw_market_data = inputs.get("market_data") or self._safe_bus_get("market_data", name, default={}) or {}
            market_data = self._build_market_data(raw_market_data)
            self._perf["data_fetch_ms"] = (time.time() - tfetch0) * 1000.0

            if self._debug_enabled and self._debug_include_market_data:
                ohlcv = market_data.get("ohlcv") if isinstance(market_data.get("ohlcv"), dict) else {}
                lens = {}
                if isinstance(ohlcv, dict):
                    for k in ("open", "high", "low", "close", "volume"):
                        seq = ohlcv.get(k)
                        if isinstance(seq, list):
                            lens[k] = len(seq)
                self._debug_log(
                    event="market_data_canonical",
                    level="standard",
                    primary_symbol=market_data.get("primary_symbol"),
                    primary_timeframe=market_data.get("primary_timeframe"),
                    current_price=market_data.get("current_price"),
                    ohlcv_lengths=lens,
                    keys=sorted(list(market_data.keys())),
                )

            # Context update
            self._update_market_context(market_data)

            position_focus = self._get_position_focus_context()
            in_position_focus_mode = bool(position_focus and position_focus.get("focus_mode_active", False))

            self._debug_log(
                event="context_updated",
                level="light",
                market_context=dict(self.market_context),
                position_focus_mode=in_position_focus_mode,
            )

            # Expert-specific proposal
            tprop0 = time.time()
            raw_proposal_obj = await self._generate_expert_specific_proposal(market_data)
            proposal = self._coerce_proposal_to_dict(raw_proposal_obj)

            # Contract normalization
            primary_symbol = str(market_data.get("primary_symbol", self.config.get("primary_symbol", "XAUUSD")))
            proposal = self._validate_and_normalize_expert_output(proposal, primary_symbol)

            # Optional: capture a forensic snapshot (only flushed on failure)
            forensic_on_fail = bool(getattr(self, "_debug_forensic_on_fail", False))
            if getattr(self, "_debug_enabled", False) and forensic_on_fail:
                try:
                    self._capture_forensic_blob(
                        market_data_snapshot={
                            "primary_symbol": market_data.get("primary_symbol"),
                            "primary_timeframe": market_data.get("primary_timeframe"),
                            "current_price": market_data.get("current_price"),
                            "market_context": dict(self.market_context),
                        },
                        proposal_snapshot={
                            k: proposal.get(k)
                            for k in ("action", "signal_strength", "confidence", "reason", "proposals")
                        },
                    )
                except Exception as e:
                    self._debug_log(event="forensic_capture_failed", level="light", error=str(e))

            self._perf["proposal_build_ms"] = (time.time() - tprop0) * 1000.0

            # Confidence
            tscore0 = time.time()
            confidence_raw = await self._calculate_expert_specific_confidence(proposal, market_data)
            try:
                confidence = float(confidence_raw)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            self._perf["scoring_ms"] = (time.time() - tscore0) * 1000.0

            self._debug_log(
                event="proposal_generated",
                level="standard",
                proposal_preview={k: proposal.get(k) for k in ("action", "signal_strength", "reason", "top_level_source_instrument")},
                confidence=confidence,
                has_per_instrument=bool(proposal.get("proposals")),
            )

            # Position focus reframing (per-instrument aware)
            if in_position_focus_mode and position_focus is not None:
                per_inst = proposal.get("proposals")
                if isinstance(per_inst, dict) and per_inst:
                    modified: Dict[str, Dict[str, Any]] = {}
                    supports_map: Dict[str, bool] = {}

                    for inst, block in per_inst.items():
                        pos = self._get_position_for_instrument(inst)
                        if pos and isinstance(block, dict):
                            mod_block, _, supports = self._evaluate_single_instrument_against_position(block, inst, pos)
                            modified[inst] = mod_block
                            supports_map[inst] = supports
                        else:
                            modified[inst] = dict(block) if isinstance(block, dict) else {"action": "abstain"}
                            supports_map[inst] = True

                    proposal["proposals"] = modified
                    proposal["per_instrument"] = modified
                    proposal["position_supports"] = supports_map
                    proposal["position_focus_mode"] = True
                else:
                    pos = self._get_position_for_instrument(primary_symbol)
                    if pos:
                        base_block = dict(proposal)
                        mod_block, confidence, supports = self._evaluate_single_instrument_against_position(base_block, primary_symbol, pos)
                        proposal.update(mod_block)
                        proposal["supports_position"] = supports
                        proposal["position_focus_mode"] = True

                # expert hook (final adjustments if needed)
                try:
                    proposal = self._expert_specific_position_evaluation(proposal, position_focus)
                except Exception as e:
                    self._debug_log(event="position_hook_failed", level="light", error=str(e))

                # keep top-level consistent after reframing
                proposal = self._validate_and_normalize_expert_output(proposal, primary_symbol)

            # Postprocess/gating
            tpost0 = time.time()
            proposal, confidence = self._postprocess_proposal_for_voting(proposal, confidence)
            self._perf["postprocess_ms"] = (time.time() - tpost0) * 1000.0

            thesis = self._generate_thesis(proposal, confidence)

            # Per-instrument votes (contract-compliant)
            piv = proposal.get("proposals") if isinstance(proposal.get("proposals"), dict) else {}
            per_instrument_votes: Dict[str, Any] = dict(piv) if isinstance(piv, dict) else {}

            # Publish to bus
            tpub0 = time.time()
            self._safe_bus_set(proposal_key, proposal, module=name, thesis=thesis, confidence=confidence)
            self._safe_bus_set(confidence_key, confidence, module=name, thesis=f"{name} confidence: {confidence:.1%}")
            self._safe_bus_set(
                per_instrument_key,
                per_instrument_votes,
                module=name,
                thesis=f"{name} per-instrument votes ({len(per_instrument_votes)})",
                confidence=confidence,
            )
            self._perf["bus_publish_ms"] = (time.time() - tpub0) * 1000.0

            # Analytics + last good
            self._record_action(str(proposal.get("action", "unknown")), confidence, proposal)
            self._record_success()
            self._last_good_proposal = dict(proposal)
            self._last_good_confidence = float(confidence)

            elapsed_ms = (time.time() - t0) * 1000.0
            self._perf["total_ms"] = elapsed_ms

            # Optional perf tracker
            try:
                if hasattr(self, "performance_tracker") and self.performance_tracker is not None:
                    self.performance_tracker.record_metric(name, "process", elapsed_ms, True)
            except Exception:
                pass

            # Health metrics (for committee weighting)
            health = self._compute_health_metrics(market_data, proposal, elapsed_ms)

            output: Dict[str, Any] = {
                "voting_proposal": proposal,
                "confidence": confidence,
                "thesis": thesis,
                "market_context": dict(self.market_context),
                "expert_analytics": dict(self.expert_analytics),
                "emergency_status": {"emergency_active": False},
                "health_metrics": health,
                "performance_breakdown": dict(self._perf),
                proposal_key: proposal,
                confidence_key: confidence,
                per_instrument_key: per_instrument_votes,
                "_thesis": thesis,
            }

            if self._debug_enabled and self._debug_include_outputs:
                self._debug_log(
                    event="process_complete",
                    level="standard",
                    elapsed_ms=elapsed_ms,
                    final_action=proposal.get("action"),
                    final_signal_strength=proposal.get("signal_strength"),
                    final_confidence=confidence,
                    thesis=thesis,
                    output_keys=sorted(list(output.keys())),
                )

            # Clear forensic blob on success
            self._debug_last_forensic_blob = None
            return output

        except Exception as e:
            # Resilience policy before tripping circuit
            cls = self._classify_error(e)
            policy = self._policy(cls)

            self._debug_log(
                event="process_exception",
                level="standard",
                error=str(e),
                error_class=cls,
                policy=policy,
                traceback=traceback.format_exc(),
            )
            self._flush_forensic_blob(reason=f"exception:{cls}:{policy}")

            if policy == "use_last_good" and isinstance(self._last_good_proposal, dict) and self._last_good_confidence is not None:
                return {
                    "voting_proposal": dict(self._last_good_proposal),
                    "confidence": float(self._last_good_confidence),
                    "thesis": f"{self.__class__.__name__} using last_good due to: {cls}",
                    "market_context": dict(self.market_context),
                    "expert_analytics": dict(self.expert_analytics),
                    "emergency_status": {"emergency_active": True, "reason": f"{cls}:{policy}"},
                    "health_metrics": {"degraded": True, "reason": cls, "policy": policy},
                    "_thesis": f"{self.__class__.__name__} last_good: {cls}",
                }

            if policy == "fallback_simple":
                try:
                    raw_market_data = inputs.get("market_data") or {}
                    md = self._build_market_data(raw_market_data)
                    fallback = self._fallback_simple_proposal(md, reason=f"{cls}:{str(e)}")
                    fallback = self._coerce_proposal_to_dict(fallback)
                    fallback = self._validate_and_normalize_expert_output(
                        fallback, str(md.get("primary_symbol", self.config.get("primary_symbol", "XAUUSD")))
                    )
                    fb_conf = 0.10
                    thesis = self._generate_thesis(fallback, fb_conf)
                    return {
                        "voting_proposal": fallback,
                        "confidence": fb_conf,
                        "thesis": thesis,
                        "market_context": dict(self.market_context),
                        "expert_analytics": dict(self.expert_analytics),
                        "emergency_status": {"emergency_active": True, "reason": f"{cls}:{policy}"},
                        "health_metrics": {"degraded": True, "reason": cls, "policy": policy},
                        "_thesis": thesis,
                    }
                except Exception:
                    # fall through to degrade_to_flat
                    policy = "degrade_to_flat"

            # Default: record error and degrade
            self._record_error(e)
            msg = None
            try:
                ep = getattr(self, "error_pinpointer", None)
                if ep is not None and hasattr(ep, "analyze_error"):
                    msg = str(ep.analyze_error(e, f"{self.__class__.__name__}_process"))
            except Exception:
                msg = None
            if not msg:
                msg = str(e)

            try:
                self.logger.error(f"[{self.__class__.__name__}] Process error: {msg}")
            except Exception:
                pass

            return self._degraded_output(f"{cls}:{policy}:{msg}")

    # -------------------------------
    # Misc helpers
    # -------------------------------

    def _generate_thesis(self, proposal: Dict[str, Any], confidence: float) -> str:
        name = self.__class__.__name__
        action = proposal.get("action", "unknown")
        signal = float(proposal.get("signal_strength", 0.0) or 0.0)
        regime = self.market_context.get("regime", "unknown")
        return f"{name}: action={action} | signal={signal:.2f} | confidence={confidence:.1%} | regime={regime}"

    def _degraded_output(self, reason: str) -> Dict[str, Any]:
        name = self.__class__.__name__
        proposal_key = VotingBusKeys.expert_proposal(name)
        confidence_key = VotingBusKeys.expert_confidence(name)
        per_instrument_key = f"{name}_per_instrument_votes"

        proposal: Dict[str, Any] = {
            "action": "abstain",
            "reason": reason,
            "signal_strength": 0.0,
            "position_size": 0.0,
        }
        proposal = self._ensure_proposal_scaffold(proposal)

        return {
            "voting_proposal": proposal,
            "confidence": 0.1,
            "thesis": f"{name} operating in degraded mode: {reason}",
            "market_context": dict(self.market_context),
            "expert_analytics": dict(self.expert_analytics),
            "emergency_status": {"emergency_active": True, "reason": reason},
            "health_metrics": {"degraded": True, "reason": reason},
            proposal_key: proposal,
            confidence_key: 0.1,
            per_instrument_key: {},
            "_thesis": f"{name} degraded: {reason}",
        }

    # Convenience builders for subclasses (optional)
    def build_proposal(
        self,
        instrument: str,
        action: str,
        signal_strength: float,
        confidence: float,
        rationale: str = "",
        **extra: Any,
    ) -> Dict[str, Any]:
        inst = self._canonicalize_instrument(instrument)
        return {
            "action": self._normalize_action(action),
            "signal_strength": max(0.0, min(1.0, float(signal_strength))),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "rationale": str(rationale or ""),
            "instrument": inst,
            **self.DEFAULT_PROPOSAL_SCAFFOLD,
            **extra,
        }

    def create_proposal(
        self,
        action: str,
        signal_strength: float,
        confidence: float,
        reason: str = "",
        **extra_fields: Any,
    ) -> VotingProposal:
        return VotingProposal(
            action=action,
            confidence=confidence,
            signal_strength=signal_strength,
            reason=reason,
            expert=self.__class__.__name__,
            timestamp=datetime.datetime.now().isoformat(),
            metadata=extra_fields,
        )

    # Optional persistence hooks (only used if your framework calls them)
    def _get_custom_state(self) -> Dict[str, Any]:
        def _deque_to_list(x: Any) -> Any:
            if isinstance(x, deque):
                return list(x)
            return x

        return {
            "market_context": dict(self.market_context),
            "expert_analytics": dict(self.expert_analytics),
            "action_history": [_deque_to_list(self.action_history)][0] if isinstance(self.action_history, deque) else [],
            "consecutive_errors": int(self._consecutive_errors),
            "circuit_open": bool(self._circuit_open),
            "circuit_open_until": str(self._circuit_open_until) if self._circuit_open_until else None,
            "indicator_cache_keys": list(self._indicator_cache.keys()),
            "state_store": dict(self.state_store),
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        try:
            mc = state.get("market_context")
            if isinstance(mc, dict):
                self.market_context.update(mc)
        except Exception:
            pass
        try:
            ea = state.get("expert_analytics")
            if isinstance(ea, dict):
                self.expert_analytics.update(ea)
        except Exception:
            pass
        try:
            ah = state.get("action_history")
            if isinstance(ah, list):
                self.action_history = deque(ah, maxlen=int(self._cfg_int("action_history_len", 100, 1, 10_000)))
        except Exception:
            pass
        try:
            self._consecutive_errors = int(state.get("consecutive_errors", self._consecutive_errors))
            self._circuit_open = bool(state.get("circuit_open", self._circuit_open))
        except Exception:
            pass
        try:
            ss = state.get("state_store")
            if isinstance(ss, dict):
                self.state_store = dict(ss)
        except Exception:
            pass
