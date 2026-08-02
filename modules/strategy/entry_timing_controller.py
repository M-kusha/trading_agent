#!/usr/bin/env python3
"""
EntryTimingController - SmartInfoBus Integration for Timing Features
====================================================================

This module wraps the pure timing_features library for live trading.
It reads inputs from SmartInfoBus and publishes timing features back.

Architecture:
- Reads: market_data_latest, atr_values, session_info
- Reads (preferred for OHLC windows): multi_timeframe_data
- Optional: position_state_summary (gracefully handled if missing)
- Computes: timing features via timing_features.compute_timing_features()
- Publishes:
    - entry_timing: per-instrument TimingFeatures as dicts
    - entry_timing_array: per-instrument fixed-size float arrays
    - entry_timing_allowed: global "any instrument allowed" bool

This is a thin adapter layer - all core logic lives in timing_features.py.

Version: 1.2.0 (v5.2: position_state_summary now optional)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from modules.core.module_base import BaseModule, module
from modules.timing.timing_features import (
    TIMING_FEATURE_DIM,
    TimingConfig,
    TimingFeatures,
    compute_timing_features,
    load_timing_config,
    timing_features_to_array,
)
from modules.utils.info_bus import InfoBusManager

logger = logging.getLogger(__name__)


@module(
    name="EntryTimingController",
    version="1.1.0",
    provides=[
        "entry_timing",
        "entry_timing_array",
        "entry_timing_allowed",
    ],
    requires=[
        "market_data_latest",
        "multi_timeframe_data",
        "atr_values",
        "session_info",
        # v5.2: position_state_summary is optional - code handles missing gracefully
    ],
    dependencies=[],
    thesis_required=False,
)
class EntryTimingController(BaseModule):
    """
    Computes entry timing features and publishes to SmartInfoBus.

    This module runs on each orchestration cycle to provide PPO
    with rich timing features for better entry/exit decisions.

    Notes:
    - It is deliberately "fail-closed" per instrument: if data is missing
      or invalid for an instrument, that instrument gets entry_allowed=False
      with block_reasons explaining why.
    - Global entry_timing_allowed is "any instrument allowed", mainly for
      high-level monitors; PPO should look at per-instrument features.
    """

    def _initialize(self) -> None:
        """Initialize the controller and load configuration."""
        self.smart_bus = InfoBusManager.get_instance()

        # Load timing configuration for all instruments
        # You can override path via module config in module_registry.yaml
        config_path = self.config.get("timing_config_path", "config/timing_policy.yaml")
        self.timing_config: TimingConfig = load_timing_config(config_path)

        # Instruments to track (default to XAUUSD-only; strict pipelines are single-instrument)
        instruments_cfg = self.config.get("instruments", ["XAUUSD"])
        # Ensure this is a list of strings
        self.instruments: List[str] = [
            str(sym) for sym in instruments_cfg
            if isinstance(sym, (str, bytes))
        ]

        # OHLC window size (minimum 2 bars to be meaningful)
        ohlc_window_size_cfg = int(self.config.get("ohlc_window_size", 50))
        self.ohlc_window_size: int = max(2, ohlc_window_size_cfg)

        logger.info(
            "EntryTimingController initialized: instruments=%s, window=%d, feat_dim=%d",
            self.instruments,
            self.ohlc_window_size,
            TIMING_FEATURE_DIM,
        )

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """
        Compute timing features for all tracked instruments.

        Args:
            **inputs: Data from orchestrator, usually mirrored from SmartInfoBus:
                - market_data_latest: dict[instrument -> OHLC/price structure]
                - multi_timeframe_data: dict[instrument -> timeframe -> OHLC arrays + current_bar] (preferred for OHLC windows)
                - atr_values: dict[instrument -> float]
                - session_info: dict with time info (hour, minute, weekday)
                - position_state_summary: dict[instrument -> state summary]

        Returns:
            {
                "entry_timing": {
                    "EURUSD": TimingFeatures.to_dict(),
                    "XAUUSD": ...
                },
                "entry_timing_array": {
                    "EURUSD": [float; TIMING_FEATURE_DIM],
                    "XAUUSD": ...
                },
                "entry_timing_allowed": bool (any instrument entry_allowed=True)
            }
        """
        results: Dict[str, Dict[str, Any]] = {}
        timing_arrays: Dict[str, List[float]] = {}
        any_allowed = False

        market_data = inputs.get("market_data_latest", {}) or {}
        multi_timeframe_data = inputs.get("multi_timeframe_data")
        atr_values = inputs.get("atr_values", {}) or {}
        session_info = inputs.get("session_info", {}) or {}
        position_summary = inputs.get("position_state_summary", {}) or {}

        if not isinstance(market_data, dict):
            logger.warning("[EntryTimingController] market_data_latest not a dict; got %r", type(market_data))
            market_data = {}

        if not isinstance(atr_values, dict):
            logger.warning("[EntryTimingController] atr_values not a dict; got %r", type(atr_values))
            atr_values = {}

        if not isinstance(session_info, dict):
            session_info = {}

        if not isinstance(position_summary, dict):
            position_summary = {}

        # Prefer a full OHLC window from multi_timeframe_data; fall back to bus read if orchestrator didn't pass it.
        if not isinstance(multi_timeframe_data, dict):
            try:
                multi_timeframe_data = self.smart_bus.get(
                    "multi_timeframe_data",
                    "EntryTimingController",
                    default={},
                )
            except Exception:
                multi_timeframe_data = {}

        for instrument in self.instruments:
            try:
                features = self._compute_for_instrument(
                    instrument=instrument,
                    market_data=market_data,
                    multi_timeframe_data=multi_timeframe_data,
                    atr_values=atr_values,
                    session_info=session_info,
                    position_summary=position_summary,
                )
                results[instrument] = features.to_dict()
                timing_arrays[instrument] = timing_features_to_array(features).tolist()

                if features.entry_allowed:
                    any_allowed = True

            except Exception as e:
                logger.warning(
                    "[EntryTimingController] Failed to compute timing for %s: %s",
                    instrument,
                    e,
                )
                # Return default "blocked" features on error
                features = TimingFeatures(entry_allowed=False)
                features.block_reasons.append(f"COMPUTE_ERROR:{type(e).__name__}")
                results[instrument] = features.to_dict()
                timing_arrays[instrument] = timing_features_to_array(features).tolist()

        # Optional: small debug summary
        allowed_count = 0
        try:
            allowed_count = sum(
                1 for inst in self.instruments
                if results.get(inst, {}).get("entry_allowed", False)
            )
            logger.debug(
                "[EntryTimingController] cycle complete: %d/%d instruments entry_allowed=True",
                allowed_count,
                len(self.instruments),
            )
        except Exception:
            # Logging errors should never break the module
            pass

        out = {
            "entry_timing": results,
            "entry_timing_array": timing_arrays,
            "entry_timing_allowed": any_allowed,
        }

        # Defensive publish: orchestrator should publish module outputs, but if that path
        # is misconfigured, publish directly to keep downstream consumers alive.
        try:
            thesis = f"Entry timing features computed (allowed={allowed_count}/{len(self.instruments)})"
            self.smart_bus.set(
                "entry_timing",
                results,
                module="EntryTimingController",
                thesis=thesis,
                confidence=0.8,
            )
            self.smart_bus.set(
                "entry_timing_array",
                timing_arrays,
                module="EntryTimingController",
                thesis=thesis,
                confidence=0.8,
            )
            self.smart_bus.set(
                "entry_timing_allowed",
                any_allowed,
                module="EntryTimingController",
                thesis=thesis,
                confidence=0.8,
            )
        except Exception as e:
            logger.debug("[EntryTimingController] Bus publish failed: %s", e)

        return out

    def _compute_for_instrument(
        self,
        instrument: str,
        market_data: Dict[str, Any],
        multi_timeframe_data: Dict[str, Any],
        atr_values: Dict[str, Any],
        session_info: Dict[str, Any],
        position_summary: Dict[str, Any],
    ) -> TimingFeatures:
        """
        Compute timing features for a single instrument.

        Args:
            instrument: Instrument symbol (e.g., "XAUUSD").
            market_data: Full market_data_latest dict from inputs (often single-bar snapshot).
            multi_timeframe_data: Full multi_timeframe_data dict (preferred for OHLC windows).
            atr_values: Full atr_values dict from inputs.
            session_info: Session info dict.
            position_summary: Per-instrument position state summary.

        Returns:
            TimingFeatures for the instrument.
        """
        # ─────────────────────────────────────────────────────────────
        # 1. Get OHLC data
        # ─────────────────────────────────────────────────────────────
        instrument_data = market_data.get(instrument, {})

        if not isinstance(instrument_data, dict):
            logger.warning(
                "[EntryTimingController] market_data_latest[%s] not a dict; got %r",
                instrument,
                type(instrument_data),
            )
            features = TimingFeatures(entry_allowed=False)
            features.block_reasons.append("NO_INSTRUMENT_DATA")
            return features

        # First attempt: market_data_latest per-instrument block (may be arrays in some modes)
        ohlc_window = self._extract_ohlc_window(instrument, instrument_data)

        # Preferred: multi_timeframe_data[instrument]["M15"] provides a true OHLC window
        if ohlc_window is None or len(ohlc_window) < 2:
            try:
                mtf_inst = multi_timeframe_data.get(instrument, {})
                if isinstance(mtf_inst, dict):
                    tf_block = mtf_inst.get("M15")
                    if isinstance(tf_block, dict):
                        ohlc_window = self._extract_ohlc_window(instrument, tf_block)
            except Exception:
                pass

        if ohlc_window is None or len(ohlc_window) < 2:
            logger.debug(
                "[EntryTimingController] No usable OHLC window for %s (len=%s)",
                instrument,
                0 if ohlc_window is None else len(ohlc_window),
            )
            features = TimingFeatures(entry_allowed=False)
            features.block_reasons.append("NO_OHLC_DATA")
            return features

        # ─────────────────────────────────────────────────────────────
        # 2. Get ATR value (compute_timing_features will still sanitize)
        # ─────────────────────────────────────────────────────────────
        raw_atr = atr_values.get(instrument, 1.0)
        try:
            atr_value = float(raw_atr)
        except (TypeError, ValueError):
            atr_value = 1.0

        # Optional local fallback if ATR missing/zero (compute_timing_features
        # also has its own safety logic; this just makes the input less insane).
        if atr_value <= 0.0:
            if len(ohlc_window) >= 14:
                ranges = ohlc_window[-14:, 1] - ohlc_window[-14:, 2]  # High - Low
                atr_value = float(np.mean(ranges))
            else:
                atr_value = float(np.mean(ohlc_window[:, 1] - ohlc_window[:, 2]))

        # ─────────────────────────────────────────────────────────────
        # 3. Get/normalize session info
        # ─────────────────────────────────────────────────────────────
        if not session_info:
            now = datetime.utcnow()
            session_info = {
                "hour": now.hour,
                "minute": now.minute,
                "weekday": now.weekday(),
            }
        else:
            # Ensure required keys exist, with sane defaults
            session_info = {
                "hour": int(session_info.get("hour", 12)),
                "minute": int(session_info.get("minute", 0)),
                "weekday": int(session_info.get("weekday", 0)),
            }

        # ─────────────────────────────────────────────────────────────
        # 4. Get position state for this instrument
        # ─────────────────────────────────────────────────────────────
        instrument_position = position_summary.get(instrument, {})
        if not isinstance(instrument_position, dict):
            instrument_position = {}

        position_state: Dict[str, Any] = {
            "side": instrument_position.get("side", 0),
            "minutes_since_entry": instrument_position.get("minutes_since_entry", 999.0),
            "minutes_since_loss": instrument_position.get("minutes_since_loss", 999.0),
            "trades_this_session": instrument_position.get("trades_this_session", 0),
            "had_recent_loss": instrument_position.get("had_recent_loss", False),
        }

        # ─────────────────────────────────────────────────────────────
        # 5. Compute features (delegated to pure timing engine)
        # ─────────────────────────────────────────────────────────────
        features = compute_timing_features(
            instrument=instrument,
            ohlc_window=ohlc_window,
            atr_value=atr_value,
            session_info=session_info,
            position_state=position_state,
            timing_config=self.timing_config,
        )

        return features

    def _extract_ohlc_window(
        self,
        instrument: str,
        instrument_data: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """
        Extract an OHLC window for a given instrument from market_data.

        Tries several common formats:
        - instrument_data["ohlc"] -> direct array-like
        - instrument_data["candles"] -> list[dict] with open/high/low/close
        - instrument_data["open"/"high"/"low"/"close"] -> parallel arrays

        Returns:
            np.ndarray of shape (N, 4) or None if extraction fails.
        """
        ohlc_window: Optional[np.ndarray] = None

        # 1) Direct "ohlc" key
        raw_ohlc = instrument_data.get("ohlc")
        if raw_ohlc is not None:
            try:
                ohlc_window = np.asarray(raw_ohlc, dtype=float)
            except Exception as e:
                logger.debug(
                    "[EntryTimingController] Invalid 'ohlc' for %s: %s",
                    instrument,
                    e,
                )

        # 2) Candles list of dicts
        if ohlc_window is None:
            candles = instrument_data.get("candles")
            if isinstance(candles, list) and candles:
                try:
                    sliced = candles[-self.ohlc_window_size :]
                    ohlc_window = np.asarray(
                        [
                            [
                                float(c.get("open", 0.0)),
                                float(c.get("high", 0.0)),
                                float(c.get("low", 0.0)),
                                float(c.get("close", 0.0)),
                            ]
                            for c in sliced
                        ],
                        dtype=float,
                    )
                except Exception as e:
                    logger.debug(
                        "[EntryTimingController] Invalid 'candles' for %s: %s",
                        instrument,
                        e,
                    )

        # 3) Parallel arrays format
        if ohlc_window is None:
            close = instrument_data.get("close")
            high = instrument_data.get("high")
            low = instrument_data.get("low")
            open_ = instrument_data.get("open")

            if close is not None and high is not None and low is not None:
                try:
                    close_arr = np.asarray(close, dtype=float)
                    high_arr = np.asarray(high, dtype=float)
                    low_arr = np.asarray(low, dtype=float)

                    n = min(len(close_arr), len(high_arr), len(low_arr))
                    if open_ is not None:
                        open_arr = np.asarray(open_, dtype=float)
                        n = min(n, len(open_arr))
                    else:
                        # If open is missing, approximate with previous close
                        open_arr = np.roll(close_arr, 1)
                        open_arr[0] = close_arr[0]

                    if n > 1:
                        ohlc_window = np.column_stack(
                            [
                                open_arr[-n:],
                                high_arr[-n:],
                                low_arr[-n:],
                                close_arr[-n:],
                            ]
                        )[-self.ohlc_window_size :]
                except Exception as e:
                    logger.debug(
                        "[EntryTimingController] Invalid OHLC arrays for %s: %s",
                        instrument,
                        e,
                    )

        if ohlc_window is not None and len(ohlc_window) > 0:
            # Ensure we always use the last N rows and correct shape
            ohlc_window = ohlc_window[-self.ohlc_window_size :]
            if ohlc_window.shape[1] != 4:
                logger.debug(
                    "[EntryTimingController] OHLC window for %s has invalid shape %s",
                    instrument,
                    ohlc_window.shape,
                )
                return None

        return ohlc_window

    def get_timing_for_instrument(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Get cached timing features for an instrument from the bus.

        This is a convenience helper for other modules to query timing.
        """
        entry_timing = self.smart_bus.get(
            "entry_timing",
            self.__class__.__name__,
            default={},
        )
        if not isinstance(entry_timing, dict):
            return None
        return entry_timing.get(instrument)

    def is_entry_allowed(self, instrument: str) -> bool:
        """
        Quick check if entry is allowed for an instrument.

        Returns True if no timing data is available (fail-open),
        to avoid hard-wiring this module as a single point of failure
        for the entire trading stack.
        """
        timing = self.get_timing_for_instrument(instrument)
        if timing is None or not isinstance(timing, dict):
            return True  # Fail-open by design
        return bool(timing.get("entry_allowed", True))
