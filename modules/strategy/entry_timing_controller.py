#!/usr/bin/env python3

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

    ],
    dependencies=[],
    thesis_required=False,
)
class EntryTimingController(BaseModule):

    def _initialize(self) -> None:
        self.smart_bus = InfoBusManager.get_instance()


        config_path = self.config.get("timing_config_path", "config/timing_policy.yaml")
        self.timing_config: TimingConfig = load_timing_config(config_path)


        instruments_cfg = self.config.get("instruments", ["XAUUSD"])

        self.instruments: List[str] = [
            str(sym) for sym in instruments_cfg
            if isinstance(sym, (str, bytes))
        ]


        ohlc_window_size_cfg = int(self.config.get("ohlc_window_size", 50))
        self.ohlc_window_size: int = max(2, ohlc_window_size_cfg)

        logger.info(
            "EntryTimingController initialized: instruments=%s, window=%d, feat_dim=%d",
            self.instruments,
            self.ohlc_window_size,
            TIMING_FEATURE_DIM,
        )

    async def process(self, **inputs: Any) -> Dict[str, Any]:
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

                features = TimingFeatures(entry_allowed=False)
                features.block_reasons.append(f"COMPUTE_ERROR:{type(e).__name__}")
                results[instrument] = features.to_dict()
                timing_arrays[instrument] = timing_features_to_array(features).tolist()


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

            pass

        out = {
            "entry_timing": results,
            "entry_timing_array": timing_arrays,
            "entry_timing_allowed": any_allowed,
        }


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


        ohlc_window = self._extract_ohlc_window(instrument, instrument_data)


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


        raw_atr = atr_values.get(instrument, 1.0)
        try:
            atr_value = float(raw_atr)
        except (TypeError, ValueError):
            atr_value = 1.0


        if atr_value <= 0.0:
            if len(ohlc_window) >= 14:
                ranges = ohlc_window[-14:, 1] - ohlc_window[-14:, 2]
                atr_value = float(np.mean(ranges))
            else:
                atr_value = float(np.mean(ohlc_window[:, 1] - ohlc_window[:, 2]))


        if not session_info:
            now = datetime.utcnow()
            session_info = {
                "hour": now.hour,
                "minute": now.minute,
                "weekday": now.weekday(),
            }
        else:

            session_info = {
                "hour": int(session_info.get("hour", 12)),
                "minute": int(session_info.get("minute", 0)),
                "weekday": int(session_info.get("weekday", 0)),
            }


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
        ohlc_window: Optional[np.ndarray] = None


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
        entry_timing = self.smart_bus.get(
            "entry_timing",
            self.__class__.__name__,
            default={},
        )
        if not isinstance(entry_timing, dict):
            return None
        return entry_timing.get(instrument)

    def is_entry_allowed(self, instrument: str) -> bool:
        timing = self.get_timing_for_instrument(instrument)
        if timing is None or not isinstance(timing, dict):
            return True
        return bool(timing.get("entry_allowed", True))
