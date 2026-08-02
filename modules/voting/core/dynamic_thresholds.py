
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

STATE_FILE = Path("state/modules/dynamic_thresholds_state.json")


try:
    from modules.utils.info_bus import InfoBusManager
    SMARTINFOBUS_AVAILABLE = True
except ImportError:
    InfoBusManager = None  # type: ignore[assignment]
    SMARTINFOBUS_AVAILABLE = False


class _NullBus:

    def get(self, key: str, module: str, default: Any = None) -> Any:
        return default

    def set(
        self,
        key: str,
        value: Any,
        module: str = "",
        thesis: str = "",
    ) -> None:
        return


class MarketRegime(Enum):
    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class InstrumentProfile:
    instrument: str


    base_confidence: float = 0.60
    base_consensus: float = 0.55
    base_intensity: float = 0.45


    confidence_threshold: float = 0.52
    consensus_threshold: float = 0.50
    intensity_threshold: float = 0.45


    typical_volatility: float = 0.015
    volatility_ema: float = 0.015


    total_signals: int = 0
    passed_signals: int = 0
    trades_taken: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0


    confidence_history: deque = field(default_factory=lambda: deque(maxlen=200))
    consensus_history: deque = field(default_factory=lambda: deque(maxlen=200))


    min_confidence: float = 0.40
    max_confidence: float = 0.75
    min_consensus: float = 0.40
    max_consensus: float = 0.75


    learning_rate: float = 0.05

    @property
    def win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.5

    @property
    def pass_rate(self) -> float:
        return (
            self.passed_signals / self.total_signals
            if self.total_signals > 0
            else 0.0
        )

    def record_signal(self, confidence: float, consensus: float, passed: bool) -> None:
        self.total_signals += 1
        if passed:
            self.passed_signals += 1

        self.confidence_history.append(float(confidence))
        self.consensus_history.append(float(consensus))

    def record_trade_outcome(self, pnl: float) -> None:
        self.trades_taken += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

    def get_percentile_thresholds(
        self, target_pass_rate: float = 0.20
    ) -> Tuple[float, float]:
        if len(self.confidence_history) < 20:
            return self.confidence_threshold, self.consensus_threshold

        percentile = (1.0 - target_pass_rate) * 100.0

        conf_threshold = float(
            np.percentile(list(self.confidence_history), percentile)
        )
        cons_threshold = float(
            np.percentile(list(self.consensus_history), percentile)
        )

        return conf_threshold, cons_threshold


@dataclass
class ThresholdConfig:


    target_pass_rate: float = 0.15


    volatility_scaling_enabled: bool = True
    volatility_scale_factor: float = 0.15


    regime_adjustment_enabled: bool = True
    trending_discount: float = 0.05
    ranging_premium: float = 0.06
    volatile_premium: float = 0.06


    performance_feedback_enabled: bool = True
    win_rate_target: float = 0.55
    feedback_sensitivity: float = 0.03


    time_adjustment_enabled: bool = True
    low_liquidity_premium: float = 0.05


    min_pass_rate: float = 0.05
    max_pass_rate: float = 0.30


    adaptation_interval_seconds: float = 300.0
    ema_alpha: float = 0.1


class DynamicThresholdManager:

    _instance: Optional["DynamicThresholdManager"] = None

    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()

        if SMARTINFOBUS_AVAILABLE and InfoBusManager is not None:
            try:
                self._bus = InfoBusManager.get_instance()
            except Exception:
                self._bus = _NullBus()
        else:
            self._bus = _NullBus()


        self._profiles: Dict[str, InstrumentProfile] = {}


        self._instrument_defaults: Dict[str, Dict[str, float]] = {
            "XAUUSD": {
                "base_confidence": 0.52,
                "base_consensus": 0.50,
                "typical_volatility": 0.025,
                "min_confidence": 0.40,
                "max_confidence": 0.75,
                "min_consensus": 0.40,
                "max_consensus": 0.80,
            },
            "EURUSD": {
                "base_confidence": 0.50,
                "base_consensus": 0.48,
                "typical_volatility": 0.008,
                "min_confidence": 0.40,
                "max_confidence": 0.72,
                "min_consensus": 0.40,
                "max_consensus": 0.75,
            },
        }

        self._last_adaptation = 0.0
        self._last_save = 0.0
        self._save_interval = 60.0


        self._load_state()


    @classmethod
    def get_instance(
        cls, config: Optional[ThresholdConfig] = None
    ) -> "DynamicThresholdManager":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        if cls._instance is not None:
            cls._instance.save_state()
        cls._instance = None


    def save_state(self) -> bool:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "_saved_at": time.time(),
                "_version": "1.0",
                "profiles": {}
            }

            for inst, profile in self._profiles.items():
                state["profiles"][inst] = {

                    "confidence_threshold": profile.confidence_threshold,
                    "consensus_threshold": profile.consensus_threshold,
                    "intensity_threshold": profile.intensity_threshold,
                    "volatility_ema": profile.volatility_ema,

                    "total_signals": profile.total_signals,
                    "passed_signals": profile.passed_signals,
                    "trades_taken": profile.trades_taken,
                    "winning_trades": profile.winning_trades,
                    "losing_trades": profile.losing_trades,
                    "total_pnl": profile.total_pnl,

                    "confidence_history": list(profile.confidence_history)[-100:],
                    "consensus_history": list(profile.consensus_history)[-100:],
                }

            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)

            self._last_save = time.time()
            logging.getLogger("voting.thresholds").debug(
                f"[DynamicThresholds] State saved: {len(self._profiles)} profiles"
            )
            return True

        except Exception as e:
            logging.getLogger("voting.thresholds").warning(
                f"[DynamicThresholds] Failed to save state: {e}"
            )
            return False

    def _load_state(self) -> bool:
        if not STATE_FILE.exists():
            logging.getLogger("voting.thresholds").info(
                "[DynamicThresholds] No saved state found, starting fresh"
            )
            return False

        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)

            saved_at = state.get("_saved_at", 0)
            age_hours = (time.time() - saved_at) / 3600


            if age_hours > 24:
                logging.getLogger("voting.thresholds").info(
                    f"[DynamicThresholds] Saved state too old ({age_hours:.1f}h), starting fresh"
                )
                return False

            profiles_data = state.get("profiles", {})
            restored_count = 0

            for inst, data in profiles_data.items():
                profile = self._get_or_create_profile(inst)


                profile.confidence_threshold = data.get(
                    "confidence_threshold", profile.confidence_threshold
                )
                profile.consensus_threshold = data.get(
                    "consensus_threshold", profile.consensus_threshold
                )
                profile.intensity_threshold = data.get(
                    "intensity_threshold", profile.intensity_threshold
                )
                profile.volatility_ema = data.get(
                    "volatility_ema", profile.volatility_ema
                )


                profile.total_signals = data.get("total_signals", 0)
                profile.passed_signals = data.get("passed_signals", 0)
                profile.trades_taken = data.get("trades_taken", 0)
                profile.winning_trades = data.get("winning_trades", 0)
                profile.losing_trades = data.get("losing_trades", 0)
                profile.total_pnl = data.get("total_pnl", 0.0)


                conf_hist = data.get("confidence_history", [])
                cons_hist = data.get("consensus_history", [])
                profile.confidence_history = deque(conf_hist, maxlen=200)
                profile.consensus_history = deque(cons_hist, maxlen=200)

                restored_count += 1

            logging.getLogger("voting.thresholds").info(
                f"[DynamicThresholds] 📥 State restored: {restored_count} profiles, "
                f"age={age_hours:.1f}h"
            )
            return True

        except Exception as e:
            logging.getLogger("voting.thresholds").warning(
                f"[DynamicThresholds] Failed to load state: {e}"
            )
            return False

    def _maybe_save(self) -> None:
        now = time.time()
        if now - self._last_save >= self._save_interval:
            self.save_state()


    @staticmethod
    def _normalize_instrument(instrument: str) -> str:
        if not instrument:
            return "UNKNOWN"
        s = str(instrument).strip()
        if not s:
            return "UNKNOWN"
        return s.replace("/", "").replace("_", "").replace("-", "").upper()

    def _get_or_create_profile(self, instrument: str) -> InstrumentProfile:
        norm_inst = self._normalize_instrument(instrument)

        if norm_inst not in self._profiles:
            defaults = self._instrument_defaults.get(norm_inst, {})
            profile = InstrumentProfile(
                instrument=norm_inst,
                base_confidence=defaults.get("base_confidence", 0.60),
                base_consensus=defaults.get("base_consensus", 0.55),
                typical_volatility=defaults.get("typical_volatility", 0.015),
                min_confidence=defaults.get("min_confidence", 0.45),
                max_confidence=defaults.get("max_confidence", 0.85),
                min_consensus=defaults.get("min_consensus", 0.40),
                max_consensus=defaults.get("max_consensus", 0.80),
            )

            profile.confidence_threshold = profile.base_confidence
            profile.consensus_threshold = profile.base_consensus
            self._profiles[norm_inst] = profile

        return self._profiles[norm_inst]


    def _get_current_volatility(self, instrument: str) -> float:
        norm_inst = self._normalize_instrument(instrument)
        try:

            vol_data = self._bus.get(
                "volatility_by_instrument", "DynamicThresholds", default={}
            )
            if isinstance(vol_data, dict):
                if norm_inst in vol_data:
                    return float(vol_data[norm_inst])

                for key, value in vol_data.items():
                    if self._normalize_instrument(key) == norm_inst:
                        return float(value)


            market_state = self._bus.get(
                "market_state", "DynamicThresholds", default={}
            )
            if isinstance(market_state, dict):
                vol = market_state.get(
                    "volatility", market_state.get("atr_pct", 0.015)
                )
                if vol:
                    return float(vol)


            features = self._bus.get(
                "feature_data", "DynamicThresholds", default={}
            )
            if isinstance(features, dict):
                inst_features = features.get(norm_inst, {})
                if isinstance(inst_features, dict):
                    vol = inst_features.get("volatility", 0.015)
                    return float(vol)

        except Exception:
            pass

        return 0.015

    def _get_market_regime(self, instrument: str) -> MarketRegime:
        norm_inst = self._normalize_instrument(instrument)
        try:

            regime_data = self._bus.get(
                "market_regime_by_instrument", "DynamicThresholds", default={}
            )
            regime_str: str
            if isinstance(regime_data, dict):
                if norm_inst in regime_data:
                    regime_str = str(regime_data[norm_inst]).lower()
                else:

                    regime_str = ""
                    for key, value in regime_data.items():
                        if self._normalize_instrument(key) == norm_inst:
                            regime_str = str(value).lower()
                            break
            else:
                regime_str = ""


            if not regime_str:
                regime_str = str(
                    self._bus.get(
                        "market_regime", "DynamicThresholds", default="unknown"
                    )
                ).lower()

            if "trend" in regime_str:
                if "strong" in regime_str:
                    return MarketRegime.TRENDING_STRONG
                return MarketRegime.TRENDING_WEAK
            if "rang" in regime_str or "sideways" in regime_str:
                return MarketRegime.RANGING
            if "volat" in regime_str or "choppy" in regime_str:
                return MarketRegime.VOLATILE

        except Exception:
            pass

        return MarketRegime.UNKNOWN

    def _get_session_liquidity(self) -> float:
        try:
            import datetime

            now = datetime.datetime.utcnow()
            hour = now.hour


            if 12 <= hour <= 16:
                return 1.0
            if 7 <= hour <= 21:
                return 0.8
            if 0 <= hour <= 7:
                return 0.5
            return 0.6
        except Exception:
            return 0.7


    def _calculate_volatility_adjustment(
        self, instrument: str, profile: InstrumentProfile
    ) -> float:
        if not self.config.volatility_scaling_enabled:
            return 0.0

        current_vol = self._get_current_volatility(instrument)
        typical_vol = profile.typical_volatility


        alpha = self.config.ema_alpha
        profile.volatility_ema = alpha * current_vol + (1 - alpha) * profile.volatility_ema


        vol_ratio = min(max(current_vol / max(typical_vol, 1e-6), 0.5), 2.0)


        adjustment = self.config.volatility_scale_factor * (vol_ratio - 1.0)
        return adjustment

    def _calculate_regime_adjustment(self, instrument: str) -> float:
        if not self.config.regime_adjustment_enabled:
            return 0.0

        regime = self._get_market_regime(instrument)

        if regime == MarketRegime.TRENDING_STRONG:
            return -self.config.trending_discount * 1.5
        if regime == MarketRegime.TRENDING_WEAK:
            return -self.config.trending_discount
        if regime == MarketRegime.RANGING:
            return self.config.ranging_premium
        if regime == MarketRegime.VOLATILE:
            return self.config.volatile_premium

        return 0.0

    def _calculate_performance_adjustment(
        self, profile: InstrumentProfile
    ) -> float:
        if not self.config.performance_feedback_enabled:
            return 0.0

        if profile.trades_taken < 5:
            return 0.0

        win_rate = profile.win_rate
        target = self.config.win_rate_target


        deviation = target - win_rate
        adjustment = deviation * self.config.feedback_sensitivity * 2.0


        return max(-0.05, min(0.10, adjustment))

    def _calculate_time_adjustment(self, instrument: str) -> float:
        if not self.config.time_adjustment_enabled:
            return 0.0

        liquidity = self._get_session_liquidity()
        norm_inst = self._normalize_instrument(instrument)


        if "XAU" in norm_inst or "GOLD" in norm_inst:
            liquidity = max(liquidity, 0.7)

        if liquidity < 0.6:
            return self.config.low_liquidity_premium * (1.0 - liquidity / 0.6)

        return 0.0

    def _calculate_percentile_adjustment(
        self, profile: InstrumentProfile
    ) -> Tuple[float, float]:
        if len(profile.confidence_history) < 30:
            return profile.base_confidence, profile.base_consensus

        current_pass_rate = profile.pass_rate
        target = self.config.target_pass_rate


        if current_pass_rate > self.config.max_pass_rate:
            effective_target = target * 0.7
        elif current_pass_rate < self.config.min_pass_rate:
            effective_target = min(target * 1.5, 0.25)
        else:
            effective_target = target

        return profile.get_percentile_thresholds(effective_target)


    def get_thresholds(self, instrument: str) -> Dict[str, Any]:
        profile = self._get_or_create_profile(instrument)


        self._maybe_adapt(instrument, profile)


        self._maybe_save()


        perc_conf, perc_cons = self._calculate_percentile_adjustment(profile)


        vol_adj = self._calculate_volatility_adjustment(instrument, profile)
        regime_adj = self._calculate_regime_adjustment(instrument)
        perf_adj = self._calculate_performance_adjustment(profile)
        time_adj = self._calculate_time_adjustment(instrument)

        total_adjustment = vol_adj + regime_adj + perf_adj + time_adj


        if len(profile.confidence_history) >= 30:
            conf_threshold = perc_conf + total_adjustment * 0.5
            cons_threshold = perc_cons + total_adjustment * 0.5
        else:
            conf_threshold = profile.base_confidence + total_adjustment
            cons_threshold = profile.base_consensus + total_adjustment


        conf_threshold = max(
            profile.min_confidence, min(profile.max_confidence, conf_threshold)
        )
        cons_threshold = max(
            profile.min_consensus, min(profile.max_consensus, cons_threshold)
        )


        alpha = self.config.ema_alpha
        profile.confidence_threshold = (
            alpha * conf_threshold + (1 - alpha) * profile.confidence_threshold
        )
        profile.consensus_threshold = (
            alpha * cons_threshold + (1 - alpha) * profile.consensus_threshold
        )


        high_conf = min(profile.confidence_threshold + 0.15, 0.90)
        strong_cons = min(profile.consensus_threshold + 0.12, 0.85)

        intensity = profile.intensity_threshold + total_adjustment * 0.3
        intensity = max(0.30, min(0.70, intensity))

        return {
            "confidence_threshold": profile.confidence_threshold,
            "consensus_threshold": profile.consensus_threshold,
            "intensity_threshold": intensity,
            "high_confidence_threshold": high_conf,
            "strong_consensus_threshold": strong_cons,
            "_adjustments": {
                "volatility": vol_adj,
                "regime": regime_adj,
                "performance": perf_adj,
                "time": time_adj,
                "total": total_adjustment,
            },
            "_profile": {
                "win_rate": profile.win_rate,
                "pass_rate": profile.pass_rate,
                "trades": profile.trades_taken,
                "signals": profile.total_signals,
            },
        }

    def record_signal(
        self,
        instrument: str,
        confidence: float,
        consensus: float,
        passed: bool,
    ) -> None:
        profile = self._get_or_create_profile(instrument)
        profile.record_signal(confidence, consensus, passed)

    def record_trade_outcome(self, instrument: str, pnl: float) -> None:
        profile = self._get_or_create_profile(instrument)
        profile.record_trade_outcome(pnl)

    def _maybe_adapt(self, instrument: str, profile: InstrumentProfile) -> None:
        now = time.time()
        if now - self._last_adaptation < self.config.adaptation_interval_seconds:
            return

        self._last_adaptation = now

        try:
            regime = self._get_market_regime(instrument)
            vol = self._get_current_volatility(instrument)

            self._bus.set(
                "adaptive_thresholds_state",
                {
                    inst: {
                        "confidence": p.confidence_threshold,
                        "consensus": p.consensus_threshold,
                        "regime": regime.value if inst == profile.instrument else "",
                        "volatility": vol if inst == profile.instrument else None,
                        "win_rate": p.win_rate,
                        "pass_rate": p.pass_rate,
                        "trades": p.trades_taken,
                        "signals": p.total_signals,
                    }
                    for inst, p in self._profiles.items()
                },
                module="DynamicThresholds",
                thesis="Adaptive threshold state update",
            )
        except Exception:

            pass

    def get_all_thresholds(self) -> Dict[str, Dict[str, float]]:
        return {inst: self.get_thresholds(inst) for inst in self._profiles.keys()}

    def get_state_summary(self) -> Dict[str, Any]:
        state_file_exists = STATE_FILE.exists()
        state_age = None
        if state_file_exists:
            try:
                state_age = (time.time() - STATE_FILE.stat().st_mtime) / 60
            except Exception:
                pass

        return {
            "instruments": list(self._profiles.keys()),
            "persistence": {
                "state_file": str(STATE_FILE),
                "file_exists": state_file_exists,
                "state_age_minutes": round(state_age, 1) if state_age else None,
                "last_save_ago": round(time.time() - self._last_save, 1) if self._last_save else None,
            },
            "config": {
                "target_pass_rate": self.config.target_pass_rate,
                "volatility_scaling": self.config.volatility_scaling_enabled,
                "regime_adjustment": self.config.regime_adjustment_enabled,
                "performance_feedback": self.config.performance_feedback_enabled,
            },
            "profiles": {
                inst: {
                    "confidence_threshold": p.confidence_threshold,
                    "consensus_threshold": p.consensus_threshold,
                    "win_rate": p.win_rate,
                    "pass_rate": p.pass_rate,
                    "trades": p.trades_taken,
                    "signals": p.total_signals,
                    "pnl": p.total_pnl,
                }
                for inst, p in self._profiles.items()
            },
        }


def get_adaptive_threshold(instrument: str, threshold_name: str) -> float:
    manager = DynamicThresholdManager.get_instance()
    thresholds = manager.get_thresholds(instrument)
    return float(thresholds.get(threshold_name, 0.60))


def get_adaptive_thresholds(instrument: str) -> Dict[str, float]:
    manager = DynamicThresholdManager.get_instance()
    return manager.get_thresholds(instrument)


def record_signal_for_adaptation(
    instrument: str,
    confidence: float,
    consensus: float,
    passed: bool,
) -> None:
    manager = DynamicThresholdManager.get_instance()
    manager.record_signal(instrument, confidence, consensus, passed)


def record_trade_for_adaptation(instrument: str, pnl: float) -> None:
    manager = DynamicThresholdManager.get_instance()
    manager.record_trade_outcome(instrument, pnl)
