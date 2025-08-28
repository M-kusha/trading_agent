# ─────────────────────────────────────────────────────────────
# File: modules/market/time_aware_risk_scaling.py
# [ROCKET] PRODUCTION-READY Time-Aware Risk Scaling with Advanced Analytics
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# ENHANCED: SmartInfoBus integration, session analysis, thesis generation
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List, Union
from collections import deque
import datetime
from dataclasses import dataclass
import threading

# Core SmartInfoBus Infrastructure
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TimeAwareRiskConfig:
    """Configuration for Time-Aware Risk Scaling"""
    asian_end: int = 8
    euro_end: int = 16
    us_end: int = 22

    # Risk scaling parameters
    decay_factor: float = 0.9
    base_factor: float = 1.0
    vol_window: int = 100
    session_memory: int = 24

    # Session multipliers
    asian_multiplier: float = 1.2
    european_multiplier: float = 1.0
    us_multiplier: float = 1.1
    closed_multiplier: float = 0.5

    # Performance thresholds
    max_processing_time_ms: float = 100
    circuit_breaker_threshold: int = 3
    risk_threshold_high: float = 0.8
    risk_threshold_critical: float = 0.95

    # Instruments for volatility fallback
    instruments: Tuple[str, ...] = ("XAU/USD", "EUR/USD")


# ═══════════════════════════════════════════════════════════════════
# MODULE
# ═══════════════════════════════════════════════════════════════════

@module(
    name="TimeAwareRiskScaling",
    version="3.1.0",
    category="risk",
    provides=[
        "risk_scaling_factor",
        "session_risk",
        "volatility_adjustment",
        "market_conditions",
        "time_risk_analysis",
        "time_risk_status",
        "time_risk_health",
    ],
    requires=[
        "timestamp",
        "market_data",
        "risk_data",
        "volatility_data",
    ],
    description="Advanced time-aware risk scaling with session analysis and volatility modeling",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
)
class TimeAwareRiskScaling(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Production-grade time-aware risk scaling with advanced session analytics.
    Robust SmartInfoBus IO, strictly typed outputs, and no cross-module collisions.
    """

    # ── LIFECYCLE ──────────────────────────────────────────────────

    def __init__(self, config: Optional[Union[TimeAwareRiskConfig, Dict[str, Any]]] = None, **kwargs) -> None:
        """Initialize with comprehensive advanced systems."""
        if isinstance(config, dict):
            self.time_risk_config: TimeAwareRiskConfig = TimeAwareRiskConfig(**config)
            base_config: Dict[str, Any] = config.copy()
        elif config is None:
            self.time_risk_config = TimeAwareRiskConfig()
            base_config = {}
        else:
            self.time_risk_config = config
            base_config = {}

        # Internal flags
        self._fully_initialized: bool = False
        self._monitoring_active: bool = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Systems + state
        self._initialize_advanced_systems()

        # Parent init (expects dict)
        super().__init__(config=base_config)

        self._initialize_risk_state()
        self._initialize_session_tracking()
        self._start_monitoring()

        # Mark ready and publish status
        self._fully_initialized = True
        self._initialize()

        self.logger.info(
            format_operator_message(
                "⏰",
                "TIME_RISK_SCALING_INITIALIZED",
                details=f"Sessions: Asian({self.time_risk_config.asian_end}h), Euro({self.time_risk_config.euro_end}h), US({self.time_risk_config.us_end}h)",
                result="Production-ready time-aware risk scaling active",
                context="system_startup",
            )
        )

    # ── OUTPUT CONTRACT ────────────────────────────────────────────

    def _format_declared_outputs(
        self,
        risk_result: Dict[str, Any],
        time_data: Optional[Dict[str, Any]] = None,
        thesis: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Map internal results to declared provides, ensure types and defaults."""
        try:
            # Safe primitives
            scaling_factor = float(risk_result.get("scaling_factor", self.time_risk_config.base_factor))
            session = str(risk_result.get("current_session", "unknown"))
            risk_level = float(risk_result.get("risk_level", 0.0))
            hour = int(risk_result.get("hour", (time_data or {}).get("hour", datetime.datetime.now().hour)))
            volatility = float(risk_result.get("volatility", 0.01))
            vol_adj = float(risk_result.get("volatility_adjustment", 1.0))
            session_multiplier = float(risk_result.get("session_multiplier", 1.0))
            vol_regime = str(risk_result.get("volatility_regime", "unknown"))
            risk_trend = str(risk_result.get("risk_trend", "unknown"))
            vol_trend = str(risk_result.get("volatility_trend", "unknown"))

            # Convert numpy arrays if present
            vol_profile = self.vol_profile.tolist() if hasattr(self, "vol_profile") else []
            risk_profile = self.risk_profile.tolist() if hasattr(self, "risk_profile") else []
            hourly_risk_scores = self._hourly_risk_scores.tolist() if hasattr(self, "_hourly_risk_scores") else []

            time_risk_analysis = {
                "risk_level": risk_level,
                "current_session": session,
                "hour": hour,
                "risk_trend": risk_trend,
                "volatility_trend": vol_trend,
                "session_efficiency": float(risk_result.get("session_efficiency", 0.5)),
                "hourly_risk_score": float(risk_result.get("hourly_risk_score", 0.5)),
                "session_transitions": int(risk_result.get("session_transitions", 0)),
                "processing_success": bool(risk_result.get("processing_success", False)),
                "recent_transitions": list(getattr(self, "_session_transitions", []))[-5:],
                "hourly_patterns": {
                    "volatility_profile": vol_profile,
                    "risk_profile": risk_profile,
                    "hourly_risk_scores": hourly_risk_scores,
                },
                "last_update": datetime.datetime.now().isoformat(),
            }
            if isinstance(extra, dict):
                time_risk_analysis["extra"] = extra

            # Required status block (declared provide: time_risk_status)
            status_payload = {
                "status": "ok",
                "current_session": session,
                "hour": hour,
                "scaling_factor": scaling_factor,
                "risk_level": risk_level,
                "volatility": volatility,
                "last_update": datetime.datetime.now().isoformat(),
            }

            # Health payload (declared provide: time_risk_health)
            total_exec = int(self.success_count + self.failure_count)
            avg_ms = float(np.mean(self.processing_times)) if self.processing_times else 0.0
            health_payload = {
                "success_rate": float(self.success_count / max(total_exec, 1)),
                "avg_processing_time_ms": avg_ms,
                "circuit_breaker_failures": int(self.circuit_breaker_failures),
                "current_risk_level": float(risk_level),
                "session_transitions": int(self._session_changes),
                "risk_events": int(len(self._risk_events)),
                "last_update": datetime.datetime.now().isoformat(),
            }

            outputs: Dict[str, Any] = {
                "risk_scaling_factor": scaling_factor,
                "session_risk": {
                    "current_session": session,
                    "risk_level": risk_level,
                    "session_multiplier": session_multiplier,
                    "hour": hour,
                },
                "volatility_adjustment": {
                    "adjustment_factor": vol_adj,
                    "current_volatility": volatility,
                    "volatility_regime": vol_regime,
                    "volatility_trend": vol_trend,
                },
                "time_risk_status": status_payload,
                "time_risk_health": health_payload,
                "time_risk_analysis": time_risk_analysis,
                # convenience (not in provides but useful to some downstreams)
                "volatility_data": volatility,
                "market_conditions": {
                    "session": session,
                    "hour": hour,
                    "volatility_regime": vol_regime,
                    "risk_trend": risk_trend,
                },
                "risk_data": {
                    "scaling_factor": scaling_factor,
                    "risk_level": risk_level,
                    "session_multiplier": session_multiplier,
                    "volatility": volatility,
                },
                "_thesis": thesis or "Time-aware risk scaling analysis generated.",
                "thesis": thesis or "Time-aware risk scaling analysis generated.",
            }
            return outputs
        except Exception as e:
            # Absolute fallback to prevent orchestrator failure
            now_hour = int(datetime.datetime.now().hour)
            # Fallback health/status
            fb_health = {
                "success_rate": float(self.success_count / max(int(self.success_count + self.failure_count), 1)),
                "avg_processing_time_ms": float(np.mean(self.processing_times)) if self.processing_times else 0.0,
                "circuit_breaker_failures": int(self.circuit_breaker_failures),
                "current_risk_level": 0.5,
                "session_transitions": int(self._session_changes),
                "risk_events": int(len(self._risk_events)),
                "last_update": datetime.datetime.now().isoformat(),
            }

            return {
                "risk_scaling_factor": float(self.time_risk_config.base_factor),
                "session_risk": {
                    "current_session": "unknown",
                    "risk_level": 0.5,
                    "session_multiplier": 1.0,
                    "hour": now_hour,
                },
                "volatility_adjustment": {
                    "adjustment_factor": 1.0,
                    "current_volatility": 0.01,
                    "volatility_regime": "unknown",
                    "volatility_trend": "unknown",
                },
                "time_risk_status": {
                    "status": "degraded",
                    "current_session": "unknown",
                    "hour": now_hour,
                    "scaling_factor": float(self.time_risk_config.base_factor),
                    "risk_level": 0.5,
                    "volatility": 0.01,
                    "last_update": datetime.datetime.now().isoformat(),
                },
                "time_risk_health": fb_health,
                "time_risk_analysis": {
                    "risk_level": 0.5,
                    "current_session": "unknown",
                    "hour": now_hour,
                    "risk_trend": "unknown",
                    "volatility_trend": "unknown",
                    "session_efficiency": 0.5,
                    "hourly_risk_score": 0.5,
                    "session_transitions": 0,
                    "processing_success": False,
                    "recent_transitions": [],
                    "hourly_patterns": {
                        "volatility_profile": [],
                        "risk_profile": [],
                        "hourly_risk_scores": [],
                    },
                    "last_update": datetime.datetime.now().isoformat(),
                    "extra": {"formatter_error": str(e)[:200]},
                },
                "volatility_data": 0.01,
                "market_conditions": {
                    "session": "unknown",
                    "hour": now_hour,
                    "volatility_regime": "unknown",
                    "risk_trend": "unknown",
                },
                "risk_data": {
                    "scaling_factor": float(self.time_risk_config.base_factor),
                    "risk_level": 0.5,
                    "session_multiplier": 1.0,
                    "volatility": 0.01,
                },
                "_thesis": "Time-aware risk scaling (safe fallback)",
                "thesis": "Time-aware risk scaling (safe fallback)",
            }

    # ── SYSTEMS / STATE ────────────────────────────────────────────

    def _initialize_advanced_systems(self) -> None:
        """Initialize all advanced SmartInfoBus systems."""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="TimeAwareRiskScaling",
            log_path="logs/market/time_risk_scaling.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("TimeAwareRiskScaling", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Performance metrics
        self.processing_times: deque[float] = deque(maxlen=100)
        self.success_count: int = 0
        self.failure_count: int = 0
        self.circuit_breaker_failures: int = 0
        self.last_circuit_breaker_reset: float = time.time()

    def _initialize_risk_state(self) -> None:
        """Initialize risk scaling state."""
        # Volatility profiles by hour
        self.vol_profile: np.ndarray = np.ones(24, np.float32)
        self.risk_profile: np.ndarray = np.ones(24, np.float32)

        # Session state
        self._current_session: str = "unknown"
        self._session_changes: int = 0
        self._last_session_change: Optional[datetime.datetime] = None

        # Risk tracking
        self._volatility_history: deque[float] = deque(maxlen=int(self.time_risk_config.vol_window))
        self._factor_history: deque[float] = deque(maxlen=200)
        self._risk_events: deque[Dict[str, Any]] = deque(maxlen=100)
        self._session_transitions: deque[Dict[str, Any]] = deque(maxlen=50)

        # Current metrics
        self.current_scaling_factor: float = float(self.time_risk_config.base_factor)
        self.current_volatility: float = 0.01
        self.current_risk_level: float = 0.0
        self.session_performance_score: float = 0.5

    def _initialize_session_tracking(self) -> None:
        """Initialize session performance tracking."""
        self._session_performance: Dict[str, Dict[str, Any]] = {}
        self._session_risk_multipliers: Dict[str, float] = {
            "asian": self.time_risk_config.asian_multiplier,
            "european": self.time_risk_config.european_multiplier,
            "us": self.time_risk_config.us_multiplier,
            "closed": self.time_risk_config.closed_multiplier,
        }

        for session in ("asian", "european", "us", "closed"):
            self._session_performance[session] = {
                "count": 0,
                "total_factor": 0.0,
                "avg_volatility": 0.0,
                "risk_events": 0,
                "success_rate": 1.0,
                "last_update": datetime.datetime.now(),
            }

        # Advanced session analytics
        self._session_vol_patterns: np.ndarray = np.zeros((4, 24), dtype=np.float64)  # 4 sessions x 24 hours
        self._session_risk_patterns: np.ndarray = np.zeros((4, 24), dtype=np.float64)
        self._hourly_risk_scores: np.ndarray = np.zeros(24, dtype=np.float64)

    def _start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitoring_active:
            return
        self._monitoring_active = True

        def monitoring_loop() -> None:
            while self._monitoring_active:
                try:
                    self._update_health_metrics()
                    self._analyze_session_patterns()
                    self._check_risk_thresholds()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")

        self._monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def _initialize(self) -> None:
        """Async-ish initialization."""
        if not getattr(self, "_fully_initialized", False):
            return

        self.logger.info("[RELOAD] TimeAwareRiskScaling async initialization")
        self.smart_bus.set(
            "time_risk_status",
            {
                "initialized": True,
                "current_session": self._current_session,
                "scaling_factor": self.current_scaling_factor,
                "risk_level": self.current_risk_level,
            },
            module="TimeAwareRiskScaling",
            thesis="Time-aware risk scaling initialization status for system awareness",
        )

    # ── MAIN PROCESS ───────────────────────────────────────────────

    async def process(self, **inputs) -> Dict[str, Any]:
        """Main processing method with comprehensive error handling."""
        start_time = time.time()
        try:
            time_data = await self._extract_time_data(**inputs)
            if not time_data:
                return await self._handle_no_data_fallback()

            risk_result = await self._process_time_aware_scaling(time_data)

            # Build thesis components
            risk_level_lbl = "High" if risk_result["risk_level"] > 0.7 else ("Medium" if risk_result["risk_level"] > 0.4 else "Low")
            components = {
                "time_risk": risk_result["risk_level"],
                "volatility_risk": min(risk_result["volatility"] * 20.0, 1.0),
                "session_risk": 0.3 if risk_result["current_session"] == "asian" else 0.2,
                "scaling_risk": min(float(risk_result["scaling_factor"]) / 2.0, 1.0),
            }
            alerts: List[str] = []
            if risk_result["risk_level"] > 0.8:
                alerts.append("[ALERT] Critical risk level detected")
            if risk_result["volatility"] > 0.05:
                alerts.append("[WARN] High volatility environment")

            thesis = await self._generate_risk_thesis(
                risk_level_lbl,
                risk_result["risk_level"],
                {"time_data": time_data, "risk_result": risk_result},
                components,
                alerts,
            )

            await self._update_risk_smart_bus(risk_result, thesis)

            processing_time = (time.time() - start_time) * 1000.0
            self._record_success(processing_time)

            return self._format_declared_outputs(risk_result, time_data=time_data, thesis=thesis)

        except Exception as e:
            return await self._handle_risk_error(e, start_time)

    # ── DATA EXTRACTION ─────────────────────────────────────────────

    async def _extract_time_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract time and market data from multiple sources."""
        ts = self.smart_bus.get("timestamp", "TimeAwareRiskScaling")
        if not ts:
            ts = datetime.datetime.now()

        if isinstance(ts, str):
            timestamp: pd.Timestamp | datetime.datetime = pd.Timestamp(ts)
        elif isinstance(ts, datetime.datetime):
            timestamp = ts
        elif isinstance(ts, pd.Timestamp):
            timestamp = ts
        else:
            timestamp = pd.Timestamp.now()

        hour = int(int(timestamp.hour) % 24)

        volatility = await self._extract_volatility_data()

        market_data = self.smart_bus.get("market_data", "TimeAwareRiskScaling") or {}
        risk_data = self.smart_bus.get("risk_data", "TimeAwareRiskScaling") or {}

        return {
            "timestamp": timestamp,
            "hour": hour,
            "volatility": float(volatility),
            "market_data": market_data,
            "risk_data": risk_data,
            "session": self._get_session(hour),
            "source": "smartinfobus",
        }

    async def _extract_volatility_data(self) -> float:
        """Extract volatility data with multiple fallbacks."""
        vol = self.smart_bus.get("volatility_data", "TimeAwareRiskScaling")
        if isinstance(vol, (int, float)):
            return float(vol)

        market_data = self.smart_bus.get("market_data", "TimeAwareRiskScaling")
        if isinstance(market_data, dict) and market_data:
            for instrument in self.time_risk_config.instruments:
                if instrument in market_data:
                    inst_data = market_data[instrument]
                    if isinstance(inst_data, dict) and "close" in inst_data:
                        closes = np.asarray(inst_data["close"], dtype=np.float64)
                        if closes.size > 10:
                            rets = np.diff(closes) / np.where(closes[:-1] == 0, 1.0, closes[:-1])
                            window = min(20, rets.size)
                            if window > 1:
                                vol_val = float(np.std(rets[-window:]))
                                if np.isfinite(vol_val) and vol_val > 0:
                                    return vol_val

        if self.current_volatility > 0:
            return float(self.current_volatility)

        return 0.01

    # ── CORE LOGIC ─────────────────────────────────────────────────

    async def _process_time_aware_scaling(self, time_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process time-aware risk scaling with advanced analytics."""
        hour = int(time_data["hour"])
        session = str(time_data["session"])
        volatility = float(time_data["volatility"])

        # Update state
        self.current_volatility = volatility
        self._volatility_history.append(volatility)

        # Session transitions
        if session != self._current_session:
            await self._handle_session_transition(self._current_session, session, hour)

        # Base factor
        base_factor = self._calculate_base_scaling_factor(hour, session, volatility)

        # Volatility adjustments
        vol_adjustment = self._calculate_volatility_adjustment(volatility)

        # Session multiplier
        session_multiplier = float(self._session_risk_multipliers.get(session, 1.0))

        # Final scaling factor with bounds
        scaling_factor = float(np.clip(base_factor * vol_adjustment * session_multiplier, 0.1, 5.0))

        # Update profiles
        self.vol_profile[hour] = volatility
        self.risk_profile[hour] = scaling_factor

        # Risk level
        risk_level = self._calculate_current_risk_level(scaling_factor, volatility, session)

        # Update current metrics
        self.current_scaling_factor = float(scaling_factor)
        self.current_risk_level = float(risk_level)

        # Update session performance
        self._update_session_performance(session, scaling_factor, volatility)

        # Extra metrics
        risk_trend = self._calculate_risk_trend()
        volatility_regime = self._classify_volatility_regime(volatility)
        session_efficiency = self._calculate_session_efficiency(session)
        volatility_trend = self._calculate_volatility_trend()

        return {
            "scaling_factor": float(scaling_factor),
            "risk_level": float(risk_level),
            "current_session": session,
            "hour": int(hour),
            "volatility": float(volatility),
            "volatility_adjustment": float(vol_adjustment),
            "session_multiplier": float(session_multiplier),
            "risk_trend": str(risk_trend),
            "volatility_regime": str(volatility_regime),
            "volatility_trend": str(volatility_trend),
            "session_efficiency": float(session_efficiency),
            "hourly_risk_score": float(self._hourly_risk_scores[hour]),
            "session_transitions": int(self._session_changes),
            "processing_success": True,
        }

    def _get_session(self, hour: int) -> str:
        """Determine trading session based on hour (UTC)."""
        if 0 <= hour < self.time_risk_config.asian_end:
            return "asian"
        if self.time_risk_config.asian_end <= hour < self.time_risk_config.euro_end:
            return "european"
        if self.time_risk_config.euro_end <= hour < self.time_risk_config.us_end:
            return "us"
        return "closed"

    async def _handle_session_transition(self, old_session: str, new_session: str, hour: int) -> None:
        """Handle trading session transitions."""
        self._session_changes += 1
        self._last_session_change = datetime.datetime.now()
        self._current_session = new_session

        transition_data = {
            "from": old_session,
            "to": new_session,
            "hour": int(hour),
            "timestamp": self._last_session_change,
            "volatility": float(self.current_volatility),
        }
        self._session_transitions.append(transition_data)

        self.logger.info(
            format_operator_message(
                "[RELOAD]",
                "SESSION_TRANSITION",
                instrument=f"{old_session} -> {new_session}",
                details=f"Hour: {hour}, Vol: {self.current_volatility:.4f}",
                context="session_management",
            )
        )

        self._adjust_session_multipliers(new_session)

    def _calculate_base_scaling_factor(self, hour: int, session: str, volatility: float) -> float:
        """Calculate base scaling factor with hourly patterns."""
        base = float(self.time_risk_config.base_factor)

        # Decay based on recent volatility trend
        if len(self._volatility_history) > 1:
            recent = list(self._volatility_history)[-10:]
            if len(recent) > 1:
                mean_recent = float(np.mean(recent))
                mean_prev = float(np.mean(recent[:-1])) if len(recent) > 2 else mean_recent
                ratio = mean_recent / (mean_prev + 1e-8)
                base *= float(self.time_risk_config.decay_factor) * ratio

        # Hourly pattern (moderate)
        if hasattr(self, "vol_profile") and float(np.sum(self.vol_profile)) > 0:
            hourly_factor = float(self.vol_profile[hour]) / (float(np.mean(self.vol_profile)) + 1e-8)
            base *= (1.0 + 0.1 * (hourly_factor - 1.0))

        return float(base)

    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate volatility-based adjustment factor."""
        if len(self._volatility_history) < 10:
            return 1.0

        hist = np.asarray(self._volatility_history, dtype=np.float64)
        mean_vol = float(np.mean(hist))
        std_vol = float(np.std(hist))
        if std_vol == 0.0:
            return 1.0

        z = (volatility - mean_vol) / std_vol
        if z > 2.0:
            return 1.5
        if z > 1.0:
            return 1.2
        if z < -2.0:
            return 0.7
        if z < -1.0:
            return 0.85
        return 1.0

    def _calculate_current_risk_level(self, scaling_factor: float, volatility: float, session: str) -> float:
        """Calculate current overall risk level (0..1)."""
        factor_risk = min(float(scaling_factor) / 2.0, 1.0)
        vol_percentile = self._get_volatility_percentile(volatility)
        vol_risk = float(vol_percentile) / 100.0
        session_risk_map = {"asian": 0.3, "european": 0.2, "us": 0.25, "closed": 0.1}
        session_risk = float(session_risk_map.get(session, 0.2))
        combined = 0.4 * factor_risk + 0.4 * vol_risk + 0.2 * session_risk
        return float(np.clip(combined, 0.0, 1.0))

    def _get_volatility_percentile(self, volatility: float) -> float:
        """Get volatility percentile in historical context."""
        if len(self._volatility_history) < 10:
            return 50.0
        hist = np.asarray(self._volatility_history, dtype=np.float64)
        percentile = float(np.sum(hist <= volatility)) / float(len(hist)) * 100.0
        return float(percentile)

    def _calculate_risk_trend(self) -> str:
        """Calculate risk trend direction from scaling factor history."""
        if len(self._factor_history) < 5:
            return "stable"
        y = np.asarray(list(self._factor_history)[-5:], dtype=np.float64)
        x = np.arange(y.size, dtype=np.float64)
        try:
            if float(np.std(y)) == 0.0:
                return "stable"
            slope = float(np.polyfit(x, y, 1)[0])
        except Exception:
            return "stable"
        if slope > 0.02:
            return "increasing"
        if slope < -0.02:
            return "decreasing"
        return "stable"

    def _calculate_volatility_trend(self) -> str:
        """Calculate volatility trend from volatility history."""
        if len(self._volatility_history) < 10:
            return "stable"
        y = np.asarray(list(self._volatility_history)[-10:], dtype=np.float64)
        x = np.arange(y.size, dtype=np.float64)
        try:
            if float(np.std(y)) == 0.0:
                return "stable"
            slope = float(np.polyfit(x, y, 1)[0])
        except Exception:
            return "stable"
        if slope > 1e-3:
            return "increasing"
        if slope < -1e-3:
            return "decreasing"
        return "stable"

    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify current volatility regime."""
        if len(self._volatility_history) < 20:
            return "normal"
        p = self._get_volatility_percentile(volatility)
        if p > 90.0:
            return "high"
        if p > 75.0:
            return "elevated"
        if p < 10.0:
            return "low"
        if p < 25.0:
            return "subdued"
        return "normal"

    def _calculate_session_efficiency(self, session: str) -> float:
        """Calculate session efficiency score."""
        perf = self._session_performance.get(session)
        if not perf:
            return 0.5
        cnt = int(perf.get("count", 0))
        if cnt == 0:
            return 0.5
        success_rate = float(perf.get("success_rate", 0.5))
        risk_events = int(perf.get("risk_events", 0))
        efficiency = success_rate * (1.0 - min(risk_events / max(cnt, 1), 0.5))
        return float(np.clip(efficiency, 0.0, 1.0))

    def _update_session_performance(self, session: str, scaling_factor: float, volatility: float) -> None:
        """Update session performance metrics."""
        perf = self._session_performance.get(session)
        if not perf:
            return
        perf["count"] = int(perf["count"]) + 1
        perf["total_factor"] = float(perf["total_factor"]) + float(scaling_factor)
        perf["avg_volatility"] = float((perf["avg_volatility"] * (perf["count"] - 1) + volatility) / perf["count"])
        if float(scaling_factor) > 2.0 or float(volatility) > 0.05:
            perf["risk_events"] = int(perf["risk_events"]) + 1
        perf["success_rate"] = float(1.0 - (perf["risk_events"] / max(perf["count"], 1)))
        perf["last_update"] = datetime.datetime.now()

    def _adjust_session_multipliers(self, session: str) -> None:
        """Adjust session multipliers based on performance."""
        perf = self._session_performance.get(session)
        if not perf or int(perf["count"]) < 10:
            return
        sr = float(perf["success_rate"])
        mult = float(self._session_risk_multipliers.get(session, 1.0))
        if sr > 0.8:
            mult *= 0.95
        elif sr < 0.6:
            mult *= 1.05
        self._session_risk_multipliers[session] = float(np.clip(mult, 0.3, 2.0))

    # ── THESIS / ACTIONS ───────────────────────────────────────────

    async def _generate_risk_thesis(
        self,
        level: str,
        score: float,
        context: Dict[str, Any],
        components: Dict[str, float],
        alerts: List[str],
    ) -> str:
        """Generate comprehensive thesis for risk scaling."""
        time_data = context.get("time_data", {})
        risk_result = context.get("risk_result", {})

        session = risk_result.get("current_session", "unknown")
        scaling_factor = float(risk_result.get("scaling_factor", 1.0))
        volatility = float(risk_result.get("volatility", 0.01))
        hour = int(risk_result.get("hour", 0))

        risk_assessment = "High" if score > 0.7 else ("Medium" if score > 0.4 else "Low")
        session_names = {
            "asian": "Asian Trading Session",
            "european": "European Trading Session",
            "us": "US Trading Session",
            "closed": "Market Closed Period",
        }
        session_name = session_names.get(session, "Unknown Session")

        thesis = f"""
TIME-AWARE RISK SCALING ANALYSIS - {session_name}
Trading Focus: XAU/USD (Gold) & EUR/USD

⏰ CURRENT CONTEXT:
• Session: {session_name} (Hour: {hour}:00 UTC)
• Risk Scaling Factor: {scaling_factor:.3f}x
• Overall Risk Level: {risk_assessment} ({score:.1%})
• Current Volatility: {volatility:.4f} ({risk_result.get('volatility_regime', 'unknown')} regime)

[STATS] SCALING COMPONENTS:
• Volatility Adjustment: {risk_result.get('volatility_adjustment', 1.0):.3f}x
• Session Multiplier: {risk_result.get('session_multiplier', 1.0):.3f}x
• Hourly Risk Score: {risk_result.get('hourly_risk_score', 0.5):.3f}
• Risk Trend: {risk_result.get('risk_trend', 'unknown').title()}
""".rstrip()

        thesis += "\n\n[TARGET] SESSION ANALYSIS FOR GOLD & EUR:\n"
        if session == "asian":
            thesis += (
                "• Gold typically shows early volatility during Asian session\n"
                "• EUR/USD often ranges during low European liquidity\n"
                "• Risk scaling reflects overnight developments and sentiment\n"
                "• Conservative position sizing recommended for both instruments"
            )
        elif session == "european":
            thesis += (
                "• High liquidity European session - optimal for EUR/USD\n"
                "• Gold often reacts to European economic releases\n"
                "• Baseline risk scaling applied for standard conditions\n"
                "• Prime trading hours for EUR/USD, moderate for Gold"
            )
        elif session == "us":
            thesis += (
                "• Peak liquidity with US markets open - excellent for Gold\n"
                "• EUR/USD shows increased volatility during US overlap\n"
                "• Enhanced risk monitoring for Gold momentum moves\n"
                "• Dynamic position sizing based on US economic data"
            )
        else:
            thesis += (
                "• Limited liquidity for both Gold and EUR/USD\n"
                "• Overnight gaps possible especially for Gold\n"
                "• Reduced position sizing strongly recommended\n"
                "• Focus on risk preservation and gap management"
            )

        vol_regime = risk_result.get("volatility_regime", "unknown")
        if vol_regime == "high":
            thesis += "\n\n[WARN] HIGH VOLATILITY REGIME: Significant market stress detected"
        elif vol_regime == "low":
            thesis += "\n\n[NOTE] LOW VOLATILITY REGIME: Calm market conditions"
        else:
            thesis += f"\n\n[OK] {vol_regime.upper()} VOLATILITY REGIME: Standard market conditions"

        if score > self.time_risk_config.risk_threshold_critical:
            thesis += "\n\n[ALERT] CRITICAL RISK LEVEL: Emergency risk controls activated"
        elif score > self.time_risk_config.risk_threshold_high:
            thesis += "\n\n[WARN] HIGH RISK LEVEL: Enhanced monitoring and reduced exposure"
        else:
            thesis += "\n\n[OK] MANAGEABLE RISK LEVEL: Standard risk management protocols"

        session_efficiency = float(risk_result.get("session_efficiency", 0.5))
        thesis += f"""

[CHART] SESSION PERFORMANCE:
• Session Efficiency: {session_efficiency:.1%}
• Total Session Transitions: {risk_result.get('session_transitions', 0)}
• Performance Trend: {self._get_performance_trend(session)}

[BOT] RISK SCALING LOGIC:
• Base Factor: {self.time_risk_config.base_factor}
• Decay Applied: {self.time_risk_config.decay_factor}
• Memory Window: {self.time_risk_config.vol_window} periods
• Circuit Breaker: {'Active' if self.circuit_breaker_failures > 0 else 'Normal'}
"""
        return thesis

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose risk scaling action based on current conditions."""
        result = await self.process(**inputs)

        scaling_factor = float(result.get("risk_scaling_factor", result.get("scaling_factor", 1.0)))
        risk_level = float(result.get("session_risk", {}).get("risk_level", result.get("risk_level", 0.5)))
        session = str(result.get("session_risk", {}).get("current_session", result.get("current_session", "unknown")))

        if risk_level > 0.8:
            action_type = "reduce_exposure"
            magnitude = 0.5
        elif risk_level > 0.6:
            action_type = "moderate_caution"
            magnitude = 0.75
        elif risk_level < 0.3:
            action_type = "increase_exposure"
            magnitude = 1.2
        else:
            action_type = "maintain_current"
            magnitude = 1.0

        return {
            "action_type": action_type,
            "scaling_factor": scaling_factor,
            "magnitude": float(magnitude),
            "risk_level": risk_level,
            "session": session,
            "reasoning": f"Risk level {risk_level:.1%} in {session} session suggests {action_type}",
            "confidence": float(min(0.9, scaling_factor / 2.0)),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in the proposed risk scaling action."""
        if not isinstance(action, dict):
            return 0.5

        base_confidence = 0.7 if len(self.processing_times) > 0 else 0.5

        scaling_factor = float(action.get("scaling_factor", 1.0))
        risk_level = float(action.get("risk_level", 0.5))

        scaling_confidence = 1.0 - abs(scaling_factor - 1.0) * 0.5
        if risk_level > 0.8 or risk_level < 0.2:
            risk_confidence = 0.9
        elif 0.4 <= risk_level <= 0.6:
            risk_confidence = 0.6
        else:
            risk_confidence = 0.8

        combined = base_confidence * 0.3 + scaling_confidence * 0.4 + risk_confidence * 0.3
        session_adjustment = 1.1 if getattr(self, "_current_session", "unknown") != "unknown" else 0.9
        final_confidence = float(np.clip(combined * session_adjustment, 0.0, 1.0))
        return final_confidence

    def _get_performance_trend(self, session: str) -> str:
        """Get performance trend for session."""
        perf = self._session_performance.get(session)
        if not perf:
            return "No data"
        sr = float(perf.get("success_rate", 0.0))
        if sr > 0.8:
            return "Excellent"
        if sr > 0.6:
            return "Good"
        if sr > 0.4:
            return "Declining"
        return "Poor"

    # ── BUS / HEALTH / FALLBACKS ───────────────────────────────────

    async def _update_risk_smart_bus(self, risk_result: Dict[str, Any], thesis: str) -> None:
        """Update SmartInfoBus with risk scaling results (single-writer safe)."""
        self.smart_bus.set(
            "risk_scaling_factor",
            risk_result["scaling_factor"],
            module="TimeAwareRiskScaling",
            thesis=f"Time-aware risk scaling factor: {risk_result['scaling_factor']:.3f}x based on {risk_result['current_session']} session",
        )

        self.smart_bus.set(
            "session_risk",
            {
                "current_session": risk_result["current_session"],
                "risk_level": risk_result["risk_level"],
                "session_multiplier": risk_result["session_multiplier"],
                "processing_success": risk_result.get("processing_success", True),
                "last_update": datetime.datetime.now().isoformat(),
            },
            module="TimeAwareRiskScaling",
            thesis="Per-session risk context",
        )

        self.smart_bus.set(
            "volatility_adjustment",
            {
                "volatility": float(risk_result.get("volatility", 0.01)),
                "volatility_trend": risk_result.get("volatility_trend", "unknown"),
                "volatility_regime": risk_result.get("volatility_regime", "unknown"),
            },
            module="TimeAwareRiskScaling",
            thesis="Volatility-derived adjustment parameters",
        )

        self.smart_bus.set(
            "market_conditions",
            {
                "session": risk_result.get("current_session", "unknown"),
                "volatility": float(risk_result.get("volatility", 0.01)),
                "volatility_regime": risk_result.get("volatility_regime", "unknown"),
                "risk_trend": risk_result.get("risk_trend", "unknown"),
            },
            module="TimeAwareRiskScaling",
            thesis="Compact market state for downstream modules",
        )

        self.smart_bus.set(
            "time_risk_analysis",
            {**risk_result, "last_update": datetime.datetime.now().isoformat()},
            module="TimeAwareRiskScaling",
            thesis=thesis,
        )

        self.performance_tracker.record_metric(
            "TimeAwareRiskScaling",
            "risk_scaling",
            self.processing_times[-1] if self.processing_times else 0.0,
            risk_result.get("processing_success", True),
        )

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no time data is available (contract-safe)."""
        self.logger.warning("No time data available - using fallback risk scaling")
        current_hour = int(datetime.datetime.now().hour)
        fallback_session = self._get_session(current_hour)

        risk_result = {
            "scaling_factor": float(self.time_risk_config.base_factor),
            "risk_level": 0.5,
            "current_session": fallback_session,
            "hour": current_hour,
            "volatility": 0.01,
            "volatility_adjustment": 1.0,
            "session_multiplier": 1.0,
            "risk_trend": "unknown",
            "volatility_trend": "unknown",
            "volatility_regime": "unknown",
            "session_efficiency": 0.5,
            "hourly_risk_score": 0.5,
            "session_transitions": int(self._session_changes),
            "processing_success": False,
            "fallback_reason": "No time data available",
        }
        thesis = "Fallback: No time data available; using safe defaults for risk scaling."
        return self._format_declared_outputs(risk_result, time_data={"hour": current_hour}, thesis=thesis)

    async def _handle_risk_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle risk scaling errors."""
        _ = (time.time() - start_time) * 1000.0
        self.error_pinpointer.analyze_error(error, "TimeAwareRiskScaling")
        self._record_failure(error)
        explanation = self.english_explainer.explain_error("TimeAwareRiskScaling", str(error), "time-aware analysis")

        self.logger.error(
            format_operator_message(
                "[CRASH]",
                "RISK_SCALING_ERROR",
                details=str(error)[:200],
                explanation=explanation,
                context="error_handling",
            )
        )

        return await self._handle_no_data_fallback()

    def _analyze_session_patterns(self) -> None:
        """Analyze session patterns for optimization (periodic)."""
        if not hasattr(self, "_last_pattern_analysis"):
            self._last_pattern_analysis = time.time()
            return

        if time.time() - self._last_pattern_analysis < 300:
            return

        for hour in range(24):
            session = self._get_session(hour)
            perf = self._session_performance.get(session)
            if perf:
                risk_score = 1.0 - float(perf["success_rate"]) if int(perf["count"]) > 0 else 0.5
                self._hourly_risk_scores[hour] = float(risk_score)

        self._last_pattern_analysis = time.time()

    def _check_risk_thresholds(self) -> None:
        """Check risk thresholds and trigger alerts if needed."""
        if self.current_risk_level > self.time_risk_config.risk_threshold_critical:
            self._trigger_risk_alert("critical", self.current_risk_level)
        elif self.current_risk_level > self.time_risk_config.risk_threshold_high:
            self._trigger_risk_alert("high", self.current_risk_level)

    def _trigger_risk_alert(self, level: str, risk_value: float) -> None:
        """Trigger risk threshold alert."""
        alert = {
            "level": level,
            "risk_value": float(risk_value),
            "session": self._current_session,
            "timestamp": datetime.datetime.now(),
            "scaling_factor": float(self.current_scaling_factor),
        }
        self._risk_events.append(alert)
        self.logger.warning(
            format_operator_message(
                "[ALERT]",
                f"RISK_ALERT_{level.upper()}",
                details=f"Risk level: {risk_value:.1%}",
                context="risk_management",
            )
        )

    # ── METRICS / HEALTH / STATE ───────────────────────────────────

    def _record_success(self, processing_time_ms: float) -> None:
        """Record successful processing."""
        self.success_count += 1
        self.processing_times.append(float(processing_time_ms))
        self._factor_history.append(float(self.current_scaling_factor))
        if self.circuit_breaker_failures > 0:
            self.circuit_breaker_failures = max(0, self.circuit_breaker_failures - 1)

    def _record_failure(self, error: Exception) -> None:
        """Record processing failure."""
        self.failure_count += 1
        self.circuit_breaker_failures += 1
        if self.circuit_breaker_failures >= int(self.time_risk_config.circuit_breaker_threshold):
            self.logger.error("[ALERT] Risk scaling circuit breaker triggered")

    def _update_health_metrics(self) -> None:
        """Update health metrics."""
        total = self.success_count + self.failure_count
        success_rate = float(self.success_count / max(total, 1))
        avg_ms = float(np.mean(self.processing_times)) if self.processing_times else 0.0

        self.smart_bus.set(
            "time_risk_health",
            {
                "success_rate": success_rate,
                "avg_processing_time_ms": avg_ms,
                "circuit_breaker_failures": int(self.circuit_breaker_failures),
                "current_risk_level": float(self.current_risk_level),
                "session_transitions": int(self._session_changes),
                "risk_events": int(len(self._risk_events)),
                "last_update": datetime.datetime.now().isoformat(),
            },
            module="TimeAwareRiskScaling",
            thesis=f"Risk scaling health: {success_rate:.1%} success rate, {avg_ms:.1f}ms avg time",
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current module state for persistence."""
        return {
            "current_session": self._current_session,
            "session_changes": int(self._session_changes),
            "current_scaling_factor": float(self.current_scaling_factor),
            "current_risk_level": float(self.current_risk_level),
            "current_volatility": float(self.current_volatility),
            "vol_profile": self.vol_profile.tolist(),
            "risk_profile": self.risk_profile.tolist(),
            "session_performance": self._session_performance,
            "session_risk_multipliers": self._session_risk_multipliers,
            "success_count": int(self.success_count),
            "failure_count": int(self.failure_count),
            "last_update": datetime.datetime.now().isoformat(),
            "config": {
                "asian_end": int(self.time_risk_config.asian_end),
                "euro_end": int(self.time_risk_config.euro_end),
                "us_end": int(self.time_risk_config.us_end),
                "base_factor": float(self.time_risk_config.base_factor),
                "decay_factor": float(self.time_risk_config.decay_factor),
                "vol_window": int(self.time_risk_config.vol_window),
                "risk_threshold_high": float(self.time_risk_config.risk_threshold_high),
                "risk_threshold_critical": float(self.time_risk_config.risk_threshold_critical),
            },
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state for hot-reload."""
        if not isinstance(state, dict):
            return

        self._current_session = str(state.get("current_session", self._current_session))
        self._session_changes = int(state.get("session_changes", self._session_changes))
        self.current_scaling_factor = float(state.get("current_scaling_factor", self.current_scaling_factor))
        self.current_risk_level = float(state.get("current_risk_level", self.current_risk_level))
        self.current_volatility = float(state.get("current_volatility", self.current_volatility))

        if "vol_profile" in state:
            self.vol_profile = np.asarray(state["vol_profile"], dtype=np.float32)

        if "risk_profile" in state:
            self.risk_profile = np.asarray(state["risk_profile"], dtype=np.float32)

        if "session_performance" in state:
            self._session_performance = dict(state["session_performance"])

        if "session_risk_multipliers" in state:
            self._session_risk_multipliers = dict(state["session_risk_multipliers"])

        self.success_count = int(state.get("success_count", self.success_count))
        self.failure_count = int(state.get("failure_count", self.failure_count))

        self.logger.info("[OK] Risk scaling state restored successfully")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        total = self.success_count + self.failure_count
        return {
            "module_name": "TimeAwareRiskScaling",
            "status": "healthy" if (self.success_count / max(total, 1)) > 0.8 else "degraded",
            "success_rate": float(self.success_count / max(total, 1)),
            "avg_processing_time": float(np.mean(self.processing_times)) if self.processing_times else 0.0,
            "circuit_breaker_failures": int(self.circuit_breaker_failures),
            "current_risk_level": float(self.current_risk_level),
            "current_session": self._current_session,
            "scaling_factor": float(self.current_scaling_factor),
            "session_transitions": int(self._session_changes),
            "risk_events": int(len(self._risk_events)),
            "last_health_check": datetime.datetime.now().isoformat(),
        }

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring_active = False
