# ─────────────────────────────────────────────────────────────
# File: modules/market/regime_performance_matrix.py
# [ROCKET] PRODUCTION-READY Regime Performance Matrix with Advanced Analytics
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# ENHANCED: Complete SmartInfoBus integration, performance tracking, thesis generation
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from modules.contracts import module_args
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple, List, Union
import datetime
from dataclasses import dataclass
import threading
import math

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
class RegimeMatrixConfig:
    """Configuration for Regime Performance Matrix"""
    n_regimes: int = 3
    decay_factor: float = 0.95
    vol_history_size: int = 500
    performance_window: int = 100
    regime_sensitivity: float = 1.0

    # Performance thresholds
    max_processing_time_ms: float = 150
    circuit_breaker_threshold: int = 3
    accuracy_threshold: float = 0.60

    # Instruments to inspect when deriving volatility from market_data
    instruments: Tuple[str, ...] = ("XAU/USD", "EUR/USD", )


# ═══════════════════════════════════════════════════════════════════
# MODULE
# ═══════════════════════════════════════════════════════════════════

@module(**module_args(
    "RegimePerformanceMatrix",
    description="Advanced regime performance tracking with stress testing and prediction accuracy",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class RegimePerformanceMatrix(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Production-grade regime performance matrix with advanced analytics.
    Robust SmartInfoBus IO, strictly typed outputs, and no cross-module collisions.
    """

    # ── LIFECYCLE ──────────────────────────────────────────────────

    def __init__(self, config: Optional[Union[RegimeMatrixConfig, Dict[str, Any]]] = None, **kwargs) -> None:
        """Initialize with comprehensive advanced systems."""
        if isinstance(config, dict):
            self.regime_config: RegimeMatrixConfig = RegimeMatrixConfig(**config)
            base_config: Dict[str, Any] = config.copy()
        elif config is None:
            self.regime_config = RegimeMatrixConfig()
            base_config = {}
        else:
            self.regime_config = config
            base_config = {}

        # Flags / safety
        self._fully_initialized: bool = False
        self._monitoring_active: bool = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Systems
        self._initialize_advanced_systems()

        # Parent (expects dict config; keep lints happy)
        super().__init__(config=base_config, **kwargs)

        # Local state
        self._initialize_matrix_state()
        self._initialize_stress_testing()
        self._start_monitoring()

        self._fully_initialized = True
        self._initialize()

        self.logger.info(
            format_operator_message(
                "[STATS]",
                "REGIME_MATRIX_INITIALIZED",
                details=f"{self.regime_config.n_regimes} regimes, decay: {self.regime_config.decay_factor}",
                result="Production-ready regime performance tracking active",
                context="system_startup",
            )
        )

    # ── ADVANCED SYSTEMS ───────────────────────────────────────────

    def _initialize_advanced_systems(self) -> None:
        """Initialize all advanced SmartInfoBus systems."""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="RegimePerformanceMatrix",
            log_path="logs/market/regime_matrix.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("RegimePerformanceMatrix", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Performance metrics
        self.processing_times: deque[float] = deque(maxlen=100)
        self.success_count: int = 0
        self.failure_count: int = 0
        self.circuit_breaker_failures: int = 0

    def _initialize_matrix_state(self) -> None:
        """Initialize regime performance matrix state."""
        n = int(self.regime_config.n_regimes)
        self.matrix: np.ndarray = np.zeros((n, n), np.float32)
        self.volatility_regimes: np.ndarray = np.array([0.1, 0.3, 0.5], np.float32)  # updated from history

        # Current state
        self._current_regime: int = 0
        self._predicted_regime: int = 0
        self.last_volatility: float = 0.0
        self.last_liquidity: float = 1.0

        # History tracking
        self.vol_history: deque[float] = deque(maxlen=int(self.regime_config.vol_history_size))
        self._performance_history: deque[float] = deque(maxlen=int(self.regime_config.performance_window))
        self._regime_history: deque[int] = deque(maxlen=200)
        self._predicted_regime_history: deque[int] = deque(maxlen=200)
        self._true_regime_history: deque[int] = deque(maxlen=200)

        # Performance tracking
        self._regime_accuracy_scores: np.ndarray = np.zeros(n, np.float32)
        self._regime_pnl_tracking: Dict[int, deque[float]] = {i: deque(maxlen=50) for i in range(n)}
        self._regime_transitions: Dict[str, Dict[str, float]] = {}

        # Regime characteristics
        self._regime_characteristics: Dict[int, Dict[str, float]] = {
            i: {
                "avg_volatility": 0.0,
                "avg_pnl": 0.0,
                "count": 0,
                "accuracy": 0.5,
                "stability_score": 0.5,
            }
            for i in range(n)
        }

    def _initialize_stress_testing(self) -> None:
        """Initialize stress testing scenarios."""
        self._stress_scenarios: Dict[str, Dict[str, float]] = {
            "flash_crash": {"vol_mult": 3.0, "liq_mult": 0.2, "duration": 5},
            "rate_spike": {"vol_mult": 2.5, "liq_mult": 0.5, "duration": 10},
            "liquidity_crisis": {"vol_mult": 1.8, "liq_mult": 0.1, "duration": 20},
            "market_meltdown": {"vol_mult": 4.0, "liq_mult": 0.15, "duration": 8},
        }
        self._stress_test_results: Dict[str, Any] = {}

    def _start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitoring_active:
            return
        self._monitoring_active = True

        def monitoring_loop() -> None:
            while self._monitoring_active:
                try:
                    self._update_health_metrics()
                    self._update_regime_accuracy()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")

        self._monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def _initialize(self) -> None:
        """Async-ish initialization (do NOT call BaseModule._initialize())."""
        if not getattr(self, "_fully_initialized", False):
            return

        self.logger.info("[RELOAD] RegimePerformanceMatrix async initialization")
        self.smart_bus.set(
            "regime_matrix_status",
            {
                "initialized": True,
                "regimes_tracked": int(self.regime_config.n_regimes),
                "accuracy_threshold": float(self.regime_config.accuracy_threshold),
                "current_regime": int(self._current_regime),
            },
            module="RegimePerformanceMatrix",
            thesis="Regime performance matrix initialization status for system awareness",
        )

        # Publish a minimal baseline health snapshot for early consumers
        try:
            self.smart_bus.set(
                "regime_matrix_health",
                {
                    "success_rate": float(self.success_count / max(int(self.success_count + self.failure_count), 1)),
                    "avg_processing_time_ms": float(self._safe_mean(self.processing_times, default=0.0)),
                    "circuit_breaker_failures": int(self.circuit_breaker_failures),
                    "overall_accuracy": float(self._calculate_overall_accuracy()),
                    "current_regime": int(self._current_regime),
                    "regime_transitions": int(len(self._regime_transitions)),
                    "last_update": datetime.datetime.now().isoformat(),
                },
                module="RegimePerformanceMatrix",
                thesis="Baseline regime matrix health published at initialization",
            )
        except Exception:
            pass

    # ── NUMERIC SAFETY ─────────────────────────────────────────────

    @staticmethod
    def _is_num(x: Any) -> bool:
        return isinstance(x, (int, float, np.floating)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

    @staticmethod
    def _safe_mean(seq: Union[List[Any], deque], default: float = 0.0) -> float:
        nums = [float(v) for v in seq if isinstance(v, (int, float, np.floating))]
        return float(np.mean(nums)) if nums else float(default)

    # Robust numeric coercion to avoid float(dict) errors
    @staticmethod
    def _to_float(x: Any, default: float = 0.0) -> float:
        try:
            if isinstance(x, (int, float, np.floating)):
                return float(x)
            if isinstance(x, dict):
                # common value holders
                for k in ("value", "overall", "avg", "mean", "score"):
                    v = x.get(k)
                    if isinstance(v, (int, float, np.floating)):
                        return float(v)
                # nested numeric list: take safe mean
                for v in x.values():
                    if isinstance(v, (list, tuple, deque, np.ndarray)):
                        try:
                            arr = [float(iv) for iv in v if isinstance(iv, (int, float, np.floating))]
                            if arr:
                                return float(np.mean(arr))
                        except Exception:
                            continue
                return float(default)
            if isinstance(x, (list, tuple, deque, np.ndarray)):
                nums = [float(v) for v in x if isinstance(v, (int, float, np.floating))]
                return float(np.mean(nums)) if nums else float(default)
            # last resort: python float
            return float(x)  # may raise
        except Exception:
            return float(default)

    # ── OUTPUT CONTRACT ────────────────────────────────────────────

    def _format_declared_outputs(
        self,
        matrix_result: Dict[str, Any],
        thesis: Optional[str] = None,
        performance_data: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Map internal results to declared provides, ensure types and defaults."""
        try:
            n = int(self.regime_config.n_regimes)

            current_regime = int(matrix_result.get("current_regime", getattr(self, "_current_regime", 0)))
            predicted_regime = int(matrix_result.get("predicted_regime", getattr(self, "_predicted_regime", 0)))
            matrix_raw = matrix_result.get("matrix", self.matrix)
            matrix_list = (np.array(matrix_raw, dtype=np.float32).reshape(n, n)).tolist()

            # FIX: Ensure regime_accuracy is handled consistently
            # If matrix_result["regime_accuracy"] is already a float, use it directly
            regime_accuracy_raw = matrix_result.get("regime_accuracy", 0.5)
            if isinstance(regime_accuracy_raw, dict):
                # If it's a dict, extract the value
                overall_accuracy = float(regime_accuracy_raw.get("value", 0.5))
            else:
                # If it's already a float/numeric, use it
                overall_accuracy = float(regime_accuracy_raw)
            
            avg_performance = float(matrix_result.get("avg_performance", 0.0))
            current_volatility = float(matrix_result.get("current_volatility", getattr(self, "last_volatility", 0.01)))
            volatility_trend = str(matrix_result.get("volatility_trend", "stable"))
            regime_characteristics = dict(matrix_result.get("regime_characteristics", self._regime_characteristics))
            processing_success = bool(matrix_result.get("processing_success", False))

            stress_results = dict(getattr(self, "_stress_test_results", {}) or {})

            # Detailed accuracy structure for consumers
            by_regime = {str(i): float(self._regime_accuracy_scores[i]) for i in range(n)}
            regime_accuracy_details = {
                "value": overall_accuracy,            # canonical overall
                "by_regime": by_regime,               # per-regime breakdown
                "current_regime_accuracy": float(self._calculate_regime_accuracy(current_regime)),
                "last_update": datetime.datetime.now().isoformat(),
            }

            # Provided keys construction
            regime_prediction = {
                "predicted": predicted_regime,
                "actual": current_regime,
                "correct": bool(current_regime == predicted_regime),
            }
            market_state = {
                "regime": current_regime,
                "volatility": current_volatility,
                "trend": volatility_trend,
            }
            performance_metrics = {
                "overall_accuracy": overall_accuracy,
                "regime_accuracy": overall_accuracy,  # Keep consistent
                "avg_performance": avg_performance,
                "processing_success": processing_success,
            }
            backtesting_data = {
                "window": int(len(self._performance_history)),
                "volatility_history": [float(v) for v in list(self.vol_history)[-50:] if self._is_num(v)],
                "timestamp": datetime.datetime.now().isoformat(),
            }

            recent_trades: List[Dict[str, Any]] = []
            if performance_data and "recent_trades" in performance_data:
                recent_trades = [t for t in performance_data["recent_trades"] if isinstance(t, dict)]
            else:
                try:
                    bus_trades = self.smart_bus.get("recent_trades", "RegimePerformanceMatrix")
                    if isinstance(bus_trades, list):
                        recent_trades = [t for t in bus_trades if isinstance(t, dict)]
                except Exception:
                    recent_trades = []

            signal = "hold"
            if overall_accuracy > max(0.6, float(self.regime_config.accuracy_threshold)):
                signal = "trade" if current_regime == predicted_regime else "reduce_exposure"
            trading_signals = {
                "signal": signal,
                "confidence": float(min(1.0, overall_accuracy + (1.0 - current_volatility) * 0.2)),
                "timestamp": datetime.datetime.now().isoformat(),
            }

            # Health/status snapshots to satisfy provides contract
            health_snapshot = {
                "success_rate": float(self.success_count / max(self.success_count + self.failure_count, 1)),
                "avg_processing_time_ms": float(self._safe_mean(self.processing_times, default=0.0)),
                "circuit_breaker_failures": int(self.circuit_breaker_failures),
                "overall_accuracy": float(self._calculate_overall_accuracy()),
                "current_regime": current_regime,
                "regime_transitions": int(len(self._regime_transitions)),
                "last_update": datetime.datetime.now().isoformat(),
            }
            status_snapshot = {
                "initialized": True,
                "regimes_tracked": n,
                "accuracy_threshold": float(self.regime_config.accuracy_threshold),
                "current_regime": current_regime,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            outputs: Dict[str, Any] = {
                # not in provides but useful blob for dashboards
                "regime_performance": {
                    "matrix": matrix_list,
                    "current_regime": current_regime,
                    "predicted_regime": predicted_regime,
                    "avg_performance": avg_performance,
                },
                # FIX: Ensure regime_accuracy is ALWAYS a structured dict to prevent subscript errors
                "regime_accuracy": {
                    "value": overall_accuracy,  # The main accuracy value
                    "overall": overall_accuracy,  # Alias for compatibility
                    "details": regime_accuracy_details,  # Full details
                    "by_regime": by_regime,  # Per-regime breakdown
                    "timestamp": datetime.datetime.now().isoformat()
                },
                "regime_accuracy_details": regime_accuracy_details,  # Keep for backward compat
                "regime_prediction": regime_prediction,
                "stress_test_results": stress_results,
                "market_regime": current_regime,  # convenience for downstreams
                "regime_data": {
                    "matrix": matrix_list,
                    "characteristics": regime_characteristics,
                    "volatility_regimes": self.volatility_regimes.tolist(),
                },
                "regime_analysis": {
                    **matrix_result,
                    "last_update": datetime.datetime.now().isoformat(),
                },
                # Mirror for contract compatibility (both keys provided)
                "regime_matrix_analysis": {
                    **matrix_result,
                    "last_update": datetime.datetime.now().isoformat(),
                },
                "market_state": market_state,
                "performance_metrics": performance_metrics,
                "backtesting_data": backtesting_data,
                "recent_trades": recent_trades,
                "trading_signals": trading_signals,
                # Health and status to satisfy provides
                "regime_matrix_health": health_snapshot,
                "regime_matrix_status": status_snapshot,
                "_thesis": thesis or "Regime performance matrix analysis generated.",
                "thesis": thesis or "Regime performance matrix analysis generated.",
            }
            if isinstance(extra, dict):
                outputs["regime_analysis"]["extra"] = extra
            return outputs

        except Exception as e:
            # Safe fallback to ensure all provides are set with proper structure
            now = datetime.datetime.now().isoformat()
            n = int(self.regime_config.n_regimes)
            fallback_accuracy = {
                "value": 0.5,
                "overall": 0.5,
                "details": {
                    "value": 0.5,
                    "by_regime": {str(i): 0.5 for i in range(n)},
                    "current_regime_accuracy": 0.5,
                    "last_update": now,
                },
                "by_regime": {str(i): 0.5 for i in range(n)},
                "timestamp": now
            }
            
            return {
                "regime_performance": {
                    "matrix": self.matrix.tolist(),
                    "current_regime": int(getattr(self, "_current_regime", 0)),
                    "predicted_regime": int(getattr(self, "_predicted_regime", 0)),
                    "avg_performance": 0.0,
                },
                "regime_accuracy": fallback_accuracy,  # Structured dict instead of float
                "regime_accuracy_details": fallback_accuracy["details"],
                "regime_prediction": {
                    "predicted": int(getattr(self, "_predicted_regime", 0)),
                    "actual": int(getattr(self, "_current_regime", 0)),
                    "correct": False,
                },
                "stress_test_results": dict(getattr(self, "_stress_test_results", {})),
                "market_regime": int(getattr(self, "_current_regime", 0)),
                "regime_data": {
                    "matrix": self.matrix.tolist(),
                    "characteristics": dict(getattr(self, "_regime_characteristics", {})),
                    "volatility_regimes": self.volatility_regimes.tolist(),
                },
                "regime_analysis": {
                    "error": str(e)[:200],
                    "processing_success": False,
                    "last_update": now,
                },
                "regime_matrix_analysis": {
                    "error": str(e)[:200],
                    "processing_success": False,
                    "last_update": now,
                },
                "market_state": {
                    "regime": int(getattr(self, "_current_regime", 0)),
                    "volatility": float(getattr(self, "last_volatility", 0.01)),
                    "trend": "unknown",
                },
                "performance_metrics": {
                    "overall_accuracy": 0.5,
                    "regime_accuracy": 0.5,
                    "avg_performance": 0.0,
                    "processing_success": False,
                },
                "backtesting_data": {"window": 0, "volatility_history": [], "timestamp": now},
                "recent_trades": [],
                "trading_signals": {"signal": "hold", "confidence": 0.5, "timestamp": now},
                "regime_matrix_health": {
                    "success_rate": 0.0,
                    "avg_processing_time_ms": 0.0,
                    "circuit_breaker_failures": int(self.circuit_breaker_failures),
                    "overall_accuracy": 0.5,
                    "current_regime": int(getattr(self, "_current_regime", 0)),
                    "regime_transitions": int(len(getattr(self, "_regime_transitions", {}))),
                    "last_update": now,
                },
                "regime_matrix_status": {
                    "initialized": True,
                    "regimes_tracked": n,
                    "accuracy_threshold": float(self.regime_config.accuracy_threshold),
                    "current_regime": int(getattr(self, "_current_regime", 0)),
                    "timestamp": now,
                },
                "_thesis": "Regime performance matrix (safe fallback)",
                "thesis": "Regime performance matrix (safe fallback)",
            }

    # ── MAIN PROCESS ───────────────────────────────────────────────

    async def process(self, **inputs) -> Dict[str, Any]:
        """Main processing method with comprehensive error handling."""
        start_time = time.time()
        try:
            performance_data = await self._extract_performance_data(**inputs)
            if not performance_data:
                return await self._handle_no_data_fallback()

            matrix_result = await self._process_regime_matrix(performance_data)
            thesis = await self._generate_matrix_thesis(performance_data, matrix_result)

            # Record timing BEFORE publishing to bus so downstream metrics reflect the real cycle time
            processing_time = (time.time() - start_time) * 1000.0
            self._record_success(processing_time)

            # Publish to bus with precise timing for performance tracker
            await self._update_matrix_smart_bus(matrix_result, thesis, processing_time_ms=processing_time)

            return self._format_declared_outputs(matrix_result, thesis=thesis, performance_data=performance_data)

        except Exception as e:
            return await self._handle_matrix_error(e, start_time)

    # ── DATA EXTRACTION ─────────────────────────────────────────────

    async def _extract_performance_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract performance data from SmartInfoBus with safe fallbacks."""
        # Predicted regime (accept both str & int inputs)
        market_regime_val = self.smart_bus.get("market_regime", "RegimePerformanceMatrix")
        if isinstance(market_regime_val, str):
            mapping = {"trending": 0, "ranging": 1, "volatile": 2}
            predicted_regime = int(mapping.get(market_regime_val.lower(), 0))
        else:
            try:
                predicted_regime = int(market_regime_val) if market_regime_val is not None else 0
            except Exception:
                predicted_regime = 0

        # Volatility
        volatility_data = self.smart_bus.get("volatility_data", "RegimePerformanceMatrix")
        if isinstance(volatility_data, (int, float, np.floating)):
            volatility = float(volatility_data)
        else:
            volatility = await self._calculate_volatility_fallback()

        # PnL or recent trades
        pnl_raw = self.smart_bus.get("pnl_data", "RegimePerformanceMatrix")
        if isinstance(pnl_raw, (int, float, np.floating)):
            pnl = float(pnl_raw)
        else:
            recent_trades = self.smart_bus.get("recent_trades", "RegimePerformanceMatrix")
            if isinstance(recent_trades, list):
                pnl_vals = []
                for t in recent_trades:
                    if isinstance(t, dict):
                        v = t.get("pnl", 0.0)
                        if isinstance(v, (int, float, np.floating)):
                            pnl_vals.append(float(v))
                pnl = float(sum(pnl_vals)) if pnl_vals else 0.0
            else:
                pnl = 0.0

        liquidity_score_raw = self.smart_bus.get("liquidity_score", "RegimePerformanceMatrix")
        liquidity_score = float(liquidity_score_raw) if isinstance(liquidity_score_raw, (int, float, np.floating)) else 1.0

        return {
            "predicted_regime": predicted_regime,
            "volatility": volatility,
            "pnl": pnl,
            "liquidity_score": liquidity_score,
            "timestamp": datetime.datetime.now(),
            "source": "smartinfobus",
        }

    async def _calculate_volatility_fallback(self) -> float:
        """Calculate volatility from market data supporting both snapshot and multi-timeframe shapes."""
        market_data = self.smart_bus.get("market_data", "RegimePerformanceMatrix")
        if not isinstance(market_data, dict) or not market_data:
            return 0.01

        # 1) Snapshot shape: {SYM: {open,high,low,close,volume,...}}
        def _from_snapshot(md: Dict[str, Any]) -> Optional[float]:
            # Without a series of closes, we cannot compute returns-based volatility.
            # This function remains for completeness and future extension.
            return None  # snapshot alone insufficient for vol calc

        # 2) Multi-timeframe shape: {SYM: {TF: {close:[...] ...}}}
        def _from_multi(md: Dict[str, Any]) -> Optional[float]:
            for code in self.regime_config.instruments:
                inst = md.get(code) or md.get(code.replace("/", "")) or md.get(code.upper()) or md.get(code.replace("/", "").upper())
                if not isinstance(inst, dict):
                    continue
                # prefer H1/H4/D1
                for tf in ("H1", "H4", "D1"):
                    tf_data = inst.get(tf)
                    if isinstance(tf_data, dict) and "close" in tf_data:
                        closes = np.asarray(tf_data["close"], dtype=np.float64)
                        if closes.size > 10:
                            denom = np.where(closes[:-1] == 0, 1.0, closes[:-1])
                            rets = np.diff(closes) / denom
                            if rets.size > 1:
                                vol = float(np.std(rets[-min(20, rets.size):]))
                                if np.isfinite(vol) and vol > 0:
                                    return vol
            return None

        # Try multi-timeframe first
        vol = _from_multi(market_data)
        if isinstance(vol, float):
            return vol

        # Try snapshot (rarely sufficient for vol calc; kept for completeness)
        vol = _from_snapshot(market_data)
        if isinstance(vol, float):
            return vol

        return 0.01

    # ── CORE LOGIC ─────────────────────────────────────────────────

    async def _process_regime_matrix(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process regime performance matrix updates."""
        predicted_regime = int(performance_data["predicted_regime"])
        volatility = float(performance_data["volatility"])
        pnl = float(performance_data["pnl"])

        # Determine true regime from volatility
        true_regime = self._determine_true_regime(volatility)

        # Update histories (only numbers!)
        self._predicted_regime_history.append(predicted_regime)
        self._true_regime_history.append(true_regime)
        self.vol_history.append(float(volatility))
        self._performance_history.append(float(pnl))
        self._regime_history.append(true_regime)

        # Exponential decay update of performance matrix
        i, j = predicted_regime, true_regime
        i = int(np.clip(i, 0, self.matrix.shape[0] - 1))
        j = int(np.clip(j, 0, self.matrix.shape[1] - 1))
        decay = float(self.regime_config.decay_factor)
        self.matrix[i, j] = self.matrix[i, j] * decay + float(pnl) * (1.0 - decay)

        # Transition handling
        if true_regime != self._current_regime:
            await self._handle_regime_transition(self._current_regime, true_regime, volatility, pnl)

        # Update current pointers
        self._current_regime = true_regime
        self._predicted_regime = predicted_regime
        self.last_volatility = volatility

        # Characteristics
        self._update_regime_characteristics(true_regime, volatility, pnl)

        # Metrics
        overall_accuracy = self._calculate_overall_accuracy()
        regime_accuracy = self._calculate_regime_accuracy(true_regime)
        avg_performance = self._safe_mean(self._performance_history, default=0.0)
        volatility_trend = self._calculate_volatility_trend()

        return {
            "current_regime": int(true_regime),
            "predicted_regime": int(predicted_regime),
            "matrix": self.matrix.tolist(),
            "overall_accuracy": float(overall_accuracy),
            "regime_accuracy": float(regime_accuracy),
            "avg_performance": float(avg_performance),
            "current_volatility": float(volatility),
            "volatility_trend": str(volatility_trend),
            "regime_characteristics": self._regime_characteristics,
            "processing_success": True,
        }

    def _determine_true_regime(self, volatility: float) -> int:
        """Determine true regime based on nearest volatility cluster."""
        if len(self.vol_history) > 50:
            self._update_volatility_regimes()

        distances = np.abs(self.volatility_regimes - float(volatility))
        idx = int(np.argmin(distances))
        return idx

    def _update_volatility_regimes(self) -> None:
        """Update volatility regime thresholds from history."""
        if len(self.vol_history) < 50:
            return
        vols = np.array([float(v) for v in self.vol_history], dtype=np.float32)
        # Percentiles (robust-ish)
        self.volatility_regimes[0] = float(np.percentile(vols, 33))
        self.volatility_regimes[1] = float(np.percentile(vols, 66))
        self.volatility_regimes[2] = float(np.percentile(vols, 90))

    async def _handle_regime_transition(self, old_regime: int, new_regime: int, volatility: float, pnl: float) -> None:
        """Handle and record regime transitions."""
        key = f"{int(old_regime)}->{int(new_regime)}"
        if key not in self._regime_transitions:
            self._regime_transitions[key] = {"count": 0, "avg_pnl": 0.0, "avg_volatility": 0.0}

        trans = self._regime_transitions[key]
        new_count = int(trans.get("count", 0)) + 1
        # Running averages
        trans["avg_pnl"] = float((trans.get("avg_pnl", 0.0) * (new_count - 1) + float(pnl)) / new_count)
        trans["avg_volatility"] = float((trans.get("avg_volatility", 0.0) * (new_count - 1) + float(volatility)) / new_count)
        trans["count"] = new_count

        self.logger.info(
            format_operator_message(
                "[RELOAD]",
                "REGIME_TRANSITION",
                instrument=f"Regime {old_regime} -> {new_regime}",
                details=f"Vol: {volatility:.4f}, PnL: {pnl:.2f}",
                context="regime_tracking",
            )
        )

    def _update_regime_characteristics(self, regime: int, volatility: float, pnl: float) -> None:
        """Update regime characteristics incrementally."""
        regime = int(regime)
        ch = self._regime_characteristics.get(regime)
        if ch is None:
            ch = {"avg_volatility": 0.0, "avg_pnl": 0.0, "count": 0, "accuracy": 0.5, "stability_score": 0.5}
            self._regime_characteristics[regime] = ch
        cnt = int(ch.get("count", 0)) + 1
        ch["count"] = cnt
        ch["avg_volatility"] = float((ch.get("avg_volatility", 0.0) * (cnt - 1) + float(volatility)) / cnt)
        ch["avg_pnl"] = float((ch.get("avg_pnl", 0.0) * (cnt - 1) + float(pnl)) / cnt)
        # Track PnL
        self._regime_pnl_tracking[regime].append(float(pnl))

    def _calculate_overall_accuracy(self) -> float:
        """Overall predicted-vs-true regime accuracy."""
        if len(self._predicted_regime_history) == 0 or len(self._predicted_regime_history) != len(self._true_regime_history):
            return 0.5
        correct = sum(1 for p, t in zip(self._predicted_regime_history, self._true_regime_history) if p == t)
        return float(correct / len(self._predicted_regime_history))

    def _calculate_regime_accuracy(self, regime: int) -> float:
        """Per-regime accuracy when that regime was true."""
        preds: List[int] = []
        truths: List[int] = []
        for p, t in zip(self._predicted_regime_history, self._true_regime_history):
            if t == regime:
                preds.append(p)
                truths.append(t)
        if not truths:
            return 0.5
        correct = sum(1 for p, t in zip(preds, truths) if p == t)
        return float(correct / len(truths))

    def _calculate_volatility_trend(self) -> str:
        """Simple volatility trend over recent observations (robust)."""
        if len(self.vol_history) < 10:
            return "stable"
        y = np.asarray([float(v) for v in list(self.vol_history)[-10:]], dtype=np.float64)
        x = np.arange(y.size, dtype=np.float64)
        try:
            # Guard against singular matrix with degenerate data
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

    # ── THESIS + BUS ───────────────────────────────────────────────

    async def _generate_matrix_thesis(self, performance_data: Dict[str, Any], matrix_result: Dict[str, Any]) -> str:
        """Generate comprehensive thesis for regime matrix."""
        current_regime = int(matrix_result["current_regime"])
        predicted_regime = int(matrix_result["predicted_regime"])
        overall_accuracy = float(matrix_result["overall_accuracy"])

        regime_names = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}
        current_name = regime_names.get(current_regime, f"Regime {current_regime}")
        predicted_name = regime_names.get(predicted_regime, f"Regime {predicted_regime}")

        prediction_status = "Correct" if current_regime == predicted_regime else "Incorrect"

        thesis = f"""
REGIME PERFORMANCE MATRIX ANALYSIS

[STATS] CURRENT STATUS:
• True Regime: {current_name} (ID: {current_regime})
• Predicted Regime: {predicted_name} (ID: {predicted_regime})
• Prediction Status: {prediction_status}
• Overall Accuracy: {overall_accuracy:.1%}

[TARGET] PERFORMANCE METRICS:
• Average Performance: ${matrix_result['avg_performance']:.2f}
• Current Volatility: {matrix_result['current_volatility']:.4f}
• Volatility Trend: {matrix_result['volatility_trend'].title()}
• Matrix Decay Factor: {self.regime_config.decay_factor}
""".rstrip()

        # Regime characteristics
        rc_lines: List[str] = []
        for rid, ch in matrix_result["regime_characteristics"].items():
            rid_int = int(rid) if not isinstance(rid, int) else rid
            nm = regime_names.get(rid_int, f"Regime {rid_int}")
            cnt = int(ch.get("count", 0))
            if cnt > 0:
                rc_lines.append(
                    f"• {nm}:\n"
                    f"  - Observations: {cnt}\n"
                    f"  - Avg Volatility: {float(ch.get('avg_volatility', 0.0)):.4f}\n"
                    f"  - Avg PnL: ${float(ch.get('avg_pnl', 0.0)):.2f}\n"
                    f"  - Accuracy: {float(ch.get('accuracy', 0.5)):.1%}"
                )
        if rc_lines:
            thesis += "\n\n[CHART] REGIME CHARACTERISTICS:\n" + "\n".join(rc_lines)

        # Accuracy assessment
        if overall_accuracy > 0.8:
            thesis += "\n\n[OK] EXCELLENT PREDICTION ACCURACY: Model performing very well"
        elif overall_accuracy > 0.6:
            thesis += "\n\n[NOTE] GOOD PREDICTION ACCURACY: Model showing solid performance"
        elif overall_accuracy > 0.4:
            thesis += "\n\n[WARN] MODERATE ACCURACY: Model needs improvement"
        else:
            thesis += "\n\n[ALERT] LOW ACCURACY: Model requires attention"

        thesis += (
            f"\n\n[SEARCH] MATRIX INSIGHTS:\n"
            f"• Total Regime Transitions: {len(self._regime_transitions)}\n"
            f"• Performance Window: {len(self._performance_history)}/{int(self.regime_config.performance_window)}\n"
            f"• Volatility History: {len(self.vol_history)}/{int(self.regime_config.vol_history_size)}\n"
            f"• Prediction vs Reality: {prediction_status} this period"
        )

        return thesis

    async def _update_matrix_smart_bus(
        self,
        matrix_result: Dict[str, Any],
        thesis: str,
        *,
        processing_time_ms: Optional[float] = None
    ) -> None:
        """Update SmartInfoBus with matrix results (scoped to this module)."""
        # Main performance data
        self.smart_bus.set(
            "regime_performance",
            {
                "matrix": matrix_result["matrix"],
                "current_regime": matrix_result["current_regime"],
                "predicted_regime": matrix_result["predicted_regime"],
                "avg_performance": matrix_result["avg_performance"],
            },
            module="RegimePerformanceMatrix",
            thesis=f"Regime performance matrix with {matrix_result['overall_accuracy']:.1%} accuracy",
        )

        # FIX: Publish regime_accuracy as a structured dict for consistency
        # This prevents downstream code from encountering subscript errors
        overall_acc = matrix_result["overall_accuracy"]
        n = int(self.regime_config.n_regimes)
        by_regime = {str(i): float(self._regime_accuracy_scores[i]) for i in range(n)}
        
        self.smart_bus.set(
            "regime_accuracy",
            {
                "value": overall_acc,  # Main value
                "overall": overall_acc,  # Alias
                "by_regime": by_regime,
                "current_regime_accuracy": float(self._calculate_regime_accuracy(matrix_result["current_regime"])),
                "timestamp": datetime.datetime.now().isoformat()
            },
            module="RegimePerformanceMatrix",
            thesis=f"Overall regime prediction accuracy: {overall_acc:.1%}",
        )

        self.smart_bus.set(
            "regime_prediction",
            {
                "predicted": matrix_result["predicted_regime"],
                "actual": matrix_result["current_regime"],
                "correct": matrix_result["current_regime"] == matrix_result["predicted_regime"],
            },
            module="RegimePerformanceMatrix",
            thesis=f"Regime prediction: {matrix_result['predicted_regime']} vs actual {matrix_result['current_regime']}",
        )

        # Do NOT publish 'market_regime' here (single-writer policy). We emit only analytics.

        self.smart_bus.set(
            "regime_data",
            {
                "matrix": matrix_result["matrix"],
                "characteristics": matrix_result["regime_characteristics"],
                "volatility_regimes": self.volatility_regimes.tolist(),
            },
            module="RegimePerformanceMatrix",
            thesis="Regime data snapshot including matrix and characteristics",
        )

        analysis_payload = {
            **matrix_result,
            "regime_transitions": self._regime_transitions,
            "volatility_regimes": self.volatility_regimes.tolist(),
            "last_update": datetime.datetime.now().isoformat(),
        }
        self.smart_bus.set("regime_analysis", analysis_payload, module="RegimePerformanceMatrix", thesis=thesis)
        # legacy/compat (kept intentionally)
        self.smart_bus.set("regime_matrix_analysis", analysis_payload, module="RegimePerformanceMatrix", thesis=thesis)

        self.smart_bus.set(
            "market_state",
            {
                "regime": matrix_result["current_regime"],
                "volatility": matrix_result["current_volatility"],
                "trend": matrix_result["volatility_trend"],
            },
            module="RegimePerformanceMatrix",
            thesis="Compact market state derived from regime analysis",
        )

        self.smart_bus.set(
            "performance_metrics",
            {
                "overall_accuracy": matrix_result["overall_accuracy"],
                "regime_accuracy": matrix_result.get("regime_accuracy", matrix_result["overall_accuracy"]),
                "avg_performance": matrix_result["avg_performance"],
                "processing_success": matrix_result["processing_success"],
            },
            module="RegimePerformanceMatrix",
            thesis="Performance metrics for regime matrix module",
        )

        self.smart_bus.set(
            "backtesting_data",
            {
                "window": len(self._performance_history),
                "volatility_history": [float(v) for v in list(self.vol_history)[-50:] if self._is_num(v)],
                "timestamp": datetime.datetime.now().isoformat(),
            },
            module="RegimePerformanceMatrix",
            thesis="Backtesting context for regime analysis consumers",
        )

        recent_trades = self.smart_bus.get("recent_trades", "RegimePerformanceMatrix") or []
        if isinstance(recent_trades, list):
            recent_trades = [t for t in recent_trades if isinstance(t, dict)]
        else:
            recent_trades = []
        self.smart_bus.set("recent_trades", recent_trades, module="RegimePerformanceMatrix", thesis="Recent trades passthrough for regime consumers")

        signal = "hold"
        if matrix_result["overall_accuracy"] > max(0.6, float(self.regime_config.accuracy_threshold)):
            signal = "trade" if matrix_result["current_regime"] == matrix_result["predicted_regime"] else "reduce_exposure"
        self.smart_bus.set(
            "trading_signals",
            {
                "signal": signal,
                "confidence": float(min(1.0, matrix_result["overall_accuracy"] + (1.0 - matrix_result["current_volatility"]) * 0.2)),
                "timestamp": datetime.datetime.now().isoformat(),
            },
            module="RegimePerformanceMatrix",
            thesis="Basic trading signal derived from regime accuracy and state",
        )

        self.smart_bus.set(
            "stress_test_results",
            dict(getattr(self, "_stress_test_results", {})),
            module="RegimePerformanceMatrix",
            thesis="Most recent regime stress test results",
        )

        # Record a precise performance metric for this cycle
        metric_ms = float(processing_time_ms if processing_time_ms is not None else (self.processing_times[-1] if self.processing_times else 0.0))
        self.performance_tracker.record_metric(
            "RegimePerformanceMatrix",
            "matrix_processing",
            metric_ms,
            matrix_result["processing_success"],
        )

    # ── FALLBACKS / ERRORS ─────────────────────────────────────────

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no performance data is available (contract-safe)."""
        self.logger.warning("No performance data available - using fallback regime matrix")
        now = datetime.datetime.now().isoformat()
        n = int(self.regime_config.n_regimes)
        matrix_result = {
            "current_regime": int(getattr(self, "_current_regime", 0)),
            "predicted_regime": int(getattr(self, "_predicted_regime", 0)),
            "matrix": self.matrix.tolist(),
            "overall_accuracy": 0.5,
            "regime_accuracy": 0.5,
            "avg_performance": 0.0,
            "current_volatility": float(getattr(self, "last_volatility", 0.01)),
            "volatility_trend": "unknown",
            "regime_characteristics": dict(getattr(self, "_regime_characteristics", {})),
            "processing_success": False,
            "fallback_reason": "No performance data available",
        }
        thesis = "Fallback: No performance data; using safe defaults for regime matrix."
        out = self._format_declared_outputs(matrix_result, thesis=thesis)
        # ensure details exist too
        try:
            acc_val = float(out.get("regime_accuracy", {}).get("value", 0.5)) if isinstance(out.get("regime_accuracy"), dict) else 0.5
        except Exception:
            acc_val = 0.5
        out["regime_accuracy_details"] = {
            "value": acc_val,
            "by_regime": {str(i): 0.5 for i in range(n)},
            "current_regime_accuracy": 0.5,
            "last_update": now,
        }
        return out

    async def _handle_matrix_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle matrix processing errors (contract-safe)."""
        _ = (time.time() - start_time) * 1000.0
        self.error_pinpointer.analyze_error(error, "RegimePerformanceMatrix")
        self._record_failure(error)
        explanation = self.english_explainer.explain_error("RegimePerformanceMatrix", str(error), "regime matrix analysis")

        self.logger.error(
            format_operator_message(
                "[CRASH]",
                "REGIME_MATRIX_ERROR",
                details=str(error)[:200],
                explanation=explanation,
                context="error_handling",
            )
        )
        return await self._handle_no_data_fallback()

    # ── METRICS / HEALTH ───────────────────────────────────────────

    def _update_regime_accuracy(self) -> None:
        """Update per-regime accuracy scores."""
        for regime in range(int(self.regime_config.n_regimes)):
            acc = self._calculate_regime_accuracy(regime)
            self._regime_accuracy_scores[regime] = float(acc)
            if regime in self._regime_characteristics:
                self._regime_characteristics[regime]["accuracy"] = float(acc)

    def _record_success(self, processing_time_ms: float) -> None:
        """Record successful processing."""
        self.success_count += 1
        self.processing_times.append(float(processing_time_ms))
        if self.circuit_breaker_failures > 0:
            self.circuit_breaker_failures = max(0, self.circuit_breaker_failures - 1)

    def _record_failure(self, error: Exception) -> None:
        """Record processing failure."""
        self.failure_count += 1
        self.circuit_breaker_failures += 1
        if self.circuit_breaker_failures >= int(self.regime_config.circuit_breaker_threshold):
            self.logger.error("[ALERT] Regime matrix circuit breaker triggered")

    def _update_health_metrics(self) -> None:
        """Update health metrics to SmartInfoBus."""
        total = self.success_count + self.failure_count
        success_rate = float(self.success_count / max(total, 1))
        avg_ms = self._safe_mean(self.processing_times, default=0.0)
        overall_accuracy = float(self._calculate_overall_accuracy())

        self.smart_bus.set(
            "regime_matrix_health",
            {
                "success_rate": success_rate,
                "avg_processing_time_ms": avg_ms,
                "circuit_breaker_failures": int(self.circuit_breaker_failures),
                "overall_accuracy": overall_accuracy,
                "current_regime": int(self._current_regime),
                "regime_transitions": int(len(self._regime_transitions)),
                "last_update": datetime.datetime.now().isoformat(),
            },
            module="RegimePerformanceMatrix",
            thesis=f"Regime matrix health: {success_rate:.1%} success rate, {overall_accuracy:.1%} accuracy",
        )

    # ── STATE IO ───────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Get current module state for persistence."""
        return {
            "matrix": self.matrix.tolist(),
            "current_regime": int(self._current_regime),
            "predicted_regime": int(self._predicted_regime),
            "volatility_regimes": self.volatility_regimes.tolist(),
            "regime_characteristics": self._regime_characteristics,
            "regime_transitions": self._regime_transitions,
            "accuracy_scores": self._regime_accuracy_scores.tolist(),
            "success_count": int(self.success_count),
            "failure_count": int(self.failure_count),
            "last_update": datetime.datetime.now().isoformat(),
            "config": {
                "n_regimes": int(self.regime_config.n_regimes),
                "decay_factor": float(self.regime_config.decay_factor),
                "vol_history_size": int(self.regime_config.vol_history_size),
                "performance_window": int(self.regime_config.performance_window),
                "accuracy_threshold": float(self.regime_config.accuracy_threshold),
            },
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state for hot-reload."""
        if not isinstance(state, dict):
            return

        if "matrix" in state:
            mat = np.array(state["matrix"], dtype=np.float32)
            if mat.shape == self.matrix.shape:
                self.matrix = mat

        self._current_regime = int(state.get("current_regime", self._current_regime))
        self._predicted_regime = int(state.get("predicted_regime", self._predicted_regime))

        if "volatility_regimes" in state:
            vr = np.array(state["volatility_regimes"], dtype=np.float32)
            if vr.size == self.volatility_regimes.size:
                self.volatility_regimes = vr

        if "regime_characteristics" in state:
            self._regime_characteristics = dict(state["regime_characteristics"])

        if "regime_transitions" in state:
            self._regime_transitions = dict(state["regime_transitions"])

        self.success_count = int(state.get("success_count", self.success_count))
        self.failure_count = int(state.get("failure_count", self.failure_count))

        self.logger.info("[OK] Regime matrix state restored successfully")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        total = self.success_count + self.failure_count
        overall_accuracy = float(self._calculate_overall_accuracy())

        return {
            "module_name": "RegimePerformanceMatrix",
            "status": "healthy" if (self.success_count / max(total, 1)) > 0.8 else "degraded",
            "success_rate": float(self.success_count / max(total, 1)),
            "avg_processing_time": self._safe_mean(self.processing_times, default=0.0),
            "circuit_breaker_failures": int(self.circuit_breaker_failures),
            "overall_accuracy": overall_accuracy,
            "current_regime": int(self._current_regime),
            "regime_transitions": int(len(self._regime_transitions)),
            "last_health_check": datetime.datetime.now().isoformat(),
        }

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring_active = False

    # ── ACTIONS / CONFIDENCE ───────────────────────────────────────

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose regime-based action recommendations."""
        try:
            performance_data = await self._extract_performance_data(**inputs)
            if not performance_data:
                return {
                    "action": "hold",
                    "regime_confidence": 0.5,
                    "rationale": "Insufficient data for regime analysis",
                    "risk_level": "medium",
                }

            current_regime = int(self._current_regime)
            predicted_regime = int(performance_data["predicted_regime"])
            volatility = float(performance_data["volatility"])
            overall_accuracy = float(self._calculate_overall_accuracy())

            if current_regime == 0:  # Low vol
                action = "buy" if (predicted_regime == current_regime and overall_accuracy > 0.7) else "hold"
                risk_level = "low"
            elif current_regime == 1:  # Medium vol
                action = "trade" if overall_accuracy > 0.6 else "reduce_exposure"
                risk_level = "medium"
            else:  # High vol
                action = "defensive" if volatility > 0.3 else "cautious_trade"
                risk_level = "high"

            regime_confidence = float(min(1.0, overall_accuracy + (1.0 - volatility) * 0.3))
            regime_names = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}
            current_name = regime_names.get(current_regime, f"Regime {current_regime}")
            rationale = f"Current regime: {current_name}, Prediction accuracy: {overall_accuracy:.1%}, Volatility: {volatility:.3f}"

            return {
                "action": action,
                "regime_confidence": regime_confidence,
                "rationale": rationale,
                "risk_level": risk_level,
                "current_regime": current_regime,
                "predicted_regime": predicted_regime,
                "regime_accuracy": overall_accuracy,
            }

        except Exception as e:
            self.logger.error(f"Error in propose_action: {e}")
            return {"action": "hold", "regime_confidence": 0.5, "rationale": f"Error in regime analysis: {str(e)}", "risk_level": "medium"}

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in the proposed action."""
        try:
            overall_accuracy = self._to_float(self._calculate_overall_accuracy(), 0.5)
            # guard regimes in case they were set from malformed state
            cur_reg = int(self._to_float(getattr(self, "_current_regime", 0), 0.0))
            pred_reg = int(self._to_float(getattr(self, "_predicted_regime", 0), 0.0))
            total_reg = max(int(self._to_float(getattr(self.regime_config, "n_regimes", 3), 3.0)) - 1, 1)
            regime_stability = 1.0 - abs(cur_reg - pred_reg) / total_reg
            data_quality = float(min(1.0, len(self.vol_history) / max(float(self._to_float(getattr(self.regime_config, "vol_history_size", 500), 500.0)), 1.0)))

            if len(self._performance_history) > 10:
                perf_std = float(np.std([self._to_float(v, 0.0) for v in list(self._performance_history)]))
                perf_consistency = float(max(0.0, 1.0 - perf_std / 100.0))
            else:
                perf_consistency = 0.5

            denom = max(int(self._to_float(getattr(self.regime_config, "n_regimes", 3), 3.0)) ** 2, 1)
            try:
                nonzero = float(np.count_nonzero(self.matrix))
            except Exception:
                # if matrix somehow corrupted, fall back to 0 coverage
                nonzero = 0.0
            matrix_coverage = float(max(0.0, min(1.0, nonzero / denom)))

            confidence = (
                overall_accuracy * 0.40
                + regime_stability * 0.25
                + data_quality * 0.15
                + perf_consistency * 0.10
                + matrix_coverage * 0.10
            )

            action_type = action.get("action", "hold")
            if action_type == "defensive" and cur_reg == 2:
                confidence *= 1.10
            elif action_type == "buy" and cur_reg == 0:
                confidence *= 1.05
            elif action_type in ("trade", "cautious_trade") and overall_accuracy < 0.6:
                confidence *= 0.80

            return float(max(0.0, min(1.0, confidence)))
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
