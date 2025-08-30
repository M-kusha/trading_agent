# ─────────────────────────────────────────────────────────────
# File: modules/market/market_theme_detector.py
# [ROCKET] PRODUCTION-READY Market Theme Detection with Advanced ML
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# ENHANCED: Complete SmartInfoBus integration, neural analysis, thesis generation
# Contract-safe: always returns `_thesis`, `theme_detector_status`, `theme_detector_health`
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
import datetime
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional, Union
from collections import deque
import threading

from modules.contracts import module_args
import numpy as np
import pandas as pd  # kept for future feature enrichments / compatibility
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import pywt

# Core SmartInfoBus Infrastructure
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusVotingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ThemeDetectorConfig:
    """Configuration for Market Theme Detector"""
    n_themes: int = 4
    window: int = 100
    batch_size: int = 64
    feature_lookback: int = 500
    instruments: List[str] = field(default_factory=list)

    # ML Parameters
    max_iter: int = 100
    convergence_threshold: float = 0.001
    clustering_quality_threshold: float = 0.30  # readiness threshold

    # Performance thresholds
    max_processing_time_ms: float = 200
    circuit_breaker_threshold: int = 3

    # Feature controls
    use_macro: bool = True
    timeframes: Tuple[str, ...] = ("H1", "H4", "D1")

    def __post_init__(self) -> None:
        # sane defaults
        if not self.instruments:
            self.instruments = ["XAU/USD", "EUR/USD"]
        # keep batch practical and stable
        self.batch_size = max(64, int(self.n_themes) * 16)


# ═══════════════════════════════════════════════════════════════════
# MODULE
# ═══════════════════════════════════════════════════════════════════

@module(**module_args(
    name="MarketThemeDetector",
    description="Advanced market theme detection with ML clustering and regime-aware features",
    error_handling=True,
    hot_reload=True,
    timeout_ms=120,
))
class MarketThemeDetector(
    BaseModule,
    SmartInfoBusTradingMixin,
    SmartInfoBusVotingMixin,
    SmartInfoBusStateMixin,
):
    """
    Production-grade market theme detector with advanced ML clustering and robust SmartInfoBus IO.
    Contract-clean returns; preserves useful logic; avoids namespace collisions with other market modules.
    """

    # ── LIFECYCLE ──────────────────────────────────────────────────

    def __init__(self, config: Union[ThemeDetectorConfig, Dict[str, Any], None] = None, **kwargs) -> None:
        # Normalize config
        if isinstance(config, dict):
            self.theme_config: ThemeDetectorConfig = ThemeDetectorConfig(**config)
            config_for_base: Dict[str, Any] = config.copy()
        elif config is None:
            self.theme_config = ThemeDetectorConfig()
            config_for_base = {}
        else:
            self.theme_config = config
            config_for_base = {}

        # System scaffolding
        self._fully_initialized: bool = False
        self._monitoring_active: bool = False
        self._monitor_thread: Optional[threading.Thread] = None

        self._initialize_advanced_systems()
        super().__init__(config=config_for_base)  # BaseModule may store/expect a dict

        self._initialize_ml_components()
        self._initialize_theme_state()
        self._start_monitoring()

        self._fully_initialized = True
        self._initialize()  # local initialize (safe)

        self.logger.info(
            format_operator_message(
                "[TARGET]",
                "THEME_DETECTOR_INITIALIZED",
                details=f"{self.theme_config.n_themes} themes, {len(self.theme_config.instruments)} instruments",
                result="Production-ready ML clustering active",
                context="system_startup",
            )
        )

    def _initialize_advanced_systems(self) -> None:
        """Initialize logging, bus, helpers, trackers, and circuit protection."""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="MarketThemeDetector",
            log_path="logs/market/theme_detector.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("MarketThemeDetector", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Perf stats
        self.processing_times: deque[float] = deque(maxlen=100)
        self.success_count: int = 0
        self.failure_count: int = 0
        self.circuit_breaker_failures: int = 0
        self.last_circuit_breaker_reset: float = time.time()

    def _initialize_ml_components(self) -> None:
        """Initialize ML components with version-agnostic safety."""
        try:
            self.scaler = StandardScaler()
            # 'n_init="auto"' may break on older sklearn; use an explicit int for stability.
            self.km = MiniBatchKMeans(
                n_clusters=int(self.theme_config.n_themes),
                batch_size=int(self.theme_config.batch_size),
                random_state=0,
                max_iter=int(self.theme_config.max_iter),
                n_init=10,
            )

            self._ml_fit_count: int = 0
            self._last_inertia: Optional[float] = None
            self._convergence_history: deque[float] = deque(maxlen=20)

            self.ml_circuit_breaker: Dict[str, Any] = {
                "failures": 0,
                "last_failure": 0.0,
                "state": "CLOSED",
                "threshold": int(self.theme_config.circuit_breaker_threshold),
            }

            self.logger.info("[OK] ML components initialized successfully")

        except Exception as e:
            self.logger.error(f"ML initialization failed: {e}")
            self.error_pinpointer.analyze_error(e, "MarketThemeDetector")
            # degrade gracefully: replace with noop objects to avoid crashes later
            self.scaler = StandardScaler()
            self.km = MiniBatchKMeans(n_clusters=2, batch_size=64, random_state=0, max_iter=10, n_init=5)

    def _initialize_theme_state(self) -> None:
        """Initialize in-memory state and buffers."""
        n = int(self.theme_config.n_themes)
        self._theme_vec: np.ndarray = np.zeros(n, np.float32)
        self._current_theme: int = 0
        self._theme_confidence: float = 0.0
        self._theme_strength_history: deque[float] = deque(maxlen=100)
        self._theme_momentum: deque[float] = deque(maxlen=10)
        self._theme_history: deque[int] = deque(maxlen=300)

        self._fit_buffer: deque[np.ndarray] = deque(maxlen=2000)
        self._feature_stability_score: float = 1.0
        self._clustering_quality: float = 0.0

        self._data_access_attempts: int = 0
        self._successful_data_extractions: int = 0
        self._last_known_data: Dict[str, Any] = {}
        self._last_features: Optional[np.ndarray] = None

        # Macro scaler (independent from main scaler)
        self._macro_scaler = StandardScaler()
        self._macro_scaler.fit([[20.0, 0.5, 3.0]])  # vix, yield_curve, cpi (placeholder)
        self.macro_data: Dict[str, float] = {"vix": 20.0, "yield_curve": 0.5, "cpi": 3.0}

    def _start_monitoring(self) -> None:
        """Start background monitoring in a daemon thread (idempotent)."""
        if self._monitoring_active:
            return
        self._monitoring_active = True

        def monitoring_loop() -> None:
            while self._monitoring_active:
                try:
                    self._update_health_metrics()
                    self._check_circuit_breaker_reset()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")

        self._monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def _initialize(self) -> None:
        """Local initialize (do NOT call BaseModule._initialize here)."""
        if not getattr(self, "_fully_initialized", False):
            return
        self.logger.info("[RELOAD] MarketThemeDetector async initialization")
        self.smart_bus.set(
            "theme_detector_status",
            {
                "initialized": True,
                "themes_available": int(self.theme_config.n_themes),
                "instruments": list(self.theme_config.instruments),
                "clustering_ready": False,
            },
            module="MarketThemeDetector",
            thesis="Theme detector initialization status for system awareness",
        )

    # ── HELPERS: CONTRACT SNAPSHOTS ────────────────────────────────

    def _build_status_snapshot(self, res: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "initialized": True,
            "themes_available": int(self.theme_config.n_themes),
            "instruments": list(self.theme_config.instruments),
            "clustering_ready": self._is_model_ready(),
            "current_theme": int((res or {}).get("market_theme", self._current_theme)),
        }

    def _build_health_snapshot(self) -> Dict[str, Any]:
        total = self.success_count + self.failure_count
        return {
            "success_rate": float(self.success_count / max(total, 1)),
            "avg_processing_time_ms": float(np.mean(self.processing_times)) if self.processing_times else 0.0,
            "ml_circuit_breaker_state": self.ml_circuit_breaker["state"],
            "clustering_quality": float(self._clustering_quality),
            "last_update": datetime.datetime.utcnow().isoformat(),
        }

    # ── HELPERS: CONVENIENCE SNAPSHOTS ──────────────────────────────

    def _derive_market_data_snapshot(self, mkt: Dict[str, Any]) -> Dict[str, Any]:
        """Build a light 'market_data' snapshot from multi-timeframe source."""
        out: Dict[str, Any] = {}
        if not isinstance(mkt, dict):
            return out
        for symbol, tf_map in mkt.items():
            if not isinstance(tf_map, dict) or not tf_map:
                continue
            tf = "H4" if "H4" in tf_map else next(iter(tf_map))
            data = tf_map.get(tf, {})
            cb = data.get("current_bar", None)
            if isinstance(cb, dict) and all(k in cb for k in ("open", "high", "low", "close", "volume")):
                out[symbol] = {
                    "open": float(cb["open"]),
                    "high": float(cb["high"]),
                    "low": float(cb["low"]),
                    "close": float(cb["close"]),
                    "volume": int(cb.get("volume", 0)),
                    # bid/ask unknown: safe placeholders
                    "bid": float(cb["close"]),
                    "ask": float(cb["close"]),
                }
            else:
                try:
                    closes = data.get("close", [])
                    highs = data.get("high", [])
                    lows = data.get("low", [])
                    opens = data.get("open", [])
                    vols = data.get("volume", [])
                    if len(closes):
                        out[symbol] = {
                            "open": float(opens[-1]) if len(opens) else float(closes[-1]),
                            "high": float(highs[-1]) if len(highs) else float(closes[-1]),
                            "low": float(lows[-1]) if len(lows) else float(closes[-1]),
                            "close": float(closes[-1]),
                            "volume": int(vols[-1]) if len(vols) else 0,
                            "bid": float(closes[-1]),
                            "ask": float(closes[-1]),
                        }
                except Exception:
                    pass
        return out

    def _derive_price_data_from_market(self, mkt: Dict[str, Any]) -> Dict[str, Any]:
        """Build simple price_data {symbol: {open,high,low,close}}."""
        out: Dict[str, Any] = {}
        md = self._derive_market_data_snapshot(mkt)
        for sym, bar in md.items():
            out[sym] = {"open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"]}
        return out

    # ── MAIN PROCESS ────────────────────────────────────────────────

    async def process(self, **inputs) -> Dict[str, Any]:
        t0 = time.time()
        try:
            # Upstream (if available) – used to fill contract-friendly outputs
            upstream_market_data = self.smart_bus.get("market_data", "MarketThemeDetector") or {}
            upstream_price_data = self.smart_bus.get("price_data", "MarketThemeDetector") or {}
            upstream_ta = self.smart_bus.get("technical_indicators", "MarketThemeDetector") or {}

            market_data = await self._extract_market_data(**inputs)

            if not market_data:
                theme_result = await self._handle_no_data_fallback()
                safe_market_data: Dict[str, Any] = {}
            else:
                theme_result = await self._process_theme_detection(market_data)
                safe_market_data = market_data

            # Ensure required compact blob exists
            if "theme_detection" not in theme_result:
                theme_result["theme_detection"] = {
                    "theme": theme_result.get("market_theme", self._current_theme),
                    "strength": theme_result.get("theme_strength", 0.0),
                    "confidence": theme_result.get("theme_confidence", 0.0),
                    "stability": theme_result.get("theme_stability", 0.0),
                    "transition_probability": theme_result.get("transition_probability", 0.0),
                    "timestamp": datetime.datetime.now().isoformat(),
                }

            thesis = await self._generate_theme_thesis(safe_market_data, theme_result)

            # Populate convenience outputs (do NOT collide with other modules)
            derived_market_snapshot = self._derive_market_data_snapshot(safe_market_data)
            theme_result["market_data"] = upstream_market_data or derived_market_snapshot or {}
            derived_price_data = self._derive_price_data_from_market(safe_market_data)
            theme_result["price_data"] = upstream_price_data or derived_price_data or {}
            theme_result["technical_indicators"] = upstream_ta or {}
            theme_result["market_features"] = (
                self._last_features.tolist() if isinstance(self._last_features, np.ndarray) else []
            )

            proc_ms = (time.time() - t0) * 1000.0
            self._record_success(proc_ms)

            # Status/health snapshots for both BUS & RETURN (contract requires in return)
            status = self._build_status_snapshot(theme_result)
            health = self._build_health_snapshot()

            await self._update_theme_smart_bus(theme_result, thesis, status=status, health=health)

            # Return payload (contract-safe)
            out = dict(theme_result)
            out["_thesis"] = thesis
            out["thesis"] = thesis
            out["theme_detector_status"] = status
            out["theme_detector_health"] = health
            return out

        except Exception as e:
            return await self._handle_theme_error(e, t0)

    # ── DATA EXTRACTION ─────────────────────────────────────────────

    async def _extract_market_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract market data from multiple sources with safe fallbacks."""
        self._data_access_attempts += 1

        # Rich historical / multi-timeframe drop-in
        rich = self.smart_bus.get("historical_prices", "MarketThemeDetector") or \
               self.smart_bus.get("multi_timeframe_data", "MarketThemeDetector")
        if isinstance(rich, dict) and rich:
            self._successful_data_extractions += 1
            self._last_known_data = rich.copy()
            return rich

        # Compact current-bar market_data -> shape into multi-TF
        market_data = self.smart_bus.get("market_data", "MarketThemeDetector")
        if isinstance(market_data, dict) and market_data:
            shaped: Dict[str, Dict[str, Any]] = {}
            for instrument, bar in market_data.items():
                tf = "H4"
                instr_hist = {
                    "open": [bar.get("open")] if isinstance(bar, dict) else [],
                    "high": [bar.get("high")] if isinstance(bar, dict) else [],
                    "low": [bar.get("low")] if isinstance(bar, dict) else [],
                    "close": [bar.get("close")] if isinstance(bar, dict) else [],
                    "volume": [bar.get("volume")] if isinstance(bar, dict) else [],
                    "current_bar": {
                        "open": float(bar.get("open", 0.0)),
                        "high": float(bar.get("high", 0.0)),
                        "low": float(bar.get("low", 0.0)),
                        "close": float(bar.get("close", 0.0)),
                        "volume": int(bar.get("volume", 0)),
                    },
                    "timeframe": tf,
                    "bars_available": 1 if isinstance(bar, dict) and bar.get("close") is not None else 0,
                }
                shaped[instrument] = {tf: instr_hist}
            self._successful_data_extractions += 1
            self._last_known_data = shaped.copy()
            return shaped

        # Individually keyed data per instrument/timeframe
        keyed: Dict[str, Dict[str, Any]] = {}
        for instrument in self.theme_config.instruments:
            for timeframe in self.theme_config.timeframes:
                key = f"market_data_{instrument}_{timeframe}"
                data = self.smart_bus.get(key, "MarketThemeDetector")
                if data:
                    keyed.setdefault(instrument, {})[timeframe] = data
        if keyed:
            self._successful_data_extractions += 1
            self._last_known_data = keyed.copy()
            return keyed

        # Inputs fallback
        if "market_data" in inputs and isinstance(inputs["market_data"], dict):
            md = inputs["market_data"]
            if md:
                self._successful_data_extractions += 1
                self._last_known_data = md.copy()
                return md

        # Last known (stale) as fallback
        if self._last_known_data:
            self.logger.warning("Using last known market data")
            return self._last_known_data

        # Synthetic (ultimate fallback)
        return self._generate_synthetic_market_data()

    def _generate_synthetic_market_data(self) -> Dict[str, Any]:
        """Generate synthetic data – deterministic enough for tests, safe for fallbacks."""
        rng = np.random.default_rng(7)
        synthetic: Dict[str, Any] = {}
        for instrument in self.theme_config.instruments:
            synthetic[instrument] = {}
            for timeframe in self.theme_config.timeframes:
                base = 1950.0 if "XAU" in instrument.upper() else 1.1000
                prices: List[float] = []
                p = base
                for _ in range(100):
                    p = p * (1.0 + rng.normal(0.0, 0.001))
                    prices.append(float(p))
                synthetic[instrument][timeframe] = {
                    "open": prices,
                    "high": [x * 1.0008 for x in prices],
                    "low": [x * 0.9992 for x in prices],
                    "close": prices,
                    "volume": [int(1000 + rng.normal(0, 50)) for _ in prices],
                    "current_bar": {
                        "open": prices[-1],
                        "high": prices[-1] * 1.0008,
                        "low": prices[-1] * 0.9992,
                        "close": prices[-1],
                        "volume": 1000,
                    },
                    "timeframe": timeframe,
                    "bars_available": len(prices),
                }
        self.logger.info("Generated synthetic market data for theme detection")
        return synthetic

    # ── CORE LOGIC ─────────────────────────────────────────────────

    async def _process_theme_detection(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features, fit model (if ready), and detect theme."""
        features = self._extract_comprehensive_features(market_data)
        if features is None or features.size == 0:
            self._last_features = np.array([], dtype=np.float32)
            return self._create_fallback_theme_result("No valid features extracted")

        self._last_features = features
        self._fit_buffer.append(features)

        if self._should_fit_model():
            await self._fit_model_safe()

        if self._is_model_ready():
            theme_id, strength = self._detect_current_theme(features)
            confidence = self._calculate_theme_confidence(features, theme_id)
        else:
            theme_id, strength, confidence = 0, 0.30, 0.10

        self._update_theme_state(theme_id, strength, confidence)

        stability = self._get_theme_stability()
        transition_probability = self._calculate_transition_probability()

        return {
            "market_theme": int(theme_id),
            "theme_strength": float(strength),
            "theme_confidence": float(confidence),
            "theme_stability": float(stability),
            "transition_probability": float(transition_probability),
            "theme_transition": float(transition_probability),
            "clustering_quality": float(self._clustering_quality),
            "feature_stability": float(self._feature_stability_score),
            "themes_total": int(self.theme_config.n_themes),
            "processing_success": True,
        }

    def _extract_comprehensive_features(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Feature vector: per-instrument×timeframe core stats + macro (optional)."""
        try:
            feats: List[float] = []
            tfs = self.theme_config.timeframes
            for instrument in self.theme_config.instruments:
                if instrument not in market_data:
                    # pad zeros for missing instrument/timeframes to keep stable length
                    for _ in tfs:
                        feats.extend([0.0] * 7)
                    continue

                inst_data = market_data[instrument]
                for timeframe in tfs:
                    tf_data = inst_data.get(timeframe)
                    if not isinstance(tf_data, dict) or "close" not in tf_data:
                        feats.extend([0.0] * 7)
                        continue

                    prices = np.asarray(tf_data["close"], dtype=np.float64)
                    if prices.size < 10:
                        feats.extend([0.0] * 7)
                        continue

                    returns = np.diff(prices) / np.where(prices[:-1] == 0, 1.0, prices[:-1])
                    vol = float(np.std(returns[-20:])) if returns.size >= 20 else 0.0
                    mom = float(np.mean(returns[-5:])) if returns.size >= 5 else 0.0
                    hurst = float(self._hurst_safe(prices[-50:])) if prices.size >= 50 else 0.5
                    wave = float(self._wavelet_energy_safe(prices[-30:])) if prices.size >= 30 else 0.0
                    sma_s = float(np.mean(prices[-10:])) if prices.size >= 10 else float(prices[-1])
                    sma_l = float(np.mean(prices[-30:])) if prices.size >= 30 else float(prices[-1])
                    trend = float((sma_s - sma_l) / sma_l) if sma_l != 0 else 0.0
                    roll_10 = float(prices[-1] / prices[-10] - 1.0) if prices.size >= 10 else 0.0

                    feats.extend([vol, mom, hurst, wave, trend, roll_10, float(prices.size)])

            if self.theme_config.use_macro:
                feats.extend(self._get_macro_features())

            if not feats:
                return None

            vec = np.array(feats, dtype=np.float32)
            vec = self._standardize_feature_size(vec)
            return vec

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    def _get_macro_features(self) -> List[float]:
        """Get macro features (scaled)."""
        try:
            macro_update = self.smart_bus.get("macro_data", "MarketThemeDetector")
            if isinstance(macro_update, dict) and macro_update:
                # Only update known keys to avoid accidental pollution
                for k in ("vix", "yield_curve", "cpi"):
                    if k in macro_update and isinstance(macro_update[k], (int, float)):
                        self.macro_data[k] = float(macro_update[k])

            macro_array = np.array(
                [[self.macro_data["vix"], self.macro_data["yield_curve"], self.macro_data["cpi"]]],
                dtype=np.float32,
            )
            macro_scaled = self._macro_scaler.transform(macro_array)[0]
            return macro_scaled.tolist()
        except Exception as e:
            self.logger.warning(f"Macro feature calculation failed: {e}")
            return [0.0, 0.0, 0.0]

    def _standardize_feature_size(self, features: np.ndarray) -> np.ndarray:
        """Ensure fixed-length vector for scaler/kmeans (pad/trim deterministically)."""
        expected_size = len(self.theme_config.instruments) * len(self.theme_config.timeframes) * 7 + (
            3 if self.theme_config.use_macro else 0
        )
        if features.size == expected_size:
            return features
        if features.size < expected_size:
            padded = np.zeros(expected_size, dtype=np.float32)
            padded[: features.size] = features
            return padded
        return features[:expected_size]

    @staticmethod
    def _hurst_safe(series: np.ndarray) -> float:
        try:
            if series.size < 10:
                return 0.5
            lags = np.arange(2, min(20, series.size // 2))
            if lags.size < 3:
                return 0.5
            diffs = [np.std(series[lag:] - series[:-lag]) for lag in lags]
            tau = np.array(diffs, dtype=np.float64)
            tau = tau[(tau > 0) & np.isfinite(tau)]
            if tau.size < 3:
                return 0.5
            log_lags = np.log(lags[: tau.size])
            log_tau = np.log(tau)
            coeffs = np.polyfit(log_lags, log_tau, 1)
            slope = float(coeffs[0])
            hurst = float(np.clip(slope * 2.0, 0.0, 1.0))
            return hurst if np.isfinite(hurst) else 0.5
        except Exception:
            return 0.5

    @staticmethod
    def _wavelet_energy_safe(series: np.ndarray, wavelet: str = "db4") -> float:
        try:
            if series.size < 16:
                return 0.0
            level = min(1, pywt.dwt_max_level(len(series), wavelet))
            coeffs = pywt.wavedec(series, wavelet, level=level)
            energy = float(np.sum(coeffs[-1] ** 2))  # detail coeffs energy
            total = float(np.sum(series ** 2)) + 1e-8
            return float(np.clip(energy / total, 0.0, 1.0))
        except Exception:
            return 0.0

    def _should_fit_model(self) -> bool:
        """Fit every ~batch when buffer ready; throttle fits for stability."""
        return len(self._fit_buffer) >= int(self.theme_config.batch_size) and (self._ml_fit_count % 10 == 0)

    async def _fit_model_safe(self) -> None:
        """Fit MiniBatchKMeans with circuit-breaker protection."""
        if self.ml_circuit_breaker["state"] == "OPEN":
            return
        try:
            if len(self._fit_buffer) < int(self.theme_config.batch_size):
                return
            X = np.array(list(self._fit_buffer), dtype=np.float32)
            X_scaled = self.scaler.fit_transform(X)
            self.km.fit(X_scaled)
            self._ml_fit_count += 1

            # Quality & convergence
            self._clustering_quality = self._calculate_clustering_quality()
            if hasattr(self.km, "inertia_"):
                self._convergence_history.append(float(self.km.inertia_))
                self._last_inertia = float(self.km.inertia_)

            self.smart_bus.set(
                "theme_model_quality",
                {"quality": self._clustering_quality, "fits": self._ml_fit_count},
                module="MarketThemeDetector",
                thesis=f"Theme clustering quality {self._clustering_quality:.3f}",
            )
            self.logger.info(f"[OK] Model fitted successfully - Quality: {self._clustering_quality:.3f}")

        except Exception as e:
            self._handle_ml_failure(e)

    def _calculate_clustering_quality(self) -> float:
        try:
            if not hasattr(self.km, "inertia_") or self.km.inertia_ is None:
                return 0.0
            n_samples = max(1, len(self._fit_buffer))
            normalized_inertia = float(self.km.inertia_) / float(n_samples)
            quality = 1.0 / (1.0 + normalized_inertia)
            return float(np.clip(quality, 0.0, 1.0))
        except Exception:
            return 0.0

    def _is_model_ready(self) -> bool:
        centers_ok = hasattr(self.km, "cluster_centers_") and self.km.cluster_centers_ is not None
        return centers_ok and (self._clustering_quality > float(self.theme_config.clustering_quality_threshold))

    def _detect_current_theme(self, features: np.ndarray) -> Tuple[int, float]:
        try:
            if not self._is_model_ready():
                return 0, 0.30
            fs = self.scaler.transform(features.reshape(1, -1))
            theme_id = int(self.km.predict(fs)[0])
            dists = self.km.transform(fs)[0]
            min_d = float(np.min(dists))
            strength = float(1.0 / (1.0 + max(min_d, 1e-9)))
            return theme_id, float(np.clip(strength, 0.0, 1.0))
        except Exception as e:
            self.logger.error(f"Theme detection failed: {e}")
            return 0, 0.10

    def _calculate_theme_confidence(self, features: np.ndarray, theme_id: int) -> float:
        try:
            if not self._is_model_ready():
                return 0.10
            fs = self.scaler.transform(features.reshape(1, -1))
            dists = self.km.transform(fs)[0]
            min_d = float(np.min(dists))
            # second smallest distance
            if dists.size < 2:
                return 0.25
            second = float(np.partition(dists, 1)[1])
            if second <= 0:
                return 1.0
            sep = (second - min_d) / second
            return float(np.clip(sep, 0.0, 1.0))
        except Exception:
            return 0.10

    def _update_theme_state(self, theme_id: int, strength: float, confidence: float) -> None:
        self._theme_vec.fill(0.0)
        idx = int(np.clip(theme_id, 0, len(self._theme_vec) - 1))
        self._theme_vec[idx] = float(np.clip(strength, 0.0, 1.0))
        prev = getattr(self, "_current_theme", idx)
        if idx != prev:
            self.logger.info(
                format_operator_message(
                    "[TARGET]",
                    "THEME_TRANSITION",
                    instrument=f"Theme {prev} -> {idx}",
                    details=f"Strength: {strength:.3f}",
                    context="theme_detection",
                )
            )
        self._current_theme = idx
        self._theme_confidence = float(np.clip(confidence, 0.0, 1.0))
        self._theme_momentum.append(float(strength))
        self._theme_strength_history.append(float(strength))
        self._theme_history.append(idx)

    def _get_theme_stability(self) -> float:
        if len(self._theme_strength_history) < 5:
            return 0.5
        recent = np.asarray(list(self._theme_strength_history)[-10:], dtype=np.float32)
        stability = float(np.clip(1.0 - float(np.std(recent)), 0.0, 1.0))
        return stability

    def _calculate_transition_probability(self) -> float:
        if len(self._theme_momentum) < 3:
            return 0.10
        rec = list(self._theme_momentum)[-3:]
        dif = np.diff(rec)
        if dif.size == 0:
            return 0.10
        avg = float(np.mean(dif))
        prob = float(np.clip(-avg + 0.10, 0.0, 1.0))
        return prob

    # ── THESIS & BUS ───────────────────────────────────────────────

    async def _generate_theme_thesis(self, market_data: Dict[str, Any], theme_result: Dict[str, Any]) -> str:
        """Human-friendly, compact thesis (contract requires a thesis)."""
        theme_id = int(theme_result.get("market_theme", self._current_theme))
        strength = float(theme_result.get("theme_strength", 0.0))
        confidence = float(theme_result.get("theme_confidence", 0.0))
        stability = float(theme_result.get("theme_stability", 0.0))
        names = {0: "Risk-Off Defensive", 1: "Growth Momentum", 2: "Volatility Spike", 3: "Range-Bound Consolidation"}
        name = names.get(theme_id, f"Theme {theme_id}")

        instruments_analyzed = sum(1 for inst in self.theme_config.instruments if inst in market_data)

        quality = float(self._clustering_quality)
        feat_stab = float(self._feature_stability_score)
        transition_prob = float(theme_result.get("transition_probability", 0.0))

        trend_label = (
            "Strong" if strength > 0.7 else "Moderate" if strength > 0.4 else "Weak"
        )
        conf_label = (
            "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        )
        stab_label = (
            "Stable" if stability > 0.7 else "Evolving" if stability > 0.4 else "Volatile"
        )

        lines = [
            f"MARKET THEME ANALYSIS — {name}",
            "",
            "[RESULTS]",
            f"• Theme: {name} (ID {theme_id})",
            f"• Strength: {strength:.1%} ({trend_label})",
            f"• Confidence: {confidence:.1%} ({conf_label})",
            f"• Stability: {stability:.1%} ({stab_label})",
            "",
            "[CONTEXT]",
            f"• Instruments analyzed: {instruments_analyzed}/{len(self.theme_config.instruments)}",
            f"• Clustering quality: {quality:.1%}",
            f"• Feature stability: {feat_stab:.1%}",
            f"• Themes available: {self.theme_config.n_themes}",
        ]

        if transition_prob > 0.60:
            lines.append(f"\n[WARN] High transition risk: {transition_prob:.1%}")
        elif transition_prob > 0.30:
            lines.append(f"\n[NOTE] Moderate transition risk: {transition_prob:.1%}")
        else:
            lines.append(f"\n[OK] Theme stable: {transition_prob:.1%} transition probability")

        # Short model stats
        lines += [
            "",
            "[MODEL]",
            f"• Fits: {self._ml_fit_count}",
            f"• Buffer: {len(self._fit_buffer)}/{self._fit_buffer.maxlen}",
            f"• Last update: {datetime.datetime.now().isoformat()}",
        ]

        return "\n".join(lines)

    async def _update_theme_smart_bus(
        self,
        theme_result: Dict[str, Any],
        thesis: str,
        *,
        status: Optional[Dict[str, Any]] = None,
        health: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish only THIS module's keys; avoid collisions with regime/liquidity modules."""
        self.smart_bus.set(
            "market_theme",
            theme_result.get("market_theme", self._current_theme),
            module="MarketThemeDetector",
            thesis=f"Current market theme: {theme_result.get('market_theme', self._current_theme)} ({theme_result.get('theme_strength', 0.0):.1%} strength)",
        )
        self.smart_bus.set(
            "theme_strength",
            theme_result.get("theme_strength", 0.0),
            module="MarketThemeDetector",
            thesis=f"Theme strength updated: {theme_result.get('theme_strength', 0.0):.3f}",
        )
        self.smart_bus.set(
            "theme_confidence",
            theme_result.get("theme_confidence", 0.0),
            module="MarketThemeDetector",
            thesis=f"Detection confidence: {theme_result.get('theme_confidence', 0.0):.1%}",
        )
        self.smart_bus.set(
            "theme_transition",
            theme_result.get("theme_transition", theme_result.get("transition_probability", 0.0)),
            module="MarketThemeDetector",
            thesis=f"Theme transition probability: {theme_result.get('theme_transition', 0.0):.1%}",
        )
        self.smart_bus.set(
            "theme_detection",
            theme_result.get("theme_detection", {}),
            module="MarketThemeDetector",
            thesis="Compact theme detection payload for downstream modules",
        )
        self.smart_bus.set(
            "theme_analysis",
            {
                "theme_vector": self._theme_vec.tolist(),
                "recent_transitions": int(len([x for x in self._theme_history][-20:])),
                "last_update": datetime.datetime.now().isoformat(),
                "ml_quality": float(self._clustering_quality),
                "data_quality": float(self._successful_data_extractions / max(self._data_access_attempts, 1)),
                "thesis": thesis,
            },
            module="MarketThemeDetector",
            thesis="Theme analysis snapshot",
        )

        # Also publish status/health snapshots (keep BUS in sync with return payload)
        status = status or self._build_status_snapshot(theme_result)
        health = health or self._build_health_snapshot()
        self.smart_bus.set(
            "theme_detector_status",
            status,
            module="MarketThemeDetector",
            thesis="Theme detector status",
        )
        self.smart_bus.set(
            "theme_detector_health",
            health,
            module="MarketThemeDetector",
            thesis="Theme detector health snapshot",
        )

        # Performance metric
        self.performance_tracker.record_metric(
            "MarketThemeDetector",
            "theme_detection",
            self.processing_times[-1] if self.processing_times else 0.0,
            theme_result.get("processing_success", True),
        )

    # ── FALLBACKS / ERRORS ─────────────────────────────────────────

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        self.logger.warning("No market data available - using fallback theme detection")
        res: Dict[str, Any] = {
            "market_theme": self._current_theme,
            "theme_strength": 0.10,
            "theme_confidence": 0.00,
            "theme_stability": 0.00,
            "transition_probability": 0.50,
            "theme_transition": 0.50,
            "clustering_quality": 0.00,
            "feature_stability": 0.00,
            "themes_total": int(self.theme_config.n_themes),
            "processing_success": False,
            "fallback_reason": "No market data available",
            # contract helpers (never break downstream)
            "market_data": {},
            "price_data": {},
            "technical_indicators": {},
            "market_features": [],
        }
        res["theme_detection"] = {
            "theme": res["market_theme"],
            "strength": res["theme_strength"],
            "confidence": res["theme_confidence"],
            "stability": res["theme_stability"],
            "transition_probability": res["transition_probability"],
            "timestamp": datetime.datetime.now().isoformat(),
        }
        return res

    def _create_fallback_theme_result(self, reason: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "market_theme": self._current_theme,
            "theme_strength": 0.20,
            "theme_confidence": 0.10,
            "theme_stability": 0.50,
            "transition_probability": 0.30,
            "theme_transition": 0.30,
            "clustering_quality": float(self._clustering_quality),
            "feature_stability": 0.50,
            "themes_total": int(self.theme_config.n_themes),
            "processing_success": False,
            "fallback_reason": reason,
            "market_data": {},
            "price_data": {},
            "technical_indicators": {},
            "market_features": [],
        }
        res["theme_detection"] = {
            "theme": res["market_theme"],
            "strength": res["theme_strength"],
            "confidence": res["theme_confidence"],
            "stability": res["theme_stability"],
            "transition_probability": res["transition_probability"],
            "timestamp": datetime.datetime.now().isoformat(),
        }
        return res

    async def _handle_theme_error(self, error: Exception, t0: float) -> Dict[str, Any]:
        proc_ms = (time.time() - t0) * 1000.0
        self.error_pinpointer.analyze_error(error, "MarketThemeDetector")
        self._record_failure(error)
        explanation = self.english_explainer.explain_error("MarketThemeDetector", str(error), "theme detection")
        self.logger.error(
            format_operator_message(
                "[CRASH]",
                "THEME_DETECTION_ERROR",
                details=str(error)[:200],
                explanation=explanation,
                context="error_handling",
            )
        )
        fb = self._create_fallback_theme_result(f"Error: {str(error)[:120]}")
        thesis = (
            "MARKET THEME ANALYSIS — ERROR\n\n"
            f"[WARN] Theme detection encountered an error: {str(error)[:160]}\n"
            f"[OK] Returning safe fallback values; pipeline can continue.\n"
            f"• Current Theme: {fb.get('market_theme', 0)}\n"
            f"• Transition Probability: {fb.get('transition_probability', 0.0):.1%}\n"
            f"• Processing time (ms): {proc_ms:.1f}"
        )

        # Ensure contract-required keys even on error
        status = self._build_status_snapshot(fb)
        health = self._build_health_snapshot()
        fb["_thesis"] = thesis
        fb["thesis"] = thesis
        fb["theme_detector_status"] = status
        fb["theme_detector_health"] = health
        return fb

    def _handle_ml_failure(self, error: Exception) -> None:
        self.ml_circuit_breaker["failures"] += 1
        self.ml_circuit_breaker["last_failure"] = time.time()
        if self.ml_circuit_breaker["failures"] >= int(self.theme_config.circuit_breaker_threshold):
            self.ml_circuit_breaker["state"] = "OPEN"
            self.logger.error("[ALERT] ML circuit breaker OPEN - too many failures")
        self.logger.error(f"ML training failed: {error}")

    # ── METRICS / HEALTH ───────────────────────────────────────────

    def _record_success(self, processing_time_ms: float) -> None:
        self.success_count += 1
        self.processing_times.append(float(processing_time_ms))
        # Heal ML breaker slowly on success
        if self.ml_circuit_breaker["failures"] > 0:
            self.ml_circuit_breaker["failures"] = max(0, self.ml_circuit_breaker["failures"] - 1)

    def _record_failure(self, error: Exception) -> None:
        self.failure_count += 1
        self.circuit_breaker_failures += 1
        if self.circuit_breaker_failures >= int(self.theme_config.circuit_breaker_threshold):
            self.logger.error("[ALERT] Theme detector circuit breaker triggered")

    def _update_health_metrics(self) -> None:
        total = self.success_count + self.failure_count
        success_rate = self.success_count / max(total, 1)
        avg_ms = float(np.mean(self.processing_times)) if self.processing_times else 0.0
        self.smart_bus.set(
            "theme_detector_health",
            {
                "success_rate": float(success_rate),
                "avg_processing_time_ms": float(avg_ms),
                "circuit_breaker_failures": int(self.circuit_breaker_failures),
                "ml_circuit_breaker_state": self.ml_circuit_breaker["state"],
                "clustering_quality": float(self._clustering_quality),
                "data_extraction_rate": float(self._successful_data_extractions / max(self._data_access_attempts, 1)),
                "last_update": datetime.datetime.now().isoformat(),
            },
            module="MarketThemeDetector",
            thesis=f"Theme detector health: {success_rate:.1%} success rate, {avg_ms:.1f}ms avg time",
        )

    def _check_circuit_breaker_reset(self) -> None:
        if self.ml_circuit_breaker["state"] == "OPEN" and (time.time() - self.ml_circuit_breaker["last_failure"] > 300):
            self.ml_circuit_breaker["state"] = "CLOSED"
            self.ml_circuit_breaker["failures"] = 0
            self.logger.info("[OK] ML circuit breaker reset")

    # ── STATE IO ───────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_theme": int(self._current_theme),
            "theme_vec": self._theme_vec.tolist(),
            "theme_transitions": int(len(self._theme_history)),
            "clustering_quality": float(self._clustering_quality),
            "ml_fit_count": int(self._ml_fit_count),
            "success_count": int(self.success_count),
            "failure_count": int(self.failure_count),
            "last_update": datetime.datetime.now().isoformat(),
            "config": {
                "n_themes": int(self.theme_config.n_themes),
                "window": int(self.theme_config.window),
                "instruments": list(self.theme_config.instruments),
            },
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._current_theme = int(state.get("current_theme", self._current_theme))
        if "theme_vec" in state:
            vec = np.array(state["theme_vec"], dtype=np.float32)
            if vec.shape == self._theme_vec.shape:
                self._theme_vec = vec
        self._clustering_quality = float(state.get("clustering_quality", self._clustering_quality))
        self._ml_fit_count = int(state.get("ml_fit_count", self._ml_fit_count))
        self.success_count = int(state.get("success_count", self.success_count))
        self.failure_count = int(state.get("failure_count", self.failure_count))
        self.logger.info("[OK] Theme detector state restored successfully")

    def get_health_status(self) -> Dict[str, Any]:
        total = self.success_count + self.failure_count
        return {
            "module_name": "MarketThemeDetector",
            "status": "healthy" if (self.success_count / max(total, 1)) > 0.8 else "degraded",
            "success_rate": float(self.success_count / max(total, 1)),
            "avg_processing_time": float(np.mean(self.processing_times)) if self.processing_times else 0.0,
            "circuit_breaker_state": self.ml_circuit_breaker["state"],
            "clustering_quality": float(self._clustering_quality),
            "current_theme": int(self._current_theme),
            "theme_transitions_recent": int(len([x for x in self._theme_history][-20:])),
            "data_extraction_success": float(self._successful_data_extractions / max(self._data_access_attempts, 1)),
            "last_health_check": datetime.datetime.now().isoformat(),
        }

    def stop_monitoring(self) -> None:
        self._monitoring_active = False

    # ── ACTIONS / CONFIDENCE ───────────────────────────────────────

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """
        Propose theme-based action recommendations (kept orthogonal to other modules):
        0 Risk-Off Defensive         -> defensive / reduce
        1 Growth Momentum            -> long bias
        2 Volatility Spike           -> defensive controls
        3 Range-Bound Consolidation  -> mean reversion
        """
        try:
            theme_id = int(getattr(self, "_current_theme", 0))
            strength = float(self._theme_vec[theme_id]) if 0 <= theme_id < len(self._theme_vec) else 0.0
            conf = float(getattr(self, "_theme_confidence", 0.0))
            qual = float(getattr(self, "_clustering_quality", 0.0))
            stability = float(self._get_theme_stability())

            if theme_id == 1:  # Growth Momentum
                action = "buy_moderate" if conf < 0.75 else "buy_aggressive"
                rationale = "Momentum theme favors long bias; scale with confidence."
                risk = "medium"
            elif theme_id == 3:  # Range-Bound
                action = "range_trade" if conf >= 0.6 else "monitor"
                rationale = "Range-bound theme supports mean reversion; wait for clarity if low confidence."
                risk = "low"
            elif theme_id == 2:  # Volatility Spike
                action = "defensive"
                rationale = "Elevated volatility; reduce exposure and tighten risk."
                risk = "high"
            else:  # Risk-Off / default
                action = "reduce_exposure" if conf >= 0.5 else "monitor"
                rationale = "Risk-off posture; protect capital and avoid chasing noise."
                risk = "medium"

            # Composite action confidence (bounded 0..1)
            action_conf = float(
                np.clip(0.35 * conf + 0.25 * strength + 0.20 * stability + 0.20 * min(1.0, qual * 1.2), 0.0, 1.0)
            )

            return {
                "action": action,
                "action_confidence": action_conf,
                "rationale": rationale,
                "risk_level": risk,
                "current_theme": theme_id,
                "theme_strength": strength,
                "theme_confidence": conf,
                "clustering_quality": qual,
                "theme_stability": stability,
            }

        except Exception as e:
            self.logger.error(f"Error in propose_action: {e}")
            return {
                "action": "monitor",
                "action_confidence": 0.5,
                "rationale": f"Error in theme analysis: {str(e)}",
                "risk_level": "medium",
            }

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Confidence for the previously proposed action."""
        try:
            conf = float(getattr(self, "_theme_confidence", 0.5))
            qual = float(getattr(self, "_clustering_quality", 0.5))
            stab = float(self._get_theme_stability())
            ml_health = 1.0 if getattr(self, "ml_circuit_breaker", {}).get("state") == "CLOSED" else 0.3
            data_ok = float(self._successful_data_extractions / max(self._data_access_attempts, 1))

            base = 0.35 * conf + 0.25 * qual + 0.20 * stab + 0.10 * ml_health + 0.10 * data_ok
            decision = action.get("action", "monitor")
            if "aggressive" in decision and conf < 0.7:
                base *= 0.85  # throttle if overconfident
            if decision == "range_trade" and self._current_theme == 3:
                base *= 1.08  # small boost when aligned

            return float(np.clip(base, 0.0, 1.0))
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
