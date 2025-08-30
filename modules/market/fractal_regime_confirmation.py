# ─────────────────────────────────────────────────────────────
# File: modules/market/fractal_regime_confirmation.py
# Fractal Regime Confirmation (Contract-Clean, Orchestration-Safe)
# ─────────────────────────────────────────────────────────────

import datetime
import time
import asyncio
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, Optional, List

from modules.contracts import module_args
import numpy as np
import pandas as pd
import pywt

# Core infrastructure
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusVotingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass
class FractalConfig:
    # core
    window: int = 100
    # metric coefficients
    coeff_h: float = 0.40
    coeff_vr: float = 0.30
    coeff_we: float = 0.30
    # hysteresis thresholds
    noise_to_volatile: float = 0.30
    volatile_to_noise: float = 0.20
    volatile_to_trending: float = 0.60
    trending_to_volatile: float = 0.50
    # logging / monitoring
    log_path: str = "logs/market/fractal_regime.log"
    enable_gpu_metrics: bool = False  # reserved


# ─────────────────────────────────────────────────────────────
# Module declaration
# ─────────────────────────────────────────────────────────────
@module(**module_args(
    "FractalRegimeConfirmation",
    description="Fractal analysis (Hurst / Variance Ratio / Wavelet Energy) with hysteresis and stability control.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=120,
))
class FractalRegimeConfirmation(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusVotingMixin):
    """
    Clean producer of regime-related keys with single-writer discipline.
    - Reads market data from kwargs or InfoBus (best-effort), with synthetic fallback.
    - Computes robust fractal metrics and a smooth strength score.
    - Applies hysteresis to avoid flip-flopping regimes.
    - Publishes only the declared keys (no duplication with other modules).
    """

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        genome: Optional[Dict[str, Any]] = None,  # backward-compat knobs
        **kwargs: Any,
    ):
        # normalize config/genome into FractalConfig
        if isinstance(config, dict):
            base = FractalConfig(**{k: config[k] for k in FractalConfig.__dataclass_fields__ if k in config})
        else:
            base = FractalConfig()

        # legacy genome overrides
        if genome:
            base.window = int(genome.get("window", base.window))
            base.coeff_h = float(genome.get("coeff_h", base.coeff_h))
            base.coeff_vr = float(genome.get("coeff_vr", base.coeff_vr))
            base.coeff_we = float(genome.get("coeff_we", base.coeff_we))

        self._cfg = base

        # BaseModule will invoke _initialize()
        super().__init__(config=asdict(self._cfg), **kwargs)

    def _initialize(self) -> None:
        """Implement abstract hook; build everything here (no super()._initialize())."""
        # core handles
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="FractalRegimeConfirmation",
            log_path=self._cfg.log_path,
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )

        # helpers
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("FractalRegimeConfirmation", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # state
        self.window: int = int(self._cfg.window)
        self.coeff_h: float = float(self._cfg.coeff_h)
        self.coeff_vr: float = float(self._cfg.coeff_vr)
        self.coeff_we: float = float(self._cfg.coeff_we)

        self._noise_to_volatile = float(self._cfg.noise_to_volatile)
        self._volatile_to_noise = float(self._cfg.volatile_to_noise)
        self._volatile_to_trending = float(self._cfg.volatile_to_trending)
        self._trending_to_volatile = float(self._cfg.trending_to_volatile)

        # working buffers / telemetry
        self._buf: deque = deque(maxlen=max(3, int(self.window * 0.75)))
        self._regime_history: deque = deque(maxlen=50)
        self._fractal_metrics_history: deque = deque(maxlen=100)

        self._last_symbols: List[str] = []
        self._last_timestamps: List[str] = []
        self._last_known_prices: Dict[str, float] = {}

        self._data_access_attempts: int = 0
        self._successful_data_extractions: int = 0

        # outputs (authoritative state)
        self.label: str = "noise"
        self.regime_strength: float = 0.0  # internal, may exceed 1.0; clamped at output
        self._trend_direction: float = 0.0  # [-1..1]
        self._regime_stability_score: float = 100.0
        self._theme_integration_score: float = 1.0

        # circuit breaker
        self.fractal_circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",  # CLOSED | HALF_OPEN | OPEN
            "threshold": 3,
        }

        # aggregate metrics
        self._regime_metrics: Dict[str, Any] = {
            "transitions": 0,
            "avg_strength": 0.0,
            "stability_trend": deque(maxlen=20),
            "performance_by_regime": {
                "noise": {"count": 0, "avg_strength": 0.0},
                "volatile": {"count": 0, "avg_strength": 0.0},
                "trending": {"count": 0, "avg_strength": 0.0},
            },
        }

        # background monitors (optional)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._health_monitor_loop())
        except RuntimeError:
            pass

        self.logger.info(
            format_operator_message(
                "🧮", "FRACTAL_READY",
                details=f"window={self.window}, coeffs(H/VR/WE)={self.coeff_h}/{self.coeff_vr}/{self.coeff_we}",
                result="ready",
                context="fractal_startup",
            )
        )

        # publish minimal initial state (declared keys only)
        self._publish_bus(
            regime=self.label,
            strength=self.regime_strength,
            trend=self._trend_direction,
            fractal_metrics={},  # declared; empty at init
            thesis="Initial regime metadata published.",
        )

    # ─────────────────────────────────────────────────────────
    # Output contract
    # ─────────────────────────────────────────────────────────
    def _format_declared_outputs(
        self,
        *,
        regime: Optional[str] = None,
        strength: Optional[float] = None,
        trend_direction: Optional[float] = None,
        fractal_metrics: Optional[Dict[str, float]] = None,
        regime_data: Optional[Dict[str, Any]] = None,
        symbols: Optional[List[str]] = None,
        timestamps: Optional[List[str]] = None,
        thesis: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ensure all declared outputs exist with coherent types.
        - market_regime: str
        - regime_strength: float in [0..1] (we clamp on output)
        - trend_direction: float in [-1..1]
        - fractal_metrics: dict
        - regime_data: dict (normalized)
        - symbols, timestamps: list[str]
        """
        out: Dict[str, Any] = {}

        # scalars
        out["market_regime"] = str(regime if regime is not None else self.label)
        out["regime_strength"] = float(np.clip(strength if strength is not None else self.regime_strength, 0.0, 1.0))
        out["trend_direction"] = float(np.clip(trend_direction if trend_direction is not None else self._trend_direction, -1.0, 1.0))

        # metrics dict (type-check first, then finiteness)
        fm = fractal_metrics if isinstance(fractal_metrics, dict) else {}
        out["fractal_metrics"] = {
            k: float(v)
            for k, v in fm.items()
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
        }

        # regime_data normalization
        _RID = {"noise": 0, "range": 1, "trend": 2, "trending": 2, "volatile": 3}
        rd = regime_data if isinstance(regime_data, dict) else {}
        if "id" not in rd or "market_regime" not in rd:
            mr = out["market_regime"]
            rd = {
                "id": _RID.get(mr, _RID.get(mr.lower(), 0)),
                "market_regime": mr,
                "regime_strength": out["regime_strength"],
                "trend_direction": out["trend_direction"],
            }
        out["regime_data"] = rd

        # list fields
        out["symbols"] = list(symbols) if isinstance(symbols, list) else list(getattr(self, "_last_symbols", []))
        out["timestamps"] = list(timestamps) if isinstance(timestamps, list) else list(getattr(self, "_last_timestamps", []))

        # thesis (explainable contract)
        safe_thesis = thesis or self._generate_thesis(out["market_regime"], out["regime_strength"])
        out["_thesis"] = safe_thesis
        out["thesis"] = safe_thesis  # convenience

        # extras (non-declared metadata allowed here)
        if extra and isinstance(extra, dict):
            try:
                out.update(extra)
            except Exception:
                pass

        # fail-fast for declared essentials
        for k in ("market_regime", "regime_strength", "trend_direction"):
            if k not in out or out[k] is None:
                raise ValueError(f"Critical output '{k}' missing in FractalRegimeConfirmation")

        return out

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────
    async def process(self, **inputs: Any) -> Dict[str, Any]:
        t0 = time.time()

        # circuit breaker short-circuit
        if not self._cb_allow():
            thesis = "Fractal circuit breaker is open; returning cached state."
            payload = self._build_regime_data(self.label, self.regime_strength)
            return self._format_declared_outputs(
                regime=self.label,
                strength=self.regime_strength,
                trend_direction=self._trend_direction,
                fractal_metrics=payload.get("latest_metrics", {}).get("metrics", {}),
                regime_data=payload,
                symbols=self._last_symbols,
                timestamps=self._last_timestamps,
                thesis=thesis,
                extra={"processing_time_ms": (time.time() - t0) * 1000.0, "success": False, "reason": "circuit_open"},
            )

        try:
            market = self._extract_market_data_comprehensive(inputs)
            if not market:
                thesis = "Fractal analysis fallback: no market data available; using cached state."
                payload = self._build_regime_data(self.label, self.regime_strength)
                return self._format_declared_outputs(
                    regime=self.label,
                    strength=self.regime_strength,
                    trend_direction=self._trend_direction,
                    fractal_metrics=payload.get("latest_metrics", {}).get("metrics", {}),
                    regime_data=payload,
                    symbols=self._last_symbols,
                    timestamps=self._last_timestamps,
                    thesis=thesis,
                    extra={"processing_time_ms": (time.time() - t0) * 1000.0, "success": True, "fallback_reason": "no_market_data"},
                )

            regime, strength = self._process_regime_detection(market)
            self._update_regime_metrics(regime, strength)

            # publish declared keys ONLY
            thesis = self._generate_thesis(regime, strength)
            latest_metrics = {}
            if len(self._fractal_metrics_history) > 0:
                latest_metrics = dict(self._fractal_metrics_history[-1].get("metrics", {}))
            self._publish_bus(regime=regime, strength=strength, trend=self._trend_direction, fractal_metrics=latest_metrics, thesis=thesis)

            # perf + cb bookkeeping
            elapsed_ms = (time.time() - t0) * 1000.0
            self._record_success(elapsed_ms)

            payload = self._build_regime_data(regime, strength)
            return self._format_declared_outputs(
                regime=regime,
                strength=strength,
                trend_direction=self._trend_direction,
                fractal_metrics=payload.get("latest_metrics", {}).get("metrics", {}),
                regime_data=payload,
                symbols=self._last_symbols,
                timestamps=self._last_timestamps,
                thesis=thesis,
                extra={"processing_time_ms": elapsed_ms, "success": True},
            )
        except Exception as e:
            return await self._handle_fractal_error(e, t0)

    # backward compatibility wrapper
    def step(self, data_dict=None, current_step=None, theme_detector=None, **kwargs) -> Tuple[str, float]:
        if data_dict is not None and current_step is not None:
            kwargs.update({"data_dict": data_dict, "current_step": current_step, "theme_detector": theme_detector})
        # run a light sync step (no bus publishing here; orchestrator calls process())
        self._data_access_attempts += 1
        market = self._extract_market_data_comprehensive(kwargs)
        if market:
            self._successful_data_extractions += 1
            regime, strength = self._process_regime_detection(market)
            self._update_regime_metrics(regime, strength)
        else:
            syn = self._create_synthetic_market_data()
            if syn:
                regime, strength = self._process_regime_detection(syn)
                self._update_regime_metrics(regime, strength)
        return self.label, float(np.clip(self.regime_strength, 0.0, 1.0))

    # observation vector for module system
    def _get_observation_impl(self) -> np.ndarray:
        return self.get_observation_components()

    def get_observation_components(self) -> np.ndarray:
        try:
            one_hot = {"noise": [1, 0, 0], "volatile": [0, 1, 0], "trending": [0, 0, 1]}.get(self.label, [0, 0, 0])
            strength = float(np.clip(self.regime_strength, 0.0, 1.0))
            trend = float(np.clip(self._trend_direction, -1.0, 1.0))
            stability = float(self._regime_stability_score) / 100.0
            obs = np.array(one_hot + [strength, trend, stability], dtype=np.float32)
            if not np.all(np.isfinite(obs)):
                return np.zeros(6, dtype=np.float32)
            return obs
        except Exception:
            return np.zeros(6, dtype=np.float32)

    # ─────────────────────────────────────────────────────────
    # InfoBus publishing (declared keys only)
    # ─────────────────────────────────────────────────────────
    def _publish_bus(self, *, regime: str, strength: float, trend: float, fractal_metrics: Dict[str, float], thesis: str) -> None:
        try:
            # market_regime
            self.smart_bus.set(
                "market_regime",
                regime,
                module="FractalRegimeConfirmation",
                thesis=f"Current market regime: {regime} (strength {np.clip(strength,0,1):.3f})",
            )
            # regime_strength
            self.smart_bus.set(
                "regime_strength",
                float(np.clip(strength, 0.0, 1.0)),
                module="FractalRegimeConfirmation",
                thesis="Regime strength updated.",
            )
            # trend_direction
            self.smart_bus.set(
                "trend_direction",
                float(np.clip(trend, -1.0, 1.0)),
                module="FractalRegimeConfirmation",
                thesis="Trend direction updated.",
            )
            # fractal_metrics (declared)
            clean_metrics = {
                k: float(v)
                for k, v in (fractal_metrics or {}).items()
                if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
            }
            self.smart_bus.set(
                "fractal_metrics",
                clean_metrics,
                module="FractalRegimeConfirmation",
                thesis="Latest fractal metrics snapshot.",
            )
            # regime_data
            _RID = {"noise": 0, "range": 1, "trend": 2, "trending": 2, "volatile": 3}
            self.smart_bus.set(
                "regime_data",
                {
                    "id": _RID.get(regime, _RID.get(regime.lower(), 0)),
                    "market_regime": regime,
                    "regime_strength": float(np.clip(strength, 0.0, 1.0)),
                    "trend_direction": float(np.clip(trend, -1.0, 1.0)),
                    "last_update": datetime.datetime.now().isoformat(),
                },
                module="FractalRegimeConfirmation",
                thesis=thesis,
            )
            # timestamps (declared)
            self.smart_bus.set(
                "timestamps",
                list(self._last_timestamps),
                module="FractalRegimeConfirmation",
                thesis="Timestamps window",
            )
            # IMPORTANT: Do NOT write 'symbols' here (not declared) to maintain single-writer discipline.
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    # ─────────────────────────────────────────────────────────
    # Data extraction (multi-path, robust)
    # ─────────────────────────────────────────────────────────
    def _extract_market_data_comprehensive(self, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            self._data_access_attempts += 1
        except Exception:
            self._data_access_attempts = 1

        # 1) kwargs legacy path: data_dict + current_step
        if isinstance(args, dict) and "data_dict" in args and "current_step" in args:
            return {
                "data_dict": args["data_dict"],
                "current_step": int(args["current_step"]),
                "theme_detector": args.get("theme_detector"),
                "source": "kwargs",
            }

        # 2) InfoBus structured (module_data.market_data)
        try:
            bus_md = self.smart_bus.get("module_data", "FractalRegimeConfirmation", default={})
            if isinstance(bus_md, dict) and "market_data" in bus_md:
                md = bus_md["market_data"]
                if isinstance(md, dict) and "data_dict" in md and "current_step" in md:
                    return {**md, "source": "infobus_structured"}
        except Exception:
            pass

        # 3) InfoBus 'prices' quick path
        try:
            prices = self.smart_bus.get("prices", "FractalRegimeConfirmation", default=None)
            if isinstance(prices, dict) and prices:
                # keep stale cache for fallback
                try:
                    self._last_known_prices.update(prices)
                except Exception:
                    self._last_known_prices = dict(prices)

                step_idx = int(self.smart_bus.get("step_idx", "FractalRegimeConfirmation", default=0) or 0)
                regime_ctx = self.smart_bus.get("market_regime", "FractalRegimeConfirmation", default="unknown")
                vol_level = self.smart_bus.get("volatility_level", "FractalRegimeConfirmation", default="medium")
                return {
                    "prices": prices,
                    "current_step": step_idx,
                    "regime_context": regime_ctx,
                    "volatility_level": vol_level,
                    "source": "infobus_prices",
                }
        except Exception:
            pass

        # 4) last known prices (stale cache)
        try:
            if isinstance(self._last_known_prices, dict) and self._last_known_prices:
                return {
                    "prices": dict(self._last_known_prices),
                    "current_step": 0,
                    "regime_context": "stale",
                    "source": "last_known_stale",
                }
        except Exception:
            pass

        return None

    # ─────────────────────────────────────────────────────────
    # Regime detection pipeline
    # ─────────────────────────────────────────────────────────
    def _process_regime_detection(self, market: Dict[str, Any]) -> Tuple[str, float]:
        try:
            src = market.get("source", "unknown")
            if src in ("kwargs", "infobus_structured") or "data_dict" in market:
                return self._process_traditional_format(market)
            if src in ("infobus_prices", "last_known_stale", "synthetic") or "prices" in market:
                return self._process_prices_format(market)
            # unknown → best effort
            return self._process_fallback_format(market)
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return self.label, self.regime_strength

    def _process_traditional_format(self, market: Dict[str, Any]) -> Tuple[str, float]:
        data_dict = market["data_dict"]
        current_step = int(market["current_step"])
        theme_detector = market.get("theme_detector")

        # select instrument
        instruments = list(data_dict.keys()) if isinstance(data_dict, dict) else []
        if not instruments:
            return self.label, self.regime_strength

        preferred = ["EUR/USD", "XAU/USD",]
        selected = next((p for p in preferred if p in instruments), instruments[0])

        # require D1 dataframe
        df = data_dict.get(selected, {}).get("D1")
        if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
            return self.label, self.regime_strength

        # indices
        if len(df) == 0 or current_step >= len(df):
            return self.label, self.regime_strength

        start_idx = max(0, current_step - self.window)
        end_idx = min(current_step + 1, len(df))
        if end_idx <= start_idx:
            return self.label, self.regime_strength

        ts = df["close"].values[start_idx:end_idx].astype(np.float32)
        self._capture_symbols_timestamps_from_df(df, start_idx, end_idx, selected)
        if ts.size < 2:
            return self.label, self.regime_strength

        # compute metrics
        self._trend_direction = self._calculate_trend_direction(ts)
        metrics = self._compute_fractal_metrics_robust(ts)
        theme_conf = self._integrate_theme_detector(theme_detector, data_dict, current_step)

        return self._process_regime_signals(metrics, theme_conf)

    def _process_prices_format(self, market: Dict[str, Any]) -> Tuple[str, float]:
        prices = market.get("prices", {})
        if not isinstance(prices, dict) or not prices:
            return self.label, self.regime_strength

        self._capture_symbols_from_prices(prices)
        src = market.get("source", "unknown")
        vals = list(map(float, prices.values()))

        if len(vals) == 1:
            base = vals[0]
        else:
            base = float(np.mean(vals))

        if src == "synthetic":
            n = min(self.window, 50)
            rets = np.random.normal(0.0, 0.01, n - 1)
            ts = np.empty(n, dtype=np.float32)
            ts[0] = base
            for i in range(1, n):
                ts[i] = ts[i - 1] * (1.0 + rets[i - 1])
        else:
            n = min(self.window, 20)
            ts = np.full(n, base, dtype=np.float32)
            ts += np.random.normal(0.0, max(1e-8, base * 0.001), n).astype(np.float32)

        self._trend_direction = self._calculate_trend_direction(ts)
        metrics = self._compute_fractal_metrics_robust(ts)
        return self._process_regime_signals(metrics, theme_conf=1.0)

    def _process_fallback_format(self, market: Dict[str, Any]) -> Tuple[str, float]:
        if "prices" in market:
            return self._process_prices_format(market)
        if "data_dict" in market:
            return self._process_traditional_format(market)
        return self.label, self.regime_strength

    # ─────────────────────────────────────────────────────────
    # Metrics & signals
    # ─────────────────────────────────────────────────────────
    def _calculate_trend_direction(self, ts: np.ndarray) -> float:
        if ts.size < 2:
            return 0.0
        try:
            if ts.size >= 10:
                recent = ts[-5:]
                older = ts[:5]
                oa = float(np.mean(older))
                ra = float(np.mean(recent))
                if abs(oa) > 1e-12:
                    return float(np.clip((ra - oa) / abs(oa), -1.0, 1.0))
            if abs(float(ts[0])) > 1e-12:
                return float(np.clip((float(ts[-1]) - float(ts[0])) / abs(float(ts[0])), -1.0, 1.0))
            return 0.0
        except Exception:
            return 0.0

    def _compute_fractal_metrics_robust(self, ts: np.ndarray) -> Dict[str, float]:
        metrics = {"H": 0.5, "VR": 1.0, "WE": 0.0}
        if ts.size < 2:
            return metrics
        try:
            if ts.size >= 10:
                metrics["H"] = self._hurst_enhanced(ts)
            metrics["VR"] = self._var_ratio_enhanced(ts)
            if ts.size >= 16:
                metrics["WE"] = self._wavelet_energy_enhanced(ts)

            # sanitize
            for k in list(metrics.keys()):
                v = float(metrics[k])
                if not np.isfinite(v):
                    metrics[k] = {"H": 0.5, "VR": 1.0, "WE": 0.0}[k]

            self._fractal_metrics_history.append(
                {"timestamp": np.datetime64("now").astype(str), "metrics": dict(metrics), "series_length": int(ts.size)}
            )
        except Exception as e:
            self.logger.error(f"Fractal metrics computation failed: {e}")
        return metrics

    @staticmethod
    def _hurst_enhanced(series: np.ndarray) -> float:
        s = series[:500].astype(float)
        if s.size < 10:
            return 0.5
        if np.std(s) < 1e-10:
            return 0.5
        try:
            max_lag = min(50, s.size // 3)
            lags = np.unique(np.logspace(0.3, np.log10(max_lag), 15).astype(int))
            lags = lags[lags >= 2]
            if lags.size < 3:
                return 0.5
            tau = []
            for lag in lags:
                if lag < s.size:
                    diff = s[lag:] - s[:-lag]
                    if diff.size > 0:
                        tau.append(np.std(diff))
            tau = np.asarray(tau, dtype=float)
            valid = (tau > 0) & np.isfinite(tau)
            if np.sum(valid) < 3:
                return 0.5
            tau = tau[valid]
            lags = lags[: tau.size][valid]
            log_lags = np.log(lags)
            log_tau = np.log(tau)
            if not (np.all(np.isfinite(log_lags)) and np.all(np.isfinite(log_tau))):
                return 0.5
            coef = np.polyfit(log_lags, log_tau, 1)
            slope = float(coef[0])
            corr = np.corrcoef(log_lags, log_tau)[0, 1]
            r2 = float(corr * corr) if np.isfinite(corr) else 0.0
            if r2 < 0.1:
                return 0.5
            hurst = float(np.clip(slope * 2.0, 0.0, 1.0))
            return hurst
        except Exception:
            return 0.5

    @staticmethod
    def _var_ratio_enhanced(ts: np.ndarray) -> float:
        t = ts[-200:].astype(float)
        if t.size < 4:
            return 1.0
        try:
            k = min(2, t.size // 2)
            if k < 2:
                return 1.0
            k_ret = t[k:] - t[:-k]
            one_ret = t[1:] - t[:-1]
            if one_ret.size < k or k_ret.size == 0:
                return 1.0
            var_k = np.var(k_ret) / k
            var_1 = np.var(one_ret)
            if var_1 <= 1e-12:
                return 1.0
            ratio = float(np.clip(var_k / var_1, 0.1, 10.0))
            return ratio if np.isfinite(ratio) else 1.0
        except Exception:
            return 1.0

    @staticmethod
    def _wavelet_energy_enhanced(series: np.ndarray, wavelet: str = "db4") -> float:
        s = series[:256].astype(float)
        if s.size < 16 or np.std(s) < 1e-10:
            return 0.0
        try:
            max_lvl = pywt.dwt_max_level(len(s), wavelet)
            lvl = min(2, max_lvl)
            if lvl < 1:
                return 0.0
            coeffs = pywt.wavedec(s, wavelet, level=lvl)
            detail_energy = 0.0
            for i in range(1, len(coeffs)):
                detail_energy += float(np.sum(np.square(coeffs[i])))
            total = float(np.sum(np.square(s)))
            if total <= 1e-12:
                return 0.0
            er = float(np.clip(detail_energy / total, 0.0, 1.0))
            return er if np.isfinite(er) else 0.0
        except Exception:
            return 0.0

    def _integrate_theme_detector(self, theme_detector: Any, data_dict: Dict[str, Any], step: int) -> float:
        conf = 1.0
        try:
            if theme_detector is not None and data_dict is not None:
                if hasattr(theme_detector, "fit_if_needed"):
                    theme_detector.fit_if_needed(data_dict, step)
                if hasattr(theme_detector, "detect"):
                    _, conf = theme_detector.detect(data_dict, step)
                    if not np.isfinite(conf):
                        conf = 1.0
                conf = float(np.clip(conf, 0.0, 1.0))
            self._theme_integration_score = conf
        except Exception as e:
            self.logger.warning(f"Theme integration failed: {e}")
            conf = 1.0
        return conf

    def _process_regime_signals(self, metrics: Dict[str, float], theme_conf: float) -> Tuple[str, float]:
        H = float(np.clip(metrics.get("H", 0.5), 0.0, 1.0))
        VR = float(np.clip(metrics.get("VR", 1.0), 0.1, 10.0))
        WE = float(np.clip(metrics.get("WE", 0.0), 0.0, 1.0))

        score = self.coeff_h * H + self.coeff_vr * VR + self.coeff_we * WE
        self._buf.append(score)

        if len(self._buf) >= 3:
            smoothed = float(np.median(list(self._buf)[-3:]))
        else:
            smoothed = float(np.mean(self._buf)) if len(self._buf) > 0 else 0.0

        strength = float(np.clip(smoothed * float(np.clip(theme_conf, 0.5, 1.0)), 0.0, 2.0))

        old = self.label
        new = self._determine_regime_with_hysteresis(old, strength)

        if new != old:
            self._log_regime_change(old, new, strength)

        self._regime_history.append((new, strength, self._trend_direction))
        self._update_regime_stability()

        return new, strength

    def _determine_regime_with_hysteresis(self, old_label: str, strength: float) -> str:
        if old_label == "noise":
            new_label = "volatile" if strength >= self._noise_to_volatile else "noise"
        elif old_label == "volatile":
            if strength >= self._volatile_to_trending:
                new_label = "trending"
            elif strength < self._volatile_to_noise:
                new_label = "noise"
            else:
                new_label = "volatile"
        else:  # trending
            new_label = "volatile" if strength < self._trending_to_volatile else "trending"

        # stability guard
        if len(self._regime_history) >= 5 and new_label != old_label:
            recent = [r[0] for r in list(self._regime_history)[-5:]]
            if len(set(recent)) > 2 and abs(strength - 0.5) < 0.3:
                new_label = old_label
        return new_label

    def _update_regime_stability(self) -> None:
        if len(self._regime_history) >= 10:
            recent = [r[0] for r in list(self._regime_history)[-10:]]
            uniq = len(set(recent))
            stability = max(0.0, 100.0 - (uniq - 1) * 15.0)
            self._regime_stability_score = stability
            self.performance_tracker.record_metric("FractalRegimeConfirmation", "regime_stability", stability, True)

    def _log_regime_change(self, old_label: str, new_label: str, strength: float) -> None:
        dur = 1
        for i in range(len(self._regime_history) - 1, -1, -1):
            if self._regime_history[i][0] == old_label:
                dur += 1
            else:
                break
        success_rate = self._successful_data_extractions / max(self._data_access_attempts, 1)
        self.logger.info(
            format_operator_message(
                "📈", "REGIME_TRANSITION",
                details=f"{old_label} → {new_label} | strength={strength:.3f} | trend={self._trend_direction:+.3f} | "
                        f"duration={dur} | stability={self._regime_stability_score:.1f}% | data_ok={success_rate:.1%}",
                result="transition",
                context="fractal_regime",
            )
        )

    # ─────────────────────────────────────────────────────────
    # State / metrics bookkeeping
    # ─────────────────────────────────────────────────────────
    def _update_regime_metrics(self, regime: str, strength: float) -> None:
        try:
            if hasattr(self, "label") and self.label != regime:
                self._regime_metrics["transitions"] += 1
            self.label = regime
            self.regime_strength = float(strength)

            if regime in self._regime_metrics["performance_by_regime"]:
                rd = self._regime_metrics["performance_by_regime"][regime]
                rd["count"] += 1
                c = rd["count"]
                rd["avg_strength"] = (rd["avg_strength"] * (c - 1) + float(strength)) / c

            if len(self._regime_history) > 0:
                strengths = [r[1] for r in self._regime_history] + [float(strength)]
                self._regime_metrics["avg_strength"] = float(np.mean(strengths[-20:]))
            else:
                self._regime_metrics["avg_strength"] = float(strength)

            self._regime_metrics["stability_trend"].append(self._regime_stability_score)

            self.performance_tracker.record_metric("FractalRegimeConfirmation", "regime_strength", float(strength), True)
            self.performance_tracker.record_metric("FractalRegimeConfirmation", "regime_transitions", float(self._regime_metrics["transitions"]), True)
            self.performance_tracker.record_metric("FractalRegimeConfirmation", "avg_regime_strength", float(self._regime_metrics["avg_strength"]), True)
        except Exception as e:
            self.logger.error(f"Regime metrics update failed: {e}")

    def _capture_symbols_timestamps_from_df(self, df: pd.DataFrame, start_idx: int, end_idx: int, inst: str) -> None:
        try:
            self._last_symbols = [str(inst)]
        except Exception:
            self._last_symbols = [str(inst)]
        try:
            idx_slice = df.index[start_idx:end_idx]
            self._last_timestamps = [str(x) for x in idx_slice]
        except Exception:
            self._last_timestamps = [str(i) for i in range(start_idx, end_idx)]

    def _capture_symbols_from_prices(self, prices: Dict[str, Any]) -> None:
        try:
            self._last_symbols = list(map(str, prices.keys()))
        except Exception:
            self._last_symbols = []
        if not hasattr(self, "_last_timestamps"):
            self._last_timestamps = []

    def _create_synthetic_market_data(self) -> Optional[Dict[str, Any]]:
        try:
            instruments = ["EUR/USD", "XAU/USD"]
            bases = {"EUR/USD": 1.10, "XAU/USD": 1950.0}
            synthetic = {}
            for ins in instruments:
                b = float(bases.get(ins, 1.0))
                vol = max(1e-6, b * 0.001)
                synthetic[ins] = max(b * 0.8, b + float(np.random.normal(0.0, vol)))
            return {"prices": synthetic, "current_step": 0, "regime_context": "synthetic", "volatility_level": "medium", "source": "synthetic"}
        except Exception as e:
            self.logger.error(f"Failed to create synthetic data: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # Errors / health
    # ─────────────────────────────────────────────────────────
    async def _handle_fractal_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        elapsed_ms = (time.time() - start_time) * 1000.0
        self.fractal_circuit_breaker["failures"] += 1
        self.fractal_circuit_breaker["last_failure"] = time.time()
        if self.fractal_circuit_breaker["failures"] >= self.fractal_circuit_breaker["threshold"]:
            self.fractal_circuit_breaker["state"] = "OPEN"

        try:
            _ = self.error_pinpointer.analyze_error(error, "FractalRegimeConfirmation")
            explanation = self.english_explainer.explain_error("FractalRegimeConfirmation", str(error), "fractal analysis")
            self.logger.error(f"Fractal analysis error: {error} | {explanation}")
        except Exception:
            pass

        thesis = f"Fractal analysis error; returning safe defaults. Reason: {str(error)}"
        payload = self._build_regime_data(self.label, self.regime_strength)
        return self._format_declared_outputs(
            regime=self.label,
            strength=self.regime_strength,
            trend_direction=self._trend_direction,
            fractal_metrics=payload.get("latest_metrics", {}).get("metrics", {}),
            regime_data=payload,
            symbols=self._last_symbols,
            timestamps=self._last_timestamps,
            thesis=thesis,
            extra={"error": str(error), "processing_time_ms": elapsed_ms, "circuit_breaker_state": self.fractal_circuit_breaker["state"], "success": False},
        )

    def _record_success(self, processing_time_ms: float) -> None:
        try:
            self.performance_tracker.record_metric("FractalRegimeConfirmation", "fractal_cycle", processing_time_ms, True)
            if self.fractal_circuit_breaker["state"] in ("OPEN", "HALF_OPEN"):
                self.fractal_circuit_breaker["failures"] = 0
                self.fractal_circuit_breaker["state"] = "CLOSED"
        except Exception:
            pass

    def _cb_allow(self) -> bool:
        state = self.fractal_circuit_breaker["state"]
        if state == "OPEN":
            if time.time() - self.fractal_circuit_breaker["last_failure"] > 120.0:
                self.fractal_circuit_breaker["state"] = "HALF_OPEN"
                return True
            return False
        return True

    async def _health_monitor_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                # stability-based micro adjust (placeholder for richer health)
                if len(self._regime_history) > 0:
                    pass
            except Exception as e:
                self.logger.error(f"Fractal health monitor error: {e}")

    # ─────────────────────────────────────────────────────────
    # Reports / explainability
    # ─────────────────────────────────────────────────────────
    def _build_regime_data(self, regime: str, strength: float) -> Dict[str, Any]:
        last_metrics = {}
        if len(self._fractal_metrics_history) > 0:
            try:
                last_metrics = dict(self._fractal_metrics_history[-1])
            except Exception:
                last_metrics = {}

        return {
            "regime": regime,
            "strength": float(strength),
            "trend_direction": float(self._trend_direction),
            "stability_score": float(self._regime_stability_score),
            "theme_confidence": float(self._theme_integration_score),
            "latest_metrics": last_metrics,
            "history_len": len(self._fractal_metrics_history),
        }

    def _generate_thesis(self, regime: str, strength: float) -> str:
        try:
            metrics = {}
            if len(self._fractal_metrics_history) > 0:
                metrics = dict(self._fractal_metrics_history[-1].get("metrics", {}))
            H = float(metrics.get("H", 0.5))
            VR = float(metrics.get("VR", 1.0))
            WE = float(metrics.get("WE", 0.0))
            trend = float(self._trend_direction)
            stability = float(self._regime_stability_score)
            theme = float(self._theme_integration_score)
            return (
                f"FRACTAL REGIME: {regime.upper()} | strength {np.clip(strength,0,1):.3f} | "
                f"trend {trend:+.3f} | stability {stability:.1f}/100 | "
                f"H={H:.3f}, VR={VR:.3f}, WE={WE:.3f} | theme_conf {theme:.3f}"
            )
        except Exception:
            return f"FRACTAL REGIME: {regime.upper()} | strength {np.clip(strength,0,1):.3f}"

    def get_regime_analysis_report(self) -> str:
        regime_distribution: Dict[str, float] = {}
        if len(self._regime_history) >= 10:
            recent = [r[0] for r in list(self._regime_history)[-10:]]
            for r in ("noise", "volatile", "trending"):
                regime_distribution[r] = recent.count(r) / len(recent)
        success_rate = self._successful_data_extractions / max(self._data_access_attempts, 1)
        return f"""
[CHART] FRACTAL REGIME ANALYSIS
══════════════════════════════════════════════════════════════
[TARGET] Current Regime: {self.label.upper()} (Strength: {np.clip(self.regime_strength,0,1):.3f})
[STATS] Trend Direction: {self._trend_direction:.3f}
[BALANCE] Stability Score: {self._regime_stability_score:.1f}/100

[METRICS COEFFS]
• Hurst: {self.coeff_h:.2f}
• Variance Ratio: {self.coeff_vr:.2f}
• Wavelet Energy: {self.coeff_we:.2f}
• Buffer Size: {len(self._buf)}/{self._buf.maxlen}

[RECENT REGIME DISTRIBUTION (10)]
• Noise: {regime_distribution.get('noise', 0.0):.1%}
• Volatile: {regime_distribution.get('volatile', 0.0):.1%}
• Trending: {regime_distribution.get('trending', 0.0):.1%}

[DATA ACCESS]
• Attempts: {self._data_access_attempts}
• Successes: {self._successful_data_extractions}
• Success Rate: {success_rate:.1%}

[THEME]
• Theme Confidence: {self._theme_integration_score:.3f}
• Transitions: {self._regime_metrics['transitions']}
• Metrics Snapshots: {len(self._fractal_metrics_history)}
""".strip()

    # ─────────────────────────────────────────────────────────
    # Action and confidence (optional helpers)
    # ─────────────────────────────────────────────────────────
    async def propose_action(self, **_inputs: Any) -> Dict[str, Any]:
        if not hasattr(self, "_action_dim"):
            self._action_dim = 2
        action = np.zeros(self._action_dim, np.float32)

        if self.label == "trending":
            base_signal = self._trend_direction * self.regime_strength
            duration = 0.7
        elif self.label == "volatile":
            base_signal = -self._trend_direction * self.regime_strength * 0.5
            duration = 0.3
        else:  # noise
            base_signal = 0.0
            duration = 0.5

        base_signal *= (self._regime_stability_score / 100.0)

        for i in range(0, self._action_dim, 2):
            action[i] = base_signal
            if i + 1 < self._action_dim:
                action[i + 1] = duration

        return {
            "action": action.tolist(),
            "confidence": self.confidence(),
            "thesis": f"Fractal regime {self.label} with strength {np.clip(self.regime_strength,0,1):.3f}",
        }

    async def calculate_confidence(self, action: Dict[str, Any], **_inputs) -> float:
        return self.confidence()

    def confidence(self, obs: Any = None, info_bus: Optional[Any] = None) -> float:
        base = float(np.clip(self.regime_strength, 0.0, 1.0))
        if self.label == "trending":
            base = min(base * 1.3, 1.0)
        elif self.label == "noise":
            base *= 0.7
        stability = self._regime_stability_score / 100.0
        adjusted = base * (0.5 + 0.5 * stability)
        success_rate = self._successful_data_extractions / max(self._data_access_attempts, 1)
        final_conf = adjusted * (0.5 + 0.5 * success_rate)
        return float(np.clip(final_conf, 0.0, 1.0))
