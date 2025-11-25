# ─────────────────────────────────────────────────────────────
# File: modules/market/market_module.py
# Unified Market Analysis Module - Production Grade
# Orchestrates all market analysis components with advanced debugging
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np
from modules.utils.session_utils import normalize_session_name

from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.core.mixins import (
    SmartInfoBusTradingMixin,
    SmartInfoBusVotingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
)

# Components
from .components.fractal_regime import FractalRegimeComponent
from .components.liquidity_heatmap import LiquidityHeatmapComponent
from .components.theme_detector import ThemeDetectorComponent
from .components.regime_matrix import RegimeMatrixComponent
from .components.time_risk import TimeRiskComponent

# Shared utilities
from .shared.base_component import ComponentResult, ComponentStatus, BaseMarketComponent
from .shared.data_extractors import UnifiedDataExtractor
from .shared.state_manager import StateManager
from .shared.metrics_tracker import MetricsTracker
from .shared.circuit_breaker import CircuitBreaker

# Debug
from .debug.trace_logger import TraceLogger, TraceLevel
from .debug.diagnostics import DiagnosticsEngine
from .debug.visualizer import DebugVisualizer


@dataclass
class MarketConfig:
    """Unified configuration for all market components"""

    # Component enable flags (re-enabled after fixing cache TTL issue)
    enable_fractal: bool = True
    enable_liquidity: bool = True   # RE-ENABLED: Cache fix resolves 54k bar processing
    enable_theme: bool = True       # RE-ENABLED: Will use cached data
    enable_regime_matrix: bool = True  # RE-ENABLED: Performance optimized
    enable_time_risk: bool = True

    # Execution strategy
    parallel_execution: bool = True
    component_timeout_ms: float = 1500
    # Optional per-component overrides (name -> ms)
    per_component_timeouts_ms: Dict[str, int] = field(default_factory=dict)
    # Optional cap on parallel workers
    max_parallel_components: int = 8

    # Debug settings
    debug_enabled: bool = True
    trace_level: TraceLevel = TraceLevel.TRACE
    debug_visualization: bool = True
    performance_profiling: bool = True

    # Logging output configuration
    # Write logs to console and/or file (JSONL with rotation)
    console_log: bool = False
    log_to_file: bool = True
    log_file: Optional[str] = None  # if None and log_to_file=True, defaults to logs/market/unified_market.jsonl
    rotate_size_mb: Optional[int] = 10
    rotate_backups: int = 3
    sample_by_level: Dict[str, float] = field(default_factory=dict)  # e.g., {"TRACE": 0.1}
    dedup_window_sec: float = 1.0

    # Resource management (optional guard)
    max_memory_mb: int = 512
    enable_gpu: bool = True
    resource_guard_enabled: bool = False
    memory_soft_limit_pct: float = 0.85  # 85% of system RAM
    degrade_components: List[str] = field(default_factory=list)

    # Component specific configs (must be Dicts for Pylance friendliness)
    fractal_config: Dict[str, Any] = field(default_factory=dict)
    liquidity_config: Dict[str, Any] = field(default_factory=dict)
    theme_config: Dict[str, Any] = field(default_factory=dict)
    regime_config: Dict[str, Any] = field(default_factory=dict)
    time_risk_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize component configs with defaults if not provided
        # Set default log file if enabled and path not provided
        if self.log_to_file and not self.log_file:
            self.log_file = "logs/market/unified_market.jsonl"
        if not self.fractal_config:
            self.fractal_config = {
                "window": 100,
                "coeff_h": 0.40,
                "coeff_vr": 0.30,
                "coeff_we": 0.30,
            }
        if not self.liquidity_config:
            self.liquidity_config = {
                "lstm_units": 64,
                "sequence_length": 20,
                "enable_gpu": self.enable_gpu,
            }
        if not self.theme_config:
            self.theme_config = {
                "n_themes": 4,
                "window": 100,
                "batch_size": 64,
            }
        if not self.regime_config:
            self.regime_config = {
                "n_regimes": 3,
                "decay_factor": 0.95,
            }
        if not self.time_risk_config:
            self.time_risk_config = {
                "asian_end": 8,
                "euro_end": 16,
                "us_end": 22,
            }


@module(
    **module_args(
        "UnifiedMarketModule",
        description="Unified market analysis orchestrating fractal, liquidity, theme, regime, and time-risk components",
        error_handling=True,
        hot_reload=True,
        timeout_ms=10000,
    )
)
class UnifiedMarketModule(
    BaseModule,
    SmartInfoBusTradingMixin,
    SmartInfoBusVotingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
):
    """
    Production-grade unified market analysis module.
    Orchestrates all market components with advanced debugging and monitoring.
    """

    def __init__(self, config: Optional[MarketConfig] = None, **kwargs):
        """Initialize unified market module with all components"""
        # Keep a strongly-typed dataclass for attribute access
        self._cfg: MarketConfig = config or MarketConfig()
        # Keep a dict for BaseModule and external consumers that expect a dict
        self.config: Dict[str, Any] = asdict(self._cfg)

        # Call parent init (expects a dict) early to satisfy BaseModule contracts
        # and allow DI/error pinpointer setup before our custom systems.
        super().__init__(config=self.config, **kwargs)

        # Initialize debug and shared systems after base init so our logger/trace
        # override the base logger cleanly.
        self._init_debug_systems()
        self._init_shared_systems()

        # Initialize components
        self._init_components()

        # Initialize orchestration
        self._init_orchestration()

        self.trace("UnifiedMarketModule initialized successfully", level=TraceLevel.INFO)

    # ─────────────────────────────────────────────────────────
    # Initialization blocks
    # ─────────────────────────────────────────────────────────
    def _initialize(self):
        """BaseModule initialization hook (no-op).
        Main initialization is performed explicitly in __init__ to control order.
        Keep this method to satisfy the abstract contract and support reset hooks.
        """
        # Safe to log if trace is available (it is, since __init__ sets it before super()).
        if hasattr(self, "trace"):
            self.trace("BaseModule _initialize hook executed (no-op)", level=TraceLevel.TRACE)
        return None

    def _init_debug_systems(self):
        """Initialize debug and monitoring systems"""
        self.logger = TraceLogger(
            name="UnifiedMarketModule",
            trace_level=self._cfg.trace_level,
            enabled=self._cfg.debug_enabled,
            console=self._cfg.console_log,
            output_file=(self._cfg.log_file if self._cfg.log_to_file else None),
            rotate_size_mb=self._cfg.rotate_size_mb,
            rotate_backups=int(self._cfg.rotate_backups),
            sample_by_level=self._cfg.sample_by_level or None,
            dedup_window_sec=float(self._cfg.dedup_window_sec),
        )

        self.diagnostics = DiagnosticsEngine(
            profiling=self._cfg.performance_profiling, memory_tracking=True
        )

        if self._cfg.debug_visualization:
            self.visualizer = DebugVisualizer()
        else:
            self.visualizer = None

        # Expose a shorthand
        self.trace = self.logger.trace

    def _init_shared_systems(self):
        """Initialize shared infrastructure"""
        self.trace("Initializing shared systems", level=TraceLevel.DEBUG)

        # Unified data extraction
        self.data_extractor = UnifiedDataExtractor(logger=self.logger, cache_enabled=True)

        # State management
        self.state_manager = StateManager(logger=self.logger, hot_reload_enabled=True)

        # Metrics tracking
        self.metrics_tracker = MetricsTracker(
            logger=self.logger, detailed_tracking=self._cfg.performance_profiling
        )

        # Circuit breakers for each component
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "fractal": CircuitBreaker(name="fractal", threshold=3),
            "liquidity": CircuitBreaker(name="liquidity", threshold=3),
            "theme": CircuitBreaker(name="theme", threshold=3),
            "regime": CircuitBreaker(name="regime", threshold=3),
            "time_risk": CircuitBreaker(name="time_risk", threshold=3),
        }

        # Concurrency limiter
        self._concurrency_sem = asyncio.Semaphore(
            max(1, int(self._cfg.max_parallel_components))
        )

        self.trace("Shared systems initialized", level=TraceLevel.DEBUG)

    def _init_components(self):
        """Initialize market analysis components"""
        self.trace("Initializing market components", level=TraceLevel.DEBUG)

        self.components: Dict[str, BaseMarketComponent] = {}

        # Initialize each component conditionally
        if self._cfg.enable_fractal:
            self.trace("Creating FractalRegimeComponent", level=TraceLevel.TRACE)
            self.components["fractal"] = FractalRegimeComponent(
                config=self._cfg.fractal_config, logger=self.logger, metrics_tracker=self.metrics_tracker
            )

        if self._cfg.enable_liquidity:
            self.trace("Creating LiquidityHeatmapComponent", level=TraceLevel.TRACE)
            self.components["liquidity"] = LiquidityHeatmapComponent(
                config=self._cfg.liquidity_config, logger=self.logger, metrics_tracker=self.metrics_tracker
            )

        if self._cfg.enable_theme:
            self.trace("Creating ThemeDetectorComponent", level=TraceLevel.TRACE)
            self.components["theme"] = ThemeDetectorComponent(
                config=self._cfg.theme_config, logger=self.logger, metrics_tracker=self.metrics_tracker
            )

        if self._cfg.enable_regime_matrix:
            self.trace("Creating RegimeMatrixComponent", level=TraceLevel.TRACE)
            self.components["regime"] = RegimeMatrixComponent(
                config=self._cfg.regime_config, logger=self.logger, metrics_tracker=self.metrics_tracker
            )

        if self._cfg.enable_time_risk:
            self.trace("Creating TimeRiskComponent", level=TraceLevel.TRACE)
            self.components["time_risk"] = TimeRiskComponent(
                config=self._cfg.time_risk_config, logger=self.logger, metrics_tracker=self.metrics_tracker
            )

        self.trace(f"Initialized {len(self.components)} components", level=TraceLevel.INFO)

    def _init_orchestration(self):
        """Initialize component orchestration logic"""
        self.trace("Initializing orchestration", level=TraceLevel.DEBUG)

        # Define component dependencies
        self.dependencies: Dict[str, Set[str]] = {
            "fractal": set(),       # No dependencies
            "liquidity": set(),     # No dependencies
            "theme": set(),         # No dependencies
            "regime": {"fractal"},  # Depends on fractal for regime detection
            "time_risk": {"liquidity"},  # Depends on liquidity for volatility
        }

        # Calculate execution order
        self.execution_order = self._calculate_execution_order()

        self.trace(f"Execution order: {self.execution_order}", level=TraceLevel.DEBUG)

    # ─────────────────────────────────────────────────────────
    # Orchestration helpers
    # ─────────────────────────────────────────────────────────
    def _calculate_execution_order(self) -> List[List[str]]:
        """Calculate parallel execution groups based on dependencies"""
        self.trace("Calculating execution order", level=TraceLevel.TRACE)

        # Topological sort with level grouping for parallel execution
        levels: List[List[str]] = []
        remaining = set(self.components.keys())
        satisfied: Set[str] = set()

        while remaining:
            level: List[str] = []
            for comp in list(remaining):
                deps = self.dependencies.get(comp, set())
                if deps.issubset(satisfied):
                    level.append(comp)

            if not level:
                # Circular dependency or missing component; bail out gracefully
                self.trace(
                    f"Warning: Could not resolve dependencies for {remaining}",
                    level=TraceLevel.WARNING,
                )
                level = list(remaining)

            levels.append(level)
            satisfied.update(level)
            remaining.difference_update(level)

        self.trace(f"Execution levels: {levels}", level=TraceLevel.TRACE)
        return levels

    def _get_component_timeout_s(self, comp_name: str) -> float:
        """Resolve timeout for a component in seconds (with per-component override)."""
        ms = self._cfg.per_component_timeouts_ms.get(comp_name, self._cfg.component_timeout_ms)
        try:
            return max(0.1, float(ms) / 1000.0)
        except Exception:
            return max(0.1, float(self._cfg.component_timeout_ms) / 1000.0)

    # ─────────────────────────────────────────────────────────
    # Public entry points
    # ─────────────────────────────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Main processing method orchestrating all components
        """
        start_time = time.time()
        self.trace("=" * 60, level=TraceLevel.INFO)
        self.trace("Starting UnifiedMarketModule.process()", level=TraceLevel.INFO)
        self.trace(f"Input keys: {list(inputs.keys())}", level=TraceLevel.DEBUG)

        # Start diagnostics
        if self._cfg.performance_profiling:
            self.diagnostics.start_profiling("process")

        try:
            # Extract unified market data
            self.trace("Extracting market data", level=TraceLevel.DEBUG)
            market_data = await self._extract_market_data(**inputs)
            self.trace(f"Market data extracted: {self._summarize_data(market_data)}", level=TraceLevel.TRACE)

            # Optionally degrade if resource guard says so (safe no-op if disabled)
            self._resource_guard_degrade_if_needed()

            # Check circuit breakers
            self._check_circuit_breakers()

            # Execute components
            self.trace("Executing components", level=TraceLevel.DEBUG)
            component_results = await self._execute_components(market_data)

            # Aggregate results
            self.trace("Aggregating results", level=TraceLevel.DEBUG)
            aggregated = await self._aggregate_results(component_results)

            # Ensure required 'timestamps' key is present per contract
            if 'timestamps' not in aggregated or not isinstance(aggregated.get('timestamps'), list):
                aggregated['timestamps'] = self._extract_timestamps_list(market_data)
            elif not aggregated['timestamps']:
                aggregated['timestamps'] = self._extract_timestamps_list(market_data)

            # Ensure required 'liquidity_thesis' is present per contract
            if not aggregated.get('liquidity_thesis'):
                aggregated['liquidity_thesis'] = self._build_liquidity_thesis(aggregated)

            # Ensure required theme health/status fields are present per contract
            if 'theme_detector_health' not in aggregated or 'theme_detector_status' not in aggregated:
                theme_result = component_results.get('theme')
                health, status = self._build_theme_health(theme_result)
                aggregated['theme_detector_health'] = aggregated.get('theme_detector_health', health)
                aggregated['theme_detector_status'] = aggregated.get('theme_detector_status', status)

            # Generate unified thesis
            thesis = await self._generate_unified_thesis(aggregated)

            # Ensure all contract-required outputs exist with safe defaults
            aggregated = self._ensure_contract_outputs(aggregated, component_results, thesis)

            # Update SmartInfoBus
            await self._update_smart_bus(aggregated, thesis)

            # Record metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics_tracker.record_success("unified_process", processing_time)

            # Visualize if enabled
            if self.visualizer:
                self.visualizer.update(aggregated)

            # Add per-symbol timeframe dynamic market_data_{symbol}_{tf} keys (fresh each step)
            try:
                from modules.utils.info_bus import InfoBusManager as _IBM
                _bus = _IBM.get_instance()
                _mtf = _bus.get('historical_prices', 'UnifiedMarketModule') or {}
                if isinstance(_mtf, dict):
                    for _sym, _tfs in _mtf.items():
                        if not isinstance(_tfs, dict):
                            continue
                        _sym_key = str(_sym).replace('/', '').replace('_', '')
                        for _tf, _rec in _tfs.items():
                            if not isinstance(_rec, dict):
                                continue
                            _cur = _rec.get('current_bar') if isinstance(_rec.get('current_bar'), dict) else {
                                'open': _rec.get('open'),
                                'high': _rec.get('high'),
                                'low': _rec.get('low'),
                                'close': _rec.get('close'),
                                'volume': _rec.get('volume'),
                            }
                            _val = {
                                'symbol': _sym,
                                'timeframe': _tf,
                                'current_bar': _cur,
                                'ts': datetime.datetime.utcnow().isoformat(),
                            }
                            # Publish both sanitized and raw symbol variants to satisfy all consumers
                            aggregated[f"market_data_{_sym_key}_{_tf}"] = _val
                            aggregated[f"market_data_{_sym}_{_tf}"] = _val
            except Exception:
                pass

            # Add metadata
            aggregated["_metadata"] = {
                "processing_time_ms": processing_time,
                "components_executed": list(component_results.keys()),
                "timestamp": datetime.datetime.now().isoformat(),
                "success": True,
            }
            aggregated["_thesis"] = thesis
            aggregated["thesis"] = thesis

            self.trace(f"Process completed in {processing_time:.2f}ms", level=TraceLevel.INFO)
            self.trace("=" * 60, level=TraceLevel.INFO)

            return aggregated

        except Exception as e:
            self.trace(f"Error in process: {e}", level=TraceLevel.ERROR, exc_info=True)
            return await self._handle_process_error(e, start_time)

        finally:
            if self._cfg.performance_profiling:
                profile = self.diagnostics.stop_profiling("process")
                self.trace(f"Performance profile: {profile}", level=TraceLevel.DEBUG)

    # ─────────────────────────────────────────────────────────
    # Execution internals
    # ─────────────────────────────────────────────────────────
    def _extract_timestamps_list(self, market_data: Dict[str, Any]) -> List[str]:
        """Derive a timestamps list from market_data or provide a safe fallback.
        Returns a list of ISO-8601 strings (UTC).
        """
        try:
            ts = market_data.get('timestamps')
            # Normalize to a python list
            if ts is None:
                # Try single 'timestamp' field
                single = market_data.get('timestamp')
                if single is not None:
                    ts = [single]
                else:
                    ts = []

            tolist_fn = getattr(ts, 'tolist', None)
            if callable(tolist_fn):
                try:
                    tmp = tolist_fn()
                    if isinstance(tmp, (list, tuple)):
                        seq = list(tmp)
                    else:
                        seq = [tmp]
                except Exception:
                    seq = [ts]
            elif isinstance(ts, (list, tuple)):
                seq = list(ts)
            else:
                seq = [ts]

            out: List[str] = []
            for item in seq:
                # numpy datetime64
                if isinstance(item, np.datetime64):
                    try:
                        out.append(np.datetime_as_string(item, unit='ms'))
                        continue
                    except Exception:
                        pass
                # datetime
                if isinstance(item, datetime.datetime):
                    dt = item
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=datetime.timezone.utc)
                    else:
                        dt = dt.astimezone(datetime.timezone.utc)
                    out.append(dt.isoformat())
                    continue
                # numeric epoch
                if isinstance(item, (int, float)):
                    try:
                        dt = datetime.datetime.utcfromtimestamp(float(item)).replace(tzinfo=datetime.timezone.utc)
                        out.append(dt.isoformat())
                        continue
                    except Exception:
                        pass
                # string or unknown
                try:
                    out.append(str(item))
                except Exception:
                    pass

            if not out:
                # Safe fallback: use now
                now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
                out = [now]
            # Cap length to avoid huge payloads
            return out[:300]
        except Exception:
            now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
            return [now]
    
    def _build_liquidity_thesis(self, aggregated: Dict[str, Any]) -> str:
        """
        Create a compact liquidity_thesis string using fields that actually exist
        in LiquidityHeatmapComponent outputs. Robust to missing pieces.
        """
        try:
            parts: List[str] = []

            # 1) Liquidity score
            score = aggregated.get('liquidity_score')
            if isinstance(score, (int, float)) and np.isfinite(score):
                parts.append(f"score={float(score):.2%}")

            # 2) Spread (prefer 'current_spread', fallback to 'average_spread')
            spread = aggregated.get('spread_analysis') or {}
            if isinstance(spread, dict):
                cur_spread = spread.get('current_spread')
                avg_spread = spread.get('average_spread')
                if isinstance(cur_spread, (int, float)) and np.isfinite(cur_spread):
                    parts.append(f"spread={float(cur_spread):.6f}")
                elif isinstance(avg_spread, (int, float)) and np.isfinite(avg_spread):
                    parts.append(f"avg_spread={float(avg_spread):.6f}")

            # 3) Depth (use 'average_depth' if available)
            depth = aggregated.get('market_depth') or {}
            if isinstance(depth, dict):
                avg_depth = depth.get('average_depth')
                if isinstance(avg_depth, (int, float)) and np.isfinite(avg_depth):
                    parts.append(f"avg_depth={float(avg_depth):.0f}")

            # 4) Session (prefer Liquidity 'session_data.active_session',
            #    fallback to TimeRisk 'session_risk.current_session')
            session_data = aggregated.get('session_data') or {}
            session_risk = aggregated.get('session_risk') or {}
            active_session = None
            if isinstance(session_data, dict):
                active_session = (
                    session_data.get('active_session')
                    or session_data.get('current_session')
                    or session_data.get('session')
                )
            if not isinstance(active_session, str) and isinstance(session_risk, dict):
                active_session = session_risk.get('current_session')

            if isinstance(active_session, str):
                parts.append(f"session={active_session}")

            return "Liquidity: " + ", ".join(parts) if parts else "Liquidity status computed"
        except Exception:
            return "Liquidity status computed"


    def _ensure_contract_outputs(
        self,
        aggregated: Dict[str, Any],
        component_results: Dict[str, 'ComponentResult'],
        thesis: str,
    ) -> Dict[str, Any]:
        """
        Fill in any missing contract-provided keys with sensible defaults.
        Keeps existing values intact. Ensures types match what downstream expects.
        """
        try:
            # Critical missing data keys that many modules depend on
            # Generate market_context from current aggregated data
            # FIX: Properly infer session if not available from components
            session_name = None
            session_data = aggregated.get('session_data', {})
            if isinstance(session_data, dict):
                session_name = session_data.get('current_session')

            # Fallback: infer from time_risk_analysis or current time
            if not session_name or session_name == 'unknown':
                time_risk = aggregated.get('time_risk_analysis', {})
                if isinstance(time_risk, dict):
                    sess_abbr = time_risk.get('session')
                    # Map abbreviated sessions to full names
                    if sess_abbr == 'AS':
                        session_name = 'asian'
                    elif sess_abbr == 'EU':
                        session_name = 'european'
                    elif sess_abbr == 'US':
                        session_name = 'american'
                    elif sess_abbr == 'OFF':
                        session_name = 'closed'

            # Try to get timestamp from market data (for training mode) - do this first
            data_timestamp = None
            timestamps_list = aggregated.get('timestamps', [])
            if timestamps_list and len(timestamps_list) > 0:
                try:
                    ts_str = timestamps_list[-1]  # Use most recent timestamp
                    if isinstance(ts_str, str):
                        # Parse ISO format timestamp
                        if ts_str.endswith('Z'):
                            ts_str = ts_str[:-1] + '+00:00'
                        data_timestamp = datetime.datetime.fromisoformat(ts_str)
                    elif isinstance(ts_str, (int, float)):
                        data_timestamp = datetime.datetime.utcfromtimestamp(ts_str)
                except Exception:
                    pass
            
            # Also try timestamp from inputs or market_data
            if data_timestamp is None:
                ts_input = aggregated.get('timestamp')
                if isinstance(ts_input, str):
                    try:
                        if ts_input.endswith('Z'):
                            ts_input = ts_input[:-1] + '+00:00'
                        data_timestamp = datetime.datetime.fromisoformat(ts_input)
                    except Exception:
                        pass

            # Final fallback: infer session from data timestamp (for training) or current UTC time (for live)
            if not session_name or session_name == 'unknown':
                from modules.utils.session_utils import infer_market_session
                # Use data timestamp for session inference (important for training!)
                session_name = infer_market_session(data_timestamp)

            # Use data timestamp for market_context if available (for training consistency)
            context_timestamp = (
                data_timestamp.isoformat() if data_timestamp 
                else datetime.datetime.utcnow().isoformat()
            )
            
            aggregated.setdefault('market_context', {
                'regime': aggregated.get('market_regime', 'unknown'),
                'volatility_level': aggregated.get('volatility_level', 'medium'),
                'session': normalize_session_name(session_name),
                'theme': aggregated.get('market_theme', 0),
                'liquidity_score': aggregated.get('liquidity_score', 0.5),
                'timestamp': context_timestamp
            })

            # Generate step_idx as incremental counter
            aggregated.setdefault('step_idx', int(time.time() * 1000) % 1000000)  # Simple step counter

            # NOTE: prices and price_data are provided by MarketDataProvider, not UnifiedMarketModule
            # Removed illegal publications to stop provider ownership conflicts

            # Fractal / Regime (legacy coverage)
            aggregated.setdefault('fractal_metrics', {})
            aggregated.setdefault('market_regime', aggregated.get('market_regime', 'unknown'))
            aggregated.setdefault('regime_data', {})
            aggregated.setdefault('regime_strength', float(aggregated.get('regime_strength', 0.0)))
            # FIX: trend_direction must be a float ([-1, 1]), not a string
            aggregated.setdefault('trend_direction', float(aggregated.get('trend_direction', 0.0)))

            # Liquidity
            aggregated.setdefault('liquidity_capabilities', {})
            aggregated.setdefault('liquidity_prediction', {})
            aggregated.setdefault('liquidity_score', float(aggregated.get('liquidity_score', 0.5)))
            aggregated.setdefault('liquidity_thesis', self._build_liquidity_thesis(aggregated))
            aggregated.setdefault('market_depth', aggregated.get('market_depth', {}))
            aggregated.setdefault('session_data', aggregated.get('session_data', {}))
            aggregated.setdefault('spread_analysis', aggregated.get('spread_analysis', {}))
            aggregated.setdefault('trading_sessions', aggregated.get('trading_sessions', {}))

            # Theme
            aggregated.setdefault('market_theme', int(aggregated.get('market_theme', 0)))
            aggregated.setdefault('theme_detection', aggregated.get('theme_detection', {}))
            if 'theme_detector_health' not in aggregated or 'theme_detector_status' not in aggregated:
                theme_result = component_results.get('theme')
                health, status = self._build_theme_health(theme_result)
                aggregated.setdefault('theme_detector_health', health)
                aggregated.setdefault('theme_detector_status', status)
            aggregated.setdefault('theme_strength', float(aggregated.get('theme_strength', 0.0)))
            aggregated.setdefault('theme_transition', float(aggregated.get('theme_transition', 0.0)))
            # REMOVED: theme_confidence to avoid conflict with EnhancedThemeExpert's canonical theme_confidence
            # EnhancedThemeExpert is the specialized voting expert that owns theme_confidence
            # UnifiedMarketModule provides raw theme_strength and theme_transition instead

            # Regime performance matrix
            aggregated.setdefault('backtesting_data', aggregated.get('backtesting_data', {}))
            aggregated.setdefault('performance_metrics', aggregated.get('performance_metrics', {}))
            aggregated.setdefault('regime_accuracy', aggregated.get('regime_accuracy', {
                'value': 0.0,
                'by_regime': {},
                'current_regime_accuracy': 0.0,
                'last_update': datetime.datetime.utcnow().isoformat() + 'Z'
            }))
            aggregated.setdefault('regime_analysis', aggregated.get('regime_analysis', {}))
            aggregated.setdefault('regime_matrix_analysis', aggregated.get('regime_matrix_analysis', {}))
            aggregated.setdefault('regime_matrix_health', aggregated.get('regime_matrix_health', {'status': 'UNKNOWN'}))
            aggregated.setdefault('regime_matrix_status', aggregated.get('regime_matrix_status', 'UNKNOWN'))
            aggregated.setdefault('regime_performance', aggregated.get('regime_performance', {
                'matrix': [],
                'current_regime': 0,
                'predicted_regime': 0,
                'avg_performance': 0.0,
            }))
            aggregated.setdefault('regime_prediction', aggregated.get('regime_prediction', {
                'predicted': 0,
                'actual': 0,
                'correct': False,
            }))
            aggregated.setdefault('stress_test_results', aggregated.get('stress_test_results', {}))

            # Time-aware risk scaling
            aggregated.setdefault('risk_scaling_factor', float(aggregated.get('risk_scaling_factor', 1.0)))
            aggregated.setdefault('session_risk', aggregated.get('session_risk', {
                'risk_level': 0.5,
                'current_session': 'unknown',
            }))
            aggregated.setdefault('time_risk_health', aggregated.get('time_risk_health', {'status': 'UNKNOWN'}))
            aggregated.setdefault('time_risk_status', aggregated.get('time_risk_status', 'UNKNOWN'))
            # Ensure a minimal, always-available time_risk_analysis baseline
            if not aggregated.get('time_risk_analysis'):
                try:
                    now = datetime.datetime.utcnow()
                    hour = now.hour
                    minute = now.minute
                    # Simple session mapping by UTC hour
                    if 0 <= hour < 8:
                        sess = 'AS'
                    elif 8 <= hour < 16:
                        sess = 'EU'
                    elif 16 <= hour < 22:
                        sess = 'US'
                    else:
                        sess = 'OFF'
                    # Rollover window hint (approximate FX rollover)
                    is_roll = bool(21 <= hour < 23)
                    vol_hint = aggregated.get('volatility_adjustment')
                    if isinstance(vol_hint, (int, float)):
                        vol_label = 'high' if float(vol_hint) > 1.2 else ('medium' if float(vol_hint) > 0.9 else 'low')
                    else:
                        vol_label = 'medium'
                    aggregated['time_risk_analysis'] = {
                        'session': sess,
                        'minute_in_session': int((hour % 24) * 60 + minute),
                        'is_rollover_window': bool(is_roll),
                        'volatility_hint': str(vol_label),
                    }
                except Exception:
                    aggregated.setdefault('time_risk_analysis', {'session': 'UNKNOWN', 'minute_in_session': 0, 'is_rollover_window': False, 'volatility_hint': 'medium'})
            aggregated.setdefault('volatility_adjustment', float(aggregated.get('volatility_adjustment', 1.0)))

            # Unified additions
            aggregated.setdefault('market_analysis_thesis', thesis)
            if 'unified_market_analysis' not in aggregated:
                try:
                    keys_preview = list(aggregated.keys())[:25]
                except Exception:
                    keys_preview = []
                aggregated['unified_market_analysis'] = {
                    'keys_preview': keys_preview,
                    'components_executed': list(aggregated.get('_metadata', {}).get('components_executed', []))
                        if isinstance(aggregated.get('_metadata'), dict) else [],
                }

            return aggregated
        except Exception:
            aggregated.setdefault('market_analysis_thesis', thesis)
            return aggregated

    def _build_theme_health(self, theme_result: Optional[ComponentResult]) -> Tuple[Dict[str, Any], str]:
        """Synthesize theme detector health/status from component result (or defaults)."""
        try:
            threshold = 0.35
            try:
                threshold = float(self._cfg.theme_config.get('clustering_quality_threshold', 0.35))
            except Exception:
                threshold = 0.35

            health: Dict[str, Any] = {
                'processing_success': False,
                'clustering_quality': 0.0,
                'fit_count': 0,
                'model_ready': False,
            }

            status = 'DISABLED' if not self._cfg.enable_theme else 'UNKNOWN'

            if theme_result is None:
                return health, status

            # Map component status to string
            comp_status = getattr(theme_result, 'status', None)
            data = getattr(theme_result, 'data', {}) or {}

            # extract fields
            quality = float(data.get('clustering_quality', 0.0)) if isinstance(data.get('clustering_quality', 0.0), (int, float)) else 0.0
            diagnostics = data.get('diagnostics') if isinstance(data.get('diagnostics'), dict) else {}
            fit_count = int(diagnostics.get('fit_count', 0)) if isinstance(diagnostics, dict) else 0
            proc_ok = bool(data.get('processing_success', False))

            model_ready = bool(quality >= threshold)

            health.update({
                'processing_success': proc_ok,
                'clustering_quality': quality,
                'fit_count': fit_count,
                'model_ready': model_ready,
            })

            # status string
            try:
                from .shared.base_component import ComponentStatus  # local import for enum
                if comp_status == ComponentStatus.SUCCESS:
                    if model_ready:
                        status = 'READY'
                    elif quality > 0.0:
                        status = 'WARMING_UP'
                    else:
                        status = 'INITIALIZING'
                elif comp_status == ComponentStatus.TIMEOUT:
                    status = 'TIMEOUT'
                elif comp_status == ComponentStatus.ERROR:
                    status = 'ERROR'
                elif comp_status == ComponentStatus.CIRCUIT_OPEN:
                    status = 'DISABLED'
                else:
                    status = 'UNKNOWN'
            except Exception:
                status = 'UNKNOWN'

            return health, status
        except Exception:
            return {
                'processing_success': False,
                'clustering_quality': 0.0,
                'fit_count': 0,
                'model_ready': False,
            }, 'UNKNOWN'
    async def _extract_market_data(self, **inputs) -> Dict[str, Any]:
        """Extract and normalize market data for all components"""
        self.trace("UnifiedDataExtractor.extract() starting", level=TraceLevel.TRACE)

        # Use unified extractor
        market_data = await self.data_extractor.extract(
            sources=["smartinfobus", "inputs", "cache"],
            **inputs,
        )

        self.trace(f"Extracted data sources: {market_data.get('_sources', [])}", level=TraceLevel.TRACE)
        return market_data

    def _check_circuit_breakers(self):
        """Check all circuit breakers before execution"""
        self.trace("Checking circuit breakers", level=TraceLevel.TRACE)

        for name, breaker in self.circuit_breakers.items():
            state = breaker.get_state()
            self.trace(f"Circuit breaker '{name}': {state}", level=TraceLevel.TRACE)

            if state == "OPEN":
                self.trace(
                    f"Circuit breaker '{name}' is OPEN, component disabled",
                    level=TraceLevel.WARNING,
                )

    async def _execute_components(self, market_data: Dict[str, Any]) -> Dict[str, ComponentResult]:
        """Execute components according to dependency order"""
        self.trace("Component execution starting", level=TraceLevel.DEBUG)

        results: Dict[str, ComponentResult] = {}
        shared_context: Dict[str, Any] = {}  # For passing data between dependent components

        for level in self.execution_order:
            self.trace(f"Executing level: {level}", level=TraceLevel.DEBUG)

            if self._cfg.parallel_execution and len(level) > 1:
                # Execute components in parallel
                level_results = await self._execute_parallel(level, market_data, shared_context)
            else:
                # Execute sequentially
                level_results = await self._execute_sequential(level, market_data, shared_context)

            # Update results and shared context
            for comp_name, result in level_results.items():
                results[comp_name] = result
                if result.status == ComponentStatus.SUCCESS:
                    shared_context[comp_name] = result.data

        self.trace(f"Component execution completed: {list(results.keys())}", level=TraceLevel.DEBUG)
        return results

    async def _execute_parallel(
        self,
        components: List[str],
        market_data: Dict[str, Any],
        shared_context: Dict[str, Any],
    ) -> Dict[str, ComponentResult]:
        """Execute components in parallel"""
        self.trace(f"Parallel execution of: {components}", level=TraceLevel.TRACE)

        # Prepare tasks
        tasks: List[Tuple[str, asyncio.Task]] = []
        for comp_name in components:
            if comp_name in self.components:
                # Wrap in semaphore to avoid unbounded parallelism
                async def run_with_sem(name=comp_name):
                    async with self._concurrency_sem:
                        return await self._execute_component(name, market_data, shared_context)

                tasks.append((comp_name, asyncio.create_task(run_with_sem())))

        results: Dict[str, ComponentResult] = {}
        for comp_name, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=self._get_component_timeout_s(comp_name))
                results[comp_name] = result
            except asyncio.TimeoutError:
                self.trace(f"Component '{comp_name}' timed out", level=TraceLevel.ERROR)
                results[comp_name] = ComponentResult(
                    component=comp_name,
                    status=ComponentStatus.TIMEOUT,
                    data={},
                    error="Component execution timed out",
                )

        return results

    async def _execute_sequential(
        self,
        components: List[str],
        market_data: Dict[str, Any],
        shared_context: Dict[str, Any],
    ) -> Dict[str, ComponentResult]:
        """Execute components sequentially"""
        self.trace(f"Sequential execution of: {components}", level=TraceLevel.TRACE)

        results: Dict[str, ComponentResult] = {}
        for comp_name in components:
            if comp_name in self.components:
                result = await self._execute_component(comp_name, market_data, shared_context)
                results[comp_name] = result

        return results

    async def _execute_component(
        self,
        comp_name: str,
        market_data: Dict[str, Any],
        shared_context: Dict[str, Any],
    ) -> ComponentResult:
        """Execute a single component with error handling"""
        self.trace(f"Executing component: {comp_name}", level=TraceLevel.TRACE)

        # Check circuit breaker
        breaker = self.circuit_breakers.get(comp_name)
        if breaker and not breaker.allow_request():
            self.trace(f"Circuit breaker blocked: {comp_name}", level=TraceLevel.WARNING)
            return ComponentResult(
                component=comp_name,
                status=ComponentStatus.CIRCUIT_OPEN,
                data={},
                error="Circuit breaker is open",
            )

        component = self.components[comp_name]
        start_time = time.time()

        try:
            # Prepare input data
            input_data = {
                "market_data": market_data,
                "shared_context": shared_context,
            }

            self.trace(
                f"Component {comp_name} input keys: {list(input_data.keys())}",
                level=TraceLevel.TRACE,
            )

            # Execute component
            result = await component.analyze(**input_data)

            # Record success
            execution_time = (time.time() - start_time) * 1000
            self.metrics_tracker.record_success(f"{comp_name}_execution", execution_time)

            if breaker:
                breaker.record_success()

            self.trace(
                f"Component {comp_name} succeeded in {execution_time:.2f}ms",
                level=TraceLevel.DEBUG,
            )

            return ComponentResult(
                component=comp_name,
                status=ComponentStatus.SUCCESS,
                data=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            # Record failure
            self.trace(f"Component {comp_name} failed: {e}", level=TraceLevel.ERROR)

            if breaker:
                breaker.record_failure()

            self.metrics_tracker.record_failure(f"{comp_name}_execution", str(e))

            return ComponentResult(
                component=comp_name,
                status=ComponentStatus.ERROR,
                data={},
                error=str(e),
            )

    # ─────────────────────────────────────────────────────────
    # Aggregation & thesis
    # ─────────────────────────────────────────────────────────
    async def _aggregate_results(self, component_results: Dict[str, ComponentResult]) -> Dict[str, Any]:
        """Aggregate results from all components"""
        self.trace("Aggregating component results", level=TraceLevel.DEBUG)

        aggregated: Dict[str, Any] = {}

        # Collect all successful results
        for comp_name, result in component_results.items():
            if result.status == ComponentStatus.SUCCESS:
                # Merge component data
                for key, value in result.data.items():
                    # Handle key collisions
                    if key in aggregated:
                        self.trace(
                            f"Key collision: {key} from {comp_name}",
                            level=TraceLevel.TRACE,
                        )
                        # Store with component prefix to avoid collision
                        aggregated[f"{comp_name}_{key}"] = value
                    else:
                        aggregated[key] = value

        # Add component health status
        aggregated["component_health"] = {
            comp_name: {
                "status": result.status.value,
                "execution_time_ms": result.execution_time_ms,
                "error": result.error,
            }
            for comp_name, result in component_results.items()
        }

        # Calculate unified metrics
        aggregated["unified_metrics"] = self._calculate_unified_metrics(component_results)

        self.trace(f"Aggregated {len(aggregated)} keys", level=TraceLevel.DEBUG)
        return aggregated

    def _calculate_unified_metrics(self, component_results: Dict[str, ComponentResult]) -> Dict[str, Any]:
        """Calculate unified metrics from component results"""
        self.trace("Calculating unified metrics", level=TraceLevel.TRACE)

        times = [r.execution_time_ms for r in component_results.values() if r.execution_time_ms is not None]
        metrics: Dict[str, Any] = {
            "total_components": len(self.components),
            "successful_components": sum(1 for r in component_results.values() if r.status == ComponentStatus.SUCCESS),
            "failed_components": sum(1 for r in component_results.values() if r.status == ComponentStatus.ERROR),
            "average_execution_time_ms": float(np.mean(times)) if times else 0.0,
        }

        # Add cross-component consensus if available
        if "fractal" in component_results and "theme" in component_results:
            fractal_data = component_results["fractal"].data
            theme_data = component_results["theme"].data

            # Example consensus calculation
            if fractal_data and theme_data:
                regime_consensus = self._calculate_regime_consensus(fractal_data, theme_data)
                metrics["regime_consensus"] = regime_consensus

        return metrics

    def _calculate_regime_consensus(self, fractal_data: Dict[str, Any], theme_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus between regime detection components"""
        self.trace("Calculating regime consensus", level=TraceLevel.TRACE)

        fractal_regime = fractal_data.get("market_regime", "unknown")
        theme_id = theme_data.get("market_theme", -1)

        # Map theme to regime (domain-specific mapping example)
        theme_to_regime = {
            0: "defensive",
            1: "growth",
            2: "volatile",
            3: "ranging",
        }
        theme_regime = theme_to_regime.get(theme_id, "unknown")

        # Calculate agreement (placeholder logic)
        agreement = 1.0 if fractal_regime == theme_regime else 0.5

        return {
            "fractal_regime": fractal_regime,
            "theme_regime": theme_regime,
            "agreement_score": agreement,
            "consensus_regime": fractal_regime if agreement > 0.7 else "mixed",
        }

    async def _generate_unified_thesis(self, aggregated: Dict[str, Any]) -> str:
        """Generate unified thesis from all components"""
        self.trace("Generating unified thesis", level=TraceLevel.DEBUG)

        thesis_parts: List[str] = []

        # Header
        thesis_parts.append("UNIFIED MARKET ANALYSIS")
        thesis_parts.append("=" * 50)

        # Component summaries
        health = aggregated.get("component_health", {})

        if "fractal" in health and health["fractal"]["status"] == "SUCCESS":
            regime = aggregated.get("market_regime", "unknown")
            strength = aggregated.get("regime_strength", 0.0)
            thesis_parts.append(f"\n[FRACTAL] Regime: {regime} (strength: {strength:.2%})")

        if "liquidity" in health and health["liquidity"]["status"] == "SUCCESS":
            liquidity_score = aggregated.get("liquidity_score", 0.0)
            thesis_parts.append(f"[LIQUIDITY] Score: {liquidity_score:.2%}")

        if "theme" in health and health["theme"]["status"] == "SUCCESS":
            theme = aggregated.get("market_theme", -1)
            strength = aggregated.get("theme_strength", 0.0)
            thesis_parts.append(f"[THEME] Theme ID: {theme} (strength: {strength:.2%})")

        if "regime" in health and health["regime"]["status"] == "SUCCESS":
            accuracy = aggregated.get("regime_accuracy", {}).get("value", 0.0)
            thesis_parts.append(f"[REGIME MATRIX] Accuracy: {accuracy:.2%}")

        if "time_risk" in health and health["time_risk"]["status"] == "SUCCESS":
            scaling = aggregated.get("risk_scaling_factor", 1.0)
            session = aggregated.get("session_risk", {}).get("current_session", "unknown")
            thesis_parts.append(f"[TIME RISK] Scaling: {scaling:.2f}x (session: {session})")

        # Unified metrics
        metrics = aggregated.get("unified_metrics", {})
        thesis_parts.append(
            f"\n[UNIFIED] {metrics.get('successful_components', 0)}/{metrics.get('total_components', 0)} components successful"
        )

        if "regime_consensus" in metrics:
            consensus = metrics["regime_consensus"]
            thesis_parts.append(f"[CONSENSUS] Agreement: {consensus.get('agreement_score', 0.0):.2%}")

        # Recommendations
        thesis_parts.append("\n[RECOMMENDATIONS]")
        recommendations = self._generate_recommendations(aggregated)
        thesis_parts.extend(recommendations)

        return "\n".join(thesis_parts)

    def _generate_recommendations(self, aggregated: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations from aggregated data"""
        recommendations: List[str] = []

        # Risk level assessment
        risk_level = aggregated.get("session_risk", {}).get("risk_level", 0.5)
        if risk_level > 0.8:
            recommendations.append("• HIGH RISK: Consider reducing exposure")
        elif risk_level < 0.3:
            recommendations.append("• LOW RISK: Favorable conditions for trading")

        # Liquidity assessment
        liquidity = aggregated.get("liquidity_score", 0.5)
        if liquidity < 0.3:
            recommendations.append("• LOW LIQUIDITY: Widen spreads and reduce size")

        # Regime alignment
        regime = aggregated.get("market_regime", "unknown")
        if regime == "trending":
            recommendations.append("• TRENDING: Follow momentum strategies")
        elif regime == "volatile":
            recommendations.append("• VOLATILE: Use range-based strategies")

        if not recommendations:
            recommendations.append("• NEUTRAL: Standard trading conditions")

        return recommendations

    async def _update_smart_bus(self, aggregated: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with unified results"""
        self.trace("Updating SmartInfoBus", level=TraceLevel.DEBUG)

        # Get InfoBus instance
        from modules.utils.info_bus import InfoBusManager

        smart_bus = InfoBusManager.get_instance()

        # Update all original keys to maintain compatibility
        updates = [
        # Update all original keys to maintain compatibility with CONTRACTS

            # Critical missing data keys that many modules depend on
            ("market_context", aggregated.get("market_context")),
            ("prices", aggregated.get("prices")),
            ("price_data", aggregated.get("price_data")),
            ("step_idx", aggregated.get("step_idx")),

            ("fractal_metrics", aggregated.get("fractal_metrics")),
            ("market_regime", aggregated.get("market_regime")),
            ("regime_data", aggregated.get("regime_data")),
            ("regime_strength", aggregated.get("regime_strength")),
            ("timestamps", aggregated.get("timestamps")),
            ("trend_direction", aggregated.get("trend_direction")),

            # Liquidity
            ("liquidity_capabilities", aggregated.get("liquidity_capabilities")),
            ("liquidity_prediction", aggregated.get("liquidity_prediction")),
            ("liquidity_score", aggregated.get("liquidity_score")),
            ("liquidity_thesis", aggregated.get("liquidity_thesis")),
            ("market_depth", aggregated.get("market_depth")),
            ("session_data", aggregated.get("session_data")),
            ("spread_analysis", aggregated.get("spread_analysis")),
            ("trading_sessions", aggregated.get("trading_sessions")),

            # Theme
            ("market_theme", aggregated.get("market_theme")),
            ("theme_detection", aggregated.get("theme_detection")),
            ("theme_detector_health", aggregated.get("theme_detector_health")),
            ("theme_detector_status", aggregated.get("theme_detector_status")),
            ("theme_strength", aggregated.get("theme_strength")),
            ("theme_transition", aggregated.get("theme_transition")),
            # REMOVED: theme_confidence - owned by EnhancedThemeExpert

            # Regime performance matrix
            ("backtesting_data", aggregated.get("backtesting_data")),
            ("performance_metrics", aggregated.get("performance_metrics")),
            ("regime_accuracy", aggregated.get("regime_accuracy")),
            ("regime_analysis", aggregated.get("regime_analysis")),
            ("regime_matrix_analysis", aggregated.get("regime_matrix_analysis")),
            ("regime_matrix_health", aggregated.get("regime_matrix_health")),
            ("regime_matrix_status", aggregated.get("regime_matrix_status")),
            ("regime_performance", aggregated.get("regime_performance")),
            ("regime_prediction", aggregated.get("regime_prediction")),
            ("stress_test_results", aggregated.get("stress_test_results")),

            # Time-aware risk scaling
            ("risk_scaling_factor", aggregated.get("risk_scaling_factor")),
            ("session_risk", aggregated.get("session_risk")),
            ("time_risk_health", aggregated.get("time_risk_health")),
            ("time_risk_status", aggregated.get("time_risk_status")),
            ("time_risk_analysis", aggregated.get("time_risk_analysis")),
            ("volatility_adjustment", aggregated.get("volatility_adjustment")),

            # Unified additions
            ("unified_market_analysis", aggregated),
            ("market_analysis_thesis", thesis),
        ]

        

        # Respect canonical owners: do not publish keys owned by other modules
        forbidden_keys = {
            'performance_metrics',   # SessionManager owns this
            'step_idx',              # MarketDataProvider owns this
            # NOTE: market_context is published by UnifiedMarketModule since it computes regime/volatility
            # MarketDataProvider only creates a basic placeholder without regime data
        }

        for key, value in updates:
            if value is None:
                continue
            if key in forbidden_keys:
                continue
            self.trace(f"Updating InfoBus: {key}", level=TraceLevel.TRACE)
            smart_bus.set(key, value, module="UnifiedMarketModule", thesis=thesis[:200])

        # Publish dynamic per-symbol timeframe keys if present
        try:
            for k, v in (aggregated or {}).items():
                if isinstance(k, str) and k.startswith('market_data_'):
                    smart_bus.set(k, v, module="UnifiedMarketModule", thesis=thesis[:200])
        except Exception:
            pass

    async def _handle_process_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle process-level errors"""
        self.trace(f"Handling process error: {error}", level=TraceLevel.ERROR)

        processing_time = (time.time() - start_time) * 1000
        self.metrics_tracker.record_failure("unified_process", str(error))

        # Return safe fallback
        return {
            "error": str(error),
            "processing_time_ms": processing_time,
            "success": False,
            "_thesis": f"Market analysis failed: {str(error)[:100]}",
            "thesis": f"Market analysis failed: {str(error)[:100]}",
            # Provide default values for critical keys
            "market_regime": "unknown",
            "regime_strength": 0.0,
            "liquidity_score": 0.5,
            "market_theme": 0,
            "risk_scaling_factor": 1.0,
            "_metadata": {
                "error": True,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        }

    # ─────────────────────────────────────────────────────────
    # Utilities & state
    # ─────────────────────────────────────────────────────────
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """Summarize data for trace logging"""
        if not self._cfg.debug_enabled:
            return "N/A"

        summary = []
        for key, value in data.items():
            if isinstance(value, dict):
                summary.append(f"{key}:dict({len(value)})")
            elif isinstance(value, list):
                summary.append(f"{key}:list({len(value)})")
            elif isinstance(value, np.ndarray):
                summary.append(f"{key}:array{value.shape}")
            else:
                summary.append(f"{key}:{type(value).__name__}")

        return ", ".join(summary[:10])  # Limit to first 10 items

    def get_state(self) -> Dict[str, Any]:
        """Get unified state from all components"""
        self.trace("Getting unified state", level=TraceLevel.DEBUG)

        state: Dict[str, Any] = {
            "config": self.config,  # dict snapshot
            "components": {},
            "metrics": self.metrics_tracker.get_summary(),
            "circuit_breakers": {
                name: breaker.get_state_dict() for name, breaker in self.circuit_breakers.items()
            },
        }

        # Get state from each component
        for name, component in self.components.items():
            state["components"][name] = component.get_state()

        return state

    def set_state(self, state: Dict[str, Any]):
        """Set unified state for all components"""
        self.trace("Setting unified state", level=TraceLevel.DEBUG)

        # Restore component states
        if "components" in state:
            for name, component_state in state["components"].items():
                if name in self.components:
                    self.components[name].set_state(component_state)

        # Restore circuit breakers
        if "circuit_breakers" in state:
            for name, breaker_state in state["circuit_breakers"].items():
                if name in self.circuit_breakers:
                    self.circuit_breakers[name].set_state(breaker_state)

        self.trace("State restored successfully", level=TraceLevel.DEBUG)

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose unified action from all components"""
        results = await self.process(**inputs)

        # Aggregate actions from components
        actions: List[Tuple[str, float]] = []

        if "market_regime" in results:
            if results["market_regime"] == "trending":
                actions.append(("follow_trend", 0.8))
            elif results["market_regime"] == "volatile":
                actions.append(("reduce_size", 0.7))

        if "liquidity_score" in results and results["liquidity_score"] < 0.3:
            actions.append(("avoid_market", 0.9))

        if "risk_scaling_factor" in results and results["risk_scaling_factor"] > 2.0:
            actions.append(("defensive", 0.85))

        # Select highest confidence action
        if actions:
            action, confidence = max(actions, key=lambda x: x[1])
        else:
            action, confidence = "hold", 0.5

        return {
            "action": action,
            "confidence": confidence,
            "rationale": results.get("_thesis", "Based on unified market analysis"),
            "components_considered": list(results.get("component_health", {}).keys()),
        }

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in proposed action"""
        results = await self.process(**inputs)

        # Base confidence on component success rate
        health = results.get("component_health", {})
        success_rate = (
            sum(1 for h in health.values() if h["status"] == "SUCCESS") / max(len(health), 1)
        )

        # Adjust based on consensus
        metrics = results.get("unified_metrics", {})
        if "regime_consensus" in metrics:
            agreement = metrics["regime_consensus"].get("agreement_score", 0.5)
            confidence = success_rate * 0.7 + agreement * 0.3
        else:
            confidence = success_rate * 0.8

        return min(max(confidence, 0.0), 1.0)

    # ─────────────────────────────────────────────────────────
    # (Optional) Resource guard
    # ─────────────────────────────────────────────────────────
    def _resource_guard_degrade_if_needed(self) -> bool:
        """
        Optional resource guard that can mark components "degraded" based on memory pressure.
        It’s a no-op unless resource_guard_enabled is True. Safe on systems without psutil.
        """
        if not self._cfg.resource_guard_enabled:
            return False

        try:
            import psutil  # local import to keep hard dep optional

            vm = psutil.virtual_memory()
            hard_cap_mb = float(self._cfg.max_memory_mb)
            soft_pct = float(self._cfg.memory_soft_limit_pct)

            # Trigger if above soft percentage OR over hard MB cap (if cap > 0)
            degrade = vm.percent >= (soft_pct * 100.0) or (
                (vm.used / (1024 * 1024)) >= hard_cap_mb > 0
            )

            if degrade:
                for comp in self._cfg.degrade_components:
                    setattr(self, f"_degraded_{comp}", True)
                self.trace(
                    f"Resource guard activated (mem%={vm.percent:.1f}, usedMB={vm.used/1024/1024:.0f}); "
                    f"degraded={self._cfg.degrade_components}",
                    level=TraceLevel.WARNING,
                )
                return True
            else:
                for comp in self._cfg.degrade_components:
                    setattr(self, f"_degraded_{comp}", False)
                return False

        except Exception as e:
            # If psutil is missing or anything else fails, just skip guard silently
            self.trace(f"Resource guard skipped: {e}", level=TraceLevel.DEBUG)
            return False
