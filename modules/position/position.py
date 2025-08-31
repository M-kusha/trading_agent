# -------------------------------------------------------------
# File: modules/position/position.py
# Enhanced with SmartInfoBus infrastructure integration
# -------------------------------------------------------------

from __future__ import annotations

import asyncio
import copy
import datetime
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Dict, List, Optional, Tuple, TypeVar, Union, cast, Deque

from modules.contracts import module_args
import numpy as np

# Optional live-broker connector
try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    mt5 = None  # Still works in back-test / unit-test mode

# New infrastructure imports
from modules.core.module_base import BaseModule, module
from modules.core.mixins import (
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
    SmartInfoBusTradingMixin,
)
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.monitoring.performance_tracker import PerformanceTracker
from modules.utils.info_bus import InfoBusManager
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from envs.config import TradingConfig
from modules.utils.audit_utils import RotatingLogger, format_operator_message, AuditConfiguration

# Module-level shared logger to avoid class-decorator type inference issues
_PM_SHARED_LOGGER: Optional[RotatingLogger] = None


# ─────────────────────────────────────────────────────────
# Helpers (decorator-safe awaiting for possibly-async wrappers)
# ─────────────────────────────────────────────────────────
T_co = TypeVar("T_co")


async def _maybe_await(x: Union[Awaitable[T_co], T_co]) -> T_co:
    """Await if coroutine, otherwise return value directly (helps with decorators that may wrap functions)."""
    if asyncio.iscoroutine(x):
        return cast(T_co, await x)
    return cast(T_co, x)


class PositionDecision(Enum):
    HOLD = "hold"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CLOSE = "close"
    EMERGENCY_CLOSE = "emergency_close"


@dataclass
class SignalContext:
    """Container for all signal inputs to position decisions"""

    instrument: str
    market_intensity: float = 0.0
    market_direction: int = 0  # -1, 0, 1
    volatility: float = 0.02
    trend_strength: float = 0.0
    momentum: float = 0.0
    volume_profile: float = 1.0
    correlation_penalty: float = 0.0

    # Market regime context
    regime: str = "normal"  # normal, volatile, trending, ranging
    liquidity_score: float = 1.0
    session: str = "unknown"

    # Portfolio context
    current_exposure: float = 0.0
    drawdown: float = 0.0
    balance: float = 10000.0

    # SmartInfoBus context
    step_idx: int = 0
    timestamp: str = ""


@dataclass
class PositionDecisionResult:
    """Result of position decision process"""

    decision: PositionDecision
    intensity: float  # 0.0 to 1.0
    size: float
    confidence: float
    rationale: Dict[str, Any]
    risk_factors: Dict[str, float]
    context: SignalContext


@module(**module_args(
    "PositionManager",
    description="Advanced position management with dynamic risk scaling and portfolio optimization",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class PositionManager(
    BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
):
    # Common attributes initialized during advanced systems setup; annotated for Pylance
    logger: RotatingLogger
    performance_tracker: PerformanceTracker
    circuit_breaker: Dict[str, Any]
    error_pinpointer: Any
    error_handler: Any
    english_explainer: Any
    system_utilities: Any
    smart_bus: Any
    # Sim execution attributes for analyzer
    simulate_execution: bool
    _sim_balance: float
    _sim_ticket_counter: int
    _trade_ledger: List[Dict[str, Any]]
    _recent_trades: Deque[Dict[str, Any]]

    # ─────────────────────────────────────────────────────────
    # Lifecycle / initialization
    # ─────────────────────────────────────────────────────────
    def __init__(
        self,
        config: Optional[TradingConfig | Dict[str, Any]] = None,
        instruments: Optional[List[str]] = None,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        # ensure instruments exist before BaseModule plumbing
        self.instruments = instruments or ["XAU/USD", "EUR/USD"]

        super().__init__()  # BaseModule may create .config (dict) etc.

        # Typed config alias (self.C) + dict mirror (self.config) to satisfy BaseModule/Pylance
        if isinstance(config, TradingConfig):
            self.C: TradingConfig = config
        elif isinstance(config, dict):
            self.C = TradingConfig(**config)
        else:
            self.C = TradingConfig()

        # Keep BaseModule's .config as a DICT (do NOT store TradingConfig there)
        self.config: Dict[str, Any] = dict(self.C.__dict__)
        self.default_max_pct = self.C.max_position_pct

        # NEW: toggle for legacy bus key probing (off by default to avoid BUS MISS spam)
        self.enable_legacy_bus_signal_probe: bool = bool(
            self.config.get("enable_legacy_bus_signal_probe", False)
        )

        # NEW: toggle to control whether PM should read instrument intensities from SmartInfoBus
        # When False (default), PM derives intensity locally from inputs/indicators only.
        # Set to True to re-enable consuming bus 'instrument_signals' and legacy fallbacks.
        self.use_bus_instrument_signals: bool = bool(
            self.config.get("use_bus_instrument_signals", False)
        )

        self._initialize_advanced_systems()
        self._initialize_genome_parameters(genome)
        self._initialize_position_state()
        self._initialize_position_tracking()

        # Start monitoring after all initialization is complete
        self._start_monitoring()

        self.env = None

        self.logger.info(
            format_operator_message(
                "🏦",
                "POSITION_MANAGER_INITIALIZED",
                instruments_count=len(self.instruments),
                initial_balance=f"€{self.C.initial_balance:,.0f}",
                max_position_pct=f"{self.C.max_position_pct:.1%}",
                details="Smart position management active",
            )
        )
        self._flush_logs()  # <-- add this

    def _flush_logs(self) -> None:
        """Safely flush shared logger buffers to disk (no-throw)."""
        try:
            lg = getattr(self, "logger", None)
            if lg and hasattr(lg, "flush"):
                lg.flush()
        except Exception:
            pass

    def _initialize(self, **kwargs: Any) -> None:
        """
        Real implementation of the required _initialize method for module system compatibility.
        Initializes advanced systems, genome parameters, position state, and tracking.
        Accepts optional config, instruments, genome, and env from kwargs for flexible hot-reload and orchestration.
        """
        cfg_in = kwargs.get("config", None)

        if isinstance(cfg_in, TradingConfig):
            self.C = cfg_in
        elif isinstance(cfg_in, dict):
            self.C = TradingConfig(**cfg_in)
        else:
            self.C = getattr(self, "C", TradingConfig())

        # Keep the BaseModule's .config as a DICT mirror
        self.config = dict(self.C.__dict__)
        self.default_max_pct = self.C.max_position_pct
        self.enable_legacy_bus_signal_probe = bool(
            self.config.get("enable_legacy_bus_signal_probe", False)
        )

        # Respect runtime-configurable behavior for reading bus intensities
        self.use_bus_instrument_signals = bool(
            self.config.get("use_bus_instrument_signals", False)
        )

        instruments = kwargs.get("instruments", None)
        genome = kwargs.get("genome", None)
        env = kwargs.get("env", None)

        if not hasattr(self, "instruments") or self.instruments is None:
            self.instruments = instruments or ["XAU/USD", "EUR/USD"]
        elif instruments is not None:
            self.instruments = instruments

        if not hasattr(self, "genome") or self.genome is None:
            self.genome = genome or {}
        elif genome is not None:
            self.genome = genome

        if env is not None:
            self.env = env

        self._initialize_advanced_systems()
        self._initialize_genome_parameters(self.genome)
        self._initialize_position_state()
        self._initialize_position_tracking()
        self.logger.info(
            format_operator_message(
                "[INIT]",
                "POSITION_MANAGER_REINITIALIZED",
                instruments_count=len(self.instruments),
                initial_balance=f"€{self.C.initial_balance:,.0f}",
                max_position_pct=f"{self.C.max_position_pct:.1%}",
                details="PositionManager _initialize called",
            )
        )

    def _initialize_advanced_systems(self) -> None:
        """Initialize advanced systems for position management"""
        self.smart_bus = InfoBusManager.get_instance()

        # ONE shared logger across instances → one header per process, single file, immediate writes
        global _PM_SHARED_LOGGER
        if _PM_SHARED_LOGGER is None:
            cfg = AuditConfiguration(
                log_level="DEBUG",          # use "DEBUG" while tuning if you want more chatter
                async_logging=False,       # <- key: write immediately, no background buffer
                flush_interval_seconds=1,
                buffer_size=200,
                info_bus_integration=True,
                publish_to_bus=True,
            )
            _PM_SHARED_LOGGER = RotatingLogger(
                name="PositionManager",
                log_path="logs/position/position.log",   # <- single target file for everything
                max_lines=1_000_000,                     # ignored for direct path rotation, but harmless
                operator_mode=True,
                plain_english=True,                      # gives you the "[LOG] ..." style prefix
                info_bus_aware=True,
                config=cfg,
            )

        self.logger = _PM_SHARED_LOGGER

        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PositionManager", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Circuit breaker for position operations
        self.circuit_breaker = {
            "failures": 0,
            "last_failure": 0,
            "state": "CLOSED",
            "threshold": self.C.position_circuit_breaker_threshold,
        }

    def _start_monitoring(self) -> None:
        """Start background monitoring for position management (idempotent)."""
        if getattr(self, "_monitoring_active", False):
            return  # already running

        def monitoring_loop() -> None:
            try:
                while getattr(self, "_monitoring_active", True):
                    try:
                        self._update_position_health()
                    except Exception as inner_e:  # noqa: BLE001
                        self.logger.warning(f"Position monitoring error: {inner_e}")
                    time.sleep(30)
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Monitoring loop failure: {e}")

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]) -> None:
        """Initialize genome-based parameters"""
        if genome:
            # Override config with genome values
            for key, value in genome.items():
                if hasattr(self.C, key):
                    setattr(self.C, key, value)
            # keep dict mirror in sync
            self.config.update(self.C.__dict__)

        # Store genome for evolution
        self.genome = genome or {}

        # Set derived parameters
        self.risk_multiplier = float(self.genome.get("risk_multiplier", 1.0))
        self.correlation_threshold = float(self.genome.get("correlation_threshold", 0.7))

        # Dynamic parameters (reset on each episode)
        self.default_max_pct = self.C.max_position_pct

    def _initialize_position_state(self) -> None:
        """Initialize position management state"""
        # Core position state
        self.consecutive_losses = 0
        self.open_positions: Dict[str, Dict[str, Any]] = {}

        # Enhanced tracking
        self._decision_history = deque(maxlen=100)
        self._portfolio_health_history = deque(maxlen=50)
        self._exposure_history = deque(maxlen=100)
        self._performance_analytics = defaultdict(list)

        # Decision tracking
        self.last_decisions: Dict[str, PositionDecisionResult] = {}
        self.position_confidence: Dict[str, float] = {}
        self.signal_history: Dict[str, List[float]] = {inst: [] for inst in self.instruments}

        # Performance metrics
        self._portfolio_health_score = 1.0
        self._total_exposure_ratio = 0.0
        self._decision_quality_score = 0.5
        self._risk_management_score = 1.0

        # Adaptive parameters
        self._adaptive_params = {
            "dynamic_max_pct": self.C.max_position_pct,
            "signal_sensitivity": 1.0,
            "risk_tolerance": 1.0,
            "confidence_threshold": 0.5,
        }

        # Live trading state
        self._forced_action = None
        self._forced_conf = None
        self._last_sync_time = None

        # Simulation execution state (for backtests/training when no broker)
        # Enabled via config key 'simulate_execution' (default True when not in live mode)
        self.simulate_execution = bool(self.config.get("simulate_execution", True))
        self._sim_balance = float(getattr(self.C, "initial_balance", 10000.0))
        self._sim_ticket_counter = 1_000_000
        self._trade_ledger = []
        self._recent_trades = deque(maxlen=200)

    def _initialize_position_tracking(self) -> None:
        """Initialize position-specific tracking"""
        # Position metadata tracking
        self._position_metadata: Dict[str, Dict[str, Any]] = {}

        # Performance tracking per position
        self._position_performance: Dict[str, Dict[str, Any]] = {}

        # Exit rule tracking
        self._exit_signals: Dict[str, List[Dict[str, Any]]] = {}

    # ─────────────────────────────────────────────────────────
    # SmartBus publishing (balances, trades, exec)
    # ─────────────────────────────────────────────────────────
    def _publish_bus_feeds(
        self,
        balance: float,
        equity: float,
        current_pnl: float,
        trades: Optional[List[Dict[str, Any]]] = None,
        execution_data: Optional[Dict[str, Any]] = None,
        order_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write accounting/execution feeds so downstream consumers have a provider."""
        trades = trades if isinstance(trades, list) else []
        execution_data = execution_data if isinstance(execution_data, dict) else {}
        order_data = order_data if isinstance(order_data, dict) else {}

        self.smart_bus.set(
            "trades", trades, module="PositionManager", thesis=f"Trade ledger snapshot: {len(trades)} items"
        )
        self.smart_bus.set(
            "recent_trades", trades[-20:], module="PositionManager", thesis="Most recent trades"
        )
        self.smart_bus.set(
            "balance", float(balance), module="PositionManager", thesis="Account balance for drawdown/rescue logic"
        )
        self.smart_bus.set(
            "equity", float(equity), module="PositionManager", thesis="Account equity for rescue logic"
        )
        self.smart_bus.set(
            "current_pnl", float(current_pnl), module="PositionManager", thesis="Current PnL for bias/audit modules"
        )
        self.smart_bus.set(
            "execution_data", execution_data, module="PositionManager", thesis="Execution diagnostics (per step)"
        )
        self.smart_bus.set(
            "order_data", order_data, module="PositionManager", thesis="Orders placed/cancelled (per step)"
        )

    def _get_liquidity(self, instrument: str) -> float:
        """
        Prefer canonical 'liquidity_score'. If absent, adapt from LiquidityHeatmapLayer's
        'liquidity_capabilities' or legacy 'market_liquidity'.
        """
        # Canonical single score
        liq_global = self.smart_bus.get("liquidity_score", "PositionManager")
        if isinstance(liq_global, (int, float)):
            return float(liq_global)

        # Adapter from LiquidityHeatmapLayer
        liq_caps = self.smart_bus.get("liquidity_capabilities", "PositionManager")
        if isinstance(liq_caps, dict):
            # Common aggregate keys
            for k in ("score", "aggregate_score", "global_score"):
                v = liq_caps.get(k, None)
                if isinstance(v, (int, float)):
                    return float(v)

            # Per-instrument subnode
            variants = {
                instrument,
                instrument.replace("/", ""),
                instrument.replace("/", "_"),
                instrument.upper(),
                instrument.replace("/", "").upper(),
            }
            for v in variants:
                node = liq_caps.get(v, None)
                if node is None:
                    continue
                if isinstance(node, (int, float)):
                    return float(node)
                if isinstance(node, dict):
                    sv = node.get("score", None)
                    if isinstance(sv, (int, float)):
                        return float(sv)

        # Legacy per-instrument map
        market_liquidity = self.smart_bus.get("market_liquidity", "PositionManager")
        if isinstance(market_liquidity, dict):
            try:
                return float(market_liquidity.get(instrument, 1.0))
            except Exception:  # noqa: BLE001
                pass

        return 1.0

    # ─────────────────────────────────────────────────────────
    # NEW: feature-derived intensity fallback (no bus signals needed)
    # ─────────────────────────────────────────────────────────
    def _derive_intensity(self, inst_dict: Dict[str, Any]) -> Optional[float]:
        """
        Derive a signed intensity in [-1, 1] from indicators:
        - trend_strength (SMA20-50 / |SMA50|)
        - momentum (MACD)
        - rsi deviation from 50
        Scaled by volatility to avoid blow-ups.
        """
        try:
            trend = float(inst_dict.get("trend_strength", 0.0))
            mom = float(inst_dict.get("momentum", 0.0))
            rsi = float(inst_dict.get("rsi", 50.0))
            vol = float(inst_dict.get("volatility", self.C.min_volatility))
            vol = max(vol, 1e-6)

            trend_term = float(np.tanh(trend * 3.0))
            macd_term = float(np.tanh(mom / (vol * 50.0)))  # normalize MACD by vol
            rsi_term = float(np.clip((rsi - 50.0) / 50.0, -1.0, 1.0))

            intensity = 0.5 * trend_term + 0.3 * macd_term + 0.2 * rsi_term
            return float(np.clip(intensity, -1.0, 1.0))
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────
    # Main processing entry
    # ─────────────────────────────────────────────────────────
    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """Main processing method — must output all contract 'provides' keys and a thesis."""
        start_time = time.time()
        try:
            # 1) Build market snapshot from inputs first, then overlay bus snapshot via deep merge
            market_from_inputs = self._extract_market_data_from_inputs(inputs)
            bus_snapshot = self._extract_market_data_from_smartbus() or {}
            market_data: Dict[str, Any] = self._merge_market_maps(bus_snapshot, market_from_inputs)

            # If still empty, fallback
            if not any(isinstance(v, dict) and v for v in market_data.values()):
                fb = self._create_fallback_response("No market data available")
                pos_health = {
                    "portfolio_health": float(self._portfolio_health_score),
                    "exposure_ratio": float(self._total_exposure_ratio),
                    "risk_management_score": float(self._risk_management_score),
                    "consecutive_losses": int(self.consecutive_losses),
                }
                # publish empty-but-valid bus feeds so consumers have providers
                self._publish_bus_feeds(
                    balance=float(getattr(self.C, "initial_balance", 0.0)),
                    equity=float(getattr(self.C, "initial_balance", 0.0)),
                    current_pnl=0.0,
                    trades=[],
                    execution_data={},
                    order_data={},
                )
                return {
                    "position_decisions": {},
                    "portfolio_state": {
                        "health_score": fb["portfolio_health"],
                        "exposure_ratio": fb["exposure_ratio"],
                        "open_positions": len(self.open_positions),
                        "decision_quality": self._decision_quality_score,
                    },
                    "position_health": pos_health,
                    "position_analysis": {"note": fb["thesis"], "thesis": fb["thesis"]},
                    "positions": copy.deepcopy(self.open_positions),
                    "pending_orders": [],
                    "position_data": {},
                    "trades": [],
                    "recent_trades": [],
                    "current_pnl": 0.0,
                    "balance": float(getattr(self.C, "initial_balance", 0.0)),
                    "equity": float(getattr(self.C, "initial_balance", 0.0)),
                    "current_positions": copy.deepcopy(self.open_positions),
                    "execution_data": {},
                    "order_data": {},
                    "_thesis": fb["thesis"],           # ← required by module system
                    "thesis": fb["thesis"],            # ← optional (kept for UIs)
                    "processing_time_ms": 0.0,
                }

            # 2) Decisions (decorator may return coroutine → normalize)
            decisions = await _maybe_await(self.process_market_signals(market_data))

            # 3) Apply simulated execution (optional, before publishing so positions reflect fills)
            sim_exec = self._apply_simulated_execution(decisions, market_data)

            # 4) Publish to SmartInfoBus (decisions + current_positions)
            await self._update_smartbus_with_decisions(decisions)

            # 5) Thesis
            thesis = await self._generate_position_thesis(market_data, decisions)

            # 6) Risk roll-up and balances
            aggregated_risks: Dict[str, float] = {}
            for inst, dr in decisions.items():
                for k, v in (dr.risk_factors or {}).items():
                    aggregated_risks[k] = max(aggregated_risks.get(k, 0.0), float(v))

            # Use simulated accounting if available; otherwise fall back to bus/initials
            if sim_exec:
                try:
                    balance = float(sim_exec.get("balance", self._sim_balance))
                except Exception:
                    balance = float(self._sim_balance)
                try:
                    equity = float(sim_exec.get("equity", balance))
                except Exception:
                    equity = float(balance)
                try:
                    current_pnl = float(sim_exec.get("current_pnl", 0.0))
                except Exception:
                    current_pnl = 0.0
                step_trades_val = sim_exec.get("trades", []) or []
                step_trades: List[Dict[str, Any]] = step_trades_val if isinstance(step_trades_val, list) else []
            else:
                balance = float(self.C.initial_balance)
                bus_port = self.smart_bus.get("portfolio_metrics", "PositionManager")
                if isinstance(bus_port, dict):
                    balance = float(bus_port.get("balance", balance))
                current_pnl = 0.0
                equity = balance + current_pnl
                step_trades = []

            # 7) Publish required feeds so downstream consumers have providers
            self._publish_bus_feeds(
                balance=balance,
                equity=equity,
                current_pnl=current_pnl,
                trades=step_trades,
                execution_data={},
                order_data={},
            )

            # 8) Contract output
            position_decisions = {
                inst: {
                    "decision": dr.decision.value,
                    "intensity": float(dr.intensity),
                    "size": float(dr.size),
                    "confidence": float(dr.confidence),
                    "rationale": dr.rationale,
                    "risk_factors": dr.risk_factors,
                }
                for inst, dr in decisions.items()
            }

            processing_time = (time.time() - start_time) * 1000.0
            self.performance_tracker.record_metric("PositionManager", "process", processing_time, True)

            pos_health = {
                "portfolio_health": float(self._portfolio_health_score),
                "exposure_ratio": float(self._total_exposure_ratio),
                "risk_management_score": float(self._risk_management_score),
                "consecutive_losses": int(self.consecutive_losses),
            }
            return {
                "position_decisions": position_decisions,
                "portfolio_state": {
                    "health_score": float(self._portfolio_health_score),
                    "exposure_ratio": float(self._total_exposure_ratio),
                    "open_positions": int(len(self.open_positions)),
                    "decision_quality": float(self._decision_quality_score),
                },
                "position_health": pos_health,
                "position_analysis": {
                    "instruments": list(self.instruments),
                    "decisions_count": len(decisions),
                    "exposure_ratio": float(self._total_exposure_ratio),
                    "risk_management_score": float(self._risk_management_score),
                    "thesis": thesis,
                },
                "positions": copy.deepcopy(self.open_positions),
                "pending_orders": [],
                "position_data": copy.deepcopy(self.open_positions),
                "trades": step_trades,
                "recent_trades": step_trades[-20:],
                "current_pnl": float(current_pnl),
                "balance": float(balance),
                "equity": float(equity),
                "current_positions": copy.deepcopy(self.open_positions),
                "execution_data": {},
                "order_data": {},
                "_thesis": thesis,                    # ← required by module system
                "thesis": thesis,                     # ← optional (kept for UIs)
                "processing_time_ms": processing_time,
                "risk_metrics": aggregated_risks,     # optional diagnostics
            }

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Position processing failed: {e}")
            fb = self._create_fallback_response(f"Processing error: {str(e)}")
            pos_health = {
                "portfolio_health": float(self._portfolio_health_score),
                "exposure_ratio": float(self._total_exposure_ratio),
                "risk_management_score": float(self._risk_management_score),
                "consecutive_losses": int(self.consecutive_losses),
            }
            # publish empty-but-valid bus feeds on error
            self._publish_bus_feeds(
                balance=float(getattr(self.C, "initial_balance", 0.0)),
                equity=float(getattr(self.C, "initial_balance", 0.0)),
                current_pnl=0.0,
                trades=[],
                execution_data={},
                order_data={},
            )
            return {
                "position_decisions": {},
                "portfolio_state": {
                    "health_score": fb["portfolio_health"],
                    "exposure_ratio": fb["exposure_ratio"],
                    "open_positions": len(self.open_positions),
                    "decision_quality": self._decision_quality_score,
                },
                "position_health": pos_health,
                "position_analysis": {"note": fb["thesis"], "thesis": fb["thesis"]},
                "positions": copy.deepcopy(self.open_positions),
                "pending_orders": [],
                "position_data": {},
                "trades": [],
                "recent_trades": [],
                "current_pnl": 0.0,
                "balance": float(getattr(self.C, "initial_balance", 0.0)),
                "equity": float(getattr(self.C, "initial_balance", 0.0)),
                "current_positions": copy.deepcopy(self.open_positions),
                "execution_data": {},
                "order_data": {},
                "_thesis": fb["thesis"],              # ← required by module system
                "thesis": fb["thesis"],               # ← optional (kept for UIs)
                "processing_time_ms": 0.0,
            }

    # ─────────────────────────────────────────────────────────
    # Input extraction (from orchestrator inputs + SmartBus)
    # ─────────────────────────────────────────────────────────
    def _extract_market_data_from_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build per-instrument snapshot from orchestrator inputs.
        Signals are a fallback if bus didn't provide them.
        """
        out: Dict[str, Any] = {}

        market_data = inputs.get("market_data") or {}
        price_data = inputs.get("price_data") or {}
        tech = inputs.get("indicators") or inputs.get("technical_indicators") or {}
        vol = inputs.get("volatility_data") or {}
        simple_prices = inputs.get("prices") or {}

        regime = inputs.get("market_regime") or inputs.get("market_context", {}).get("volatility_regime")
        session = inputs.get("trading_session") or inputs.get("market_context", {}).get("session")

        # Accept many signal keys
        signals_map = (
            inputs.get("instrument_signals")
            or inputs.get("signals")
            or inputs.get("alpha_signals")
            or inputs.get("action_signals")
            or inputs.get("trading_signals")
            or {}
        )

        def variants(inst: str):
            core = inst.replace("/", "").replace("_", "")
            return {
                inst, inst.replace("/", ""), inst.replace("/", "_"),
                inst.upper(), inst.lower(), core.upper(), core.lower()
            }

        def pick(src: Dict[str, Any], inst: str):
            if not isinstance(src, dict):
                return None
            for k in variants(inst):
                if k in src:
                    return src[k]
            return None

        def extract_intensity(sig_payload: Any) -> Optional[float]:
            if isinstance(sig_payload, (int, float)):
                return float(sig_payload)
            if isinstance(sig_payload, dict):
                for key in ("intensity", "signal", "score", "value"):
                    v = sig_payload.get(key)
                    if isinstance(v, (int, float)):
                        return float(v)
            return None

        for inst in self.instruments:
            inst_dict: Dict[str, Any] = {}

            md = pick(market_data, inst)
            if isinstance(md, dict):
                inst_dict["current_price"] = float(md.get("close", md.get("price", md.get("bid", 0.0))))
                inst_dict["volatility"] = float(md.get("atr", md.get("volatility", self.C.min_volatility)))

            pd = pick(price_data, inst)
            if isinstance(pd, dict):
                last = pd.get("last", pd.get("close"))
                if last is not None:
                    inst_dict["current_price"] = float(last)

            ti = pick(tech, inst)
            if isinstance(ti, dict):
                sma20 = float(ti.get("sma_20", 0.0))
                sma50 = float(ti.get("sma_50", 0.0)) or 1.0
                inst_dict["trend_strength"] = (sma20 - sma50) / (abs(sma50) or 1.0)
                inst_dict["momentum"] = float(ti.get("macd", 0.0))
                inst_dict["rsi"] = float(ti.get("rsi", 50.0))

            vd = pick(vol, inst)
            if isinstance(vd, dict):
                inst_dict["volatility"] = float(vd.get("atr", vd.get("volatility", inst_dict.get("volatility", self.C.min_volatility))))

            sp = pick(simple_prices, inst)
            if sp is not None and "current_price" not in inst_dict:
                if isinstance(sp, (int, float)):
                    inst_dict["current_price"] = float(sp)
                elif isinstance(sp, dict):
                    for k in ("last", "close", "price", "bid", "ask"):
                        v = sp.get(k)
                        if isinstance(v, (int, float)):
                            inst_dict["current_price"] = float(v)
                            break

            # Signals from inputs (fallback only)
            sig = pick(signals_map, inst)
            inten = extract_intensity(sig)
            if inten is not None:
                inst_dict["intensity"] = float(max(-1.0, min(1.0, inten)))
                inst_dict["intensity_source"] = "inputs"

            # NEW: derive intensity from indicators if still missing
            if "intensity" not in inst_dict:
                di = self._derive_intensity(inst_dict)
                if di is not None:
                    inst_dict["intensity"] = di
                    inst_dict["intensity_source"] = "derived"

            if session:
                inst_dict["session"] = session

            out[inst] = inst_dict

        if regime:
            out["market_regime"] = regime

        return out

    def _extract_market_data_from_smartbus(self) -> Optional[Dict[str, Any]]:
        """
        Build a per-instrument snapshot from SmartInfoBus.
        Prefers:
        - instrument_signals[intensity]  (Strategy/Voting output)
        - price_data / prices
        - indicators / technical_indicators
        - volatility_data
        - market_context / market_conditions (regime, session)

        Robust against None/empty values and falls back to the
        position_decision_{instrument} nodes we publish.
        """
        out: Dict[str, Any] = {}

        # Canonical aggregates
        price_map = self.smart_bus.get('price_data', 'PositionManager') or {}
        simple_prices = self.smart_bus.get('prices', 'PositionManager') or {}
        tech_map = (self.smart_bus.get('indicators', 'PositionManager')
                    or self.smart_bus.get('technical_indicators', 'PositionManager')
                    or {})
        vol_map = self.smart_bus.get('volatility_data', 'PositionManager') or {}

        market_context = self.smart_bus.get('market_context', 'PositionManager') or {}
        market_conditions = self.smart_bus.get('market_conditions', 'PositionManager') or {}
        regime = market_context.get('volatility_regime', market_conditions.get('volatility_regime'))
        session = market_context.get('session')

        # Voting/arbiter outputs (gated): optionally consume bus-provided intensities
        bus_signals = {}
        if self.use_bus_instrument_signals:
            bus_signals = self.smart_bus.get('instrument_signals', 'PositionManager') or {}

        def variants(inst: str) -> List[str]:
            core = inst.replace("/", "").replace("_", "")
            return [inst, inst.replace("/", ""), inst.replace("/", "_"),
                    inst.upper(), inst.lower(), core.upper(), core.lower()]

        def pick(src: Dict[str, Any], inst: str):
            if not isinstance(src, dict):
                return None
            for k in variants(inst):
                if k in src:
                    return src[k]
            return None

        have_any = False
        for inst in self.instruments:
            inst_dict: Dict[str, Any] = {}

            # price_data / prices
            pd = pick(price_map, inst)
            if isinstance(pd, dict):
                last = pd.get("last", pd.get("close"))
                if isinstance(last, (int, float)):
                    inst_dict["current_price"] = float(last)
            if "current_price" not in inst_dict:
                sp = pick(simple_prices, inst)
                if isinstance(sp, (int, float)):
                    inst_dict["current_price"] = float(sp)
                elif isinstance(sp, dict):
                    for k in ("last", "close", "price", "bid", "ask"):
                        v = sp.get(k)
                        if isinstance(v, (int, float)):
                            inst_dict["current_price"] = float(v)
                            break

            # indicators
            ti = pick(tech_map, inst)
            if isinstance(ti, dict):
                sma20 = float(ti.get("sma_20", 0.0))
                sma50 = float(ti.get("sma_50", 0.0)) or 1.0
                inst_dict["trend_strength"] = (sma20 - sma50) / (abs(sma50) or 1.0)
                inst_dict["momentum"] = float(ti.get("macd", 0.0))
                inst_dict["rsi"] = float(ti.get("rsi", 50.0))

            # volatility
            vd = pick(vol_map, inst)
            if isinstance(vd, dict):
                vol_val = vd.get("atr", vd.get("volatility", self.C.min_volatility))
                if isinstance(vol_val, (int, float)):
                    inst_dict["volatility"] = float(vol_val)

            # INTENSITY — optionally from bus first
            if self.use_bus_instrument_signals:
                sig = pick(bus_signals, inst)
                if isinstance(sig, dict):
                    iv = sig.get("intensity")
                    if isinstance(iv, (int, float)):
                        inst_dict["intensity"] = float(np.clip(iv, -1.0, 1.0))
                        inst_dict["intensity_source"] = "bus"
                elif isinstance(sig, (int, float)):
                    inst_dict["intensity"] = float(np.clip(sig, -1.0, 1.0))
                    inst_dict["intensity_source"] = "bus"

            # LEGACY PROBE (optional) — single canonical key only
            if (
                self.use_bus_instrument_signals
                and "intensity" not in inst_dict
                and self.enable_legacy_bus_signal_probe
            ):
                legacy_key = f"signal_{inst.replace('/', '').upper()}"
                entry = self.smart_bus.get(legacy_key, "PositionManager")
                if isinstance(entry, dict):
                    iv = entry.get("intensity")
                    if isinstance(iv, (int, float)):
                        inst_dict["intensity"] = float(np.clip(iv, -1.0, 1.0))
                        inst_dict["intensity_source"] = "bus_legacy"

            # FALLBACK: use our own published position_decision_{instrument}
            if self.use_bus_instrument_signals and "intensity" not in inst_dict:
                pd_node = self.smart_bus.get(f"position_decision_{inst}", "PositionManager")
                if isinstance(pd_node, dict):
                    iv = pd_node.get("intensity")
                    if isinstance(iv, (int, float)):
                        inst_dict["intensity"] = float(np.clip(iv, -1.0, 1.0))
                        inst_dict["intensity_source"] = "bus_decision"
                    else:
                        # infer sign from decision if provided
                        dec = str(pd_node.get("decision", "")).lower()
                        if dec in ("open_long", "scale_up"):
                            inst_dict["intensity"] = 0.5
                            inst_dict["intensity_source"] = "bus_decision_inferred"
                        elif dec in ("open_short", "scale_down", "close", "emergency_close"):
                            inst_dict["intensity"] = -0.5
                            inst_dict["intensity_source"] = "bus_decision_inferred"

            # NEW: derive intensity from indicators if still missing
            if "intensity" not in inst_dict:
                di = self._derive_intensity(inst_dict)
                if di is not None:
                    inst_dict["intensity"] = di
                    inst_dict["intensity_source"] = "derived"

            if session:
                inst_dict["session"] = session

            out[inst] = inst_dict
            have_any = have_any or bool(inst_dict)

        if regime:
            out["market_regime"] = regime

        return out if have_any else None


    def _merge_market_snapshots(self, bus_snap: Dict[str, Any], in_snap: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge per-instrument dicts, preferring bus for 'intensity' (and session/regime).
        Other fields: input can override (often more recent for prices).
        """
        merged: Dict[str, Any] = {}

        # top-level regime: prefer bus
        if isinstance(bus_snap, dict) and "market_regime" in bus_snap:
            merged["market_regime"] = bus_snap["market_regime"]
        elif isinstance(in_snap, dict) and "market_regime" in in_snap:
            merged["market_regime"] = in_snap["market_regime"]

        for inst in self.instruments:
            b = (bus_snap or {}).get(inst, {}) or {}
            i = (in_snap or {}).get(inst, {}) or {}
            out = dict(i)  # start with inputs

            # keep bus intensity if present
            if "intensity" in b:
                out["intensity"] = b["intensity"]
                if "intensity_source" in b:
                    out["intensity_source"] = b["intensity_source"]

            # prefer bus session if present
            if "session" in b:
                out["session"] = b["session"]

            # ensure we don’t lose current_price from either side
            if "current_price" not in out and "current_price" in b:
                out["current_price"] = b["current_price"]

            merged[inst] = out

        return merged

    def _merge_market_maps(self, bus_map: Dict[str, Any], in_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Non-destructive merge of bus snapshot with inputs.
        - Preserves bus per-instrument fields.
        - Overlays only non-empty input fields for each instrument.
        - Prefer **numeric** bus intensity when present (ignore None).
        - Carries over non-instrument global keys from inputs when present.
        """
        out: Dict[str, Any] = dict(bus_map) if isinstance(bus_map, dict) else {}

        # Per-instrument deep merge
        for inst in self.instruments:
            b = bus_map.get(inst, {}) if isinstance(bus_map, dict) else {}
            i = in_map.get(inst, {}) if isinstance(in_map, dict) else {}

            merged = dict(b)
            if isinstance(i, dict) and i:
                merged.update(i)

            # Explicitly prefer bus intensity if it is numeric
            if isinstance(b, dict) and isinstance(b.get("intensity", None), (int, float)):
                merged["intensity"] = b["intensity"]
                if "intensity_source" in b:
                    merged["intensity_source"] = b["intensity_source"]
            else:
                # Guard against None coming from inputs
                iv = merged.get("intensity", None)
                if not isinstance(iv, (int, float)):
                    merged.pop("intensity", None)
                    merged.pop("intensity_source", None)

            # Prefer bus session if present
            if isinstance(b, dict) and "session" in b:
                merged["session"] = b["session"]

            if merged:
                out[inst] = merged

        # Global (non-instrument) keys from inputs (e.g., market_regime)
        for k, v in (in_map or {}).items():
            if k in self.instruments:
                continue
            if isinstance(v, dict):
                if v:  # only non-empty dicts
                    out[k] = v
            elif v is not None:
                out[k] = v

        # Prefer bus market_regime if present
        if isinstance(bus_map, dict) and "market_regime" in bus_map:
            out["market_regime"] = bus_map["market_regime"]

        return out


    async def _update_smartbus_with_decisions(self, decisions: Dict[str, PositionDecisionResult]) -> None:
        """Update SmartInfoBus with position decisions + publish canonical instrument_signals."""
        instrument_signals: Dict[str, Any] = {}

        for instrument, decision in decisions.items():
            # Publish per-instrument decision node
            self.smart_bus.set(
                f"position_decision_{instrument}",
                {
                    "decision": decision.decision.value,
                    "intensity": float(decision.intensity),
                    "size": float(decision.size),
                    "confidence": float(decision.confidence),
                    "risk_factors": decision.risk_factors,
                },
                module="PositionManager",
                thesis=f"Position decision for {instrument}: {decision.decision.value} with {decision.confidence:.2f} confidence",
            )

            # Build canonical instrument signal (signed intensity)
            try:
                signed = self._map_decision_to_intensity(decision)
            except Exception:
                signed = 0.0

            instrument_signals[instrument] = {
                "intensity": float(np.clip(signed, -1.0, 1.0)),
                "decision": decision.decision.value,
                "confidence": float(decision.confidence),
            }

        # Set portfolio state
        self.smart_bus.set(
            "portfolio_state",
            {
                "health_score": float(self._portfolio_health_score),
                "exposure_ratio": float(self._total_exposure_ratio),
                "open_positions": int(len(self.open_positions)),
                "decision_quality": float(self._decision_quality_score),
            },
            module="PositionManager",
            thesis="Current portfolio health and exposure metrics",
        )

        # Publish canonical instrument_signals map
        self.smart_bus.set(
            "instrument_signals",
            instrument_signals,
            module="PositionManager",
            thesis="Canonical instrument signals from PositionManager",
        )

        # Also publish the current open positions for downstream consumers
        try:
            self.smart_bus.set(
                "current_positions",
                copy.deepcopy(self.open_positions),
                module="PositionManager",
                thesis="Snapshot of current open positions",
            )
        except Exception:
            self.smart_bus.set(
                "current_positions",
                self.open_positions,
                module="PositionManager",
                thesis="Snapshot of current open positions",
            )

    async def _generate_position_thesis(
        self, market_data: Dict[str, Any], decisions: Dict[str, PositionDecisionResult]
    ) -> str:
        """Generate concise thesis for operator UIs."""
        return (
            f"PortfolioHealth={self._portfolio_health_score:.2f} | "
            f"Exposure={self._total_exposure_ratio:.1%} | "
            f"Decisions={len(decisions)} | "
            f"RiskMgmt={self._risk_management_score:.2f}"
        )

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error conditions"""
        return {
            "decisions": {},
            "portfolio_health": self._portfolio_health_score,
            "exposure_ratio": self._total_exposure_ratio,
            "error": reason,
            "processing_time_ms": 0,
            "thesis": f"Position manager fallback: {reason}",
        }

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()

        # Reset position manager state
        self.C.max_position_pct = self.default_max_pct
        self.config.update(self.C.__dict__)  # keep dict mirror fresh
        self.consecutive_losses = 0
        self.open_positions.clear()
        self.last_decisions.clear()
        self.position_confidence.clear()

        # Reset tracking
        self._decision_history.clear()
        self._portfolio_health_history.clear()
        self._exposure_history.clear()
        self._performance_analytics.clear()

        # Reset signal history
        for inst in self.instruments:
            self.signal_history[inst].clear()

        # Reset position tracking
        self._position_metadata.clear()
        self._position_performance.clear()
        self._exit_signals.clear()

        # Reset forced values
        self._forced_action = None
        self._forced_conf = None

        # Reset performance metrics
        self._portfolio_health_score = 1.0
        self._total_exposure_ratio = 0.0
        self._decision_quality_score = 0.5
        self._risk_management_score = 1.0

        # Reset adaptive parameters
        self._adaptive_params = {
            "dynamic_max_pct": self.C.max_position_pct,
            "signal_sensitivity": 1.0,
            "risk_tolerance": 1.0,
            "confidence_threshold": 0.5,
        }

        # Ensure minimum allocation capability
        if self.C.max_position_pct < 1e-5:
            self.logger.warning(
                format_operator_message(
                    "[WARN]",
                    "MAX_POSITION_PCT_TOO_LOW",
                    current=f"{self.C.max_position_pct:.6f}",
                    default=f"{self.default_max_pct:.4f}",
                    action="Restoring to default",
                )
            )
            self.C.max_position_pct = self.default_max_pct
            self.config.update(self.C.__dict__)

    def set_env(self, env: Any) -> None:
        """Set environment reference"""
        self.env = env

    # ─────────────────────────────────────────────────────────
    # Decision pipeline
    # ─────────────────────────────────────────────────────────
    @create_error_handler("process_market_signals")
    def process_market_signals(self, market_data: Dict[str, Any]) -> Dict[str, PositionDecisionResult]:
        """
        Main entry point for processing market signals into position decisions.

        This method implements the hierarchical decision making:
        1. Strategic: Assess market regime and portfolio health
        2. Tactical: Make instrument-specific decisions
        3. Execution: Apply risk management and sizing
        """
        decisions: Dict[str, PositionDecisionResult] = {}

        # 1. Strategic Layer: Portfolio-level assessment
        portfolio_health = self._assess_portfolio_health()
        market_regime = self._assess_market_regime(market_data)

        self.logger.info(
            format_operator_message(
                "[STATS]",
                "PORTFOLIO_ASSESSMENT",
                health_score=f"{portfolio_health['overall_health']:.3f}",
                market_regime=market_regime,
                exposure_ratio=f"{portfolio_health['exposure_ratio']:.2%}",
                consecutive_losses=self.consecutive_losses,
            )
        )

        # 2. Tactical Layer: Per-instrument decisions
        for instrument in self.instruments:
            # Extract signal context for this instrument
            signal_context = self._extract_signal_context(instrument, market_data, portfolio_health)

            # Make position decision
            decision_result = self._make_position_decision(signal_context)

            # Store decision and update tracking
            decisions[instrument] = decision_result
            self.last_decisions[instrument] = decision_result

            # Update signal history
            self.signal_history[instrument].append(signal_context.market_intensity)
            if len(self.signal_history[instrument]) > 50:  # Keep last 50 signals
                self.signal_history[instrument].pop(0)

            if decision_result.decision != PositionDecision.HOLD:
                self.logger.info(
                    format_operator_message(
                        "[MONEY]",
                        "POSITION_DECISION",
                        instrument=instrument,
                        decision=decision_result.decision.value,
                        intensity=f"{decision_result.intensity:.3f}",
                        size=f"€{decision_result.size:.0f}",
                        confidence=f"{decision_result.confidence:.3f}",
                        rationale=decision_result.rationale.get("stage", "unknown"),
                    )
                )
        self._flush_logs()  # ensure decision lines hit disk now
        return decisions

    def _assess_portfolio_health(self) -> Dict[str, float]:
        """Assess overall portfolio health metrics (canonical-aware risk & balances)."""

        # Balance / drawdown (prefers market_state → portfolio_metrics → env)
        balance = float(self.C.initial_balance)
        drawdown = 0.0

        market_state = self.smart_bus.get("market_state", "PositionManager")  # optional upstream
        if isinstance(market_state, dict):
            b = market_state.get("balance", balance)
            balance = float(b if b is not None else balance)
            d = market_state.get("drawdown", 0.0)
            drawdown = float(d if d is not None else 0.0)
        else:
            portfolio_metrics = self.smart_bus.get("portfolio_metrics", "PositionManager")
            if isinstance(portfolio_metrics, dict):
                b = portfolio_metrics.get("balance", balance)
                balance = float(b if b is not None else balance)
                d = portfolio_metrics.get("drawdown", 0.0)
                drawdown = float(d if d is not None else 0.0)
            elif self.env:
                b = getattr(self.env, "balance", balance)
                balance = float(b if b is not None else balance)
                d = getattr(self.env, "current_drawdown", 0.0)
                drawdown = float(d if d is not None else 0.0)
            else:
                env_cfg = self.smart_bus.get("environment_config", "PositionManager")
                if isinstance(env_cfg, dict):
                    b = env_cfg.get("initial_balance", balance)
                    balance = float(b if b is not None else balance)

        # Exposure
        total_exposure = 0.0
        for pos_data in self.open_positions.values():
            if "size" in pos_data:
                total_exposure += abs(pos_data["size"])
            else:
                total_exposure += abs(pos_data.get("lots", 0)) * pos_data.get("price_open", 1) * 100_000
        exposure_ratio = total_exposure / max(balance, 1.0)

        # Health components
        dd_health = max(0.0, 1.0 - drawdown * 2.0)
        exposure_health = max(0.0, 1.0 - exposure_ratio / max(self.C.max_instrument_concentration, 1e-9))
        streak_health = max(0.1, 1.0 - self.consecutive_losses / max(self.C.max_consecutive_losses, 1))

        # Canonical risk first
        risk_level = 0.0
        tri = self.smart_bus.get("time_risk_analysis", "PositionManager") or {}
        mc = self.smart_bus.get("market_conditions", "PositionManager") or {}

        if isinstance(tri, dict):
            rl = tri.get("risk_level")
            if rl is None:
                rl = tri.get("risk_score", 0.0)
            risk_level = float(rl if rl is not None else 0.0)
        elif isinstance(mc, dict):
            rl = mc.get("risk_level")
            if rl is None:
                rl = mc.get("risk_score", 0.0)
            risk_level = float(rl if rl is not None else 0.0)
        else:
            # Legacy fallback
            rs = self.smart_bus.get("risk_score", "PositionManager")
            try:
                if isinstance(rs, dict):
                    rl = rs.get("risk_level")
                    risk_level = float(rl if rl is not None else 0.0)
                else:
                    risk_level = float(rs if rs is not None else 0.0)
            except Exception:  # noqa: BLE001
                risk_level = 0.0

        risk_level = float(np.clip(risk_level, 0.0, 1.0))
        risk_health = 1.0 - risk_level

        overall_health = (dd_health + exposure_health + streak_health + risk_health) / 4.0

        # Update internal trackers
        self._portfolio_health_score = overall_health
        self._total_exposure_ratio = exposure_ratio
        self._portfolio_health_history.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "health_score": overall_health,
                "exposure_ratio": exposure_ratio,
                "drawdown": drawdown,
                "balance": balance,
            }
        )

        return {
            "drawdown_health": dd_health,
            "exposure_health": exposure_health,
            "streak_health": streak_health,
            "risk_health": risk_health,
            "overall_health": overall_health,
            "total_exposure": total_exposure,
            "exposure_ratio": exposure_ratio,
            "balance": balance,
            "drawdown": drawdown,
        }

    def _assess_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Determine current market regime"""
        # Check if market regime already provided
        if "market_regime" in market_data:
            return str(market_data["market_regime"])

        # Extract volatility indicators
        avg_volatility = 0.0
        trend_strength = 0.0
        momentum_count = 0

        for instrument in self.instruments:
            inst_data = market_data.get(instrument, {})

            # Get volatility (from market data or default)
            vol = float(inst_data.get("volatility", self.C.min_volatility))
            avg_volatility += vol

            # Get trend indicators
            if "trend_strength" in inst_data:
                trend_strength += abs(float(inst_data["trend_strength"]))

            # Count momentum signals
            if abs(float(inst_data.get("momentum", 0.0))) > 0.3:
                momentum_count += 1

        if self.instruments:
            avg_volatility /= len(self.instruments)
            trend_strength /= len(self.instruments)

        # Classify regime
        if avg_volatility > 0.04:
            return "volatile"
        elif trend_strength > 0.5:
            return "trending"
        elif momentum_count >= len(self.instruments) * 0.6:
            return "momentum"
        else:
            return "ranging"

    def _extract_signal_context(
        self, instrument: str, market_data: Dict[str, Any], portfolio_health: Dict[str, float]
    ) -> SignalContext:
        """Extract and structure signal context for decision making (canonical-aware)."""
        inst_data = market_data.get(instrument, {}) or {}

        # Market signals (robust to None)
        raw_intensity = inst_data.get("intensity", 0.0)
        market_intensity = float(raw_intensity) if isinstance(raw_intensity, (int, float)) else 0.0
        market_direction = int(np.sign(market_intensity))

        volatility = inst_data.get("volatility", self.C.min_volatility)
        volatility = float(volatility) if isinstance(volatility, (int, float)) else self.C.min_volatility
        volatility = max(volatility, self.C.min_volatility)

        trend_strength = float(inst_data.get("trend_strength", 0.0) or 0.0)
        momentum = float(inst_data.get("momentum", 0.0) or 0.0)
        volume_profile = float(inst_data.get("volume_profile", 1.0) or 1.0)

        # Regime/session
        regime = market_data.get("market_regime", None)
        if regime is None:
            regime = self.smart_bus.get("market_regime", "PositionManager") or "normal"

        market_conditions = self.smart_bus.get("market_conditions", "PositionManager") or {}
        session = inst_data.get("session") or market_conditions.get("session", "unknown")

        # Correlation penalty (optional)
        correlation_penalty = 0.0
        correlation_data = self.smart_bus.get("correlation_matrix", "PositionManager")
        if isinstance(correlation_data, dict):
            try:
                # Handle two formats:
                # 1) PortfolioRiskSystem matrix-like: {instrument: {avg_correlation: x, ...}}
                node = correlation_data.get(instrument)
                if isinstance(node, dict):
                    ac = node.get("avg_correlation")
                    if isinstance(ac, (int, float)):
                        correlation_penalty = min(abs(float(ac)) * 0.5, 0.8)
                else:
                    # 2) CorrelatedRiskController pairwise string-keyed map: "('EUR/USD','XAU/USD')": corr
                    # Compute per-instrument average absolute correlation
                    total = 0.0
                    count = 0
                    for k, v in correlation_data.items():
                        if not isinstance(k, str):
                            continue
                        if instrument in k:
                            try:
                                corr_val = float(v)
                            except Exception:
                                continue
                            if np.isfinite(corr_val):
                                total += abs(corr_val)
                                count += 1
                    if count > 0:
                        avg_abs_corr = total / count
                        correlation_penalty = min(avg_abs_corr * 0.5, 0.8)
            except Exception:
                correlation_penalty = 0.0

        # Liquidity
        liquidity_score = self._get_liquidity(instrument)

        # Normalize portfolio fields
        bal_raw = portfolio_health.get("balance")
        balance_val = float(bal_raw) if isinstance(bal_raw, (int, float)) else float(self.C.initial_balance)
        dd_raw = portfolio_health.get("drawdown", 0.0)
        dd_val = float(dd_raw) if isinstance(dd_raw, (int, float)) else 0.0

        return SignalContext(
            instrument=instrument,
            market_intensity=market_intensity,
            market_direction=market_direction,
            volatility=volatility,
            trend_strength=trend_strength,
            momentum=momentum,
            volume_profile=volume_profile,
            correlation_penalty=correlation_penalty,
            regime=str(regime),
            liquidity_score=liquidity_score,
            session=session,
            current_exposure=portfolio_health.get("exposure_ratio", 0.0) or 0.0,
            drawdown=dd_val,
            balance=balance_val,
            step_idx=0,
            timestamp=datetime.datetime.now().isoformat(),
        )

    def _make_position_decision(self, context: SignalContext) -> PositionDecisionResult:
        """
        Core position decision logic implementing hierarchical decision making
        """
        instrument = context.instrument
        has_position = instrument in self.open_positions

        # Initialize decision components
        decision = PositionDecision.HOLD
        intensity = 0.0
        size = 0.0
        confidence = 0.5
        risk_factors: Dict[str, float] = {}
        rationale: Dict[str, Any] = {"stage": "initial", "factors": []}

        # Get absolute signal strength
        signal_strength = abs(context.market_intensity)
        signal_direction = np.sign(context.market_intensity)

        # Stage 1: Emergency conditions check
        if self._check_emergency_conditions(context):
            if has_position:
                decision = PositionDecision.EMERGENCY_CLOSE
                intensity = 1.0
                confidence = 0.9
                rationale["stage"] = "emergency"
                rationale["factors"].append("Emergency conditions detected")

                self.logger.warning(
                    format_operator_message(
                        "[ALERT]",
                        "EMERGENCY_CLOSE",
                        instrument=instrument,
                        drawdown=f"{context.drawdown:.1%}",
                        consecutive_losses=self.consecutive_losses,
                        exposure=f"{context.current_exposure:.1%}",
                    )
                )
            return PositionDecisionResult(decision, intensity, size, confidence, rationale, risk_factors, context)

        # Stage 2: Signal strength filtering
        if signal_strength < self.C.min_signal_threshold:
            rationale["stage"] = "signal_filter"
            rationale["factors"].append(
                f"Signal strength {signal_strength:.3f} below threshold {self.C.min_signal_threshold}"
            )
            return PositionDecisionResult(decision, intensity, size, confidence, rationale, risk_factors, context)

        # Stage 3: Portfolio health checks
        portfolio_health_score = self._calculate_portfolio_health_score(context)
        if portfolio_health_score < 0.3:
            rationale["stage"] = "portfolio_health"
            rationale["factors"].append(f"Portfolio health {portfolio_health_score:.3f} too low")
            # Still allow closes but no new positions
            if has_position and signal_strength > 0.7:
                decision = PositionDecision.CLOSE
                intensity = 0.8
                confidence = 0.7
                rationale["factors"].append("Closing due to poor portfolio health")
            return PositionDecisionResult(decision, intensity, size, confidence, rationale, risk_factors, context)

        # Stage 4: Position-specific decision logic
        if not has_position:
            # New position logic
            if signal_strength >= self.C.min_signal_threshold:
                decision = PositionDecision.OPEN_LONG if signal_direction > 0 else PositionDecision.OPEN_SHORT
                intensity = signal_strength
                confidence = self._calculate_confidence(context, decision)
                size = self._calculate_position_size(context, intensity, confidence)
                rationale["stage"] = "new_position"
                rationale["factors"].append(f"Strong signal {signal_strength:.3f} for new position")
        else:
            # Existing position management
            current_side = self.open_positions[instrument].get("side", 0)
            position_pnl, _ = self._calc_unrealised_pnl(instrument, self.open_positions[instrument])

            # Check if signal aligns with current position
            signal_aligns = (current_side > 0 and signal_direction > 0) or (current_side < 0 and signal_direction < 0)

            if signal_aligns and signal_strength > self.C.position_scale_threshold:
                # Scale up position
                decision = PositionDecision.SCALE_UP
                intensity = min(signal_strength * 0.8, 0.9)  # Conservative scaling
                confidence = self._calculate_confidence(context, decision)
                size = self._calculate_position_size(context, intensity, confidence) * 0.5  # Smaller scale
                rationale["stage"] = "scale_up"
                rationale["factors"].append(f"Signal {signal_strength:.3f} aligns with position, scaling up")

            elif not signal_aligns and signal_strength > 0.5:
                # Consider closing or reversing
                if signal_strength > 0.8:
                    decision = PositionDecision.CLOSE
                    intensity = 0.9
                    confidence = 0.8
                    rationale["stage"] = "close_reverse"
                    rationale["factors"].append(f"Strong opposing signal {signal_strength:.3f}")
                else:
                    decision = PositionDecision.SCALE_DOWN
                    intensity = 0.6
                    confidence = 0.6
                    rationale["stage"] = "scale_down"
                    rationale["factors"].append(f"Opposing signal {signal_strength:.3f}, reducing exposure")

            elif position_pnl < -self.C.hard_loss_eur * 0.5:  # Approaching hard loss
                decision = PositionDecision.CLOSE
                intensity = 0.8
                confidence = 0.9
                rationale["stage"] = "risk_management"
                rationale["factors"].append(f"Position approaching loss limit: €{position_pnl:.2f}")

        # Stage 5: Final risk adjustments
        risk_factors = self._assess_risk_factors(context)
        final_intensity = intensity * (1.0 - max(risk_factors.values()) if risk_factors else 1.0)
        final_confidence = confidence * portfolio_health_score

        # Ensure minimum viable size or zero
        if decision in [PositionDecision.OPEN_LONG, PositionDecision.OPEN_SHORT, PositionDecision.SCALE_UP]:
            if size < context.balance * self.C.min_size_pct and final_intensity > 0.3:
                size = context.balance * self.C.min_size_pct
            elif size < context.balance * self.C.min_size_pct:
                size = 0.0
                decision = PositionDecision.HOLD
                rationale["factors"].append("Size too small, holding instead")

        return PositionDecisionResult(
            decision=decision,
            intensity=final_intensity,
            size=size,
            confidence=final_confidence,
            rationale=rationale,
            risk_factors=risk_factors,
            context=context,
        )

    def _check_emergency_conditions(self, context: SignalContext) -> bool:
        """Check for emergency conditions requiring immediate action"""
        emergency_conditions = [
            context.drawdown > 0.15,  # 15% drawdown
            self.consecutive_losses >= self.C.max_consecutive_losses,
            context.current_exposure > self.C.max_instrument_concentration * 1.5,
            context.liquidity_score < 0.3,
        ]
        return any(emergency_conditions)

    def _calculate_portfolio_health_score(self, context: SignalContext) -> float:
        """Calculate overall portfolio health score"""
        drawdown_component = max(0.0, 1.0 - context.drawdown * 3.0)
        exposure_component = max(0.0, 1.0 - context.current_exposure / max(self.C.max_instrument_concentration, 1e-9))
        streak_component = max(0.1, 1.0 - self.consecutive_losses / max(self.C.max_consecutive_losses, 1))
        liquidity_component = context.liquidity_score

        return (drawdown_component + exposure_component + streak_component + liquidity_component) / 4.0

    def _calculate_confidence(self, context: SignalContext, decision: PositionDecision) -> float:
        """Calculate confidence in the position decision"""
        base_confidence = 0.5

        # Signal strength contribution
        signal_confidence = min(abs(context.market_intensity) * 1.2, 0.4)

        # Trend alignment
        trend_confidence = min(abs(context.trend_strength) * 0.3, 0.2)

        # Volatility penalty (high vol = lower confidence)
        vol_penalty = min(context.volatility / 0.05, 0.2)

        # Portfolio health contribution
        health_boost = self._calculate_portfolio_health_score(context) * 0.2

        # Decision-specific adjustments
        decision_adjustment = 0.0
        if decision in [PositionDecision.CLOSE, PositionDecision.EMERGENCY_CLOSE]:
            decision_adjustment = 0.1  # Higher confidence in exits
        elif decision == PositionDecision.SCALE_DOWN:
            decision_adjustment = 0.05

        total_confidence = (
            base_confidence + signal_confidence + trend_confidence - vol_penalty + health_boost + decision_adjustment
        )
        return float(np.clip(total_confidence, 0.1, 1.0))

    def _assess_risk_factors(self, context: SignalContext) -> Dict[str, float]:
        """Assess various risk factors that might reduce position size"""
        risk_factors: Dict[str, float] = {}

        # Volatility risk
        risk_factors["volatility"] = min((context.volatility - self.C.min_volatility) / 0.05, 0.5)

        # Correlation risk
        risk_factors["correlation"] = context.correlation_penalty

        # Drawdown risk
        risk_factors["drawdown"] = min(context.drawdown * 2.0, 0.8)

        # Concentration risk
        denom = max(self.C.max_instrument_concentration, 1e-9)
        risk_factors["concentration"] = min(context.current_exposure / denom, 0.9)

        # Liquidity risk
        risk_factors["liquidity"] = max(0.0, 1.0 - context.liquidity_score)

        # Session risk (trading outside optimal hours)
        session_risk = 0.0
        if context.session == "closed":
            session_risk = 0.3
        elif context.session == "asian":
            session_risk = 0.1  # Lower liquidity
        risk_factors["session"] = session_risk

        return risk_factors

    def _calculate_position_size(self, context: SignalContext, intensity: float, confidence: float) -> float:
        """Calculate position size using the enhanced sizing logic"""
        return self.calculate_size(
            volatility=context.volatility,
            intensity=intensity,
            balance=context.balance,
            drawdown=context.drawdown,
            correlation=context.correlation_penalty,
            current_exposure=context.current_exposure,
        )

    def calculate_size(
        self,
        volatility: float,
        intensity: float,
        balance: float,
        drawdown: float,
        correlation: Optional[float] = None,
        current_exposure: Optional[float] = None,
    ) -> float:
        """
        Enhanced position sizing that integrates with hierarchical decision making
        """
        # Input sanitization
        volatility = max(float(np.nan_to_num(volatility, nan=self.C.min_volatility)), self.C.min_volatility)
        intensity = float(np.nan_to_num(intensity, nan=0.0))
        balance = max(float(balance), 100.0)  # Minimum balance
        drawdown = float(np.nan_to_num(drawdown, nan=0.0))

        # Clip intensity to reasonable range
        intensity = float(np.clip(intensity, -1.0, 1.0))

        # Base risk budget calculation
        risk_pct = max(float(self._adaptive_params["dynamic_max_pct"]), 0.01)  # Ensure minimum risk allocation
        risk_budget = balance * risk_pct

        # Volatility-adjusted base size
        vol_adjusted_budget = risk_budget / volatility
        base_size = intensity * vol_adjusted_budget

        # Apply portfolio health modifiers
        portfolio_health = self._portfolio_health_score

        # Health-based size adjustment
        health_multiplier = max(0.1, portfolio_health)  # Never go completely to zero
        adjusted_size = base_size * health_multiplier

        # Risk tolerance adjustments
        risk_tolerance = float(self._adaptive_params.get("risk_tolerance", 1.0))
        adjusted_size *= risk_tolerance

        # Correlation penalty
        if correlation is not None:
            corr_penalty = 1.0 - min(abs(float(correlation)) * 0.3, 0.5)  # Less aggressive penalty
            adjusted_size *= corr_penalty

        # Loss streak reduction
        if self.consecutive_losses >= self.C.max_consecutive_losses:
            streak_reduction = max(0.1, self.C.loss_reduction)  # Never reduce below 10%
            adjusted_size *= streak_reduction
            self.logger.info(
                format_operator_message(
                    "📉",
                    "LOSS_STREAK_REDUCTION",
                    reduction_factor=f"{streak_reduction:.2f}",
                    consecutive_losses=self.consecutive_losses,
                )
            )

        # Ensure minimum viable size or zero
        abs_size = abs(adjusted_size)
        min_viable_size = balance * self.C.min_size_pct

        if abs_size < min_viable_size and abs(intensity) > 0.3:
            # Strong signal but small size - use minimum
            adjusted_size = np.sign(adjusted_size or intensity) * min_viable_size
        elif abs_size < min_viable_size:
            # Weak signal and small size - zero out
            adjusted_size = 0.0

        # Final safety bounds
        max_single_position = balance * risk_pct
        final_size = float(np.clip(adjusted_size, -max_single_position, max_single_position))

        return float(np.nan_to_num(final_size, nan=0.0, posinf=0.0, neginf=0.0))

    def _apply_position_management(self) -> None:
        """Apply position management rules including live sync and exits"""

        # Ensure minimum allocation capability
        min_cap = 0.01  # 1% minimal allocation
        if float(self._adaptive_params["dynamic_max_pct"]) < min_cap:
            self.logger.warning(
                format_operator_message(
                    "[WARN]",
                    "DYNAMIC_MAX_PCT_LOW",
                    current=f"{float(self._adaptive_params['dynamic_max_pct']):.6f}",
                    minimum=f"{min_cap:.4f}",
                    action="Resetting to default",
                )
            )
            self._adaptive_params["dynamic_max_pct"] = self.default_max_pct

        # Live‐mode: sync & apply exit rules
        if self.env and getattr(self.env, "live_mode", False):
            self._sync_live_positions()
            self._apply_exit_rules()

        # Decay position confidence over time
        for inst in list(self.position_confidence.keys()):
            self.position_confidence[inst] *= self.C.confidence_decay

    # ─────────────────────────────────────────────────────────
    # Simulated execution helpers
    # ─────────────────────────────────────────────────────────
    def _get_current_price(self, instrument: str, fallback: Optional[float] = None) -> Optional[float]:
        """Best-effort current price for an instrument from SmartBus/prices maps."""
        sym_variants = [instrument, instrument.replace('/', ''), instrument.replace('/', '_'), instrument.upper(), instrument.lower()]
        # Try price_data first
        pd = self.smart_bus.get('price_data', 'PositionManager') or {}
        if isinstance(pd, dict):
            for k in sym_variants:
                node = pd.get(k)
                if isinstance(node, dict):
                    for key in ('last', 'close', 'price', 'bid', 'ask'):
                        v = node.get(key)
                        if isinstance(v, (int, float)):
                            return float(v)
        # Try simple prices
        sp = self.smart_bus.get('prices', 'PositionManager') or {}
        if isinstance(sp, dict):
            for k in sym_variants:
                v = sp.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict):
                    for key in ('last', 'close', 'price', 'bid', 'ask'):
                        vv = v.get(key)
                        if isinstance(vv, (int, float)):
                            return float(vv)
        return fallback

    def _apply_simulated_execution(self, decisions: Dict[str, PositionDecisionResult], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply a minimal immediate-fill simulated execution. Returns accounting snapshot if applied."""
        try:
            if not self.simulate_execution or (self.env and getattr(self.env, 'live_mode', False)):
                return None

            step_trades: List[Dict[str, Any]] = []

            # Iterate decisions and open/scale/close accordingly
            for inst, dr in decisions.items():
                # Determine a price
                inst_map = market_data.get(inst, {}) or {}
                price = inst_map.get('current_price')
                if not isinstance(price, (int, float)):
                    price = self._get_current_price(inst)
                if not isinstance(price, (int, float)):
                    # Skip if no price
                    continue

                decision = dr.decision
                size_eur = float(dr.size)
                if decision in (PositionDecision.OPEN_LONG, PositionDecision.OPEN_SHORT):
                    if size_eur <= 0:
                        continue
                    if inst in self.open_positions:
                        # Already open; treat as scale if same side else close+open
                        cur_side = self.open_positions[inst].get('side', 0)
                        new_side = 1 if decision == PositionDecision.OPEN_LONG else -1
                        if cur_side == new_side:
                            decision = PositionDecision.SCALE_UP
                        else:
                            # Close existing then open new
                            self._close_position_sim(inst, price, reason='reverse')
                            step_trades.append(self._recent_trades[-1]) if self._recent_trades else None
                    if decision in (PositionDecision.OPEN_LONG, PositionDecision.OPEN_SHORT):
                        lots = max(size_eur / max(price, 1e-6) / 100000.0, 0.0)
                        self._open_position_sim(inst, price, 1 if decision == PositionDecision.OPEN_LONG else -1, lots)
                elif decision == PositionDecision.SCALE_UP:
                    if inst in self.open_positions and size_eur > 0:
                        lots = max(size_eur / max(price, 1e-6) / 100000.0, 0.0)
                        self._scale_position_sim(inst, price, lots)
                elif decision in (PositionDecision.CLOSE, PositionDecision.EMERGENCY_CLOSE, PositionDecision.SCALE_DOWN):
                    if inst in self.open_positions:
                        # For SCALE_DOWN, reduce half; treat as partial close
                        if decision == PositionDecision.SCALE_DOWN:
                            self._partial_close_position_sim(inst, price, portion=0.5)
                        else:
                            self._close_position_sim(inst, price, reason=decision.value)
                        if self._recent_trades:
                            step_trades.append(self._recent_trades[-1])

            # Compute unrealized PnL and equity
            unreal = 0.0
            for inst, pos in self.open_positions.items():
                price = self._get_current_price(inst, pos.get('price_open'))
                if not isinstance(price, (int, float)):
                    continue
                points = (price - pos.get('price_open', price)) * pos.get('side', 0)
                unreal += points * 100000.0 * float(pos.get('lots', 0.0))

            equity = self._sim_balance + unreal
            current_pnl = unreal

            return {
                'balance': float(self._sim_balance),
                'equity': float(equity),
                'current_pnl': float(current_pnl),
                'trades': list(step_trades),
            }
        except Exception:
            return None

    # Simulated execution primitives
    def _open_position_sim(self, inst: str, price: float, side: int, lots: float) -> None:
        lots = float(max(lots, 0.0))
        if lots <= 0:
            return
        self._sim_ticket_counter += 1
        eur_exposure = lots * float(price) * 100000.0
        self.open_positions[inst] = {
            'ticket': self._sim_ticket_counter,
            'side': int(np.sign(side) or 1),
            'lots': lots,
            'price_open': float(price),
            'size': float(abs(eur_exposure)),  # EUR exposure for visualization/exposure
            'peak_profit': 0.0,
        }
        self.logger.info(format_operator_message('[GREEN]', 'OPEN_SIM', instrument=inst, side=('LONG' if side>0 else 'SHORT'), lots=f"{lots:.2f}", price=f"{price:.5f}"))

    def _scale_position_sim(self, inst: str, price: float, add_lots: float) -> None:
        if inst not in self.open_positions or add_lots <= 0:
            return
        pos = self.open_positions[inst]
        # VWAP update for price_open
        total_lots = float(pos.get('lots', 0.0)) + float(add_lots)
        if total_lots <= 0:
            return
        vwap = (pos.get('price_open', price) * float(pos.get('lots', 0.0)) + price * float(add_lots)) / total_lots
        pos['lots'] = total_lots
        pos['price_open'] = float(vwap)
        pos['size'] = float(abs(total_lots * price * 100000.0))
        self.logger.info(
            format_operator_message(
                '[GREEN]', 'SCALE_SIM', instrument=inst,
                add_lots=f"{add_lots:.2f}", new_lots=f"{total_lots:.2f}", vwap=f"{vwap:.5f}"
            )
        )

    def _partial_close_position_sim(self, inst: str, price: float, portion: float = 0.5) -> None:
        if inst not in self.open_positions:
            return
        pos = self.open_positions[inst]
        close_lots = float(pos.get('lots', 0.0)) * float(np.clip(portion, 0.0, 1.0))
        if close_lots <= 0:
            return
        remaining_lots = float(pos.get('lots', 0.0)) - close_lots
        pnl = (price - pos.get('price_open', price)) * pos.get('side', 0) * 100000.0 * close_lots
        self._sim_balance += float(pnl)
        trade = {
            'instrument': inst,
            'side': 'SELL' if pos.get('side', 0) > 0 else 'BUY',
            'lots': close_lots,
            'price_open': float(pos.get('price_open', price)),
            'price_close': float(price),
            'pnl': float(pnl),
            'timestamp': datetime.datetime.now().isoformat(),
            'ticket': pos.get('ticket'),
            'type': 'partial_close',
        }
        self._trade_ledger.append(trade)
        self._recent_trades.append(trade)
        if remaining_lots <= 0:
            self.open_positions.pop(inst, None)
        else:
            pos['lots'] = remaining_lots
            pos['size'] = float(abs(remaining_lots * price * 100000.0))
        self.logger.info(format_operator_message('[RED]', 'PARTIAL_CLOSE_SIM', instrument=inst, lots=f"{close_lots:.2f}", price=f"{price:.5f}", pnl=f"{pnl:+.2f}", remain=f"{remaining_lots:.2f}"))

    def _close_position_sim(self, inst: str, price: float, reason: str = 'close') -> None:
        if inst not in self.open_positions:
            return
        pos = self.open_positions.pop(inst)
        pnl = (price - pos.get('price_open', price)) * pos.get('side', 0) * 100000.0 * float(pos.get('lots', 0.0))
        self._sim_balance += float(pnl)
        trade = {
            'instrument': inst,
            'side': 'SELL' if pos.get('side', 0) > 0 else 'BUY',
            'lots': float(pos.get('lots', 0.0)),
            'price_open': float(pos.get('price_open', price)),
            'price_close': float(price),
            'pnl': float(pnl),
            'timestamp': datetime.datetime.now().isoformat(),
            'ticket': pos.get('ticket'),
            'type': reason,
        }
        self._trade_ledger.append(trade)
        self._recent_trades.append(trade)
        self.logger.info(format_operator_message('[RED]', 'CLOSE_SIM', instrument=inst, reason=reason, price=f"{price:.5f}", pnl=f"{pnl:+.2f}"))

    def _update_position_health(self) -> None:
        """Update position health metrics (canonical-aware)."""
        try:
            if not hasattr(self, "open_positions") or not hasattr(self, "_exposure_history"):
                return

            current_exposure = self._calculate_current_exposure_ratio()
            self._total_exposure_ratio = current_exposure

            self._exposure_history.append(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "exposure_ratio": current_exposure,
                    "position_count": len(self.open_positions),
                    "consecutive_losses": self.consecutive_losses,
                }
            )

            # Risk via canonical feeds first
            risk_level = 0.0
            tri = self.smart_bus.get("time_risk_analysis", "PositionManager") or {}
            mc = self.smart_bus.get("market_conditions", "PositionManager") or {}
            if isinstance(tri, dict):
                rl = tri.get("risk_level")
                if rl is None:
                    rl = tri.get("risk_score", 0.0)
                risk_level = float(rl if rl is not None else 0.0)
            elif isinstance(mc, dict):
                rl = mc.get("risk_level")
                if rl is None:
                    rl = mc.get("risk_score", 0.0)
                risk_level = float(rl if rl is not None else 0.0)
            else:
                # Legacy fallback
                rs = self.smart_bus.get("risk_score", "PositionManager")
                try:
                    if isinstance(rs, dict):
                        rl = rs.get("risk_level")
                        risk_level = float(rl if rl is not None else 0.0)
                    else:
                        risk_level = float(rs if rs is not None else 0.0)
                except Exception:  # noqa: BLE001
                    risk_level = 0.0

            risk_level = float(np.clip(risk_level, 0.0, 1.0))
            self._risk_management_score = max(0.1, 1.0 - risk_level)

            # Param adaptation & bus write
            self._adapt_parameters()
            self.smart_bus.set(
                "position_health",
                {
                    "portfolio_health": self._portfolio_health_score,
                    "exposure_ratio": current_exposure,
                    "risk_management_score": self._risk_management_score,
                    "consecutive_losses": self.consecutive_losses,
                },
                module="PositionManager",
                thesis="Position manager health update",
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Position health update failed: {e}")

    def _adapt_parameters(self) -> None:
        """Adapt position management parameters based on performance (consistent keys)."""
        try:
            # Adapt risk ceiling around config.max_position_pct
            if len(self._decision_history) >= 10:
                recent = list(self._decision_history)[-10:]
                avg_ph = np.mean([d.get("portfolio_health", self._portfolio_health_score) for d in recent])

                current = float(self._adaptive_params.get("dynamic_max_pct", self.C.max_position_pct))
                base = float(self.C.max_position_pct)

                if avg_ph > 0.8:
                    self._adaptive_params["dynamic_max_pct"] = min(current * 1.05, base * 1.5)
                elif avg_ph < 0.4:
                    self._adaptive_params["dynamic_max_pct"] = max(current * 0.90, base * 0.3)
                else:
                    # mean-revert to base slowly
                    self._adaptive_params["dynamic_max_pct"] = current * 0.95 + base * 0.05

            # Adapt signal sensitivity based on recent non-hold confidences
            if len(self._decision_history) >= 5:
                rec = list(self._decision_history)[-5:]
                confs: List[float] = []
                for record in rec:
                    for d in record.get("decisions", {}).values():
                        if d.get("decision") != "hold":
                            confs.append(float(d.get("confidence", 0.5)))
                if confs:
                    avgc = float(np.mean(confs))
                    if avgc > 0.7:
                        self._adaptive_params["signal_sensitivity"] = min(
                            1.2, float(self._adaptive_params["signal_sensitivity"]) * 1.02
                        )
                    elif avgc < 0.4:
                        self._adaptive_params["signal_sensitivity"] = max(
                            0.7, float(self._adaptive_params["signal_sensitivity"]) * 0.98
                        )

            # Risk tolerance vs loss streak
            if self.consecutive_losses == 0:
                self._adaptive_params["risk_tolerance"] = min(
                    1.3, float(self._adaptive_params["risk_tolerance"]) * 1.01
                )
            elif self.consecutive_losses >= 3:
                self._adaptive_params["risk_tolerance"] = max(
                    0.5, float(self._adaptive_params["risk_tolerance"]) * 0.95
                )

        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Parameter adaptation failed: {e}")

    def _sync_live_positions(self) -> None:
        """Sync positions from live broker"""
        broker_positions: List[Dict[str, Any]] = []

        # 1) env.broker
        if self.env and hasattr(self.env, "broker") and self.env.broker is not None:
            try:
                broker_positions = self.env.broker.get_positions()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Broker position sync failed: {exc}")

        # 2) MT5 fallback
        elif mt5 is not None:
            raw = getattr(mt5, "positions_get", lambda: [])() or []
            for p in raw:
                broker_positions.append(
                    dict(
                        instrument=f"{p.symbol[:3]}/{p.symbol[3:]}",
                        ticket=p.ticket,
                        side=1 if p.type == mt5.POSITION_TYPE_BUY else -1,
                        lots=p.volume,
                        price_open=p.price_open,
                    )
                )

        if not broker_positions:
            return

        new_positions: Dict[str, Dict[str, Any]] = {}
        for pos in broker_positions:
            inst = pos["instrument"]
            d = {
                "ticket": pos["ticket"],
                "side": pos["side"],
                "lots": pos["lots"],
                "price_open": pos["price_open"],
                "peak_profit": self.open_positions.get(inst, {}).get("peak_profit", 0.0),
                "size": pos["lots"],
            }
            new_positions[inst] = d

        self.open_positions = new_positions
        self._last_sync_time = datetime.datetime.now()

        self.logger.info(
            format_operator_message(
                "[RELOAD]", "POSITIONS_SYNCED", position_count=len(self.open_positions), timestamp=self._last_sync_time.isoformat()
            )
        )

    def _apply_exit_rules(self) -> None:
        """Apply automated exit rules to open positions"""
        for inst, data in list(self.open_positions.items()):
            pnl_eur, _ = self._calc_unrealised_pnl(inst, data)
            # update peak
            if pnl_eur > data.get("peak_profit", 0.0):
                data["peak_profit"] = float(pnl_eur)

            # hard‐loss
            if pnl_eur <= -self.C.hard_loss_eur:
                self._close_position(inst, "hard_loss")
                continue

            # trailing‐profit
            if data.get("peak_profit", 0.0) > 0:
                drawdown_eur = float(data["peak_profit"]) - float(pnl_eur)
                trigger = max(float(data["peak_profit"]) * float(self.C.trail_pct), float(self.C.trail_abs_eur))
                if drawdown_eur >= trigger:
                    self._close_position(inst, "trail_stop")

    def _close_position(self, inst: str, reason: str) -> None:
        """Close a position via broker or simulation"""
        # env.broker
        if self.env and getattr(self.env, "broker", None):
            ok = self.env.broker.close_position(inst, comment=reason)
            if ok:
                self.logger.info(
                    format_operator_message("[RED]", "POSITION_CLOSED", instrument=inst, reason=reason, via="broker")
                )
                self.open_positions.pop(inst, None)
            else:
                self.logger.error(f"Broker close failed: {inst}, reason: {reason}")
            return

        # MT5 close
        if mt5 is not None:
            data = self.open_positions[inst]
            side = data["side"]
            lots = data["lots"]
            sym = inst.replace("/", "")
            tick = getattr(mt5, "symbol_info_tick", lambda x: None)(sym)
            price = (tick.bid if side > 0 else tick.ask) if tick else 0.0

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym,
                "volume": lots,
                "type": (mt5.ORDER_TYPE_SELL if side > 0 else mt5.ORDER_TYPE_BUY),
                "price": price,
                "deviation": self.C.pips_tolerance,
                "position": data["ticket"],
                "magic": 10001,
                "comment": f"auto-exit:{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            order_send = getattr(mt5, "order_send", None)
            res = order_send(request) if order_send else None
            if res and getattr(res, "retcode", -1) == getattr(mt5, "TRADE_RETCODE_DONE", 10009):
                self.logger.info(
                    format_operator_message(
                        "[RED]", "POSITION_CLOSED_MT5", instrument=inst, ticket=data["ticket"], reason=reason
                    )
                )
                self.open_positions.pop(inst, None)
            else:
                self.logger.error(f"MT5 close failed: {inst}, error: {str(res)}")
            return

        # backtest fallback
        self.logger.info(
            format_operator_message("[RED]", "POSITION_CLOSED_SIM", instrument=inst, reason=reason, mode="simulation")
        )
        self.open_positions.pop(inst, None)

    def _calc_unrealised_pnl(self, inst: str, data: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate unrealized P&L for a position"""
        sym = inst.replace("/", "")
        # price
        price: Optional[float] = None
        if self.env and getattr(self.env, "broker", None):
            price = self.env.broker.get_price(sym, side=data["side"])
        elif mt5 is not None:
            tick = getattr(mt5, "symbol_info_tick", lambda x: None)(sym)
            if tick:
                price = tick.bid if data["side"] > 0 else tick.ask
        if price is None or not np.isfinite(price):
            return 0.0, 0.0

        # contract size
        contract_size = 100_000
        if mt5 is not None:
            info = getattr(mt5, "symbol_info", lambda x: None)(sym)
            if info and getattr(info, "trade_contract_size", None):
                contract_size = info.trade_contract_size

        points = (price - data["price_open"]) * data["side"]
        pnl_eur = points * contract_size * data["lots"]
        pnl_pct = pnl_eur / (abs(data["price_open"]) * contract_size * data["lots"])
        return float(pnl_eur), float(pnl_pct)

    def _calculate_current_exposure_ratio(self) -> float:
        """Calculate current exposure as ratio of balance"""
        if not self.open_positions:
            return 0.0

        balance: float = float(self.C.initial_balance)

        # Try to get current balance from SmartInfoBus
        portfolio_metrics = self.smart_bus.get("portfolio_metrics", "PositionManager")
        if portfolio_metrics:
            balance = float(portfolio_metrics.get("balance", self.C.initial_balance))
        elif self.env:
            balance = float(getattr(self.env, "balance", self.C.initial_balance))

        total_exposure = 0.0

        for pos_data in self.open_positions.values():
            if "size" in pos_data:
                total_exposure += abs(float(pos_data["size"]))
            else:
                # Live mode estimation
                total_exposure += abs(float(pos_data.get("lots", 0))) * float(pos_data.get("price_open", 1)) * 100_000

        return total_exposure / max(balance, 1.0)

    def _assess_signal_quality(self) -> float:
        """Assess the quality of recent signals"""
        if not self.signal_history:
            return 0.5

        quality_scores: List[float] = []
        for _, history in self.signal_history.items():
            if len(history) >= 5:
                # Check signal consistency and strength
                recent_signals = history[-5:]
                signal_strength = float(np.mean(np.abs(recent_signals)))
                signal_consistency = float(1.0 - np.std(recent_signals))
                inst_quality = (signal_strength + max(0.0, float(signal_consistency))) / 2.0
                quality_scores.append(inst_quality)

        return float(np.mean(quality_scores)) if quality_scores else 0.5

    # ================================================================
    # ENHANCED STATE MANAGEMENT
    # ================================================================
    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence"""
        return {
            "config": dict(self.C.__dict__),  # persist typed config as dict
            "genome": self.genome,
            "open_positions": self.open_positions,
            "portfolio_health_score": self._portfolio_health_score,
            "exposure_ratio": self._total_exposure_ratio,
            "consecutive_losses": self.consecutive_losses,
            "adaptive_params": self._adaptive_params,
            "position_confidence": self.position_confidence,
            "signal_history": {k: v[-20:] for k, v in self.signal_history.items()},  # Keep last 20
            "decision_history": list(self._decision_history)[-20:],  # Keep recent history
            "portfolio_health_history": list(self._portfolio_health_history)[-20:],
            "last_decisions": {
                k: {
                    "decision": v.decision.value,
                    "intensity": v.intensity,
                    "confidence": v.confidence,
                    "rationale": v.rationale,
                }
                for k, v in self.last_decisions.items()
            },
            "success_count": getattr(self, "success_count", 0),
            "failure_count": getattr(self, "failure_count", 0),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload"""
        if "config" in state:
            cfg = state["config"]
            if isinstance(cfg, dict):
                self.C = TradingConfig(**cfg)
                self.config.update(self.C.__dict__)
                self.default_max_pct = self.C.max_position_pct
                self.enable_legacy_bus_signal_probe = bool(
                    self.config.get("enable_legacy_bus_signal_probe", False)
                )

        if "genome" in state:
            self.genome = state["genome"]
        if "open_positions" in state:
            self.open_positions = state["open_positions"]
        if "portfolio_health_score" in state:
            self._portfolio_health_score = float(state["portfolio_health_score"])
        if "exposure_ratio" in state:
            self._total_exposure_ratio = float(state["exposure_ratio"])
        if "consecutive_losses" in state:
            self.consecutive_losses = int(state["consecutive_losses"])
        if "adaptive_params" in state:
            self._adaptive_params = state["adaptive_params"]
        if "position_confidence" in state:
            self.position_confidence = state["position_confidence"]
        if "signal_history" in state:
            for inst, history in state["signal_history"].items():
                if inst in self.signal_history:
                    self.signal_history[inst] = history

        # Restore decision history
        if "decision_history" in state:
            self._decision_history = deque(state["decision_history"], maxlen=100)
        if "portfolio_health_history" in state:
            self._portfolio_health_history = deque(state["portfolio_health_history"], maxlen=50)

        # Restore last decisions
        if "last_decisions" in state:
            self.last_decisions.clear()
            for inst, decision_data in state["last_decisions"].items():
                try:
                    decision = PositionDecision(decision_data["decision"])
                    context = SignalContext(instrument=inst)  # Minimal context
                    result = PositionDecisionResult(
                        decision=decision,
                        intensity=float(decision_data.get("intensity", 0.0)),
                        size=0.0,
                        confidence=float(decision_data.get("confidence", 0.5)),
                        rationale=decision_data.get("rationale", {}),
                        risk_factors={},
                        context=context,
                    )
                    self.last_decisions[inst] = result
                except Exception:  # noqa: BLE001
                    pass

        # Restore counts
        if "success_count" in state:
            self.success_count = int(state["success_count"])
        if "failure_count" in state:
            self.failure_count = int(state["failure_count"])

    # ================================================================
    # EVOLUTIONARY METHODS
    # ================================================================
    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()

    def set_genome(self, genome: Dict[str, Any]) -> None:
        """Set evolutionary genome"""
        self.C.max_position_pct = float(np.clip(genome.get("max_position_pct", self.C.max_position_pct), 0.01, 0.25))
        self.C.max_consecutive_losses = int(np.clip(genome.get("max_consecutive_losses", self.C.max_consecutive_losses), 1, 20))
        self.C.loss_reduction = float(np.clip(genome.get("loss_reduction", self.C.loss_reduction), 0.05, 1.0))
        self.C.max_instrument_concentration = float(
            np.clip(genome.get("max_instrument_concentration", self.C.max_instrument_concentration), 0.05, 0.5)
        )
        self.C.min_volatility = float(np.clip(genome.get("min_volatility", self.C.min_volatility), 0.001, 0.10))
        self.C.hard_loss_eur = float(np.clip(genome.get("hard_loss_eur", self.C.hard_loss_eur), 10.0, 100.0))
        self.C.trail_pct = float(np.clip(genome.get("trail_pct", self.C.trail_pct), 0.05, 0.3))
        self.C.trail_abs_eur = float(np.clip(genome.get("trail_abs_eur", self.C.trail_abs_eur), 5.0, 50.0))
        self.C.min_signal_threshold = float(
            np.clip(genome.get("min_signal_threshold", self.C.min_signal_threshold), 0.05, 0.5)
        )
        self.C.position_scale_threshold = float(
            np.clip(genome.get("position_scale_threshold", self.C.position_scale_threshold), 0.2, 0.8)
        )
        self.C.emergency_close_threshold = float(
            np.clip(genome.get("emergency_close_threshold", self.C.emergency_close_threshold), 0.7, 0.95)
        )
        self.C.confidence_decay = float(np.clip(genome.get("confidence_decay", self.C.confidence_decay), 0.90, 0.99))
        self.risk_multiplier = float(np.clip(genome.get("risk_multiplier", self.risk_multiplier), 0.5, 2.0))
        self.correlation_threshold = float(np.clip(genome.get("correlation_threshold", self.correlation_threshold), 0.3, 0.9))

        # update genome snapshot + keep dict mirror fresh
        self.genome = {
            "max_position_pct": self.C.max_position_pct,
            "max_consecutive_losses": self.C.max_consecutive_losses,
            "loss_reduction": self.C.loss_reduction,
            "max_instrument_concentration": self.C.max_instrument_concentration,
            "min_volatility": self.C.min_volatility,
            "hard_loss_eur": self.C.hard_loss_eur,
            "trail_pct": self.C.trail_pct,
            "trail_abs_eur": self.C.trail_abs_eur,
            "min_signal_threshold": self.C.min_signal_threshold,
            "position_scale_threshold": self.C.position_scale_threshold,
            "emergency_close_threshold": self.C.emergency_close_threshold,
            "confidence_decay": self.C.confidence_decay,
            "risk_multiplier": self.risk_multiplier,
            "correlation_threshold": self.correlation_threshold,
        }
        self.config.update(self.C.__dict__)

    def mutate(self, mutation_rate: float = 0.2) -> None:
        """Enhanced mutation with correct genome keys."""
        # start from current genome snapshot
        g = self.genome.copy() or {
            "max_position_pct": self.C.max_position_pct,
            "max_consecutive_losses": self.C.max_consecutive_losses,
            "loss_reduction": self.C.loss_reduction,
            "min_signal_threshold": self.C.min_signal_threshold,
            "hard_loss_eur": self.C.hard_loss_eur,
            "trail_pct": self.C.trail_pct,
            "trail_abs_eur": self.C.trail_abs_eur,
        }
        mutations: List[str] = []

        if np.random.rand() < mutation_rate:
            old = g["max_position_pct"]
            g["max_position_pct"] = float(np.clip(old + np.random.uniform(-0.02, 0.02), 0.01, 0.25))
            mutations.append(f"max_position_pct: {old:.3f} -> {g['max_position_pct']:.3f}")

        if np.random.rand() < mutation_rate:
            old = g["max_consecutive_losses"]
            g["max_consecutive_losses"] = int(np.clip(old + np.random.choice([-1, 0, 1]), 1, 20))
            mutations.append(f"max_consecutive_losses: {old} -> {g['max_consecutive_losses']}")

        if np.random.rand() < mutation_rate:
            old = g["loss_reduction"]
            g["loss_reduction"] = float(np.clip(old + np.random.uniform(-0.1, 0.1), 0.05, 1.0))
            mutations.append(f"loss_reduction: {old:.2f} -> {g['loss_reduction']:.2f}")

        if np.random.rand() < mutation_rate:
            old = g["min_signal_threshold"]
            g["min_signal_threshold"] = float(np.clip(old + np.random.uniform(-0.05, 0.05), 0.05, 0.5))
            mutations.append(f"min_signal_threshold: {old:.2f} -> {g['min_signal_threshold']:.2f}")

        if np.random.rand() < mutation_rate:
            old = g["hard_loss_eur"]
            g["hard_loss_eur"] = float(np.clip(old + np.random.uniform(-5, 5), 10.0, 100.0))
            mutations.append(f"hard_loss_eur: €{old:.0f} -> €{g['hard_loss_eur']:.0f}")

        if np.random.rand() < mutation_rate:
            old = g["trail_pct"]
            g["trail_pct"] = float(np.clip(old + np.random.uniform(-0.02, 0.02), 0.05, 0.30))
            mutations.append(f"trail_pct: {old:.2f} -> {g['trail_pct']:.2f}")

        if np.random.rand() < mutation_rate:
            old = g["trail_abs_eur"]
            g["trail_abs_eur"] = float(np.clip(old + np.random.uniform(-5, 5), 5.0, 50.0))
            mutations.append(f"trail_abs_eur: €{old:.0f} -> €{g['trail_abs_eur']:.0f}")

        if mutations:
            self.logger.info(format_operator_message("🧬", "MUTATION_APPLIED", changes=", ".join(mutations)))

        self.set_genome(g)

    def crossover(self, other: "PositionManager") -> "PositionManager":
        """Enhanced crossover with performance-based selection"""
        if not isinstance(other, PositionManager):
            self.logger.warning("Crossover with incompatible type")
            return self

        # Performance-based crossover
        self_performance = getattr(self, "_portfolio_health_score", 0.5)
        other_performance = getattr(other, "_portfolio_health_score", 0.5)

        # Favor higher performance parent
        bias = 0.7 if self_performance > other_performance else 0.3

        new_g = {k: (self.genome[k] if np.random.rand() < bias else getattr(other, "genome", {}).get(k, v)) for k, v in self.genome.items()}

        child = PositionManager(**{"config": self.C, "instruments": getattr(self, "instruments", []), "genome": new_g})

        # Inherit beneficial state from better parent
        if self_performance > other_performance:
            self_signal_history = getattr(self, "signal_history", {})
            if self_signal_history:
                setattr(child, "signal_history", copy.deepcopy(self_signal_history))
        else:
            other_signal_history = getattr(other, "signal_history", {})
            if other_signal_history:
                setattr(child, "signal_history", copy.deepcopy(other_signal_history))

        return child

    # ================================================================
    # API AND INTERFACE METHODS
    # ================================================================
    def force_action(self, value: float) -> None:
        """Force a specific action value for testing/debugging"""
        self._forced_action = float(value)

    def force_confidence(self, value: float) -> None:
        """Force a specific confidence value for testing/debugging"""
        self._forced_conf = float(value)

    def clear_forced(self) -> None:
        """Clear any forced values"""
        self._forced_action = None
        self._forced_conf = None

    def get_last_rationale(self) -> Dict[str, Any]:
        """Get rationale from last decision"""
        rationales: Dict[str, Any] = {}
        for inst, decision in self.last_decisions.items():
            rationales[inst] = decision.rationale
        return rationales

    def get_full_audit(self) -> Dict[str, Any]:
        """Get comprehensive audit information"""
        return {
            "positions": copy.deepcopy(self.open_positions),
            "last_decisions": {
                k: {
                    "decision": v.decision.value,
                    "intensity": v.intensity,
                    "confidence": v.confidence,
                    "rationale": v.rationale,
                    "risk_factors": v.risk_factors,
                }
                for k, v in self.last_decisions.items()
            },
            "position_confidence": copy.deepcopy(self.position_confidence),
            "consecutive_losses": self.consecutive_losses,
            "adaptive_params": copy.deepcopy(self._adaptive_params),
            "performance_metrics": {
                "portfolio_health": self._portfolio_health_score,
                "total_exposure": self._total_exposure_ratio,
                "decision_quality": self._decision_quality_score,
                "risk_management": self._risk_management_score,
            },
            "signal_history_summary": {
                k: {
                    "length": len(v),
                    "recent_avg": float(np.mean(v[-5:])) if len(v) >= 5 else 0.0,
                    "recent_std": float(np.std(v[-5:])) if len(v) >= 5 else 0.0,
                }
                for k, v in self.signal_history.items()
            },
            "genome": self.genome.copy(),
            "circuit_breaker": self.circuit_breaker.copy(),
        }

    def get_position_manager_report(self) -> str:
        """Generate operator-friendly position manager report (fixed keys)."""
        # Portfolio status
        if self._portfolio_health_score > 0.8:
            portfolio_status = "[ROCKET] Excellent"
        elif self._portfolio_health_score > 0.6:
            portfolio_status = "[OK] Good"
        elif self._portfolio_health_score > 0.4:
            portfolio_status = "[FAST] Fair"
        else:
            portfolio_status = "[WARN] Poor"

        # Risk status
        if self.consecutive_losses == 0:
            risk_status = "[GREEN] Safe"
        elif self.consecutive_losses < self.C.max_consecutive_losses // 2:
            risk_status = "[YELLOW] Caution"
        else:
            risk_status = "[RED] High Risk"

        active_decisions = len([d for d in self.last_decisions.values() if d.decision != PositionDecision.HOLD])

        recent_avg_confidence = 0.0
        if self._decision_history:
            recent_decisions = list(self._decision_history)[-5:]
            all_conf: List[float] = []
            for record in recent_decisions:
                for decision_data in record.get("decisions", {}).values():
                    if decision_data.get("decision") != "hold":
                        all_conf.append(float(decision_data.get("confidence", 0.0)))
            recent_avg_confidence = float(np.mean(all_conf)) if all_conf else 0.0

        dyn_max = float(self._adaptive_params.get("dynamic_max_pct", self.C.max_position_pct))

        return f"""
[STATS] ENHANCED POSITION MANAGER
=======================================
💼 Portfolio: {portfolio_status} ({self._portfolio_health_score:.3f})
[WARN] Risk Status: {risk_status}
[CHART] Exposure: {self._total_exposure_ratio:.1%}
📍 Open Positions: {len(self.open_positions)}

[TOOL] RISK PARAMETERS
* Max Position %: {self.C.max_position_pct:.1%} (dynamic: {dyn_max:.1%})
* Hard Loss Limit: €{self.C.hard_loss_eur:.0f}
* Trail Stop: {self.C.trail_pct:.1%} / €{self.C.trail_abs_eur:.0f}
* Signal Threshold: {self.C.min_signal_threshold:.2f}
* Consecutive Losses: {self.consecutive_losses}/{self.C.max_consecutive_losses}

[STATS] PERFORMANCE METRICS
* Decision Quality: {self._decision_quality_score:.3f}
* Risk Management: {self._risk_management_score:.3f}
* Signal Quality: {self._assess_signal_quality():.3f}
* Recent Confidence: {recent_avg_confidence:.3f}

[TARGET] ADAPTIVE PARAMETERS
* Signal Sensitivity: {self._adaptive_params['signal_sensitivity']:.2f}
* Risk Tolerance: {self._adaptive_params['risk_tolerance']:.2f}
* Confidence Threshold: {self._adaptive_params['confidence_threshold']:.2f}

💡 RECENT ACTIVITY
* Active Decisions: {active_decisions}
* Decision History: {len(self._decision_history)} records
* Portfolio Health Trend: {len([h for h in self._portfolio_health_history if h['health_score'] > 0.7])} good periods
* Circuit Breaker: {self.circuit_breaker['state']}

[RELOAD] INSTRUMENTS ({len(self.instruments)})
{chr(10).join([f"* {inst}: {len(self.signal_history.get(inst, []))} signals, confidence: {self.position_confidence.get(inst, 0.5):.2f}" for inst in self.instruments[:5]])}
        """

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with position metrics (fixed keys)."""
        try:
            portfolio_health = self._portfolio_health_score
            total_exposure = self._total_exposure_ratio
            decision_quality = self._decision_quality_score

            position_count = len(self.open_positions)
            avg_position_confidence = (
                float(np.mean(list(self.position_confidence.values()))) if self.position_confidence else 0.5
            )

            consecutive_losses_ratio = self.consecutive_losses / max(self.C.max_consecutive_losses, 1)
            risk_management_score = self._risk_management_score

            dynamic_risk_ratio = float(self._adaptive_params.get("dynamic_max_pct", self.C.max_position_pct)) / max(
                self.C.max_position_pct, 1e-9
            )
            signal_sensitivity = float(self._adaptive_params["signal_sensitivity"])

            recent_decision_count = 0.0
            if self._decision_history:
                recent = list(self._decision_history)[-5:]
                if recent:
                    recent_decision_count = sum(
                        len([d for d in record.get("decisions", {}).values() if d.get("decision") != "hold"])
                        for record in recent
                    ) / float(len(recent))

            balance = self.C.initial_balance
            drawdown = 0.0
            portfolio_metrics = self.smart_bus.get("portfolio_metrics", "PositionManager")
            if portfolio_metrics:
                balance = float(portfolio_metrics.get("balance", balance))
                drawdown = float(portfolio_metrics.get("drawdown", 0.0))
            elif self.env:
                balance = float(getattr(self.env, "balance", self.C.initial_balance))
                drawdown = float(getattr(self.env, "current_drawdown", 0.0))

            balance_ratio = balance / max(self.C.initial_balance, 1e-9)

            observation = np.array(
                [
                    float(portfolio_health),
                    float(total_exposure),
                    float(decision_quality),
                    float(position_count) / 10.0,
                    float(avg_position_confidence),
                    float(consecutive_losses_ratio),
                    float(risk_management_score),
                    float(dynamic_risk_ratio),
                    float(signal_sensitivity),
                    float(recent_decision_count) / 5.0,
                    float(balance_ratio),
                    float(drawdown),
                ],
                dtype=np.float32,
            )

            return observation

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Observation generation failed: {e}")
            return np.zeros(12, dtype=np.float32)

    # ─────────────────────────────────────────────────────────
    # Action proposal (actor interface)
    # ─────────────────────────────────────────────────────────
    async def propose_action(self, **inputs: Any) -> Dict[str, Any]:
        """
        Propose trading actions based on independent signal interpretation.

        Uses the same merge path as `process()` so intensity never disappears
        when SmartBus lacks / returns None.
        """
        try:
            obs = inputs.get("obs", None)  # reserved for future use

            if self._forced_action is not None:
                action_array = np.array([self._forced_action] * len(self.instruments) * 2, dtype=np.float32)
                return {
                    "action_type": "position_management",
                    "action_array": action_array.tolist(),
                    "forced": True,
                    "confidence": self._forced_conf or 0.8,
                }

            # Build market snapshot exactly like process()
            market_from_inputs = self._extract_market_data_from_inputs(inputs or {})
            bus_snapshot = self._extract_market_data_from_smartbus() or {}
            market_data = self._merge_market_maps(bus_snapshot, market_from_inputs)

            if not market_data or not any(market_data.get(i) for i in self.instruments):
                action_array = np.zeros(len(self.instruments) * 2, dtype=np.float32)
                return {
                    "action_type": "position_management",
                    "action_array": action_array.tolist(),
                    "fallback_reason": "no_market_data",
                    "confidence": 0.1,
                }

            # Process signals through hierarchical decision making
            decisions = await _maybe_await(self.process_market_signals(market_data))

            # Convert decisions to action signals
            action_details: List[Dict[str, Any]] = []
            signals: List[float] = []

            for inst in self.instruments:
                decision_result = decisions.get(inst, None)

                if decision_result is None:
                    intensity_val = 0.0
                    duration = 1.0
                    decision_name = "hold"
                    conf_val = 0.5
                else:
                    intensity_val = self._map_decision_to_intensity(decision_result)
                    duration = 1.0
                    decision_name = decision_result.decision.value
                    conf_val = float(decision_result.confidence)
                    self.position_confidence[inst] = conf_val

                action_details.append(
                    {
                        "instrument": inst,
                        "intensity": float(intensity_val),
                        "duration": float(duration),
                        "decision": decision_name,
                        "confidence": float(conf_val),
                    }
                )

                self.logger.info(
                    format_operator_message(
                        "📡",
                        "ACTION_PROPOSAL",
                        instrument=inst,
                        intensity=f"{float(intensity_val):.3f}",
                        duration=f"{float(duration):.1f}",
                        decision=decision_name,
                    )
                )
                self._flush_logs()

                signals.extend([float(intensity_val), float(duration)])

            action_array = np.array(signals, dtype=np.float32)
            overall_confidence = self.confidence() if hasattr(self, "confidence") else 0.5

            return {
                "action_type": "position_management",
                "action_array": action_array.tolist(),
                "action_details": action_details,
                "confidence": float(overall_confidence),
                "market_data_available": True,
                "decisions_count": len(decisions),
                "timestamp": datetime.datetime.now().isoformat(),
            }

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Action proposal failed: {e}")
            action_array = np.zeros(len(self.instruments) * 2, dtype=np.float32)
            return {
                "action_type": "position_management",
                "action_array": action_array.tolist(),
                "error": str(e),
                "confidence": 0.1,
            }


    async def calculate_confidence(self, action: Dict[str, Any], **inputs: Any) -> float:
        """Calculate confidence in position management decisions"""
        try:
            # Use the existing confidence calculation logic
            if self._forced_conf is not None:
                return float(self._forced_conf)

            # Calculate confidence components
            confidence_components: List[float] = []

            # Portfolio health confidence
            confidence_components.append(float(self._portfolio_health_score))

            # Position-specific confidence
            if self.position_confidence:
                avg_position_conf = float(np.mean(list(self.position_confidence.values())))
                confidence_components.append(avg_position_conf)
            else:
                confidence_components.append(0.5)  # Neutral when no positions

            # Signal quality confidence
            signal_quality = float(self._assess_signal_quality())
            confidence_components.append(signal_quality)

            # Risk management confidence
            confidence_components.append(float(self._risk_management_score))

            # Circuit breaker confidence
            circuit_confidence = 1.0 if self.circuit_breaker["state"] == "CLOSED" else 0.2
            confidence_components.append(float(circuit_confidence))

            # Decision quality confidence
            confidence_components.append(float(self._decision_quality_score))

            # Calculate weighted average
            weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]  # Portfolio, positions, signals, risk, circuit, decisions
            final_confidence = float(np.average(confidence_components, weights=weights))

            return float(np.clip(final_confidence, 0.1, 1.0))

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence

    # Legacy method for backward compatibility
    def propose_action_legacy(self, obs: Any = None) -> np.ndarray:  # noqa: D401
        """Legacy propose_action method that returns numpy array"""
        try:
            # Try to run async method
            result = asyncio.run(self.propose_action(obs=obs))
            return np.array(result.get("action_array", []), dtype=np.float32)
        except RuntimeError:
            # If we're already in an event loop, create a task instead
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.propose_action(obs=obs))
                # For this legacy method, we'll provide a simple fallback
                return np.zeros(len(self.instruments) * 2, dtype=np.float32)
            except Exception:
                # Final fallback
                return np.zeros(len(self.instruments) * 2, dtype=np.float32)

    def _map_decision_to_intensity(self, decision_result: PositionDecisionResult) -> float:
        """Map position decision to trading intensity (signed)."""
        decision = decision_result.decision
        base_intensity = float(decision_result.intensity)

        if decision == PositionDecision.HOLD:
            return 0.0
        elif decision == PositionDecision.OPEN_LONG:
            return base_intensity
        elif decision == PositionDecision.OPEN_SHORT:
            return -base_intensity
        elif decision == PositionDecision.SCALE_UP:
            # For scaling, use moderate intensity
            return base_intensity * 0.6
        elif decision == PositionDecision.SCALE_DOWN:
            return -base_intensity * 0.4
        elif decision in [PositionDecision.CLOSE, PositionDecision.EMERGENCY_CLOSE]:
            # For closes, use opposite of current position direction
            return -base_intensity * 0.8

        return 0.0

    def confidence(self, obs: Any = None) -> float:  # noqa: D401
        """Calculate overall confidence based on portfolio and position health."""
        if self._forced_conf is not None:
            return float(self._forced_conf)

        # Calculate confidence components
        confidence_components: List[float] = []

        # Portfolio health confidence
        confidence_components.append(float(self._portfolio_health_score))

        # Position-specific confidence
        if self.position_confidence:
            avg_position_conf = float(np.mean(list(self.position_confidence.values())))
            confidence_components.append(avg_position_conf)
        else:
            confidence_components.append(0.5)  # Neutral when no positions

        # Signal quality confidence
        signal_quality = float(self._assess_signal_quality())
        confidence_components.append(signal_quality)

        # Risk management confidence
        confidence_components.append(float(self._risk_management_score))

        # Circuit breaker confidence
        circuit_confidence = 1.0 if self.circuit_breaker["state"] == "CLOSED" else 0.2
        confidence_components.append(float(circuit_confidence))

        # Decision quality confidence
        confidence_components.append(float(self._decision_quality_score))

        # Calculate weighted average
        weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]  # Portfolio, positions, signals, risk, circuit, decisions
        final_confidence = float(np.average(confidence_components, weights=weights))

        return float(np.clip(final_confidence, 0.1, 1.0))

    # Backward compatibility
    def step(self, **kwargs: Any) -> Dict[str, Any]:
        """Backward compatibility step method"""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.process(**kwargs))
        finally:
            loop.close()
        return result
