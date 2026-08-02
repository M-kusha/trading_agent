# -------------------------------------------------------------
# File: modules/position/position_base.py
# PositionManagerBase — infrastructure, logging, safety gates,
# P&L peak tracking, and SmartInfoBus I/O utilities.
#
# Part 2 will provide the subclass "PositionManager" with the
# decision logic, sizing, and profit-take / loss-cut application.
# -------------------------------------------------------------

from __future__ import annotations

import asyncio
import copy
import datetime as _dt
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Awaitable, Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np

from envs.core.config import TradingConfig

# Core infra
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.core.mixins import (
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
    SmartInfoBusTradingMixin,
)
from modules.core.module_base import BaseModule
from modules.monitoring.performance_tracker import PerformanceTracker
from modules.utils.audit_utils import AuditConfiguration, RotatingLogger, format_operator_message
from modules.utils.info_bus import InfoBusManager
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities

# Debug system
from .position_debug import DebugLevel, PositionDebugSystem
from .position_logger import UnifiedPositionLogger

# ===============================
# Debug / decision scaffolding
# ===============================

class ActionType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SCALE_UP = "ADD_MORE"
    SCALE_DOWN = "REDUCE"
    CLOSE_POSITION = "CLOSE"
    EMERGENCY_EXIT = "EMERGENCY"


class PositionDecision(Enum):
    HOLD = "hold"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CLOSE = "close"
    EMERGENCY_CLOSE = "emergency_close"


@dataclass
class DebugSnapshot:
    timestamp: str
    instrument: str
    action: str
    is_buying: bool
    size_eur: float
    confidence: float
    signal_strength: float
    volatility: float
    portfolio_health: float
    risk_score: float
    plain_english_reason: str
    will_execute: bool
    execution_blocked_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===============================
# Async helper
# ===============================
T_co = TypeVar("T_co")


async def _maybe_await(x: Union[Awaitable[T_co], T_co]) -> T_co:
    if asyncio.iscoroutine(x):
        return cast(T_co, await x)
    return cast(T_co, x)


# ===============================
# Decision payloads
# ===============================
@dataclass
class SignalContext:
    instrument: str
    market_intensity: float = 0.0
    market_direction: int = 0  # -1, 0, 1
    volatility: float = 0.02
    trend_strength: float = 0.0
    momentum: float = 0.0
    volume_profile: float = 1.0
    correlation_penalty: float = 0.0
    regime: str = "normal"  # normal, volatile, trending, ranging
    liquidity_score: float = 1.0
    session: str = "unknown"
    current_exposure: float = 0.0
    drawdown: float = 0.0
    balance: float = 1000.0
    current_price: float = 0.0
    step_idx: int = 0
    timestamp: str = ""


@dataclass
class PositionDecisionResult:
    decision: PositionDecision
    intensity: float
    size: float
    confidence: float
    rationale: Dict[str, Any]
    risk_factors: Dict[str, float]
    context: SignalContext


# ===============================================================
# Shared logger to keep file handlers stable across instances
# ===============================================================
_PM_SHARED_LOGGER: Optional[RotatingLogger] = None


# ===============================================================
# Profit / loss peak tracking for trailing take-profit
# ===============================================================
class ProfitTracker:
    """Tracks per-instrument running P&L peaks for trailing profit logic."""

    def __init__(self) -> None:
        self._peak: Dict[str, float] = defaultdict(float)
        self._last_pnl: Dict[str, float] = defaultdict(float)

    def update(self, instrument: str, current_unrealized_eur: float) -> None:
        self._last_pnl[instrument] = float(current_unrealized_eur)
        if current_unrealized_eur > self._peak[instrument]:
            self._peak[instrument] = float(current_unrealized_eur)

    def peak(self, instrument: str) -> float:
        return float(self._peak[instrument])

    def last(self, instrument: str) -> float:
        return float(self._last_pnl[instrument])

    def reset(self, instrument: str) -> None:
        """
        Reset peak tracking for an instrument when position is closed.
        CRITICAL: Must be called when a position closes to prevent stale peaks
        from affecting new positions on the same instrument.
        """
        self._peak[instrument] = 0.0
        self._last_pnl[instrument] = 0.0

    def should_trail_close(
        self,
        instrument: str,
        trailing_pct: float,
        min_activation_eur: float,
        favors_down: bool,
    ) -> bool:
        peak = self._peak[instrument]
        cur = self._last_pnl[instrument]

        # Require positive, meaningful peak profit
        if peak < max(0.0, min_activation_eur):
            return False
        if peak <= 0.0:
            return False

        draw = peak - cur
        drop_pct = draw / peak

        # Require both a meaningful retrace and degrading favorability
        return (drop_pct >= max(0.0, trailing_pct)) and bool(favors_down)


# ===============================================================
# Base Manager (no decorator). Subclass & decorate in Part 2.
# ===============================================================
class PositionManagerBase(
    BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
):
    """
    Base layer: SmartBus I/O, logging, safety gates, profit-tracking, and
    generic process() flow that calls abstract decision hooks provided in Part 2.
    """

    # Fields that should NOT be passed to TradingConfig.__init__()
    # These are internal fields with init=False or env-specific fields
    _CONFIG_NON_INIT_FIELDS = frozenset({
        "_risk_policy_loaded",
        "max_steps_per_episode",
    })

    @classmethod
    def _sanitize_config_dict(cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Remove fields that are not valid __init__ parameters for TradingConfig."""
        return {k: v for k, v in cfg.items() if k not in cls._CONFIG_NON_INIT_FIELDS}

    # ---------- lifecycle
    def __init__(
        self,
        config: Optional[TradingConfig | Dict[str, Any]] = None,
        instruments: Optional[List[str]] = None,
        genome: Optional[Dict[str, Any]] = None,
        enable_debug: bool = True,
        debug_log_dir: str = "logs/debug",
        **kwargs: Any,
    ):
        # Debugger: heavy output only in files, not on console
        debug_verbosity = DebugLevel.CRITICAL  # Only log critical errors by default
        self.debugger = PositionDebugSystem(
            log_dir=debug_log_dir,
            enable=enable_debug,
            verbosity=debug_verbosity,
            console_output=False,       # No console spam
            file_output=enable_debug,   # CSV/JSON output only
        )

        self._instruments_forced = instruments is not None
        self.instruments = instruments or ["XAUUSD"]
        self.genome = genome or {}
        self.env = None

        # Base module init (may set metadata, etc.)
        super().__init__()

        # Config wiring (robust against missing fields)
        if isinstance(config, TradingConfig):
            self.C: TradingConfig = config
        elif isinstance(config, dict):
            self.C = TradingConfig(**self._sanitize_config_dict(config))
        else:
            self.C = TradingConfig()

        self.config: Dict[str, Any] = dict(self.C.__dict__)
        self.default_max_pct = self.Cval("max_position_pct", 0.10)

        # Runtime toggles
        self.enable_legacy_bus_signal_probe = bool(
            self.config.get("enable_legacy_bus_signal_probe", False)
        )
        raw_bus_signal_flag = self.config.get("use_bus_instrument_signals")
        # Default: True unless explicitly disabled
        self.use_bus_instrument_signals = (
            True if raw_bus_signal_flag is None else bool(raw_bus_signal_flag)
        )
        self.debug = bool(
            self.config.get("debug", False) or self.config.get("debug_decisions", False)
        )

        # Robust systems & runtime state
        self._initialize_advanced_systems()
        self._sync_from_bus_env()
        self._initialize_genome_parameters(genome)
        self._initialize_position_state()
        self._initialize_position_tracking()
        self._start_monitoring()

        # One-time init logging (debounced)
        if not getattr(self, "_init_logged", False):
            bal, _ = self._read_balance_and_drawdown()
            self.logger.info(
                format_operator_message(
                    "INIT",
                    "POSITION_MANAGER_INITIALIZED",
                    instruments_count=len(self.instruments),
                    initial_balance=f"EUR {bal:,.0f}",
                    max_position_pct=f"{self.Cval('max_position_pct', 0.10):.1%}",
                    details=(
                        "Decider-only with Integrated Debugging"
                        if enable_debug
                        else "Decider-only"
                    ),
                )
            )
            self._flush_logs()
            self._init_logged = True

    def _initialize(self, **kwargs: Any) -> None:
        """Module-system friendly re-init, with debounced logs and state preservation."""
        cfg_in = kwargs.get("config", None)

        if isinstance(cfg_in, TradingConfig):
            self.C = cfg_in
        elif isinstance(cfg_in, dict):
            self.C = TradingConfig(**self._sanitize_config_dict(cfg_in))
        else:
            self.C = getattr(self, "C", TradingConfig())

        self.config = dict(self.C.__dict__)
        self.default_max_pct = self.Cval("max_position_pct", 0.10)

        self.enable_legacy_bus_signal_probe = bool(
            self.config.get("enable_legacy_bus_signal_probe", False)
        )
        raw_bus_signal_flag = self.config.get("use_bus_instrument_signals")
        self.use_bus_instrument_signals = (
            True if raw_bus_signal_flag is None else bool(raw_bus_signal_flag)
        )
        self.debug = bool(
            self.config.get("debug", False) or self.config.get("debug_decisions", False)
        )

        instruments = kwargs.get("instruments", None)
        if instruments is not None:
            self._instruments_forced = True
            self.instruments = instruments or ["XAUUSD"]

        self.genome = kwargs.get("genome", None) or self.genome or {}
        self.env = kwargs.get("env", None) or self.env

        self._initialize_advanced_systems()
        self._sync_from_bus_env()
        self._initialize_genome_parameters(self.genome)
        # Preserve analytics where possible
        self._initialize_position_state(reset_hist_only=True)
        self._initialize_position_tracking(recreate=False)

        # Re-init banner
        bal, _ = self._read_balance_and_drawdown()
        self.logger.info(
            format_operator_message(
                "[INIT]",
                "POSITION_MANAGER_REINITIALIZED",
                instruments_count=len(self.instruments),
                initial_balance=f"EUR {bal:,.0f}",
                max_position_pct=f"{self.Cval('max_position_pct', 0.10):.1%}",
                details="Clean decider mode + Debugger",
            )
        )

    # ---------- config helper
    def Cval(self, key: str, default: Any) -> Any:
        """Safe config accessor with sane fallbacks."""
        try:
            v = getattr(self.C, key)
            return v if v is not None else default
        except Exception:
            return default

    # ---------- systems
    def _initialize_advanced_systems(self) -> None:
        self.smart_bus = InfoBusManager.get_instance()
        
        # FIX: Publish default order_queue immediately to prevent BUS MISS
        try:
            existing = self.smart_bus.get("order_queue", "PositionManager", default=None)
            if existing is None:
                self.smart_bus.set(
                    "order_queue",
                    [],
                    module="PositionManager",
                    thesis="Order queue initialized (empty)"
                )
        except Exception:
            pass

        global _PM_SHARED_LOGGER
        if _PM_SHARED_LOGGER is None:
            cfg = AuditConfiguration(
                log_level="DEBUG",
                async_logging=False,
                flush_interval_seconds=1,
                buffer_size=200,
                info_bus_integration=True,
                publish_to_bus=True,
            )
            _PM_SHARED_LOGGER = RotatingLogger(
                name="PositionManager",
                log_path="logs/position/position.log",
                max_lines=1_000_000,
                operator_mode=True,
                plain_english=True,
                info_bus_aware=True,
                config=cfg,
            )
        self.logger = _PM_SHARED_LOGGER

        # Ensure the debugger mirrors into the shared logger if available
        try:
            if hasattr(self, "debugger") and self.debugger:
                attach_fn = getattr(self.debugger, "attach_shared_logger", None)
                if callable(attach_fn):
                    self.debugger.attach_shared_logger(self.logger)
        except Exception:
            pass

        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PositionManager", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Unified logger for clean, structured logs
        self.unified_logger = UnifiedPositionLogger(self.logger, self.smart_bus)

        # Simple circuit breaker scaffold for future use
        self.circuit_breaker = {
            "failures": 0,
            "last_failure": 0,
            "state": "CLOSED",
            "threshold": self.Cval("position_circuit_breaker_threshold", 5),
        }

        self._profit_tracker = ProfitTracker()
        self._scale_cooldown_until: Dict[str, float] = defaultdict(float)

    def _start_monitoring(self) -> None:
        """Background loop that periodically updates portfolio health."""
        if getattr(self, "_monitoring_active", False):
            return

        def monitoring_loop() -> None:
            try:
                while getattr(self, "_monitoring_active", True):
                    try:
                        self._update_position_health()
                    except Exception as inner_e:
                        self.logger.warning(f"Position monitoring error: {inner_e}")
                    time.sleep(30)
            except Exception as e:
                # Last-resort guard; monitoring should never crash the process
                self.logger.error(f"Monitoring loop failure: {e}")

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]) -> None:
        if genome:
            for key, value in genome.items():
                if hasattr(self.C, key):
                    setattr(self.C, key, value)
            self.config.update(self.C.__dict__)

        self.genome = genome or {}
        self.risk_multiplier = float(self.genome.get("risk_multiplier", 1.0))
        self.correlation_threshold = float(self.genome.get("correlation_threshold", 0.7))
        self.default_max_pct = self.Cval("max_position_pct", 0.10)

    def _initialize_position_state(self, reset_hist_only: bool = False) -> None:
        if not reset_hist_only:
            self.consecutive_losses = 0
            # Track consecutive scale-downs per instrument
            self.consecutive_scale_downs: Dict[str, int] = defaultdict(int)
            self.open_positions: Dict[str, Dict[str, Any]] = {}
            # Trade cooldown to prevent rapid-fire trading
            self._last_new_position_time: Dict[str, float] = {}
            self._trade_cooldown_seconds = float(
                self.config.get("trade_cooldown_seconds", 60.0)
            )
            # Startup grace period: skip signal-based exits until signals stabilize
            self._process_call_count: int = 0
            self._startup_grace_calls: int = 3  # Same as SmartPositionManager

        self._decision_history = deque(maxlen=100)
        self._portfolio_health_history = deque(maxlen=50)
        self._exposure_history = deque(maxlen=100)
        self._performance_analytics = defaultdict(list)
        self.last_decisions: Dict[str, PositionDecisionResult] = {}
        self.position_confidence: Dict[str, float] = {}
        self.signal_history: Dict[str, List[float]] = {
            inst: [] for inst in self.instruments
        }
        self._portfolio_health_score = 1.0
        self._total_exposure_ratio = 0.0
        self._decision_quality_score = 0.5
        self._risk_management_score = 1.0
        self._adaptive_params = {
            "dynamic_max_pct": self.Cval("max_position_pct", 0.10),
            "signal_sensitivity": 1.0,
            "risk_tolerance": 1.0,
            "confidence_threshold": 0.5,
        }
        self._forced_action = None
        self._forced_conf = None
        self._last_sync_time: Optional[_dt.datetime] = None

    def _initialize_position_tracking(self, recreate: bool = True) -> None:
        if recreate:
            self._position_metadata: Dict[str, Dict[str, Any]] = {}
            self._position_performance: Dict[str, Dict[str, Any]] = {}
            self._exit_signals: Dict[str, List[Dict[str, Any]]] = {}

    # ---------- bus execution interface (hook)
    def _publish_bus_feeds(
        self,
        balance: float,
        equity: float,
        current_pnl: float,
        trades: Optional[List[Dict[str, Any]]] = None,
        execution_data: Optional[Dict[str, Any]] = None,
        order_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Hook for external infra (e.g. dashboards, observers).
        The base implementation is a no-op by design.
        """
        return

    def _refresh_positions_from_bus(self) -> None:
        """Synchronise open_positions with canonical bus representation."""
        try:
            pos = self.smart_bus.get("positions", "PositionManager")
            if not isinstance(pos, dict):
                return
            new_positions: Dict[str, Dict[str, Any]] = {}
            for inst, p in pos.items():
                try:
                    side_val = p.get("side", 0)
                    units = float(p.get("units", 0.0) or 0.0)
                    # Prefer explicit side, otherwise derive from units
                    side = int(np.sign(side_val if side_val != 0 else units))
                    entry = float(p.get("entry_price", 0.0) or 0.0)
                    notional = float(p.get("notional_eur", abs(units) * entry) or 0.0)
                    new_positions[str(inst)] = {
                        "side": side,
                        "units": units,
                        "price_open": entry,
                        "size": notional,
                        "open_time": p.get("open_time"),
                    }
                except Exception:
                    continue
            self.open_positions = new_positions
        except Exception:
            # As a safety net, preserve last known self.open_positions
            pass

    def _get_unrealised_pnl_from_bus(self, inst: str) -> float:
        try:
            pos = self.smart_bus.get("positions", "PositionManager")
            if isinstance(pos, dict):
                node = pos.get(inst, {})
                v = node.get("unrealized_pnl_eur", 0.0)
                return float(v if v is not None else 0.0)
        except Exception:
            pass
        return 0.0

    # ---------- env alignment
    def _sync_from_bus_env(self) -> None:
        """Align config / instruments with environment_config, with safe overrides."""
        try:
            env_cfg = self.smart_bus.get("environment_config", "PositionManager")
            if not isinstance(env_cfg, dict):
                return

            accept_override = bool(
                self.config.get("accept_env_balance_override", False)
            )
            ib = env_cfg.get("initial_balance")
            if isinstance(ib, (int, float)) and ib > 0:
                if accept_override:
                    self.C.initial_balance = float(ib)
                    self.config["initial_balance"] = float(ib)
                else:
                    try:
                        configured = float(self.C.initial_balance)
                        if (
                            abs(float(ib) - configured)
                            / max(1.0, configured)
                            > 0.05
                        ):
                            self.logger.warning(
                                format_operator_message(
                                    "[WARN]",
                                    "ENV_BALANCE_OVERRIDE_IGNORED",
                                    configured=f"EUR {configured:,.0f}",
                                    bus_value=f"EUR {float(ib):,.0f}",
                                    hint=(
                                        "Set accept_env_balance_override=true to allow"
                                    ),
                                )
                            )
                    except Exception:
                        pass

            env_insts = env_cfg.get("instruments")
            if isinstance(env_insts, list) and env_insts and not self._instruments_forced:
                self.instruments = [str(x) for x in env_insts]

            self.default_max_pct = self.Cval("max_position_pct", 0.10)
        except Exception:
            # Non-fatal; falls back to local config
            pass

    def _read_balance_and_drawdown(self) -> Tuple[float, float]:
        """Read balance and drawdown with priority: portfolio_metrics > market_state > env override."""
        balance = float(self.C.initial_balance)
        drawdown = 0.0

        try:
            market_state = self.smart_bus.get("market_state", "PositionManager")
            if isinstance(market_state, dict):
                balance = float(market_state.get("balance", balance))
                drawdown = float(market_state.get("drawdown", drawdown))

            portfolio_metrics = self.smart_bus.get(
                "portfolio_metrics", "PositionManager"
            )
            if isinstance(portfolio_metrics, dict):
                balance = float(portfolio_metrics.get("balance", balance))
                drawdown = float(
                    portfolio_metrics.get(
                        "current_drawdown",
                        portfolio_metrics.get("drawdown", drawdown),
                    )
                )

            if bool(self.config.get("accept_env_balance_override", False)):
                env_cfg = self.smart_bus.get("environment_config", "PositionManager")
                if isinstance(env_cfg, dict):
                    ib = env_cfg.get("initial_balance")
                    if isinstance(ib, (int, float)) and ib > 0:
                        balance = float(ib)
        except Exception:
            pass

        return balance, drawdown

    # ---------- orders
    def _build_order(
        self,
        instrument: str,
        side: int,
        intent: str,
        size_eur: float,
        confidence: float,
        rationale: Dict[str, Any],
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """Construct a normalized order payload understood by Env/Executor."""
        order_id = str(uuid.uuid4())
        order = {
            "id": order_id,
            "ts": _dt.datetime.utcnow().isoformat() + "Z",
            "source": "PositionManager",
            "instrument": instrument,
            "side": int(np.sign(side)),  # +1 buy / -1 sell
            "intent": intent,            # 'open' | 'scale_up' | 'scale_down' | 'close' | 'emergency_close'
            "size_eur": float(max(0.0, abs(size_eur))),
            "reduce_only": bool(reduce_only),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "rationale": rationale,
        }
        if self.debug:
            self.unified_logger.log_order_build(instrument, order)
            self._flush_logs()
        return order

    def _get_strategy_confidence_boost(self) -> Tuple[float, str]:
        """
        Read strategy module outputs and calculate a confidence boost.

        Returns:
            (boost_amount, reason) - boost is -0.10 to +0.15, reason explains why.

        Strategy signals that can boost confidence:
        - market_thesis with high confidence
        - best_thesis strong hypothesis
        - bias_analysis showing no harmful biases
        - trading_mode == "aggressive" or "normal" (boost) or "defensive" (penalty)
        """
        boost = 0.0
        reasons: List[str] = []

        try:
            # 1. Market thesis boost (from ThesisEvolutionEngine)
            market_thesis = self.smart_bus.get("market_thesis", "PositionManager")
            if isinstance(market_thesis, dict):
                thesis_conf = float(market_thesis.get("confidence", 0.0) or 0.0)
                thesis_direction = market_thesis.get("direction", "")
                if thesis_conf > 0.6 and thesis_direction in ("bullish", "bearish"):
                    boost += 0.05
                    reasons.append(f"thesis:{thesis_direction}@{thesis_conf:.0%}")

            # 2. Best thesis boost (strongest current hypothesis)
            best_thesis = self.smart_bus.get("best_thesis", "PositionManager")
            if isinstance(best_thesis, dict):
                best_conf = float(best_thesis.get("confidence", 0.0) or 0.0)
                if best_conf > 0.7:
                    boost += 0.05
                    reasons.append(f"best_thesis@{best_conf:.0%}")

            # 3. Bias analysis (from BiasAuditor) - boost if no dangerous biases
            bias_analysis = self.smart_bus.get("bias_analysis", "PositionManager")
            if isinstance(bias_analysis, dict):
                # BiasAuditor outputs 'individual_biases' dict with bias_type -> strength
                individual_biases = bias_analysis.get("individual_biases", {})
                aggregate = bias_analysis.get("aggregate_metrics", {})
                total_bias_score = aggregate.get("total_bias_score", 0.0)

                if total_bias_score > 1.0:
                    # High bias score = penalty
                    boost -= 0.05
                    reasons.append(f"bias_penalty:{total_bias_score:.2f}")
                elif total_bias_score < 0.3 and not individual_biases:
                    boost += 0.03
                    reasons.append("no_severe_bias")

            # 3b. Bias adjustments (position size multipliers from BiasAuditor)
            bias_adjustments = self.smart_bus.get("bias_adjustments", "PositionManager")
            if isinstance(bias_adjustments, dict):
                # Find the minimum adjustment (most restrictive bias)
                min_adjustment = min(bias_adjustments.values()) if bias_adjustments else 1.0
                if min_adjustment < 0.8:
                    # Significant bias detected - apply penalty
                    penalty = (1.0 - min_adjustment) * 0.1
                    boost -= penalty
                    dominant_bias = (
                        min(bias_adjustments.items(), key=lambda x: x[1])[0]
                        if bias_adjustments
                        else "unknown"
                    )
                    reasons.append(f"bias_adj:{dominant_bias}@{min_adjustment:.0%}")

            # 4. Trading mode boost (from TradingModeManager)
            trading_mode = self.smart_bus.get("trading_mode", "PositionManager")
            if isinstance(trading_mode, str):
                mode_l = trading_mode.lower()
                if mode_l in ("aggressive", "opportunity"):
                    boost += 0.02
                    reasons.append(f"mode:{trading_mode}")
                elif mode_l in ("defensive", "cautious"):
                    boost -= 0.02
                    reasons.append(f"mode_penalty:{trading_mode}")

        except Exception as e:
            if self.debug:
                self.logger.debug(f"[STRATEGY] Boost calculation error: {e}")

        # Cap the boost
        boost = max(-0.10, min(0.15, boost))
        reason = ", ".join(reasons) if reasons else "no_strategy_signals"
        return boost, reason

    def _check_voting_consensus(self) -> bool:
        """
        SIMPLIFIED RISK-ONLY CONSENSUS GATE (v3.3.0 - PPO MASTER).

        ═══════════════════════════════════════════════════════════════════
        PPO IS THE MASTER DECISION MAKER.
        Experts and voting system are ADVISORY ONLY - they do NOT block PPO.
        ═══════════════════════════════════════════════════════════════════

        This gate ONLY checks for hard safety blocks:
        1. RISK VETOES: DynamicRiskController/PortfolioRiskSystem HALT/EMERGENCY

        REMOVED: ABSTAIN check - experts cannot block PPO decisions.
        PPO makes all trading decisions. Experts provide context/signals only.
        """
        if not bool(self.config.get("require_voting_consensus", True)):
            return True

        try:
            # ═══════════════════════════════════════════════════════════════════
            # CHECK 1: RISK VETO - Hard safety gate (ONLY blocking check)
            # If any risk module (DynamicRiskController, etc.) says HALT/EMERGENCY
            # ═══════════════════════════════════════════════════════════════════
            expert_votes = self.smart_bus.get("expert_votes", "PositionManager")
            if isinstance(expert_votes, list):
                risk_voters = [
                    "DynamicRiskController",
                    "PortfolioRiskSystem",
                    "EnhancedAnomalyDetector",
                    "ExecutionQualityMonitor",
                ]
                RISK_BLOCK_ACTIONS = {"halt", "emergency", "block"}

                for vote in expert_votes:
                    if not isinstance(vote, dict):
                        continue
                    expert = vote.get("expert", "")
                    vote_obj = vote.get("vote", {}) or {}
                    if not isinstance(vote_obj, dict):
                        vote_obj = {}
                    action = str(vote_obj.get("action", "")).lower()
                    confidence = float(vote.get("confidence", 0.0) or 0.0)

                    # Only block on high-confidence risk vetoes
                    if expert in risk_voters and action in RISK_BLOCK_ACTIONS and confidence > 0.7:
                        if self.debug:
                            self.logger.debug(
                                f"[GATE] Risk veto by {expert}: action={action}, conf={confidence:.1%}"
                            )
                        return False

            # ═══════════════════════════════════════════════════════════════════
            # REMOVED: ABSTAIN check (v3.3.0)
            # PPO is MASTER - experts/voting are ADVISORY ONLY.
            # Experts cannot block PPO decisions via ABSTAIN votes.
            # ═══════════════════════════════════════════════════════════════════

            # ═══════════════════════════════════════════════════════════════════
            # CHECK 2: EXTREME FRAGILITY WARNING (advisory only - does NOT block)
            # ═══════════════════════════════════════════════════════════════════
            fragility = 0.0
            try:
                inst_frag = self.smart_bus.get("instrument_fragility", "PositionManager")
                if isinstance(inst_frag, dict) and inst_frag:
                    fragility = min(inst_frag.values())
                else:
                    fragility = float(
                        self.smart_bus.get("fragility", "PositionManager") or 0.0
                    )
            except Exception:
                pass

            if fragility >= 0.95:
                self.logger.warning(
                    f"[GATE] ⚠️ HIGH FRAGILITY WARNING: {fragility:.2f} - proceeding anyway"
                )

            # ═══════════════════════════════════════════════════════════════════
            # DEFAULT: ALLOW - PPO is master, experts are advisory only
            # ═══════════════════════════════════════════════════════════════════
            return True

        except Exception as e:
            if self.debug:
                self.logger.warning(f"[GATE] Consensus check error: {e}")
            # On error, default to allowing (fail-open for trading, not fail-closed)
            return True

    def _check_trade_cooldown(self, instrument: str, intent: str) -> bool:
        """
        Check if we are in a cooldown period for this instrument.
        Returns True if trade is ALLOWED, False if still in cooldown.

        Only applies to new position opens, not closes or scale operations.

        Uses step-based cooldown in simulation mode (fast training) or
        real-time cooldown in live mode.
        """
        if intent not in ("open", "open_long", "open_short"):
            return True  # Closes and scales are always allowed

        # Check execution mode - use step-based cooldown for training
        exec_mode = "simulation"
        try:
            exec_mode = str(
                self.smart_bus.get("execution_mode", "PositionManager")
                or "simulation"
            ).lower()
        except Exception:
            pass

        if exec_mode in ("live", "paper"):
            # Real-time cooldown for live trading
            now = time.time()
            last_trade = getattr(self, "_last_new_position_time", {}).get(
                instrument, 0.0
            )
            cooldown = getattr(self, "_trade_cooldown_seconds", 60.0)
            elapsed = now - last_trade
            if elapsed < cooldown:
                if self.debug:
                    self.logger.warning(
                        f"⏳ COOLDOWN: {instrument} - {cooldown - elapsed:.0f}s remaining "
                        f"(min {cooldown:.0f}s between new positions)"
                    )
                return False
        else:
            # Step-based cooldown for simulation/training
            current_step = 0
            try:
                current_step = int(
                    self.smart_bus.get("step_idx", "PositionManager") or 0
                )
            except Exception:
                pass

            last_trade_step = getattr(self, "_last_new_position_step", {}).get(
                instrument, -999
            )
            step_cooldown = int(self.config.get("trade_cooldown_steps", 10))

            steps_elapsed = current_step - last_trade_step
            if steps_elapsed < step_cooldown:
                if self.debug:
                    self.logger.debug(
                        f"⏳ COOLDOWN: {instrument} - "
                        f"{step_cooldown - steps_elapsed} steps remaining"
                    )
                return False

        return True

    def _record_trade_time(self, instrument: str) -> None:
        """Record when a new position was opened for cooldown tracking."""
        if not hasattr(self, "_last_new_position_time"):
            self._last_new_position_time = {}
        self._last_new_position_time[instrument] = time.time()

        # Also record step-based for simulation mode
        if not hasattr(self, "_last_new_position_step"):
            self._last_new_position_step = {}
        try:
            current_step = int(
                self.smart_bus.get("step_idx", "PositionManager") or 0
            )
            self._last_new_position_step[instrument] = current_step
        except Exception:
            self._last_new_position_step[instrument] = 0

    def _enqueue_orders(self, orders: List[Dict[str, Any]]) -> None:
        """
        Append orders to shared 'order_queue' with consensus gate,
        cooldown, and safety exceptions.
        """
        if not orders:
            return

        have_consensus = self._check_voting_consensus()
        allow_safety = bool(
            self.config.get("allow_safety_orders_without_consensus", True)
        )

        try:
            # STEP 1: Apply cooldown filter for new positions
            cooldown_filtered: List[Dict[str, Any]] = []
            for o in orders:
                intent = o.get("intent", "")
                instrument = o.get("instrument", "")
                if self._check_trade_cooldown(instrument, intent):
                    cooldown_filtered.append(o)
                else:
                    if self.debug:
                        self.logger.info(f"[COOLDOWN] Blocked {intent} on {instrument}")
            orders = cooldown_filtered

            if not orders:
                return

            # STEP 2: If no consensus, only allow reduce-only orders
            if not have_consensus:
                safe = [o for o in orders if bool(o.get("reduce_only"))]
                blocked = len(orders) - len(safe)
                if blocked > 0 and self.debug:
                    try:
                        consensus = self.smart_bus.get(
                            "committee_consensus", "PositionManager"
                        )
                        trade_vote = self.smart_bus.get(
                            "trade_vote_v2", "PositionManager"
                        )
                        strength = 0.0
                        vote_conf = 0.0
                        consensus_score = 0.0
                        if isinstance(consensus, dict):
                            strength = float(
                                consensus.get("consensus_strength", 0.0) or 0.0
                            )
                        if isinstance(trade_vote, dict):
                            vote_conf = float(
                                trade_vote.get("confidence", 0.0) or 0.0
                            )
                            consensus_score = float(
                                trade_vote.get("consensus_score", 0.0) or 0.0
                            )

                        strategy_boost, boost_reason = self._get_strategy_confidence_boost()

                        self.logger.warning(
                            f"⚠️  SMART GATE: Blocked {blocked} order(s) | "
                            f"Vote: {vote_conf:.1%}+{strategy_boost:+.1%}="
                            f"{vote_conf + strategy_boost:.1%} | "
                            f"Consensus: {consensus_score:.1%} | "
                            f"Strategy: {boost_reason} | "
                            f"Need: conf>28% + consensus>60%"
                        )
                    except Exception:
                        self.logger.warning(
                            format_operator_message(
                                "[GATE]",
                                "ORDERS_BLOCKED_NO_CONSENSUS",
                                count=blocked,
                                reason=(
                                    "Voting system has not produced consensus - "
                                    "blocking non-safety orders"
                                ),
                            )
                        )
                orders = safe if allow_safety else []
                if not orders:
                    return

            # STEP 3: Enqueue orders and record cooldown times
            existing = self.smart_bus.get("order_queue", "PositionManager")
            if not isinstance(existing, list):
                existing = []
            existing_ids = {o.get("id") for o in existing if isinstance(o, dict)}

            for o in orders:
                oid = o.get("id")
                if oid not in existing_ids:
                    existing.append(o)
                    existing_ids.add(oid)
                    # Record trade time for cooldown (only for new position opens)
                    intent = o.get("intent", "")
                    if intent in ("open", "open_long", "open_short"):
                        self._record_trade_time(o.get("instrument", ""))

            self.smart_bus.set(
                "order_queue",
                existing,
                module="PositionManager",
                thesis="Enqueued {n} order(s) for Env/Executor execution".format(
                    n=len(orders)
                ),
            )
        except Exception as e:
            self.logger.error(f"Failed to enqueue orders: {e}")

    def _is_signal_valid_for_exits(self) -> bool:
        """
        Check if signals have been calculated enough times to trust signal-based exits.
        
        This prevents closing positions on startup when signals haven't stabilized yet.
        Same logic as SmartPositionManager._startup_grace_calls.
        """
        count = getattr(self, "_process_call_count", 0)
        grace = getattr(self, "_startup_grace_calls", 3)
        return count > grace

    def _check_cooldown(self, instrument: str) -> bool:
        """
        Check if the instrument is past its trade cooldown period.
        
        Returns True if trading is allowed (cooldown expired or never traded).
        Returns False if still in cooldown (should wait before new entry).
        """
        last_trade_times = getattr(self, "_last_new_position_time", {})
        cooldown_seconds = getattr(self, "_trade_cooldown_seconds", 60.0)
        
        last_trade = last_trade_times.get(instrument)
        if last_trade is None:
            return True  # Never traded this instrument, allow
        
        elapsed = time.time() - last_trade
        return elapsed >= cooldown_seconds

    # ---------- process() — calls abstract decision pipeline provided by Part 2
    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """
        Main processing flow.

        Part 2 provides `process_market_signals()` and produces decisions.
        This layer handles:
        - Market snapshot assembly (inputs + bus)
        - Time-budgeted decision pipeline
        - Bus publishing
        - Order translation + enqueue under consensus and cooldown gates
        - Contract-shaped payload for the caller
        """
        t0 = time.time()
        
        # Track calls for startup grace period (signal-based exits skip first N calls)
        if hasattr(self, "_process_call_count"):
            self._process_call_count += 1

        # Robust metadata access: metadata is optional
        metadata = getattr(self, "metadata", None)
        try:
            budget_ms = float(
                getattr(metadata, "timeout_ms", 3000.0) if metadata is not None else 3000.0
            )
        except Exception:
            budget_ms = 3000.0
        if budget_ms <= 0:
            budget_ms = 3000.0

        try:
            self._refresh_positions_from_bus()

            # Build market snapshot (inputs + Bus)
            market_from_inputs = self._extract_market_data_from_inputs(inputs)
            bus_snapshot = self._extract_market_data_from_smartbus() or {}
            market_data: Dict[str, Any] = self._merge_market_maps(
                bus_snapshot, market_from_inputs
            )

            await asyncio.sleep(0)

            # If we have no usable data for any configured instrument, just report health and hold
            if not any(
                isinstance(market_data.get(i, {}), dict) and market_data.get(i)
                for i in self.instruments
            ):
                metrics = self._read_env_metrics()
                current_queue = self._safe_get_order_queue()
                return self._contract_payload(
                    decisions={},
                    orders_created=[],
                    processing_ms=(time.time() - t0) * 1000.0,
                    instrument_signals=self._default_signals(),
                    order_queue=current_queue,
                    thesis="No market data; holding",
                    **metrics,
                )

            # decisions from Part 2
            decisions = await _maybe_await(self.process_market_signals(market_data))
            await asyncio.sleep(0)

            elapsed_ms = (time.time() - t0) * 1000.0
            if elapsed_ms > budget_ms * 0.95:
                # Under time pressure: publish decisions to bus but skip order translation
                await self._update_smartbus_with_decisions(decisions)
                metrics = self._read_env_metrics()
                current_queue = self._safe_get_order_queue()
                return self._contract_payload(
                    decisions=decisions,
                    orders_created=[],
                    processing_ms=elapsed_ms,
                    thesis="Aborted post-processing due to time budget",
                    instrument_signals=self._signals_map_from_decisions(decisions),
                    order_queue=current_queue,
                    **metrics,
                )

            # Publish artifacts (no execution yet)
            await self._update_smartbus_with_decisions(decisions)
            await asyncio.sleep(0)

            # Translate + enqueue (with consensus gate safety)
            orders_created = self._translate_decisions_to_orders(decisions)
            self._enqueue_orders(orders_created)

            current_queue = self._safe_get_order_queue()
            thesis = await self._generate_position_thesis(market_data, decisions)
            metrics = self._read_env_metrics()

            # Flush unified logger at end of processing
            if self.debug:
                self._flush_logs()

            # History bookkeeping for adaptive logic
            self._decision_history.append(
                {
                    "ts": _dt.datetime.utcnow().isoformat() + "Z",
                    "decisions": {
                        k: {"decision": v.decision.value, "confidence": v.confidence}
                        for k, v in decisions.items()
                    },
                    "portfolio_health": self._portfolio_health_score,
                }
            )

            return self._contract_payload(
                decisions=decisions,
                orders_created=orders_created,
                processing_ms=(time.time() - t0) * 1000.0,
                thesis=thesis,
                instrument_signals=self._signals_map_from_decisions(decisions),
                order_queue=current_queue,
                **metrics,
            )

        except Exception as e:
            self.logger.error(f"Position processing failed: {e}")
            metrics = self._read_env_metrics()
            current_queue = self._safe_get_order_queue()
            return self._contract_payload(
                decisions={},
                orders_created=[],
                processing_ms=0.0,
                thesis=f"Processing error: {e}",
                instrument_signals=self._default_signals(),
                order_queue=current_queue,
                **metrics,
            )

    # ---------- abstract hooks (implemented in Part 2)
    @create_error_handler("process_market_signals")
    def process_market_signals(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, PositionDecisionResult]:
        raise NotImplementedError("Provided in Part 2 (position_logic.py)")

    def _make_position_decision(self, context: SignalContext) -> PositionDecisionResult:
        raise NotImplementedError("Provided in Part 2 (position_logic.py)")

    # ---------- translation / payload
    def _translate_decisions_to_orders(
        self, decisions: Dict[str, PositionDecisionResult]
    ) -> List[Dict[str, Any]]:
        """
        Translate per-instrument decisions into executable order payloads.

        Critical fix: SCALE_UP / SCALE_DOWN / CLOSE now respect the *existing*
        position side per instrument, so shorts are scaled/closed correctly.
        """
        orders: List[Dict[str, Any]] = []

        for inst, dr in decisions.items():
            if dr.decision == PositionDecision.HOLD:
                continue

            existing = self.open_positions.get(inst, {})
            existing_side = 0
            try:
                existing_side = int(
                    np.sign(existing.get("side", 0) or existing.get("units", 0.0) or 0.0)
                )
            except Exception:
                existing_side = 0

            # Decide intent
            intent = {
                PositionDecision.OPEN_LONG: "open",
                PositionDecision.OPEN_SHORT: "open",
                PositionDecision.SCALE_UP: "scale_up",
                PositionDecision.SCALE_DOWN: "scale_down",
                PositionDecision.CLOSE: "close",
                PositionDecision.EMERGENCY_CLOSE: "emergency_close",
            }[dr.decision]

            # Decide side with proper handling of existing positions
            if dr.decision == PositionDecision.OPEN_LONG:
                side = 1
            elif dr.decision == PositionDecision.OPEN_SHORT:
                side = -1
            elif dr.decision == PositionDecision.SCALE_UP:
                # Increase in the direction of the current position if known,
                # otherwise fall back to market_direction or default long.
                if existing_side != 0:
                    side = existing_side
                else:
                    md = int(np.sign(dr.context.market_direction or 1))
                    side = 1 if md >= 0 else -1
            elif dr.decision in (
                PositionDecision.SCALE_DOWN,
                PositionDecision.CLOSE,
                PositionDecision.EMERGENCY_CLOSE,
            ):
                # Reduce/close: trade opposite to the existing position side.
                # For CLOSE/EMERGENCY_CLOSE with unknown side: send side=0 with close_all flag
                # The executor should interpret this as "close ALL positions for this symbol"
                if existing_side != 0:
                    side = -existing_side
                elif dr.decision in (PositionDecision.CLOSE, PositionDecision.EMERGENCY_CLOSE):
                    # Unknown side but explicit CLOSE - use side=0 to signal "close all"
                    # This handles hedged positions (BUY+SELL that net to 0)
                    side = 0
                    self.logger.info(
                        f"[CLOSE_ALL] {dr.decision.value} for {inst}: "
                        f"existing_side=0, sending close_all intent to executor."
                    )
                else:
                    # SCALE_DOWN with unknown side - skip to avoid creating hedge
                    self.logger.warning(
                        f"[HEDGE_PREVENTION] Skipping {dr.decision.value} for {inst}: "
                        f"existing_side=0 (unknown). Cannot scale down unknown position safely."
                    )
                    continue  # Skip this order to prevent hedge creation
            else:
                # Fallback: should not happen, but keep safe
                side = 0

            reduce_only = dr.decision in (
                PositionDecision.SCALE_DOWN,
                PositionDecision.CLOSE,
                PositionDecision.EMERGENCY_CLOSE,
            )

            order = self._build_order(
                instrument=inst,
                side=side,
                intent=intent,
                size_eur=dr.size,
                confidence=dr.confidence,
                rationale=dr.rationale,
                reduce_only=reduce_only,
            )
            orders.append(order)

        return orders

    def _contract_payload(
        self,
        decisions: Dict[str, PositionDecisionResult],
        orders_created: List[Dict[str, Any]],
        processing_ms: float,
        thesis: Optional[str] = None,
        balance: float = 0.0,
        equity: float = 0.0,
        current_pnl: float = 0.0,
        positions_snapshot: Optional[Dict[str, Any]] = None,
        trades_snapshot: Optional[List[Dict[str, Any]]] = None,
        execution_reports: Optional[List[Dict[str, Any]]] = None,
        instrument_signals: Optional[Dict[str, Any]] = None,
        order_queue: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
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

        pos_health = {
            "portfolio_health": float(self._portfolio_health_score),
            "exposure_ratio": float(self._total_exposure_ratio),
            "risk_management_score": float(self._risk_management_score),
            "consecutive_losses": int(self.consecutive_losses),
        }

        positions_snapshot = positions_snapshot or {}
        trades_snapshot = trades_snapshot or []
        execution_reports = execution_reports or []
        instrument_signals = instrument_signals or self._default_signals()
        order_queue = order_queue or []

        return {
            "position_decisions": position_decisions,
            "portfolio_state": {
                "health_score": float(self._portfolio_health_score),
                "exposure_ratio": float(self._total_exposure_ratio),
                "open_positions": len(positions_snapshot),
                "decision_quality": float(self._decision_quality_score),
            },
            "position_health": pos_health,
            "position_analysis": {
                "instruments": list(self.instruments),
                "decisions_count": len(decisions),
                "exposure_ratio": float(self._total_exposure_ratio),
                "risk_management_score": float(self._risk_management_score),
                "thesis": thesis or "",
            },
            "positions": copy.deepcopy(positions_snapshot),
            "pending_orders": copy.deepcopy(orders_created),
            "trades": copy.deepcopy(trades_snapshot),
            "recent_trades": copy.deepcopy(trades_snapshot[-20:]),
            "current_pnl": float(current_pnl),
            "balance": float(balance),
            "equity": float(equity),
            "current_positions": copy.deepcopy(positions_snapshot),
            "execution_data": copy.deepcopy(execution_reports),
            "order_data": copy.deepcopy(orders_created),
            "instrument_signals": copy.deepcopy(instrument_signals),
            "order_queue": copy.deepcopy(order_queue),
            "position_manager_data": {
                "decisions": position_decisions,
                "health": pos_health,
                "portfolio_state": {
                    "health_score": float(self._portfolio_health_score),
                    "exposure_ratio": float(self._total_exposure_ratio),
                    "open_positions": len(positions_snapshot),
                    "decision_quality": float(self._decision_quality_score),
                },
                "positions": copy.deepcopy(positions_snapshot),
                "balance": float(balance),
                "equity": float(equity),
                "current_pnl": float(current_pnl),
            },
            "_thesis": thesis or "",
            "thesis": thesis or "",
            "processing_time_ms": float(processing_ms),
        }

    # ---------- env metrics & signals
    def _read_env_metrics(self) -> Dict[str, Any]:
        balance, drawdown = self._read_balance_and_drawdown()
        equity = balance
        current_pnl = 0.0
        positions_snapshot: Dict[str, Any] = {}
        trades_snapshot: List[Dict[str, Any]] = []
        execution_reports: List[Dict[str, Any]] = []

        try:
            pm = self.smart_bus.get("portfolio_metrics", "PositionManager")
            if isinstance(pm, dict):
                equity = float(pm.get("equity", equity))
                current_pnl = float(pm.get("current_pnl", current_pnl))
        except Exception:
            pass

        try:
            positions = self.smart_bus.get("positions", "PositionManager")
            if isinstance(positions, dict):
                positions_snapshot = positions
                # update peak P&L for trailing TP
                for inst, node in positions.items():
                    pnl = float(node.get("unrealized_pnl_eur", 0.0) or 0.0)
                    self._profit_tracker.update(inst, pnl)
        except Exception:
            pass

        try:
            trades = self.smart_bus.get("trades", "PositionManager")
            if isinstance(trades, list):
                trades_snapshot = trades
        except Exception:
            pass

        try:
            exec_r = self.smart_bus.get("execution_reports", "PositionManager")
            if isinstance(exec_r, list):
                execution_reports = exec_r
        except Exception:
            pass

        return dict(
            balance=balance,
            equity=equity,
            current_pnl=current_pnl,
            positions_snapshot=positions_snapshot,
            trades_snapshot=trades_snapshot,
            execution_reports=execution_reports,
        )

    def _safe_get_order_queue(self) -> List[Dict[str, Any]]:
        try:
            q = self.smart_bus.get("order_queue", "PositionManager")
            if isinstance(q, list):
                return q
        except Exception:
            pass
        return []

    # ---------- market extraction (inputs + bus)
    def _derive_intensity(self, inst_dict: Dict[str, Any]) -> Optional[float]:
        """Derive a signed intensity from trend/momentum/RSI when no explicit signal exists."""
        try:
            trend = float(inst_dict.get("trend_strength", 0.0))
            mom = float(inst_dict.get("momentum", 0.0))
            rsi = float(inst_dict.get("rsi", 50.0))
            vol = float(inst_dict.get("volatility", self.Cval("min_volatility", 0.015)))
            vol = max(vol, self.Cval("min_volatility", 0.015))

            trend_term = float(np.tanh(trend * 3.0))
            macd_term = float(np.tanh(mom / (vol * 50.0)))
            rsi_term = float(np.clip((rsi - 50.0) / 50.0, -1.0, 1.0))

            intensity = 0.5 * trend_term + 0.3 * macd_term + 0.2 * rsi_term
            return float(np.clip(intensity, -1.0, 1.0))
        except Exception:
            return None

    def _extract_market_data_from_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize market data from process() inputs."""
        out: Dict[str, Any] = {}

        market_data = inputs.get("market_data") or {}
        price_data = inputs.get("price_data") or {}
        tech = inputs.get("indicators") or inputs.get("technical_indicators") or {}
        vol = inputs.get("volatility_data") or {}
        simple_prices = inputs.get("prices") or {}
        regime = inputs.get("market_regime") or inputs.get(
            "market_context", {}
        ).get("volatility_regime")
        session = inputs.get("trading_session") or inputs.get(
            "market_context", {}
        ).get("session")
        liq_caps = inputs.get("liquidity_capabilities") or {}
        liq_score = inputs.get("liquidity_score", None)
        market_liquidity = inputs.get("market_liquidity") or {}

        signals_map = (
            inputs.get("instrument_signals")
            or inputs.get("kernel_instrument_signals")
            or inputs.get("arbiter_instrument_signals")
            or inputs.get("signals")
            or inputs.get("alpha_signals")
            or inputs.get("action_signals")
            or inputs.get("trading_signals")
            or {}
        )

        def variants(inst: str) -> set[str]:
            core = inst.replace("/", "").replace("_", "")
            return {
                inst,
                inst.replace("/", ""),
                inst.replace("/", "_"),
                inst.upper(),
                inst.lower(),
                core.upper(),
                core.lower(),
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
                inst_dict["current_price"] = float(
                    md.get("close", md.get("price", md.get("bid", 0.0)))
                )
                raw_vol = float(
                    md.get(
                        "atr",
                        md.get("volatility", self.Cval("min_volatility", 0.015)),
                    )
                )
                # Clamp volatility to reasonable bounds (0.1% to 100%).
                # ATR values > 1.0 are likely raw price ATR, not percentage - normalize.
                if raw_vol > 1.0 and inst_dict.get("current_price", 0) > 0:
                    raw_vol = raw_vol / inst_dict["current_price"]
                inst_dict["volatility"] = float(np.clip(raw_vol, 0.001, 1.0))

            pd = pick(price_data, inst)
            if isinstance(pd, dict):
                last = pd.get("last", pd.get("close"))
                if isinstance(last, (int, float)):
                    inst_dict["current_price"] = float(last)

            ti = pick(tech, inst)
            if isinstance(ti, dict):
                sma20 = float(ti.get("sma_20", 0.0))
                sma50 = float(ti.get("sma_50", 0.0))
                # Normalize trend_strength to [-1, 1] and handle missing SMA data
                if sma50 == 0.0:
                    inst_dict["trend_strength"] = 0.0
                else:
                    raw_trend = (sma20 - sma50) / abs(sma50)
                    inst_dict["trend_strength"] = float(
                        np.clip(raw_trend, -1.0, 1.0)
                    )
                inst_dict["momentum"] = float(ti.get("macd", 0.0))
                inst_dict["rsi"] = float(ti.get("rsi", 50.0))

            vd = pick(vol, inst)
            if isinstance(vd, dict):
                raw_vol = float(
                    vd.get(
                        "atr",
                        vd.get(
                            "volatility",
                            inst_dict.get(
                                "volatility", self.Cval("min_volatility", 0.015)
                            ),
                        ),
                    )
                )
                if raw_vol > 1.0 and inst_dict.get("current_price", 0) > 0:
                    raw_vol = raw_vol / inst_dict["current_price"]
                inst_dict["volatility"] = float(np.clip(raw_vol, 0.001, 1.0))

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

            inten = extract_intensity(pick(signals_map, inst))
            if inten is not None:
                inst_dict["intensity"] = float(np.clip(inten, -1.0, 1.0))
                inst_dict["intensity_source"] = "inputs"
            elif "intensity" not in inst_dict:
                di = self._derive_intensity(inst_dict)
                if di is not None:
                    inst_dict["intensity"] = di
                    inst_dict["intensity_source"] = "derived"

            if session:
                inst_dict["session"] = session

            if isinstance(market_liquidity, dict) and inst in market_liquidity:
                try:
                    inst_dict["liquidity_hint"] = float(
                        market_liquidity.get(inst, 1.0)
                    )
                except Exception:
                    pass

            out[inst] = inst_dict

        if regime:
            out["market_regime"] = regime
        if isinstance(liq_score, (int, float)):
            out["liquidity_score"] = float(liq_score)
        if isinstance(liq_caps, dict):
            out["liquidity_capabilities"] = liq_caps

        return out

    def _extract_market_data_from_smartbus(self) -> Optional[Dict[str, Any]]:
        """Extract and normalize market data from SmartInfoBus."""
        out: Dict[str, Any] = {}

        price_map = self.smart_bus.get("price_data", "PositionManager") or {}
        simple_prices = self.smart_bus.get("prices", "PositionManager") or {}
        tech_map = (
            self.smart_bus.get("indicators", "PositionManager")
            or self.smart_bus.get("technical_indicators", "PositionManager")
            or {}
        )
        vol_map = self.smart_bus.get("volatility_data", "PositionManager") or {}
        market_context = self.smart_bus.get("market_context", "PositionManager") or {}
        liq_caps = self.smart_bus.get("liquidity_capabilities", "PositionManager") or {}
        liq_score = self.smart_bus.get("liquidity_score", "PositionManager")
        market_liquidity = self.smart_bus.get("market_liquidity", "PositionManager") or {}

        regime = market_context.get("volatility_regime") or market_context.get("regime")
        session = market_context.get("session")

        bus_signals: Dict[str, Any] = {}
        if self.use_bus_instrument_signals:
            for key in (
                "instrument_signals",
                "kernel_instrument_signals",
                "arbiter_instrument_signals",
                "position_manager_instrument_signals",
            ):
                candidate = self.smart_bus.get(key, "PositionManager")
                if isinstance(candidate, dict) and candidate:
                    bus_signals = candidate
                    break

        def variants(inst: str) -> List[str]:
            core = inst.replace("/", "").replace("_", "")
            aliases = [
                inst,
                inst.replace("/", ""),
                inst.replace("/", "_"),
                inst.replace("_", "/"),
                inst.replace("_", ""),
                inst.upper(),
                inst.lower(),
                core.upper(),
                core.lower(),
            ]
            seen: set[str] = set()
            out_aliases: List[str] = []
            for a in aliases:
                if a not in seen:
                    out_aliases.append(a)
                    seen.add(a)
            return out_aliases

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

            ti = pick(tech_map, inst)
            if isinstance(ti, dict):
                sma20 = float(ti.get("sma_20", 0.0))
                sma50 = float(ti.get("sma_50", 0.0))
                if sma50 == 0.0:
                    inst_dict["trend_strength"] = 0.0
                else:
                    raw_trend = (sma20 - sma50) / abs(sma50)
                    inst_dict["trend_strength"] = float(
                        np.clip(raw_trend, -1.0, 1.0)
                    )
                inst_dict["momentum"] = float(ti.get("macd", 0.0))
                inst_dict["rsi"] = float(ti.get("rsi", 50.0))

            vd = pick(vol_map, inst)
            if isinstance(vd, dict):
                vol_val = vd.get(
                    "atr",
                    vd.get("volatility", self.Cval("min_volatility", 0.015)),
                )
                if isinstance(vol_val, (int, float)):
                    raw_vol = float(vol_val)
                    if raw_vol > 1.0 and inst_dict.get("current_price", 0) > 0:
                        raw_vol = raw_vol / inst_dict["current_price"]
                    inst_dict["volatility"] = float(np.clip(raw_vol, 0.001, 1.0))

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

            if "intensity" not in inst_dict:
                di = self._derive_intensity(inst_dict)
                if di is not None:
                    inst_dict["intensity"] = di
                    inst_dict["intensity_source"] = "derived"

            if session:
                inst_dict["session"] = session

            if isinstance(market_liquidity, dict) and inst in market_liquidity:
                try:
                    inst_dict["liquidity_hint"] = float(
                        market_liquidity.get(inst, 1.0)
                    )
                except Exception:
                    pass

            if self.debug and inst_dict:
                try:
                    src = inst_dict.get("intensity_source", "none")
                    iv = inst_dict.get("intensity", 0.0)
                    vol_v = inst_dict.get(
                        "volatility", self.Cval("min_volatility", 0.015)
                    )
                    tr = inst_dict.get("trend_strength", 0.0)
                    mo = inst_dict.get("momentum", 0.0)
                    self.unified_logger.log_signal_mapping(
                        instrument=inst,
                        source=str(src),
                        intensity=float(
                            iv if isinstance(iv, (int, float)) else 0.0
                        ),
                        volatility=float(
                            vol_v
                            if isinstance(vol_v, (int, float))
                            else self.Cval("min_volatility", 0.015)
                        ),
                        trend=float(tr if isinstance(tr, (int, float)) else 0.0),
                        momentum=float(mo if isinstance(mo, (int, float)) else 0.0),
                    )
                except Exception:
                    pass

            out[inst] = inst_dict
            have_any = have_any or bool(inst_dict)

        if regime:
            out["market_regime"] = regime
        if isinstance(liq_score, (int, float)):
            out["liquidity_score"] = float(liq_score)
        if isinstance(liq_caps, dict):
            out["liquidity_capabilities"] = liq_caps

        return out if have_any else None

    def _merge_market_maps(
        self, bus_map: Dict[str, Any], in_map: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge bus-derived and input-derived market views with sensible precedence."""
        out: Dict[str, Any] = dict(bus_map) if isinstance(bus_map, dict) else {}

        for inst in self.instruments:
            b = bus_map.get(inst, {}) if isinstance(bus_map, dict) else {}
            i = in_map.get(inst, {}) if isinstance(in_map, dict) else {}

            merged = dict(b)
            if isinstance(i, dict) and i:
                merged.update(i)

            # Preserve bus intensity if present, otherwise clean invalid entries
            if isinstance(b, dict) and isinstance(b.get("intensity", None), (int, float)):
                merged["intensity"] = b["intensity"]
                if "intensity_source" in b:
                    merged["intensity_source"] = b["intensity_source"]
            else:
                iv = merged.get("intensity", None)
                if not isinstance(iv, (int, float)):
                    merged.pop("intensity", None)
                    merged.pop("intensity_source", None)

            # Bus session overrides local if available
            if isinstance(b, dict) and "session" in b:
                merged["session"] = b["session"]

            if merged:
                out[inst] = merged

        # Copy any non-instrument keys from input (e.g., market_regime)
        for k, v in (in_map or {}).items():
            if k in self.instruments:
                continue
            if isinstance(v, dict):
                if v:
                    out[k] = v
            elif v is not None:
                out[k] = v

        # market_regime: bus overrides inputs if present
        if isinstance(bus_map, dict) and "market_regime" in bus_map:
            out["market_regime"] = bus_map["market_regime"]

        return out

    # ---------- thesis / logs
    async def _generate_position_thesis(
        self, market_data: Dict[str, Any], decisions: Dict[str, PositionDecisionResult]
    ) -> str:
        # Lightweight, constant-time summary string suitable for rapid training loops.
        return (
            f"PortfolioHealth={self._portfolio_health_score:.2f} | "
            f"Exposure={self._total_exposure_ratio:.1%} | "
            f"Decisions={len(decisions)} | "
            f"RiskMgmt={self._risk_management_score:.2f}"
        )

    def _flush_logs(self) -> None:
        """Flush both shared logger and debugger safely."""
        try:
            lg = getattr(self, "logger", None)
            if lg and hasattr(lg, "flush"):
                lg.flush()
        except Exception:
            pass
        try:
            dbg = getattr(self, "debugger", None)
            if dbg and hasattr(dbg, "flush"):
                dbg.flush()
        except Exception:
            pass

    # ---------- health / adaptation
    def _read_risk_level(self) -> float:
        """Read a normalized [0,1] risk level from canonical bus feeds."""
        risk_level = 0.0
        try:
            tri = self.smart_bus.get("time_risk_analysis", "PositionManager") or {}
            mc = self.smart_bus.get("market_conditions", "PositionManager") or {}
            node: Dict[str, Any] = {}
            if isinstance(tri, dict) and tri:
                node = tri
            elif isinstance(mc, dict) and mc:
                node = mc
            if isinstance(node, dict):
                rl = node.get("risk_level", node.get("risk_score", 0.0))
                risk_level = float(rl if rl is not None else 0.0)
        except Exception:
            risk_level = 0.0

        return float(np.clip(risk_level, 0.0, 1.0))

    def _update_position_health(self) -> None:
        """Periodic portfolio health update used by monitoring thread."""
        try:
            self._refresh_positions_from_bus()
            current_exposure = self._calculate_current_exposure_ratio()
            self._total_exposure_ratio = current_exposure

            self._exposure_history.append(
                {
                    "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
                    "exposure_ratio": current_exposure,
                    "position_count": len(self.open_positions),
                    "consecutive_losses": self.consecutive_losses,
                }
            )

            risk_level = self._read_risk_level()
            self._risk_management_score = max(0.1, 1.0 - risk_level)

            # Adapt params based on recent decisions and performance
            self._adapt_parameters()

            # Publish compact health node
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
        except Exception as e:
            self.logger.warning(f"Position health update failed: {e}")

    def _adapt_parameters(self) -> None:
        """Slowly adjust dynamic max_pct, sensitivity, and risk_tolerance."""
        try:
            # Dynamic max position percentage based on portfolio health
            if len(self._decision_history) >= 10:
                recent = list(self._decision_history)[-10:]
                avg_ph = np.mean(
                    [
                        d.get("portfolio_health", self._portfolio_health_score)
                        for d in recent
                    ]
                )

                current = float(
                    self._adaptive_params.get(
                        "dynamic_max_pct", self.Cval("max_position_pct", 0.10)
                    )
                )
                base = float(self.Cval("max_position_pct", 0.10))

                if avg_ph > 0.8:
                    self._adaptive_params["dynamic_max_pct"] = min(
                        current * 1.05, base * 1.5
                    )
                elif avg_ph < 0.4:
                    self._adaptive_params["dynamic_max_pct"] = max(
                        current * 0.90, base * 0.3
                    )
                else:
                    self._adaptive_params["dynamic_max_pct"] = current * 0.95 + base * 0.05

            # Signal sensitivity based on average non-hold confidence
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
                            1.2,
                            float(self._adaptive_params["signal_sensitivity"]) * 1.02,
                        )
                    elif avgc < 0.4:
                        self._adaptive_params["signal_sensitivity"] = max(
                            0.7,
                            float(self._adaptive_params["signal_sensitivity"]) * 0.98,
                        )

            # Risk tolerance based on loss streak
            if self.consecutive_losses == 0:
                self._adaptive_params["risk_tolerance"] = min(
                    1.3, float(self._adaptive_params["risk_tolerance"]) * 1.01
                )
            elif self.consecutive_losses >= 3:
                self._adaptive_params["risk_tolerance"] = max(
                    0.5, float(self._adaptive_params["risk_tolerance"]) * 0.95
                )
        except Exception as e:
            self.logger.warning(f"Parameter adaptation failed: {e}")

    def _calculate_current_exposure_ratio(self) -> float:
        """Total notional exposure from bus positions / balance."""
        balance, _ = self._read_balance_and_drawdown()
        total_exposure = 0.0
        try:
            positions = self.smart_bus.get("positions", "PositionManager")
            if isinstance(positions, dict):
                for p in positions.values():
                    try:
                        notional = float(p.get("notional_eur", 0.0))
                        if notional == 0.0:
                            units = float(p.get("units", 0.0) or 0.0)
                            entry = float(p.get("entry_price", 0.0) or 0.0)
                            notional = abs(units) * entry
                        total_exposure += abs(notional)
                    except Exception:
                        continue
        except Exception:
            pass

        return total_exposure / max(balance, 1.0)

    def _assess_portfolio_health(self) -> Dict[str, float]:
        """Compute a composite health score based on DD, exposure, streak, and risk feeds."""
        balance, drawdown = self._read_balance_and_drawdown()

        total_exposure = 0.0
        for pos_data in self.open_positions.values():
            try:
                total_exposure += abs(float(pos_data.get("size", 0.0)))
            except Exception:
                continue

        exposure_ratio = total_exposure / max(balance, 1.0)

        max_conc = self.Cval("max_instrument_concentration", 0.3)
        dd_health = max(0.0, 1.0 - drawdown * 2.0)
        exposure_health = max(0.0, 1.0 - exposure_ratio / max(max_conc, 1e-9))
        streak_health = max(
            0.1,
            1.0 - self.consecutive_losses / max(
                self.Cval("max_consecutive_losses", 5), 1
            ),
        )

        risk_level = self._read_risk_level()
        risk_health = 1.0 - risk_level

        overall_health = (dd_health + exposure_health + streak_health + risk_health) / 4.0

        self._portfolio_health_score = overall_health
        self._total_exposure_ratio = exposure_ratio
        self._portfolio_health_history.append(
            {
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
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
        """Classify the high-level market regime from volatility/trend/momentum."""
        if "market_regime" in market_data:
            return str(market_data["market_regime"])

        avg_volatility = 0.0
        trend_strength = 0.0
        momentum_count = 0

        for instrument in self.instruments:
            inst_data = market_data.get(instrument, {})
            if not isinstance(inst_data, dict):
                continue
            vol = float(inst_data.get("volatility", self.Cval("min_volatility", 0.015)))
            avg_volatility += vol
            trend_strength += abs(float(inst_data.get("trend_strength", 0.0)))
            if abs(float(inst_data.get("momentum", 0.0))) > 0.3:
                momentum_count += 1

        n = max(1, len(self.instruments))
        avg_volatility /= n
        trend_strength /= n

        if avg_volatility > 0.04:
            return "volatile"
        elif trend_strength > 0.5:
            return "trending"
        elif momentum_count >= n * 0.6:
            return "momentum"
        else:
            return "ranging"

    # ---------- signals compactors
    def _map_decision_to_intensity(self, dr: PositionDecisionResult) -> float:
        d = dr.decision
        mag = float(np.clip(dr.intensity, 0.0, 1.0))
        if d == PositionDecision.OPEN_LONG:
            return +mag
        if d == PositionDecision.OPEN_SHORT:
            return -mag
        if d == PositionDecision.SCALE_UP:
            return float(np.sign(dr.context.market_direction or 1)) * mag
        if d == PositionDecision.SCALE_DOWN:
            return -float(np.sign(dr.context.market_direction or 1)) * mag
        return 0.0

    def _signals_map_from_decisions(
        self, decisions: Dict[str, PositionDecisionResult]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for inst in self.instruments:
            dr = decisions.get(inst)
            if dr is None:
                out[inst] = {"intensity": 0.0, "decision": "hold", "confidence": 0.0}
                continue
            out[inst] = {
                "intensity": float(
                    np.clip(self._map_decision_to_intensity(dr), -1.0, 1.0)
                ),
                "decision": dr.decision.value,
                "confidence": float(np.clip(dr.confidence, 0.0, 1.0)),
            }
        return out

    def _default_signals(self) -> Dict[str, Any]:
        return {
            inst: {"intensity": 0.0, "decision": "hold", "confidence": 0.0}
            for inst in self.instruments
        }

    # ---------- SmartBus publishing of decisions (aggregated + namespaced)
    async def _update_smartbus_with_decisions(
        self, decisions: Dict[str, PositionDecisionResult]
    ) -> None:
        instrument_signals: Dict[str, Any] = {}
        position_decisions_map: Dict[str, Any] = {}

        if not hasattr(self, "_pm_registered_keys"):
            self._pm_registered_keys = set()  # type: ignore[attr-defined]

        def _register_keys(keys: List[str]) -> None:
            bus = getattr(self, "smart_bus", None)
            if not bus:
                return
            reg_fn = getattr(bus, "register_provider", None)
            if not callable(reg_fn):
                return
            new_keys = [
                k for k in keys if k not in self._pm_registered_keys  # type: ignore[attr-defined]
            ]
            if not new_keys:
                return
            try:
                reg_fn("PositionManager", new_keys)  # type: ignore[misc]
                self._pm_registered_keys.update(new_keys)  # type: ignore[attr-defined]
            except Exception:
                pass

        to_register: List[str] = [
            "position_decisions",
            "portfolio_state",
            "order_queue",
        ]

        for instrument, decision in decisions.items():
            ctx = decision.context
            payload = {
                "decision": decision.decision.value,
                "intensity": float(decision.intensity),
                "size": float(decision.size),
                "confidence": float(decision.confidence),
                "risk_factors": decision.risk_factors,
                "context": {
                    "market_intensity": float(getattr(ctx, "market_intensity", 0.0) or 0.0),
                    "market_direction": int(getattr(ctx, "market_direction", 0) or 0),
                    "volatility": float(getattr(ctx, "volatility", 0.0) or 0.0),
                    "trend_strength": float(getattr(ctx, "trend_strength", 0.0) or 0.0),
                    "momentum": float(getattr(ctx, "momentum", 0.0) or 0.0),
                    "correlation_penalty": float(getattr(ctx, "correlation_penalty", 0.0) or 0.0),
                    "session": str(getattr(ctx, "session", "unknown") or "unknown"),
                    "current_exposure": float(getattr(ctx, "current_exposure", 0.0) or 0.0),
                    "drawdown": float(getattr(ctx, "drawdown", 0.0) or 0.0),
                    "balance": float(getattr(ctx, "balance", 0.0) or 0.0),
                },
            }

            key_verbatim = f"position_decision_{instrument}"
            self.smart_bus.set(
                key_verbatim,
                payload,
                module="PositionManager",
                thesis=f"Decision for {instrument}",
            )

            canon_core = instrument.replace("/", "").replace("_", "")
            variants = {
                canon_core,
                canon_core.upper(),
                canon_core.lower(),
                instrument.replace("/", "_"),
            }
            for v in variants:
                if v and v != instrument:
                    self.smart_bus.set(
                        f"position_decision_{v}",
                        payload,
                        module="PositionManager",
                        thesis=f"Decision for {instrument} (canonical: {v})",
                    )

            position_decisions_map[instrument] = payload

            signed = self._map_decision_to_intensity(decision)
            instrument_signals[instrument] = {
                "intensity": float(np.clip(signed, -1.0, 1.0)),
                "decision": decision.decision.value,
                "confidence": float(decision.confidence),
            }

            per_inst_keys = [key_verbatim] + [f"position_decision_{v}" for v in variants]
            to_register.extend(per_inst_keys)

        self.smart_bus.set(
            "position_decisions",
            position_decisions_map,
            module="PositionManager",
            thesis="Aggregated position decisions from PM",
        )

        self.smart_bus.set(
            "portfolio_state",
            {
                "health_score": float(self._portfolio_health_score),
                "exposure_ratio": float(self._total_exposure_ratio),
                "open_positions": len(self.open_positions),
                "decision_quality": float(self._decision_quality_score),
            },
            module="PositionManager",
            thesis="Portfolio health (diagnostic)",
        )

        # Publish under namespaced key to avoid stepping on StrategyArbiter
        self.smart_bus.set(
            "position_manager_instrument_signals",
            instrument_signals,
            module="PositionManager",
            thesis="PositionManager instrument signals (namespaced)",
        )

        _register_keys(to_register)

    # ---------- state persistence
    def get_state(self) -> Dict[str, Any]:
        return {
            "config": dict(self.C.__dict__),
            "genome": self.genome,
            "open_positions": self.open_positions,
            "portfolio_health_score": self._portfolio_health_score,
            "exposure_ratio": self._total_exposure_ratio,
            "consecutive_losses": self.consecutive_losses,
            "adaptive_params": self._adaptive_params,
            "position_confidence": self.position_confidence,
            "signal_history": {k: v[-20:] for k, v in self.signal_history.items()},
            "decision_history": list(self._decision_history)[-20:],
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
        if "config" in state and isinstance(state["config"], dict):
            cfg_in = self._sanitize_config_dict(state["config"])
            try:
                self.C = TradingConfig(**cfg_in)
            except TypeError:
                # Sanitize unknown keys - get only fields that are valid init params
                # Use dataclass fields() to check which have init=True
                from dataclasses import fields as dc_fields
                try:
                    init_fields = {f.name for f in dc_fields(TradingConfig) if f.init}
                except Exception:
                    # Fallback: use vars but exclude private/internal fields
                    default_cfg = TradingConfig()
                    init_fields = {k for k in vars(default_cfg).keys() if not k.startswith('_')}
                sanitized = {k: v for k, v in cfg_in.items() if k in init_fields}
                self.C = TradingConfig(**sanitized)
            self.config.update(self.C.__dict__)
            self.default_max_pct = self.Cval("max_position_pct", 0.10)

        if isinstance(state.get("genome"), dict):
            self._initialize_genome_parameters(state["genome"])

        if isinstance(state.get("open_positions"), dict):
            try:
                self.open_positions = {
                    str(k): dict(v) for k, v in state["open_positions"].items()
                }
            except Exception:
                self.open_positions = {}

        try:
            self._portfolio_health_score = float(
                state.get("portfolio_health_score", self._portfolio_health_score)
            )
        except Exception:
            pass
        try:
            self._total_exposure_ratio = float(
                state.get("exposure_ratio", self._total_exposure_ratio)
            )
        except Exception:
            pass
        try:
            self.consecutive_losses = int(
                state.get("consecutive_losses", self.consecutive_losses)
            )
        except Exception:
            pass

        if isinstance(state.get("adaptive_params"), dict):
            try:
                self._adaptive_params.update(
                    {k: state["adaptive_params"][k] for k in state["adaptive_params"]}
                )
            except Exception:
                pass

        if isinstance(state.get("position_confidence"), dict):
            try:
                self.position_confidence = {
                    str(k): float(v)
                    for k, v in state["position_confidence"].items()
                    if isinstance(v, (int, float))
                }
            except Exception:
                self.position_confidence = {}

        if isinstance(state.get("signal_history"), dict):
            try:
                for inst in self.instruments:
                    vals = state["signal_history"].get(inst, [])
                    if isinstance(vals, list):
                        self.signal_history[inst] = [float(x) for x in vals][-50:]
            except Exception:
                pass

        if isinstance(state.get("decision_history"), list):
            self._decision_history.clear()
            for d in state["decision_history"][-100:]:
                try:
                    self._decision_history.append(d)
                except Exception:
                    continue

        if isinstance(state.get("portfolio_health_history"), list):
            self._portfolio_health_history.clear()
            for d in state["portfolio_health_history"][-50:]:
                try:
                    self._portfolio_health_history.append(d)
                except Exception:
                    continue

        if isinstance(state.get("last_decisions"), dict):
            self.last_decisions.clear()
            for inst, d in state["last_decisions"].items():
                try:
                    dec = PositionDecision(str(d.get("decision", "hold")))
                except Exception:
                    dec = PositionDecision.HOLD
                intensity = float(d.get("intensity", 0.0) or 0.0)
                confidence = float(d.get("confidence", 0.5) or 0.5)
                rationale = (
                    d.get("rationale", {})
                    if isinstance(d.get("rationale", {}), dict)
                    else {}
                )
                ctx = SignalContext(instrument=str(inst))
                self.last_decisions[str(inst)] = PositionDecisionResult(
                    decision=dec,
                    intensity=float(np.clip(intensity, -1.0, 1.0)),
                    size=0.0,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    rationale=rationale,
                    risk_factors={},
                    context=ctx,
                )

        try:
            self.success_count = int(
                state.get("success_count", getattr(self, "success_count", 0))
            )
        except Exception:
            self.success_count = getattr(self, "success_count", 0)
        try:
            self.failure_count = int(
                state.get("failure_count", getattr(self, "failure_count", 0))
            )
        except Exception:
            self.failure_count = getattr(self, "failure_count", 0)

    # ==========================================================
    # Helpers used by Part 2 (profit trailing, cooldown, favors)
    # ==========================================================
    def update_profit_tracker(self, instrument: str) -> None:
        """Call after reading bus to refresh P&L peak for instrument."""
        pnl = self._get_unrealised_pnl_from_bus(instrument)
        self._profit_tracker.update(instrument, pnl)

    def should_close_for_trailing_profit(
        self,
        instrument: str,
        trailing_pct: float,
        min_activation_eur: float,
        favors_down: bool,
    ) -> bool:
        return self._profit_tracker.should_trail_close(
            instrument, trailing_pct, min_activation_eur, favors_down
        )

    def favors_trend_down(
        self, instrument: str, current_intensity: float, lookback: int = 5, eps: float = 0.05
    ) -> bool:
        """Rough 'favors down' detector: negative slope of recent intensity with buffer."""
        hist = self.signal_history.get(instrument, [])
        if not hist:
            return current_intensity < 0.0
        seq = (hist + [current_intensity])[-max(3, lookback) :]
        if len(seq) < 3:
            return current_intensity < 0.0
        # Simple slope vs index
        xs = np.arange(len(seq), dtype=float)
        x_mean = xs.mean()
        y_mean = np.mean(seq)
        num = float(np.sum((xs - x_mean) * (np.array(seq) - y_mean)))
        den = float(np.sum((xs - x_mean) ** 2)) or 1.0
        slope = num / den
        return slope <= -abs(eps)

    def scale_cooldown_ok(self, instrument: str, now: Optional[float] = None) -> bool:
        now = now or time.time()
        return now >= float(self._scale_cooldown_until[instrument])

    def arm_scale_cooldown(self, instrument: str, seconds: float = 15.0) -> None:
        self._scale_cooldown_until[instrument] = time.time() + max(0.0, seconds)

    # ---------- cleanup
    def __del__(self):
        """
        Best-effort cleanup.

        We cannot rely on __del__ for critical logic, but we can try to
        stop the monitoring loop and flush logs to avoid noisy shutdowns.
        """
        try:
            if getattr(self, "_monitoring_active", False):
                self._monitoring_active = False
        except Exception:
            pass
        try:
            self._flush_logs()
        except Exception:
            pass
