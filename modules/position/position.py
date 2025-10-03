# -------------------------------------------------------------
# File: modules/position/position.py
# PositionManager — Decider-only (Env/Executor handle execution)
# with Integrated Debug System (plain-English BUY/SELL tracking)
# -------------------------------------------------------------

from __future__ import annotations

import asyncio
import copy
import datetime
import threading
import time
import uuid
import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np

from modules.contracts import module_args
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


####################################################################################################################
# Integrated Debug System
####################################################################################################################

class ActionType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SCALE_UP = "ADD_MORE"
    SCALE_DOWN = "REDUCE"
    CLOSE_POSITION = "CLOSE"
    EMERGENCY_EXIT = "EMERGENCY"


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


class IntegratedDebugger:
    """Lightweight decision debugger writing CSV/JSON + console banners."""

    def __init__(self, log_dir: str = "logs/debug", enable: bool = True):
        # SmartInfoBus handle (injected later by PositionManager). Adding upfront
        # silences static analysis complaining about missing attribute where
        # _should_suppress_alerts accesses self.smart_bus.
        self.smart_bus: Optional[Any] = None
        self.enabled = bool(enable)
        if not self.enabled:
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "trade_signals").mkdir(exist_ok=True)

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.decision_file = self.log_dir / f"decisions_{timestamp}.csv"
        self.summary_file = self.log_dir / f"summary_{timestamp}.txt"
        self.buy_signals_file = self.log_dir / "trade_signals" / f"buy_signals_{timestamp}.json"
        self.sell_signals_file = self.log_dir / "trade_signals" / f"sell_signals_{timestamp}.json"

        self.stats = {
            "total_decisions": 0,
            "buy_decisions": 0,
            "sell_decisions": 0,
            "hold_decisions": 0,
            "executed": 0,
            "blocked": 0,
        }

        with open(self.decision_file, "w") as f:
            f.write(
                "timestamp,instrument,action,is_buying,size_eur,confidence,"
                "signal_strength,volatility,portfolio_health,risk_score,"
                "executed,reason\n"
            )

        self.buy_signals: List[Dict[str, Any]] = []
        self.sell_signals: List[Dict[str, Any]] = []

    def _parse_action(self, decision: str, intensity: float) -> Tuple[str, bool]:
        d = (decision or "").lower()
        if "open_long" in d:
            return "BUY", True
        if "open_short" in d:
            return "SELL", False
        if "scale_up" in d:
            return "SCALE_UP", intensity > 0
        if "scale_down" in d:
            return "SCALE_DOWN", False
        if "emergency_close" in d:
            return "EMERGENCY_EXIT", False
        if "close" in d:
            return "CLOSE_POSITION", False
        return "HOLD", False

    def _calculate_risk_score(self, context: Dict[str, Any]) -> float:
        drawdown = float(context.get("drawdown", 0.0) or 0.0)
        exposure = float(context.get("current_exposure", 0.0) or 0.0)
        volatility = float(context.get("volatility", 0.02) or 0.02)
        risk = (drawdown * 0.4 + exposure * 0.3 + min(volatility / 0.05, 1.0) * 0.3)
        return float(min(risk, 1.0))

    def _generate_reason(
        self,
        action: str,
        is_buying: bool,
        instrument: str,
        signal_strength: float,
        confidence: float,
        rationale: Dict[str, Any],
    ) -> str:
        stage = rationale.get("stage", "unknown")
        factors = rationale.get("factors", [])
        if action == "BUY":
            reason = f"Opening LONG on {instrument}: bullish {signal_strength:.2f} | conf {confidence:.1%}"
        elif action == "SELL":
            reason = f"Opening SHORT on {instrument}: bearish {signal_strength:.2f} | conf {confidence:.1%}"
        elif action == "CLOSE_POSITION":
            reason = f"Closing {instrument}: {'risk rule' if 'risk' in stage else 'take profit/cut loss'}"
        elif action == "EMERGENCY_EXIT":
            reason = f"EMERGENCY EXIT {instrument}: critical risk"
        elif action == "SCALE_UP":
            reason = f"Scaling up {instrument}: signal {signal_strength:.2f}"
        elif action == "SCALE_DOWN":
            reason = f"Scaling down {instrument}: opposing or risk"
        else:
            reason = f"Holding {instrument}: waiting for stronger signals"
        if factors:
            reason += f" | {factors[0]}"
        return reason

    def _write_to_csv(self, snapshot: DebugSnapshot) -> None:
        try:
            with open(self.decision_file, "a") as f:
                f.write(
                    f"{snapshot.timestamp},{snapshot.instrument},{snapshot.action},"
                    f"{snapshot.is_buying},{snapshot.size_eur:.2f},{snapshot.confidence:.3f},"
                    f"{snapshot.signal_strength:.3f},{snapshot.volatility:.4f},"
                    f"{snapshot.portfolio_health:.3f},{snapshot.risk_score:.3f},"
                    f"{snapshot.will_execute},\"{snapshot.plain_english_reason}\"\n"
                )
        except Exception:
            pass

    def _write_to_summary(self, snapshot: DebugSnapshot) -> None:
        try:
            with open(self.summary_file, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Time: {snapshot.timestamp}\n")
                f.write(f"Instrument: {snapshot.instrument}\n")
                f.write(f"Action: {snapshot.action}\n")
                f.write(f"Size: EUR {snapshot.size_eur:,.2f}\n")
                f.write(f"Confidence: {snapshot.confidence:.1%}\n")
                f.write(f"Reason: {snapshot.plain_english_reason}\n")
                f.write(f"Status: {'EXECUTED' if snapshot.will_execute else 'BLOCKED'}\n")
        except Exception:
            pass

    def _save_signals(self) -> None:
        try:
            if self.buy_signals:
                with open(self.buy_signals_file, "w") as f:
                    json.dump(self.buy_signals, f, indent=2)
            if self.sell_signals:
                with open(self.sell_signals_file, "w") as f:
                    json.dump(self.sell_signals, f, indent=2)
        except Exception:
            pass

    def _should_suppress_alerts(self) -> bool:
        """Check if alerts should be suppressed during training/simulation mode."""
        bus = getattr(self, "smart_bus", None)
        if bus is None:
            return False
        try:
            env_cfg = bus.get("environment_config", "PositionManager")
            if isinstance(env_cfg, dict):
                mode = env_cfg.get("mode", "")
                return mode == "sim"
        except Exception:
            return False
        return False

    def _alert_buy(self, instrument: str, snapshot: DebugSnapshot) -> None:
        # Console alerts disabled - use beautiful visualizer instead
        return

    def _alert_sell(self, instrument: str, snapshot: DebugSnapshot) -> None:
        # Console alerts disabled - use beautiful visualizer instead
        return

    def log_decision(
        self,
        instrument: str,
        decision: str,
        intensity: float,
        size: float,
        confidence: float,
        context: Dict[str, Any],
        rationale: Dict[str, Any],
        portfolio_health: float,
    ) -> Optional[DebugSnapshot]:
        if not self.enabled:
            return None
        try:
            action, is_buying = self._parse_action(decision, intensity)
            signal_strength = abs(float(intensity))
            volatility = float(context.get("volatility", 0.0) or 0.0)
            risk_score = self._calculate_risk_score(context)
            plain_english = self._generate_reason(action, is_buying, instrument, signal_strength, confidence, rationale)

            will_execute = (size or 0.0) > 0 and (confidence or 0.0) > 0.3
            blocked_reason = None if will_execute else "Size or confidence too low"

            snapshot = DebugSnapshot(
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                instrument=instrument,
                action=action,
                is_buying=is_buying,
                size_eur=float(size or 0.0),
                confidence=float(np.clip(confidence or 0.0, 0.0, 1.0)),
                signal_strength=signal_strength,
                volatility=volatility,
                portfolio_health=float(np.clip(portfolio_health, 0.0, 1.0)),
                risk_score=risk_score,
                plain_english_reason=plain_english,
                will_execute=bool(will_execute),
                execution_blocked_reason=blocked_reason,
            )

            self.stats["total_decisions"] += 1
            if is_buying and action in ("BUY", "SCALE_UP"):
                self.stats["buy_decisions"] += 1
                self.buy_signals.append(snapshot.to_dict())
                self._alert_buy(instrument, snapshot)
            elif action in ("SELL", "CLOSE_POSITION", "EMERGENCY_EXIT", "SCALE_DOWN"):
                self.stats["sell_decisions"] += 1
                self.sell_signals.append(snapshot.to_dict())
                # Avoid noisy alerts for emergency exits that cannot execute (size=0)
                if not (action == "EMERGENCY_EXIT" and snapshot.size_eur <= 0.0):
                    self._alert_sell(instrument, snapshot)
            else:
                self.stats["hold_decisions"] += 1

            if will_execute:
                self.stats["executed"] += 1
            else:
                self.stats["blocked"] += 1

            self._write_to_csv(snapshot)
            self._write_to_summary(snapshot)
            self._save_signals()
            return snapshot
        except Exception:
            return None

    def print_summary(self) -> None:
        """Print aggregated decision statistics if debugger enabled."""
        # Disabled - using beautiful visualizer instead
        return


####################################################################################################################
# Async helper
####################################################################################################################
T_co = TypeVar("T_co")


async def _maybe_await(x: Union[Awaitable[T_co], T_co]) -> T_co:
    if asyncio.iscoroutine(x):
        return cast(T_co, await x)
    return cast(T_co, x)


####################################################################################################################
# Decision enums / payloads
####################################################################################################################
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


# Shared logger to keep file handlers stable
_PM_SHARED_LOGGER: Optional[RotatingLogger] = None


@module(
    **module_args(
        "PositionManager",
        description="Position decision maker that enqueues orders; Env/Executor execute & publish fills.",
        error_handling=True,
        hot_reload=True,
        timeout_ms=3000,
    )
)
class PositionManager(
    BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
):
    """
    CLEAN ARCH + DEBUG:
    - This module ONLY decides and enqueues orders on the SmartInfoBus.
    - Env/Executor execute orders, update balances/positions/trades, and publish execution state.
    - Integrated debugger records BUY/SELL/HOLD with plain-English reasons to CSV/JSON.
    """

    # Class-level hints
    signal_history: Dict[str, List[float]]
    genome: Dict[str, Any]
    env: Optional[Any]

    ####################################################################################################################
    # Lifecycle / initialization
    ####################################################################################################################
    def __init__(
        self,
        config: Optional[TradingConfig | Dict[str, Any]] = None,
        instruments: Optional[List[str]] = None,
        genome: Optional[Dict[str, Any]] = None,
        enable_debug: bool = True,
        debug_log_dir: str = "logs/debug",
        **kwargs: Any,
    ):
        # Integrated debugger (silent mode)
        self.debugger = IntegratedDebugger(log_dir=debug_log_dir, enable=enable_debug)

        # whether caller hard-forced instruments
        self._instruments_forced = instruments is not None

        # set instruments early for BaseModule plumbing
        self.instruments = instruments or ["XAU_USD", "EUR_USD"]
        self.genome = genome or {}
        self.env = None

        super().__init__()

        # Config wiring
        if isinstance(config, TradingConfig):
            self.C: TradingConfig = config
        elif isinstance(config, dict):
            self.C = TradingConfig(**config)
        else:
            self.C = TradingConfig()

        self.config: Dict[str, Any] = dict(self.C.__dict__)
        self.default_max_pct = self.C.max_position_pct

        # runtime toggles
        self.enable_legacy_bus_signal_probe: bool = bool(self.config.get("enable_legacy_bus_signal_probe", False))
        raw_bus_signal_flag = self.config.get("use_bus_instrument_signals")
        self.use_bus_instrument_signals: bool = True if raw_bus_signal_flag is None else bool(raw_bus_signal_flag)
        self.debug: bool = bool(self.config.get("debug", False) or self.config.get("debug_decisions", False))

        self._initialize_advanced_systems()
        self._sync_from_bus_env()
        self._initialize_genome_parameters(genome)
        self._initialize_position_state()
        self._initialize_position_tracking()
        self._start_monitoring()

        bal, _ = self._read_balance_and_drawdown()
        self.logger.info(
            format_operator_message(
                "INIT",
                "POSITION_MANAGER_INITIALIZED",
                instruments_count=len(self.instruments),
                initial_balance=f"EUR {bal:,.0f}",
                max_position_pct=f"{self.C.max_position_pct:.1%}",
                details="Decider-only with Integrated Debugging" if enable_debug else "Decider-only",
            )
        )
        self._flush_logs()

    def _flush_logs(self) -> None:
        try:
            lg = getattr(self, "logger", None)
            if lg and hasattr(lg, "flush"):
                lg.flush()
        except Exception:
            pass

    def _initialize(self, **kwargs: Any) -> None:
        """Module-system friendly re-init."""
        cfg_in = kwargs.get("config", None)

        if isinstance(cfg_in, TradingConfig):
            self.C = cfg_in
        elif isinstance(cfg_in, dict):
            self.C = TradingConfig(**cfg_in)
        else:
            self.C = getattr(self, "C", TradingConfig())

        self.config = dict(self.C.__dict__)
        self.default_max_pct = self.C.max_position_pct

        self.enable_legacy_bus_signal_probe = bool(self.config.get("enable_legacy_bus_signal_probe", False))
        raw_bus_signal_flag = self.config.get("use_bus_instrument_signals")
        self.use_bus_instrument_signals = True if raw_bus_signal_flag is None else bool(raw_bus_signal_flag)
        self.debug = bool(self.config.get("debug", False) or self.config.get("debug_decisions", False))

        instruments = kwargs.get("instruments", None)
        if instruments is not None:
            self._instruments_forced = True
            self.instruments = instruments or ["XAU_USD", "EUR_USD"]

        self.genome = kwargs.get("genome", None) or self.genome or {}
        self.env = kwargs.get("env", None) or self.env

        self._initialize_advanced_systems()
        self._sync_from_bus_env()
        self._initialize_genome_parameters(self.genome)
        self._initialize_position_state()
        self._initialize_position_tracking()

        bal, _ = self._read_balance_and_drawdown()
        self.logger.info(
            format_operator_message(
                "[INIT]",
                "POSITION_MANAGER_REINITIALIZED",
                instruments_count=len(self.instruments),
                initial_balance=f"EUR {bal:,.0f}",
                max_position_pct=f"{self.C.max_position_pct:.1%}",
                details="Clean decider mode + Debugger",
            )
        )

    def _initialize_advanced_systems(self) -> None:
        self.smart_bus = InfoBusManager.get_instance()
        # Propagate bus handle to debugger if available
        try:
            if hasattr(self, "debugger") and hasattr(self.debugger, "smart_bus"):
                self.debugger.smart_bus = self.smart_bus
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

        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PositionManager", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # simple circuit breaker state
        self.circuit_breaker = {
            "failures": 0,
            "last_failure": 0,
            "state": "CLOSED",
            "threshold": self.C.position_circuit_breaker_threshold,
        }

    def _start_monitoring(self) -> None:
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
        self.default_max_pct = self.C.max_position_pct

    def _initialize_position_state(self) -> None:
        self.consecutive_losses = 0
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self._decision_history = deque(maxlen=100)
        self._portfolio_health_history = deque(maxlen=50)
        self._exposure_history = deque(maxlen=100)
        self._performance_analytics = defaultdict(list)
        self.last_decisions: Dict[str, PositionDecisionResult] = {}
        self.position_confidence: Dict[str, float] = {}
        self.signal_history: Dict[str, List[float]] = {inst: [] for inst in self.instruments}
        self._portfolio_health_score = 1.0
        self._total_exposure_ratio = 0.0
        self._decision_quality_score = 0.5
        self._risk_management_score = 1.0
        self._adaptive_params = {
            "dynamic_max_pct": self.C.max_position_pct,
            "signal_sensitivity": 1.0,
            "risk_tolerance": 1.0,
            "confidence_threshold": 0.5,
        }
        self._forced_action = None
        self._forced_conf = None
        self._last_sync_time: Optional[datetime.datetime] = None

    def _initialize_position_tracking(self) -> None:
        self._position_metadata: Dict[str, Dict[str, Any]] = {}
        self._position_performance: Dict[str, Dict[str, Any]] = {}
        self._exit_signals: Dict[str, List[Dict[str, Any]]] = {}

    ####################################################################################################################
    # SmartBus execution interface (ENV/Executor execute)
    ####################################################################################################################
    def _publish_bus_feeds(
        self,
        balance: float,
        equity: float,
        current_pnl: float,
        trades: Optional[List[Dict[str, Any]]] = None,
        execution_data: Optional[Dict[str, Any]] = None,
        order_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """No-op: Env/Executor are the sources of truth for execution/balances/trades."""
        return

    def _refresh_positions_from_bus(self) -> None:
        """
        Mirror Env/Executor positions snapshot (published by Executor) into PM.open_positions.
        Expected schema per instrument (example):
          positions = {
            "EUR_USD": {
               "side": +1/-1,
               "units": 125000,
               "entry_price": 1.0831,
               "notional_eur": 135000.0,
               "unrealized_pnl_eur": 230.5,
               "open_time": "iso",
            }, ...
          }
        """
        try:
            pos = self.smart_bus.get("positions", "PositionManager")
            if not isinstance(pos, dict):
                return
            new_positions: Dict[str, Dict[str, Any]] = {}
            for inst, p in pos.items():
                try:
                    side = int(np.sign(p.get("side", 0)))
                    units = float(p.get("units", 0.0) or 0.0)
                    entry = float(p.get("entry_price", 0.0) or 0.0)
                    notional = float(p.get("notional_eur", abs(units) * entry) or 0.0)
                    new_positions[inst] = {
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

    ####################################################################################################################
    # Env / balance alignment
    ####################################################################################################################
    def _sync_from_bus_env(self) -> None:
        """Align PM config with the env's published environment_config (balance, instruments).

        Core rule: Do not silently override configured initial_balance from a stale bus value.
        Only accept an override if 'accept_env_balance_override' is explicitly enabled in config.
        """
        try:
            env_cfg = self.smart_bus.get("environment_config", "PositionManager")
            if not isinstance(env_cfg, dict):
                return

            # Optional opt-in to accept bus overrides for initial balance
            accept_override = bool(self.config.get("accept_env_balance_override", False))

            ib = env_cfg.get("initial_balance")
            if isinstance(ib, (int, float)) and ib > 0:
                if accept_override:
                    self.C.initial_balance = float(ib)
                    self.config["initial_balance"] = float(ib)
                else:
                    # Warn on mismatch but keep configured value
                    try:
                        if abs(float(ib) - float(self.C.initial_balance)) / max(1.0, float(self.C.initial_balance)) > 0.05:
                            self.logger.warning(
                                format_operator_message(
                                    "[WARN]",
                                    "ENV_BALANCE_OVERRIDE_IGNORED",
                                    configured=f"EUR {self.C.initial_balance:,.0f}",
                                    bus_value=f"EUR {float(ib):,.0f}",
                                    hint="Set accept_env_balance_override=true to allow"
                                )
                            )
                    except Exception:
                        pass

            env_insts = env_cfg.get("instruments")
            if isinstance(env_insts, list) and env_insts and not self._instruments_forced:
                self.instruments = [str(x) for x in env_insts]

            self.default_max_pct = self.C.max_position_pct
        except Exception:
            pass

    def _read_balance_and_drawdown(self) -> Tuple[float, float]:
        """Read balance & drawdown from env/optional executor snapshots.

        Priority:
          1) Configured initial_balance (authoritative per run)
          2) market_state / portfolio_metrics live balances
          3) environment_config.initial_balance IF explicitly allowed via accept_env_balance_override
        """
        balance = float(self.C.initial_balance)
        drawdown = 0.0

        try:
            # Live anchors first (if available)
            market_state = self.smart_bus.get("market_state", "PositionManager")
            if isinstance(market_state, dict):
                try:
                    balance = float(market_state.get("balance", balance))
                except Exception:
                    pass
                try:
                    drawdown = float(market_state.get("drawdown", drawdown))
                except Exception:
                    pass

            portfolio_metrics = self.smart_bus.get("portfolio_metrics", "PositionManager")
            if isinstance(portfolio_metrics, dict):
                try:
                    balance = float(portfolio_metrics.get("balance", balance))
                except Exception:
                    pass
                try:
                    drawdown = float(
                        portfolio_metrics.get("current_drawdown", portfolio_metrics.get("drawdown", drawdown))
                    )
                except Exception:
                    pass

            # Only accept environment_config override if explicitly enabled
            accept_override = bool(self.config.get("accept_env_balance_override", False))
            if accept_override:
                env_cfg = self.smart_bus.get("environment_config", "PositionManager")
                if isinstance(env_cfg, dict):
                    ib = env_cfg.get("initial_balance")
                    if isinstance(ib, (int, float)) and ib > 0:
                        balance = float(ib)
        except Exception:
            pass

        return balance, drawdown

    ####################################################################################################################
    # Order building / enqueue
    ####################################################################################################################
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
        """Build canonical order payload for Env/Executor execution."""
        order_id = str(uuid.uuid4())
        order = {
            "id": order_id,
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
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
            self.logger.info(format_operator_message("ORDER", "ORDER_BUILT", **order))
            self._flush_logs()
        return order

    def _check_voting_consensus(self) -> bool:
        """
        Check if voting system has produced valid consensus before allowing orders.
        Prevents unchecked trades from bypassing the voting committee.
        """
        try:
            # Check for VotingKernel consensus
            consensus = self.smart_bus.get("committee_consensus", "PositionManager")
            if isinstance(consensus, dict):
                # Check if consensus exists and has minimum strength
                consensus_exists = consensus.get("consensus_exists")
                consensus_strength = consensus.get("consensus_strength", 0.0)

                # Accept if consensus exists OR if strength is above threshold
                if consensus_exists or (isinstance(consensus_strength, (int, float)) and float(consensus_strength) > 0.3):
                    return True

            # Fallback: Check for trade_vote_v2 from VotingKernel
            trade_vote = self.smart_bus.get("trade_vote_v2", "PositionManager")
            if isinstance(trade_vote, dict) and trade_vote.get("action"):
                return True

            # No valid consensus found
            return False
        except Exception:
            # On error, be conservative and block orders
            return False

    def _enqueue_orders(self, orders: List[Dict[str, Any]]) -> None:
        """Append orders to shared 'order_queue' on the bus, but only if voting consensus exists."""
        if not orders:
            return

        # CRITICAL FIX: Check voting consensus before enqueueing orders
        if not self._check_voting_consensus():
            if self.debug:
                self.logger.warning(
                    format_operator_message(
                        "[GATE]",
                        "ORDERS_BLOCKED_NO_CONSENSUS",
                        count=len(orders),
                        reason="Voting system has not produced consensus - blocking orders to prevent unchecked trades"
                    )
                )
            return

        try:
            existing = self.smart_bus.get("order_queue", "PositionManager")
            if not isinstance(existing, list):
                existing = []
            existing_ids = {o.get("id") for o in existing if isinstance(o, dict)}
            for o in orders:
                if o.get("id") not in existing_ids:
                    existing.append(o)
                    existing_ids.add(o.get("id"))
            self.smart_bus.set(
                "order_queue",
                existing,
                module="PositionManager",
                thesis=f"Enqueued {len(orders)} order(s) for Env/Executor execution (voting consensus verified)",
            )
        except Exception as e:
            self.logger.error(f"Failed to enqueue orders: {e}")

    def _translate_decisions_to_orders(
        self, decisions: Dict[str, PositionDecisionResult]
    ) -> List[Dict[str, Any]]:
        """Convert decisions to concrete orders for Env/Executor."""
        orders: List[Dict[str, Any]] = []
        for inst, dr in decisions.items():
            if dr.decision == PositionDecision.HOLD:
                continue

            side = 1 if dr.decision in (PositionDecision.OPEN_LONG, PositionDecision.SCALE_UP) else -1
            intent = {
                PositionDecision.OPEN_LONG: "open",
                PositionDecision.OPEN_SHORT: "open",
                PositionDecision.SCALE_UP: "scale_up",
                PositionDecision.SCALE_DOWN: "scale_down",
                PositionDecision.CLOSE: "close",
                PositionDecision.EMERGENCY_CLOSE: "emergency_close",
            }[dr.decision]

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

            if self.debug:
                self.logger.info(
                    format_operator_message(
                        "SIGNAL",
                        "ORDER_ENQUEUED_PREVIEW",
                        instrument=inst,
                        decision=dr.decision.value,
                        size_eur=f"{dr.size:.2f}",
                        reduce_only=str(reduce_only),
                        confidence=f"{dr.confidence:.3f}",
                    )
                )
        if orders and self.debug:
            self._flush_logs()
        return orders

    ####################################################################################################################
    # Main processing entry
    ####################################################################################################################
    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """
    Main processing method - outputs PM-owned contract keys and enqueues orders.
        Provides: position_decisions, instrument_signals, position_health, portfolio_state, order_queue
        """
        t0 = time.time()
        budget_ms = float(getattr(self.metadata, "timeout_ms", 3000))
        try:
            # Mirror current positions first (optional)
            self._refresh_positions_from_bus()

            # Build market snapshot from inputs + SmartBus and merge
            market_from_inputs = self._extract_market_data_from_inputs(inputs)
            bus_snapshot = self._extract_market_data_from_smartbus() or {}
            market_data: Dict[str, Any] = self._merge_market_maps(bus_snapshot, market_from_inputs)

            await asyncio.sleep(0)

            # No market data? return minimal
            if not any(isinstance(market_data.get(i, {}), dict) and market_data.get(i) for i in self.instruments):
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

            # Decisions (hierarchical)
            decisions = await _maybe_await(self.process_market_signals(market_data))
            await asyncio.sleep(0)

            if (time.time() - t0) * 1000.0 > budget_ms * 0.95:
                await self._update_smartbus_with_decisions(decisions)
                metrics = self._read_env_metrics()
                current_queue = self._safe_get_order_queue()
                return self._contract_payload(
                    decisions=decisions,
                    orders_created=[],
                    processing_ms=(time.time() - t0) * 1000.0,
                    thesis="Aborted post-processing due to time budget",
                    instrument_signals=self._signals_map_from_decisions(decisions),
                    order_queue=current_queue,
                    **metrics,
                )

            # Publish decision artifacts (no execution here)
            await self._update_smartbus_with_decisions(decisions)
            await asyncio.sleep(0)

            # Convert to orders + enqueue
            orders_created = self._translate_decisions_to_orders(decisions)
            self._enqueue_orders(orders_created)

            # Read live queue
            current_queue = self._safe_get_order_queue()

            # Thesis + metrics for output
            thesis = await self._generate_position_thesis(market_data, decisions)
            metrics = self._read_env_metrics()

            # Minimal history bookkeeping
            self._decision_history.append(
                {
                    "ts": datetime.datetime.utcnow().isoformat() + "Z",
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
    ####################################################################################################################
    # Contract output builder
    ####################################################################################################################
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
                "open_positions": int(len(positions_snapshot)),
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
            # read-only mirrors (if Executor published them)
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
            # PM-owned keys
            "instrument_signals": copy.deepcopy(instrument_signals),
            "order_queue": copy.deepcopy(order_queue),
            # RENAMED: position_data -> position_manager_data to avoid conflict with Executor's position_data
            # Executor owns canonical position_data (actual executed positions)
            # PM owns decision data (what PM decided to do)
            "position_manager_data": {
                "decisions": position_decisions,
                "health": pos_health,
                "portfolio_state": {
                    "health_score": float(self._portfolio_health_score),
                    "exposure_ratio": float(self._total_exposure_ratio),
                    "open_positions": int(len(positions_snapshot)),
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

    def _read_env_metrics(self) -> Dict[str, Any]:
        """Read live metrics strictly from Env/Executor-published bus keys (optional)."""
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
    ####################################################################################################################
    # Decision pipeline
    ####################################################################################################################
    @create_error_handler("process_market_signals")
    def process_market_signals(self, market_data: Dict[str, Any]) -> Dict[str, PositionDecisionResult]:
        """Hierarchical decision making: portfolio â†’ instrument â†’ sizing."""
        decisions: Dict[str, PositionDecisionResult] = {}

        # Portfolio-level assessment
        portfolio_health = self._assess_portfolio_health()
        market_regime = self._assess_market_regime(market_data)

        if self.debug:
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

        # Per-instrument decisions
        for instrument in self.instruments:
            signal_context = self._extract_signal_context(instrument, market_data, portfolio_health)
            decision_result = self._make_position_decision(signal_context)  # DEBUG integrated inside

            decisions[instrument] = decision_result
            self.last_decisions[instrument] = decision_result

            self.signal_history[instrument].append(signal_context.market_intensity)
            if len(self.signal_history[instrument]) > 50:
                self.signal_history[instrument].pop(0)

            if self.debug and decision_result.decision != PositionDecision.HOLD:
                self.logger.info(
                    format_operator_message(
                        "[MONEY]",
                        "POSITION_DECISION",
                        instrument=instrument,
                        decision=decision_result.decision.value,
                        intensity=f"{decision_result.intensity:.3f}",
                        size=f"EUR {decision_result.size:.0f}",
                        confidence=f"{decision_result.confidence:.3f}",
                        rationale=decision_result.rationale.get("stage", "unknown"),
                    )
                )

        if self.debug:
            self._flush_logs()
        return decisions

    def _assess_portfolio_health(self) -> Dict[str, float]:
        balance, drawdown = self._read_balance_and_drawdown()

        # exposure from mirrored open_positions
        total_exposure = 0.0
        for pos_data in self.open_positions.values():
            try:
                total_exposure += abs(float(pos_data.get("size", 0.0)))
            except Exception:
                continue
        exposure_ratio = total_exposure / max(balance, 1.0)

        dd_health = max(0.0, 1.0 - drawdown * 2.0)
        exposure_health = max(0.0, 1.0 - exposure_ratio / max(self.C.max_instrument_concentration, 1e-9))
        streak_health = max(0.1, 1.0 - self.consecutive_losses / max(self.C.max_consecutive_losses, 1))

        # time/market risk (from required feeds)
        risk_level = 0.0
        tri = self.smart_bus.get("time_risk_analysis", "PositionManager") or {}
        mc = self.smart_bus.get("market_conditions", "PositionManager") or {}
        if isinstance(tri, dict):
            rl = tri.get("risk_level", tri.get("risk_score", 0.0))
            try:
                risk_level = float(rl if rl is not None else 0.0)
            except Exception:
                risk_level = 0.0
        elif isinstance(mc, dict):
            rl = mc.get("risk_level", mc.get("risk_score", 0.0))
            try:
                risk_level = float(rl if rl is not None else 0.0)
            except Exception:
                risk_level = 0.0

        risk_level = float(np.clip(risk_level, 0.0, 1.0))
        risk_health = 1.0 - risk_level

        overall_health = (dd_health + exposure_health + streak_health + risk_health) / 4.0

        self._portfolio_health_score = overall_health
        self._total_exposure_ratio = exposure_ratio
        self._portfolio_health_history.append(
            {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
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
        if "market_regime" in market_data:
            return str(market_data["market_regime"])

        avg_volatility = 0.0
        trend_strength = 0.0
        momentum_count = 0

        for instrument in self.instruments:
            inst_data = market_data.get(instrument, {})
            if not isinstance(inst_data, dict):
                continue
            vol = float(inst_data.get("volatility", self.C.min_volatility))
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

    def _extract_signal_context(
        self, instrument: str, market_data: Dict[str, Any], portfolio_health: Dict[str, float]
    ) -> SignalContext:
        inst_data = market_data.get(instrument, {}) or {}

        raw_intensity = inst_data.get("intensity", 0.0)
        market_intensity = float(raw_intensity) if isinstance(raw_intensity, (int, float)) else 0.0
        market_direction = int(np.sign(market_intensity))

        volatility = inst_data.get("volatility", self.C.min_volatility)
        try:
            volatility = float(volatility)
        except Exception:
            volatility = self.C.min_volatility
        volatility = max(volatility, self.C.min_volatility)

        trend_strength = float(inst_data.get("trend_strength", 0.0) or 0.0)
        momentum = float(inst_data.get("momentum", 0.0) or 0.0)
        volume_profile = float(inst_data.get("volume_profile", 1.0) or 1.0)

        regime = market_data.get("market_regime", None)
        if regime is None:
            mc = self.smart_bus.get("market_regime", "PositionManager")
            regime = mc or "normal"

        market_conditions = self.smart_bus.get("market_conditions", "PositionManager") or {}
        session = inst_data.get("session") or market_conditions.get("session", "unknown")

        # correlation penalty (optional)
        correlation_penalty = 0.0
        correlation_data = self.smart_bus.get("correlation_matrix", "PositionManager")
        if isinstance(correlation_data, dict):
            try:
                node = correlation_data.get(instrument)
                if isinstance(node, dict) and "avg_correlation" in node:
                    ac = node.get("avg_correlation")
                    if isinstance(ac, (int, float)):
                        correlation_penalty = min(abs(float(ac)) * 0.5, 0.8)
                else:
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

        # liquidity
        liquidity_score = self._get_liquidity(instrument)

        balance_val = float(portfolio_health.get("balance", self.C.initial_balance))
        dd_val = float(portfolio_health.get("drawdown", 0.0))

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
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

    def _get_liquidity(self, instrument: str) -> float:
        liq_global = self.smart_bus.get("liquidity_score", "PositionManager")
        if isinstance(liq_global, (int, float)):
            return float(liq_global)
        liq_caps = self.smart_bus.get("liquidity_capabilities", "PositionManager")
        if isinstance(liq_caps, dict):
            for k in ("score", "aggregate_score", "global_score"):
                v = liq_caps.get(k, None)
                if isinstance(v, (int, float)):
                    return float(v)
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
        market_liquidity = self.smart_bus.get("market_liquidity", "PositionManager")
        if isinstance(market_liquidity, dict):
            try:
                return float(market_liquidity.get(instrument, 1.0))
            except Exception:
                pass
        return 1.0

    def _derive_intensity(self, inst_dict: Dict[str, Any]) -> Optional[float]:
        try:
            trend = float(inst_dict.get("trend_strength", 0.0))
            mom = float(inst_dict.get("momentum", 0.0))
            rsi = float(inst_dict.get("rsi", 50.0))
            vol = float(inst_dict.get("volatility", self.C.min_volatility))
            vol = max(vol, 1e-6)
            trend_term = float(np.tanh(trend * 3.0))
            macd_term = float(np.tanh(mom / (vol * 50.0)))
            rsi_term = float(np.clip((rsi - 50.0) / 50.0, -1.0, 1.0))
            intensity = 0.5 * trend_term + 0.3 * macd_term + 0.2 * rsi_term
            return float(np.clip(intensity, -1.0, 1.0))
        except Exception:
            return None
    ####################################################################################################################
    # SmartBus I/O (inputs + bus)
    ####################################################################################################################
    def _extract_market_data_from_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        market_data = inputs.get("market_data") or {}
        price_data = inputs.get("price_data") or {}
        tech = inputs.get("indicators") or inputs.get("technical_indicators") or {}
        vol = inputs.get("volatility_data") or {}
        simple_prices = inputs.get("prices") or {}
        regime = inputs.get("market_regime") or inputs.get("market_context", {}).get("volatility_regime")
        session = inputs.get("trading_session") or inputs.get("market_context", {}).get("session")
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

        def variants(inst: str):
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
                inst_dict["current_price"] = float(md.get("close", md.get("price", md.get("bid", 0.0))))
                inst_dict["volatility"] = float(md.get("atr", md.get("volatility", self.C.min_volatility)))

            pd = pick(price_data, inst)
            if isinstance(pd, dict):
                last = pd.get("last", pd.get("close"))
                if isinstance(last, (int, float)):
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
                inst_dict["volatility"] = float(
                    vd.get("atr", vd.get("volatility", inst_dict.get("volatility", self.C.min_volatility)))
                )

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
                    inst_dict["liquidity_hint"] = float(market_liquidity.get(inst, 1.0))
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
        market_conditions = self.smart_bus.get("market_conditions", "PositionManager") or {}
        liq_caps = self.smart_bus.get("liquidity_capabilities", "PositionManager") or {}
        liq_score = self.smart_bus.get("liquidity_score", "PositionManager")
        market_liquidity = self.smart_bus.get("market_liquidity", "PositionManager") or {}

        regime = market_context.get("volatility_regime", market_conditions.get("volatility_regime"))
        session = market_context.get("session")

        bus_signals = {}
        if self.use_bus_instrument_signals:
            signal_keys = ("instrument_signals", "kernel_instrument_signals", "arbiter_instrument_signals", "position_manager_instrument_signals")
            for key in signal_keys:
                candidate = self.smart_bus.get(key, "PositionManager")
                if isinstance(candidate, dict) and candidate:
                    bus_signals = candidate
                    break

        def variants(inst: str) -> List[str]:
            core = inst.replace("/", "").replace("_", "")
            return [
                inst,
                inst.replace("/", ""),
                inst.replace("/", "_"),
                inst.upper(),
                inst.lower(),
                core.upper(),
                core.lower(),
            ]

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
                sma50 = float(ti.get("sma_50", 0.0)) or 1.0
                inst_dict["trend_strength"] = (sma20 - sma50) / (abs(sma50) or 1.0)
                inst_dict["momentum"] = float(ti.get("macd", 0.0))
                inst_dict["rsi"] = float(ti.get("rsi", 50.0))

            vd = pick(vol_map, inst)
            if isinstance(vd, dict):
                vol_val = vd.get("atr", vd.get("volatility", self.C.min_volatility))
                if isinstance(vol_val, (int, float)):
                    inst_dict["volatility"] = float(vol_val)

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
                    inst_dict["liquidity_hint"] = float(market_liquidity.get(inst, 1.0))
                except Exception:
                    pass

            # Per-instrument mapping debug (source + values)
            try:
                if self.debug and inst_dict:
                    src = inst_dict.get("intensity_source", "none")
                    iv = inst_dict.get("intensity", 0.0)
                    vol_v = inst_dict.get("volatility", self.C.min_volatility)
                    tr = inst_dict.get("trend_strength", 0.0)
                    mo = inst_dict.get("momentum", 0.0)
                    self.logger.debug(
                        format_operator_message(
                            icon="[MAP]",
                            message="Signal mapping",
                            instrument=inst,
                            source=str(src),
                            intensity=f"{float(iv if isinstance(iv, (int, float)) else 0.0):.3f}",
                            volatility=f"{float(vol_v if isinstance(vol_v, (int, float)) else self.C.min_volatility):.4f}",
                            trend=f"{float(tr if isinstance(tr, (int, float)) else 0.0):.3f}",
                            momentum=f"{float(mo if isinstance(mo, (int, float)) else 0.0):.3f}",
                        )
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

    def _merge_market_maps(self, bus_map: Dict[str, Any], in_map: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(bus_map) if isinstance(bus_map, dict) else {}

        for inst in self.instruments:
            b = bus_map.get(inst, {}) if isinstance(bus_map, dict) else {}
            i = in_map.get(inst, {}) if isinstance(in_map, dict) else {}

            merged = dict(b)
            if isinstance(i, dict) and i:
                merged.update(i)

            if isinstance(b, dict) and isinstance(b.get("intensity", None), (int, float)):
                merged["intensity"] = b["intensity"]
                if "intensity_source" in b:
                    merged["intensity_source"] = b["intensity_source"]
            else:
                iv = merged.get("intensity", None)
                if not isinstance(iv, (int, float)):
                    merged.pop("intensity", None)
                    merged.pop("intensity_source", None)

            if isinstance(b, dict) and "session" in b:
                merged["session"] = b["session"]

            if merged:
                out[inst] = merged

        for k, v in (in_map or {}).items():
            if k in self.instruments:
                continue
            if isinstance(v, dict):
                if v:
                    out[k] = v
            elif v is not None:
                out[k] = v

        if isinstance(bus_map, dict) and "market_regime" in bus_map:
            out["market_regime"] = bus_map["market_regime"]

        return out

    async def _update_smartbus_with_decisions(self, decisions: Dict[str, PositionDecisionResult]) -> None:
        """
        Publish decision artifacts to the InfoBus (NO execution/balance/trades).

        Emits (PM-owned):
    - 'position_decisions' (aggregated)
    - 'instrument_signals' (compact)
    - 'portfolio_state' (light diagnostics)
        Also writes per-instrument 'position_decision_<inst>' variants for convenience.
        """
        instrument_signals: Dict[str, Any] = {}
        position_decisions_map: Dict[str, Any] = {}

        # Lazy registry for provider keys
        if not hasattr(self, "_pm_registered_keys"):
            self._pm_registered_keys = set()  # type: ignore[attr-defined]

        def _register_keys(keys: List[str]) -> None:
            bus = getattr(self, "smart_bus", None)
            if not bus:
                return
            reg_fn = getattr(bus, "register_provider", None)
            if not callable(reg_fn):
                return
            new_keys = [k for k in keys if k not in self._pm_registered_keys]  # type: ignore[attr-defined]
            if not new_keys:
                return
            try:
                reg_fn("PositionManager", new_keys)  # type: ignore[misc]
                self._pm_registered_keys.update(new_keys)  # type: ignore[attr-defined]
            except Exception:
                pass

        to_register: List[str] = ["position_decisions", "portfolio_state", "order_queue"]

        for instrument, decision in decisions.items():
            payload = {
                "decision": decision.decision.value,
                "intensity": float(decision.intensity),
                "size": float(decision.size),
                "confidence": float(decision.confidence),
                "risk_factors": decision.risk_factors,
            }

            # 1) Per-instrument (verbatim, e.g., "EUR_USD")
            key_verbatim = f"position_decision_{instrument}"
            self.smart_bus.set(key_verbatim, payload, module="PositionManager", thesis=f"Decision for {instrument}")

            # Canonical variants (EURUSD, eurusd, EUR_USD)
            canon_core = instrument.replace("/", "").replace("_", "")
            variants = {canon_core, canon_core.upper(), canon_core.lower(), instrument.replace("/", "_")}

            for v in variants:
                if v and v != instrument:
                    self.smart_bus.set(
                        f"position_decision_{v}",
                        payload,
                        module="PositionManager",
                        thesis=f"Decision for {instrument} (canonical: {v})",
                    )

            position_decisions_map[instrument] = payload

            # Compact instrument_signals
            signed = self._map_decision_to_intensity(decision)
            instrument_signals[instrument] = {
                "intensity": float(np.clip(signed, -1.0, 1.0)),
                "decision": decision.decision.value,
                "confidence": float(decision.confidence),
            }

            per_inst_keys = [key_verbatim] + [f"position_decision_{v}" for v in variants]
            to_register.extend(per_inst_keys)

        # Aggregated decisions
        self.smart_bus.set(
            "position_decisions",
            position_decisions_map,
            module="PositionManager",
            thesis="Aggregated position decisions from PM",
        )

        # Lightweight portfolio state (diagnostic only)
        self.smart_bus.set(
            "portfolio_state",
            {
                "health_score": float(self._portfolio_health_score),
                "exposure_ratio": float(self._total_exposure_ratio),
                "open_positions": int(len(self.open_positions)),
                "decision_quality": float(self._decision_quality_score),
            },
            module="PositionManager",
            thesis="Portfolio health (diagnostic)",
        )

        # Publish PM-internal signals under a namespaced key to avoid duplicate providers
        # StrategyArbiter is the single writer for 'instrument_signals'.
        self.smart_bus.set(
            "position_manager_instrument_signals",
            instrument_signals,
            module="PositionManager",
            thesis="PositionManager instrument signals (namespaced)",
        )

        # Register providers for owned keys
        _register_keys(to_register)

    async def _generate_position_thesis(
        self, market_data: Dict[str, Any], decisions: Dict[str, PositionDecisionResult]
    ) -> str:
        return (
            f"PortfolioHealth={self._portfolio_health_score:.2f} | "
            f"Exposure={self._total_exposure_ratio:.1%} | "
            f"Decisions={len(decisions)} | "
            f"RiskMgmt={self._risk_management_score:.2f}"
        )

    def reset(self) -> None:
        super().reset()
        self.C.max_position_pct = self.default_max_pct
        self.config.update(self.C.__dict__)
        self.consecutive_losses = 0
        self.open_positions.clear()
        self.last_decisions.clear()
        self.position_confidence.clear()
        self._decision_history.clear()
        self._portfolio_health_history.clear()
        self._exposure_history.clear()
        self._performance_analytics.clear()
        for inst in self.instruments:
            self.signal_history[inst].clear()
        self._position_metadata.clear()
        self._position_performance.clear()
        self._exit_signals.clear()
        self._forced_action = None
        self._forced_conf = None
        self._portfolio_health_score = 1.0
        self._total_exposure_ratio = 0.0
        self._decision_quality_score = 0.5
        self._risk_management_score = 1.0
        self._adaptive_params = {
            "dynamic_max_pct": self.C.max_position_pct,
            "signal_sensitivity": 1.0,
            "risk_tolerance": 1.0,
            "confidence_threshold": 0.5,
        }
    ####################################################################################################################
    # Core decision logic WITH Integrated Debug logging
    ####################################################################################################################
    def _check_emergency_conditions(self, context: SignalContext) -> bool:
        """Return True if any emergency condition is met.

        Added diagnostics:
          - Logs WHICH specific triggers fired (once per step) with their raw values.
          - Optional suppression if executor appears inactive (to avoid noisy spam while no closer can act).

        Suppression flag precedence:
          TradingConfig.suppress_emergency_without_executor (bool, default True if missing)
        Detection of executor activity (cheap heuristics):
          - Any of: execution_data, executor_debug, positions keys recently published.
        """
        triggers = {
            "drawdown": context.drawdown > float(getattr(self.C, 'emergency_drawdown_trigger', 0.15)),
            "loss_streak": self.consecutive_losses >= self.C.max_consecutive_losses,
            "exposure": context.current_exposure > self.C.max_instrument_concentration * 1.5,
            "liquidity": context.liquidity_score < 0.3,
        }
        # Do not trigger emergency on liquidity alone; require another hard trigger
        active = bool(triggers["drawdown"] or triggers["loss_streak"] or triggers["exposure"])
        if not active:
            return False

        # Executor activity check with recency guard (avoid stale bus data)
        executor_active = False
        try:
            for key in ("execution_data", "executor_debug", "positions"):
                try:
                    md = getattr(self.smart_bus, "get_with_metadata", None)
                    if md is not None:
                        rec = md(key, "PositionManager")
                        if rec and rec.value is not None and hasattr(rec, "age_seconds") and rec.age_seconds() < 10.0:
                            executor_active = True
                            break
                    else:
                        val = self.smart_bus.get(key, "PositionManager", default=None)
                        if val:
                            executor_active = True
                            break
                except Exception:
                    continue
        except Exception:
            pass

        suppress = getattr(self.C, "suppress_emergency_without_executor", True) and not executor_active

        # Only log once per step / context timestamp; keep lightweight
        stamp = getattr(self, "_last_emergency_diag_stamp", None)
        current_stamp = (context.timestamp or f"step_{context.step_idx}")
        if stamp != current_stamp:
            try:
                self.logger.warning(
                    format_operator_message(
                        icon="[ALERT]" if not suppress else "[INFO]",
                        message="Emergency condition evaluated" + (" (SUPPRESSED)" if suppress else ""),
                        drawdown=f"{context.drawdown:.4f}",
                        consecutive_losses=self.consecutive_losses,
                        loss_streak_trigger=triggers["loss_streak"],
                        drawdown_trigger=triggers["drawdown"],
                        exposure=f"{context.current_exposure:.4f}",
                        exposure_trigger=triggers["exposure"],
                        liquidity=f"{context.liquidity_score:.3f}",
                        liquidity_trigger=triggers["liquidity"],
                        executor_active=executor_active,
                        suppressed=suppress,
                    )
                )
            except Exception:
                pass
            self._last_emergency_diag_stamp = current_stamp

        if suppress:
            # Do not treat as emergency (executor inactive) â€“ still allow other logic to proceed normally
            return False

        return True

    def _calculate_portfolio_health_score(self, context: SignalContext) -> float:
        drawdown_component = max(0.0, 1.0 - context.drawdown * 3.0)
        denom = max(self.C.max_instrument_concentration, 1e-9)
        exposure_component = max(0.0, 1.0 - context.current_exposure / denom)
        streak_component = max(0.1, 1.0 - self.consecutive_losses / max(self.C.max_consecutive_losses, 1))
        liquidity_component = context.liquidity_score
        return (drawdown_component + exposure_component + streak_component + liquidity_component) / 4.0

    def _calculate_confidence(self, context: SignalContext, decision: PositionDecision) -> float:
        base_confidence = 0.5
        signal_confidence = min(abs(context.market_intensity) * 1.2, 0.4)
        trend_confidence = min(abs(context.trend_strength) * 0.3, 0.2)
        vol_penalty = min(context.volatility / 0.05, 0.2)
        health_boost = self._calculate_portfolio_health_score(context) * 0.2
        decision_adjustment = 0.0
        if decision in [PositionDecision.CLOSE, PositionDecision.EMERGENCY_CLOSE]:
            decision_adjustment = 0.1
        elif decision == PositionDecision.SCALE_DOWN:
            decision_adjustment = 0.05
        total_confidence = (
            base_confidence + signal_confidence + trend_confidence - vol_penalty + health_boost + decision_adjustment
        )
        return float(np.clip(total_confidence, 0.1, 1.0))

    def _assess_risk_factors(self, context: SignalContext) -> Dict[str, float]:
        risk_factors: Dict[str, float] = {}
        risk_factors["volatility"] = min((context.volatility - self.C.min_volatility) / 0.05, 0.5)
        risk_factors["correlation"] = context.correlation_penalty
        risk_factors["drawdown"] = min(context.drawdown * 2.0, 0.8)
        denom = max(self.C.max_instrument_concentration, 1e-9)
        risk_factors["concentration"] = min(context.current_exposure / denom, 0.9)
        session_risk = 0.0
        if context.session == "closed":
            session_risk = 0.3
        elif context.session == "asian":
            session_risk = 0.1
        risk_factors["session"] = session_risk
        return risk_factors

    def _calculate_position_size(self, context: SignalContext, intensity: float, confidence: float) -> float:
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
        current_exposure: Optional[float] = None,  # kept for signature parity
    ) -> float:
        volatility = max(float(np.nan_to_num(volatility, nan=self.C.min_volatility)), self.C.min_volatility)
        intensity = float(np.nan_to_num(intensity, nan=0.0))
        balance = max(float(balance), 100.0)
        drawdown = float(np.nan_to_num(drawdown, nan=0.0))

        intensity = float(np.clip(intensity, -1.0, 1.0))
        risk_pct = max(float(self._adaptive_params["dynamic_max_pct"]), 0.01)
        risk_budget = balance * risk_pct
        vol_adjusted_budget = risk_budget / volatility
        base_size = intensity * vol_adjusted_budget

        portfolio_health = self._portfolio_health_score
        health_multiplier = max(0.1, portfolio_health)
        adjusted_size = base_size * health_multiplier

        risk_tolerance = float(self._adaptive_params.get("risk_tolerance", 1.0))
        adjusted_size *= risk_tolerance

        # ═══════════════════════════════════════════════════════════════════
        # TRADING MODE MANAGER INTEGRATION
        # Apply risk_multiplier from TradingModeManager (0.5x → 2.0x)
        # ═══════════════════════════════════════════════════════════════════
        try:
            mode_config = self.smart_bus.get('mode_config', 'PositionManager') or {}
            risk_multiplier = float(mode_config.get('risk_multiplier', 1.0))
            trading_mode = self.smart_bus.get('trading_mode', 'PositionManager') or 'normal'

            # Apply mode risk multiplier
            adjusted_size *= risk_multiplier

            if self.debug and risk_multiplier != 1.0:
                self.logger.info(format_operator_message(
                    icon="🎛️",
                    message="Trading mode risk adjustment applied",
                    mode=trading_mode,
                    multiplier=f"{risk_multiplier:.2f}x",
                    size_before=f"{base_size * health_multiplier * risk_tolerance:.2f}",
                    size_after=f"{adjusted_size:.2f}"
                ))
        except Exception as e:
            # Graceful fallback - don't break sizing if mode manager unavailable
            if self.debug:
                self.logger.warning(f"Trading mode integration failed, using default multiplier: {e}")
        # ═══════════════════════════════════════════════════════════════════

        if correlation is not None:
            corr_penalty = 1.0 - min(abs(float(correlation)) * 0.3, 0.5)
            adjusted_size *= corr_penalty

        if self.consecutive_losses >= self.C.max_consecutive_losses:
            streak_reduction = max(0.1, self.C.loss_reduction)
            adjusted_size *= streak_reduction
            if self.debug:
                self.logger.info(
                    format_operator_message(
                        "SELL",
                        "LOSS_STREAK_REDUCTION",
                        reduction_factor=f"{streak_reduction:.2f}",
                        consecutive_losses=self.consecutive_losses,
                    )
                )

        abs_size = abs(adjusted_size)
        min_viable_size = balance * self.C.min_size_pct

        if abs_size < min_viable_size and abs(intensity) > 0.3:
            adjusted_size = np.sign(adjusted_size or intensity) * min_viable_size
        elif abs_size < min_viable_size:
            adjusted_size = 0.0

        max_single_position = balance * risk_pct
        final_size = float(np.clip(adjusted_size, -max_single_position, max_single_position))
        return float(np.nan_to_num(final_size, nan=0.0, posinf=0.0, neginf=0.0))

    def _make_position_decision(self, context: SignalContext) -> PositionDecisionResult:
        """Core decision logic + integrated debug logging."""
        instrument = context.instrument
        has_position = instrument in self.open_positions

        decision = PositionDecision.HOLD
        intensity = 0.0
        size = 0.0
        confidence = 0.5
        risk_factors: Dict[str, float] = {}
        rationale: Dict[str, Any] = {"stage": "initial", "factors": []}

        signal_strength = abs(context.market_intensity)
        signal_direction = int(np.sign(context.market_intensity))

        # Emergency gate
        if self._check_emergency_conditions(context):
            if has_position:
                decision = PositionDecision.EMERGENCY_CLOSE
                intensity = 1.0
                confidence = 0.9
                rationale["stage"] = "emergency"
                rationale["factors"].append("Emergency conditions detected")
                if self.debug:
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
            # Debug log + return
            debug_ctx = {
                "volatility": context.volatility,
                "current_exposure": context.current_exposure,
                "drawdown": context.drawdown,
            }
            result = PositionDecisionResult(
                decision=decision,
                intensity=intensity,
                size=size,
                confidence=confidence,
                rationale=rationale,
                risk_factors=risk_factors,
                context=context,
            )
            self.debugger.log_decision(
                instrument=instrument,
                decision=result.decision.value,
                intensity=result.intensity,
                size=result.size,
                confidence=result.confidence,
                context=debug_ctx,
                rationale=result.rationale,
                portfolio_health=self._portfolio_health_score,
            )
            return result

        # Signal quality filter
        if signal_strength < self.C.min_signal_threshold:
            rationale["stage"] = "signal_filter"
            rationale["factors"].append(
                f"Signal strength {signal_strength:.3f} below threshold {self.C.min_signal_threshold}"
            )
            # Debug log + return
            debug_ctx = {
                "volatility": context.volatility,
                "current_exposure": context.current_exposure,
                "drawdown": context.drawdown,
            }
            result = PositionDecisionResult(
                decision=decision,
                intensity=intensity,
                size=size,
                confidence=confidence,
                rationale=rationale,
                risk_factors=risk_factors,
                context=context,
            )
            self.debugger.log_decision(
                instrument=instrument,
                decision=result.decision.value,
                intensity=result.intensity,
                size=result.size,
                confidence=result.confidence,
                context=debug_ctx,
                rationale=result.rationale,
                portfolio_health=self._portfolio_health_score,
            )
            return result

        # Portfolio health gate
        portfolio_health_score = self._calculate_portfolio_health_score(context)
        if portfolio_health_score < 0.3:
            rationale["stage"] = "portfolio_health"
            rationale["factors"].append(f"Portfolio health {portfolio_health_score:.3f} too low")
            if has_position and signal_strength > 0.7:
                decision = PositionDecision.CLOSE
                intensity = 0.8
                confidence = 0.7
                rationale["factors"].append("Closing due to poor portfolio health")

            debug_ctx = {
                "volatility": context.volatility,
                "current_exposure": context.current_exposure,
                "drawdown": context.drawdown,
            }
            result = PositionDecisionResult(
                decision=decision,
                intensity=intensity,
                size=size,
                confidence=confidence,
                rationale=rationale,
                risk_factors=risk_factors,
                context=context,
            )
            self.debugger.log_decision(
                instrument=instrument,
                decision=result.decision.value,
                intensity=result.intensity,
                size=result.size,
                confidence=result.confidence,
                context=debug_ctx,
                rationale=result.rationale,
                portfolio_health=self._portfolio_health_score,
            )
            return result

        # Core branching
        if not has_position:
            decision = PositionDecision.OPEN_LONG if signal_direction > 0 else PositionDecision.OPEN_SHORT
            intensity = signal_strength
            confidence = self._calculate_confidence(context, decision)
            size = self._calculate_position_size(context, intensity, confidence)
            rationale["stage"] = "new_position"
            rationale["factors"].append(f"Strong signal {signal_strength:.3f} for new position")
        else:
            current_side = int(np.sign(self.open_positions[instrument].get("side", 0)))
            pnl_eur = self._get_unrealised_pnl_from_bus(instrument)
            signal_aligns = (current_side > 0 and signal_direction > 0) or (current_side < 0 and signal_direction < 0)

            if signal_aligns and signal_strength > self.C.position_scale_threshold:
                decision = PositionDecision.SCALE_UP
                intensity = min(signal_strength * 0.8, 0.9)
                confidence = self._calculate_confidence(context, decision)
                size = self._calculate_position_size(context, intensity, confidence) * 0.5
                rationale["stage"] = "scale_up"
                rationale["factors"].append(f"Signal {signal_strength:.3f} aligns with position, scaling up")
            elif not signal_aligns and signal_strength > 0.5:
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
                    size = max(self._calculate_position_size(context, intensity, confidence) * 0.5, 0.0)
                    rationale["stage"] = "scale_down"
                    rationale["factors"].append(f"Opposing signal {signal_strength:.3f}, reducing exposure")
            elif pnl_eur < -self.C.hard_loss_eur * 0.5:
                decision = PositionDecision.CLOSE
                intensity = 0.8
                confidence = 0.9
                rationale["stage"] = "risk_management"
                rationale["factors"].append(f"Position approaching loss limit: EUR {pnl_eur:.2f}")

        # Risk adjustments
        risk_factors = self._assess_risk_factors(context)
        final_intensity = intensity * (1.0 - max(risk_factors.values()) if risk_factors else 1.0)
        final_confidence = self._calculate_confidence(context, decision) * portfolio_health_score if decision != PositionDecision.HOLD else confidence * portfolio_health_score

        # Minimum viable sizing rules
        if decision in [PositionDecision.OPEN_LONG, PositionDecision.OPEN_SHORT, PositionDecision.SCALE_UP]:
            if size < context.balance * self.C.min_size_pct and final_intensity > 0.3:
                size = context.balance * self.C.min_size_pct
            elif size < context.balance * self.C.min_size_pct:
                size = 0.0
                decision = PositionDecision.HOLD
                rationale["factors"].append("Size too small, holding instead")

        # Build result
        result = PositionDecisionResult(
            decision=decision,
            intensity=float(np.clip(final_intensity, 0.0, 1.0)),
            size=float(size),
            confidence=float(np.clip(final_confidence, 0.0, 1.0)),
            rationale=rationale,
            risk_factors=risk_factors,
            context=context,
        )

        # Integrated debug log
        debug_context = {
            "volatility": context.volatility,
            "current_exposure": context.current_exposure,
            "drawdown": context.drawdown,
        }
        # Integrated debug log
        try:
            if self.debug:
                self.logger.debug(
                    format_operator_message(
                        icon="[POS]",
                        message="Decision",
                        instrument=instrument,
                        decision=result.decision.value,
                        intensity=f"{result.intensity:.3f}",
                        size_eur=f"{result.size:.2f}",
                        confidence=f"{result.confidence:.3f}",
                        signal_strength=f"{signal_strength:.3f}",
                        volatility=f"{context.volatility:.4f}",
                    )
                )
        except Exception:
            pass
        self.debugger.log_decision(
            instrument=instrument,
            decision=result.decision.value,
            intensity=result.intensity,
            size=result.size,
            confidence=result.confidence,
            context=debug_context,
            rationale=result.rationale,
            portfolio_health=self._portfolio_health_score,
        )

        return result

    def _update_position_health(self) -> None:
        try:
            self._refresh_positions_from_bus()
            current_exposure = self._calculate_current_exposure_ratio()
            self._total_exposure_ratio = current_exposure

            self._exposure_history.append(
                {
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
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
                rl = tri.get("risk_level", tri.get("risk_score", 0.0))
                try:
                    risk_level = float(rl if rl is not None else 0.0)
                except Exception:
                    risk_level = 0.0
            elif isinstance(mc, dict):
                rl = mc.get("risk_level", mc.get("risk_score", 0.0))
                try:
                    risk_level = float(rl if rl is not None else 0.0)
                except Exception:
                    risk_level = 0.0

            risk_level = float(np.clip(risk_level, 0.0, 1.0))
            self._risk_management_score = max(0.1, 1.0 - risk_level)

            # Adapt parameters and publish a compact health node
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
        except Exception as e:
            self.logger.warning(f"Position health update failed: {e}")

    def _adapt_parameters(self) -> None:
        """Adapt risk & sensitivity based on recent outcomes (env-only)."""
        try:
            # Risk ceiling adaptation around config.max_position_pct
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

            # Signal sensitivity based on non-hold confidence
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
        except Exception as e:
            self.logger.warning(f"Parameter adaptation failed: {e}")

    def _calculate_current_exposure_ratio(self) -> float:
        """Î£|notional_eur| / balance from optional Executor bus snapshots."""
        balance, _ = self._read_balance_and_drawdown()

        total_exposure = 0.0
        try:
            positions = self.smart_bus.get("positions", "PositionManager")
            if isinstance(positions, dict):
                for p in positions.values():
                    try:
                        notional = float(p.get("notional_eur", 0.0))
                        if notional == 0.0:
                            # fallback: abs(units*entry)
                            units = float(p.get("units", 0.0) or 0.0)
                            entry = float(p.get("entry_price", 0.0) or 0.0)
                            notional = abs(units) * entry
                        total_exposure += abs(notional)
                    except Exception:
                        continue
        except Exception:
            pass

        return total_exposure / max(balance, 1.0)

    def _assess_signal_quality(self) -> float:
        """Heuristic signal quality from recent intensities."""
        if not self.signal_history:
            return 0.5
        quality_scores: List[float] = []
        for _, history in self.signal_history.items():
            if len(history) >= 5:
                recent = history[-5:]
                signal_strength = float(np.mean(np.abs(recent)))
                signal_consistency = float(1.0 - np.std(recent))
                inst_quality = (signal_strength + max(0.0, signal_consistency)) / 2.0
                quality_scores.append(inst_quality)
        return float(np.mean(quality_scores)) if quality_scores else 0.5

    # ================================================================
    # STATE PERSISTENCE
    # ================================================================
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
            # TradingConfig has computed/alias fields (e.g., max_steps_per_episode with init=False)
            # Filter out non-init keys to avoid __init__ errors on restore.
            cfg_in = dict(state["config"])  # shallow copy
            cfg_in.pop("max_steps_per_episode", None)
            try:
                self.C = TradingConfig(**cfg_in)
            except TypeError:
                # In case other non-init keys sneak in, drop unknowns conservatively.
                # Keep only attributes present on a default instance that are not callables/dunder.
                default_cfg = TradingConfig()
                allowed = {k for k in vars(default_cfg).keys()}
                sanitized = {k: v for k, v in cfg_in.items() if k in allowed}
                self.C = TradingConfig(**sanitized)
            self.config.update(self.C.__dict__)
            # keep dependent cached values in sync
            self.default_max_pct = self.C.max_position_pct

        # Genome / params
        if isinstance(state.get("genome"), dict):
            self._initialize_genome_parameters(state["genome"])

        # Positions snapshot (mirror from Executor)
        if isinstance(state.get("open_positions"), dict):
            try:
                self.open_positions = {str(k): dict(v) for k, v in state["open_positions"].items()}
            except Exception:
                self.open_positions = {}

        # Simple scalars
        try:
            self._portfolio_health_score = float(state.get("portfolio_health_score", self._portfolio_health_score))
        except Exception:
            pass
        try:
            self._total_exposure_ratio = float(state.get("exposure_ratio", self._total_exposure_ratio))
        except Exception:
            pass
        try:
            self.consecutive_losses = int(state.get("consecutive_losses", self.consecutive_losses))
        except Exception:
            pass

        # Adaptive params
        if isinstance(state.get("adaptive_params"), dict):
            try:
                self._adaptive_params.update({k: state["adaptive_params"][k] for k in state["adaptive_params"]})
            except Exception:
                pass

        # Position confidence
        if isinstance(state.get("position_confidence"), dict):
            try:
                self.position_confidence = {
                    str(k): float(v) for k, v in state["position_confidence"].items() if isinstance(v, (int, float))
                }
            except Exception:
                self.position_confidence = {}

        # Signal history (cap 50 like runtime)
        if isinstance(state.get("signal_history"), dict):
            try:
                for inst in self.instruments:
                    vals = state["signal_history"].get(inst, [])
                    if isinstance(vals, list):
                        self.signal_history[inst] = [float(x) for x in vals][-50:]
            except Exception:
                pass

        # Decision history
        if isinstance(state.get("decision_history"), list):
            self._decision_history.clear()
            for d in state["decision_history"][-100:]:
                try:
                    self._decision_history.append(d)
                except Exception:
                    continue

        # Portfolio health history
        if isinstance(state.get("portfolio_health_history"), list):
            self._portfolio_health_history.clear()
            for d in state["portfolio_health_history"][-50:]:
                try:
                    self._portfolio_health_history.append(d)
                except Exception:
                    continue

        # Last decisions (reconstruct minimal objects)
        if isinstance(state.get("last_decisions"), dict):
            self.last_decisions.clear()
            for inst, d in state["last_decisions"].items():
                try:
                    dec = PositionDecision(str(d.get("decision", "hold")))
                except Exception:
                    dec = PositionDecision.HOLD
                intensity = float(d.get("intensity", 0.0) or 0.0)
                confidence = float(d.get("confidence", 0.5) or 0.5)
                rationale = d.get("rationale", {}) if isinstance(d.get("rationale", {}), dict) else {}
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

        # Success / failure counters
        try:
            self.success_count = int(state.get("success_count", getattr(self, "success_count", 0)))
        except Exception:
            self.success_count = getattr(self, "success_count", 0)
        try:
            self.failure_count = int(state.get("failure_count", getattr(self, "failure_count", 0)))
        except Exception:
            self.failure_count = getattr(self, "failure_count", 0)
    ####################################################################################################################
    # Small helpers
    ####################################################################################################################
    def _map_decision_to_intensity(self, dr: PositionDecisionResult) -> float:
        """Map a decision into a signed intensity for compact instrument_signals."""
        d = dr.decision
        mag = float(np.clip(dr.intensity, 0.0, 1.0))
        if d == PositionDecision.OPEN_LONG:
            return +mag
        if d == PositionDecision.OPEN_SHORT:
            return -mag
        if d == PositionDecision.SCALE_UP:
            # assume scaling in current direction; use sign from market_direction
            return float(np.sign(dr.context.market_direction or 1)) * mag
        if d == PositionDecision.SCALE_DOWN:
            return -float(np.sign(dr.context.market_direction or 1)) * mag
        # close/emergency_close/hold -> neutral (0)
        return 0.0

    def _signals_map_from_decisions(self, decisions: Dict[str, PositionDecisionResult]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for inst in self.instruments:
            dr = decisions.get(inst)
            if dr is None:
                out[inst] = {"intensity": 0.0, "decision": "hold", "confidence": 0.0}
                continue
            out[inst] = {
                "intensity": float(np.clip(self._map_decision_to_intensity(dr), -1.0, 1.0)),
                "decision": dr.decision.value,
                "confidence": float(np.clip(dr.confidence, 0.0, 1.0)),
            }
        return out

    def _default_signals(self) -> Dict[str, Any]:
        return {inst: {"intensity": 0.0, "decision": "hold", "confidence": 0.0} for inst in self.instruments}

    def __del__(self):
        """Cleanup - debug summary disabled."""
        # Summary printing disabled - using beautiful visualizer instead
        pass

