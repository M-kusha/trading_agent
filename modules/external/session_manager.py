from __future__ import annotations

import time
import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Deque, List, Union, Tuple
from collections import deque

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.utils.audit_utils import RotatingLogger

# Optional: use the real InfoBus if available
try:
    from modules.utils.info_bus import InfoBusManager  # type: ignore
except Exception:  # pragma: no cover
    InfoBusManager = None  # type: ignore

@dataclass
class SessionConfig:
    """Enhanced configuration for Session Manager with validation."""

    session_duration: int = field(default=3600, metadata={
        'validator': lambda x: isinstance(x, int) and x > 0,
        'description': 'Duration in seconds before suggesting a reset/roll'
    })
    performance_window: int = field(default=500, metadata={
        'validator': lambda x: isinstance(x, int) and x > 0,
        'description': 'History length for internal counters'
    })
    enable_health_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_error_pinpointing: bool = True

    def __post_init__(self):
        """Validate configuration values after initialization."""
        for field_name, field_info in self.__dataclass_fields__.items():
            validator = field_info.metadata.get('validator')
            if validator:
                value = getattr(self, field_name)
                if not validator(value):
                    raise ValueError(f"Invalid {field_name}: {value}")

class SessionTimeHelper:
    """Optimized time helper to cache UTC calls and reduce datetime overhead."""

    def __init__(self):
        self._cached_now: Optional[float] = None
        self._cached_utcnow: Optional[datetime.datetime] = None
        self._cache_time: float = 0
        self._CACHE_DURATION: float = 0.1  # Cache for 100ms

    def get_current_time(self) -> float:
        """Get cached current time with refresh interval."""
        now = time.time()
        if now - self._cache_time > self._CACHE_DURATION or self._cached_now is None:
            self._cached_now = now
            self._cache_time = now
        return self._cached_now

    def get_utcnow(self) -> datetime.datetime:
        """Get cached UTC datetime with refresh interval."""
        now = time.time()
        if now - self._cache_time > self._CACHE_DURATION or self._cached_utcnow is None:
            self._cached_utcnow = datetime.datetime.utcnow()
            self._cache_time = now
        return self._cached_utcnow

@dataclass
class SessionData:
    """Structured data class for session information."""
    metrics: Dict[str, Any]
    health: Dict[str, Any]
    context: Dict[str, Any]
    pnl_data: Dict[str, Any]
    trading_result: Dict[str, Any]
    performance_data: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    system_performance: Dict[str, Any]
    system_health: Dict[str, Any]

@module(**module_args(
    "SessionManager",
    description="Enhanced session timing & health context. Returns all contract keys; PnL keys are pass-through of env outputs.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class SessionManager(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Enhanced Session Manager with optimizations and better error handling.

    Responsibilities:
        - Track session lifecycle (start, duration) with optimized timing
        - Expose compact session/health/performance context
        - Strict contract discipline: all required top-level keys always present
        - PnL keys namespaced to avoid clashes with Executor
        - Health snapshot namespaced to avoid clashes with HealthMonitor
        - Cached time calculations and improved confidence logic
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Validate and set configuration first
        try:
            self.cfg = SessionConfig(**(config or {}))
        except ValueError as e:
            # Create logger for error reporting if validation fails
            logger = RotatingLogger("SessionManager", log_path="logs/external/session_manager.log")
            logger.error(f"Invalid SessionConfig parameters: {e}")
            raise

        # Cached time helper for performance optimizations - create before BaseModule init
        self.time_helper = SessionTimeHelper()

        # Session state with improved typing - initialize before BaseModule init
        self.session_start_ts: float = self.time_helper.get_current_time()
        self.session_id: str = f"session_{int(self.session_start_ts)}"
        self.session_status: str = "active"

        # Performance counters with enhanced tracking
        self._success: int = 0
        self._fail: int = 0
        self._proc_times: Deque[float] = deque(maxlen=self.cfg.performance_window)

        # Health / alerts with structured logging
        self.system_alerts: List[Dict[str, Any]] = []
        self._last_health_check: float = self.time_helper.get_current_time()

        # Enhanced session labels with caching
        self._last_label_update: float = 0
        self.trading_session: str = "london"
        self.session_type: str = "normal"

        # Now safe to initialize BaseModule (which calls _initialize)
        super().__init__(config=asdict(self.cfg))

        # Finally create logger after BaseModule init
        self.logger = RotatingLogger("SessionManager", log_path="logs/external/session_manager.log")

    def _initialize(self) -> None:
        """Enhanced initialization with validation."""
        self._update_session_labels()
        self.logger.info("[OK] Enhanced SessionManager initialized with improvements.")
        self.logger.debug(f"Configuration: session_duration={self.cfg.session_duration}s, performance_window={self.cfg.performance_window}")
        # Seed canonical mode keys early to avoid initial BUS MISS
        try:
            # Prefer mixin bus if present; otherwise use global instance
            bus = None
            try:
                bus = getattr(self, 'smart_bus', None)
            except Exception:
                bus = None
            if bus is None:
                try:
                    from modules.utils.info_bus import InfoBusManager as _IBM
                    bus = _IBM.get_instance() if _IBM else None
                except Exception:
                    bus = None

            if bus is not None:
                try:
                    bus.declare_owner('execution_mode', 'SessionManager')
                except Exception:
                    pass
                mode_value = 'sim'
                try:
                    # If env config already present, respect its mode
                    env_cfg = bus.get('environment_config', 'SessionManager') or {}
                    mv = str(env_cfg.get('mode', mode_value)).lower()
                    if mv in ('sim', 'live'):
                        mode_value = mv
                except Exception:
                    pass
                try:
                    bus.set('execution_mode', mode_value, module='SessionManager', thesis='Canonical execution mode (init)')
                except Exception:
                    pass
                try:
                    bus.set('env_mode', mode_value, module='SessionManager', thesis='Legacy alias: env_mode (init)')
                except Exception:
                    pass
        except Exception:
            pass

    def _update_session_labels(self, current_hour: Optional[int] = None) -> None:
        """Optimized session labels with explicit time parameter for testing."""
        # Use cached UTC now instead of multiple calls
        if current_hour is None:
            utcnow = self.time_helper.get_utcnow()
            current_hour = utcnow.hour

        # Trading session based on hour
        if 8 <= current_hour < 16:
            self.trading_session = "london"
        elif 13 <= current_hour < 21:
            self.trading_session = "new_york"
        elif 21 <= current_hour or current_hour < 6:
            self.trading_session = "sydney"
        else:
            self.trading_session = "tokyo"

        # Session type based on hour
        if 9 <= current_hour < 17:
            self.session_type = "main"
        elif 17 <= current_hour < 21:
            self.session_type = "overlap"
        else:
            self.session_type = "overnight"

    def _session_canonical(self, current_hour: Optional[int] = None) -> str:
        """Optimized session canonical calculation."""
        if current_hour is None:
            utcnow = self.time_helper.get_utcnow()
            current_hour = utcnow.hour

        if 0 <= current_hour < 8:
            return "asian"
        if 8 <= current_hour < 16:
            return "european"
        if 16 <= current_hour < 22:
            return "us"
        return "closed"

    def _get_bus_data(self) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Any], Dict[str, Any], Optional[Any]]:
        """Extract common bus data retrieval logic."""
        # Use getattr to satisfy type checker for mixin methods
        bus_get = getattr(self, '_bus_get', None)

        if bus_get is None:
            return {}, {}, None, {}, None  # Return defaults if method unavailable

        perf_data_bus = bus_get("performance_data", None)
        portfolio_metrics = bus_get("portfolio_metrics", None)
        trading_result_bus = bus_get("trading_result", None)
        env_cfg_bus = bus_get("environment_config", None)
        system_health_bus = bus_get("system_health", None)

        return (
            perf_data_bus if isinstance(perf_data_bus, dict) else {},
            env_cfg_bus if isinstance(env_cfg_bus, dict) else {},
            portfolio_metrics,  # This can be None
            trading_result_bus if isinstance(trading_result_bus, dict) else {},
            system_health_bus  # This can be None
        )

    def _build_system_performance(self, is_error_case: bool = False) -> Dict[str, Any]:
        """Extracted system performance calculation."""
        total_operations = self._success + self._fail
        return {
            "success_count": int(self._success),
            "failure_count": int(self._fail),
            "success_rate": float(self._success / max(1, total_operations)),
            "avg_processing_time_ms": float(np.mean(self._proc_times)) if self._proc_times else 0.0,
            "last_check": self.time_helper.get_utcnow().isoformat(),
        }

    def _build_session_metrics(self, current_time: float) -> Dict[str, Any]:
        """Extracted session metrics calculation."""
        return {
            "session_id": self.session_id,
            "duration": float(current_time - self.session_start_ts),
            "status": self.session_status,
            "start_time": datetime.datetime.utcfromtimestamp(self.session_start_ts).isoformat(),
        }

    def _build_session_health(self) -> Dict[str, Any]:
        """Extracted session health calculation."""
        return {
            "status": "healthy" if len(self.system_alerts) == 0 else "degraded",
            "alerts": list(self.system_alerts[-25:]),
        }

    def _build_session_context(self, current_hour: Optional[int] = None) -> Dict[str, Any]:
        """Extracted session context calculation with caching."""
        return {
            "session_canonical": self._session_canonical(current_hour),
            "trading_session": self.trading_session,
            "session_type": self.session_type,
        }

    def _build_system_health(self, session_health: Dict[str, Any], system_performance: Dict[str, Any],
                           system_health_bus: Optional[Any]) -> Dict[str, Any]:
        """Extracted system health calculation."""
        if isinstance(system_health_bus, dict):
            return system_health_bus

        return {
            "status": session_health.get("status", "unknown"),
            "alerts": list(self.system_alerts[-25:]),
            "last_check": self.time_helper.get_utcnow().isoformat(),
            "success_rate": system_performance.get("success_rate"),
            "avg_processing_time_ms": system_performance.get("avg_processing_time_ms"),
        }

    def _build_pnl_data(self, portfolio_metrics: Optional[Any], trading_result_bus: Optional[Any]) -> Dict[str, Any]:
        """Extracted PnL data calculation."""
        session_pnl_data: Dict[str, Any] = {}

        if isinstance(portfolio_metrics, dict) and portfolio_metrics:
            session_pnl_data.update({
                "balance": portfolio_metrics.get("balance"),
                "equity": portfolio_metrics.get("equity"),
                "current_pnl": portfolio_metrics.get("current_pnl"),
                "step": portfolio_metrics.get("step"),
            })

            if isinstance(trading_result_bus, dict) and "pnl" in trading_result_bus:
                session_pnl_data["last_step_pnl"] = trading_result_bus.get("pnl")

        return session_pnl_data

    def _build_snapshot(self, session_data: SessionData, error_message: Optional[str] = None) -> Dict[str, Any]:
        """Build the complete snapshot from session data."""
        snapshot: Dict[str, Any] = {
            # Required contract outputs (always present)
            "consensus_data": {},
            "emergency_mode": False,
            "episode_data": {},
            "episode_summary": {},
            "market_open": True,
            "memory_usage": {},
            "mistakes": [],
            "module_performance": {},
            "performance_data": session_data.performance_data,
            "playbook_entries": [],
            "playbook_memory": {},
            "session_pnl_data": session_data.pnl_data,
            "performance_metrics": session_data.performance_metrics,
            "session_context": session_data.context,
            "session_metrics": session_data.metrics,
            "system_alerts": list(self.system_alerts[-25:]),
            "session_health": session_data.health,
            "system_performance": session_data.system_performance,
            "system_health": session_data.system_health,
            "environment_config": {},
            "trading_result": session_data.trading_result,
            "execution_mode": "sim",
        }

        # Build thesis with error context if present
        snapshot["_thesis"] = self._build_session_thesis(
            session_metrics=session_data.metrics,
            session_health=session_data.health,
            system_performance=session_data.system_performance,
            session_pnl_data=session_data.pnl_data,
            trading_result=session_data.trading_result,
            session_context=session_data.context,
            performance_data=session_data.performance_data,
            performance_metrics=session_data.performance_metrics,
            error_message=error_message,
        )

        return snapshot

    def _build_session_thesis(self, *,
                             session_metrics: Dict[str, Any],
                             session_health: Dict[str, Any],
                             system_performance: Dict[str, Any],
                             session_pnl_data: Dict[str, Any],
                             trading_result: Dict[str, Any],
                             session_context: Dict[str, Any],
                             performance_data: Dict[str, Any],
                             performance_metrics: Dict[str, Any],
                             error_message: Optional[str] = None) -> str:
        """Enhanced thesis builder broken into focused sub-methods."""
        parts: List[str] = []

        # Session overview section
        parts.extend(self._build_thesis_session_overview(session_metrics, session_context))

        # Health section
        parts.append(self._build_thesis_health_section(session_health))

        # Performance sections
        parts.append(self._build_thesis_performance_section(system_performance))
        parts.append(self._build_thesis_pnl_section(session_pnl_data))
        parts.append(self._build_thesis_performance_data_section(performance_data))

        # Portfolio and trading sections
        parts.append(self._build_thesis_portfolio_section(performance_metrics))
        parts.append(self._build_thesis_trading_section(trading_result))

        # Error section if applicable
        if error_message:
            parts.append(f"Operating in degraded mode due to {self._excerpt_text(error_message)}.")

        return " ".join(part.strip() for part in parts if part).strip() or "Session status available but no detailed context was produced."

    def _build_thesis_session_overview(self, session_metrics: Dict[str, Any], session_context: Dict[str, Any]) -> List[str]:
        """Build session overview section."""
        parts = []

        metrics_source = session_metrics or {}
        session_id = metrics_source.get("session_id") or "session"
        status = metrics_source.get("status") or "unknown"

        duration_minutes: Optional[float] = None
        try:
            duration_minutes = float(metrics_source.get("duration", 0.0)) / 60.0
        except (TypeError, ValueError):
            pass

        context_bits: List[str] = []
        context_source = session_context or {}
        for key in ("trading_session", "session_type", "session_canonical"):
            val = context_source.get(key)
            if val and str(val) not in context_bits:
                context_bits.append(str(val))

        base_sentence = f"Session {session_id} {status}"
        if duration_minutes is not None:
            base_sentence += f" for {duration_minutes:.1f}m"
        if context_bits:
            base_sentence += f" ({'/'.join(context_bits)})"

        parts.append(base_sentence + ".")
        return parts

    def _build_thesis_health_section(self, session_health: Dict[str, Any]) -> str:
        """Build health section."""
        health_source = session_health or {}
        health_status = health_source.get("status") or "unknown"
        alerts = health_source.get("alerts") or []

        if alerts:
            latest_alert = alerts[-1]
            alert_message = ""
            if isinstance(latest_alert, dict):
                alert_message = latest_alert.get("message") or latest_alert.get("detail") or ""
            if alert_message:
                return f"Health {health_status} with {len(alerts)} alert(s); latest: {self._excerpt_text(alert_message)}."
            else:
                return f"Health {health_status} with {len(alerts)} alert(s)."
        else:
            return f"Health {health_status} with no outstanding alerts."

    def _build_thesis_performance_section(self, system_performance: Dict[str, Any]) -> str:
        """Build performance stats section."""
        perf_source = system_performance or {}
        perf_bits: List[str] = []

        success_count = perf_source.get("success_count")
        failure_count = perf_source.get("failure_count")
        if success_count is not None or failure_count is not None:
            perf_bits.append(f"{int(success_count or 0)} success / {int(failure_count or 0)} fail")

        success_rate = self._format_percentage(perf_source.get("success_rate"))
        if success_rate:
            perf_bits.append(f"success rate {success_rate}")

        avg_latency = self._format_float(perf_source.get("avg_processing_time_ms"), precision=1)
        if avg_latency:
            perf_bits.append(f"avg {avg_latency} ms latency")

        return "Processing stats: " + ", ".join(perf_bits) + "." if perf_bits else ""

    def _build_thesis_pnl_section(self, session_pnl_data: Dict[str, Any]) -> str:
        """Build PnL section."""
        pnl_source = session_pnl_data if isinstance(session_pnl_data, dict) else {}
        if not pnl_source:
            return ""

        pnl_bits: List[str] = []

        # Add step information
        step_value = pnl_source.get("step")
        if step_value is not None:
            try:
                pnl_bits.append(f"step {int(step_value)}")
            except (TypeError, ValueError):
                pnl_bits.append(f"step {step_value}")

        # Add PnL metrics
        for key, label in (
            ("current_pnl", "current PnL"),
            ("balance", "balance"),
            ("equity", "equity"),
            ("last_step_pnl", "last step PnL"),
        ):
            if key in pnl_source:
                formatted = self._format_float(pnl_source.get(key), precision=2)
                pnl_bits.append(
                    f"{label} {formatted}" if formatted is not None else f"{label} {pnl_source.get(key)}"
                )

        return "PnL snapshot: " + ", ".join(pnl_bits) + "." if pnl_bits else ""

    def _build_thesis_performance_data_section(self, performance_data: Dict[str, Any]) -> str:
        """Build performance data section."""
        perf_data_source = performance_data if isinstance(performance_data, dict) else {}
        perf_data_bits: List[str] = []

        win_rate = self._format_percentage(perf_data_source.get("win_rate"))
        if win_rate:
            perf_data_bits.append(f"win rate {win_rate}")

        expectancy = self._format_float(perf_data_source.get("expectancy"), precision=2)
        if expectancy:
            perf_data_bits.append(f"expectancy {expectancy}")

        avg_trade = self._format_float(perf_data_source.get("avg_trade_pnl"), precision=2)
        if avg_trade:
            perf_data_bits.append(f"avg trade PnL {avg_trade}")

        return "Performance data: " + ", ".join(perf_data_bits) + "." if perf_data_bits else ""

    def _build_thesis_portfolio_section(self, performance_metrics: Dict[str, Any]) -> str:
        """Build portfolio metrics section."""
        portfolio_metrics = {}
        if isinstance(performance_metrics, dict):
            candidate = performance_metrics.get("portfolio")
            if isinstance(candidate, dict):
                portfolio_metrics = candidate

        if not portfolio_metrics:
            return ""

        portfolio_bits: List[str] = []
        for key, label in (
            ("max_drawdown", "max drawdown"),
            ("exposure", "exposure"),
            ("volatility", "volatility"),
        ):
            formatted = self._format_float(portfolio_metrics.get(key), precision=2)
            if formatted:
                portfolio_bits.append(f"{label} {formatted}")

        return "Portfolio metrics: " + ", ".join(portfolio_bits) + "." if portfolio_bits else ""

    def _build_thesis_trading_section(self, trading_result: Dict[str, Any]) -> str:
        """Build trading result section."""
        trading_notes: List[str] = []

        if isinstance(trading_result, dict) and trading_result:
            # Status information
            status_value = trading_result.get("status")
            if status_value:
                trading_notes.append(str(status_value))

            # PnL information
            pnl_value = self._format_float(trading_result.get("pnl"), precision=2)
            if pnl_value:
                trading_notes.append(f"pnl {pnl_value}")

            # Position and order information
            for field_key, display_name in [
                ("open_positions", "open positions"),
                ("position_count", "open positions"),
                ("executed_orders", "executed orders")
            ]:
                value = trading_result.get(field_key)
                if value is not None:
                    try:
                        trading_notes.append(f"{int(value)} {display_name}")
                    except (TypeError, ValueError):
                        trading_notes.append(f"{display_name} {value}")

        return "Trading result: " + ", ".join(trading_notes) + "." if trading_notes else ""

    def _format_float(self, value: Any, precision: int = 1) -> Optional[str]:
        """Enhanced float formatting with better error handling."""
        try:
            if isinstance(value, (int, float)):
                return f"{float(value):.{precision}f}"
            else:
                return None
        except (TypeError, ValueError, OverflowError):
            return None

    def _format_percentage(self, value: Any) -> Optional[str]:
        """Enhanced percentage formatting."""
        try:
            if isinstance(value, (int, float)):
                return f"{float(value) * 100:.1f}%"
            else:
                return None
        except (TypeError, ValueError, OverflowError):
            return None

    def _excerpt_text(self, text: Any, limit: int = 160) -> str:
        """Enhanced text excerpting."""
        if not text:
            return ""
        value = " ".join(str(text).split())
        return value if len(value) <= limit else value[: limit - 3] + "..."

    async def calculate_confidence(self, action: Optional[Dict[str, Any]] = None, **inputs) -> float:
        """
        Enhanced confidence calculation with better logic.

        Confidence reflects recency of health checks and meaningful counter data.
        Fixed logic to properly validate counter usage rather than always returning 1.0.
        """
        now = self.time_helper.get_current_time()
        freshness = max(0.0, 1.0 - (now - self._last_health_check) / 60.0)

        # Fixed: Check if counters are meaningful (have been used)
        total_operations = self._success + self._fail
        counters_ok = 1.0 if total_operations > 0 else 0.0  # Changed from always 1.0

        # Enhanced weighting - freshness more important for recent activity
        confidence = 0.7 * freshness + 0.3 * counters_ok

        return float(min(1.0, max(0.0, confidence)))

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Enhanced session maintenance proposal with timing optimization."""
        current_time = self.time_helper.get_current_time()
        duration = current_time - self.session_start_ts

        return {
            "update_session": True,
            "reset_session": bool(duration >= self.cfg.session_duration),
            "reason": "roll session after configured duration" if duration >= self.cfg.session_duration else None,
        }

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Enhanced process method with extracted common logic and optimizations.

        CONTRACT-ALIGNED TOP-LEVEL OUTPUTS (always present):
            consensus_data, emergency_mode, episode_data, episode_summary, expert_votes,
            market_open, memory_usage, mistakes, module_performance, performance_data,
            playbook_entries, playbook_memory, session_pnl_data, session_context, session_metrics,
            system_alerts, session_health, system_performance, trading_result

        PnL policy:
          - Forward env bus values when available.
          - Otherwise return {} for PnL-related keys (never fabricate numbers).
        """
        t0 = self.time_helper.get_current_time()
        # Pre-initialize to satisfy type checker and ensure availability in error path
        env_cfg_out: Dict[str, Any] = {}

        try:
            # Optimized session updates with caching
            self._update_session_labels()
            current_time = self.time_helper.get_current_time()
            self._last_health_check = current_time

            # Build core session data using helper methods
            session_metrics = self._build_session_metrics(current_time)
            system_performance = self._build_system_performance()
            session_health = self._build_session_health()
            session_context = self._build_session_context()

            # Extract bus data once for both success and error cases
            (performance_data, environment_config, portfolio_metrics,
             trading_result, system_health_bus) = self._get_bus_data()

            # Build derived data
            session_pnl_data = self._build_pnl_data(portfolio_metrics, trading_result)
            system_health = self._build_system_health(session_health, system_performance, system_health_bus)

            # Build performance metrics
            performance_metrics: Dict[str, Any] = {
                "system_performance": system_performance,
                "system_health": session_health,
                "environment_config": environment_config,
                "session_pnl": dict(session_pnl_data) if session_pnl_data else {},
            }
            if isinstance(portfolio_metrics, dict) and portfolio_metrics:
                performance_metrics["portfolio"] = dict(portfolio_metrics)

            # Create structured session data
            session_data = SessionData(
                metrics=session_metrics,
                health=session_health,
                context=session_context,
                pnl_data=session_pnl_data,
                trading_result=trading_result,
                performance_data=performance_data,
                performance_metrics=performance_metrics,
                system_performance=system_performance,
                system_health=system_health
            )

            # Canonicalize environment_config using performance_data as authority for initial_balance
            try:
                env_cfg_out = dict(environment_config) if isinstance(environment_config, dict) else {}
                pd_ib = None
                try:
                    if isinstance(performance_data, dict):
                        if isinstance(performance_data.get('initial_balance'), (int, float)):
                            pd_ib = float(performance_data['initial_balance'])
                        elif isinstance(performance_data.get('starting_balance'), (int, float)):
                            pd_ib = float(performance_data['starting_balance'])
                except Exception:
                    pd_ib = None
                if pd_ib is not None:
                    # Enforce authoritative initial balance from env/provider
                    env_cfg_out['initial_balance'] = pd_ib
                # Publish canonical environment_config (SessionManager is the owner)
                try:
                    # Reinforce ownership to avoid churn from any prior Environment attempts
                    try:
                        self.smart_bus.declare_owner('environment_config', 'SessionManager')
                    except Exception:
                        pass
                    self.smart_bus.set(
                        'environment_config',
                        env_cfg_out,
                        module='SessionManager',
                        thesis='Canonical environment config (synced)'
                    )
                    mode_value = str(env_cfg_out.get('mode', 'sim')).lower()
                    try:
                        self.smart_bus.declare_owner('execution_mode', 'SessionManager')
                    except Exception:
                        pass
                    self.smart_bus.set(
                        'execution_mode',
                        mode_value,
                        module='SessionManager',
                        thesis='Canonical execution mode'
                    )
                    try:
                        self.smart_bus.set(
                            'env_mode',
                            mode_value,
                            module='SessionManager',
                            thesis='Legacy alias: env_mode'
                        )
                    except Exception:
                        pass
                    
                    # FIX: Publish emergency_mode to SmartInfoBus (required by contract)
                    try:
                        self.smart_bus.declare_owner('emergency_mode', 'SessionManager')
                    except Exception:
                        pass
                    self.smart_bus.set(
                        'emergency_mode',
                        False,  # Currently hardcoded to False; enhance later if needed
                        module='SessionManager',
                        thesis='Emergency mode status (system-wide kill switch)'
                    )
                except Exception:
                    pass
            except Exception:
                env_cfg_out = {}

            # Build final snapshot
            snapshot = self._build_snapshot(session_data)
            # Include environment_config explicitly to avoid empty contract field
            try:
                snapshot["environment_config"] = env_cfg_out
            except Exception:
                snapshot["environment_config"] = {}

            # Update success metrics
            self._success += 1
            self._proc_times.append((self.time_helper.get_current_time() - t0) * 1000.0)

            return snapshot

        except Exception as exc:
            # Enhanced error handling with structured logging
            self._fail += 1
            error_message = f"{type(exc).__name__}: {str(exc)[:200]}"

            # Log structured error information
            self.system_alerts.append({
                "level": "error",
                "message": f"process() exception: {str(exc)[:200]}",
                "timestamp": self.time_helper.get_current_time(),
                "context": {
                    "success_count": self._success,
                    "fail_count": self._fail,
                    "last_health_check": self._last_health_check
                }
            })

            # Extract bus data (same logic as success case)
            (performance_data, environment_config, portfolio_metrics,
             trading_result, system_health_bus) = self._get_bus_data()

            # Build error case data
            session_pnl_data = self._build_pnl_data(portfolio_metrics, trading_result)
            system_performance_error = self._build_system_performance(is_error_case=True)
            session_health_error = self._build_session_health()

            # Build performance metrics for error case
            performance_metrics: Dict[str, Any] = {
                "system_performance": system_performance_error,
                "system_health": session_health_error,
                "environment_config": environment_config,
                "session_pnl": dict(session_pnl_data) if session_pnl_data else {},
            }
            if isinstance(portfolio_metrics, dict) and portfolio_metrics:
                performance_metrics["portfolio"] = dict(portfolio_metrics)

            # Enhanced error context
            current_time = self.time_helper.get_current_time()
            session_data_error = SessionData(
                metrics={
                    "session_id": self.session_id,
                    "duration": float(current_time - self.session_start_ts),
                    "status": "error",
                    "start_time": datetime.datetime.utcfromtimestamp(self.session_start_ts).isoformat(),
                },
                health={"status": "degraded", "alerts": list(self.system_alerts[-25:])},
                context=self._build_session_context(),
                pnl_data=session_pnl_data,
                trading_result=trading_result,
                performance_data=performance_data,
                performance_metrics=performance_metrics,
                system_performance=system_performance_error,
                system_health={"status": "degraded", "alerts": list(self.system_alerts[-25:])}
            )

            # Build error snapshot
            payload = self._build_snapshot(session_data_error, error_message)
            try:
                payload["environment_config"] = env_cfg_out if isinstance(env_cfg_out, dict) else {}
            except Exception:
                payload["environment_config"] = {}
            return payload
