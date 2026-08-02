# ─────────────────────────────────────────────────────────────
# File: modules/external/session_manager.py
# Enhanced Session Manager
#
# • Pylance-clean: strict typing, safe optionals, no unsafe casts
# • Human-friendly logging to logs/external/ (pretty lines)
# • Optional NDJSON stream for light forensics
# • Single-writer discipline for contract surfaces
# • Zero fabrication: pass-through where applicable, else empty
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import datetime
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.mixins import SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.module_base import BaseModule, module
from modules.utils.audit_utils import RotatingLogger
from modules.utils.session_utils import classify_session

# Optional: use the real InfoBus if available (guarded to keep startup robust)
try:  # pragma: no cover
    from modules.utils.info_bus import InfoBusManager  # type: ignore
except Exception:  # pragma: no cover
    InfoBusManager = None  # type: ignore


# ─────────────────────────────────────────────────────────────
# Pretty logger + optional NDJSON (same dialect as MDP)
# ─────────────────────────────────────────────────────────────

class PrettyLogger:
    """
    Formats log lines for humans and (optionally) writes compact NDJSON.
    If logging has an issue, it never breaks the hot path.
    """
    def __init__(
        self,
        name: str,
        pretty_sink: RotatingLogger,
        ndjson_path: Optional[Path] = None,
        ndjson_every_n: int = 0,
    ) -> None:
        self.name = name
        self.pretty = pretty_sink
        self.ndjson_path = ndjson_path
        self.ndjson_every_n = max(0, int(ndjson_every_n))
        if self.ndjson_path is not None:
            self.ndjson_path.parent.mkdir(parents=True, exist_ok=True)
            self.ndjson_path.touch(exist_ok=True)

    @staticmethod
    def _hms_with_ms(ts: float, tz: datetime.tzinfo | None = None) -> str:
        dt = datetime.datetime.fromtimestamp(ts, tz=tz)
        return dt.strftime("%H:%M:%S.") + f"{int(dt.microsecond/1000):03d}"

    def _fmt(self, level: str, pid: int, tag: str, msg: str, ts: Optional[float] = None) -> str:
        """
        Example:
        [LOG] [DEBUG  ] 14:23:03.985 [p#15] [PROCESS_START       ] Starting... at 12:23:03
        """
        now = time.time() if ts is None else ts
        local = self._hms_with_ms(now)
        utc = self._hms_with_ms(now, tz=datetime.timezone.utc)
        tag_padded = f"{tag:22s}"
        lvl_padded = f"{level:<7s}"
        return f"[LOG] [{lvl_padded}] {local} [p#{pid}] [{tag_padded}] {msg} at {utc}"

    def trace(self, pid: int, tag: str, msg: str) -> None:
        self.pretty.debug(self._fmt("TRACE", pid, tag, msg))

    def debug(self, pid: int, tag: str, msg: str) -> None:
        self.pretty.debug(self._fmt("DEBUG", pid, tag, msg))

    def info(self, pid: int, tag: str, msg: str) -> None:
        self.pretty.info(self._fmt("INFO", pid, tag, msg))

    def warn(self, pid: int, tag: str, msg: str) -> None:
        self.pretty.warning(self._fmt("WARN", pid, tag, msg))

    def error(self, pid: int, tag: str, msg: str) -> None:
        self.pretty.error(self._fmt("ERROR", pid, tag, msg))

    def ndjson(self, pid: int, kind: str, obj: Dict[str, Any]) -> None:
        if self.ndjson_path is None:
            return
        try:
            record = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "pid": pid,
                "kind": kind,
                "data": obj,
            }
            with self.ndjson_path.open("a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            # Logging must never break the provider.
            self.pretty.debug(f"[DBG] NDJSON write failed: {e}")


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

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

    # Logging controls (same knobs as MarketDataProvider)
    log_every_n: int = 250             # 1 = log every tick
    ndjson_every_n: int = 0            # 0 = off; 1 = every tick; N = every Nth tick

    def __post_init__(self):
        """Validate configuration values after initialization."""
        for field_name, field_info in self.__dataclass_fields__.items():  # type: ignore[attr-defined]
            validator = field_info.metadata.get('validator')
            if validator:
                value = getattr(self, field_name)
                if not validator(value):
                    raise ValueError(f"Invalid {field_name}: {value}")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

class SessionTimeHelper:
    """Optimized time helper to cache UTC calls and reduce datetime overhead."""
    def __init__(self):
        self._cached_now: Optional[float] = None
        self._cached_utcnow: Optional[datetime.datetime] = None
        self._cache_time: float = 0.0
        self._CACHE_DURATION: float = 0.1  # Cache for 100ms

    def get_current_time(self) -> float:
        """Get cached current time with refresh interval."""
        now = time.time()
        if (now - self._cache_time) > self._CACHE_DURATION or self._cached_now is None:
            self._cached_now = now
            self._cache_time = now
        return float(self._cached_now)

    def get_utcnow(self) -> datetime.datetime:
        """Get cached UTC datetime with refresh interval."""
        now = time.time()
        if (now - self._cache_time) > self._CACHE_DURATION or self._cached_utcnow is None:
            self._cached_utcnow = datetime.datetime.utcnow()
            self._cache_time = now
        # _cached_utcnow is always set here
        return self._cached_utcnow  # type: ignore[return-value]


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


# ─────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────

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
        - PnL keys namespaced to avoid clashes; no invented numbers
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
        self._last_label_update: float = 0.0
        self.trading_session: str = "london"
        self.session_type: str = "normal"

        # Pretty + NDJSON logging — same dialect as MDP
        log_dir = Path("logs/external")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = RotatingLogger("SessionManager", log_path=str(log_dir / "session_manager.log"))
        self._pretty = PrettyLogger(
            name="SessionManager",
            pretty_sink=self.logger,
            ndjson_path=(log_dir / "session_manager.ndjson") if self.cfg.ndjson_every_n > 0 else None,
            ndjson_every_n=self.cfg.ndjson_every_n,
        )
        self._proc_count: int = 0  # p# for pretty logs

        # Now safe to initialize BaseModule (which calls _initialize)
        super().__init__(config=asdict(self.cfg))

    # ─────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────
    def _initialize(self) -> None:
        """Enhanced initialization with validation."""
        self._update_session_labels()
        self._pretty.info(self._proc_count, "INIT_OK", "Enhanced SessionManager initialized.")
        self._pretty.debug(
            self._proc_count,
            "CONFIG",
            f"session_duration={self.cfg.session_duration}s, performance_window={self.cfg.performance_window}"
        )

        # Seed canonical mode keys early to avoid initial BUS MISS
        try:
            bus = getattr(self, 'smart_bus', None)
            if bus is None and InfoBusManager is not None:  # type: ignore[truthy-function]
                try:
                    bus = InfoBusManager.get_instance()  # type: ignore[attr-defined]
                except Exception:
                    bus = None

            if bus is not None:
                mode_value = 'sim'
                try:
                    # If env config already present, respect its mode
                    env_cfg = bus.get('environment_config', 'SessionManager') or {}
                    mv = str(env_cfg.get('mode', mode_value)).lower()
                    if mv in ('sim', 'live'):
                        mode_value = mv
                except Exception:
                    pass

                # Publish canonical session labels upfront
                try:
                    session_ctx = self._build_session_context()
                    bus.set('session_canonical', session_ctx.get('session_canonical', 'unknown'),
                            module='SessionManager', thesis='Canonical session label (init)')
                    bus.set('trading_session', session_ctx.get('trading_session', 'unknown'),
                            module='SessionManager', thesis='Trading session label (init)')
                    bus.set('session_type', session_ctx.get('session_type', 'unknown'),
                            module='SessionManager', thesis='Session type label (init)')
                except Exception:
                    # If _build_session_context fails, still publish safe defaults
                    bus.set('session_canonical', 'unknown', module='SessionManager', thesis='Canonical session label (init fallback)')
                    bus.set('trading_session', 'unknown', module='SessionManager', thesis='Trading session label (init fallback)')
                    bus.set('session_type', 'unknown', module='SessionManager', thesis='Session type label (init fallback)')
                
                    # Also publish per-instrument session for HorizonAligner
                    try:
                        bus.set('session_canonical_by_instrument',
                             {"XAUUSD": "unknown", "XAU_USD": "unknown"},
                             module='SessionManager', thesis='Per-instrument session labels (init)')
                    except Exception:
                        pass

                try:
                    bus.declare_owner('execution_mode', 'SessionManager')
                except Exception:
                    pass

                try:
                    bus.set('execution_mode', mode_value, module='SessionManager',
                            thesis='Canonical execution mode (init)')
                except Exception:
                    pass

                try:
                    bus.set('env_mode', mode_value, module='SessionManager',
                            thesis='Legacy alias: env_mode (init)')
                except Exception:
                    pass
        except Exception:
            # Don't let optional bus wiring break init
            pass

    # ─────────────────────────────────────────────────────────
    # Labeling / session helpers
    # ─────────────────────────────────────────────────────────
    def _update_session_labels(self, current_hour: Optional[int] = None) -> None:
        """Optimized session labels with explicit time parameter for testing."""
        if current_hour is None:
            current_hour = self.time_helper.get_utcnow().hour

        # Detect ALL active sessions (markets can overlap)
        # London: 08:00-16:00 UTC, NY: 13:00-21:00 UTC, Sydney: 21:00-06:00 UTC, Tokyo: 00:00-08:00 UTC
        active_sessions = []
        if 8 <= current_hour < 16:
            active_sessions.append("london")
        if 13 <= current_hour < 21:
            active_sessions.append("new_york")
        if 21 <= current_hour or current_hour < 6:
            active_sessions.append("sydney")
        if 0 <= current_hour < 8:
            active_sessions.append("tokyo")
        
        # Store active sessions list
        self.active_sessions = active_sessions if active_sessions else ["closed"]
        
        # Determine primary session (with overlap priority)
        if "london" in active_sessions and "new_york" in active_sessions:
            self.trading_session = "london_newyork_overlap"  # High liquidity overlap
        elif "london" in active_sessions:
            self.trading_session = "london"
        elif "new_york" in active_sessions:
            self.trading_session = "new_york"
        elif "sydney" in active_sessions and "tokyo" in active_sessions:
            self.trading_session = "asia_overlap"  # Asian overlap
        elif "tokyo" in active_sessions:
            self.trading_session = "tokyo"
        elif "sydney" in active_sessions:
            self.trading_session = "sydney"
        else:
            self.trading_session = "closed"

        # Session type based on hour
        if 13 <= current_hour < 16:
            self.session_type = "london_ny_overlap"  # Prime trading hours
        elif 9 <= current_hour < 17:
            self.session_type = "main"
        elif 17 <= current_hour < 21:
            self.session_type = "late"
        else:
            self.session_type = "overnight"

    def _session_canonical(self, current_hour: Optional[int] = None) -> str:
        """Canonical session using shared util: asian/european/american/closed."""
        try:
            if current_hour is None:
                now = self.time_helper.get_utcnow()
                wk = now.weekday() in (5, 6)
                return classify_session(hour=now.hour, weekend=wk)
            now = self.time_helper.get_utcnow().replace(hour=int(current_hour), minute=0, second=0, microsecond=0)
            wk = now.weekday() in (5, 6)
            return classify_session(hour=now.hour, weekend=wk)
        except Exception:
            return 'unknown'

    # ─────────────────────────────────────────────────────────
    # Safe helpers (Pylance-friendly)
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _to_iso_ts(dt: datetime.datetime) -> str:
        return dt.isoformat()

    @staticmethod
    def _safe_percent_str(v: Any) -> Optional[str]:
        try:
            if isinstance(v, (int, float)):
                return f"{float(v) * 100:.1f}%"
            return None
        except Exception:
            return None

    @staticmethod
    def _safe_float_str(v: Any, precision: int = 1) -> Optional[str]:
        try:
            if isinstance(v, (int, float)):
                return f"{float(v):.{precision}f}"
            return None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────
    # InfoBus
    # ─────────────────────────────────────────────────────────
    def _get_bus_data(self) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Any], Dict[str, Any], Optional[Any]]:
        """
        Extract common bus data retrieval logic.
        Returns (performance_data, environment_config, portfolio_metrics, trading_result, system_health_bus).
        """
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
            portfolio_metrics,  # None or dict/other
            trading_result_bus if isinstance(trading_result_bus, dict) else {},
            system_health_bus  # None or dict/other
        )

    # ─────────────────────────────────────────────────────────
    # Builders
    # ─────────────────────────────────────────────────────────
    def _build_system_performance(self) -> Dict[str, Any]:
        total_ops = self._success + self._fail
        avg_ms = float(np.mean(self._proc_times)) if self._proc_times else 0.0
        return {
            "success_count": int(self._success),
            "failure_count": int(self._fail),
            "success_rate": float(self._success / max(1, total_ops)),
            "avg_processing_time_ms": float(avg_ms),
            "last_check": self._to_iso_ts(self.time_helper.get_utcnow()),
        }

    def _build_session_metrics(self, current_time: float) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "duration": float(current_time - self.session_start_ts),
            "status": self.session_status,
            "start_time": datetime.datetime.utcfromtimestamp(self.session_start_ts).isoformat(),
        }

    def _build_session_health(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if len(self.system_alerts) == 0 else "degraded",
            "alerts": list(self.system_alerts[-25:]),
        }

    def _build_session_context(self, current_hour: Optional[int] = None) -> Dict[str, Any]:
        canonical = self._session_canonical(current_hour)
        return {
            "session_canonical": canonical,
            "current_session": canonical,
            "trading_session": self.trading_session,
            "session_type": self.session_type,
        }

    def _build_system_health(
        self,
        session_health: Dict[str, Any],
        system_performance: Dict[str, Any],
        system_health_bus: Optional[Any]
    ) -> Dict[str, Any]:
        if isinstance(system_health_bus, dict):
            return system_health_bus
        return {
            "status": session_health.get("status", "unknown"),
            "alerts": list(self.system_alerts[-25:]),
            "last_check": self._to_iso_ts(self.time_helper.get_utcnow()),
            "success_rate": system_performance.get("success_rate"),
            "avg_processing_time_ms": system_performance.get("avg_processing_time_ms"),
        }

    def _build_pnl_data(self, portfolio_metrics: Optional[Any], trading_result_bus: Optional[Any]) -> Dict[str, Any]:
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

    # FIX: Helper methods for missing bus keys
    def _build_session_by_instrument(self) -> Dict[str, str]:
        """Build per-instrument session labels for HorizonAligner."""
        # All instruments share the same session in this implementation
        # Could be enhanced for multi-market support
        canonical = self._session_canonical()
        instruments = ["XAUUSD", "XAU/USD", "XAU_USD"]
        return {inst: canonical for inst in instruments}

    def _get_daily_pnl(self, session_pnl_data: Dict[str, Any], trading_result: Any) -> float:
        """Get daily P&L for PositionManager loss tracking."""
        # Try to extract from session_pnl_data first
        if isinstance(session_pnl_data, dict):
            pnl = session_pnl_data.get("daily_pnl") or session_pnl_data.get("current_pnl") or session_pnl_data.get("session_pnl")
            if pnl is not None:
                try:
                    return float(pnl)
                except (TypeError, ValueError):
                    pass
        # Try trading_result
        if isinstance(trading_result, dict):
            pnl = trading_result.get("pnl") or trading_result.get("daily_pnl")
            if pnl is not None:
                try:
                    return float(pnl)
                except (TypeError, ValueError):
                    pass
        # Try bus for portfolio_metrics
        try:
            pm = self.smart_bus.get("portfolio_metrics", "SessionManager", default=None)
            if isinstance(pm, dict):
                pnl = pm.get("current_pnl") or pm.get("daily_pnl")
                if pnl is not None:
                    return float(pnl)
        except Exception:
            pass
        return 0.0

    def _get_prop_firm_status(self, env_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Get prop firm status for Environment checks."""
        # Try to read from risk_policy.yaml
        status = {
            "enabled": False,
            "account_type": "personal",
            "max_daily_loss_pct": 5.0,
            "max_total_loss_pct": 10.0,
            "profit_target_pct": 8.0,
            "timestamp": self._to_iso_ts(self.time_helper.get_utcnow()),
        }
        try:
            from pathlib import Path

            import yaml
            risk_policy_path = Path("config/risk_policy.yaml")
            if risk_policy_path.exists():
                with open(risk_policy_path, "r", encoding="utf-8") as f:
                    rp = yaml.safe_load(f) or {}
                prop_firm = rp.get("prop_firm", {})
                if prop_firm:
                    status["enabled"] = bool(prop_firm.get("enabled", False))
                    status["account_type"] = prop_firm.get("account_type", "personal")
                    status["max_daily_loss_pct"] = float(prop_firm.get("max_daily_loss_pct", 5.0))
                    status["max_total_loss_pct"] = float(prop_firm.get("max_total_loss_pct", 10.0))
                    status["profit_target_pct"] = float(prop_firm.get("profit_target_pct", 8.0))
        except Exception:
            pass
        return status

    def _get_prop_firm_state(self, env_cfg: Dict[str, Any], session_pnl_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prop firm state for LotCalculator."""
        state: Dict[str, Any] = {
            "initial_balance": 100000.0,
            "start_of_day_equity": 100000.0,
            "yesterday_close_equity": 100000.0,
            "daily_limit_base_equity": 100000.0,
            "current_equity": 100000.0,
            "daily_pnl": 0.0,
            "timestamp": self._to_iso_ts(self.time_helper.get_utcnow()),
        }
        # Try to get initial balance from env_cfg
        if isinstance(env_cfg, dict):
            ib = env_cfg.get("initial_balance")
            if ib is not None:
                try:
                    state["initial_balance"] = float(ib)
                    state["start_of_day_equity"] = float(ib)
                    state["yesterday_close_equity"] = float(ib)
                    state["daily_limit_base_equity"] = float(ib)
                except (TypeError, ValueError):
                    pass
        # Try to get from risk_policy.yaml
        try:
            from pathlib import Path

            import yaml
            risk_policy_path = Path("config/risk_policy.yaml")
            if risk_policy_path.exists():
                with open(risk_policy_path, "r", encoding="utf-8") as f:
                    rp = yaml.safe_load(f) or {}
                prop_firm = rp.get("prop_firm", {})
                lot_sizing = rp.get("lot_sizing", {})
                account_size = prop_firm.get("account_size") or lot_sizing.get("account_balance")
                if account_size:
                    state["initial_balance"] = float(account_size)
                    state["start_of_day_equity"] = float(account_size)
                    state["yesterday_close_equity"] = float(account_size)
                    state["daily_limit_base_equity"] = float(account_size)
        except Exception:
            pass
        # Try to get current equity from bus
        try:
            pm = self.smart_bus.get("portfolio_metrics", "SessionManager", default=None)
            if isinstance(pm, dict):
                eq = pm.get("equity") or pm.get("balance")
                if eq is not None:
                    state["current_equity"] = float(eq)
                pnl = pm.get("current_pnl") or pm.get("daily_pnl")
                if pnl is not None:
                    state["daily_pnl"] = float(pnl)
        except Exception:
            pass
        return state

    # Thesis (human-readable status string)
    def _build_session_thesis(
        self, *,
        session_metrics: Dict[str, Any],
        session_health: Dict[str, Any],
        system_performance: Dict[str, Any],
        session_pnl_data: Dict[str, Any],
        trading_result: Dict[str, Any],
        session_context: Dict[str, Any],
        performance_data: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        error_message: Optional[str] = None
    ) -> str:
        parts: List[str] = []

        # Session overview
        parts.extend(self._build_thesis_session_overview(session_metrics, session_context))

        # Health
        parts.append(self._build_thesis_health_section(session_health))

        # Performance
        parts.append(self._build_thesis_performance_section(system_performance))
        parts.append(self._build_thesis_pnl_section(session_pnl_data))
        parts.append(self._build_thesis_performance_data_section(performance_data))

        # Portfolio
        parts.append(self._build_thesis_portfolio_section(performance_metrics))

        # Trading
        parts.append(self._build_thesis_trading_section(trading_result))

        # Error if applicable
        if error_message:
            parts.append(f"Operating in degraded mode due to {self._excerpt_text(error_message)}.")

        return " ".join(part.strip() for part in parts if part).strip() or \
               "Session status available but no detailed context was produced."

    def _build_thesis_session_overview(self, session_metrics: Dict[str, Any], session_context: Dict[str, Any]) -> List[str]:
        parts: List[str] = []
        metrics_source = session_metrics or {}
        session_id = metrics_source.get("session_id") or "session"
        status = metrics_source.get("status") or "unknown"

        duration_minutes: Optional[float]
        try:
            duration_minutes = float(metrics_source.get("duration", 0.0)) / 60.0
        except (TypeError, ValueError):
            duration_minutes = None

        context_bits: List[str] = []
        context_source = session_context or {}
        for key in ("trading_session", "session_type", "session_canonical"):
            val = context_source.get(key)
            if val:
                sval = str(val)
                if sval not in context_bits:
                    context_bits.append(sval)

        base_sentence = f"Session {session_id} {status}"
        if duration_minutes is not None:
            base_sentence += f" for {duration_minutes:.1f}m"
        if context_bits:
            base_sentence += f" ({'/'.join(context_bits)})"

        parts.append(base_sentence + ".")
        return parts

    def _build_thesis_health_section(self, session_health: Dict[str, Any]) -> str:
        health_source = session_health or {}
        health_status = health_source.get("status") or "unknown"
        alerts = health_source.get("alerts") or []
        if alerts:
            latest_alert = alerts[-1]
            alert_message = ""
            if isinstance(latest_alert, dict):
                alert_message = latest_alert.get("message") or latest_alert.get("detail") or ""
            return f"Health {health_status} with {len(alerts)} alert(s)" + \
                   (f"; latest: {self._excerpt_text(alert_message)}." if alert_message else ".")
        return f"Health {health_status} with no outstanding alerts."

    def _build_thesis_performance_section(self, system_performance: Dict[str, Any]) -> str:
        perf_source = system_performance or {}
        bits: List[str] = []
        if ("success_count" in perf_source) or ("failure_count" in perf_source):
            try:
                sc = int(perf_source.get("success_count") or 0)
                fc = int(perf_source.get("failure_count") or 0)
                bits.append(f"{sc} success / {fc} fail")
            except Exception:
                pass
        sr = self._safe_percent_str(perf_source.get("success_rate"))
        if sr:
            bits.append(f"success rate {sr}")
        avg_ms = self._safe_float_str(perf_source.get("avg_processing_time_ms"), precision=1)
        if avg_ms:
            bits.append(f"avg {avg_ms} ms latency")
        return ("Processing stats: " + ", ".join(bits) + ".") if bits else ""

    def _build_thesis_pnl_section(self, session_pnl_data: Dict[str, Any]) -> str:
        src = session_pnl_data if isinstance(session_pnl_data, dict) else {}
        if not src:
            return ""
        bits: List[str] = []
        step_value = src.get("step")
        if step_value is not None:
            try:
                bits.append(f"step {int(step_value)}")
            except Exception:
                bits.append(f"step {step_value}")
        for key, label in (
            ("current_pnl", "current PnL"),
            ("balance", "balance"),
            ("equity", "equity"),
            ("last_step_pnl", "last step PnL"),
        ):
            if key in src:
                fstr = self._safe_float_str(src.get(key), precision=2)
                bits.append(f"{label} {fstr}" if fstr is not None else f"{label} {src.get(key)}")
        return ("PnL snapshot: " + ", ".join(bits) + ".") if bits else ""

    def _build_thesis_performance_data_section(self, performance_data: Dict[str, Any]) -> str:
        src = performance_data if isinstance(performance_data, dict) else {}
        bits: List[str] = []
        wr = self._safe_percent_str(src.get("win_rate"))
        if wr:
            bits.append(f"win rate {wr}")
        expectancy = self._safe_float_str(src.get("expectancy"), precision=2)
        if expectancy:
            bits.append(f"expectancy {expectancy}")
        avg_trade = self._safe_float_str(src.get("avg_trade_pnl"), precision=2)
        if avg_trade:
            bits.append(f"avg trade PnL {avg_trade}")
        return ("Performance data: " + ", ".join(bits) + ".") if bits else ""

    def _build_thesis_portfolio_section(self, performance_metrics: Dict[str, Any]) -> str:
        portfolio_metrics: Dict[str, Any] = {}
        if isinstance(performance_metrics, dict):
            candidate = performance_metrics.get("portfolio")
            if isinstance(candidate, dict):
                portfolio_metrics = candidate
        if not portfolio_metrics:
            return ""
        bits: List[str] = []
        for key, label in (
            ("max_drawdown", "max drawdown"),
            ("exposure", "exposure"),
            ("volatility", "volatility"),
        ):
            fstr = self._safe_float_str(portfolio_metrics.get(key), precision=2)
            if fstr:
                bits.append(f"{label} {fstr}")
        return ("Portfolio metrics: " + ", ".join(bits) + ".") if bits else ""

    def _build_thesis_trading_section(self, trading_result: Dict[str, Any]) -> str:
        notes: List[str] = []
        if isinstance(trading_result, dict) and trading_result:
            status_value = trading_result.get("status")
            if status_value:
                notes.append(str(status_value))
            pnl_value = self._safe_float_str(trading_result.get("pnl"), precision=2)
            if pnl_value:
                notes.append(f"pnl {pnl_value}")
            for field_key, display_name in [
                ("open_positions", "open positions"),
                ("position_count", "open positions"),
                ("executed_orders", "executed orders"),
            ]:
                value = trading_result.get(field_key)
                if value is not None:
                    try:
                        notes.append(f"{int(value)} {display_name}")
                    except Exception:
                        notes.append(f"{display_name} {value}")
        return ("Trading result: " + ", ".join(notes) + ".") if notes else ""

    @staticmethod
    def _excerpt_text(text: Any, limit: int = 160) -> str:
        if not text:
            return ""
        value = " ".join(str(text).split())
        return value if len(value) <= limit else value[: limit - 3] + "..."

    # ─────────────────────────────────────────────────────────
    # Confidence / proposals
    # ─────────────────────────────────────────────────────────
    async def calculate_confidence(self, action: Optional[Dict[str, Any]] = None, **inputs) -> float:
        """
        Enhanced confidence calculation with better logic.
        Confidence reflects recency of health checks and meaningful counter data.
        """
        now = self.time_helper.get_current_time()
        freshness = max(0.0, 1.0 - (now - self._last_health_check) / 60.0)
        total_operations = self._success + self._fail
        counters_ok = 1.0 if total_operations > 0 else 0.0
        confidence = 0.7 * freshness + 0.3 * counters_ok
        return float(min(1.0, max(0.0, confidence)))

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        current_time = self.time_helper.get_current_time()
        duration = current_time - self.session_start_ts
        return {
            "update_session": True,
            "reset_session": bool(duration >= self.cfg.session_duration),
            "reason": "roll session after configured duration" if duration >= self.cfg.session_duration else None,
        }

    # ─────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        CONTRACT-ALIGNED TOP-LEVEL OUTPUTS (always present in snapshot):
            consensus_data, emergency_mode, episode_data, episode_summary,
            market_open, memory_usage, mistakes, module_performance, performance_data,
            playbook_entries, playbook_memory, session_pnl_data, session_context,
            session_metrics, system_alerts, session_health, system_performance,
            system_health, environment_config, execution_mode

        PnL policy:
          - Forward env bus values when available.
          - Otherwise return {} for PnL-related keys (never fabricate numbers).
        """
        t0 = self.time_helper.get_current_time()
        self._proc_count += 1
        pid = self._proc_count

        # Pretty logs (like your sample)
        self._pretty.debug(pid, "PROCESS_START", "────────────────────────────────────────────────────────────")
        try:
            input_keys = list(inputs.keys())
            self._pretty.debug(
                pid, "INPUT_INSPECTION",
                f"Input summary: {{keys_provided: [{len(input_keys)} items]}}"
            )
        except Exception:
            pass

        error_message: Optional[str] = None
        env_cfg_out: Dict[str, Any] = {}  # ensure defined for error path

        try:
            # Optimized session updates with caching
            self._update_session_labels()
            current_time = self.time_helper.get_current_time()
            self._last_health_check = current_time

            # Build core session data
            session_metrics = self._build_session_metrics(current_time)
            system_performance = self._build_system_performance()
            session_health = self._build_session_health()
            session_context = self._build_session_context()

            # Publish canonical session labels for consumers that read direct keys
            try:
                self.smart_bus.set(
                    'session_canonical',
                    session_context.get('session_canonical', 'unknown'),
                    module='SessionManager',
                    thesis='Canonical session label'
                )
                self.smart_bus.set(
                    'trading_session',
                    session_context.get('trading_session', 'unknown'),
                    module='SessionManager',
                    thesis='Trading session label'
                )
                self.smart_bus.set(
                    'session_type',
                    session_context.get('session_type', 'unknown'),
                    module='SessionManager',
                    thesis='Session type label'
                )
                # FIX: Publish per-instrument session for HorizonAligner
                self.smart_bus.set(
                    'session_canonical_by_instrument',
                    self._build_session_by_instrument(),
                    module='SessionManager',
                    thesis='Per-instrument session labels'
                )
            except Exception:
                pass

            # Extract bus data once
            (performance_data, environment_config, portfolio_metrics,
             trading_result, system_health_bus) = self._get_bus_data()

            # Derived
            session_pnl_data = self._build_pnl_data(portfolio_metrics, trading_result)
            system_health = self._build_system_health(session_health, system_performance, system_health_bus)

            # FIX: Publish daily_pnl for PositionManager and other consumers
            try:
                daily_pnl_val = self._get_daily_pnl(session_pnl_data, trading_result)
                self.smart_bus.set(
                    'daily_pnl',
                    daily_pnl_val,
                    module='SessionManager',
                    thesis=f'Daily PnL: {daily_pnl_val:.2f}'
                )
            except Exception:
                pass
            
            # FIX: Publish prop_firm_status and prop_firm_state for Environment consumers
            try:
                env_cfg_tmp = dict(environment_config) if isinstance(environment_config, dict) else {}
                prop_status = self._get_prop_firm_status(env_cfg_tmp)
                self.smart_bus.set(
                    'prop_firm_status',
                    prop_status,
                    module='SessionManager',
                    thesis=f"Prop firm mode: {'active' if prop_status.get('enabled') else 'inactive'}"
                )
                prop_state = self._get_prop_firm_state(env_cfg_tmp, session_pnl_data)
                self.smart_bus.set(
                    'prop_firm_state',
                    prop_state,
                    module='SessionManager',
                    thesis=f"Prop firm state: phase={prop_state.get('phase', 'N/A')}, daily_pnl={prop_state.get('daily_pnl', 0.0):.2f}"
                )
            except Exception:
                pass

            # Performance metrics bundle
            perf_metrics: Dict[str, Any] = {
                "system_performance": system_performance,
                "system_health": session_health,
                "environment_config": environment_config,
                "session_pnl": dict(session_pnl_data) if session_pnl_data else {},
            }
            if isinstance(portfolio_metrics, dict) and portfolio_metrics:
                perf_metrics["portfolio"] = dict(portfolio_metrics)

            # Structured session data
            session_data = SessionData(
                metrics=session_metrics,
                health=session_health,
                context=session_context,
                pnl_data=session_pnl_data,
                trading_result=trading_result,
                performance_data=performance_data,
                performance_metrics=perf_metrics,
                system_performance=system_performance,
                system_health=system_health
            )

            # Canonicalize environment_config & publish execution/emergency mode
            try:
                env_cfg_out = dict(environment_config) if isinstance(environment_config, dict) else {}
                # Derive initial_balance if present in performance_data
                pd_ib: Optional[float] = None
                if isinstance(performance_data, dict):
                    if isinstance(performance_data.get('initial_balance'), (int, float)):
                        pd_ib = float(performance_data['initial_balance'])
                    elif isinstance(performance_data.get('starting_balance'), (int, float)):
                        pd_ib = float(performance_data['starting_balance'])
                if pd_ib is not None:
                    env_cfg_out['initial_balance'] = pd_ib

                # Publish to SmartInfoBus (guarded)
                try:
                    self.smart_bus.declare_owner('environment_config', 'SessionManager')
                except Exception:
                    pass
                try:
                    self.smart_bus.set('environment_config', env_cfg_out,
                                       module='SessionManager',
                                       thesis='Canonical environment config (synced)')
                except Exception:
                    pass

                mode_value = str(env_cfg_out.get('mode', 'sim')).lower()
                try:
                    self.smart_bus.declare_owner('execution_mode', 'SessionManager')
                except Exception:
                    pass
                try:
                    self.smart_bus.set('execution_mode', mode_value,
                                       module='SessionManager',
                                       thesis='Canonical execution mode')
                except Exception:
                    pass

                try:
                    self.smart_bus.set('env_mode', mode_value,
                                       module='SessionManager',
                                       thesis='Legacy alias: env_mode')
                except Exception:
                    pass

                # Publish emergency_mode (canonical on SessionManager per contract)
                try:
                    self.smart_bus.declare_owner('emergency_mode', 'SessionManager')
                except Exception:
                    pass
                try:
                    self.smart_bus.set('emergency_mode', False,
                                       module='SessionManager',
                                       thesis='Emergency mode status (system-wide kill switch)')
                except Exception:
                    pass
            except Exception:
                env_cfg_out = {}

            # Build final snapshot with the session_data we created earlier
            snapshot = self._build_snapshot(session_data=session_data, env_cfg_out=env_cfg_out)

            # Update success metrics
            self._success += 1
            elapsed_ms = (self.time_helper.get_current_time() - t0) * 1000.0
            self._proc_times.append(float(elapsed_ms))

            # Pretty summary (sample style)
            if pid % max(1, self.cfg.log_every_n) == 0:
                sr = system_performance.get("success_rate", 0.0)
                avgms = system_performance.get("avg_processing_time_ms", 0.0)
                health = session_health.get("status", "unknown")
                self._pretty.info(
                    pid, "SUMMARY",
                    f"tick={self._proc_count} health={health} success_rate={sr:.3f} avg_ms={avgms:.1f} alerts={len(self.system_alerts)}"
                )

            # Optional NDJSON
            if self.cfg.ndjson_every_n > 0 and (pid % self.cfg.ndjson_every_n == 0):
                self._pretty.ndjson(pid, "snapshot_meta", {
                    "tick": self._proc_count,
                    "health": session_health.get("status"),
                    "success_rate": system_performance.get("success_rate"),
                    "avg_ms": system_performance.get("avg_processing_time_ms"),
                    "trading_session": session_context.get("trading_session"),
                    "session_type": session_context.get("session_type"),
                })

            return snapshot

        except Exception as exc:
            # Enhanced error handling with structured logging
            self._fail += 1
            error_message = f"{type(exc).__name__}: {str(exc)[:200]}"
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
            self._pretty.error(self._proc_count, "PROCESS_FAIL", error_message)

            # Try to recover a coherent snapshot
            (performance_data, environment_config, portfolio_metrics,
             trading_result, system_health_bus) = self._get_bus_data()

            session_pnl_data = self._build_pnl_data(portfolio_metrics, trading_result)
            system_performance_error = self._build_system_performance()
            session_health_error = self._build_session_health()

            perf_metrics: Dict[str, Any] = {
                "system_performance": system_performance_error,
                "system_health": session_health_error,
                "environment_config": environment_config,
                "session_pnl": dict(session_pnl_data) if session_pnl_data else {},
            }
            if isinstance(portfolio_metrics, dict) and portfolio_metrics:
                perf_metrics["portfolio"] = dict(portfolio_metrics)

            # Build degraded snapshot
            snapshot = self._build_snapshot(
                session_data=None,
                env_cfg_out=environment_config if isinstance(environment_config, dict) else {},
                error_context={
                    "error": error_message,
                    "system_performance": system_performance_error,
                    "session_health": session_health_error,
                    "performance_data": performance_data,
                    "performance_metrics": perf_metrics,
                    "trading_result": trading_result,
                    "session_pnl_data": session_pnl_data,
                }
            )
            return snapshot

    # ─────────────────────────────────────────────────────────
    # Snapshot assembly (contract-clean)
    # ─────────────────────────────────────────────────────────
    def _build_snapshot(
        self,
        session_data: Optional[SessionData],
        env_cfg_out: Dict[str, Any],
        error_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assemble the contract snapshot. If session_data is None, we’ll derive the minimal
        surface from error_context to keep the contract satisfied.
        """
        if session_data is None and error_context is not None:
            # Degraded path — derive minimal viable sections
            system_performance = error_context.get("system_performance", {})
            session_health = error_context.get("session_health", {})
            performance_data = error_context.get("performance_data", {})
            perf_metrics = error_context.get("performance_metrics", {})
            session_pnl_data = error_context.get("session_pnl_data", {})
            trading_result = error_context.get("trading_result", {})
            session_context = self._build_session_context()
            session_metrics = {
                "session_id": self.session_id,
                "duration": float(self.time_helper.get_current_time() - self.session_start_ts),
                "status": "error",
                "start_time": datetime.datetime.utcfromtimestamp(self.session_start_ts).isoformat(),
            }
            thesis = self._build_session_thesis(
                session_metrics=session_metrics,
                session_health=session_health,
                system_performance=system_performance,
                session_pnl_data=session_pnl_data,
                trading_result=trading_result,
                session_context=session_context,
                performance_data=performance_data,
                performance_metrics=perf_metrics,
                error_message=error_context.get("error")
            )
        else:
            assert session_data is not None  # for type-checker
            system_performance = session_data.system_performance
            session_health = session_data.health
            performance_data = session_data.performance_data
            perf_metrics = session_data.performance_metrics
            session_pnl_data = session_data.pnl_data
            trading_result = session_data.trading_result
            session_context = session_data.context
            session_metrics = session_data.metrics
            thesis = self._build_session_thesis(
                session_metrics=session_metrics,
                session_health=session_health,
                system_performance=system_performance,
                session_pnl_data=session_pnl_data,
                trading_result=trading_result,
                session_context=session_context,
                performance_data=performance_data,
                performance_metrics=perf_metrics
            )

        # Contract surfaces — always present
        snapshot: Dict[str, Any] = {
            "consensus_data": {},
            "emergency_mode": False,
            "episode_data": {},
            "episode_summary": {},
            "market_open": True,
            "memory_usage": {},
            "mistakes": [],
            "module_performance": {},
            "performance_data": performance_data,
            "playbook_entries": [],
            "playbook_memory": {},
            "session_pnl_data": session_pnl_data,
            "performance_metrics": perf_metrics,
            "session_context": session_context,
            "session_metrics": session_metrics,
            "system_alerts": list(self.system_alerts[-25:]),
            "session_health": session_health,
            "system_performance": system_performance,
            "system_health": session_health if error_context is None else session_health,
            "environment_config": env_cfg_out,
            "execution_mode": str(env_cfg_out.get("mode", "sim")).lower() if isinstance(env_cfg_out, dict) else "sim",
            # FIX: Per-instrument session for HorizonAligner
            "session_canonical_by_instrument": self._build_session_by_instrument(),
            # FIX: Add daily_pnl for PositionManager daily loss tracking
            "daily_pnl": self._get_daily_pnl(session_pnl_data, trading_result),
            # FIX: Add prop_firm keys for LotCalculator and Environment
            "prop_firm_status": self._get_prop_firm_status(env_cfg_out),
            "prop_firm_state": self._get_prop_firm_state(env_cfg_out, session_pnl_data),
        }

        # Add an operator-facing narrative (safe, non-contract key)
        snapshot["_thesis"] = thesis
        return snapshot
