"""
Enhanced Drawdown Rescue with SmartInfoBus Integration
Intelligent drawdown monitoring and rescue mechanisms
(Contract-tight, production-ready)
"""

from __future__ import annotations

from modules.contracts import module_args
import numpy as np
import datetime
import time
import threading
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict

from modules.core.module_base import BaseModule, module
from modules.core.mixins import (
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
    SmartInfoBusTradingMixin,
)
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


# ─────────────────────────────────────────────────────────────
# Typed, lint-safe config + namespaced health/status keys
# ─────────────────────────────────────────────────────────────
def _load_drawdown_rescue_config_from_yaml() -> Dict[str, Any]:
    """Load drawdown rescue config values from risk_policy.yaml."""
    import yaml
    import os

    defaults: Dict[str, Any] = {}
    try:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "config",
            "risk_policy.yaml",
        )
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}

            limits = policy.get("limits", {})
            escalation = policy.get("escalation", {})
            modules_cfg = policy.get("modules", {}).get("DrawdownRescue", {})

            # Map escalation thresholds to rescue thresholds
            defaults["dd_limit"] = float(limits.get("max_drawdown", 0.085))
            defaults["warning_dd"] = float(escalation.get("warning_threshold", 0.025))
            defaults["info_dd"] = float(escalation.get("alert_threshold", 0.015))
            defaults["recovery_threshold"] = float(modules_cfg.get("recovery_target", 0.05))
    except Exception:
        # YAML load failure should not break the module; fall back to defaults
        pass
    return defaults


@dataclass
class DrawdownRescueConfig:
    """Configuration loaded from risk_policy.yaml"""

    # Thresholds (fractions 0..1) - from risk_policy.yaml
    dd_limit: float = 0.085         # From limits.max_drawdown
    warning_dd: float = 0.025       # From escalation.warning_threshold
    info_dd: float = 0.015          # From escalation.alert_threshold
    recovery_threshold: float = 0.05  # From modules.DrawdownRescue.recovery_target

    # Windows & dynamics
    velocity_window: int = 10       # steps for velocity history
    lookback_balance: int = 120     # stored balance/equity points

    # Velocity/accel thresholds (per step)
    rapid_velocity: float = 0.02    # +2% dd per step → rapid deterioration
    moderate_velocity: float = 0.01 # +1% dd per step → moderate deterioration
    recovery_velocity: float = -0.01 # -1% dd per step → recovery
    accel_threshold: float = 0.01   # +1% change in velocity → accelerating decline

    # Rescue controls
    enabled: bool = True
    rescue_mode_default: bool = True
    adaptive_thresholds: bool = True

    # Risk adjustment
    risk_smoothing: float = 0.30    # EMA smoothing for risk_adjustment_factor
    rescue_decay_per_min: float = 0.001  # risk reduction 0.1% per minute in rescue

    # Monitoring / health
    health_check_interval: int = 30
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown_sec: float = 20.0
    max_processing_time_ms: float = 60.0
    status_key: str = "drawdown_rescue_status"
    health_key: str = "drawdown_rescue_health"

    def __post_init__(self) -> None:
        """Load values from risk_policy.yaml after init."""
        yaml_config = _load_drawdown_rescue_config_from_yaml()
        for key, value in yaml_config.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ─────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────
@module(
    **module_args(
        "DrawdownRescue",
        description="Enhanced drawdown monitoring with intelligent rescue mechanisms and risk adjustment",
        error_handling=True,
        hot_reload=True,
        timeout_ms=3000,
    )
)
class DrawdownRescue(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin):
    """
    Contract guarantees:
    - Writes ONLY its 'provides' keys to SmartInfoBus. Health/status use namespaced keys.
    - Returns ALL 'provides' keys plus '_thesis' and 'success' on success, fallback, or error.
    - Numeric types are plain Python; timestamps are ISO-8601.
    - Circuit breaker with safe fallback; background health monitor.
    """

    # ── init & systems ───────────────────────────────────────
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        # Proper layering: defaults -> YAML -> explicit config
        base_cfg = DrawdownRescueConfig()  # includes YAML overrides via __post_init__
        if isinstance(config, dict):
            for k, v in config.items():
                if hasattr(base_cfg, k):
                    setattr(base_cfg, k, v)
        self._cfg = base_cfg
        self.config = config or {}

        # Initialize low-level systems BEFORE BaseModule may call _initialize()
        self._initialize_advanced_systems()

        # Debug flag for conditional logging in methods
        self.debug: bool = bool(getattr(self, "debug", False))

        # Circuit breaker & health state
        self.circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",
            "threshold": int(self._cfg.circuit_breaker_threshold),
            "cooldown_sec": float(self._cfg.circuit_breaker_cooldown_sec),
        }
        self._processing_times: deque[float] = deque(maxlen=100)  # seconds per cycle
        self._health_status: str = "healthy"
        self._monitoring_active: bool = False

        # Configuration mirror (for fast access)
        self.dd_limit: float = float(self._cfg.dd_limit)
        self.warning_dd: float = float(self._cfg.warning_dd)
        self.info_dd: float = float(self._cfg.info_dd)
        self.recovery_threshold: float = float(self._cfg.recovery_threshold)
        self.velocity_window: int = int(self._cfg.velocity_window)
        self.enabled: bool = bool(self._cfg.enabled)
        self.rescue_mode_enabled: bool = bool(self._cfg.rescue_mode_default)
        self.adaptive_thresholds: bool = bool(self._cfg.adaptive_thresholds)

        # Core state
        self.current_dd: float = 0.0
        self.max_dd: float = 0.0
        self.peak_balance: float = 0.0
        self.dd_velocity: float = 0.0
        self.dd_acceleration: float = 0.0
        self.severity_level: str = "normal"

        # Rescue system state
        self.rescue_mode: bool = False
        self.rescue_start_time: Optional[datetime.datetime] = None
        self.risk_adjustment_factor: float = 1.0
        self.rescue_intervention_count: int = 0

        # Analytics / history
        self.dd_history: deque[float] = deque(maxlen=self.velocity_window)
        self.balance_history: deque[Dict[str, Any]] = deque(maxlen=int(self._cfg.lookback_balance))
        self.recovery_events: List[Dict[str, Any]] = []
        self.regime_drawdowns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Performance counters
        self.step_count: int = 0
        self.successful_recoveries: int = 0
        self.false_alarms: int = 0
        self.emergency_interventions: int = 0

        # Dynamic thresholds (mutable)
        self.current_thresholds: Dict[str, float] = {
            "dd_limit": self.dd_limit,
            "warning_dd": self.warning_dd,
            "info_dd": self.info_dd,
        }

        # Context-aware multipliers
        self.regime_multipliers: Dict[str, Dict[str, float]] = {
            "volatile": {"warning": 1.2, "critical": 1.15},
            "trending": {"warning": 0.9, "critical": 0.95},
            "ranging": {"warning": 1.0, "critical": 1.0},
            "crisis": {"warning": 0.85, "critical": 0.90},
        }

        super().__init__()  # may call _initialize()
        self._start_monitoring()

        self.logger.info(
            format_operator_message(
                message="Enhanced Drawdown Rescue initialized",
                icon="🛟",
                critical_limit=f"{self.dd_limit:.1%}",
                warning_threshold=f"{self.warning_dd:.1%}",
                rescue_mode=self.rescue_mode_enabled,
                adaptive_thresholds=self.adaptive_thresholds,
                enabled=self.enabled,
            )
        )

    def _initialize_advanced_systems(self) -> None:
        """Initialize advanced monitoring and error handling systems"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="DrawdownRescue",
            log_path="logs/risk/drawdown_rescue.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("DrawdownRescue", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

    # ── BaseModule hook ──────────────────────────────────────
    def _initialize(self) -> None:
        """Initialize the drawdown rescue module (required by BaseModule)"""
        try:
            # namespaced status (NOT part of provides)
            status = {
                "enabled": bool(self.enabled),
                "severity_level": str(self.severity_level),
                "current_dd": float(self.current_dd),
                "ts": datetime.datetime.now().isoformat(),
            }
            self.smart_bus.set(
                self._cfg.status_key,
                status,
                module="DrawdownRescue",
                thesis="Initial drawdown rescue status",
            )

            # reset mutable state
            self.current_dd = 0.0
            self.max_dd = 0.0
            self.peak_balance = 0.0
            self.dd_velocity = 0.0
            self.dd_acceleration = 0.0
            self.severity_level = "normal"

            self.rescue_mode = False
            self.rescue_start_time = None
            self.risk_adjustment_factor = 1.0
            self.rescue_intervention_count = 0

            self.dd_history.clear()
            self.balance_history.clear()
            self.recovery_events.clear()
            self.regime_drawdowns.clear()

            self.step_count = 0
            self.successful_recoveries = 0
            self.false_alarms = 0
            self.emergency_interventions = 0

            self.logger.info("Drawdown Rescue module initialization completed successfully")
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "rescue_initialization")
            self.logger.error(f"Rescue initialization failed: {error_context}")

    # ── background monitor ───────────────────────────────────
    def _start_monitoring(self) -> None:
        if getattr(self, "_monitoring_active", False):
            return

        def loop() -> None:
            self._monitoring_active = True
            self.logger.info("[MONITOR] DrawdownRescue health monitor started.")
            while self._monitoring_active:
                try:
                    self._update_health()

                    # publish namespaced health snapshot
                    health = self.get_health_status()
                    self.smart_bus.set(
                        self._cfg.health_key,
                        health,
                        module="DrawdownRescue",
                        thesis="Drawdown rescue health heartbeat",
                    )

                    # circuit breaker cooldown auto-reset
                    if self.circuit_breaker["state"] == "OPEN":
                        if (
                            time.time() - self.circuit_breaker["last_failure"]
                            >= self.circuit_breaker["cooldown_sec"]
                        ):
                            self.circuit_breaker["state"] = "CLOSED"
                            self.circuit_breaker["failures"] = 0
                            self.logger.info("[MONITOR] Circuit breaker auto-reset to CLOSED.")
                except Exception as e:
                    self.logger.error(f"Drawdown monitoring error: {e}")
                time.sleep(max(1, int(self._cfg.health_check_interval)))

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def stop_monitoring(self) -> None:
        self._monitoring_active = False

    def _update_health(self) -> None:
        try:
            self._health_status = "healthy"
            if len(self._processing_times) >= 10:
                avg_ms = float(np.mean(list(self._processing_times)[-10:]) * 1000.0)
                if avg_ms > float(self._cfg.max_processing_time_ms):
                    self._health_status = "warning"
            if self.circuit_breaker["state"] == "OPEN":
                self._health_status = "warning"
        except Exception as e:
            self.logger.error(f"Drawdown health update failed: {e}")
            self._health_status = "warning"

    def get_health_status(self) -> Dict[str, Any]:
        avg_ms = float(np.mean(self._processing_times) * 1000.0) if self._processing_times else 0.0
        return {
            "status": self._health_status,
            "avg_processing_time_ms": avg_ms,
            "circuit_breaker_state": self.circuit_breaker["state"],
            "ts": datetime.datetime.now().isoformat(),
        }

    # ── confidence & actions (optional API) ──────────────────
    async def calculate_confidence(self, action: Dict[str, Any], **kwargs: Any) -> float:
        """Calculate confidence score for drawdown rescue assessment"""
        try:
            confidence = 0.9

            severity_penalties = {
                "normal": 1.0,
                "info": 0.9,
                "warning": 0.8,
                "critical": 0.6,
                "error": 0.3,
                "disabled": 0.1,
            }
            confidence *= float(severity_penalties.get(self.severity_level, 0.6))

            # Rescue mode → more conservative (lower confidence)
            if self.rescue_mode:
                confidence *= 0.8

            # Velocity penalty/boost
            v = float(self.dd_velocity)
            if v > self._cfg.rapid_velocity:
                confidence *= max(0.5, 1.0 - (v - self._cfg.rapid_velocity) * 10.0)
            elif v < self._cfg.recovery_velocity:
                confidence *= 1.0  # recovering does not penalize

            # Data sufficiency
            data_factor = min(1.0, len(self.dd_history) / max(5.0, float(self.velocity_window)))
            confidence *= max(0.5, data_factor)

            # Track record
            if self.rescue_intervention_count > 0:
                success_rate = self.successful_recoveries / max(1, self.rescue_intervention_count)
                confidence *= (0.5 + 0.5 * success_rate)

            return float(np.clip(confidence, 0.0, 1.0))
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    async def propose_action(self, **kwargs: Any) -> Dict[str, Any]:
        """Propose drawdown rescue actions based on current state"""
        try:
            proposal: Dict[str, Any] = {
                "action_type": "drawdown_rescue",
                "timestamp": time.time(),
                "current_drawdown": float(self.current_dd),
                "severity_level": str(self.severity_level),
                "rescue_mode": bool(self.rescue_mode),
                "risk_adjustment_factor": float(self.risk_adjustment_factor),
                "recommendations": [],
                "warnings": [],
                "adjustments": {},
            }

            # Severity-based actions
            if self.severity_level == "critical":
                proposal["recommendations"].append(
                    {
                        "type": "emergency_intervention",
                        "reason": f"Critical drawdown {self.current_dd:.1%}",
                        "suggested_action": "Immediately reduce positions and halt new entries",
                        "priority": "critical",
                    }
                )
                proposal["adjustments"]["position_reduction"] = 0.5
                proposal["adjustments"]["new_entry_halt"] = True
            elif self.severity_level == "warning":
                proposal["recommendations"].append(
                    {
                        "type": "risk_reduction",
                        "reason": f"Warning drawdown {self.current_dd:.1%}",
                        "suggested_action": "Reduce sizes and tighten stops",
                        "priority": "high",
                    }
                )
                proposal["adjustments"]["position_reduction"] = 0.3
                proposal["adjustments"]["tighter_stops"] = True
            elif self.severity_level == "info":
                proposal["recommendations"].append(
                    {
                        "type": "caution",
                        "reason": f"Elevated drawdown {self.current_dd:.1%}",
                        "suggested_action": "Monitor closely and prepare controls",
                        "priority": "medium",
                    }
                )

            # Velocity-based signals
            if self.dd_velocity > self._cfg.rapid_velocity:
                proposal["warnings"].append(
                    {
                        "type": "rapid_deterioration",
                        "velocity": float(self.dd_velocity),
                        "threshold": float(self._cfg.rapid_velocity),
                    }
                )
                proposal["recommendations"].append(
                    {
                        "type": "velocity_control",
                        "reason": f"Rapid dd increase {self.dd_velocity:.2%}/step",
                        "suggested_action": "Immediate size reduction",
                        "priority": "high",
                    }
                )

            # Rescue mode context
            if self.rescue_mode:
                proposal["recommendations"].append(
                    {
                        "type": "rescue_active",
                        "reason": "Rescue mode is active",
                        "suggested_action": "Maintain conservative risk until recovery",
                        "priority": "ongoing",
                    }
                )
                if self.rescue_start_time:
                    dur_h = (datetime.datetime.now() - self.rescue_start_time).total_seconds() / 3600.0
                    if dur_h > 24:
                        proposal["warnings"].append(
                            {"type": "prolonged_rescue", "duration_hours": float(dur_h)}
                        )

            if self.risk_adjustment_factor < 0.8:
                proposal["adjustments"]["conservative_mode"] = True
                proposal["adjustments"]["risk_factor"] = float(self.risk_adjustment_factor)

            return proposal
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "action_proposal")
            self.logger.error(f"Action proposal failed: {error_context}")
            return {
                "action_type": "drawdown_rescue",
                "timestamp": time.time(),
                "error": str(e),
                "recommendations": [],
                "warnings": [],
                "adjustments": {},
            }

    # ── canonical account snapshot ───────────────────────────
    def _read_account_snapshot(self) -> Dict[str, float]:
        """
        Canonical read for balance/equity/PNL from SmartInfoBus.
        Falls back safely if fields are missing.
        """
        pm = self.smart_bus.get("portfolio_metrics", "DrawdownRescue") or {}
        balance = pm.get("balance")
        equity = pm.get("equity")
        pnl = pm.get("current_pnl")

        balance = float(balance) if balance is not None else 0.0
        equity = float(equity) if equity is not None else balance
        pnl = float(pnl) if pnl is not None else 0.0
        return {"balance": balance, "equity": equity, "current_pnl": pnl}

    # ── contract-safe process ────────────────────────────────
    async def process(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Enhanced drawdown monitoring with comprehensive rescue mechanisms
        Returns all provides + _thesis + success
        """
        start_time = time.time()
        try:
            if not self.enabled:
                payload = self._handle_disabled_fallback()
                self._write_bus_from_payload(payload, payload["_thesis"])
                return payload

            if self.circuit_breaker["state"] == "OPEN":
                thesis = "Circuit breaker OPEN; drawdown rescue safe fallback emitted."
                payload = self._fallback_payload(thesis)
                self._write_bus_from_payload(payload, thesis)
                return payload

            self.step_count += 1

            # Canonical reads
            acct = self._read_account_snapshot()
            balance, equity = acct["balance"], acct["equity"]
            # keep 'positions' access to satisfy contract 'requires' (even if unused)
            _positions = self.smart_bus.get("positions", "DrawdownRescue") or []
            market_context = self.smart_bus.get("market_context", "DrawdownRescue") or {}

            # Update balance tracking and peak
            self._update_balance_tracking(balance, equity)

            # Analyze drawdown and compute metrics
            dd_results = await self._analyze_drawdown_comprehensive(balance, equity, market_context)

            # Update rescue system
            rescue_status = self._update_rescue_system(dd_results, market_context)

            # Compute risk adjustment
            risk_adj = self._calculate_risk_adjustment(dd_results, rescue_status)

            # Thesis
            thesis = await self._generate_drawdown_thesis(dd_results, rescue_status, market_context, risk_adj)

            # Format payload (strict)
            payload = self._format_provides_output(dd_results, rescue_status, risk_adj, thesis)

            # Write provides to SmartInfoBus
            self._write_bus_from_payload(payload, thesis)

            # Success metrics
            processing_time_sec = float(time.time() - start_time)
            self._record_success(processing_time_sec)
            try:
                self.performance_tracker.record_metric(
                    "DrawdownRescue", "analysis_cycle", processing_time_sec * 1000.0, True
                )
            except Exception:
                pass

            return payload

        except Exception as e:
            processing_time_sec = float(time.time() - start_time)
            payload = self._handle_error(e, processing_time_sec)
            try:
                self._write_bus_from_payload(payload, payload.get("_thesis", "Drawdown error"))
            except Exception:
                pass
            return payload

    # ── SmartInfoBus I/O (single-writer) ─────────────────────
    def _write_bus_from_payload(self, payload: Dict[str, Any], thesis: str) -> None:
        try:
            self.smart_bus.set(
                "drawdown_risk",
                payload["drawdown_risk"],
                module="DrawdownRescue",
                thesis=thesis,
            )
            self.smart_bus.set(
                "rescue_status",
                payload["rescue_status"],
                module="DrawdownRescue",
                thesis="Rescue status update",
            )
            self.smart_bus.set(
                "risk_adjustment",
                payload["risk_adjustment"],
                module="DrawdownRescue",
                thesis=f"Risk adjustment factor: {payload['risk_adjustment']:.1%}",
            )
        except Exception as e:
            err = self.error_pinpointer.analyze_error(e, "bus_write")
            self.logger.error(f"SmartInfoBus update failed: {err}")

    # ── payload formatter (contract enforcer) ────────────────
    def _format_provides_output(
        self,
        dd_results: Dict[str, Any],
        rescue_status: Dict[str, Any],
        risk_adj: Dict[str, Any],
        thesis: str,
    ) -> Dict[str, Any]:
        """
        Format the provides output for SmartInfoBus.
        
        v3.1.0: Added dd_velocity, dd_acceleration, risk_adjustment_factor
        for integration with DynamicRiskController.
        """
        drawdown_payload = {
            "current_drawdown": float(self.current_dd),
            "max_drawdown": float(self.max_dd),
            "severity_level": str(self.severity_level),
            # v3.1.0: Include velocity/acceleration for DynamicRiskController integration
            "dd_velocity": float(self.dd_velocity),
            "dd_acceleration": float(self.dd_acceleration),
            "risk_adjustment_factor": float(self.risk_adjustment_factor),
            "rescue_mode": bool(self.rescue_mode),
            # Detailed analysis
            "drawdown_analysis": dd_results,
            "rescue_status": rescue_status,
            "risk_adjustment": risk_adj,
            "thesis": thesis,
        }
        rescue_payload = {
            "rescue_mode": bool(self.rescue_mode),
            "rescue_duration": float(rescue_status.get("rescue_duration_minutes", 0.0)),
            "intervention_count": int(self.rescue_intervention_count),
            "emergency_active": bool(rescue_status.get("emergency_intervention", False)),
            # v3.1.0: Include risk_adjustment_factor for DynamicRiskController
            "risk_adjustment_factor": float(self.risk_adjustment_factor),
        }
        return {
            "drawdown_risk": drawdown_payload,
            "rescue_status": rescue_payload,
            "risk_adjustment": float(self.risk_adjustment_factor),
            "_thesis": thesis,
            "success": True,
        }

    # ── histories & analytics ────────────────────────────────
    def _update_balance_tracking(self, balance: float, equity: float) -> None:
        """Update balance and equity tracking, update peak balance."""
        try:
            effective_balance = equity if equity > 0 else balance
            self.balance_history.append(
                {
                    "balance": float(balance),
                    "equity": float(equity),
                    "effective_balance": float(effective_balance),
                    "timestamp": datetime.datetime.now(),
                }
            )
            if effective_balance > self.peak_balance:
                prev_peak = self.peak_balance
                self.peak_balance = float(effective_balance)
                if self.step_count > 1 and prev_peak > 0:
                    self.logger.info(
                        format_operator_message(
                            message="New peak balance achieved",
                            icon="[CHART]",
                            peak_balance=f"€{self.peak_balance:,.2f}",
                            previous_max_dd=f"{self.max_dd:.1%}",
                        )
                    )
                    # On new peak, reset max drawdown baseline
                    self.max_dd = 0.0
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "balance_tracking")
            self.logger.warning(f"Balance tracking failed: {error_context}")

    async def _analyze_drawdown_comprehensive(
        self,
        balance: float,
        equity: float,
        market_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Comprehensive drawdown analysis with advanced metrics"""
        start_time = datetime.datetime.now()
        try:
            effective_balance = equity if equity > 0 else balance

            # Calculate current drawdown
            if self.peak_balance > 0:
                self.current_dd = float(
                    np.clip((self.peak_balance - effective_balance) / self.peak_balance, 0.0, 1.0)
                )
            else:
                self.current_dd = 0.0

            # Update maximum drawdown
            self.max_dd = max(float(self.max_dd), float(self.current_dd))

            # Velocity and acceleration
            velocity_metrics = self._calculate_velocity_metrics()

            # Context thresholds
            context_thresholds = self._calculate_context_adjusted_thresholds(market_context)

            # Trading mode integration (mode-level drawdown limits)
            try:
                mode_config = self.smart_bus.get("mode_config", "DrawdownRescue") or {}
                trading_mode = self.smart_bus.get("trading_mode", "DrawdownRescue") or "normal"
                mode_drawdown_limit = float(mode_config.get("drawdown_limit", self.dd_limit))

                # Mode's limit becomes the effective critical threshold
                context_thresholds["dd_limit"] = mode_drawdown_limit
                context_thresholds["warning_dd"] = mode_drawdown_limit * 0.8
                context_thresholds["info_dd"] = mode_drawdown_limit * 0.6

                # Keep our internal mirror aligned
                self.current_thresholds = dict(context_thresholds)

                if self.debug and self.current_dd > mode_drawdown_limit * 0.5:
                    self.logger.info(
                        format_operator_message(
                            icon="🎛️",
                            message="Trading mode drawdown limit integrated",
                            mode=trading_mode,
                            mode_limit=f"{mode_drawdown_limit:.1%}",
                            current_dd=f"{self.current_dd:.1%}",
                            proximity=f"{(self.current_dd / mode_drawdown_limit):.1%}",
                        )
                    )
            except Exception as e:
                if self.debug:
                    self.logger.warning(f"Trading mode integration failed in drawdown rescue: {e}")

            # Severity
            severity_assessment = self._assess_drawdown_severity(velocity_metrics, context_thresholds)

            # Recovery
            recovery_metrics = self._calculate_recovery_metrics()

            # Regime patterns
            regime_analysis = self._analyze_regime_patterns(market_context)

            processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000.0

            return {
                "current_drawdown": float(self.current_dd),
                "max_drawdown": float(self.max_dd),
                "peak_balance": float(self.peak_balance),
                "effective_balance": float(effective_balance),
                "velocity_metrics": velocity_metrics,
                "context_thresholds": context_thresholds,
                "severity_assessment": severity_assessment,
                "recovery_metrics": recovery_metrics,
                "regime_analysis": regime_analysis,
                "processing_time_ms": float(processing_time),
                "market_context": dict(market_context),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "drawdown_analysis")
            self.logger.error(f"Drawdown analysis failed: {error_context}")
            return self._generate_analysis_error_response(str(error_context))

    def _calculate_velocity_metrics(self) -> Dict[str, Any]:
        """Calculate drawdown velocity and acceleration metrics"""
        try:
            self.dd_history.append(float(self.current_dd))

            # velocity
            if len(self.dd_history) >= 2:
                self.dd_velocity = float(self.dd_history[-1] - self.dd_history[-2])
            else:
                self.dd_velocity = 0.0

            # acceleration
            if len(self.dd_history) >= 3:
                prev_v = float(self.dd_history[-2] - self.dd_history[-3])
                self.dd_acceleration = float(self.dd_velocity - prev_v)
            else:
                self.dd_acceleration = 0.0

            # trend & momentum
            if self.dd_velocity > 0:
                trend_direction = "deteriorating"
            elif self.dd_velocity < 0:
                trend_direction = "improving"
            else:
                trend_direction = "stable"

            momentum = 0.0
            if len(self.dd_history) >= 5:
                recent = list(self.dd_history)[-5:]
                try:
                    momentum = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
                except Exception:
                    momentum = 0.0

            volatility = float(np.std(list(self.dd_history))) if len(self.dd_history) >= 3 else 0.0

            return {
                "velocity": float(self.dd_velocity),
                "acceleration": float(self.dd_acceleration),
                "trend_direction": trend_direction,
                "momentum": float(momentum),
                "volatility": float(volatility),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "velocity_calculation")
            self.logger.warning(f"Velocity calculation failed: {error_context}")
            return {
                "velocity": 0.0,
                "acceleration": 0.0,
                "trend_direction": "unknown",
                "momentum": 0.0,
                "volatility": 0.0,
            }

    def _calculate_context_adjusted_thresholds(self, market_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate context-adjusted drawdown thresholds"""
        try:
            if not self.adaptive_thresholds:
                return dict(self.current_thresholds)

            regime = str(market_context.get("regime", "unknown"))
            volatility_level = str(market_context.get("volatility_level", "medium"))

            adjusted: Dict[str, float] = {
                "info_dd": float(self.info_dd),
                "warning_dd": float(self.warning_dd),
                "dd_limit": float(self.dd_limit),
            }

            # Regime multipliers
            if regime in self.regime_multipliers:
                m = self.regime_multipliers[regime]
                adjusted["warning_dd"] *= float(m["warning"])
                adjusted["dd_limit"] *= float(m["critical"])

            # Volatility multipliers
            vol_multipliers = {"low": 0.9, "medium": 1.0, "high": 1.15, "extreme": 1.3}
            vm = float(vol_multipliers.get(volatility_level, 1.0))
            for k in adjusted:
                adjusted[k] *= vm

            # Bounds (heuristic, not hard prop limits)
            adjusted["info_dd"] = float(np.clip(adjusted["info_dd"], 0.03, 0.15))
            adjusted["warning_dd"] = float(np.clip(adjusted["warning_dd"], 0.08, 0.30))
            adjusted["dd_limit"] = float(np.clip(adjusted["dd_limit"], 0.15, 0.60))

            self.current_thresholds = dict(adjusted)
            return adjusted
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "threshold_adjustment")
            self.logger.warning(f"Threshold adjustment failed: {error_context}")
            return dict(self.current_thresholds)

    def _assess_drawdown_severity(
        self,
        velocity_metrics: Dict[str, float],
        thresholds: Dict[str, float],
    ) -> Dict[str, Any]:
        """Assess drawdown severity with velocity consideration"""
        try:
            # Base severity
            if self.current_dd >= thresholds["dd_limit"]:
                level = "critical"
            elif self.current_dd >= thresholds["warning_dd"]:
                level = "warning"
            elif self.current_dd >= thresholds["info_dd"]:
                level = "info"
            else:
                level = "normal"

            # Velocity & acceleration
            v = float(velocity_metrics["velocity"])
            a = float(velocity_metrics["acceleration"])
            factors: List[str] = []

            if v > self._cfg.rapid_velocity:
                if level == "info":
                    level = "warning"
                elif level == "warning":
                    level = "critical"
                factors.append("rapid_deterioration")

            if a > self._cfg.accel_threshold and level in ("normal", "info"):
                level = "warning"
                factors.append("accelerating_decline")

            if v < self._cfg.recovery_velocity and level == "warning":
                if self.current_dd < thresholds["warning_dd"] * 0.9:
                    level = "info"
                    factors.append("recovering")

            self.severity_level = level
            return {
                "level": level,
                "severity_factors": factors,
                "velocity_impact": v > (self._cfg.rapid_velocity * 0.75),
                "acceleration_impact": a > (self._cfg.accel_threshold * 0.8),
                "threshold_used": float(thresholds.get(f"{level}_dd", thresholds["info_dd"])),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "severity_assessment")
            self.logger.warning(f"Severity assessment failed: {error_context}")
            return {"level": "unknown", "severity_factors": []}

    def _calculate_recovery_metrics(self) -> Dict[str, Any]:
        """Calculate recovery progress and metrics"""
        try:
            if self.max_dd > 0:
                progress = float(np.clip((self.max_dd - self.current_dd) / self.max_dd, 0.0, 1.0))
            else:
                progress = 0.0

            if len(self.dd_history) >= 3:
                recent_recovery = float(self.dd_history[-3] - self.current_dd)
                recovery_velocity = float(recent_recovery / 3.0)
            else:
                recovery_velocity = 0.0

            # Stability heuristic
            stability = 0.0
            if len(self.dd_history) >= 5:
                recent = list(self.dd_history)[-5:]
                stability = 1.0 if all(dd <= recent[0] for dd in recent) else 0.5

            milestone_achieved = False
            if progress >= self.recovery_threshold and self.max_dd > self.current_thresholds["info_dd"]:
                milestone_achieved = True
                self.recovery_events.append(
                    {
                        "timestamp": datetime.datetime.now(),
                        "max_dd": float(self.max_dd),
                        "recovery_progress": float(progress),
                        "step_count": int(self.step_count),
                    }
                )
                self.successful_recoveries += 1
                self.logger.info(
                    format_operator_message(
                        message="Recovery milestone achieved",
                        icon="[TARGET]",
                        progress=f"{progress:.1%}",
                        max_dd=f"{self.max_dd:.1%}",
                        current_dd=f"{self.current_dd:.1%}",
                    )
                )

            return {
                "recovery_progress": float(progress),
                "recovery_velocity": float(recovery_velocity),
                "recovery_stability": float(stability),
                "milestone_achieved": bool(milestone_achieved),
                "total_recoveries": int(self.successful_recoveries),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recovery_metrics")
            self.logger.warning(f"Recovery metrics calculation failed: {error_context}")
            return {
                "recovery_progress": 0.0,
                "recovery_velocity": 0.0,
                "recovery_stability": 0.0,
                "milestone_achieved": False,
            }

    def _analyze_regime_patterns(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime-specific drawdown patterns"""
        try:
            regime = str(market_context.get("regime", "unknown"))
            if regime != "unknown":
                self.regime_drawdowns[regime].append(
                    {
                        "drawdown": float(self.current_dd),
                        "timestamp": datetime.datetime.now(),
                        "step_count": int(self.step_count),
                    }
                )

            stats: Dict[str, Dict[str, float]] = {}
            for name, entries in self.regime_drawdowns.items():
                if entries:
                    values = [float(e["drawdown"]) for e in entries]
                    stats[name] = {
                        "avg_drawdown": float(np.mean(values)),
                        "max_drawdown": float(np.max(values)),
                        "drawdown_volatility": float(np.std(values)),
                        "sample_count": int(len(values)),
                    }

            assessment = "normal"
            if regime in stats:
                cur = stats[regime]
                if self.current_dd > cur["avg_drawdown"] * 1.5:
                    assessment = "above_average"
                elif self.current_dd > cur["max_drawdown"] * 0.9:
                    assessment = "near_maximum"
                elif self.current_dd < cur["avg_drawdown"] * 0.5:
                    assessment = "below_average"

            pattern = self._identify_regime_pattern(regime, stats)

            return {
                "current_regime": regime,
                "regime_stats": stats,
                "regime_assessment": assessment,
                "regime_pattern": pattern,
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_analysis")
            self.logger.warning(f"Regime analysis failed: {error_context}")
            return {
                "current_regime": "unknown",
                "regime_stats": {},
                "regime_assessment": "unknown",
                "regime_pattern": "unknown",
            }

    def _identify_regime_pattern(
        self,
        current_regime: str,
        stats: Dict[str, Dict[str, float]],
    ) -> str:
        try:
            if current_regime not in stats or len(stats) < 2:
                return "insufficient_data"
            cur_avg = stats[current_regime]["avg_drawdown"]
            others = [v["avg_drawdown"] for k, v in stats.items() if k != current_regime]
            if not others:
                return "no_comparison"
            other_avg = float(np.mean(others))
            if cur_avg > other_avg * 1.3:
                return "high_risk_regime"
            if cur_avg < other_avg * 0.7:
                return "low_risk_regime"
            return "normal_risk_regime"
        except Exception:
            return "pattern_analysis_error"

    def _update_rescue_system(
        self,
        dd_results: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update rescue system status and interventions"""
        del market_context  # currently unused, reserved for future context-aware rescue logic
        try:
            sev = dd_results["severity_assessment"]["level"]
            v = float(dd_results["velocity_metrics"]["velocity"])
            a = float(dd_results["velocity_metrics"]["acceleration"])

            was_active = bool(self.rescue_mode)

            # Activation logic
            should_activate = (
                sev in ("warning", "critical")
                or (v > self._cfg.rapid_velocity and self.current_dd > self.current_thresholds["info_dd"])
                or (a > self._cfg.accel_threshold)
            )

            if should_activate and not self.rescue_mode and self.rescue_mode_enabled:
                self.rescue_mode = True
                self.rescue_start_time = datetime.datetime.now()
                self.rescue_intervention_count += 1
                self.logger.warning(
                    format_operator_message(
                        message="Rescue mode ACTIVATED",
                        icon="🛟",
                        drawdown=f"{self.current_dd:.1%}",
                        velocity=f"{v:+.2%}",
                        reason=sev,
                        intervention_count=self.rescue_intervention_count,
                    )
                )

            # Deactivation logic
            if (
                self.rescue_mode
                and sev == "normal"
                and v < 0
                and dd_results["recovery_metrics"]["recovery_progress"] > 0.3
            ):
                dur_min = (
                    (datetime.datetime.now() - (self.rescue_start_time or datetime.datetime.now()))
                    .total_seconds()
                    / 60.0
                )
                self.rescue_mode = False
                self.logger.info(
                    format_operator_message(
                        message="Rescue mode DEACTIVATED",
                        icon="[OK]",
                        duration=f"{dur_min:.1f} minutes",
                        final_dd=f"{self.current_dd:.1%}",
                        recovery=f"{dd_results['recovery_metrics']['recovery_progress']:.1%}",
                    )
                )

            # Emergency intervention (hard brake)
            emergency_intervention = False
            if (
                self.current_dd > self.current_thresholds["dd_limit"] * 1.2
                and v > self._cfg.rapid_velocity * 1.5
            ):
                emergency_intervention = True
                self.emergency_interventions += 1
                self.logger.error(
                    format_operator_message(
                        message="EMERGENCY INTERVENTION",
                        icon="[ALERT]",
                        drawdown=f"{self.current_dd:.1%}",
                        velocity=f"{v:+.2%}",
                        intervention_number=self.emergency_interventions,
                    )
                )

            return {
                "rescue_mode": bool(self.rescue_mode),
                "rescue_activated": (not was_active) and self.rescue_mode,
                "rescue_deactivated": was_active and (not self.rescue_mode),
                "rescue_duration_minutes": (
                    (datetime.datetime.now() - self.rescue_start_time).total_seconds() / 60.0
                    if self.rescue_mode and self.rescue_start_time
                    else 0.0
                ),
                "intervention_count": int(self.rescue_intervention_count),
                "emergency_intervention": bool(emergency_intervention),
                "emergency_count": int(self.emergency_interventions),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "rescue_system")
            self.logger.error(f"Rescue system update failed: {error_context}")
            return {"rescue_mode": False, "error": error_context}

    def _calculate_risk_adjustment(
        self,
        dd_results: Dict[str, Any],
        rescue_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate intelligent risk adjustment factor (EMA-smoothed)"""
        try:
            level = str(dd_results["severity_assessment"]["level"])
            v = float(dd_results["velocity_metrics"]["velocity"])
            rec = dd_results["recovery_metrics"]

            base_map = {"normal": 1.0, "info": 0.9, "warning": 0.7, "critical": 0.4}
            base_adj = float(base_map.get(level, 0.5))

            # Velocity modifier
            if v > self._cfg.rapid_velocity:
                vel_adj = 0.7
            elif v > self._cfg.moderate_velocity:
                vel_adj = 0.85
            elif v < self._cfg.recovery_velocity:
                vel_adj = min(1.2, 1.0 + abs(v) * 5.0)
            else:
                vel_adj = 1.0

            # Recovery modifier
            if float(rec["recovery_progress"]) > 0.5:
                rec_adj = 1.1
            elif float(rec["recovery_velocity"]) > 0.01:
                rec_adj = 1.05
            else:
                rec_adj = 1.0

            # Rescue modifier (progressive damping while active)
            rescue_adj = 1.0
            if rescue_status["rescue_mode"]:
                minutes = float(rescue_status.get("rescue_duration_minutes", 0.0))
                rescue_adj = max(0.3, 1.0 - minutes * float(self._cfg.rescue_decay_per_min))

            # Emergency clamp
            if rescue_status.get("emergency_intervention", False):
                rescue_adj = min(rescue_adj, 0.25)

            raw = float(base_adj * vel_adj * rec_adj * rescue_adj)

            # EMA smoothing
            alpha = float(np.clip(self._cfg.risk_smoothing, 0.0, 1.0))
            self.risk_adjustment_factor = float(
                (1.0 - alpha) * float(self.risk_adjustment_factor) + alpha * raw
            )
            # Bounds
            self.risk_adjustment_factor = float(np.clip(self.risk_adjustment_factor, 0.1, 1.5))

            return {
                "risk_adjustment_factor": float(self.risk_adjustment_factor),
                "base_adjustment": float(base_adj),
                "velocity_adjustment": float(vel_adj),
                "recovery_adjustment": float(rec_adj),
                "rescue_adjustment": float(rescue_adj),
                "raw_adjustment": float(raw),
                "adjustment_reason": self._determine_adjustment_reason(level, rescue_status),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "risk_adjustment")
            self.logger.error(f"Risk adjustment calculation failed: {error_context}")
            return {"risk_adjustment_factor": 0.5, "error": error_context}

    def _determine_adjustment_reason(
        self,
        level: str,
        rescue_status: Dict[str, Any],
    ) -> str:
        if rescue_status.get("emergency_intervention", False):
            return "emergency_intervention"
        if rescue_status.get("rescue_mode", False):
            return "rescue_mode_active"
        if level == "critical":
            return "critical_drawdown"
        if level == "warning":
            return "warning_drawdown"
        if level == "info":
            return "elevated_drawdown"
        return "normal_operation"

    def _calculate_drawdown_metrics(
        self,
        dd_results: Dict[str, Any],
        rescue_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """(Kept for external diagnostics; not used for provides formatting directly)"""
        try:
            v = float(dd_results["velocity_metrics"]["velocity"])
            rec = dd_results["recovery_metrics"]
            total_int = int(self.rescue_intervention_count + self.emergency_interventions)
            success_rate = float(self.successful_recoveries / max(1, total_int))
            trend = "improving" if v < 0 else "deteriorating" if v > 0 else "stable"

            avg_recovery_time = 0.0
            if self.recovery_events:
                recovery_steps: List[int] = []
                prev_step: Optional[int] = None
                for ev in self.recovery_events:
                    sc = int(ev.get("step_count", 0))
                    if prev_step is not None and sc > prev_step:
                        recovery_steps.append(sc - prev_step)
                    prev_step = sc
                if recovery_steps:
                    avg_recovery_time = float(np.mean(recovery_steps))

            return {
                "current_drawdown": float(self.current_dd),
                "max_drawdown": float(self.max_dd),
                "severity_level": str(self.severity_level),
                "risk_adjustment_factor": float(self.risk_adjustment_factor),
                "drawdown_velocity": float(v),
                "recovery_progress": float(rec["recovery_progress"]),
                "rescue_interventions": int(self.rescue_intervention_count),
                "emergency_interventions": int(self.emergency_interventions),
                "successful_recoveries": int(self.successful_recoveries),
                "intervention_success_rate": float(success_rate),
                "risk_trend": trend,
                "avg_recovery_time": float(avg_recovery_time),
                "rescue_mode_active": bool(rescue_status["rescue_mode"]),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "drawdown_metrics")
            self.logger.error(f"Drawdown metrics calculation failed: {error_context}")
            return {"current_drawdown": float(self.current_dd), "error": error_context}

    async def _generate_drawdown_thesis(
        self,
        dd_results: Dict[str, Any],
        rescue_status: Dict[str, Any],
        market_context: Dict[str, Any],
        risk_adj: Dict[str, Any],
    ) -> str:
        """Generate intelligent thesis explaining drawdown analysis"""
        try:
            parts: List[str] = []
            if self.current_dd > 0:
                parts.append(
                    f"Portfolio drawdown {self.current_dd:.1%} (max {self.max_dd:.1%}) from peak €{self.peak_balance:,.0f}"
                )
            else:
                parts.append("Portfolio at/near peak with no material drawdown")

            sev = dd_results["severity_assessment"]["level"]
            v = float(dd_results["velocity_metrics"]["velocity"])

            if sev == "critical":
                parts.append("CRITICAL drawdown level → immediate intervention required")
            elif sev == "warning":
                parts.append("WARNING drawdown level with active monitoring")
            elif sev == "info":
                parts.append("Elevated drawdown within acceptable bounds")
            else:
                parts.append("Drawdown normal and well-controlled")

            if v > self._cfg.rapid_velocity:
                parts.append(f"RAPID deterioration (velocity {v:+.2%}/step)")
            elif v > self._cfg.moderate_velocity:
                parts.append(f"Moderate deterioration (velocity {v:+.2%}/step)")
            elif v < self._cfg.recovery_velocity:
                parts.append(f"Recovery in progress (velocity {v:+.2%}/step)")

            if rescue_status["rescue_mode"]:
                dur = float(rescue_status.get("rescue_duration_minutes", 0.0))
                parts.append(f"Rescue mode ACTIVE for {dur:.1f} min → conservative profile")
            elif rescue_status.get("emergency_intervention", False):
                parts.append("EMERGENCY intervention engaged → maximum risk reduction")

            rec_prog = float(dd_results["recovery_metrics"]["recovery_progress"])
            if rec_prog > 0.5:
                parts.append(f"Strong recovery progress {rec_prog:.1%}")
            elif rec_prog > 0.2:
                parts.append(f"Moderate recovery underway {rec_prog:.1%}")

            regime = market_context.get("regime", "unknown")
            if regime != "unknown" and self.adaptive_thresholds:
                adj_warn = float(dd_results["context_thresholds"]["warning_dd"])
                parts.append(f"Thresholds adjusted for {regime} regime (warning {adj_warn:.1%})")

            parts.append(
                f"Risk adjustment factor {self.risk_adjustment_factor:.1%} ({risk_adj.get('adjustment_reason','normal')})"
            )
            return " | ".join(parts)
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Thesis generation failed: {error_context}"

    # ── fallbacks & errors (contract-safe) ───────────────────
    def _fallback_payload(self, thesis: str) -> Dict[str, Any]:
        dd = {
            "current_drawdown": float(self.current_dd),
            "severity_level": str(self.severity_level),
            "drawdown_analysis": {},
            "rescue_status": {"rescue_mode": bool(self.rescue_mode)},
            "risk_adjustment": {"risk_adjustment_factor": float(self.risk_adjustment_factor)},
            "thesis": thesis,
        }
        rs = {
            "rescue_mode": bool(self.rescue_mode),
            "rescue_duration": 0.0,
            "intervention_count": int(self.rescue_intervention_count),
            "emergency_active": False,
        }
        return {
            "drawdown_risk": dd,
            "rescue_status": rs,
            "risk_adjustment": float(self.risk_adjustment_factor),
            "_thesis": thesis,
            "success": True,
        }

    def _handle_disabled_fallback(self) -> Dict[str, Any]:
        self.severity_level = "disabled"
        self.risk_adjustment_factor = 1.0
        thesis = "Drawdown Rescue is disabled"
        return self._fallback_payload(thesis)

    def _handle_error(self, error: Exception, processing_time_sec: float) -> Dict[str, Any]:
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()
        if self.circuit_breaker["failures"] >= int(self._cfg.circuit_breaker_threshold):
            self.circuit_breaker["state"] = "OPEN"
            self._health_status = "warning"

        explanation = self.english_explainer.explain_error("DrawdownRescue", str(error), "drawdown analysis")
        self.logger.error(
            format_operator_message(
                message="Drawdown module error",
                icon="[CRASH]",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time_sec * 1000.0,
                circuit_breaker_state=self.circuit_breaker["state"],
            )
        )
        self.severity_level = "error"
        self.risk_adjustment_factor = max(0.5, float(self.risk_adjustment_factor))
        return self._fallback_payload(thesis=f"Drawdown error fallback: {str(error)}")

    # ── bookkeeping ──────────────────────────────────────────
    def _record_success(self, processing_time_sec: float) -> None:
        try:
            self._processing_times.append(float(processing_time_sec))
            if self.circuit_breaker["state"] == "CLOSED":
                self.circuit_breaker["failures"] = max(0, self.circuit_breaker["failures"] - 1)
        except Exception:
            pass

    # ── disabled/error helpers required by legacy callers ────
    def _generate_disabled_response(self) -> Dict[str, Any]:
        return self._handle_disabled_fallback()

    def _generate_error_response(self, error_context: str) -> Dict[str, Any]:
        return self._fallback_payload(thesis=f"Drawdown analysis failed: {error_context}")

    def _generate_analysis_error_response(self, error_context: str) -> Dict[str, Any]:
        return {
            "current_drawdown": float(self.current_dd),
            "max_drawdown": float(self.max_dd),
            "peak_balance": float(self.peak_balance),
            "velocity_metrics": {"velocity": 0.0, "acceleration": 0.0},
            "severity_assessment": {"level": "unknown"},
            "recovery_metrics": {"recovery_progress": 0.0},
            "processing_time_ms": 0.0,
            "error": error_context,
        }

    # ── state & health API ───────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        """Get complete module state for hot-reload"""
        return {
            "current_dd": float(self.current_dd),
            "max_dd": float(self.max_dd),
            "peak_balance": float(self.peak_balance),
            "dd_velocity": float(self.dd_velocity),
            "dd_acceleration": float(self.dd_acceleration),
            "severity_level": str(self.severity_level),
            "rescue_mode": bool(self.rescue_mode),
            "rescue_start_time": self.rescue_start_time.isoformat() if self.rescue_start_time else None,
            "risk_adjustment_factor": float(self.risk_adjustment_factor),
            "rescue_intervention_count": int(self.rescue_intervention_count),
            "successful_recoveries": int(self.successful_recoveries),
            "emergency_interventions": int(self.emergency_interventions),
            "step_count": int(self.step_count),
            "config": dict(self.config),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state for hot-reload"""
        try:
            self.current_dd = float(state.get("current_dd", 0.0))
            self.max_dd = float(state.get("max_dd", 0.0))
            self.peak_balance = float(state.get("peak_balance", 0.0))
            self.dd_velocity = float(state.get("dd_velocity", 0.0))
            self.dd_acceleration = float(state.get("dd_acceleration", 0.0))
            self.severity_level = str(state.get("severity_level", "normal"))
            self.rescue_mode = bool(state.get("rescue_mode", False))

            t = state.get("rescue_start_time")
            self.rescue_start_time = datetime.datetime.fromisoformat(t) if t else None

            self.risk_adjustment_factor = float(state.get("risk_adjustment_factor", 1.0))
            self.rescue_intervention_count = int(state.get("rescue_intervention_count", 0))
            self.successful_recoveries = int(state.get("successful_recoveries", 0))
            self.emergency_interventions = int(state.get("emergency_interventions", 0))
            self.step_count = int(state.get("step_count", 0))
            self.config.update(dict(state.get("config", {})))
        except Exception as e:
            self.logger.warning(f"set_state failed: {e}")

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring"""
        return {
            "current_drawdown": float(self.current_dd),
            "max_drawdown": float(self.max_dd),
            "severity_level": str(self.severity_level),
            "rescue_mode": bool(self.rescue_mode),
            "risk_adjustment_factor": float(self.risk_adjustment_factor),
            "intervention_success_rate": float(
                self.successful_recoveries / max(1, self.rescue_intervention_count)
            ),
            "rescue_interventions": int(self.rescue_intervention_count),
            "emergency_interventions": int(self.emergency_interventions),
            "enabled": bool(self.enabled),
        }
