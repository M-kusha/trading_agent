"""
Enhanced Compliance Module with SmartInfoBus Integration
Comprehensive trade validation and regulatory compliance monitoring
(Contract-tight, production-ready, instrument-aware)
"""

from __future__ import annotations

from modules.contracts import module_args
import numpy as np
import datetime
import time
import os
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union, Tuple, Set
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
def _load_compliance_config_from_yaml() -> Dict[str, Any]:
    """Load compliance config values from risk_policy.yaml."""
    import yaml

    defaults: Dict[str, Any] = {}
    try:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "risk_policy.yaml"
        )
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}

            limits = policy.get("limits", {})
            lot_sizing = policy.get("lot_sizing", {})

            # Core limits from risk_policy.yaml
            defaults["max_leverage"] = float(limits.get("max_leverage", 30.0))
            defaults["max_position_risk"] = float(limits.get("max_position_size", 0.05))
            defaults["max_total_risk"] = float(
                limits.get("max_exposure_pct", 0.05) * 2
            )  # Total = 2x position
            defaults["max_drawdown"] = float(limits.get("max_drawdown", 0.085))
            defaults["max_daily_loss"] = float(limits.get("max_daily_loss", 0.042))

            # Lot sizing constraints
            defaults["min_trade_size"] = float(lot_sizing.get("min_lot", 0.01))
            defaults["max_trade_size"] = float(lot_sizing.get("max_lot", 10.0))
    except Exception:
        # On any failure we just keep dataclass defaults
        pass
    return defaults


@dataclass
class ComplianceConfig:
    # Monitoring / health
    health_check_interval: int = 30
    circuit_breaker_threshold: int = 5
    max_processing_time_ms: float = 50.0
    status_key: str = "compliance_module_status"
    health_key: str = "compliance_module_health"

    # Core limits - defaults loaded from risk_policy.yaml
    max_leverage: float = 5.0          # From limits.max_leverage
    max_position_risk: float = 0.05    # From limits.max_position_size
    max_total_risk: float = 0.10       # 2x max_position_size
    max_daily_trades: int = 100
    min_trade_size: float = 0.01       # From lot_sizing.min_lot
    max_trade_size: float = 10.0       # From lot_sizing.max_lot
    # Drawdown / daily loss, usually checked by global DD guardian,
    # but exposed here in risk_limits for env / dashboard.
    max_drawdown: float = 0.085
    max_daily_loss: float = 0.042      # From limits.max_daily_loss

    # Flags
    enabled: bool = True
    dynamic_limits: bool = True
    regime_aware: bool = True

    def __post_init__(self):
        """Override defaults with values from risk_policy.yaml."""
        yaml_config = _load_compliance_config_from_yaml()
        for key, value in yaml_config.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ─────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────
@module(
    **module_args(
        "ComplianceModule",
        description=(
            "Deterministic multi-window feature extraction with circuit breaker, "
            "monitoring, explainability and instrument-aware risk controls."
        ),
        error_handling=True,
        hot_reload=True,
        timeout_ms=3000,
    )
)
class ComplianceModule(
    BaseModule, SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin
):
    """
    Contract guarantees:
    - Writes ONLY its 'provides' keys to SmartInfoBus. Health/status use namespaced keys.
    - Returns ALL 'provides' keys plus '_thesis' and 'success' on success, fallback, or error.
    - Numpy → Python scalars/lists; timestamps are ISO-8601.
    - Background monitor posts namespaced health/status; circuit breaker with safe fallback.

    Risk design:
    - Instrument-aware notional via INSTRUMENT_META (contract_size/category).
    - Per-instrument max_position_risk on top of global limits.
    - Global exposure / leverage + per-instrument exposure snapshot.
    """

    # Default allowed instruments (normalized, we’ll accept EURUSD/EUR_USD/EUR/USD, etc.)
    DEFAULT_ALLOWED_INSTRUMENTS = {
        "EURUSD",
        "XAUUSD",
    }

    # ── init & systems ───────────────────────────────────────
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Merge typed config with dict overrides (only known keys)
        cfg_dict = asdict(ComplianceConfig())
        if isinstance(config, dict):
            for k, v in config.items():
                if k in cfg_dict:
                    cfg_dict[k] = v
        self._cfg = ComplianceConfig(**cfg_dict)

        # Keep raw config for feature-specific items (e.g., allowlist, instrument overrides)
        self.config: Dict[str, Any] = config or {}

        # Initialize low-level systems BEFORE BaseModule may call _initialize()
        self._initialize_advanced_systems()

        # Circuit breaker & health state
        self.circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",
            "threshold": int(self._cfg.circuit_breaker_threshold),
            "cooldown_sec": 20.0,
        }
        self._processing_times: deque[float] = deque(maxlen=100)  # store seconds per cycle
        self._health_status: str = "healthy"
        self._monitoring_active: bool = False
        self._lock = threading.RLock()

        # Core compliance configuration (honor overrides or typed defaults)
        self.max_leverage = float(self.config.get("max_leverage", self._cfg.max_leverage))
        self.max_position_risk = float(
            self.config.get("max_position_risk", self._cfg.max_position_risk)
        )
        self.max_total_risk = float(
            self.config.get("max_total_risk", self._cfg.max_total_risk)
        )
        self.max_daily_trades = int(
            self.config.get("max_daily_trades", self._cfg.max_daily_trades)
        )
        self.min_trade_size = float(
            self.config.get("min_trade_size", self._cfg.min_trade_size)
        )
        self.max_trade_size = float(
            self.config.get("max_trade_size", self._cfg.max_trade_size)
        )
        self.enabled = bool(self.config.get("enabled", self._cfg.enabled))

        # Dynamic risk management
        self.dynamic_limits = bool(
            self.config.get("dynamic_limits", self._cfg.dynamic_limits)
        )
        self.regime_aware = bool(self.config.get("regime_aware", self._cfg.regime_aware))

        # Allowed instruments & restrictions
        self.allowed_instruments: Set[str] = self._initialize_allowed_instruments()
        self.restricted_hours: Set[int] = set(self.config.get("restricted_hours", []))

        # Instrument meta and per-instrument limits
        self.instrument_meta: Dict[str, Dict[str, Any]] = self._initialize_instrument_meta()
        self.max_position_risk_by_instrument: Dict[str, float] = (
            self._initialize_per_instrument_limits()
        )

        # State tracking
        self.daily_trade_count: int = 0
        self.last_trade_date: Optional[datetime.date] = None
        self.total_exposure: float = 0.0
        self.current_leverage: float = 0.0
        self.exposure_by_instrument: Dict[str, float] = {}

        # Risk/compliance stats
        self.risk_budget_usage: float = 0.0
        self.position_limits: Dict[str, Any] = {}
        self.compliance_score: float = 1.0

        self.validation_stats: Dict[str, Any] = {
            "total_validations": 0,
            "approved": 0,
            "rejected": 0,
            "violations": defaultdict(int),
        }
        self.rejection_history: deque[Dict[str, Any]] = deque(maxlen=100)
        self.approval_rate_history: deque[float] = deque(maxlen=50)
        self.compliance_violations: deque[Dict[str, Any]] = deque(maxlen=200)

        # Regime-aware limit tweaks
        self.regime_adjustments: Dict[str, Dict[str, float]] = {
            "volatile": {"leverage": 0.7, "position_risk": 0.8, "daily_trades": 1.2},
            "trending": {"leverage": 1.1, "position_risk": 1.0, "daily_trades": 0.9},
            "ranging": {"leverage": 1.0, "position_risk": 1.1, "daily_trades": 1.0},
            "crisis": {"leverage": 0.5, "position_risk": 0.6, "daily_trades": 0.7},
        }

        super().__init__()  # may call _initialize()

        # Start background heartbeat AFTER BaseModule init
        self._start_monitoring()

        self.logger.info(
            format_operator_message(
                message="Enhanced Compliance Module initialized (instrument-aware)",
                icon="[SAFE]",
                max_leverage=f"{self.max_leverage:.1f}x",
                position_risk_limit=f"{self.max_position_risk:.1%}",
                allowed_instruments=len(self.allowed_instruments),
                dynamic_limits=self.dynamic_limits,
                enabled=self.enabled,
            )
        )

    # ── advanced systems ─────────────────────────────────────
    def _initialize_advanced_systems(self) -> None:
        """Initialize advanced monitoring and error handling systems."""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="ComplianceModule",
            log_path="logs/risk/compliance.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("ComplianceModule", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

    # ── BaseModule hook ──────────────────────────────────────
    def _initialize(self) -> None:
        """Initialize the compliance module (required by BaseModule)."""
        try:
            status = {
                "enabled": bool(self.enabled),
                "compliance_score": float(self.compliance_score),
                "risk_budget_usage": float(self.risk_budget_usage),
                "ts": datetime.datetime.now().isoformat(),
            }
            self.smart_bus.set(
                self._cfg.status_key,
                status,
                module="ComplianceModule",
                thesis="Initial compliance module status",
            )

            # Publish a safe, contract-compliant baseline so consumers don't BUS MISS
            baseline = self._fallback_payload("Initial compliance safe defaults")
            self._write_bus_from_payload(
                baseline, baseline.get("_thesis", "Initial compliance baseline")
            )
            self.logger.info("Compliance module initialization completed successfully")
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "compliance_initialization"
            )
            self.logger.error(f"Compliance initialization failed: {error_context}")

    # ── background monitor ───────────────────────────────────
    def _start_monitoring(self) -> None:
        if self._monitoring_active:
            return

        def loop() -> None:
            self._monitoring_active = True
            self.logger.info("[MONITOR] ComplianceModule health monitor started.")
            while self._monitoring_active:
                try:
                    self._update_compliance_health()

                    # publish namespaced health snapshot
                    health = self.get_health_status()
                    self.smart_bus.set(
                        self._cfg.health_key,
                        health,
                        module="ComplianceModule",
                        thesis="Compliance health heartbeat",
                    )

                    # circuit breaker cooldown auto-reset
                    if self.circuit_breaker["state"] == "OPEN":
                        if (
                            time.time() - self.circuit_breaker["last_failure"]
                            >= self.circuit_breaker["cooldown_sec"]
                        ):
                            self.circuit_breaker["state"] = "CLOSED"
                            self.circuit_breaker["failures"] = 0
                            self.logger.info(
                                "[MONITOR] Circuit breaker auto-reset to CLOSED."
                            )
                except Exception as e:
                    self.logger.error(f"Compliance monitoring error: {e}")
                time.sleep(max(1, int(self._cfg.health_check_interval)))

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def stop_monitoring(self) -> None:
        self._monitoring_active = False

    def _update_compliance_health(self) -> None:
        try:
            self._health_status = "healthy"
            # processing time check
            if len(self._processing_times) >= 10:
                avg_ms = float(np.mean(list(self._processing_times)[-10:]) * 1000.0)
                if avg_ms > float(self._cfg.max_processing_time_ms):
                    self._health_status = "warning"

            # approval trend check
            if self.approval_rate_history:
                avg_approval = float(np.mean(self.approval_rate_history))
                if avg_approval < 0.5:
                    self._health_status = "warning"

            if self.circuit_breaker["state"] == "OPEN":
                self._health_status = "warning"
        except Exception as e:
            self.logger.error(f"Compliance health update failed: {e}")
            self._health_status = "warning"

    def get_health_status(self) -> Dict[str, Any]:
        avg_ms = (
            float(np.mean(self._processing_times) * 1000.0)
            if self._processing_times
            else 0.0
        )
        return {
            "status": self._health_status,
            "avg_processing_time_ms": avg_ms,
            "circuit_breaker_state": self.circuit_breaker["state"],
            "ts": datetime.datetime.now().isoformat(),
        }

    # ── contract-safe process ────────────────────────────────
    async def process(self, **kwargs: Any) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if not self.enabled:
                payload = self._handle_disabled_fallback()
                self._write_bus_from_payload(payload, payload["_thesis"])
                return payload

            # Circuit breaker gating
            if self.circuit_breaker["state"] == "OPEN":
                thesis = "Circuit breaker OPEN; compliance safe fallback emitted."
                payload = self._fallback_payload(thesis)
                self._write_bus_from_payload(payload, thesis)
                return payload

            # Extract inputs from bus (single source of truth)
            market_context = self.smart_bus.get("market_context", "ComplianceModule") or {}
            positions = self.smart_bus.get("positions", "ComplianceModule") or []
            pending_orders = self.smart_bus.get("pending_orders", "ComplianceModule") or []
            balance = self.smart_bus.get("balance", "ComplianceModule") or 10000.0

            # Harden types to match contract expectations
            market_context, positions, pending_orders, balance = self._coerce_bus_inputs(
                market_context, positions, pending_orders, balance
            )

            # Update daily tracking
            self._update_daily_tracking()

            # Adjust limits based on market context
            current_limits = self._calculate_dynamic_limits(market_context)

            # Validate pending orders
            validation_results = await self._validate_pending_orders_comprehensive(
                pending_orders, positions, float(balance), market_context, current_limits
            )

            # Assess current risk exposure (global + per-instrument)
            risk_assessment = self._assess_current_risk_exposure(
                positions, float(balance), current_limits
            )

            # Calculate compliance metrics
            compliance_metrics = self._calculate_compliance_metrics(
                validation_results, risk_assessment
            )

            # Generate thesis (human readable)
            thesis = await self._generate_compliance_thesis(
                validation_results, risk_assessment, market_context
            )

            # Format payload (strict)
            payload = self._format_provides_output(
                validation_results,
                risk_assessment,
                compliance_metrics,
                current_limits,
                thesis,
            )

            # Publish to SmartInfoBus (single-writer for provides)
            self._write_bus_from_payload(payload, thesis)

            # Success metrics
            processing_time_sec = float(time.time() - start_time)
            self._record_success(processing_time_sec)

            # Performance tracker (non-critical)
            try:
                self.performance_tracker.record_metric(
                    "ComplianceModule",
                    "validation_cycle",
                    processing_time_sec * 1000.0,
                    True,
                )
            except Exception:
                pass

            return payload

        except Exception as e:
            processing_time_sec = float(time.time() - start_time)
            payload = self._handle_compliance_error(e, processing_time_sec)
            try:
                self._write_bus_from_payload(
                    payload, payload.get("_thesis", "Compliance error")
                )
            except Exception:
                pass
            return payload

    # ── SmartInfoBus I/O (single-writer) ─────────────────────
    def _write_bus_from_payload(self, payload: Dict[str, Any], thesis: str) -> None:
        try:
            self.smart_bus.set(
                "compliance",
                payload["compliance"],
                module="ComplianceModule",
                thesis="Compliance summary update",
            )
            self.smart_bus.set(
                "compliance_status",
                payload["compliance_status"],
                module="ComplianceModule",
                thesis=thesis,
            )
            self.smart_bus.set(
                "validation_results",
                payload["validation_results"],
                module="ComplianceModule",
                thesis="Order validation results update",
            )
            self.smart_bus.set(
                "risk_limits",
                payload["risk_limits"],
                module="ComplianceModule",
                thesis="Risk limits update",
            )
        except Exception as e:
            err = self.error_pinpointer.analyze_error(e, "bus_write")
            self.logger.error(f"SmartInfoBus update failed: {err}")

    # ── payload formatter (contract enforcer) ────────────────
    def _format_provides_output(
        self,
        validation_results: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        compliance_metrics: Dict[str, Any],
        current_limits: Dict[str, float],
        thesis: str,
    ) -> Dict[str, Any]:
        # Compliance status view (all python types)
        status_payload = {
            "compliance_score": float(
                risk_assessment.get("compliance_score", self.compliance_score)
            ),
            "risk_budget_usage": float(
                risk_assessment.get("risk_budget_usage", self.risk_budget_usage)
            ),
            "validation_results": {
                "total_orders": int(validation_results.get("total_orders", 0)),
                "processing_time_ms": float(
                    validation_results.get("processing_time_ms", 0.0)
                ),
                "violations": list(validation_results.get("violations", []))[:50],
            },
            "risk_assessment": {
                "total_positions": int(risk_assessment.get("total_positions", 0)),
                "current_exposure": float(risk_assessment.get("current_exposure", 0.0)),
                "current_leverage": float(risk_assessment.get("current_leverage", 0.0)),
                "leverage_usage": float(risk_assessment.get("leverage_usage", 0.0)),
                "exposure_usage": float(risk_assessment.get("exposure_usage", 0.0)),
                "daily_trades_usage": float(
                    risk_assessment.get("daily_trades_usage", 0.0)
                ),
                "violation_rate": float(risk_assessment.get("violation_rate", 0.0)),
                "per_instrument_exposure": {
                    str(k): float(v)
                    for k, v in dict(
                        risk_assessment.get("per_instrument_exposure", {})
                    ).items()
                },
            },
            "compliance_metrics": {
                "approval_rate": float(compliance_metrics.get("approval_rate", 1.0)),
                "rejection_rate": float(
                    compliance_metrics.get("rejection_rate", 0.0)
                ),
                "avg_approval_rate": float(
                    compliance_metrics.get("avg_approval_rate", 1.0)
                ),
                "total_validations": int(
                    compliance_metrics.get("total_validations", 0)
                ),
                "total_violations": int(
                    compliance_metrics.get("total_violations", 0)
                ),
                "daily_trade_utilization": float(
                    compliance_metrics.get("daily_trade_utilization", 0.0)
                ),
                "violation_breakdown": {
                    str(k): int(v)
                    for k, v in dict(
                        compliance_metrics.get("violation_breakdown", {})
                    ).items()
                },
            },
            "current_limits": {
                "max_leverage": float(
                    current_limits.get("max_leverage", self.max_leverage)
                ),
                "max_position_risk": float(
                    current_limits.get(
                        "max_position_risk", self.max_position_risk
                    )
                ),
                "max_total_risk": float(
                    current_limits.get("max_total_risk", self.max_total_risk)
                ),
                "max_daily_trades": int(
                    current_limits.get("max_daily_trades", self.max_daily_trades)
                ),
                "min_trade_size": float(
                    current_limits.get("min_trade_size", self.min_trade_size)
                ),
                "max_trade_size": float(
                    current_limits.get("max_trade_size", self.max_trade_size)
                ),
                "max_drawdown": float(
                    current_limits.get("max_drawdown", self._cfg.max_drawdown)
                ),
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # top violations (type → count) for compact `compliance` summary
        vio_counts: Dict[str, int] = {}
        for v in validation_results.get("violations", []):
            t = str(v.get("type", "unknown"))
            vio_counts[t] = vio_counts.get(t, 0) + 1
        top_vios = sorted(vio_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_vios_payload = [{"type": k, "count": int(v)} for k, v in top_vios]

        # Validation results (provides)
        vr_payload = {
            "approved_orders": list(validation_results.get("approved", []))[:200],
            "rejected_orders": list(validation_results.get("rejected", []))[:200],
            "approval_rate": float(compliance_metrics.get("approval_rate", 1.0)),
        }

        # Risk limits (provides)
        rl_payload = {
            "max_leverage": float(
                current_limits.get("max_leverage", self.max_leverage)
            ),
            "max_position_risk": float(
                current_limits.get(
                    "max_position_risk", self.max_position_risk
                )
            ),
            "max_total_risk": float(
                current_limits.get("max_total_risk", self.max_total_risk)
            ),
            "max_drawdown": float(
                current_limits.get("max_drawdown", self._cfg.max_drawdown)
            ),
            "current_leverage": float(self.current_leverage),
            "risk_budget_usage": float(self.risk_budget_usage),
        }

        # Compliance (provides) — compact summary used by downstream risk modules
        comp_status = "ok"
        if self.circuit_breaker["state"] == "OPEN":
            comp_status = "open_circuit"
        elif (
            self._health_status != "healthy"
            or float(risk_assessment.get("violation_rate", 0.0)) > 0.25
        ):
            comp_status = "warning"

        compliance_payload = {
            "status": comp_status,
            "score": float(
                risk_assessment.get("compliance_score", self.compliance_score)
            ),
            "risk_budget_usage": float(
                risk_assessment.get("risk_budget_usage", self.risk_budget_usage)
            ),
            "violation_rate": float(risk_assessment.get("violation_rate", 0.0)),
            "top_violations": top_vios_payload,
            "limits": {
                "max_leverage": float(rl_payload["max_leverage"]),
                "max_total_risk": float(rl_payload["max_total_risk"]),
                "max_position_risk": float(rl_payload["max_position_risk"]),
            },
            "ts": datetime.datetime.now().isoformat(),
        }

        return {
            "compliance": compliance_payload,
            "compliance_status": status_payload,
            "validation_results": vr_payload,
            "risk_limits": rl_payload,
            "_thesis": thesis,
            "success": True,
        }

    # ── daily tracking ───────────────────────────────────────
    def _update_daily_tracking(self) -> None:
        """Update daily trade count tracking."""
        try:
            current_date = datetime.datetime.now().date()

            # Reset counter for new day
            if self.last_trade_date != current_date:
                if self.last_trade_date and self.daily_trade_count > 0:
                    self.logger.info(
                        format_operator_message(
                            message="Daily trading summary",
                            icon="[STATS]",
                            date=str(self.last_trade_date),
                            trades=self.daily_trade_count,
                            limit=self.max_daily_trades,
                        )
                    )

                self.daily_trade_count = 0
                self.last_trade_date = current_date

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "daily_tracking")
            self.logger.warning(f"Daily tracking update failed: {error_context}")

    # ── dynamic limits ───────────────────────────────────────
    def _calculate_dynamic_limits(
        self, market_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate dynamic limits based on market context."""
        try:
            # Start with base limits
            current_limits: Dict[str, float] = {
                "max_leverage": float(self.max_leverage),
                "max_position_risk": float(self.max_position_risk),
                "max_total_risk": float(self.max_total_risk),
                "max_daily_trades": int(self.max_daily_trades),
                "min_trade_size": float(self.min_trade_size),
                "max_trade_size": float(self.max_trade_size),
                # Provide drawdown limit even if not dynamically adjusted
                "max_drawdown": float(self._cfg.max_drawdown),
            }

            if not self.dynamic_limits:
                return current_limits

            regime = str(market_context.get("regime", "unknown"))
            volatility_level = str(market_context.get("volatility_level", "medium"))

            # Regime adjustments
            if regime in self.regime_adjustments:
                adjustments = self.regime_adjustments[regime]
                current_limits["max_leverage"] *= float(adjustments["leverage"])
                current_limits["max_position_risk"] *= float(
                    adjustments["position_risk"]
                )
                current_limits["max_daily_trades"] = int(
                    current_limits["max_daily_trades"]
                    * float(adjustments["daily_trades"])
                )

            # Volatility adjustments
            volatility_multipliers: Dict[str, Dict[str, float]] = {
                "low": {"leverage": 1.1, "position_risk": 1.1, "trade_size": 1.0},
                "medium": {"leverage": 1.0, "position_risk": 1.0, "trade_size": 1.0},
                "high": {"leverage": 0.8, "position_risk": 0.8, "trade_size": 0.9},
                "extreme": {"leverage": 0.6, "position_risk": 0.6, "trade_size": 0.8},
            }

            if volatility_level in volatility_multipliers:
                vol_adj = volatility_multipliers[volatility_level]
                current_limits["max_leverage"] *= float(vol_adj["leverage"])
                current_limits["max_position_risk"] *= float(
                    vol_adj["position_risk"]
                )
                current_limits["max_trade_size"] *= float(vol_adj["trade_size"])

            # Apply bounds to prevent extreme adjustments
            current_limits["max_leverage"] = max(
                1.0, min(100.0, current_limits["max_leverage"])
            )
            current_limits["max_position_risk"] = max(
                0.01, min(0.5, current_limits["max_position_risk"])
            )
            current_limits["max_total_risk"] = max(
                0.05, min(1.0, current_limits["max_total_risk"])
            )

            return current_limits

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "dynamic_limits"
            )
            self.logger.warning(
                f"Dynamic limits calculation failed, using static limits: {error_context}"
            )
            return {
                "max_leverage": float(self.max_leverage),
                "max_position_risk": float(self.max_position_risk),
                "max_total_risk": float(self.max_total_risk),
                "max_daily_trades": int(self.max_daily_trades),
                "min_trade_size": float(self.min_trade_size),
                "max_trade_size": float(self.max_trade_size),
                "max_drawdown": float(self._cfg.max_drawdown),
            }

    # ── validations ──────────────────────────────────────────
    async def _validate_pending_orders_comprehensive(
        self,
        pending_orders: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
        balance: float,
        market_context: Dict[str, Any],
        current_limits: Dict[str, float],
    ) -> Dict[str, Any]:
        """Comprehensive validation of pending orders."""
        start_time = datetime.datetime.now()

        validation_results: Dict[str, Any] = {
            "total_orders": int(len(pending_orders)),
            "approved": [],
            "rejected": [],
            "violations": [],
            "validation_details": [],
        }

        try:
            for order in pending_orders:
                order_validation = await self._validate_single_order(
                    order, positions, balance, market_context, current_limits
                )

                validation_results["validation_details"].append(order_validation)

                if order_validation["approved"]:
                    validation_results["approved"].append(order)
                    self.validation_stats["approved"] += 1
                else:
                    validation_results["rejected"].append(order)
                    validation_results["violations"].extend(
                        order_validation["violations"]
                    )

                    # Track rejection reasons
                    for violation in order_validation["violations"]:
                        self.validation_stats["violations"][
                            violation["type"]
                        ] += 1

                self.validation_stats["total_validations"] += 1

            # Calculate processing time
            processing_time = (
                datetime.datetime.now() - start_time
            ).total_seconds() * 1000.0
            validation_results["processing_time_ms"] = float(processing_time)

            # Update approval rate
            if self.validation_stats["total_validations"] > 0:
                approval_rate = (
                    self.validation_stats["approved"]
                    / self.validation_stats["total_validations"]
                )
                self.approval_rate_history.append(float(approval_rate))

            return validation_results

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "order_validation"
            )
            self.logger.error(f"Order validation failed: {error_context}")
            # Return a contract-complete but empty result on error
            return {
                "error": error_context,
                "total_orders": int(len(pending_orders)),
                "approved": [],
                "rejected": [],
                "violations": [],
                "validation_details": [],
                "processing_time_ms": 0.0,
            }

    async def _validate_single_order(
        self,
        order: Dict[str, Any],
        positions: List[Dict[str, Any]],
        balance: float,
        market_context: Dict[str, Any],
        current_limits: Dict[str, float],
    ) -> Dict[str, Any]:
        """Validate a single order with comprehensive checks."""
        order_details: Dict[str, Any] = {
            "instrument": "UNKNOWN",
            "size": 0.0,
            "side": "UNKNOWN",
            "price": 1.0,
        }

        try:
            violations: List[Dict[str, Any]] = []

            raw_instrument: str = str(
                order.get("instrument", order.get("symbol", "UNKNOWN"))
            )
            instrument = self._normalize_single_instrument(raw_instrument)

            size_raw = order.get("size", order.get("volume", None))
            size = (
                abs(float(size_raw))
                if size_raw is not None
                else 0.0
            )
            side = order.get(
                "side", "BUY" if float(order.get("size", 0.0)) > 0 else "SELL"
            )
            price = float(order.get("price", order.get("current_price", 1.0)))

            order_details.update(
                {
                    "instrument": instrument,
                    "size": size,
                    "side": side,
                    "price": price,
                }
            )

            # 1. Daily trade limit check
            if self.daily_trade_count >= int(current_limits["max_daily_trades"]):
                violations.append(
                    {
                        "type": "daily_trade_limit",
                        "message": (
                            f"Daily trade limit exceeded: "
                            f"{self.daily_trade_count}/{int(current_limits['max_daily_trades'])}"
                        ),
                        "severity": "critical",
                    }
                )

            # 2. Instrument allowlist check
            if not self._is_instrument_allowed(instrument):
                violations.append(
                    {
                        "type": "instrument_not_allowed",
                        "message": f"Instrument {instrument} not in allowlist",
                        "severity": "critical",
                    }
                )

            # 3. Trading hours check
            if self._is_trading_restricted():
                violations.append(
                    {
                        "type": "restricted_hours",
                        "message": "Trading restricted during current hour",
                        "severity": "warning",
                    }
                )

            # 4. Trade size validation
            size_violations = self._validate_trade_size(size, current_limits)
            violations.extend(size_violations)

            # 5. Position risk validation (instrument-aware)
            position_risk_violations = self._validate_position_risk(
                order_details, positions, balance, current_limits
            )
            violations.extend(position_risk_violations)

            # 6. Total exposure validation (global + per instrument)
            exposure_violations = self._validate_total_exposure(
                order_details, positions, balance, current_limits
            )
            violations.extend(exposure_violations)

            # 7. Leverage validation (using updated total_exposure)
            leverage_violations = self._validate_leverage_limits(
                order_details, positions, balance, current_limits
            )
            violations.extend(leverage_violations)

            # 8. Market context validation
            context_violations = self._validate_market_context(
                order_details, market_context
            )
            violations.extend(context_violations)

            # Determine approval status
            critical_violations = [
                v for v in violations if v.get("severity") == "critical"
            ]
            approved = len(critical_violations) == 0

            if approved:
                # Increment trade counter on approval
                self.daily_trade_count += 1

            return {
                "order_details": order_details,
                "approved": bool(approved),
                "violations": violations,
                "risk_score": float(len(violations) / 8.0),  # Normalize by number of checks
                "validation_timestamp": datetime.datetime.now().isoformat(),
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "single_order_validation"
            )
            return {
                "order_details": order_details,
                "approved": False,
                "violations": [
                    {
                        "type": "validation_error",
                        "message": error_context,
                        "severity": "critical",
                    }
                ],
                "risk_score": 1.0,
                "error": error_context,
            }

    # ── atomic validation helpers ────────────────────────────
    def _is_instrument_allowed(self, instrument: str) -> bool:
        """Check if instrument is in allowlist (normalized)."""
        symbol = self._normalize_single_instrument(instrument)
        return symbol in self.allowed_instruments

    def _is_trading_restricted(self) -> bool:
        """Check if trading is restricted at current hour."""
        if not self.restricted_hours:
            return False
        current_hour = datetime.datetime.now().hour
        return current_hour in self.restricted_hours

    def _validate_trade_size(
        self, size: float, current_limits: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Validate trade size against limits."""
        violations: List[Dict[str, Any]] = []

        if size < float(current_limits["min_trade_size"]):
            violations.append(
                {
                    "type": "size_too_small",
                    "message": (
                        f"Trade size {size:.4f} below minimum "
                        f"{float(current_limits['min_trade_size']):.4f}"
                    ),
                    "severity": "warning",
                }
            )

        if size > float(current_limits["max_trade_size"]):
            violations.append(
                {
                    "type": "size_too_large",
                    "message": (
                        f"Trade size {size:.4f} exceeds maximum "
                        f"{float(current_limits['max_trade_size']):.4f}"
                    ),
                    "severity": "critical",
                }
            )

        return violations

    def _validate_position_risk(
        self,
        order_details: Dict[str, Any],
        positions: List[Dict[str, Any]],
        balance: float,
        current_limits: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Validate position risk limits (instrument-aware)."""
        violations: List[Dict[str, Any]] = []

        try:
            instrument = self._normalize_single_instrument(
                order_details.get("instrument", "UNKNOWN")
            )
            size = float(order_details["size"])
            price = float(order_details["price"])

            position_value = self._calculate_notional_value(
                instrument, size, price
            )

            # Calculate risk as percentage of balance
            position_risk = position_value / max(float(balance), 1.0)

            # Per-instrument override, fallback to global limit
            inst_limit = self.max_position_risk_by_instrument.get(instrument)
            max_risk_allowed = float(
                inst_limit
                if inst_limit is not None
                else current_limits["max_position_risk"]
            )

            if position_risk > max_risk_allowed:
                violations.append(
                    {
                        "type": "position_risk_exceeded",
                        "message": (
                            f"Position risk {position_risk:.1%} exceeds "
                            f"limit {max_risk_allowed:.1%} for {instrument}"
                        ),
                        "severity": "critical",
                    }
                )

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "position_risk_validation"
            )
            violations.append(
                {
                    "type": "position_risk_calculation_error",
                    "message": f"Risk calculation failed: {error_context}",
                    "severity": "warning",
                }
            )

        return violations

    def _validate_total_exposure(
        self,
        order_details: Dict[str, Any],
        positions: List[Dict[str, Any]],
        balance: float,
        current_limits: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Validate total portfolio exposure (instrument-aware notional)."""
        violations: List[Dict[str, Any]] = []

        try:
            existing_exposure = 0.0
            exposure_by_instrument: Dict[str, float] = {}

            # Existing positions
            for position in positions:
                raw_inst = position.get("instrument", position.get("symbol", "UNKNOWN"))
                instrument = self._normalize_single_instrument(raw_inst)
                pos_size = abs(
                    float(position.get("size", position.get("volume", 0.0)))
                )
                pos_price = float(
                    position.get("current_price", position.get("price", 1.0))
                )
                notional = self._calculate_notional_value(
                    instrument, pos_size, pos_price
                )
                existing_exposure += notional
                exposure_by_instrument[instrument] = (
                    exposure_by_instrument.get(instrument, 0.0) + notional
                )

            # New order exposure
            new_instrument = self._normalize_single_instrument(
                order_details.get("instrument", "UNKNOWN")
            )
            new_size = float(order_details["size"])
            new_price = float(order_details["price"])
            new_notional = self._calculate_notional_value(
                new_instrument, new_size, new_price
            )

            total_exposure = existing_exposure + new_notional
            exposure_by_instrument[new_instrument] = (
                exposure_by_instrument.get(new_instrument, 0.0) + new_notional
            )

            # Calculate total risk
            total_risk = total_exposure / max(float(balance), 1.0)

            if total_risk > float(current_limits["max_total_risk"]):
                violations.append(
                    {
                        "type": "total_risk_exceeded",
                        "message": (
                            f"Total risk {total_risk:.1%} exceeds limit "
                            f"{float(current_limits['max_total_risk']):.1%}"
                        ),
                        "severity": "critical",
                    }
                )

            # Update exposure tracking
            self.total_exposure = float(total_exposure)
            self.exposure_by_instrument = {
                str(k): float(v) for k, v in exposure_by_instrument.items()
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "total_exposure_validation"
            )
            violations.append(
                {
                    "type": "exposure_calculation_error",
                    "message": f"Exposure calculation failed: {error_context}",
                    "severity": "warning",
                }
            )

        return violations

    def _validate_leverage_limits(
        self,
        order_details: Dict[str, Any],
        positions: List[Dict[str, Any]],
        balance: float,
        current_limits: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Validate leverage limits."""
        violations: List[Dict[str, Any]] = []

        try:
            # Leverage after applying _validate_total_exposure (includes new order)
            leverage = self.total_exposure / max(float(balance), 1.0)
            self.current_leverage = float(leverage)

            if leverage > float(current_limits["max_leverage"]):
                violations.append(
                    {
                        "type": "leverage_exceeded",
                        "message": (
                            f"Leverage {leverage:.1f}x exceeds limit "
                            f"{float(current_limits['max_leverage']):.1f}x"
                        ),
                        "severity": "critical",
                    }
                )

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "leverage_validation"
            )
            violations.append(
                {
                    "type": "leverage_calculation_error",
                    "message": f"Leverage calculation failed: {error_context}",
                    "severity": "warning",
                }
            )

        return violations

    def _validate_market_context(
        self, order_details: Dict[str, Any], market_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate against market context restrictions."""
        violations: List[Dict[str, Any]] = []

        try:
            volatility_level = str(market_context.get("volatility_level", "medium"))
            regime = str(market_context.get("regime", "unknown"))

            size = float(order_details["size"])

            # Restrict large trades during extreme volatility
            if (
                volatility_level == "extreme"
                and size > float(self.max_trade_size) * 0.5
            ):
                violations.append(
                    {
                        "type": "extreme_volatility_restriction",
                        "message": "Large trade restricted during extreme volatility",
                        "severity": "warning",
                    }
                )

            # Crisis regime restrictions
            if regime == "crisis" and size > float(self.max_trade_size) * 0.3:
                violations.append(
                    {
                        "type": "crisis_regime_restriction",
                        "message": "Trade size restricted during crisis regime",
                        "severity": "warning",
                    }
                )

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "market_context_validation"
            )
            violations.append(
                {
                    "type": "context_validation_error",
                    "message": f"Context validation failed: {error_context}",
                    "severity": "info",
                }
            )

        return violations

    # ── risk & metrics ───────────────────────────────────────
    def _assess_current_risk_exposure(
        self,
        positions: List[Dict[str, Any]],
        balance: float,
        current_limits: Dict[str, float],
    ) -> Dict[str, Any]:
        """Assess current risk exposure and compliance status."""
        try:
            # Recompute exposures from current positions (no pending orders)
            exposure_by_instrument: Dict[str, float] = {}
            total_exposure = 0.0

            for position in positions:
                raw_inst = position.get("instrument", position.get("symbol", "UNKNOWN"))
                instrument = self._normalize_single_instrument(raw_inst)
                pos_size = abs(
                    float(position.get("size", position.get("volume", 0.0)))
                )
                pos_price = float(
                    position.get("current_price", position.get("price", 1.0))
                )
                notional = self._calculate_notional_value(
                    instrument, pos_size, pos_price
                )
                total_exposure += notional
                exposure_by_instrument[instrument] = (
                    exposure_by_instrument.get(instrument, 0.0) + notional
                )

            self.total_exposure = float(total_exposure)
            self.exposure_by_instrument = {
                str(k): float(v) for k, v in exposure_by_instrument.items()
            }

            total_positions = int(len(positions))
            current_exposure = float(self.total_exposure)
            current_leverage = (
                current_exposure / max(float(balance), 1.0)
                if balance > 0
                else 0.0
            )
            self.current_leverage = float(current_leverage)

            # Calculate risk budget usage
            leverage_usage = current_leverage / max(
                float(current_limits.get("max_leverage", self.max_leverage)), 1e-6
            )
            exposure_usage = (current_exposure / max(float(balance), 1.0)) / max(
                float(current_limits.get("max_total_risk", self.max_total_risk)),
                1e-6,
            )
            daily_trades_usage = self.daily_trade_count / max(
                int(current_limits.get("max_daily_trades", self.max_daily_trades)), 1
            )

            # Overall risk budget usage
            self.risk_budget_usage = float(
                max(leverage_usage, exposure_usage, daily_trades_usage)
            )

            # Violations so far
            violations_count = int(
                sum(self.validation_stats["violations"].values())
            )
            total_validations = max(
                int(self.validation_stats["total_validations"]), 1
            )
            violation_rate = float(violations_count / total_validations)

            # Score is diminished by violation rate and by budget usage beyond 1.0
            overuse_penalty = max(0.0, (self.risk_budget_usage - 1.0) * 0.5)
            self.compliance_score = float(
                max(0.0, 1.0 - violation_rate - overuse_penalty)
            )

            return {
                "total_positions": total_positions,
                "current_exposure": current_exposure,
                "current_leverage": current_leverage,
                "risk_budget_usage": float(self.risk_budget_usage),
                "compliance_score": float(self.compliance_score),
                "daily_trade_count": int(self.daily_trade_count),
                "leverage_usage": float(leverage_usage),
                "exposure_usage": float(exposure_usage),
                "daily_trades_usage": float(daily_trades_usage),
                "violation_rate": float(violation_rate),
                "per_instrument_exposure": dict(self.exposure_by_instrument),
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "risk_assessment"
            )
            self.logger.error(f"Risk assessment failed: {error_context}")
            return {
                "error": error_context,
                "compliance_score": 0.5,
                "per_instrument_exposure": dict(self.exposure_by_instrument),
            }

    def _calculate_compliance_metrics(
        self,
        validation_results: Dict[str, Any],
        risk_assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate comprehensive compliance metrics."""
        try:
            # Validation metrics
            total_orders = int(validation_results.get("total_orders", 0))
            approved_orders = int(len(validation_results.get("approved", [])))
            rejected_orders = int(len(validation_results.get("rejected", [])))

            approval_rate = float(approved_orders / max(total_orders, 1))
            rejection_rate = float(rejected_orders / max(total_orders, 1))

            # Risk metrics
            risk_budget_usage = float(
                risk_assessment.get("risk_budget_usage", 0.0)
            )
            compliance_score = float(
                risk_assessment.get("compliance_score", 1.0)
            )

            # Historical metrics
            avg_approval_rate = (
                float(np.mean(self.approval_rate_history))
                if self.approval_rate_history
                else 1.0
            )

            # Violation breakdown
            violation_breakdown = dict(self.validation_stats["violations"])

            return {
                "approval_rate": float(approval_rate),
                "rejection_rate": float(rejection_rate),
                "avg_approval_rate": float(avg_approval_rate),
                "compliance_score": float(compliance_score),
                "risk_budget_usage": float(risk_budget_usage),
                "total_validations": int(
                    self.validation_stats["total_validations"]
                ),
                "total_violations": int(sum(violation_breakdown.values())),
                "violation_breakdown": {
                    str(k): int(v) for k, v in violation_breakdown.items()
                },
                "daily_trade_utilization": float(
                    self.daily_trade_count / max(self.max_daily_trades, 1)
                ),
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "compliance_metrics"
            )
            self.logger.error(
                f"Compliance metrics calculation failed: {error_context}"
            )
            return {"compliance_score": 0.5, "error": error_context}

    # ── thesis & recommendations ─────────────────────────────
    async def _generate_compliance_thesis(
        self,
        validation_results: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> str:
        """Generate intelligent thesis explaining compliance decisions."""
        try:
            thesis_parts: List[str] = []

            # Validation overview
            total_orders = int(validation_results.get("total_orders", 0))
            approved = int(len(validation_results.get("approved", [])))
            rejected = int(len(validation_results.get("rejected", [])))

            if total_orders > 0:
                thesis_parts.append(
                    f"Processed {total_orders} orders: {approved} approved, "
                    f"{rejected} rejected ({approved / max(total_orders, 1):.1%} approval rate)"
                )
            else:
                thesis_parts.append("No pending orders to validate")

            # Risk assessment
            compliance_score = float(
                risk_assessment.get("compliance_score", 1.0)
            )
            risk_budget_usage = float(
                risk_assessment.get("risk_budget_usage", 0.0)
            )

            if compliance_score >= 0.9:
                thesis_parts.append(
                    f"EXCELLENT compliance maintained ({compliance_score:.1%})"
                )
            elif compliance_score >= 0.7:
                thesis_parts.append(
                    f"GOOD compliance status ({compliance_score:.1%})"
                )
            elif compliance_score >= 0.5:
                thesis_parts.append(
                    f"FAIR compliance with room for improvement ({compliance_score:.1%})"
                )
            else:
                thesis_parts.append(
                    f"POOR compliance requiring immediate attention ({compliance_score:.1%})"
                )

            # Risk budget analysis
            if risk_budget_usage > 0.8:
                thesis_parts.append(
                    f"HIGH risk budget utilization ({risk_budget_usage:.1%}) - approaching limits"
                )
            elif risk_budget_usage > 0.5:
                thesis_parts.append(
                    f"MODERATE risk budget usage ({risk_budget_usage:.1%})"
                )
            else:
                thesis_parts.append(
                    f"Conservative risk budget usage ({risk_budget_usage:.1%})"
                )

            # Violation analysis
            violations = validation_results.get("violations", [])
            if violations:
                violation_types = {
                    str(v.get("type", "unknown")) for v in violations
                }
                thesis_parts.append(
                    f"Detected {len(violations)} violations: "
                    f"{', '.join(list(violation_types)[:3])}"
                )

            # Market context impact
            regime = str(market_context.get("regime", "unknown"))
            volatility = str(market_context.get("volatility_level", "medium"))

            if self.dynamic_limits:
                thesis_parts.append(
                    f"Dynamic limits adjusted for {regime} regime and {volatility} volatility"
                )

            # Daily trading status
            daily_usage = self.daily_trade_count / max(
                self.max_daily_trades, 1
            )
            if daily_usage > 0.8:
                thesis_parts.append(
                    f"Daily trade limit utilization HIGH ({daily_usage:.1%})"
                )

            return " | ".join(thesis_parts)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "thesis_generation"
            )
            return f"Thesis generation failed: {error_context}"

    # ── instruments & config helpers ─────────────────────────
    def _normalize_single_instrument(self, instrument: Union[str, Any]) -> str:
        """Normalize instrument symbol to a stable key (e.g., EURUSD, XAUUSD)."""
        s = str(instrument).strip().upper()
        if not s:
            return "UNKNOWN"
        for sep in ("/", "_", "-", " "):
            s = s.replace(sep, "")
        return s or "UNKNOWN"

    def _initialize_allowed_instruments(self) -> Set[str]:
        """Initialize allowed instruments from config or environment."""
        try:
            # Environment variable wins
            env_instruments = os.getenv("COMPLIANCE_INSTRUMENTS")
            if env_instruments:
                instruments = [
                    inst.strip() for inst in env_instruments.split(",")
                ]
            else:
                # Use config or defaults
                config_instruments = self.config.get(
                    "allowed_instruments",
                    list(self.DEFAULT_ALLOWED_INSTRUMENTS),
                )
                instruments = list(config_instruments)

            normalized: Set[str] = set()
            for inst in instruments:
                norm = self._normalize_single_instrument(inst)
                if norm and norm != "UNKNOWN":
                    normalized.add(norm)
            return normalized

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "instrument_initialization"
            )
            self.logger.warning(
                f"Instrument initialization failed, using defaults: {error_context}"
            )
            normalized: Set[str] = set()
            for inst in self.DEFAULT_ALLOWED_INSTRUMENTS:
                norm = self._normalize_single_instrument(inst)
                if norm and norm != "UNKNOWN":
                    normalized.add(norm)
            return normalized

    def _initialize_instrument_meta(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize instrument metadata including contract_size and category.

        Defaults:
        - FX pairs (e.g., EURUSD): contract_size ≈ 100_000
        - XAUUSD: contract_size ≈ 100
        """
        meta: Dict[str, Dict[str, Any]] = {}

        # Sensible defaults
        default_meta: Dict[str, Dict[str, Any]] = {
            "EURUSD": {"contract_size": 100_000.0, "category": "fx"},
            "XAUUSD": {"contract_size": 100.0, "category": "metal"},
        }

        for sym, info in default_meta.items():
            norm = self._normalize_single_instrument(sym)
            meta[norm] = dict(info)

        # Config-level overrides
        cfg_meta = self.config.get("instrument_meta", {})
        if isinstance(cfg_meta, dict):
            for key, info in cfg_meta.items():
                norm = self._normalize_single_instrument(key)
                if not norm or norm == "UNKNOWN":
                    continue
                base = dict(meta.get(norm, {}))
                if isinstance(info, dict):
                    if "contract_size" in info:
                        try:
                            base["contract_size"] = float(info["contract_size"])
                        except Exception:
                            pass
                    if "category" in info:
                        base["category"] = str(info["category"])
                meta[norm] = base

        # risk_policy.yaml overrides
        try:
            import yaml

            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "config", "risk_policy.yaml"
            )
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    policy = yaml.safe_load(f) or {}
                yaml_meta = policy.get("instrument_meta", {})
                if isinstance(yaml_meta, dict):
                    for key, info in yaml_meta.items():
                        norm = self._normalize_single_instrument(key)
                        if not norm or norm == "UNKNOWN":
                            continue
                        base = dict(meta.get(norm, {}))
                        if isinstance(info, dict):
                            if "contract_size" in info:
                                try:
                                    base["contract_size"] = float(
                                        info["contract_size"]
                                    )
                                except Exception:
                                    pass
                            if "category" in info:
                                base["category"] = str(info["category"])
                        meta[norm] = base
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "instrument_meta_initialization"
            )
            self.logger.warning(
                f"Instrument meta initialization failed partially: {error_context}"
            )

        return meta

    def _initialize_per_instrument_limits(self) -> Dict[str, float]:
        """
        Initialize per-instrument max position risk (fraction of balance).

        Order of precedence:
        - config["max_position_risk_by_instrument"]
        - risk_policy.yaml["limits_by_instrument"][symbol]["max_position_size"]
        """
        limits: Dict[str, float] = {}

        # Config-level mapping
        cfg_limits = self.config.get("max_position_risk_by_instrument", {})
        if isinstance(cfg_limits, dict):
            for key, val in cfg_limits.items():
                norm = self._normalize_single_instrument(key)
                if not norm or norm == "UNKNOWN":
                    continue
                try:
                    limits[norm] = float(val)
                except Exception:
                    continue

        # risk_policy.yaml mapping
        try:
            import yaml

            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "config", "risk_policy.yaml"
            )
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    policy = yaml.safe_load(f) or {}
                per_inst = policy.get("limits_by_instrument", {})
                if isinstance(per_inst, dict):
                    for key, info in per_inst.items():
                        norm = self._normalize_single_instrument(key)
                        if not norm or norm == "UNKNOWN":
                            continue
                        if not isinstance(info, dict):
                            continue
                        val = info.get("max_position_size", info.get("max_position_risk"))
                        if val is None:
                            continue
                        try:
                            limits[norm] = float(val)
                        except Exception:
                            continue
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(
                e, "per_instrument_limits_initialization"
            )
            self.logger.warning(
                f"Per-instrument limits initialization had issues: {error_context}"
            )

        return limits

    def _calculate_notional_value(
        self, instrument: str, size: float, price: float
    ) -> float:
        """
        Compute notional exposure for a given instrument/size/price using
        instrument_meta.contract_size where available.

        If metadata is missing, falls back to 100_000 as a generic contract size.
        """
        norm = self._normalize_single_instrument(instrument)
        meta = self.instrument_meta.get(norm, {})
        contract_size = float(meta.get("contract_size", 100_000.0))
        return float(size) * float(price) * contract_size

    # ── fallbacks & errors (contract-safe) ───────────────────
    def _fallback_payload(self, thesis: str) -> Dict[str, Any]:
        """Build a minimal but contract-complete snapshot."""
        compliance_score = float(self.compliance_score)
        risk_budget_usage = float(self.risk_budget_usage)
        status = (
            "open_circuit"
            if self.circuit_breaker["state"] == "OPEN"
            else "warning"
            if self._health_status != "healthy"
            else "ok"
        )

        return {
            "compliance": {
                "status": status,
                "score": compliance_score,
                "risk_budget_usage": risk_budget_usage,
                "violation_rate": 0.0,
                "top_violations": [],
                "limits": {
                    "max_leverage": float(self.max_leverage),
                    "max_total_risk": float(self.max_total_risk),
                    "max_position_risk": float(self.max_position_risk),
                },
                "ts": datetime.datetime.now().isoformat(),
            },
            "compliance_status": {
                "compliance_score": compliance_score,
                "risk_budget_usage": risk_budget_usage,
                "validation_results": {
                    "total_orders": 0,
                    "processing_time_ms": 0.0,
                    "violations": [],
                },
                "risk_assessment": {
                    "total_positions": 0,
                    "current_exposure": float(self.total_exposure),
                    "current_leverage": float(self.current_leverage),
                    "leverage_usage": 0.0,
                    "exposure_usage": 0.0,
                    "daily_trades_usage": 0.0,
                    "violation_rate": 0.0,
                    "per_instrument_exposure": dict(self.exposure_by_instrument),
                },
                "compliance_metrics": {
                    "approval_rate": 1.0,
                    "rejection_rate": 0.0,
                    "avg_approval_rate": float(
                        np.mean(self.approval_rate_history)
                    )
                    if self.approval_rate_history
                    else 1.0,
                    "total_validations": int(
                        self.validation_stats.get("total_validations", 0)
                    ),
                    "total_violations": int(
                        sum(self.validation_stats.get("violations", {}).values())
                    )
                    if isinstance(self.validation_stats.get("violations"), dict)
                    else 0,
                    "violation_breakdown": {
                        str(k): int(v)
                        for k, v in dict(
                            self.validation_stats.get("violations", {})
                        ).items()
                    },
                    "daily_trade_utilization": 0.0,
                },
                "current_limits": {
                    "max_leverage": float(self.max_leverage),
                    "max_position_risk": float(self.max_position_risk),
                    "max_total_risk": float(self.max_total_risk),
                    "max_daily_trades": int(self.max_daily_trades),
                    "min_trade_size": float(self.min_trade_size),
                    "max_trade_size": float(self.max_trade_size),
                    "max_drawdown": float(self._cfg.max_drawdown),
                },
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "validation_results": {
                "approved_orders": [],
                "rejected_orders": [],
                "approval_rate": 1.0,
            },
            "risk_limits": {
                "max_leverage": float(self.max_leverage),
                "max_position_risk": float(self.max_position_risk),
                "max_total_risk": float(self.max_total_risk),
                "max_drawdown": float(self._cfg.max_drawdown),
                "current_leverage": float(self.current_leverage),
                "risk_budget_usage": float(self.risk_budget_usage),
            },
            "_thesis": thesis,
            "success": True,
        }

    def _handle_disabled_fallback(self) -> Dict[str, Any]:
        thesis = "Compliance Module is disabled"
        return self._fallback_payload(thesis)

    def _handle_compliance_error(
        self, error: Exception, processing_time_sec: float
    ) -> Dict[str, Any]:
        """Handle unexpected errors in a contract-safe manner."""
        # circuit breaker update
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()
        if self.circuit_breaker["failures"] >= int(
            self._cfg.circuit_breaker_threshold
        ):
            self.circuit_breaker["state"] = "OPEN"
            self._health_status = "warning"

        explanation = self.english_explainer.explain_error(
            "ComplianceModule", str(error), "compliance validation"
        )
        self.logger.error(
            format_operator_message(
                message="Compliance module error",
                icon="[CRASH]",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time_sec * 1000.0,
                circuit_breaker_state=self.circuit_breaker["state"],
            )
        )
        self._record_failure(error)

        # Keep state minimally pessimistic
        self.compliance_score = float(max(0.5, float(self.compliance_score)))
        return self._fallback_payload(
            thesis=f"Compliance error fallback: {str(error)}"
        )

    # ── bookkeeping ──────────────────────────────────────────
    def _record_success(self, processing_time_sec: float) -> None:
        try:
            self._processing_times.append(float(processing_time_sec))
            # On successful cycles, ease circuit breaker a bit
            if self.circuit_breaker["state"] == "CLOSED":
                self.circuit_breaker["failures"] = max(
                    0, self.circuit_breaker["failures"] - 1
                )
        except Exception:
            pass

    def _record_failure(self, error: Exception) -> None:
        try:
            self.rejection_history.append(
                {"ts": time.time(), "error": str(error)}
            )
        except Exception:
            pass

    # ── state & health API ───────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        """Get complete module state for hot-reload."""
        return {
            "daily_trade_count": int(self.daily_trade_count),
            "last_trade_date": self.last_trade_date.isoformat()
            if self.last_trade_date
            else None,
            "total_exposure": float(self.total_exposure),
            "current_leverage": float(self.current_leverage),
            "risk_budget_usage": float(self.risk_budget_usage),
            "compliance_score": float(self.compliance_score),
            "validation_stats": {
                "total_validations": int(
                    self.validation_stats.get("total_validations", 0)
                ),
                "approved": int(self.validation_stats.get("approved", 0)),
                "rejected": int(self.validation_stats.get("rejected", 0)),
                "violations": {
                    str(k): int(v)
                    for k, v in dict(
                        self.validation_stats.get("violations", {})
                    ).items()
                },
            },
            "config": dict(self.config),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state for hot-reload."""
        self.daily_trade_count = int(state.get("daily_trade_count", 0))
        last_date_str = state.get("last_trade_date")
        self.last_trade_date = (
            datetime.datetime.fromisoformat(last_date_str).date()
            if last_date_str
            else None
        )
        self.total_exposure = float(state.get("total_exposure", 0.0))
        self.current_leverage = float(state.get("current_leverage", 0.0))
        self.risk_budget_usage = float(state.get("risk_budget_usage", 0.0))
        self.compliance_score = float(state.get("compliance_score", 1.0))

        # validation stats
        vs = state.get("validation_stats", {})
        self.validation_stats["total_validations"] = int(
            vs.get("total_validations", 0)
        )
        self.validation_stats["approved"] = int(vs.get("approved", 0))
        self.validation_stats["rejected"] = int(vs.get("rejected", 0))
        self.validation_stats["violations"].clear()
        for k, v in dict(vs.get("violations", {})).items():
            self.validation_stats["violations"][str(k)] = int(v)

        # config passthrough (non-critical)
        self.config.update(dict(state.get("config", {})))

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring."""
        try:
            approval_rate = self.validation_stats["approved"] / max(
                self.validation_stats["total_validations"], 1
            )
        except Exception:
            approval_rate = 1.0
        return {
            "compliance_score": float(self.compliance_score),
            "risk_budget_usage": float(self.risk_budget_usage),
            "approval_rate": float(approval_rate),
            "daily_trade_count": int(self.daily_trade_count),
            "current_leverage": float(self.current_leverage),
            "total_violations": int(
                sum(self.validation_stats["violations"].values())
            ),
            "enabled": bool(self.enabled),
        }

    # ── Trading mixin contract (no-op implementations) ───────
    async def propose_action(self, **inputs: Any) -> Dict[str, Any]:
        """Compliance module doesn't place trades; return a well-formed no-op action."""
        return {
            "action": "noop",
            "module": "ComplianceModule",
            "reason": "Compliance module does not propose trades",
            "enabled": bool(self.enabled),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    async def calculate_confidence(
        self, action: Dict[str, Any], **inputs: Any
    ) -> float:
        """Return a deterministic confidence score in [0,1]."""
        try:
            score = float(self.compliance_score)
        except Exception:
            score = 1.0
        # clip to [0,1]
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        return score

    # ── input hardening ──────────────────────────────────────
    def _coerce_bus_inputs(
        self,
        market_context: Any,
        positions: Any,
        pending_orders: Any,
        balance: Any,
    ) -> Tuple[
        Dict[str, Any],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        float,
    ]:
        """Coerce bus inputs to safe types per contract."""
        mc = market_context if isinstance(market_context, dict) else {}
        pos = positions if isinstance(positions, list) else []
        po = pending_orders if isinstance(pending_orders, list) else []
        try:
            bal = float(balance)
        except Exception:
            bal = 10000.0
        return mc, pos, po, bal
