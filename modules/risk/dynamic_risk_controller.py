

import asyncio
import datetime
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Union

import numpy as np

from modules.contracts import module_args
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.core.mixins import (
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
    SmartInfoBusTradingMixin,
)
from modules.core.module_base import BaseModule, module
from modules.monitoring.performance_tracker import PerformanceTracker
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import InfoBusManager
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities


class RiskControlMode(Enum):
    INITIALIZATION = "initialization"
    CALIBRATION = "calibration"
    NORMAL = "normal"
    PROTECTIVE = "protective"
    AGGRESSIVE_REDUCTION = "aggressive_reduction"
    EMERGENCY = "emergency"
    RECOVERY = "recovery"


def _load_dynamic_risk_config_from_yaml() -> Dict[str, Any]:
    import os

    import yaml
    defaults: Dict[str, Any] = {}
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "risk_policy.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}

            limits = policy.get("limits", {})
            modules_cfg = policy.get("modules", {}).get("DynamicRiskController", {})
            escalation = policy.get("escalation", {})


            defaults["dd_threshold"] = float(limits.get("max_drawdown", 0.085))
            defaults["base_risk_scale"] = float(modules_cfg.get("base_risk_scale", 1.0))
            defaults["emergency_scaling"] = float(modules_cfg.get("emergency_scaling", 0.3))
            defaults["recovery_multiplier"] = float(modules_cfg.get("recovery_multiplier", 1.2))
    except Exception:

        pass
    return defaults


@dataclass
class DynamicRiskConfig:

    base_risk_scale: float = 1.0
    min_risk_scale: float = 0.1
    max_risk_scale: float = 1.5


    vol_history_len: int = 30
    dd_threshold: float = 0.085
    vol_ratio_threshold: float = 2.0


    recovery_speed: float = 0.15
    risk_decay: float = 0.95
    adaptive_scaling: bool = True
    regime_sensitivity: float = 1.0
    correlation_sensitivity: float = 0.8


    emergency_scaling: float = 0.3
    recovery_multiplier: float = 1.2


    max_processing_time_ms: float = 100
    circuit_breaker_threshold: int = 5
    min_risk_quality: float = 0.3


    circuit_breaker_cooldown_sec: int = 300
    health_check_interval_sec: int = 30


    adaptive_learning_rate: float = 0.02
    risk_adaptation_speed: float = 1.0

    def __post_init__(self):
        yaml_config = _load_dynamic_risk_config_from_yaml()
        for key, value in yaml_config.items():
            if hasattr(self, key):
                setattr(self, key, value)


@module(**module_args(
    "DynamicRiskController",
    description="Advanced dynamic risk scaling with intelligent adaptation and comprehensive market analysis (voting member)",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,

    is_voting_member=True,
))
class DynamicRiskController(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):

    def __init__(
        self,
        config: Optional[Union[DynamicRiskConfig, Dict[str, Any]]] = None,
        action_dim: int = 1,
        adaptive_scaling: bool = True,
        regime_aware: bool = True,
        **kwargs
    ):

        if config is None:
            self._cfg = DynamicRiskConfig()
        elif isinstance(config, dict):
            self._cfg = DynamicRiskConfig(**config)
        else:
            self._cfg = config

        self.action_dim = int(action_dim)
        self.adaptive_scaling = adaptive_scaling
        self.regime_aware = regime_aware


        try:
            self.current_mode = RiskControlMode.INITIALIZATION
            self.current_risk_scale = float(self._cfg.base_risk_scale)
            self._risk_quality = 0.5

            self.market_regime = "normal"

            self.smart_bus = InfoBusManager.get_instance()

            self.debug: bool = bool(getattr(self, "debug", False))
        except Exception:

            if not hasattr(self, "debug"):
                self.debug = False
            if not hasattr(self, "market_regime"):
                self.market_regime = "normal"


        original_cfg = self._cfg
        super().__init__()

        self._cfg = original_cfg


        self._initialize_advanced_systems()
        self._initialize_risk_control_state()

        self.logger.info(
            format_operator_message(
                message="Enhanced dynamic risk controller ready",
                icon="⚙️",
                adaptive=adaptive_scaling,
                regime_aware=regime_aware,
                config_loaded=True,
            )
        )

    def _initialize_advanced_systems(self):
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="DynamicRiskController",
            log_path="logs/risk/dynamic_risk_controller.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("DynamicRiskController", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()


        self.circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",
            "threshold": self._cfg.circuit_breaker_threshold,
            "cooldown_sec": float(self._cfg.circuit_breaker_cooldown_sec),
        }


        self._health_status = "healthy"
        self._last_health_check = time.time()


    def _initialize_risk_control_state(self):

        self._initialize_risk_state()
        self._initialize_trading_state()
        self._initialize_state_management()


        self.current_mode = RiskControlMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()

        self._stop_event = threading.Event()


        self.current_risk_scale = float(self._cfg.base_risk_scale)
        self.risk_factors: Dict[str, float] = {
            "drawdown": 1.0,
            "volatility": 1.0,
            "correlation": 1.0,
            "losing_streak": 1.0,
            "market_stress": 1.0,
            "liquidity": 1.0,
            "news_sentiment": 1.0,
            "execution_quality": 1.0,
            "portfolio_concentration": 1.0,
        }


        self.vol_history: Deque[float] = deque(maxlen=self._cfg.vol_history_len)
        self.dd_history: Deque[float] = deque(maxlen=50)
        self.risk_scale_history: Deque[float] = deque(maxlen=100)
        self.consecutive_losses = 0
        self.last_pnl = 0.0


        self.market_regime = "normal"
        self.market_regime_history: Deque[Dict[str, Any]] = deque(maxlen=20)
        self.volatility_regime = "medium"
        self.market_session = "unknown"


        self.risk_events: Deque[Dict[str, Any]] = deque(maxlen=100)
        self.risk_adjustments_made = 0
        self.emergency_interventions = 0


        self.risk_analytics: Dict[str, Any] = defaultdict(list)
        self.regime_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(list))
        self._last_significant_change = 0


        self.external_risk_scale = 1.0
        self.external_signals: Dict[str, float] = {}


        self._adaptive_params: Dict[str, Any] = {
            "dynamic_penalty_scaling": 1.0,
            "regime_sensitivity_multiplier": 1.0,
            "volatility_tolerance": 1.0,
            "risk_adaptation_confidence": 0.5,
            "learning_momentum": 0.0,
            "emergency_threshold_adaptation": 1.0,
        }


        self._risk_quality = 0.5
        self._risk_effectiveness_history: Deque[float] = deque(maxlen=50)


        self._start_monitoring()

    def _start_monitoring(self):

        def monitoring_loop():
            while getattr(self, "_monitoring_active", True):
                try:
                    self._update_risk_health()
                    self._analyze_risk_effectiveness()
                    self._adapt_risk_parameters()
                    self._stop_event.wait(max(1, int(self._cfg.health_check_interval_sec)))
                except Exception as e:
                    self.logger.error(f"Risk control monitoring error: {e}")
                    self._stop_event.wait(2)

        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    def _initialize(self) -> None:
        try:

            initial_status = {
                "current_mode": self.current_mode.value,
                "current_risk_scale": self.current_risk_scale,
                "base_risk_scale": self._cfg.base_risk_scale,
                "adaptive_scaling": self.adaptive_scaling,
                "regime_aware": self.regime_aware,
            }

            self.smart_bus.set(
                "risk_scaling",
                initial_status,
                module="DynamicRiskController",
                thesis="Initial dynamic risk controller status",
            )


            self.smart_bus.set(
                "risk_level",
                "NORMAL",
                module="DynamicRiskController",
                thesis="Initial risk level: NORMAL",
            )

            self.smart_bus.set(
                "risk_scale",
                float(self.current_risk_scale),
                module="DynamicRiskController",
                thesis=f"Initial risk scale: {self.current_risk_scale:.2f}",
            )

            self.smart_bus.set(
                "risk_assessment",
                {
                    "risk_level": "NORMAL",
                    "risk_scale": float(self.current_risk_scale),
                    "risk_quality": float(self._risk_quality),
                    "market_regime": self.market_regime,
                    "mode": self.current_mode.value,
                    "adaptive_scaling": self.adaptive_scaling,
                    "consecutive_losses": 0,
                    "emergency_active": False,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                module="DynamicRiskController",
                thesis="Initial risk assessment for API consumption",
            )


            try:
                baseline_vote = {
                    "member": "DynamicRiskController",
                    "type": "risk_posture",
                    "posture": "maintain",
                    "target_scale": float(self.current_risk_scale),
                    "bounds": [float(self._cfg.min_risk_scale), float(self._cfg.max_risk_scale)],
                    "rationale": "baseline initialization",
                    "confidence": 0.1,
                    "timestamp": time.time(),
                }
                self.smart_bus.set(
                    "DynamicRiskController_voting_proposal",
                    baseline_vote,
                    module="DynamicRiskController",
                    thesis="Baseline voting proposal during initialization",
                )
                self.smart_bus.set(
                    "DynamicRiskController_confidence",
                    0.1,
                    module="DynamicRiskController",
                    thesis="Baseline voting confidence during initialization",
                )
            except Exception:

                pass

        except Exception as e:
            self.logger.error(f"Risk controller initialization failed: {e}")

    async def calculate_confidence(self, action: Dict[str, Any], **kwargs) -> float:
        try:

            confidence = 0.9


            factors: Dict[str, Any] = {}


            factors["risk_quality"] = self._risk_quality
            confidence *= self._risk_quality


            if self.circuit_breaker["state"] == "OPEN":
                factors["circuit_breaker_penalty"] = 0.3
                confidence *= 0.3
            else:
                factors["circuit_breaker_penalty"] = 1.0


            regime_confidence_map = {
                "normal": 1.0,
                "trending": 0.95,
                "volatile": 0.8,
                "ranging": 0.9,
                "unknown": 0.7,
            }
            regime_factor = regime_confidence_map.get(self.market_regime, 0.7)
            factors["regime_factor"] = regime_factor
            confidence *= regime_factor


            data_availability = min(len(self.vol_history) / 10.0, 1.0)
            factors["data_availability"] = data_availability
            confidence *= data_availability


            if self._risk_effectiveness_history:
                recent_values = list(self._risk_effectiveness_history)[-5:]
                recent_effectiveness = sum(recent_values) / float(len(recent_values))
                factors["recent_effectiveness"] = recent_effectiveness
                confidence *= recent_effectiveness


            if len(self.risk_scale_history) >= 5:
                recent_scales = list(self.risk_scale_history)[-5:]
                scale_volatility = np.std(recent_scales) if len(recent_scales) > 1 else 0.0
                stability_factor = max(0.5, float(1.0 - scale_volatility * 2))
                factors["stability_factor"] = stability_factor
                confidence *= stability_factor


            if self.current_mode == RiskControlMode.EMERGENCY:
                factors["emergency_penalty"] = 0.6
                confidence *= 0.6


            confidence = max(0.0, min(1.0, confidence))

            self.logger.debug(f"Risk controller confidence: {confidence:.3f}, factors: {factors}")
            return float(confidence)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "confidence_calculation")
            self.logger.error(f"Confidence calculation failed: {error_context}")
            return 0.5

    async def propose_action(self, **kwargs) -> Dict[str, Any]:
        try:
            action_proposal: Dict[str, Any] = {
                "action_type": "risk_scaling",
                "timestamp": time.time(),
                "current_risk_scale": self.current_risk_scale,
                "current_mode": self.current_mode.value,
                "market_regime": self.market_regime,
                "recommendations": [],
                "warnings": [],
                "adjustments": {},
            }


            if self.current_mode == RiskControlMode.EMERGENCY:
                action_proposal["recommendations"].append(
                    {
                        "type": "emergency_risk_reduction",
                        "reason": "Emergency mode active",
                        "suggested_action": "Maintain minimum risk exposure until conditions improve",
                        "priority": "critical",
                    }
                )

                action_proposal["adjustments"]["emergency_mode"] = True
                action_proposal["adjustments"]["risk_scale"] = self._cfg.min_risk_scale

            elif self.current_mode == RiskControlMode.AGGRESSIVE_REDUCTION:
                action_proposal["recommendations"].append(
                    {
                        "type": "aggressive_risk_reduction",
                        "reason": "Multiple risk factors elevated",
                        "suggested_action": "Significantly reduce position sizes and avoid new entries",
                        "priority": "high",
                    }
                )

                action_proposal["adjustments"]["position_reduction"] = 0.3

            elif self.current_mode == RiskControlMode.PROTECTIVE:
                action_proposal["recommendations"].append(
                    {
                        "type": "protective_measures",
                        "reason": "Risk factors showing warning signals",
                        "suggested_action": "Reduce risk exposure and tighten risk management",
                        "priority": "medium",
                    }
                )

                action_proposal["adjustments"]["position_reduction"] = 0.7
                action_proposal["adjustments"]["tighter_stops"] = True


            critical_factors = [name for name, value in self.risk_factors.items() if value < 0.5]
            if critical_factors:
                action_proposal["warnings"].append(
                    {"type": "critical_risk_factors", "factors": critical_factors, "risk_level": "high"}
                )

                action_proposal["recommendations"].append(
                    {
                        "type": "factor_based_adjustment",
                        "reason": f'Critical risk factors detected: {", ".join(critical_factors)}',
                        "suggested_action": "Address specific risk factor causes",
                        "priority": "high",
                    }
                )


            if self.volatility_regime == "extreme":
                action_proposal["recommendations"].append(
                    {
                        "type": "volatility_adjustment",
                        "reason": "Extreme volatility detected",
                        "suggested_action": "Reduce position sizes and increase monitoring frequency",
                        "priority": "high",
                    }
                )


            if self.consecutive_losses > 5:
                action_proposal["warnings"].append(
                    {"type": "losing_streak", "consecutive_losses": self.consecutive_losses, "risk_level": "medium"}
                )

                action_proposal["recommendations"].append(
                    {
                        "type": "streak_management",
                        "reason": f"{self.consecutive_losses} consecutive losses detected",
                        "suggested_action": "Consider trading break or strategy review",
                        "priority": "medium",
                    }
                )


            if self.circuit_breaker["state"] == "OPEN":
                action_proposal["warnings"].append(
                    {
                        "type": "circuit_breaker_open",
                        "failures": self.circuit_breaker["failures"],
                        "risk_level": "critical",
                    }
                )

                action_proposal["recommendations"].append(
                    {
                        "type": "system_recovery",
                        "reason": "Circuit breaker triggered",
                        "suggested_action": "System recovery mode - minimal risk until stabilized",
                        "priority": "critical",
                    }
                )


            if self.external_signals:
                low_signals = [name for name, value in self.external_signals.items() if value < 0.5]
                if low_signals:
                    action_proposal["recommendations"].append(
                        {
                            "type": "external_risk_signals",
                            "reason": f'External risk signals warning: {", ".join(low_signals)}',
                            "suggested_action": "Consider external risk factor implications",
                            "priority": "medium",
                        }
                    )


            if self.current_risk_scale < 0.3:
                action_proposal["adjustments"]["recovery_readiness"] = True
            elif self.current_risk_scale > 1.2:
                action_proposal["adjustments"]["risk_monitoring"] = "increased"

            self.logger.debug(
                f"Risk action proposed: {len(action_proposal['recommendations'])} recommendations, "
                f"{len(action_proposal['warnings'])} warnings"
            )

            return action_proposal

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "action_proposal")
            self.logger.error(f"Action proposal failed: {error_context}")
            return {
                "action_type": "risk_scaling",
                "timestamp": time.time(),
                "error": str(e),
                "recommendations": [],
                "warnings": [],
                "adjustments": {},
            }

    async def process(self, **inputs) -> Dict[str, Any]:
        start_time = time.time()
        try:

            if self.circuit_breaker["state"] == "OPEN":
                since = time.time() - float(self.circuit_breaker["last_failure"] or 0.0)
                if since < float(self.circuit_breaker["cooldown_sec"]):
                    thesis = "Circuit breaker OPEN - conservative risk posture maintained."
                    fallback = await self._handle_no_data_fallback()
                    fallback["_thesis"] = thesis

                    vote_payload = await self.vote()
                    fallback["DynamicRiskController_voting_proposal"] = vote_payload
                    fallback["DynamicRiskController_confidence"] = float(vote_payload.get("confidence", 0.0))
                    await self._update_risk_smart_bus(fallback, thesis)
                    return fallback
                else:

                    self.circuit_breaker["state"] = "CLOSED"
                    self.circuit_breaker["failures"] = 0


            risk_data = await self._extract_risk_data(**inputs)
            if not risk_data:
                fallback = await self._handle_no_data_fallback()

                vote_payload = await self.vote()
                fallback["DynamicRiskController_voting_proposal"] = vote_payload
                fallback["DynamicRiskController_confidence"] = float(vote_payload.get("confidence", 0.0))
                await self._update_risk_smart_bus(fallback, fallback.get("_thesis", "No data fallback"))
                return fallback


            pos = risk_data.get("position_data", {}) or {}
            pos_count = int(pos.get("count", len(pos.get("positions", [])) if isinstance(pos, dict) else 0))
            low_drawdown = float(risk_data.get("drawdown", 0.0)) <= 0.005
            low_vol = float(risk_data.get("volatility", 0.01)) <= 0.01
            if pos_count == 0 and low_drawdown and low_vol:
                thesis = "Risk stable (no positions, low drawdown/volatility); fast-path applied"
                result: Dict[str, Any] = {
                    "success": True,
                    "risk_alerts": [],
                    "risk_analytics": {},
                    "risk_factors": self.risk_factors.copy(),
                    "risk_scaling": {
                        "current_mode": self.current_mode.value,
                        "current_risk_scale": float(self.current_risk_scale),
                        "base_risk_scale": float(self._cfg.base_risk_scale),
                        "timestamp": datetime.datetime.now().isoformat(),
                    },
                    "risk_level": "NORMAL",
                    "risk_scale": float(self.current_risk_scale),
                    "risk_assessment": {
                        "risk_level": "NORMAL",
                        "risk_scale": float(self.current_risk_scale),
                        "risk_quality": float(self._risk_quality),
                        "market_regime": self.market_regime,
                        "mode": self.current_mode.value,
                        "adaptive_scaling": self.adaptive_scaling,
                        "consecutive_losses": self.consecutive_losses,
                        "emergency_active": False,
                        "timestamp": datetime.datetime.now().isoformat(),
                    },
                    "DynamicRiskController_voting_proposal": await self.vote(),
                    "DynamicRiskController_confidence": 0.5,
                    "_thesis": thesis,
                }
                await self._update_risk_smart_bus(result, thesis)

                processing_time = (time.time() - start_time) * 1000.0
                self._record_success(processing_time)
                return result


            context_result = await self._update_market_context_async(risk_data)


            external_result = await self._update_external_integrations_async(risk_data)


            adjustment_result = await self._adjust_risk_comprehensive_async(risk_data)


            adaptive_result: Dict[str, Any] = {}
            if self.adaptive_scaling:
                adaptive_result = await self._apply_adaptive_scaling_async(risk_data)

                if adaptive_result.get("adaptive_scaling_applied", False):
                    final_adaptation = float(adaptive_result.get("final_adaptation", 1.0))
                    self.current_risk_scale = float(
                        np.clip(
                            self.current_risk_scale * final_adaptation,
                            self._cfg.min_risk_scale,
                            self._cfg.max_risk_scale,
                        )
                    )


            emergency_result = await self._apply_emergency_interventions_async(risk_data)


            final_result = await self._calculate_final_risk_scale_async()


            mode_result = await self._update_operational_mode_async(risk_data)


            result: Dict[str, Any] = {
                **context_result,
                **external_result,
                **adjustment_result,
                **adaptive_result,
                **emergency_result,
                **final_result,
                **mode_result,
            }


            thesis = await self._generate_comprehensive_risk_thesis(risk_data, result)


            scaling_data = {
                "current_mode": self.current_mode.value,
                "current_risk_scale": self.current_risk_scale,
                "base_risk_scale": self._cfg.base_risk_scale,
                "min_risk_scale": self._cfg.min_risk_scale,
                "max_risk_scale": self._cfg.max_risk_scale,
                "adaptive_scaling": self.adaptive_scaling,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            factors_data = {
                "risk_factors": self.risk_factors.copy(),
                "external_risk_scale": self.external_risk_scale,
                "external_signals": self.external_signals.copy(),
                "consecutive_losses": self.consecutive_losses,
                "risk_adjustments_made": self.risk_adjustments_made,
                "emergency_interventions": self.emergency_interventions,
            }

            analytics_data = {
                "risk_quality": self._risk_quality,
                "adaptive_params": self._adaptive_params.copy(),
                "risk_events": len(self.risk_events),
                "scale_history_size": len(self.risk_scale_history),
                "volatility_history_size": len(self.vol_history),
            }

            alerts_data = {
                "emergency_interventions": self.emergency_interventions,
                "risk_adjustments_made": self.risk_adjustments_made,
                "critical_mode": self.current_mode
                in [RiskControlMode.EMERGENCY, RiskControlMode.AGGRESSIVE_REDUCTION],
                "low_risk_quality": self._risk_quality < self._cfg.min_risk_quality,
            }


            risk_level_str = "NORMAL"
            if self.current_mode == RiskControlMode.EMERGENCY:
                risk_level_str = "CRITICAL"
            elif self.current_mode == RiskControlMode.AGGRESSIVE_REDUCTION:
                risk_level_str = "HIGH"
            elif self.current_mode == RiskControlMode.PROTECTIVE:
                risk_level_str = "ELEVATED"
            elif self.current_mode == RiskControlMode.RECOVERY:
                risk_level_str = "LOW"


            risk_assessment = {
                "risk_level": risk_level_str,
                "risk_scale": float(self.current_risk_scale),
                "risk_quality": float(self._risk_quality),
                "market_regime": self.market_regime,
                "mode": self.current_mode.value,
                "adaptive_scaling": self.adaptive_scaling,
                "consecutive_losses": self.consecutive_losses,
                "emergency_active": self.current_mode == RiskControlMode.EMERGENCY,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            result.update(
                {
                    "success": True,
                    "risk_scaling": scaling_data,
                    "risk_factors": factors_data,
                    "risk_analytics": analytics_data,
                    "risk_alerts": alerts_data,
                    "risk_level": risk_level_str,
                    "risk_scale": float(self.current_risk_scale),
                    "risk_assessment": risk_assessment,
                    "_thesis": thesis,
                }
            )


            vote_payload = await self.vote(risk_scale=self.current_risk_scale)
            result["DynamicRiskController_voting_proposal"] = vote_payload
            result["DynamicRiskController_confidence"] = float(vote_payload.get("confidence", 0.0))


            await self._update_risk_smart_bus(result, thesis)


            processing_time = (time.time() - start_time) * 1000.0
            self._record_success(processing_time)

            return result

        except Exception as e:
            return await self._handle_risk_error(e, start_time)

    async def _extract_risk_data(self, **inputs) -> Optional[Dict[str, Any]]:
        try:

            risk_data_bus = self.smart_bus.get("risk_data", "DynamicRiskController") or {}
            performance_data = self.smart_bus.get("performance_data", "DynamicRiskController") or {}
            market_data = self.smart_bus.get("market_data", "DynamicRiskController") or {}


            position_data = self.smart_bus.get("position_data", "DynamicRiskController") or {}
            if not position_data:
                snap = (
                    self.smart_bus.get("positions", "DynamicRiskController")
                    or self.smart_bus.get("current_positions", "DynamicRiskController")
                    or {}
                )
                if isinstance(snap, dict) and snap:
                    positions = []
                    for inst, p in snap.items():
                        notional = float(p.get("notional_eur", 0.0) or 0.0)
                        units = float(p.get("units", 0.0) or 0.0)
                        entry_price = float(p.get("entry_price", 0.0) or 0.0)
                        size = abs(notional) if abs(notional) > 0 else (
                            abs(units * entry_price) if (units and entry_price) else abs(units)
                        )
                        entry: Dict[str, Any] = {"instrument": inst, "size": float(size)}

                        for k, v in p.items():
                            if k != "instrument":
                                entry[k] = v
                        positions.append(entry)
                    position_data = {"positions": positions}


            drawdown = inputs.get("drawdown", inputs.get("current_drawdown", 0.0))
            volatility = inputs.get("volatility", 0.01)
            pnl = inputs.get("pnl", 0.0)
            balance = inputs.get("balance", inputs.get("current_balance", 0.0))


            risk_snapshot = risk_data_bus.get("risk_snapshot", {}) if isinstance(risk_data_bus, dict) else {}
            if not drawdown and "current_drawdown" in risk_snapshot:
                drawdown = risk_snapshot["current_drawdown"]
            if not balance and "balance" in risk_snapshot:
                balance = risk_snapshot["balance"]


            correlation = inputs.get("correlation", 0.0)
            if "correlation_risk" in risk_snapshot:
                correlation = risk_snapshot["correlation_risk"]

            return {
                "drawdown": float(drawdown or 0.0),
                "volatility": float(volatility or 0.01),
                "pnl": float(pnl or 0.0),
                "balance": float(balance or 0.0),
                "correlation": float(correlation or 0.0),
                "risk_data": risk_data_bus,
                "performance_data": performance_data,
                "market_data": market_data,
                "position_data": position_data,
                "timestamp": datetime.datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to extract risk data: {e}")
            return None

    async def _update_market_context_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        try:

            market_context = self.smart_bus.get("market_context", "DynamicRiskController") or {}

            top_level_regime = self.smart_bus.get("market_regime", "DynamicRiskController")


            old_regime = self.market_regime
            proposed_regime = market_context.get("regime")
            if not proposed_regime or proposed_regime == "unknown":
                if isinstance(top_level_regime, str) and top_level_regime:
                    proposed_regime = top_level_regime
                else:
                    proposed_regime = "unknown"
            self.market_regime = proposed_regime


            self.volatility_regime = market_context.get(
                "volatility_level", market_context.get("volatility_regime", "medium")
            )

            self.market_session = market_context.get("session", market_context.get("session_type", "unknown"))


            if self.market_regime != old_regime:
                self.market_regime_history.append(
                    {
                        "regime": self.market_regime,
                        "timestamp": risk_data.get("timestamp", datetime.datetime.now().isoformat()),
                        "old_regime": old_regime,
                    }
                )

                self.logger.info(
                    format_operator_message(
                        message="Market regime changed",
                        icon="[STATS]",
                        old_regime=old_regime,
                        new_regime=self.market_regime,
                        volatility=self.volatility_regime,
                        session=self.market_session,
                    )
                )


                if self.regime_aware:
                    await self._update_regime_risk_factors_async()

            return {
                "market_context_updated": True,
                "regime_change": old_regime != self.market_regime,
                "current_regime": self.market_regime,
                "volatility_regime": self.volatility_regime,
                "market_session": self.market_session,
            }

        except Exception as e:
            self.logger.error(f"Market context update failed: {e}")
            return {"market_context_updated": False, "error": str(e)}

    async def _update_regime_risk_factors_async(self) -> None:
        try:

            regime_adjustments = {
                "trending": {"market_stress": 0.9, "volatility": 1.1},
                "volatile": {"market_stress": 0.7, "volatility": 0.8},
                "ranging": {"market_stress": 1.1, "volatility": 1.2},
                "unknown": {"market_stress": 1.0, "volatility": 1.0},
            }


            vol_adjustments = {
                "low": {"volatility": 1.2, "market_stress": 1.1},
                "medium": {"volatility": 1.0, "market_stress": 1.0},
                "high": {"volatility": 0.8, "market_stress": 0.8},
                "extreme": {"volatility": 0.6, "market_stress": 0.6},
            }


            if self.market_regime in regime_adjustments:
                for factor, multiplier in regime_adjustments[self.market_regime].items():
                    if factor in self.risk_factors:
                        sensitivity = self._adaptive_params["regime_sensitivity_multiplier"]
                        self.risk_factors[factor] *= multiplier * self._cfg.regime_sensitivity * sensitivity


            if self.volatility_regime in vol_adjustments:
                for factor, multiplier in vol_adjustments[self.volatility_regime].items():
                    if factor in self.risk_factors:
                        self.risk_factors[factor] *= multiplier

        except Exception as e:
            self.logger.warning(f"Regime risk factor update failed: {e}")

    async def _update_external_integrations_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        try:

            portfolio_risk_data = self.smart_bus.get("portfolio_risk", "DynamicRiskController") or {}
            execution_quality_data = self.smart_bus.get("execution_quality", "DynamicRiskController") or {}
            anomaly_data = self.smart_bus.get("anomaly_detector", "DynamicRiskController") or {}

            external_signals: Dict[str, float] = {}


            if "risk_adjustment" in portfolio_risk_data:
                external_signals["portfolio_risk"] = float(portfolio_risk_data["risk_adjustment"])


            if "quality_score" in execution_quality_data:
                quality_score = float(execution_quality_data["quality_score"])
                external_signals["execution_quality"] = quality_score
                self.risk_factors["execution_quality"] = quality_score


            if "anomaly_score" in anomaly_data:
                anomaly_score = float(anomaly_data["anomaly_score"])
                external_signals["anomaly_risk"] = 1.0 - anomaly_score


            compliance_data = self.smart_bus.get("compliance", "DynamicRiskController") or {}
            if "risk_budget_used" in compliance_data:
                compliance_factor = 1.0 - float(compliance_data["risk_budget_used"])
                external_signals["compliance"] = compliance_factor


            drawdown_risk_data = self.smart_bus.get("drawdown_risk", "DynamicRiskController") or {}
            rescue_status_data = self.smart_bus.get("rescue_status", "DynamicRiskController") or {}

            if drawdown_risk_data or rescue_status_data:

                dd_rescue_adjustment = float(drawdown_risk_data.get("risk_adjustment_factor",
                                            rescue_status_data.get("risk_adjustment_factor", 1.0)))
                external_signals["drawdown_rescue"] = dd_rescue_adjustment


                rescue_mode_active = bool(rescue_status_data.get("rescue_mode", False) or
                                         drawdown_risk_data.get("rescue_mode", False))
                if rescue_mode_active:

                    self.risk_factors["drawdown"] *= 0.6
                    external_signals["rescue_mode_penalty"] = 0.6

                    if self.debug:
                        self.logger.info(format_operator_message(
                            icon="🛟",
                            message="DrawdownRescue rescue mode active - applying risk reduction",
                            rescue_adjustment=f"{dd_rescue_adjustment:.2f}",
                            drawdown_factor=f"{self.risk_factors['drawdown']:.2f}"
                        ))


                dd_velocity = float(drawdown_risk_data.get("dd_velocity", 0.0))
                dd_acceleration = float(drawdown_risk_data.get("dd_acceleration", 0.0))


                if dd_velocity > 0.01:
                    velocity_penalty = max(0.5, 1.0 - dd_velocity * 5.0)
                    self.risk_factors["drawdown"] *= velocity_penalty
                    external_signals["dd_velocity_penalty"] = velocity_penalty

                    if self.debug:
                        self.logger.info(format_operator_message(
                            icon="📉",
                            message="DrawdownRescue velocity penalty applied",
                            velocity=f"{dd_velocity:.3f}",
                            penalty=f"{velocity_penalty:.2f}x"
                        ))


                if dd_acceleration > 0.005:
                    accel_penalty = max(0.7, 1.0 - dd_acceleration * 10.0)
                    self.risk_factors["drawdown"] *= accel_penalty
                    external_signals["dd_accel_penalty"] = accel_penalty


            correlation_risk_data = self.smart_bus.get("correlation_risk", "DynamicRiskController") or {}
            diversification_score = self.smart_bus.get("diversification_score", "DynamicRiskController")

            if correlation_risk_data:

                corr_risk_score = float(correlation_risk_data.get("correlation_risk_score", 0.0))

                corr_factor = max(0.3, 1.0 - corr_risk_score * 0.7)
                external_signals["correlation_risk"] = corr_factor
                self.risk_factors["correlation"] = corr_factor


                corr_severity = str(correlation_risk_data.get("severity_level", "normal"))
                if corr_severity == "critical":
                    self.risk_factors["correlation"] *= 0.6
                    external_signals["corr_severity_penalty"] = 0.6
                elif corr_severity == "warning":
                    self.risk_factors["correlation"] *= 0.8
                    external_signals["corr_severity_penalty"] = 0.8

                if self.debug and corr_risk_score > 0.3:
                    self.logger.info(format_operator_message(
                        icon="🔗",
                        message="CorrelatedRiskController integration",
                        corr_risk_score=f"{corr_risk_score:.2f}",
                        corr_factor=f"{corr_factor:.2f}",
                        severity=corr_severity
                    ))


            if diversification_score is not None:
                div_score = float(diversification_score)

                if div_score > 0.7:
                    div_boost = min(1.15, 1.0 + (div_score - 0.7) * 0.5)
                    self.risk_factors["portfolio_concentration"] *= div_boost
                    external_signals["diversification_boost"] = div_boost
                elif div_score < 0.3:
                    div_penalty = max(0.7, div_score / 0.3)
                    self.risk_factors["portfolio_concentration"] *= div_penalty
                    external_signals["diversification_penalty"] = div_penalty

                external_signals["diversification_score"] = div_score


            duration_risk_data = self.smart_bus.get("position_duration_risk", "DynamicRiskController") or {}
            duration_alerts = self.smart_bus.get("duration_alerts", "DynamicRiskController") or {}

            if duration_risk_data:

                duration_severity = str(duration_risk_data.get("severity_level", "normal"))
                severity_map = {
                    "normal": 1.0,
                    "info": 0.95,
                    "warning": 0.80,
                    "critical": 0.55,
                    "error": 0.40,
                }
                duration_factor = severity_map.get(duration_severity, 0.7)
                external_signals["position_duration"] = duration_factor


                duration_risk_score = float(duration_risk_data.get("risk_score", 0.0))
                if duration_risk_score > 0.5:

                    duration_penalty = max(0.6, 1.0 - duration_risk_score * 0.5)
                    external_signals["duration_risk_penalty"] = duration_penalty

                    if self.debug:
                        self.logger.info(format_operator_message(
                            icon="⏱️",
                            message="ActiveTradeMonitor duration risk penalty",
                            severity=duration_severity,
                            risk_score=f"{duration_risk_score:.2f}",
                            penalty=f"{duration_penalty:.2f}x"
                        ))


            if duration_alerts:
                critical_alerts = duration_alerts.get("critical", [])
                warning_alerts = duration_alerts.get("warning", [])


                if critical_alerts:
                    alert_penalty = max(0.5, 1.0 - len(critical_alerts) * 0.15)
                    external_signals["duration_critical_alerts"] = alert_penalty

                    if self.debug:
                        self.logger.info(format_operator_message(
                            icon="🚨",
                            message="ActiveTradeMonitor critical duration alerts",
                            critical_count=len(critical_alerts),
                            penalty=f"{alert_penalty:.2f}x"
                        ))
                elif warning_alerts:
                    alert_penalty = max(0.7, 1.0 - len(warning_alerts) * 0.08)
                    external_signals["duration_warning_alerts"] = alert_penalty


            self.external_signals = external_signals


            if external_signals:

                primary_signals = ['portfolio_risk', 'execution_quality', 'drawdown_rescue',
                                   'correlation_risk', 'position_duration']
                primary_values = [v for k, v in external_signals.items() if k in primary_signals]


                penalty_signals = [v for k, v in external_signals.items()
                                   if 'penalty' in k or 'boost' in k]

                if primary_values:
                    primary_avg = float(np.mean(primary_values))
                else:
                    primary_avg = 1.0

                if penalty_signals:
                    penalty_product = float(np.prod(penalty_signals))
                else:
                    penalty_product = 1.0


                self.external_risk_scale = float(np.clip(primary_avg * penalty_product, 0.1, 1.5))
            else:
                self.external_risk_scale = 1.0

            return {
                "external_integrations_updated": True,
                "external_signals_count": len(external_signals),
                "external_risk_scale": self.external_risk_scale,
                "external_signals": external_signals.copy(),
                "modules_integrated": [
                    "PortfolioRiskSystem",
                    "ExecutionQualityMonitor",
                    "EnhancedAnomalyDetector",
                    "ComplianceModule",
                    "DrawdownRescue",
                    "CorrelatedRiskController",
                    "ActiveTradeMonitor"
                ],
            }

        except Exception as e:
            self.logger.warning(f"External integration update failed: {e}")
            return {"external_integrations_updated": False, "error": str(e)}

    async def _adjust_risk_comprehensive_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            old_scale = self.current_risk_scale


            factor_results: Dict[str, Any] = {}
            factor_results["drawdown"] = await self._update_drawdown_factor_async(risk_data.get("drawdown", 0.0))
            factor_results["volatility"] = await self._update_volatility_factor_async(risk_data.get("volatility", 0.01))
            factor_results["correlation"] = await self._update_correlation_factor_async(risk_data.get("correlation", 0.0))
            factor_results["losing_streak"] = await self._update_losing_streak_factor_async(risk_data.get("pnl", 0.0))
            factor_results["liquidity"] = await self._update_liquidity_factor_async(risk_data)
            factor_results["news_sentiment"] = await self._update_news_sentiment_factor_async(risk_data)
            factor_results["portfolio_concentration"] = await self._update_portfolio_concentration_factor_async(risk_data)


            try:
                mode_config = self.smart_bus.get('mode_config', 'DynamicRiskController') or {}
                decision_factors = self.smart_bus.get('decision_factors', 'DynamicRiskController') or {}
                trading_mode = self.smart_bus.get('trading_mode', 'DynamicRiskController') or 'normal'

                mode_risk_score = float(decision_factors.get('risk_score', 0.5))
                mode_drawdown_limit = float(mode_config.get('drawdown_limit', 0.10))


                if mode_risk_score < 0.5:
                    tightening_factor = 0.7 + (mode_risk_score * 0.6)
                    for factor_name in ['drawdown', 'volatility', 'losing_streak']:
                        if factor_name in self.risk_factors:
                            self.risk_factors[factor_name] *= tightening_factor

                    if self.debug:
                        self.logger.info(format_operator_message(
                            icon="🎛️",
                            message="Trading mode risk tightening applied",
                            mode=trading_mode,
                            mode_risk_score=f"{mode_risk_score:.2f}",
                            tightening=f"{tightening_factor:.2f}x",
                            affected_factors=['drawdown', 'volatility', 'losing_streak']
                        ))


                if 'drawdown' in self.risk_factors:
                    current_dd = float(risk_data.get('drawdown', 0.0))
                    if current_dd > mode_drawdown_limit * 0.8:
                        proximity = current_dd / mode_drawdown_limit
                        penalty = max(0.5, 1.0 - (proximity - 0.8) * 2.0)
                        self.risk_factors['drawdown'] *= penalty

                        if self.debug:
                            self.logger.info(format_operator_message(
                                icon="⚠️",
                                message="Mode drawdown limit proximity penalty",
                                mode=trading_mode,
                                current_dd=f"{current_dd:.1%}",
                                mode_limit=f"{mode_drawdown_limit:.1%}",
                                penalty=f"{penalty:.2f}x"
                            ))

            except Exception as e:
                if self.debug:
                    self.logger.warning(f"Trading mode integration failed in risk adjustment: {e}")


            try:

                memory_gate = self.smart_bus.get('memory_gate', 'DynamicRiskController') or {}
                if isinstance(memory_gate, dict):

                    memory_risk_mult = float(memory_gate.get('risk_multiplier', 1.0))
                    if memory_risk_mult < 1.0:

                        self.risk_factors['market_stress'] *= memory_risk_mult
                        if self.debug:
                            self.logger.info(format_operator_message(
                                icon="🧠",
                                message="Memory risk multiplier applied",
                                multiplier=f"{memory_risk_mult:.2f}x"
                            ))


                danger_zones = self.smart_bus.get('danger_zones', 'DynamicRiskController') or {}
                if isinstance(danger_zones, dict):
                    danger_similarity = float(danger_zones.get('similarity', 0.0))
                    if danger_similarity > 0.5:
                        danger_penalty = max(0.6, 1.0 - danger_similarity * 0.5)
                        self.risk_factors['market_stress'] *= danger_penalty
                        if self.debug:
                            self.logger.info(format_operator_message(
                                icon="⚠️",
                                message="Memory danger zone penalty applied",
                                similarity=f"{danger_similarity:.2f}",
                                penalty=f"{danger_penalty:.2f}x"
                            ))


                mistake_avoidance = self.smart_bus.get('mistake_avoidance', 'DynamicRiskController') or {}
                if isinstance(mistake_avoidance, dict):
                    avoidance_signal = float(mistake_avoidance.get('avoidance_signal', 0.0))
                    if avoidance_signal > 0.5:
                        avoidance_penalty = max(0.7, 1.0 - avoidance_signal * 0.4)
                        self.risk_factors['losing_streak'] *= avoidance_penalty
                        if self.debug:
                            self.logger.info(format_operator_message(
                                icon="🚫",
                                message="Memory mistake avoidance applied",
                                signal=f"{avoidance_signal:.2f}",
                                penalty=f"{avoidance_penalty:.2f}x"
                            ))


                intuition = self.smart_bus.get('intuition_vector', 'DynamicRiskController') or {}
                if isinstance(intuition, dict):
                    intuition_strength = float(intuition.get('strength', 0.5))
                    if intuition_strength > 0.7:

                        confidence_boost = min(1.2, 1.0 + (intuition_strength - 0.7) * 0.3)
                        self.risk_factors['market_stress'] *= confidence_boost
                        if self.debug:
                            self.logger.info(format_operator_message(
                                icon="💡",
                                message="Memory intuition confidence boost",
                                strength=f"{intuition_strength:.2f}",
                                boost=f"{confidence_boost:.2f}x"
                            ))

            except Exception as e:
                if self.debug:
                    self.logger.warning(f"Memory integration failed in risk adjustment: {e}")


            preliminary_scale = await self._calculate_preliminary_risk_scale_async()


            scale_change = abs(preliminary_scale - old_scale)
            if scale_change > 0.1:
                self.risk_adjustments_made += 1
                await self._record_risk_adjustment_event_async(old_scale, preliminary_scale, risk_data)


            self.current_risk_scale = preliminary_scale

            return {
                "risk_adjustment_completed": True,
                "old_scale": old_scale,
                "preliminary_scale": preliminary_scale,
                "scale_change": scale_change,
                "factor_results": factor_results,
                "significant_change": scale_change > 0.1,
            }

        except Exception as e:
            self.logger.error(f"Comprehensive risk adjustment failed: {e}")

            self.current_risk_scale = max(self._cfg.min_risk_scale, self.current_risk_scale * 0.9)
            return {"risk_adjustment_completed": False, "error": str(e)}

    async def _update_drawdown_factor_async(self, drawdown: float) -> Dict[str, Any]:
        try:
            self.dd_history.append(drawdown)

            if drawdown <= 0.05:
                factor = 1.0
                severity = "normal"
            elif drawdown <= self._cfg.dd_threshold:
                reduction = (drawdown - 0.05) / (self._cfg.dd_threshold - 0.05) * 0.4
                factor = 1.0 - reduction
                severity = "elevated"
            else:
                excess = drawdown - self._cfg.dd_threshold
                factor = 0.6 * float(np.exp(-excess * 8))
                severity = "critical"


            if self.market_regime == "volatile":
                factor *= 1.1
            elif self.market_regime == "trending":
                factor *= 0.9

            self.risk_factors["drawdown"] = float(factor)

            return {
                "drawdown_factor": float(factor),
                "drawdown_value": float(drawdown),
                "severity": severity,
                "regime_adjusted": self.market_regime != "unknown",
            }

        except Exception as e:
            self.logger.warning(f"Drawdown factor update failed: {e}")
            return {"drawdown_factor": 0.8, "error": str(e)}

    async def _update_volatility_factor_async(self, volatility: float) -> Dict[str, Any]:
        try:

            self.vol_history.append(volatility)

            if len(self.vol_history) >= 5:
                recent = list(self.vol_history)[-10:]
                avg_vol = float(np.mean(recent)) if recent else float(volatility)
                vol_ratio = float(volatility) / (avg_vol + 1e-8)

                if vol_ratio <= 1.2:
                    factor = 1.0
                    severity = "normal"
                elif vol_ratio <= self._cfg.vol_ratio_threshold:
                    reduction = (vol_ratio - 1.2) / (self._cfg.vol_ratio_threshold - 1.2) * 0.3
                    factor = 1.0 - reduction
                    severity = "elevated"
                else:
                    excess = vol_ratio - self._cfg.vol_ratio_threshold
                    factor = 0.7 * float(np.exp(-excess * 3))
                    severity = "extreme"
            else:
                factor = 1.0
                severity = "insufficient_data"
                vol_ratio = 1.0


            tolerance = float(self._adaptive_params["volatility_tolerance"])
            factor = min(1.0, float(factor * tolerance))

            self.risk_factors["volatility"] = float(factor)

            return {
                "volatility_factor": float(factor),
                "volatility_value": float(volatility),
                "vol_ratio": float(vol_ratio),
                "severity": severity,
                "tolerance_applied": tolerance != 1.0,
            }

        except Exception as e:
            self.logger.warning(f"Volatility factor update failed: {e}")
            return {"volatility_factor": 0.8, "error": str(e)}

    async def _update_correlation_factor_async(self, correlation_risk: float) -> Dict[str, Any]:
        try:
            if correlation_risk <= 0.3:
                factor = 1.0
                severity = "low"
            elif correlation_risk <= 0.6:
                reduction = (correlation_risk - 0.3) / 0.3 * 0.2
                factor = 1.0 - reduction
                severity = "moderate"
            else:
                excess = correlation_risk - 0.6
                factor = 0.8 * (1.0 - excess * self._cfg.correlation_sensitivity)
                factor = max(0.1, float(factor))
                severity = "high"

            self.risk_factors["correlation"] = float(factor)

            return {
                "correlation_factor": float(factor),
                "correlation_risk": float(correlation_risk),
                "severity": severity,
            }

        except Exception as e:
            self.logger.warning(f"Correlation factor update failed: {e}")
            return {"correlation_factor": 1.0, "error": str(e)}

    async def _update_losing_streak_factor_async(self, pnl: float) -> Dict[str, Any]:
        try:

            if pnl < 0:
                self.consecutive_losses += 1
            elif pnl > 0:
                self.consecutive_losses = 0


            self.last_pnl = float(pnl)


            if self.consecutive_losses <= 2:
                factor = 1.0
                severity = "normal"
            elif self.consecutive_losses <= 5:

                reduction = (self.consecutive_losses - 2) * 0.15
                factor = 1.0 - reduction
                severity = "elevated"
            else:

                factor = 0.4
                severity = "critical"

            self.risk_factors["losing_streak"] = float(factor)

            return {
                "losing_streak_factor": float(factor),
                "consecutive_losses": int(self.consecutive_losses),
                "pnl": float(pnl),
                "severity": severity,
            }

        except Exception as e:
            self.logger.warning(f"Losing streak factor update failed: {e}")
            return {"losing_streak_factor": 1.0, "error": str(e)}

    async def _update_liquidity_factor_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        try:

            market_data = risk_data.get("market_data", {})
            market_status = market_data.get("market_status", {})
            liquidity_score = float(market_status.get("liquidity_score", 1.0))

            if liquidity_score >= 0.8:
                factor = 1.0
                severity = "normal"
            elif liquidity_score >= 0.5:
                factor = 0.8 + liquidity_score * 0.2
                severity = "reduced"
            else:
                factor = 0.5 + liquidity_score * 0.3
                severity = "low"

            self.risk_factors["liquidity"] = float(factor)

            return {"liquidity_factor": float(factor), "liquidity_score": liquidity_score, "severity": severity}

        except Exception:
            self.risk_factors["liquidity"] = 1.0
            return {"liquidity_factor": 1.0, "severity": "unknown"}

    async def _update_news_sentiment_factor_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        try:

            market_data = risk_data.get("market_data", {})
            market_context = market_data.get("market_context", {})
            news_sentiment = float(market_context.get("news_sentiment", 0.0))

            if news_sentiment >= -0.2:
                factor = 1.0
                severity = "positive"
            elif news_sentiment >= -0.5:
                factor = 0.9
                severity = "negative"
            else:
                factor = 0.7
                severity = "very_negative"

            self.risk_factors["news_sentiment"] = float(factor)

            return {
                "news_sentiment_factor": float(factor),
                "news_sentiment": news_sentiment,
                "severity": severity,
            }

        except Exception:
            self.risk_factors["news_sentiment"] = 1.0
            return {"news_sentiment_factor": 1.0, "severity": "unknown"}

    async def _update_portfolio_concentration_factor_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        try:

            position_data = risk_data.get("position_data", {})
            positions = position_data.get("positions", [])
            herfindahl = 0.0

            if not positions:
                factor = 1.0
                severity = "no_positions"
            else:

                total_exposure = sum(abs(pos.get("size", 0)) for pos in positions)
                if total_exposure > 0:
                    concentrations = [(abs(pos.get("size", 0)) / total_exposure) ** 2 for pos in positions]
                    herfindahl = float(sum(concentrations))

                    if herfindahl <= 0.3:
                        factor = 1.0
                        severity = "diversified"
                    elif herfindahl <= 0.6:
                        factor = 0.9
                        severity = "moderate"
                    else:
                        factor = 0.7
                        severity = "concentrated"
                else:
                    factor = 1.0
                    severity = "no_exposure"
                    herfindahl = 0.0

            self.risk_factors["portfolio_concentration"] = float(factor)

            return {
                "portfolio_concentration_factor": float(factor),
                "position_count": len(positions),
                "concentration_index": float(herfindahl),
                "severity": severity,
            }

        except Exception as e:
            self.logger.warning(f"Portfolio concentration factor update failed: {e}")
            self.risk_factors["portfolio_concentration"] = 1.0
            return {"portfolio_concentration_factor": 1.0, "error": str(e)}

    async def _calculate_preliminary_risk_scale_async(self) -> float:
        try:
            scale = float(self._cfg.base_risk_scale)


            for factor_value in self.risk_factors.values():
                scale *= float(max(0.0, factor_value))


            scale *= float(max(0.0, self.external_risk_scale))


            return float(np.clip(scale, self._cfg.min_risk_scale, self._cfg.max_risk_scale))

        except Exception as e:
            self.logger.error(f"Preliminary risk scale calculation failed: {e}")
            return float(self._cfg.min_risk_scale)

    async def _apply_adaptive_scaling_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if len(self.risk_scale_history) < 10:
                return {"adaptive_scaling_applied": False, "reason": "insufficient_history"}


            recent_scales = list(self.risk_scale_history)[-10:]
            recent_performance = await self._calculate_recent_performance_async(risk_data)

            adaptation_factor = 1.0
            adaptation_reason = "no_change"

            avg_recent_scale = float(np.mean(recent_scales)) if recent_scales else 1.0


            if avg_recent_scale < 0.7 and recent_performance < -0.1:
                adaptation_factor = 1.2
                adaptation_reason = "poor_performance_conservative"

            elif avg_recent_scale < 0.7 and recent_performance > 0.1:
                adaptation_factor = 0.9
                adaptation_reason = "good_performance_conservative"

            elif avg_recent_scale > 0.8 and recent_performance < -0.1:
                adaptation_factor = 0.8
                adaptation_reason = "poor_performance_aggressive"
            else:
                adaptation_factor = 1.0
                adaptation_reason = "stable_performance"


            learning_rate = float(self._cfg.adaptive_learning_rate)
            current_confidence = float(self._adaptive_params["risk_adaptation_confidence"])


            if abs(adaptation_factor - 1.0) > 0.1:
                self._adaptive_params["risk_adaptation_confidence"] = min(1.0, current_confidence + learning_rate)
            else:
                self._adaptive_params["risk_adaptation_confidence"] = max(0.1, current_confidence - learning_rate * 0.5)


            final_adaptation = 1.0 + (adaptation_factor - 1.0) * float(self._adaptive_params["risk_adaptation_confidence"])

            return {
                "adaptive_scaling_applied": True,
                "adaptation_factor": float(adaptation_factor),
                "final_adaptation": float(final_adaptation),
                "adaptation_reason": adaptation_reason,
                "recent_performance": float(recent_performance),
                "adaptation_confidence": float(self._adaptive_params["risk_adaptation_confidence"]),
            }

        except Exception as e:
            self.logger.warning(f"Adaptive scaling failed: {e}")
            return {"adaptive_scaling_applied": False, "error": str(e)}

    async def _calculate_recent_performance_async(self, risk_data: Dict[str, Any]) -> float:
        try:

            drawdown = float(risk_data.get("drawdown", 0.0))
            pnl = float(risk_data.get("pnl", 0.0))

            drawdown_score = max(0.0, 1.0 - drawdown * 5.0)
            pnl_score = float(np.tanh(pnl / 100.0))


            if len(self.vol_history) >= 5:
                recent_vol = float(np.mean(list(self.vol_history)[-5:]))
                vol_adjustment = 1.0 - min(0.3, float(recent_vol * 10.0))
            else:
                vol_adjustment = 1.0

            performance = (drawdown_score + pnl_score) / 2.0 * vol_adjustment
            return float(performance)

        except Exception:
            return 0.0

    async def _apply_emergency_interventions_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            interventions_applied: List[str] = []
            emergency_scale = float(self.current_risk_scale)


            drawdown = float(risk_data.get("drawdown", 0.0))
            emergency_dd_threshold = 0.2 * float(self._adaptive_params["emergency_threshold_adaptation"])
            if drawdown > emergency_dd_threshold:
                emergency_scale = min(emergency_scale, 0.3)
                interventions_applied.append(f"extreme_drawdown_{drawdown:.1%}")


            if len(self.vol_history) > 5 and float(risk_data.get("volatility", 0.0)) > float(np.mean(self.vol_history)) * 3.0:
                emergency_scale = min(emergency_scale, 0.4)
                interventions_applied.append("extreme_volatility")


            active_risk_factors = sum(1 for factor in self.risk_factors.values() if float(factor) < 0.8)
            if active_risk_factors >= 4:
                emergency_scale = min(emergency_scale, 0.5)
                interventions_applied.append(f"multiple_risk_factors_{active_risk_factors}")


            correlation = float(risk_data.get("correlation", 0.0))
            if correlation > 0.8:
                emergency_scale = min(emergency_scale, 0.6)
                interventions_applied.append(f"extreme_correlation_{correlation:.2f}")


            if interventions_applied:
                self.emergency_interventions += 1
                self.logger.warning(
                    format_operator_message(
                        message="Emergency risk intervention triggered",
                        icon="[ALERT]",
                        interventions=len(interventions_applied),
                        old_scale=f"{self.current_risk_scale:.2f}",
                        new_scale=f"{emergency_scale:.2f}",
                        reasons=", ".join(interventions_applied[:2]),
                    )
                )


            self.current_risk_scale = float(emergency_scale)

            return {
                "emergency_interventions_applied": len(interventions_applied) > 0,
                "interventions_count": len(interventions_applied),
                "interventions": interventions_applied,
                "emergency_scale": float(emergency_scale),
                "total_emergency_interventions": self.emergency_interventions,
            }

        except Exception as e:
            self.logger.warning(f"Emergency intervention failed: {e}")
            return {"emergency_interventions_applied": False, "error": str(e)}

    async def _calculate_final_risk_scale_async(self) -> Dict[str, Any]:
        try:

            if self.current_risk_scale < self._cfg.base_risk_scale:
                recovery_adjustment = (self._cfg.base_risk_scale - self.current_risk_scale) * (1.0 - self._cfg.risk_decay)
                self.current_risk_scale = min(
                    self._cfg.base_risk_scale,
                    self.current_risk_scale + float(recovery_adjustment),
                )


            self.current_risk_scale = float(
                np.clip(self.current_risk_scale, self._cfg.min_risk_scale, self._cfg.max_risk_scale)
            )


            self.risk_scale_history.append(self.current_risk_scale)


            await self._calculate_risk_quality_async()

            return {
                "final_risk_scale_calculated": True,
                "final_risk_scale": float(self.current_risk_scale),
                "risk_quality": float(self._risk_quality),
                "scale_within_bounds": self._cfg.min_risk_scale <= self.current_risk_scale <= self._cfg.max_risk_scale,
            }

        except Exception as e:
            self.logger.warning(f"Final risk scale calculation failed: {e}")
            return {"final_risk_scale_calculated": False, "error": str(e)}

    async def _calculate_risk_quality_async(self) -> None:
        try:
            quality_factors: List[float] = []


            risk_factor_health = float(np.mean([max(0.3, float(factor)) for factor in self.risk_factors.values()]))
            if risk_factor_health > 0.8:
                scale_appropriateness = 1.0 - abs(self.current_risk_scale - self._cfg.base_risk_scale)
            else:
                scale_appropriateness = 1.0 - self.current_risk_scale
            quality_factors.append(max(0.0, float(scale_appropriateness)))


            if len(self.risk_scale_history) >= 10:
                recent_scales = list(self.risk_scale_history)[-10:]
                scale_stability = 1.0 - float(np.std(recent_scales))
                quality_factors.append(max(0.0, float(scale_stability)))


            adaptation_confidence = float(self._adaptive_params.get("risk_adaptation_confidence", 0.5))
            quality_factors.append(adaptation_confidence)


            emergency_frequency = max(0.0, 1.0 - (self.emergency_interventions / 10.0))
            quality_factors.append(float(emergency_frequency))

            self._risk_quality = float(np.mean(quality_factors)) if quality_factors else 0.5


            self._risk_effectiveness_history.append(self._risk_quality)

        except Exception as e:
            self.logger.warning(f"Risk quality calculation failed: {e}")
            self._risk_quality = 0.5

    async def _update_operational_mode_async(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            old_mode = self.current_mode


            if self.emergency_interventions > 0 and self.current_risk_scale < 0.4:
                new_mode = RiskControlMode.EMERGENCY
            elif self.current_risk_scale < 0.6 or sum(1 for f in self.risk_factors.values() if f < 0.7) >= 3:
                new_mode = RiskControlMode.AGGRESSIVE_REDUCTION
            elif self.current_risk_scale < 0.8 or risk_data.get("drawdown", 0.0) > 0.1:
                new_mode = RiskControlMode.PROTECTIVE
            elif self._risk_quality < 0.3:
                new_mode = RiskControlMode.CALIBRATION
            elif len(self.risk_scale_history) >= 20 and self.current_risk_scale > self._cfg.base_risk_scale * 0.8:
                new_mode = RiskControlMode.RECOVERY
            else:
                new_mode = RiskControlMode.NORMAL


            mode_changed = False
            if new_mode != old_mode:
                self.current_mode = new_mode
                self.mode_start_time = datetime.datetime.now()
                mode_changed = True

                self.logger.info(
                    format_operator_message(
                        message="Risk mode changed",
                        icon="[RELOAD]",
                        old_mode=old_mode.value,
                        new_mode=new_mode.value,
                        risk_scale=f"{self.current_risk_scale:.2f}",
                        risk_quality=f"{self._risk_quality:.2f}",
                    )
                )

            return {
                "mode_updated": True,
                "current_mode": self.current_mode.value,
                "mode_changed": mode_changed,
                "old_mode": old_mode.value if mode_changed else None,
                "mode_duration": (datetime.datetime.now() - self.mode_start_time).total_seconds(),
            }

        except Exception as e:
            self.logger.warning(f"Mode update failed: {e}")
            return {"mode_updated": False, "error": str(e)}

    async def _record_risk_adjustment_event_async(
        self, old_scale: float, new_scale: float, risk_data: Dict[str, Any]
    ) -> None:
        try:
            event = {
                "timestamp": datetime.datetime.now().isoformat(),
                "old_scale": float(old_scale),
                "new_scale": float(new_scale),
                "change": float(new_scale - old_scale),
                "risk_data": risk_data.copy(),
                "risk_factors": self.risk_factors.copy(),
                "reason": await self._determine_adjustment_reason_async(old_scale, new_scale, risk_data),
            }

            self.risk_events.append(event)


            while len(self.risk_events) > 50:
                self.risk_events.popleft()


            if abs(new_scale - old_scale) > 0.2:
                self.logger.warning(
                    format_operator_message(
                        message="Significant risk adjustment made",
                        icon="⚙️",
                        old_scale=f"{old_scale:.2f}",
                        new_scale=f"{new_scale:.2f}",
                        reason=event["reason"],
                        regime=self.market_regime,
                    )
                )

        except Exception as e:
            self.logger.warning(f"Risk event recording failed: {e}")

    async def _determine_adjustment_reason_async(
        self, old_scale: float, new_scale: float, risk_data: Dict[str, Any]
    ) -> str:
        try:
            if new_scale < old_scale:
                if risk_data.get("drawdown", 0.0) > 0.1:
                    return "drawdown_protection"
                elif risk_data.get("correlation", 0.0) > 0.6:
                    return "correlation_risk"
                elif self.consecutive_losses > 3:
                    return "losing_streak"
                elif len(self.vol_history) > 5 and risk_data.get("volatility", 0.0) > np.mean(self.vol_history) * 2.0:
                    return "volatility_protection"
                else:
                    return "general_risk_reduction"
            else:
                if self._cfg.recovery_speed > 0.1:
                    return "recovery_mode"
                else:
                    return "favorable_conditions"

        except Exception:
            return "unknown"

    async def _generate_comprehensive_risk_thesis(self, risk_data: Dict[str, Any], result: Dict[str, Any]) -> str:
        try:

            risk_scale = self.current_risk_scale
            mode = self.current_mode.value
            risk_quality = self._risk_quality

            thesis_parts = [
                f"Risk Control: {mode.upper()} mode with {risk_scale:.1%} scaling factor",
                f"Risk Quality: {risk_quality:.2f} assessment score",
            ]


            if risk_scale < 0.5:
                thesis_parts.append("DEFENSIVE: Significant risk reduction active")
            elif risk_scale > 0.9:
                thesis_parts.append("AGGRESSIVE: Near-normal risk exposure")


            active_factors = [name for name, value in self.risk_factors.items() if value < 0.9]
            if active_factors:
                thesis_parts.append(f"Active factors: {', '.join(active_factors[:3])}")


            vol_text = self.volatility_regime.upper() if hasattr(self.volatility_regime, "upper") else str(
                self.volatility_regime
            ).upper()
            thesis_parts.append(f"Market: {self.market_regime.upper()} regime, {vol_text} volatility")


            if result.get("significant_change", False):
                change = result.get("scale_change", 0.0)
                thesis_parts.append(f"Adjustment: {change:+.2f} scale change applied")


            if result.get("emergency_interventions_applied", False):
                interventions = result.get("interventions_count", 0)
                thesis_parts.append(f"EMERGENCY: {interventions} interventions triggered")


            external_count = len(self.external_signals)
            if external_count > 0:
                thesis_parts.append(f"External signals: {external_count} integrated")

            return " | ".join(thesis_parts)

        except Exception as e:
            return f"Risk thesis generation failed: {e!s} - Core risk scaling functional"

    async def _update_risk_smart_bus(self, result: Dict[str, Any], thesis: str):
        try:

            scaling_data = {
                "current_mode": self.current_mode.value,
                "current_risk_scale": self.current_risk_scale,
                "base_risk_scale": self._cfg.base_risk_scale,
                "min_risk_scale": self._cfg.min_risk_scale,
                "max_risk_scale": self._cfg.max_risk_scale,
                "adaptive_scaling": self.adaptive_scaling,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            self.smart_bus.set("risk_scaling", scaling_data, module="DynamicRiskController", thesis=thesis)


            factors_data = {
                "risk_factors": self.risk_factors.copy(),
                "external_risk_scale": self.external_risk_scale,
                "external_signals": self.external_signals.copy(),
                "consecutive_losses": self.consecutive_losses,
                "risk_adjustments_made": self.risk_adjustments_made,
                "emergency_interventions": self.emergency_interventions,
            }

            self.smart_bus.set(
                "risk_factors",
                factors_data,
                module="DynamicRiskController",
                thesis="Current risk factors and external signal integration",
            )


            analytics_data = {
                "risk_quality": self._risk_quality,
                "adaptive_params": self._adaptive_params.copy(),
                "regime_performance": {
                    regime: {
                        "scale_history": len(data["risk_scales"]),
                        "avg_scale": float(np.mean(data["risk_scales"][-10:])) if data["risk_scales"] else self.current_risk_scale,
                    }
                    for regime, data in self.regime_performance.items()
                },
                "risk_events": len(self.risk_events),
                "scale_history_size": len(self.risk_scale_history),
                "volatility_history_size": len(self.vol_history),
            }

            self.smart_bus.set(
                "risk_analytics",
                analytics_data,
                module="DynamicRiskController",
                thesis="Risk control analytics and performance tracking",
            )


            alerts_data = {
                "emergency_interventions": self.emergency_interventions,
                "risk_adjustments_made": self.risk_adjustments_made,
                "critical_mode": self.current_mode in [RiskControlMode.EMERGENCY, RiskControlMode.AGGRESSIVE_REDUCTION],
                "low_risk_quality": self._risk_quality < self._cfg.min_risk_quality,
                "recent_events": len(
                    [
                        e
                        for e in self.risk_events
                        if (datetime.datetime.now() - datetime.datetime.fromisoformat(e["timestamp"])).total_seconds()
                        < 3600
                    ]
                ),
            }

            self.smart_bus.set(
                "risk_alerts",
                alerts_data,
                module="DynamicRiskController",
                thesis="Risk control alerts and emergency status tracking",
            )


            risk_level_str = "NORMAL"
            if self.current_mode == RiskControlMode.EMERGENCY:
                risk_level_str = "CRITICAL"
            elif self.current_mode == RiskControlMode.AGGRESSIVE_REDUCTION:
                risk_level_str = "HIGH"
            elif self.current_mode == RiskControlMode.PROTECTIVE:
                risk_level_str = "ELEVATED"
            elif self.current_mode == RiskControlMode.RECOVERY:
                risk_level_str = "LOW"

            self.smart_bus.set(
                "risk_level",
                risk_level_str,
                module="DynamicRiskController",
                thesis=f"Current risk level: {risk_level_str}",
            )

            self.smart_bus.set(
                "risk_scale",
                float(self.current_risk_scale),
                module="DynamicRiskController",
                thesis=f"Current risk scale: {self.current_risk_scale:.2f}",
            )


            risk_assessment = {
                "risk_level": risk_level_str,
                "risk_scale": float(self.current_risk_scale),
                "risk_quality": float(self._risk_quality),
                "market_regime": self.market_regime,
                "mode": self.current_mode.value,
                "adaptive_scaling": self.adaptive_scaling,
                "consecutive_losses": self.consecutive_losses,
                "emergency_active": self.current_mode == RiskControlMode.EMERGENCY,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            self.smart_bus.set(
                "risk_assessment",
                risk_assessment,
                module="DynamicRiskController",
                thesis="Risk assessment summary for API consumption",
            )


            if "DynamicRiskController_voting_proposal" in result:
                proposal = result["DynamicRiskController_voting_proposal"]
                confidence = float(result.get("DynamicRiskController_confidence", 0.0))


                try:
                    self.smart_bus.set(
                        "DynamicRiskController_voting_proposal",
                        proposal,
                        module="DynamicRiskController",
                        thesis="DynamicRiskController voting proposal",
                    )
                    self.smart_bus.set(
                        "DynamicRiskController_confidence",
                        confidence,
                        module="DynamicRiskController",
                        thesis=f"DynamicRiskController voting confidence: {confidence:.1%}",
                    )
                except Exception as e:
                    self.logger.warning(f"Publishing canonical voting keys failed: {e}")


                try:

                    if bool(self.config.get("publish_to_expert_votes_feed", False)):
                        feed_key = "expert_votes"
                        entry = {
                            "expert": "DynamicRiskController",
                            "vote": dict(proposal) if isinstance(proposal, dict) else {},
                            "confidence": confidence,
                            "timestamp": datetime.datetime.now().isoformat(),
                        }
                        buf = self.smart_bus.get(feed_key, "DynamicRiskController") or []
                        if not isinstance(buf, list):
                            buf = []

                        buf = [e for e in buf if e.get("expert") != "DynamicRiskController"]
                        buf.append(entry)

                        cap = int(self.config.get("max_expert_votes_buffer", 200))
                        if len(buf) > cap:
                            buf = buf[-cap:]
                        self.smart_bus.set(feed_key, buf, module="DynamicRiskController", thesis="Published normalized vote entry")
                except Exception:
                    pass


                try:
                    self.smart_bus.publish(
                        "vote",
                        {
                            "expert": "DynamicRiskController",
                            "vote": dict(proposal) if isinstance(proposal, dict) else {},
                            "confidence": float(confidence),
                            "timestamp": datetime.datetime.now().isoformat(),
                        },
                        module="DynamicRiskController",
                        thesis="DynamicRiskController vote stream",
                    )
                except Exception:
                    pass

        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        self.logger.warning("No risk data available - maintaining current scale")
        thesis = "No risk data available - maintaining current risk posture"


        risk_level_str = "NORMAL"
        if self.current_mode == RiskControlMode.EMERGENCY:
            risk_level_str = "CRITICAL"
        elif self.current_mode == RiskControlMode.AGGRESSIVE_REDUCTION:
            risk_level_str = "HIGH"
        elif self.current_mode == RiskControlMode.PROTECTIVE:
            risk_level_str = "ELEVATED"
        elif self.current_mode == RiskControlMode.RECOVERY:
            risk_level_str = "LOW"

        return {
            "success": True,
            "current_mode": self.current_mode.value,
            "current_risk_scale": self.current_risk_scale,
            "risk_quality": self._risk_quality,
            "fallback_reason": "no_risk_data",
            "risk_scaling": {
                "current_mode": self.current_mode.value,
                "current_risk_scale": self.current_risk_scale,
                "base_risk_scale": self._cfg.base_risk_scale,
                "min_risk_scale": self._cfg.min_risk_scale,
                "max_risk_scale": self._cfg.max_risk_scale,
                "adaptive_scaling": self.adaptive_scaling,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "risk_factors": {
                "risk_factors": self.risk_factors.copy(),
                "external_risk_scale": self.external_risk_scale,
                "external_signals": self.external_signals.copy(),
                "consecutive_losses": self.consecutive_losses,
                "risk_adjustments_made": self.risk_adjustments_made,
                "emergency_interventions": self.emergency_interventions,
            },
            "risk_analytics": {
                "risk_quality": self._risk_quality,
                "adaptive_params": self._adaptive_params.copy(),
                "risk_events": len(self.risk_events),
                "scale_history_size": len(self.risk_scale_history),
                "volatility_history_size": len(self.vol_history),
            },
            "risk_alerts": {
                "emergency_interventions": self.emergency_interventions,
                "risk_adjustments_made": self.risk_adjustments_made,
                "critical_mode": self.current_mode in [RiskControlMode.EMERGENCY, RiskControlMode.AGGRESSIVE_REDUCTION],
                "low_risk_quality": self._risk_quality < self._cfg.min_risk_quality,
            },
            "risk_level": risk_level_str,
            "risk_scale": float(self.current_risk_scale),
            "risk_assessment": {
                "risk_level": risk_level_str,
                "risk_scale": float(self.current_risk_scale),
                "risk_quality": float(self._risk_quality),
                "market_regime": self.market_regime,
                "mode": self.current_mode.value,
                "adaptive_scaling": self.adaptive_scaling,
                "consecutive_losses": self.consecutive_losses,
                "emergency_active": self.current_mode == RiskControlMode.EMERGENCY,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "_thesis": thesis,
        }

    async def _handle_risk_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        processing_time = (time.time() - start_time) * 1000.0


        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()

        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
            self.circuit_breaker["state"] = "OPEN"
            self._health_status = "warning"


        error_context = self.error_pinpointer.analyze_error(error, "DynamicRiskController")
        explanation = self.english_explainer.explain_error("DynamicRiskController", str(error), "risk scaling")

        self.logger.error(
            format_operator_message(
                message="Risk controller error",
                icon="[CRASH]",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                circuit_breaker_state=self.circuit_breaker["state"],
            )
        )


        self._record_failure(error)

        return self._create_error_fallback_response(f"error: {error!s}")

    def _create_error_fallback_response(self, reason: str) -> Dict[str, Any]:
        thesis = f"Risk control error fallback engaged: {reason}"
        return {
            "success": False,
            "current_mode": RiskControlMode.EMERGENCY.value,
            "current_risk_scale": self._cfg.min_risk_scale,
            "risk_quality": 0.1,
            "circuit_breaker_state": self.circuit_breaker["state"],
            "fallback_reason": reason,
            "risk_scaling": {
                "current_mode": RiskControlMode.EMERGENCY.value,
                "current_risk_scale": self._cfg.min_risk_scale,
                "base_risk_scale": self._cfg.base_risk_scale,
                "min_risk_scale": self._cfg.min_risk_scale,
                "max_risk_scale": self._cfg.max_risk_scale,
                "adaptive_scaling": self.adaptive_scaling,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "risk_factors": {
                "risk_factors": self.risk_factors.copy(),
                "external_risk_scale": self.external_risk_scale,
                "external_signals": self.external_signals.copy(),
                "consecutive_losses": self.consecutive_losses,
                "risk_adjustments_made": self.risk_adjustments_made,
                "emergency_interventions": self.emergency_interventions,
            },
            "risk_analytics": {
                "risk_quality": 0.1,
                "adaptive_params": self._adaptive_params.copy(),
                "risk_events": len(self.risk_events),
                "scale_history_size": len(self.risk_scale_history),
                "volatility_history_size": len(self.vol_history),
            },
            "risk_alerts": {
                "emergency_interventions": self.emergency_interventions,
                "risk_adjustments_made": self.risk_adjustments_made,
                "critical_mode": True,
                "low_risk_quality": True,
            },
            "risk_level": "CRITICAL",
            "risk_scale": float(self._cfg.min_risk_scale),
            "risk_assessment": {
                "risk_level": "CRITICAL",
                "risk_scale": float(self._cfg.min_risk_scale),
                "risk_quality": 0.1,
                "market_regime": self.market_regime,
                "mode": RiskControlMode.EMERGENCY.value,
                "adaptive_scaling": self.adaptive_scaling,
                "consecutive_losses": self.consecutive_losses,
                "emergency_active": True,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "DynamicRiskController_voting_proposal": {
                "member": "DynamicRiskController",
                "type": "risk_posture",
                "action": "reduce_risk",
                "posture": "halt",
                "target_scale": float(self._cfg.min_risk_scale),
                "bounds": [float(self._cfg.min_risk_scale), float(self._cfg.max_risk_scale)],
                "confidence": 0.1,
                "rationale": "Error fallback - holding due to risk control failure",
                "context": {
                    "mode": RiskControlMode.EMERGENCY.value,
                    "reason": reason,
                },
            },
            "DynamicRiskController_confidence": 0.1,
            "_thesis": thesis,
        }

    def _update_risk_health(self):
        try:

            if not hasattr(self, "_risk_quality"):
                return


            if self._risk_quality < self._cfg.min_risk_quality:
                self._health_status = "warning"
            else:
                self._health_status = "healthy"


            if self.circuit_breaker["state"] == "OPEN":
                self._health_status = "warning"


            if self.emergency_interventions > 5:
                self._health_status = "warning"

            self._last_health_check = time.time()

        except Exception as e:
            self.logger.error(f"Risk health check failed: {e}")
            self._health_status = "warning"

    def _analyze_risk_effectiveness(self):
        try:

            if not hasattr(self, "risk_scale_history"):
                return

            if len(self.risk_scale_history) >= 20:
                effectiveness = self._risk_quality

                if effectiveness > 0.8:
                    self.logger.info(
                        format_operator_message(
                            message="High risk effectiveness achieved",
                            icon="[TARGET]",
                            quality_score=f"{effectiveness:.2f}",
                            current_scale=f"{self.current_risk_scale:.2f}",
                        )
                    )
                elif effectiveness < 0.4:
                    self.logger.warning(
                        format_operator_message(
                            message="Low risk effectiveness detected",
                            icon="[WARN]",
                            quality_score=f"{effectiveness:.2f}",
                            emergency_interventions=self.emergency_interventions,
                        )
                    )

        except Exception as e:
            self.logger.error(f"Risk effectiveness analysis failed: {e}")

    def _adapt_risk_parameters(self):
        try:

            if not hasattr(self, "market_regime") or not hasattr(self, "market_regime_history"):
                return


            if self.market_regime == "volatile":
                self._adaptive_params["emergency_threshold_adaptation"] = min(
                    1.3, self._adaptive_params["emergency_threshold_adaptation"] * 1.005
                )
            else:
                self._adaptive_params["emergency_threshold_adaptation"] = max(
                    0.8, self._adaptive_params["emergency_threshold_adaptation"] * 0.999
                )


            if len(self.market_regime_history) >= 5:
                recent_changes = len([h for h in list(self.market_regime_history)[-5:]])
                if recent_changes > 2:
                    self._adaptive_params["regime_sensitivity_multiplier"] = min(
                        1.5, self._adaptive_params["regime_sensitivity_multiplier"] * 1.01
                    )
                else:
                    self._adaptive_params["regime_sensitivity_multiplier"] = max(
                        0.7, self._adaptive_params["regime_sensitivity_multiplier"] * 0.995
                    )

        except Exception as e:
            self.logger.warning(f"Risk parameter adaptation failed: {e}")

    def _record_success(self, processing_time: float):
        self.performance_tracker.record_metric("DynamicRiskController", "risk_scaling", processing_time, True)


        if self.circuit_breaker["state"] == "OPEN":
            self.circuit_breaker["failures"] = 0
            self.circuit_breaker["state"] = "CLOSED"

    def _record_failure(self, error: Exception):
        self.performance_tracker.record_metric("DynamicRiskController", "risk_scaling", 0, False)


    async def vote(self, risk_scale: Optional[float] = None) -> Dict[str, Any]:
        try:
            now = time.time()
            current_scale = float(self.current_risk_scale if risk_scale is None else risk_scale)


            if self.circuit_breaker["state"] == "OPEN" or self.current_mode == RiskControlMode.EMERGENCY:
                posture = "halt"
                target_scale = max(self._cfg.min_risk_scale, min(current_scale, 0.2))
                rationale = "Circuit breaker/emergency posture."
            else:

                if self.current_mode in (RiskControlMode.AGGRESSIVE_REDUCTION, RiskControlMode.PROTECTIVE):
                    posture = "reduce"
                    target_scale = max(self._cfg.min_risk_scale, min(current_scale, 0.6))
                    rationale = "Protective posture due to risk factors."
                elif self.current_mode == RiskControlMode.RECOVERY:
                    posture = "increase" if current_scale < 0.9 else "maintain"
                    target_scale = min(self._cfg.max_risk_scale, max(current_scale, 0.9))
                    rationale = "Recovery mode with improving conditions."
                else:


                    weak_factors = sum(1 for v in self.risk_factors.values() if v < 0.8)
                    if weak_factors >= 3 or self.external_risk_scale < 0.9:
                        posture = "reduce"
                        target_scale = max(self._cfg.min_risk_scale, min(current_scale, 0.8))
                        rationale = "Multiple weak risk factors."
                    else:
                        posture = "maintain"
                        target_scale = float(current_scale)
                        rationale = "Risk factors broadly acceptable."


            proposed_action = {
                "action_type": "risk_posture",
                "posture": posture,
                "target_scale": target_scale,
                "mode": self.current_mode.value,
                "risk_quality": self._risk_quality,
                "circuit_breaker": self.circuit_breaker["state"],
            }
            confidence = await self.calculate_confidence(proposed_action)


            jump = abs(target_scale - current_scale)
            if jump > 0.25:
                confidence *= 0.9


            confidence = float(max(0.0, min(1.0, confidence)))


            action_map = {
                "halt": "reduce_risk",
                "reduce": "reduce_risk",
                "increase": "increase_risk",
                "maintain": "hold",
            }

            payload = {
                "member": "DynamicRiskController",
                "action": action_map.get(posture, "hold"),
                "type": "risk_posture",
                "posture": posture,
                "target_scale": float(np.clip(target_scale, self._cfg.min_risk_scale, self._cfg.max_risk_scale)),
                "bounds": [float(self._cfg.min_risk_scale), float(self._cfg.max_risk_scale)],
                "rationale": rationale,
                "confidence": confidence,
                "context": {
                    "mode": self.current_mode.value,
                    "market_regime": self.market_regime,
                    "volatility_regime": self.volatility_regime,
                    "risk_quality": float(self._risk_quality),
                    "external_risk_scale": float(self.external_risk_scale),
                    "weak_factors": sum(1 for v in self.risk_factors.values() if v < 0.8),
                },
                "timestamp": now,
            }
            return payload

        except Exception as e:
            self.logger.warning(f"Vote generation failed: {e}")
            return {
                "member": "DynamicRiskController",
                "type": "risk_posture",
                "action": "hold",
                "posture": "maintain",
                "target_scale": float(self.current_risk_scale),
                "bounds": [float(self._cfg.min_risk_scale), float(self._cfg.max_risk_scale)],
                "rationale": f"fallback: {e}",
                "confidence": 0.5,
                "context": {},
                "timestamp": time.time(),
            }


    def get_current_risk_scale(self) -> float:
        return float(self.current_risk_scale)

    def set_external_risk_scale(self, scale: float) -> None:
        self.external_risk_scale = float(np.clip(scale, 0.1, 2.0))

    def get_risk_factors(self) -> Dict[str, float]:
        return self.risk_factors.copy()

    def force_emergency_mode(self, reason: str = "manual_override") -> None:
        old_scale = float(self.current_risk_scale)
        self.current_risk_scale = float(self._cfg.min_risk_scale)
        self.emergency_interventions += 1
        self.current_mode = RiskControlMode.EMERGENCY

        self.logger.error(
            format_operator_message(
                message="Emergency mode forced",
                icon="[ALERT]",
                reason=reason,
                old_scale=f"{old_scale:.2f}",
                new_scale=f"{self.current_risk_scale:.2f}",
            )
        )

    def get_observation_components(self) -> np.ndarray:
        try:
            return np.array(
                [
                    float(self.current_risk_scale),
                    float(self.risk_factors["drawdown"]),
                    float(self.risk_factors["volatility"]),
                    float(self.risk_factors["correlation"]),
                    float(self.risk_factors["losing_streak"]),
                    float(min(self.consecutive_losses / 10.0, 1.0)),
                    float(self.external_risk_scale),
                    float(1.0 if self.market_regime in ["volatile", "extreme"] else 0.0),
                    float(self._risk_quality),
                    float(1.0 if self.current_mode in [RiskControlMode.EMERGENCY, RiskControlMode.AGGRESSIVE_REDUCTION] else 0.0),
                ],
                dtype=np.float32,
            )

        except Exception as e:
            self.logger.error(f"Risk observation generation failed: {e}")
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0], dtype=np.float32)

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "status": self._health_status,
            "last_check": self._last_health_check,
            "circuit_breaker": self.circuit_breaker["state"],
            "current_mode": self.current_mode.value,
            "current_risk_scale": self.current_risk_scale,
            "risk_quality": self._risk_quality,
            "emergency_interventions": self.emergency_interventions,
        }

    def stop_monitoring(self):
        self._monitoring_active = False

    def get_risk_control_report(self) -> str:


        if self.current_risk_scale < 0.3:
            risk_status = "[ALERT] Emergency"
        elif self.current_risk_scale < 0.6:
            risk_status = "[WARN] High Reduction"
        elif self.current_risk_scale < 0.8:
            risk_status = "[FAST] Moderate Reduction"
        else:
            risk_status = "[OK] Normal"


        mode_emoji = {
            RiskControlMode.INITIALIZATION: "[RELOAD]",
            RiskControlMode.CALIBRATION: "[TOOL]",
            RiskControlMode.NORMAL: "[OK]",
            RiskControlMode.PROTECTIVE: "[SAFE]",
            RiskControlMode.AGGRESSIVE_REDUCTION: "[WARN]",
            RiskControlMode.EMERGENCY: "🆘",
            RiskControlMode.RECOVERY: "[CHART]",
        }

        mode_status = f"{mode_emoji.get(self.current_mode, '❓')} {self.current_mode.value.upper()}"


        health_emoji = "[OK]" if self._health_status == "healthy" else "[WARN]"
        cb_status = "[RED] OPEN" if self.circuit_breaker["state"] == "OPEN" else "[GREEN] CLOSED"


        risk_factor_lines: List[str] = []
        for factor_name, factor_value in self.risk_factors.items():
            if factor_value < 0.9:
                if factor_value < 0.5:
                    emoji = "[ALERT]"
                elif factor_value < 0.7:
                    emoji = "[WARN]"
                else:
                    emoji = "[FAST]"
                risk_factor_lines.append(f"  {emoji} {factor_name.replace('_', ' ').title()}: {factor_value:.1%}")

        return f"""
⚙️ ENHANCED DYNAMIC RISK CONTROLLER v4.0
═══════════════════════════════════════════════════
[TARGET] Risk Status: {risk_status} ({self.current_risk_scale:.1%} scale)
[TOOL] Control Mode: {mode_status}
[STATS] Market Regime: {self.market_regime.title()}
[CRASH] Volatility Level: {self.volatility_regime.title()}
🕐 Market Session: {self.market_session.title()}

[HEALTH] SYSTEM HEALTH
• Status: {health_emoji} {self._health_status.upper()}
• Circuit Breaker: {cb_status}
• Risk Quality: {self._risk_quality:.2f}

[BALANCE] RISK SCALE CONFIGURATION
• Current Scale: {self.current_risk_scale:.1%}
• Base Scale: {self._cfg.base_risk_scale:.1%}
• Min Scale: {self._cfg.min_risk_scale:.1%}
• Max Scale: {self._cfg.max_risk_scale:.1%}
• External Scale: {self.external_risk_scale:.1%}

[STATS] ACTIVE RISK FACTORS
{chr(10).join(risk_factor_lines) if risk_factor_lines else "  [OK] All risk factors normal"}

[TOOL] CONTROLLER PERFORMANCE
• Risk Adjustments: {self.risk_adjustments_made}
• Emergency Interventions: {self.emergency_interventions}
• Consecutive Losses: {self.consecutive_losses}
• Adaptive Scaling: {'[OK] Enabled' if self.adaptive_scaling else '[FAIL] Disabled'}
• Regime Awareness: {'[OK] Enabled' if self.regime_aware else '[FAIL] Disabled'}

[CHART] ADAPTIVE PARAMETERS
• Dynamic Penalty Scaling: {self._adaptive_params['dynamic_penalty_scaling']:.2f}
• Regime Sensitivity: {self._adaptive_params['regime_sensitivity_multiplier']:.2f}
• Volatility Tolerance: {self._adaptive_params['volatility_tolerance']:.2f}
• Adaptation Confidence: {self._adaptive_params['risk_adaptation_confidence']:.2f}

🔗 EXTERNAL INTEGRATIONS
• External Signals: {len(self.external_signals)}
{chr(10).join([f"  • {name}: {value:.2f}" for name, value in self.external_signals.items()]) if self.external_signals else "  📭 No external signals"}

💡 SYSTEM STATUS
• History Tracking: {len(self.risk_scale_history)} records
• Volatility History: {len(self.vol_history)} records
• Risk Events: {len(self.risk_events)} events
• Recovery Speed: {self._cfg.recovery_speed:.1%}
        """


    def step(self, **kwargs) -> Dict[str, Any]:
        return asyncio.run(self.process(**kwargs))

    def reset(self) -> None:

        self.current_risk_scale = float(self._cfg.base_risk_scale)
        self.risk_factors = {
            "drawdown": 1.0,
            "volatility": 1.0,
            "correlation": 1.0,
            "losing_streak": 1.0,
            "market_stress": 1.0,
            "liquidity": 1.0,
            "news_sentiment": 1.0,
            "execution_quality": 1.0,
            "portfolio_concentration": 1.0,
        }


        self.vol_history.clear()
        self.dd_history.clear()
        self.risk_scale_history.clear()
        self.market_regime_history.clear()


        self.consecutive_losses = 0
        self.last_pnl = 0.0
        self.risk_adjustments_made = 0
        self.emergency_interventions = 0


        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"


        self.risk_analytics.clear()
        self.regime_performance.clear()
        self.risk_events.clear()


        self.external_risk_scale = 1.0
        self.external_signals.clear()


        self.current_mode = RiskControlMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()


        self.circuit_breaker["failures"] = 0
        self.circuit_breaker["state"] = "CLOSED"
        self._health_status = "healthy"


        self._adaptive_params = {
            "dynamic_penalty_scaling": 1.0,
            "regime_sensitivity_multiplier": 1.0,
            "volatility_tolerance": 1.0,
            "risk_adaptation_confidence": 0.5,
            "learning_momentum": 0.0,
            "emergency_threshold_adaptation": 1.0,
        }


        self._risk_quality = 0.5
        if hasattr(self, "_risk_effectiveness_history"):
            self._risk_effectiveness_history.clear()
        else:
            from collections import deque
            self._risk_effectiveness_history = deque(maxlen=50)


        self._health_status = "healthy"
        self._last_health_check = time.time()


        if hasattr(self, "_stop_event"):
            self._stop_event.clear()
        self._monitoring_active = True


        self.logger.info(
            format_operator_message(
                message="Dynamic risk controller state reset",
                icon="[RELOAD]",
                current_scale=f"{self.current_risk_scale:.2f}",
                mode=self.current_mode.value,
            )
        )
