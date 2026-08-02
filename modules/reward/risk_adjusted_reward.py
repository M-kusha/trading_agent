
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from typing import Any, Dict, List, Optional

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

from .components.adaptation_manager import AdaptationManager
from .components.analytics_engine import RewardAnalyticsEngine
from .components.data_extractor import RewardDataExtractor
from .components.reward_calculator import RewardCalculator
from .debug.reward_debug_manager import RewardDebugManager
from .shared.reward_config import RewardConfig, RewardMode
from .shared.reward_state import RewardState
from .shared.utils import RewardUtils


@module(**module_args(
    "RiskAdjustedReward",
    description="Risk-adjusted reward system with comprehensive debugging and analytics",
    error_handling=True,
    hot_reload=True,
    timeout_ms=30000,
))
class RiskAdjustedReward(
    BaseModule,
    SmartInfoBusTradingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin
):


    def __init__(
        self,
        config: Optional[RewardConfig | Dict[str, Any]] = None,
        genome: Optional[Dict[str, Any]] = None,
        env: Any = None,
        debug: bool = False,
        debug_level: str = "INFO",
        **kwargs,
    ):

        self.logger = RotatingLogger(
            name="RiskAdjustedReward",
            log_path="logs/reward/risk_adjusted_reward.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.smart_bus = InfoBusManager.get_instance()
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("RiskAdjustedReward", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()


        if isinstance(config, dict):
            self.cfg = RewardConfig(**config)
        elif config is None:
            self.cfg = RewardConfig()
        else:
            self.cfg = config


        self.env = env


        self.state = RewardState(config=self.cfg)


        self.debug_manager = RewardDebugManager(
            enabled=debug,
            level=debug_level,
            logger=self.logger,
            smart_bus=self.smart_bus
        )


        try:
            if debug and str(debug_level).upper() == "TRACE":
                self.debug_manager.enable_memory_tracking()
        except Exception:
            pass


        self._initialize_components()


        self._initialize_genome_parameters(genome)


        self._process_lock: asyncio.Lock = asyncio.Lock()
        self._state_lock: threading.RLock = threading.RLock()
        self._stop_event: threading.Event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None


        self._last_bus_payloads: Dict[str, str] = {}
        self._bus_tick_count: int = 0
        self._analytics_interval: int = 3


        self._schemas = {
            "shaped_reward": {"reward", "components", "calculation_method", "timestamp"},
            "reward_analytics": {"performance_metrics", "component_analysis", "regime_analysis"},
            "reward_performance": {
                "reward_quality","sharpe_ratio","consistency_score","win_rate",
                "avg_reward","reward_volatility","adaptive_params","health_status",
                "circuit_breaker_state"
            },

        }


        super().__init__(config=asdict(self.cfg))


        self.logger.info(
            format_operator_message(
                "🎯",
                "REWARD_SYSTEM_INITIALIZED",
                details=f"Debug={debug}, Level={debug_level}",
                result="Reward system ready with separated components (hardened)",
                context="reward_initialization",
            )
        )


        self._start_monitoring()

    def __del__(self):

        try:
            self.stop_monitoring()
        except Exception:
            pass

    def _initialize_components(self) -> None:
        self.data_extractor = RewardDataExtractor(
            smart_bus=self.smart_bus,
            logger=self.logger,
            debug_manager=self.debug_manager,
            env=self.env
        )
        self.calculator = RewardCalculator(
            config=self.cfg,
            state=self.state,
            logger=self.logger,
            debug_manager=self.debug_manager
        )
        self.analytics_engine = RewardAnalyticsEngine(
            state=self.state,
            logger=self.logger,
            debug_manager=self.debug_manager
        )
        self.adaptation_manager = AdaptationManager(
            config=self.cfg,
            state=self.state,
            logger=self.logger,
            debug_manager=self.debug_manager
        )
        self.utils = RewardUtils()


    def _initialize(self) -> None:
        try:
            self._initialize_bus_values()
            if self.debug_manager.enabled:
                with self._safe_debug():
                    self.debug_manager.log_initialization_state()
        except Exception as e:
            self.logger.error(f"Reward system initialization failed: {e}")
            if self.debug_manager.enabled:
                with self._safe_debug():
                    self.debug_manager.log_error("INITIALIZATION_FAILED", e)

    def _initialize_bus_values(self) -> None:
        shaped_baseline = {
            "reward": 0.0,
            "components": {"init": True, "reason": "baseline"},
            "calculation_method": "baseline",
            "timestamp": self.utils.utcnow(),
        }
        self._bus_set_if_changed(
            "shaped_reward",
            shaped_baseline,
            "Initial shaped_reward baseline",
        )

        self._bus_set_if_changed(
            "reward_components",
            {"init": True, "reason": "baseline"},
            "Initial reward components baseline",
        )

        analytics_baseline = self.analytics_engine.get_baseline_analytics()
        self._bus_set_if_changed(
            "reward_analytics",
            analytics_baseline,
            "Initial reward analytics baseline",
        )

        performance_baseline = self.state.get_performance_metrics()
        self._bus_set_if_changed(
            "reward_performance",
            performance_baseline,
            "Initial reward performance metrics",
        )


    def _start_monitoring(self) -> None:
        def monitoring_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    with self._state_lock:

                        with self._time_block("monitoring.update_health"):
                            self._update_health_status_locked()

                        with self._time_block("monitoring.analytics"):
                            self.analytics_engine.analyze_effectiveness()

                        with self._time_block("monitoring.adapt"):
                            self.adaptation_manager.adapt_parameters()

                    if self.debug_manager.enabled:
                        with self._safe_debug():
                            self.debug_manager.log_monitoring_cycle()
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    if self.debug_manager.enabled:
                        with self._safe_debug():
                            self.debug_manager.log_error("MONITORING_ERROR", e)

                self._stop_event.wait(1)

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        if self.debug_manager.enabled:
            with self._safe_debug():
                self.debug_manager.log_shutdown()


    async def process(self, **inputs) -> Dict[str, Any]:
        async with self._process_lock:
            start_time = time.time()
            budget_s = getattr(self.cfg, "max_processing_time_ms", 30000) / 1000.0

            try:
                if self.debug_manager.enabled:
                    with self._safe_debug():
                        self.debug_manager.log_process_start(inputs)


                if hasattr(asyncio, "timeout"):
                    async with asyncio.timeout(budget_s):
                        self.logger.info("process: before extract_reward_data")
                        with self._time_block("extract_reward_data"):
                            start_extract = time.time()
                            reward_data = await self.data_extractor.extract_reward_data(**inputs)
                            self.logger.info(f"extract_reward_data took {time.time() - start_extract} seconds")
                        self.logger.info("process: after extract_reward_data")

                        if self.debug_manager.enabled:
                            with self._safe_debug():
                                self.debug_manager.log_extracted_data(reward_data)

                        if not reward_data or reward_data.get("data_quality") == "invalid":
                            return await self._handle_no_data_fallback(reward_data)


                        self.logger.info("process: before calculate_enhanced_reward")
                        with self._memory_block("calc_enhanced_reward"), self._time_block("calc_enhanced_reward"):
                            start_calc = time.time()
                            reward_result = await self.calculator.calculate_enhanced_reward(reward_data)
                            self.logger.info(f"calculate_enhanced_reward took {time.time() - start_calc} seconds")
                        self.logger.info("process: after calculate_enhanced_reward")

                        if self.debug_manager.enabled:
                            with self._safe_debug():
                                self.debug_manager.log_calculation_result(reward_result)

                        self.logger.info("process: before update_analytics")
                        with self._time_block("analytics.update"):
                            start_analytics = time.time()
                            analytics_result = await self.analytics_engine.update_analytics(reward_result, reward_data)
                            self.logger.info(f"update_analytics took {time.time() - start_analytics} seconds")
                        self.logger.info("process: after update_analytics")

                        self.logger.info("process: before update_adaptive_learning")
                        with self._time_block("adaptation.update"):
                            start_adapt = time.time()
                            adaptation_result = await self.adaptation_manager.update_adaptive_learning(reward_result)
                            self.logger.info(f"update_adaptive_learning took {time.time() - start_adapt} seconds")
                        self.logger.info("process: after update_adaptive_learning")
                else:

                    self.logger.info("process: before extract_reward_data")
                    with self._time_block("extract_reward_data"):
                        start_extract = time.time()
                        reward_data = await self.data_extractor.extract_reward_data(**inputs)
                        self.logger.info(f"extract_reward_data took {time.time() - start_extract} seconds")
                    self.logger.info("process: after extract_reward_data")

                    if self.debug_manager.enabled:
                        with self._safe_debug():
                            self.debug_manager.log_extracted_data(reward_data)

                    if not reward_data or reward_data.get("data_quality") == "invalid":
                        return await self._handle_no_data_fallback(reward_data)


                    self.logger.info("process: before calculate_enhanced_reward")
                    with self._memory_block("calc_enhanced_reward"), self._time_block("calc_enhanced_reward"):
                        start_calc = time.time()
                        reward_result = await self.calculator.calculate_enhanced_reward(reward_data)
                        self.logger.info(f"calculate_enhanced_reward took {time.time() - start_calc} seconds")
                    self.logger.info("process: after calculate_enhanced_reward")

                    if self.debug_manager.enabled:
                        with self._safe_debug():
                            self.debug_manager.log_calculation_result(reward_result)

                    self.logger.info("process: before update_analytics")
                    with self._time_block("analytics.update"):
                        start_analytics = time.time()
                        analytics_result = await self.analytics_engine.update_analytics(reward_result, reward_data)
                        self.logger.info(f"update_analytics took {time.time() - start_analytics} seconds")
                    self.logger.info("process: after update_analytics")

                    self.logger.info("process: before update_adaptive_learning")
                    with self._time_block("adaptation.update"):
                        start_adapt = time.time()
                        adaptation_result = await self.adaptation_manager.update_adaptive_learning(reward_result)
                        self.logger.info(f"update_adaptive_learning took {time.time() - start_adapt} seconds")
                    self.logger.info("process: after update_adaptive_learning")

                result = {**reward_result, **analytics_result, **adaptation_result}

                self.logger.info("process: before _generate_reward_thesis")
                with self._time_block("thesis.generate"):
                    thesis = await self._generate_reward_thesis(reward_data, result)
                self.logger.info("process: after _generate_reward_thesis")

                output = self._prepare_output(result, thesis)

                self.logger.info("process: before _update_smart_bus")
                with self._time_block("bus.update"):
                    await self._update_smart_bus(result, thesis)
                self.logger.info("process: after _update_smart_bus")

                self._record_success((time.time() - start_time) * 1000.0)

                if self.debug_manager.enabled:
                    with self._safe_debug():
                        self.debug_manager.log_process_complete(output, time.time() - start_time)

                return output

            except asyncio.TimeoutError as e:
                if self.debug_manager.enabled:
                    with self._safe_debug():
                        self.debug_manager.log_error("PROCESS_TIMEOUT", e, inputs)
                return await self._handle_reward_error(TimeoutError("process timeout"), start_time)
            except Exception as e:
                if self.debug_manager.enabled:
                    with self._safe_debug():
                        self.debug_manager.log_error("PROCESS_ERROR", e, inputs)
                return await self._handle_reward_error(e, start_time)

    def _prepare_output(self, result: Dict[str, Any], thesis: str) -> Dict[str, Any]:
        shaped_payload = {
            "reward": result.get("shaped_reward", 0.0),
            "components": result.get("reward_components", {}),
            "calculation_method": result.get("calculation_method", "enhanced_async"),
            "timestamp": self.utils.utcnow(),
        }
        analytics_payload = result.get("reward_analytics", self.analytics_engine.get_baseline_analytics())
        performance_payload = self.state.get_performance_metrics()


        self._validate_schema("shaped_reward", shaped_payload)
        if isinstance(analytics_payload, dict):
            self._validate_schema("reward_analytics", analytics_payload)
        self._validate_schema("reward_performance", performance_payload)

        return {
            "shaped_reward": shaped_payload,
            "reward_components": result.get("reward_components", {}),
            "reward_analytics": analytics_payload,
            "reward_performance": performance_payload,
            "_thesis": thesis,
            "success": True,
        }


    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        try:
            base_confidence = self.state.calculate_base_confidence()


            if self.state.circuit_breaker["state"] == "OPEN":
                base_confidence *= 0.5
            elif self.state.circuit_breaker["failures"] > 0:
                base_confidence *= 0.8


            if self.state.current_mode == RewardMode.EMERGENCY:
                base_confidence *= 0.7
            elif self.state.current_mode == RewardMode.LIVE_TRADING:
                base_confidence += 0.1

            if self.debug_manager.enabled:
                with self._safe_debug():
                    self.debug_manager.log_confidence_calculation(base_confidence, action, inputs)

            return float(np.clip(base_confidence, 0.1, 1.0))
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.6

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        try:
            recommendations = self.analytics_engine.generate_recommendations()
            state_summary = self.state.get_state_summary()
            proposal = {
                "action_type": "reward_optimization",
                "current_reward": state_summary["last_reward"],
                "reward_trend": state_summary["reward_trend"],
                "recommendations": recommendations,
                "sharpe_ratio": state_summary["sharpe_ratio"],
                "reward_quality": state_summary["reward_quality"],
                "confidence": await self.calculate_confidence({}, **inputs),
                "timestamp": self.utils.utcnow(),
                "system_health": {
                    "circuit_breaker": self.state.circuit_breaker["state"],
                    "mode": self.state.current_mode.value,
                    "health_status": self.state.health_status,
                },
            }
            if self.debug_manager.enabled:
                with self._safe_debug():
                    self.debug_manager.log_action_proposal(proposal)
            return proposal
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                "action_type": "reward_optimization",
                "error": str(e),
                "recommendations": [{
                    "action": "system_check",
                    "reason": "Error in reward analysis",
                    "priority": "high"
                }],
                "confidence": 0.1,
            }


    async def _handle_no_data_fallback(
        self,
        reward_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self.logger.warning("No valid reward data - using fallback")
        if self.debug_manager.enabled:
            try:
                self.debug_manager.increment_fallback()
            except Exception:
                pass
            if reward_data:
                with self._safe_debug():
                    self.debug_manager.log_missing_data(reward_data)

        fallback_reward = -0.1
        thesis = "No valid reward data - fallback applied"
        return self._prepare_output(
            {
                "shaped_reward": fallback_reward,
                "reward_components": {
                    "fallback_penalty": -0.1,
                    "reason": "no_valid_data",
                    "missing_keys": reward_data.get("missing_keys", []) if isinstance(reward_data, dict) else []
                },
                "calculation_method": "fallback"
            },
            thesis
        )

    async def _handle_reward_error(
        self,
        error: Exception,
        start_time: float
    ) -> Dict[str, Any]:
        processing_time = (time.time() - start_time) * 1000.0
        self.state.record_failure()


        try:
            self.performance_tracker.record_metric(
                'RiskAdjustedReward', 'reward_calculation', float(processing_time), False
            )
        except Exception as e:
            self.logger.debug(f"Performance tracking (failure) failed: {e}")

        explanation = self.english_explainer.explain_error(
            "RiskAdjustedReward", str(error), "reward calculation"
        )
        self.logger.error(
            format_operator_message(
                "❌",
                "REWARD_CALCULATION_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                circuit_breaker_state=self.state.circuit_breaker["state"],
                context="reward_error",
            )
        )
        if self.debug_manager.enabled:
            with self._safe_debug():
                self.debug_manager.log_error_details(error, processing_time)

        return self._create_error_fallback_response(str(error))

    def _create_error_fallback_response(self, reason: str) -> Dict[str, Any]:
        if self.debug_manager.enabled:
            try:
                self.debug_manager.increment_fallback()
            except Exception:
                pass

        error_reward = -0.5 if self.state.circuit_breaker["state"] == "OPEN" else -0.2
        thesis = f"Reward error fallback: {reason}"
        return self._prepare_output(
            {
                "shaped_reward": error_reward,
                "reward_components": {
                    "error_penalty": error_reward,
                    "reason": reason
                },
                "calculation_method": "error_fallback"
            },
            thesis
        )


    async def _update_smart_bus(self, result: Dict[str, Any], thesis: str) -> None:
        try:
            self._bus_tick_count += 1

            with self._time_block("bus.shaped_reward"):
                reward_data = {
                    "reward": result.get("shaped_reward", 0.0),
                    "components": result.get("reward_components", {}),
                    "calculation_method": result.get("calculation_method", "enhanced_async"),
                    "timestamp": self.utils.utcnow(),
                }
                self._bus_set_if_changed("shaped_reward", reward_data, thesis)

            with self._time_block("bus.components"):
                self._bus_set_if_changed(
                    "reward_components", result.get("reward_components", {}), "Reward components breakdown"
                )


            should_push_analytics = (self._bus_tick_count % self._analytics_interval == 1)
            analytics_payload = result.get("reward_analytics", {})
            perf_payload = self.state.get_performance_metrics()

            if should_push_analytics or self._payload_changed("reward_analytics", analytics_payload):
                with self._time_block("bus.analytics"):
                    self._bus_set_if_changed("reward_analytics", analytics_payload, "Reward analytics & effectiveness")

            if should_push_analytics or self._payload_changed("reward_performance", perf_payload):
                with self._time_block("bus.performance"):
                    self._bus_set_if_changed("reward_performance", perf_payload, "Real-time reward performance")

            if self.debug_manager.enabled:
                with self._safe_debug():
                    self.debug_manager.log_bus_updates([
                        "shaped_reward", "reward_components", "reward_analytics", "reward_performance"
                    ])

        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")


    async def _generate_reward_thesis(
        self,
        reward_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> str:
        try:
            return self.utils.generate_thesis(reward_data, result, self.state)
        except Exception as e:
            return f"Reward calculation completed (thesis generation failed: {e})"

    def _update_health_status(self) -> None:

        with self._state_lock:
            self._update_health_status_locked()

    def _update_health_status_locked(self) -> None:
        try:
            self.state.update_health_status()
            if self.debug_manager.enabled:
                with self._safe_debug():
                    self.debug_manager.log_health_status(self.state.get_health_status())
        except Exception as e:
            self.logger.error(f"Health status update failed: {e}")

    def _record_success(self, processing_time: float) -> None:
        self.state.record_success()
        try:
            self.performance_tracker.record_metric(
                'RiskAdjustedReward',
                'reward_calculation',
                float(processing_time),
                True
            )
        except Exception as e:
            self.logger.debug(f"Performance tracking failed: {e}")

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]) -> None:
        if genome:
            self.state.apply_genome(genome)
            self.genome = genome.copy()
        else:
            self.genome = self.state.get_genome()


    def reset(self) -> None:
        with self._state_lock:
            self.state.reset()
            self.analytics_engine.reset()
            self.adaptation_manager.reset()

            if hasattr(self, 'calculator') and self.calculator:
                self.calculator.reset()
        if self.debug_manager.enabled:
            with self._safe_debug():
                self.debug_manager.log_reset()

    def get_observation_components(self) -> np.ndarray:
        return self.state.get_observation_components()

    def get_state(self) -> Dict[str, Any]:
        state_dict = {
            "state": self.state.get_state(),
            "analytics": self.analytics_engine.get_state(),
            "adaptation": self.adaptation_manager.get_state(),
            "debug": self.debug_manager.get_statistics() if self.debug_manager.enabled else {}
        }
        if self.debug_manager.enabled:
            with self._safe_debug():
                self.debug_manager.log_state_retrieval(state_dict)
        return state_dict

    def set_state(self, state: Dict[str, Any]) -> None:
        with self._state_lock:
            if "state" in state:
                self.state.set_state(state["state"])
            if "analytics" in state:
                self.analytics_engine.set_state(state["analytics"])
            if "adaptation" in state:
                self.adaptation_manager.set_state(state["adaptation"])
        if self.debug_manager.enabled:
            with self._safe_debug():
                self.debug_manager.log_state_setting(state)

    def get_health_status(self) -> Dict[str, Any]:
        return self.state.get_health_status()

    def get_weights(self) -> Dict[str, Any]:

        if hasattr(self.cfg, "get_weights") and callable(self.cfg.get_weights):
            return self.cfg.get_weights()
        return {
            "regime_weights": getattr(self.cfg, "regime_weights", [0.3, 0.4, 0.3]),
            "dd_pen_weight": getattr(self.cfg, "dd_pen_weight", 2.0),
            "risk_pen_weight": getattr(self.cfg, "risk_pen_weight", 0.1),
            "tail_pen_weight": getattr(self.cfg, "tail_pen_weight", 0.5),
            "mistake_pen_weight": getattr(self.cfg, "mistake_pen_weight", 0.3),
            "no_trade_penalty_weight": getattr(self.cfg, "no_trade_penalty_weight", 0.05),
            "win_bonus_weight": getattr(self.cfg, "win_bonus_weight", 1.0),
            "consistency_bonus_weight": getattr(self.cfg, "consistency_bonus_weight", 0.5),
            "sharpe_bonus_weight": getattr(self.cfg, "sharpe_bonus_weight", 0.3),
            "trade_frequency_bonus": getattr(self.cfg, "trade_frequency_bonus", 0.2),
            "volatility_adjustment": getattr(self.cfg, "volatility_adjustment", 1.0),
            "regime_bonus_weight": getattr(self.cfg, "regime_bonus_weight", 0.2),
            "momentum_bonus_weight": getattr(self.cfg, "momentum_bonus_weight", 0.1),
        }

    def get_genome(self) -> Dict[str, Any]:
        return self.genome.copy()

    def set_genome(self, genome: Dict[str, Any]) -> None:
        self.genome = genome.copy()
        self.state.apply_genome(genome)

    def mutate(self, mutation_rate: float = 0.2) -> None:
        self.state.mutate_genome(mutation_rate)
        self.genome = self.state.get_genome()

    def get_audit_trail(self, n: int = 20) -> List[Dict[str, Any]]:
        return self.state.get_audit_trail(n)

    def get_reward_system_report(self) -> str:
        report = self.utils.generate_system_report(
            self.state,
            self.analytics_engine,
            self.adaptation_manager,
            self.cfg
        )
        if self.debug_manager.enabled:
            report += "\n" + self.debug_manager.get_debug_report()
        return report


    def _payload_changed(self, key: str, payload: Any) -> bool:
        try:
            blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except Exception:
            blob = str(payload)
        h = hashlib.sha256(blob.encode()).hexdigest()
        return self._last_bus_payloads.get(key) != h

    def _bus_set_if_changed(self, key: str, payload: Any, thesis: str) -> None:
        try:
            try:
                blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            except Exception:
                blob = str(payload)
            h = hashlib.sha256(blob.encode()).hexdigest()
            if self._last_bus_payloads.get(key) != h:
                self.smart_bus.set(key, payload, module="RiskAdjustedReward", thesis=thesis)
                self._last_bus_payloads[key] = h
        except Exception as e:
            self.logger.error(f"Bus update failed for {key}: {e}")

    def _validate_schema(self, key: str, payload: Dict[str, Any]) -> None:
        spec = self._schemas.get(key)
        if not spec:
            return
        try:
            missing = spec - set(payload.keys())
            if missing:
                self.logger.warning(f"{key} missing fields: {missing}")
        except Exception:

            pass


    def _time_block(self, name: str):
        cm = getattr(self.debug_manager, "time_block", None)
        if cm and self.debug_manager.enabled:
            return cm(name)
        return nullcontext()

    def _memory_block(self, name: str):
        cm = getattr(self.debug_manager, "memory_block", None)
        if cm and self.debug_manager.enabled:
            return cm(name)
        return nullcontext()

    @contextmanager
    def _safe_debug(self):
        try:
            yield
        except Exception as e:

            try:
                self.logger.debug(f"Debug hook failure: {e}")
            except Exception:
                pass
