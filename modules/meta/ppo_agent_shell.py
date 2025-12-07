#!/usr/bin/env python3
"""
PPO Agent Shell - SmartInfoBus Gateway
======================================

This module is the thin shell that wraps PPOCore and ArbiterLogic.
It handles:
- SmartInfoBus reads/writes
- Module lifecycle (BaseModule integration)
- Health monitoring
- Model persistence
- Bus signal gathering and publishing

This is the ONLY component that knows about SmartInfoBus.

Version: 3.0.0 (Multi-instrument architecture)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.core.mixins import (
    SmartInfoBusTradingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
)
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.audit_utils import RotatingLogger
from modules.utils.info_bus import InfoBusManager, SmartInfoBus
from modules.voting.core.per_instrument import (
    PerInstrumentVote,
    InstrumentProposal,
    DEFAULT_INSTRUMENTS,
)

from modules.meta.ppo_core import PPOCore, PPOCoreConfig
from modules.meta.arbiter_logic import ArbiterLogic
from modules.meta.ppo_types import (
    InstrumentDecision,
    ArbiterMultiDecision,
    MemoryGateInfo,
    RiskInfo,
    PRIMARY_INSTRUMENT,
)

from modules.meta.ppo_observation_builder import (
    PPOObservationBuilder,
    get_ppo_observation_builder,
)

from modules.meta.arbiter_logic import StrategyInfo, TradingModeInfo, WorldModelInfo


# ═══════════════════════════════════════════════════════════════════
# PPO AGENT SHELL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PPOShellConfig:
    """Configuration for PPO Agent Shell (v3.0)."""
    
    # Core config
    core_config: PPOCoreConfig = field(default_factory=PPOCoreConfig)
    
    # Instruments
    instruments: List[str] = field(default_factory=lambda: DEFAULT_INSTRUMENTS.copy())
    primary_instrument: str = PRIMARY_INSTRUMENT
    
    # Performance thresholds
    max_processing_time_ms: float = 500.0
    circuit_breaker_threshold: int = 3
    min_performance_score: float = 0.3  # interpreted as min success rate
    
    # Monitoring
    health_check_interval: float = 30.0
    
    # Debug
    debug: bool = False


# ═══════════════════════════════════════════════════════════════════
# PPO AGENT SHELL
# ═══════════════════════════════════════════════════════════════════

@module(**module_args("PPOAgent"))
class PPOAgentShell(
    BaseModule,
    SmartInfoBusTradingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
):
    """
    PPO Agent Shell - SmartInfoBus Gateway.
    
    This is a thin wrapper that:
    - Gathers signals from SmartInfoBus
    - Delegates to ArbiterLogic for decision making
    - Publishes decisions back to SmartInfoBus
    - Manages model persistence and lifecycle
    
    The shell is deliberately thin. All domain logic lives in ArbiterLogic.
    """
    
    def __init__(
        self,
        config: Optional[PPOShellConfig] = None,
        model_path: Optional[str] = None,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._cfg: PPOShellConfig = config or PPOShellConfig()
        
        # Apply genome overrides to core_config
        if genome:
            for key, val in genome.items():
                if hasattr(self._cfg.core_config, key):
                    setattr(self._cfg.core_config, key, val)
        
        # Set debug from config
        self.debug: bool = self._cfg.debug or self._cfg.core_config.debug
        
        # Initialize BaseModule
        super().__init__(**kwargs)
        
        # Internal state placeholders (assigned in setup methods)
        self.logger: RotatingLogger
        self.error_handler: Any
        self.smart_bus: SmartInfoBus
        self.ppo_core: PPOCore
        self.arbiter: ArbiterLogic
        self.obs_builder: PPOObservationBuilder
        
        self._last_observations: Dict[str, np.ndarray] = {}
        self._last_decisions: Dict[str, InstrumentDecision] = {}
        self._last_multi_decision: Optional[ArbiterMultiDecision] = None
        
        self._health_status: str = "healthy"
        self.circuit_breaker: Dict[str, Any] = {}
        self._performance_metrics: Dict[str, Any] = {}
        self._monitoring_active: bool = False
        
        # Setup components
        self._setup_logging()
        self._setup_smart_bus()
        self._setup_core_components(model_path)
        self._setup_health_tracking()
        
        # Start monitoring
        self._start_monitoring()
        
        self.logger.info(
            "[PPOAgentShell] Initialized v3.0 | "
            f"instruments={self._cfg.instruments} | "
            f"obs_size={self._cfg.core_config.obs_size}"
        )
    
    # ─────────────────────────────────────────────────────────────
    # Setup Methods
    # ─────────────────────────────────────────────────────────────
    
    def _setup_logging(self) -> None:
        """Initialize logging and error handling."""
        self.logger = RotatingLogger(
            "PPOAgentShell",
            log_path="logs/meta/ppo_agent_shell.log",
            operator_mode=True,
        )
        self.error_handler = create_error_handler("PPOAgentShell", ErrorPinpointer())
    
    def _setup_smart_bus(self) -> None:
        """Initialize SmartInfoBus connection."""
        self.smart_bus = InfoBusManager.get_instance()
    
    def _setup_core_components(self, model_path: Optional[str]) -> None:
        """Initialize PPOCore, ArbiterLogic, and observation builder."""
        # Core PPO
        self.ppo_core = PPOCore(config=self._cfg.core_config)
        
        # Load model if path provided
        if model_path:
            self.ppo_core.load(model_path)
        
        # Arbiter logic
        self.arbiter = ArbiterLogic(
            ppo_core=self.ppo_core,
            instruments=self._cfg.instruments,
            debug=self.debug,
        )
        
        # Observation builder (v2/v3 compatible)
        self.obs_builder = get_ppo_observation_builder()
        
        # Caches
        self._last_observations = {}
        self._last_decisions = {}
        self._last_multi_decision = None
    
    def _setup_health_tracking(self) -> None:
        """Initialize health tracking state and circuit breaker."""
        self._health_status = "healthy"
        
        self.circuit_breaker = {
            "state": "CLOSED",
            "failures": 0,
            "last_failure": None,
            "cooldown_until": None,  # datetime or None
        }
        
        self._performance_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "avg_processing_time_ms": 0.0,
            "last_decision_time": None,
        }
    
    def _start_monitoring(self) -> None:
        """Start background health monitoring loop."""
        
        def monitoring_loop() -> None:
            while getattr(self, "_monitoring_active", False):
                try:
                    self._update_health()
                    time.sleep(self._cfg.health_check_interval)
                except Exception as e:  # noqa: BLE001
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    # ─────────────────────────────────────────────────────────────
    # Main Process Method
    # ─────────────────────────────────────────────────────────────
    
    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """
        Main process entrypoint - handles decision making (and optional training).
        
        Workflow:
        1. Gather signals from SmartInfoBus
        2. Build observations for each instrument
        3. Delegate to ArbiterLogic for decisions
        4. Publish decisions to SmartInfoBus
        5. Handle training updates if experience provided
        """
        start_time = time.time()
        
        try:
            # 1) Gather all signals from bus
            committee_data = self._gather_committee_consensus()
            expert_signals = self._gather_expert_signals()
            memory_info = self._gather_memory_info(expert_signals)
            risk_info = self._gather_risk_info(expert_signals)
            
            # 1b) Gather strategy module signals (BiasAuditor, CurriculumPlanner, ThesisEvolution)
            strategy_info = self._gather_strategy_info()
            
            # 1c) v4.0: Gather trading mode and world model signals
            trading_mode_info = self._gather_trading_mode_info()
            world_model_info = self._gather_world_model_info()
            
            # 2) Build observations for each instrument
            observations = self._build_observations_for_instruments()
            
            # 3) Make multi-instrument decision (with full integration)
            multi_decision = self.arbiter.make_multi_instrument_decision(
                observations=observations,
                committee_data=committee_data,
                expert_signals=expert_signals,
                memory_info=memory_info,
                risk_info=risk_info,
                strategy_info=strategy_info,
                trading_mode_info=trading_mode_info,
                world_model_info=world_model_info,
            )
            
            # Cache decisions
            self._last_multi_decision = multi_decision
            self._last_observations = observations
            self._last_decisions = multi_decision.instruments
            
            # 4) Build result dict
            result = self._build_process_result(multi_decision)
            
            # Generate thesis for primary instrument
            primary_decision = multi_decision.instruments.get(self._cfg.primary_instrument)
            if primary_decision:
                thesis = self.arbiter.generate_explanation(primary_decision)
            else:
                thesis = "Multi-instrument decision made"
            result["_thesis"] = thesis
            
            # Publish to bus
            await self._publish_to_bus(multi_decision, thesis)
            
            # 5) Optional training path
            if "experience" in inputs:
                training_result = await self._process_training(inputs["experience"])
                result.update(training_result)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000.0
            self._record_success(processing_time)
            
            if self.debug:
                self.logger.debug(
                    f"[PPOAgentShell] process() completed in {processing_time:.2f}ms"
                )
            
            return result
        
        except Exception as e:  # noqa: BLE001
            return await self._handle_error(e, start_time)
    
    # ─────────────────────────────────────────────────────────────
    # Signal Gathering (SmartInfoBus reads)
    # ─────────────────────────────────────────────────────────────
    
    def _gather_committee_consensus(self) -> Dict[str, Any]:
        """Gather committee consensus from SmartInfoBus."""
        name = "PPOAgentShell"
        
        committee_decision = self.smart_bus.get("committee_decision", name) or {}
        if isinstance(committee_decision, str):
            committee_decision = {"action": committee_decision}
        
        return {
            "action": str(committee_decision.get("action", "hold")).lower(),
            "confidence": float(self.smart_bus.get("committee_confidence", name) or 0.5),
            "consensus_score": float(self.smart_bus.get("consensus_score", name) or 0.5),
            "fragility": float(self.smart_bus.get("fragility", name) or 0.5),
            "regime": self.smart_bus.get("market_regime", name) or "unknown",
            "regime_strength": float(self.smart_bus.get("regime_strength", name) or 0.5),
        }
    
    def _gather_expert_signals(self) -> Dict[str, Any]:
        """Gather expert voting, risk, and memory signals from SmartInfoBus."""
        name = "PPOAgentShell"
        
        def _expert_block(vote_key: str, conf_key: str) -> Dict[str, Any]:
            raw = self.smart_bus.get(vote_key, name) or "flat"
            conf = self.smart_bus.get(conf_key, name)
            try:
                conf_val = float(conf) if conf is not None else 0.0
            except Exception:  # noqa: BLE001
                conf_val = 0.0
            return {"proposal": raw, "confidence": conf_val}
        
        expert_signals = {
            "trend": _expert_block("TrendExpert_voting_proposal", "TrendExpert_confidence"),
            "momentum": _expert_block("MomentumExpert_voting_proposal", "MomentumExpert_confidence"),
            "theme": _expert_block("ThemeExpert_voting_proposal", "ThemeExpert_confidence"),
            "seasonality": _expert_block(
                "SeasonalityRiskExpert_voting_proposal",
                "SeasonalityRiskExpert_confidence",
            ),
        }
        
        market_context = {
            "regime": self.smart_bus.get("market_regime", name) or "unknown",
            "regime_strength": float(self.smart_bus.get("regime_strength", name) or 0.5),
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # v4.2.0: Enhanced risk signal gathering from DynamicRiskController
        # ═══════════════════════════════════════════════════════════════════
        risk_signals = {
            "risk_data": self.smart_bus.get("risk_data", name) or {},
            "portfolio_risk": self.smart_bus.get("portfolio_risk", name) or {},
            # NEW: DynamicRiskController specific outputs
            "risk_scaling": self.smart_bus.get("risk_scaling", name) or {},
            "risk_assessment": self.smart_bus.get("risk_assessment", name) or {},
            "risk_scale": self.smart_bus.get("risk_scale", name),  # Direct 0.1-1.5 scale
            "risk_level": self.smart_bus.get("risk_level", name),  # NORMAL/ELEVATED/HIGH/CRITICAL
        }
        
        # Memory signals (gate + danger zones)
        raw_memory_gate = self.smart_bus.get("memory_gate", name)
        raw_danger_zones = self.smart_bus.get("danger_zones", name)
        
        if isinstance(raw_memory_gate, dict):
            memory_gate_meta = raw_memory_gate
        else:
            try:
                memory_gate_value = float(raw_memory_gate) if raw_memory_gate else 1.0
            except Exception:  # noqa: BLE001
                memory_gate_value = 1.0
            memory_gate_meta = {
                "risk_multiplier": memory_gate_value,
                "veto": False,
                "reasons": [],
            }
        
        if isinstance(raw_danger_zones, dict):
            dz_dict = raw_danger_zones
        elif isinstance(raw_danger_zones, list):
            dz_dict = {"zones": raw_danger_zones, "zone_count": len(raw_danger_zones)}
        else:
            dz_dict = {"zones": [], "zone_count": 0}
        
        memory_signals = {
            "memory_gate": memory_gate_meta,
            "danger_zones": dz_dict,
        }
        
        return {
            "experts": expert_signals,
            "market": market_context,
            "risk": risk_signals,
            "memory": memory_signals,
        }
    
    def _gather_memory_info(self, expert_signals: Dict[str, Any]) -> MemoryGateInfo:
        """Extract normalized MemoryGateInfo from aggregated signals."""
        memory = expert_signals.get("memory", {}) or {}
        gate = memory.get("memory_gate", 1.0)
        danger = memory.get("danger_zones", {})

        # Use the canonical translator from ppo_types
        return MemoryGateInfo.from_bus_data(
            memory_gate=gate,
            danger_zones=danger,
        )

    
    def _gather_risk_info(self, expert_signals: Dict[str, Any]) -> RiskInfo:
        """
        Extract normalized global RiskInfo from aggregated signals.
        
        v4.2.0: Enhanced to pass DynamicRiskController's risk_scaling 
        and risk_assessment for full risk intelligence consumption.
        """
        risk = expert_signals.get("risk", {}) or {}
        risk_data = risk.get("risk_data", {}) or {}
        portfolio_risk = risk.get("portfolio_risk", {}) or {}
        risk_scaling = risk.get("risk_scaling", {}) or {}
        risk_assessment = risk.get("risk_assessment", {}) or {}

        # Delegate schema decoding and clamping to RiskInfo.from_bus_data
        return RiskInfo.from_bus_data(
            risk_data=risk_data,
            portfolio_risk=portfolio_risk,
            risk_scaling=risk_scaling,
            risk_assessment=risk_assessment,
            instrument="",  # global risk gate; per-instrument can be added later
        )

    def _gather_strategy_info(self) -> StrategyInfo:
        """
        Gather strategy module signals from SmartInfoBus.
        
        Integrates outputs from:
        - BiasAuditor: Psychological bias adjustments
        - CurriculumPlannerPlus: Learning stage constraints  
        - ThesisEvolutionEngine: Best thesis recommendations
        
        v3.0: Full strategy integration for intelligent trading
        """
        name = "PPOAgentShell"
        
        # BiasAuditor outputs
        bias_adjustments = self.smart_bus.get("bias_adjustments", name)
        bias_analysis = self.smart_bus.get("bias_analysis", name)
        psychological_state = self.smart_bus.get("psychological_state", name)
        
        # CurriculumPlannerPlus outputs
        curriculum_stage = self.smart_bus.get("curriculum_stage", name)
        learning_constraints = self.smart_bus.get("learning_constraints", name)
        mastery_assessment = self.smart_bus.get("mastery_assessment", name)
        
        # ThesisEvolutionEngine outputs
        best_thesis = self.smart_bus.get("best_thesis", name)
        
        # Create normalized StrategyInfo
        return StrategyInfo.from_bus_data(
            bias_adjustments=bias_adjustments if isinstance(bias_adjustments, dict) else None,
            bias_analysis=bias_analysis if isinstance(bias_analysis, dict) else None,
            psychological_state=psychological_state if isinstance(psychological_state, dict) else None,
            curriculum_stage=curriculum_stage if isinstance(curriculum_stage, dict) else None,
            learning_constraints=learning_constraints if isinstance(learning_constraints, dict) else None,
            mastery_assessment=mastery_assessment if isinstance(mastery_assessment, dict) else None,
            best_thesis=best_thesis if isinstance(best_thesis, dict) else None,
        )

    def _gather_trading_mode_info(self) -> TradingModeInfo:
        """
        Gather trading mode information from SmartInfoBus.
        
        Integrates outputs from TradingModeManager for intelligent position sizing
        and risk adjustment based on current market conditions.
        
        v4.0: Full trading mode integration
        """
        name = "PPOAgentShell"
        
        # TradingModeManager outputs
        trading_mode = self.smart_bus.get("trading_mode", name)
        mode_config = self.smart_bus.get("mode_config", name)
        mode_effectiveness = self.smart_bus.get("mode_effectiveness", name)
        decision_factors = self.smart_bus.get("decision_factors", name)
        
        # Handle scalar vs dict for mode_effectiveness
        eff_value = None
        if mode_effectiveness is not None:
            if isinstance(mode_effectiveness, dict):
                eff_value = mode_effectiveness.get("value", mode_effectiveness.get("effectiveness"))
            else:
                try:
                    eff_value = float(mode_effectiveness)
                except (TypeError, ValueError):
                    eff_value = None
        
        # Create normalized TradingModeInfo
        return TradingModeInfo.from_bus_data(
            trading_mode=trading_mode if isinstance(trading_mode, str) else None,
            mode_config=mode_config if isinstance(mode_config, dict) else None,
            mode_effectiveness=eff_value,
            decision_factors=decision_factors if isinstance(decision_factors, dict) else None,
        )

    def _gather_world_model_info(self) -> WorldModelInfo:
        """
        Gather world model predictions from SmartInfoBus.
        
        Integrates outputs from EnhancedWorldModel for predictive trading decisions
        based on LSTM price/volatility/regime forecasts.
        
        v4.0: Full world model integration
        """
        name = "PPOAgentShell"
        
        # EnhancedWorldModel outputs
        market_predictions = self.smart_bus.get("market_predictions", name)
        prediction_confidence = self.smart_bus.get("prediction_confidence", name)
        scenario_generation = self.smart_bus.get("scenario_generation", name)
        world_model_analytics = self.smart_bus.get("world_model_analytics", name)
        
        # Handle scalar vs dict for prediction_confidence
        conf_value = None
        if prediction_confidence is not None:
            if isinstance(prediction_confidence, dict):
                conf_value = prediction_confidence.get("value", prediction_confidence.get("confidence"))
            else:
                try:
                    conf_value = float(prediction_confidence)
                except (TypeError, ValueError):
                    conf_value = None
        
        # Create normalized WorldModelInfo
        return WorldModelInfo.from_bus_data(
            market_predictions=market_predictions if isinstance(market_predictions, dict) else None,
            prediction_confidence=conf_value,
            scenario_generation=scenario_generation if isinstance(scenario_generation, dict) else None,
            world_model_analytics=world_model_analytics if isinstance(world_model_analytics, dict) else None,
        )

    
    # ─────────────────────────────────────────────────────────────
    # Observation Building (v3.0)
    # ─────────────────────────────────────────────────────────────
    
    def _build_observations_for_instruments(self) -> Dict[str, np.ndarray]:
        """
        Build observation vectors for each instrument.
        
        Uses the v3.0 observation builder which supports per-instrument observations.
        If per-instrument data is unavailable, falls back to global observation.
        """
        observations: Dict[str, np.ndarray] = {}
        
        for instrument in self._cfg.instruments:
            try:
                # v3.0: Use build_for_instrument for per-instrument observations
                obs = self.obs_builder.build_for_instrument(
                    instrument=instrument,
                    smart_bus=self.smart_bus,
                    module_name="PPOAgentShell",
                )
                observations[instrument] = obs
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"Failed to build observation for {instrument}: {e}")
                observations[instrument] = np.zeros(
                    self._cfg.core_config.obs_size,
                    dtype=np.float32,
                )
        
        return observations
    
    # ─────────────────────────────────────────────────────────────
    # Result Building
    # ─────────────────────────────────────────────────────────────
    
    def _build_process_result(self, multi_decision: ArbiterMultiDecision) -> Dict[str, Any]:
        """Build the result dict from multi-instrument decision."""
        primary = multi_decision.instruments.get(self._cfg.primary_instrument)
        
        agent_perf: Dict[str, Any] = {
            "decisions_made": self._performance_metrics["total_decisions"],
            "avg_processing_time_ms": self._performance_metrics["avg_processing_time_ms"],
            "health_status": self._health_status,
            "circuit_breaker": self.circuit_breaker["state"],
        }
        
        # Build policy_actions from primary decision
        primary_obs = self._last_observations.get(self._cfg.primary_instrument)
        policy_actions: Dict[str, Any] = {
            "action": [1.0 if primary and primary.direction == "long" else -1.0 if primary and primary.direction == "short" else 0.0],
            "log_prob": 0.0,  # Not available in inference mode
            "value_estimate": primary.confidence if primary else 0.0,
            "action_std": [1.0],
            "exploration_level": 0.0,  # No exploration in inference mode
        }
        
        # Training-related outputs (empty in inference mode, populated during training)
        training_stats = getattr(self.ppo_core, "_training_stats", {})
        policy_gradients: Dict[str, Any] = {
            "gradient_norm": training_stats.get("gradient_norm", 0.0),
            "policy_loss": training_stats.get("policy_loss", 0.0),
            "value_loss": training_stats.get("value_loss", 0.0),
        }
        
        rewards: List[float] = list(getattr(self.ppo_core, "_recent_rewards", []))[-10:]
        
        training_data: Dict[str, Any] = {
            "total_steps": training_stats.get("total_steps", 0),
            "episodes": training_stats.get("episodes", 0),
            "mode": "inference",
        }
        
        training_metrics: Dict[str, Any] = {
            "avg_reward": training_stats.get("avg_reward", 0.0),
            "avg_episode_length": training_stats.get("avg_episode_length", 0),
            "explained_variance": training_stats.get("explained_variance", 0.0),
        }
        
        training_signals: Dict[str, Any] = {
            "gradient_norm": training_stats.get("gradient_norm", 0.0),
            "explained_variance": training_stats.get("explained_variance", 0.0),
            "policy_loss": training_stats.get("policy_loss", 0.0),
            "value_loss": training_stats.get("value_loss", 0.0),
        }

        if primary:
            result: Dict[str, Any] = {
                # Primary decision (for backward compatibility)
                "ppo_final_decision": primary.to_dict(),
                "ppo_gate_passed": primary.gate_passed,
                "ppo_position_size": primary.position_size,
                
                # Multi-instrument data
                "ppo_multi_decision": multi_decision.to_dict(),
                "ppo_instrument_stats": self.arbiter.get_instrument_stats(),

                # Agent performance diagnostics
                "agent_performance": agent_perf,
                
                # Legacy voting outputs
                "PPOAgent_voting_proposal": self._build_voting_payload(multi_decision),
                "PPOAgent_confidence": primary.confidence,
                
                # Contract-required training outputs (even if empty in inference mode)
                "policy_actions": policy_actions,
                "policy_gradients": policy_gradients,
                "rewards": rewards,
                "training_data": training_data,
                "training_metrics": training_metrics,
                "training_signals": training_signals,
            }
        else:
            result = {
                "ppo_final_decision": {
                    "direction": "hold",
                    "confidence": 0.0,
                    "gate_passed": False,
                },
                "ppo_gate_passed": False,
                "ppo_position_size": 0.0,
                "ppo_multi_decision": multi_decision.to_dict(),
                "ppo_instrument_stats": self.arbiter.get_instrument_stats(),
                "agent_performance": agent_perf,
                "PPOAgent_voting_proposal": {},
                "PPOAgent_confidence": 0.0,
                # Contract-required training outputs
                "policy_actions": policy_actions,
                "policy_gradients": policy_gradients,
                "rewards": rewards,
                "training_data": training_data,
                "training_metrics": training_metrics,
                "training_signals": training_signals,
            }
        
        return result
    
    def _build_voting_payload(self, multi_decision: ArbiterMultiDecision) -> Dict[str, Any]:
        """Build legacy voting payload from multi-decision."""
        vote = PerInstrumentVote(member="PPOAgent")
        
        for instrument, decision in multi_decision.instruments.items():
            proposal = InstrumentProposal(
                instrument=instrument,
                action=decision.direction,
                confidence=decision.confidence,
                magnitude=decision.position_size,
                horizon="intraday",
                rationale=decision.reasoning,
                meta=decision.meta,
            )
            vote.set_proposal(proposal)
        
        return vote.to_dict()
    
    # ─────────────────────────────────────────────────────────────
    # Bus Publishing
    # ─────────────────────────────────────────────────────────────
    
    async def _publish_to_bus(
        self,
        multi_decision: ArbiterMultiDecision,
        thesis: str,
    ) -> None:
        """Publish decisions and metrics to SmartInfoBus."""
        try:
            # Primary decision
            primary = multi_decision.instruments.get(self._cfg.primary_instrument)
            if primary:
                self.smart_bus.set(
                    "ppo_final_decision",
                    primary.to_dict(),
                    module="PPOAgent",
                    thesis=thesis,
                )
                
                self.smart_bus.set(
                    "ppo_gate_passed",
                    primary.gate_passed,
                    module="PPOAgent",
                    thesis=f"Gate {'PASS' if primary.gate_passed else 'BLOCK'}",
                )
                
                self.smart_bus.set(
                    "ppo_position_size",
                    primary.position_size,
                    module="PPOAgent",
                    thesis=f"Position size: {primary.position_size:.2%}",
                )
            
            # Multi-instrument data
            self.smart_bus.set(
                "ppo_multi_decision",
                multi_decision.to_dict(),
                module="PPOAgent",
                thesis="Multi-instrument decision payload",
            )
            
            # Per-instrument stats
            self.smart_bus.set(
                "ppo_instrument_stats",
                self.arbiter.get_instrument_stats(),
                module="PPOAgent",
                thesis="Per-instrument statistics",
            )
            
            # Legacy voting
            self.smart_bus.set(
                "PPOAgent_voting_proposal",
                self._build_voting_payload(multi_decision),
                module="PPOAgent",
                thesis=thesis,
            )
            
            if primary:
                self.smart_bus.set(
                    "PPOAgent_confidence",
                    primary.confidence,
                    module="PPOAgent",
                    thesis=f"Confidence: {primary.confidence:.2f}",
                )
            
            # Agent performance
            self.smart_bus.set(
                "agent_performance",
                {
                    "decisions_made": self._performance_metrics["total_decisions"],
                    "avg_processing_time_ms": self._performance_metrics["avg_processing_time_ms"],
                    "health_status": self._health_status,
                    "circuit_breaker": self.circuit_breaker["state"],
                },
                module="PPOAgent",
                thesis="Agent performance metrics",
            )
        
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to publish to bus: {e}")
    
    # ─────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────
    
    async def _process_training(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Process training experience through PPOCore (optional path)."""
        try:
            obs = experience.get("observation")
            action = experience.get("action")
            reward = experience.get("reward")
            done = experience.get("done", False)
            log_prob = experience.get("log_prob")
            value = experience.get("value")
            
            if obs is not None and action is not None and reward is not None:
                # Record step
                self.ppo_core.record_step(
                    obs=np.array(obs, dtype=np.float32),
                    action=np.array(action, dtype=np.float32),
                    reward=float(reward),
                    done=bool(done),
                    log_prob=float(log_prob) if log_prob is not None else None,
                    value=float(value) if value is not None else None,
                )
                
                # Try update
                update_result = self.ppo_core.update()
                
                return {
                    "training_updated": update_result is not None,
                    "training_stats": update_result or {},
                }
            
            return {"training_updated": False}
        
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Training processing failed: {e}")
            return {"training_updated": False, "training_error": str(e)}
    
    # ─────────────────────────────────────────────────────────────
    # Health & Error Handling
    # ─────────────────────────────────────────────────────────────
    
    def _record_success(self, processing_time_ms: float) -> None:
        """Record successful decision and update rolling performance metrics."""
        self._performance_metrics["total_decisions"] += 1
        self._performance_metrics["successful_decisions"] += 1
        self._performance_metrics["last_decision_time"] = datetime.now().isoformat()
        
        # Rolling average
        n = self._performance_metrics["total_decisions"]
        old_avg = self._performance_metrics["avg_processing_time_ms"]
        self._performance_metrics["avg_processing_time_ms"] = (
            old_avg * (n - 1) + processing_time_ms
        ) / max(n, 1)
        
        # Reset circuit breaker on success in HALF_OPEN
        if self.circuit_breaker["state"] == "HALF_OPEN":
            self.circuit_breaker["state"] = "CLOSED"
            self.circuit_breaker["failures"] = 0
    
    def _update_health(self) -> None:
        """Update health status based on circuit breaker and performance."""
        now = datetime.now()
        
        # Circuit breaker cooldown
        if self.circuit_breaker["state"] == "OPEN":
            cooldown_until = self.circuit_breaker.get("cooldown_until")
            if isinstance(cooldown_until, datetime) and now >= cooldown_until:
                self.circuit_breaker["state"] = "HALF_OPEN"
        
        # Base health from breaker
        if self.circuit_breaker["state"] == "OPEN":
            status = "degraded"
        elif self.circuit_breaker["failures"] > 0:
            status = "warning"
        else:
            status = "healthy"
        
        # Performance-based adjustment (simple success-rate heuristic)
        total = self._performance_metrics["total_decisions"]
        successes = self._performance_metrics["successful_decisions"]
        avg_ms = self._performance_metrics["avg_processing_time_ms"]
        
        success_rate = successes / total if total > 0 else 1.0
        
        if success_rate < self._cfg.min_performance_score:
            status = "warning"
        if avg_ms > self._cfg.max_processing_time_ms * 2:
            status = "degraded"
        
        self._health_status = status
    
    async def _handle_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing error, update breaker, and return safe fallback."""
        processing_time = (time.time() - start_time) * 1000.0
        
        self.logger.error(f"[PPOAgentShell] Error: {error}")
        
        # Count failure
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = datetime.now().isoformat()
        
        # Circuit breaker OPEN logic
        if self.circuit_breaker["failures"] >= self._cfg.circuit_breaker_threshold:
            self.circuit_breaker["state"] = "OPEN"
            self.circuit_breaker["cooldown_until"] = datetime.now() + timedelta(seconds=60)
        
        # Performance metrics still see this processing time (as implicit load)
        total = self._performance_metrics["total_decisions"] + 1
        old_avg = self._performance_metrics["avg_processing_time_ms"]
        self._performance_metrics["avg_processing_time_ms"] = (
            old_avg * self._performance_metrics["total_decisions"] + processing_time
        ) / max(total, 1)
        self._performance_metrics["total_decisions"] = total
        
        # Safe fallback decision
        fallback_multi: Dict[str, Any] = {}
        if self._last_multi_decision is not None:
            fallback_multi = self._last_multi_decision.to_dict()
        
        return {
            "ppo_final_decision": {
                "direction": "hold",
                "confidence": 0.0,
                "reasoning": f"Error: {error}",
                "gate_passed": False,
            },
            "ppo_gate_passed": False,
            "ppo_position_size": 0.0,
            "_thesis": f"Error in PPOAgentShell: {error}",
            "ppo_multi_decision": fallback_multi,
            "ppo_instrument_stats": self.arbiter.get_instrument_stats(),
            "agent_performance": {
                "decisions_made": self._performance_metrics["total_decisions"],
                "avg_processing_time_ms": self._performance_metrics["avg_processing_time_ms"],
                "health_status": self._health_status,
                "circuit_breaker": self.circuit_breaker["state"],
            },
            "PPOAgent_voting_proposal": {},
            "PPOAgent_confidence": 0.0,
        }
    
    # ─────────────────────────────────────────────────────────────
    # Model Persistence
    # ─────────────────────────────────────────────────────────────
    
    def save_model(self, path: str) -> None:
        """Save PPO model to disk."""
        self.ppo_core.save(path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> bool:
        """Load PPO model from disk."""
        try:
            self.ppo_core.load(path)
            self.logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    # ─────────────────────────────────────────────────────────────
    # Properties and Lifecycle
    # ─────────────────────────────────────────────────────────────
    
    @property
    def network(self) -> Any:
        """Access underlying network (for compatibility)."""
        return self.ppo_core.network
    
    @property
    def core_config(self) -> PPOCoreConfig:
        """Access core PPO configuration."""
        return self._cfg.core_config
    
    def _initialize(self) -> None:
        """Initialize module (called by BaseModule)."""
        try:
            if hasattr(self, "smart_bus"):
                self.smart_bus.set(
                    "agent_performance",
                    {
                        "decisions_made": 0,
                        "avg_processing_time_ms": 0.0,
                        "health_status": "healthy",
                    },
                    module="PPOAgent",
                    thesis="Initial PPOAgentShell status",
                )
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Initialization failed: {e}")
    
    def cleanup(self) -> None:
        """Cleanup resources and stop monitoring."""
        self._monitoring_active = False
        self.logger.info("[PPOAgentShell] Cleanup completed")
