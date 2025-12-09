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

Version: 3.1.0 (Multi-instrument architecture, position-focus aware)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Mapping

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

from modules.meta.ppo_core import PPOCore, PPOCoreConfig
from modules.meta.arbiter_logic import ArbiterLogic
from modules.meta.ppo_types import (
    InstrumentDecision,
    ArbiterMultiDecision,
    MemoryGateInfo,
    RiskInfo,
    PRIMARY_INSTRUMENT,
    DEFAULT_INSTRUMENTS,
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
    """Configuration for PPO Agent Shell (v3.x)."""

    # Core PPO configuration
    core_config: PPOCoreConfig = field(default_factory=PPOCoreConfig)

    # Instruments: should be aligned with DEFAULT_INSTRUMENTS in ppo_types
    instruments: List[str] = field(default_factory=lambda: DEFAULT_INSTRUMENTS.copy())
    primary_instrument: str = PRIMARY_INSTRUMENT

    # Performance thresholds
    max_processing_time_ms: float = 500.0
    circuit_breaker_threshold: int = 3
    min_performance_score: float = 0.3  # interpreted as min success rate

    # Monitoring
    health_check_interval: float = 30.0
    
    # Position focus mode settings
    # When True, having a position in one instrument blocks NEW entries in OTHER instruments
    # When False, each instrument is evaluated independently (multi-position allowed)
    position_focus_blocks_other_instruments: bool = False

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

    All domain logic lives in ArbiterLogic; this shell is glue code.
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

        # Debug flag (core_config.debug can also enable it)
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
            "[PPOAgentShell] Initialized v3.1.0 | "
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

        # Observation builder (v3.0+ with per-instrument support)
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

    def _check_trade_outcomes_for_autonomy(self) -> None:
        """
        Check for trade outcomes from Executor and update PPO autonomy tracker.

        This enables adaptive leadership transition based on PPO's actual
        trading performance. The position manager publishes trade outcomes
        to 'trade_outcome_for_autonomy' on the SmartInfoBus.
        """
        try:
            outcome = self.smart_bus.get(
                "trade_outcome_for_autonomy",
                "PPOAgentShell",
                default=None,
            )
            if not outcome or not isinstance(outcome, dict):
                return

            # Check if this is a new outcome (avoid double-counting)
            outcome_ts = outcome.get("timestamp", 0)
            last_processed = getattr(self, "_last_autonomy_outcome_ts", 0)
            if outcome_ts <= last_processed:
                return

            # Update the arbiter's autonomy tracker
            self.arbiter.record_trade_outcome(
                instrument=outcome.get("instrument", "UNKNOWN"),
                ppo_direction=outcome.get("ppo_direction", "flat"),
                expert_direction=outcome.get("expert_direction", "flat"),
                pnl=float(outcome.get("pnl", 0.0)),
                ppo_confidence=float(outcome.get("ppo_confidence", 0.0)),
                was_ppo_led=bool(outcome.get("was_ppo_led", False)),
            )

            # Mark as processed
            self._last_autonomy_outcome_ts = outcome_ts

            # Log the autonomy update
            state = self.arbiter.get_autonomy_state()
            self.logger.info(
                f"PPO Autonomy updated: Phase={state['phase']}, Level={state['autonomy_level']:.2f}, "
                f"WinRate={state['ppo_win_rate'] * 100.0:.1f}%, Trades={state['total_trades_evaluated']}"
            )

        except Exception as e:  # noqa: BLE001
            self.logger.debug(f"Failed to process trade outcome for autonomy: {e}")

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
            # 0) Check for trade outcomes and update autonomy tracker
            self._check_trade_outcomes_for_autonomy()

            # 0.5) CHECK POSITION FOCUS MODE
            # If we have active positions, switch to position management mode
            position_focus: Optional[Dict[str, Any]] = (
                self._gather_position_focus_context()
            )
            in_position_focus_mode: bool = bool(
                position_focus
                and position_focus.get("focus_mode_active", False)
            )

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

            # 1d) If in position focus mode, adjust committee data to reflect position management
            if in_position_focus_mode and position_focus is not None:
                committee_data = self._adjust_committee_for_position_focus(
                    committee_data,
                    position_focus,
                )
                self.logger.info(
                    f"[PPO] POSITION FOCUS MODE: {position_focus.get('primary_instrument')} "
                    f"side={position_focus.get('primary_side')}, pnl={float(position_focus.get('primary_pnl', 0.0)):.2f}"
                )

            # 2) Build observations for each instrument
            observations = self._build_observations_for_instruments()

            # DEBUG: Log observation summary
            self.logger.info(
                f"[PPO] Processing: obs_dims={[len(v) for v in observations.values()]}, "
                f"committee={bool(committee_data)}, risk={risk_info.portfolio_risk:.2f}"
                f"{', POSITION_FOCUS' if in_position_focus_mode else ''}"
            )

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

            # 3.5) Apply position focus mode adjustments to decision
            if in_position_focus_mode and position_focus is not None:
                multi_decision = self._apply_position_focus_to_decision(
                    multi_decision,
                    position_focus,
                )

            # Cache decisions
            self._last_multi_decision = multi_decision
            self._last_observations = observations
            self._last_decisions = multi_decision.instruments

            # Log autonomy phase (important for debugging)
            autonomy_meta = multi_decision.global_meta.get("ppo_autonomy", {})
            autonomy_phase = autonomy_meta.get("phase", "UNKNOWN")
            autonomy_level = autonomy_meta.get("autonomy_level", 0.0)
            
            # Log per-instrument decisions
            for inst, decision in multi_decision.instruments.items():
                self.logger.info(
                    f"[PPO] {inst}: dir={decision.direction}, conf={decision.confidence:.2f}, "
                    f"gate={'PASS' if decision.gate_passed else 'BLOCK'}, "
                    f"trust={decision.trust_score:.2f}, regime={decision.regime}, "
                    f"phase={autonomy_phase}"
                )

            # 4) Build result dict
            result = self._build_process_result(multi_decision)

            # Generate thesis for primary instrument
            primary_decision = multi_decision.instruments.get(
                self._cfg.primary_instrument
            )
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
        """
        Gather committee consensus from SmartInfoBus.

        CRITICAL FIX: Use per-instrument committee decisions from
        'committee_decisions_by_instrument' to ensure each instrument gets its
        correct direction (e.g., XAUUSD SHORT vs EURUSD LONG).
        """
        name = "PPOAgentShell"

        # Authoritative per-instrument decisions
        per_inst_decisions = (
            self.smart_bus.get(
                "committee_decisions_by_instrument",
                name,
                default={},
            )
            or {}
        )

        # Global fallback for backward compatibility
        committee_decision = (
            self.smart_bus.get("committee_decision", name, default={}) or {}
        )
        if isinstance(committee_decision, str):
            committee_decision = {"action": committee_decision}

        result: Dict[str, Any] = {
            "action": str(committee_decision.get("action", "hold")).lower(),
            "confidence": float(
                self.smart_bus.get("committee_confidence", name, default=0.5) or 0.5
            ),
            "consensus_score": float(
                self.smart_bus.get("consensus_score", name, default=0.5) or 0.5
            ),
            "fragility": float(
                self.smart_bus.get("fragility", name, default=0.5) or 0.5
            ),
            "regime": self.smart_bus.get("market_regime", name, default="unknown")
            or "unknown",
            "regime_strength": float(
                self.smart_bus.get("regime_strength", name, default=0.5) or 0.5
            ),
            # Per-instrument decisions used by ArbiterLogic
            "instruments": {},
        }

        if isinstance(per_inst_decisions, dict):
            for inst, inst_data in per_inst_decisions.items():
                if not isinstance(inst_data, dict):
                    continue
                inst_action = str(
                    inst_data.get("action", inst_data.get("direction", "hold"))
                ).lower()
                inst_conf = float(inst_data.get("confidence", inst_data.get("weight", 0.5)) or 0.5)
                inst_cons = float(inst_data.get("consensus_score", 0.5) or 0.5)

                result["instruments"][inst] = {
                    "action": inst_action,
                    "confidence": inst_conf,
                    "consensus_score": inst_cons,
                }
                self.logger.debug(
                    f"Per-instrument committee: {inst} -> {inst_action} (conf={inst_conf:.2f})"
                )

        return result

    def _gather_expert_signals(self) -> Dict[str, Any]:
        """
        Gather expert voting, risk, and memory signals from SmartInfoBus.

        This aggregates:
        - Expert directional votes (Trend, Momentum, Theme, Seasonality)
        - Market regime context
        - DynamicRiskController outputs
        - MemoryGate & danger zones from UnifiedMemory
        """
        name = "PPOAgentShell"

        def _expert_block(vote_key: str, conf_key: str) -> Dict[str, Any]:
            raw = self.smart_bus.get(vote_key, name, default="flat") or "flat"
            conf = self.smart_bus.get(conf_key, name, default=None)
            try:
                conf_val = float(conf) if conf is not None else 0.0
            except Exception:  # noqa: BLE001
                conf_val = 0.0
            return {"proposal": raw, "confidence": conf_val}

        expert_signals = {
            "trend": _expert_block(
                "TrendExpert_voting_proposal", "TrendExpert_confidence"
            ),
            "momentum": _expert_block(
                "MomentumExpert_voting_proposal", "MomentumExpert_confidence"
            ),
            "theme": _expert_block(
                "ThemeExpert_voting_proposal", "ThemeExpert_confidence"
            ),
            "seasonality": _expert_block(
                "SeasonalityRiskExpert_voting_proposal",
                "SeasonalityRiskExpert_confidence",
            ),
        }

        market_context = {
            "regime": self.smart_bus.get("market_regime", name, default="unknown")
            or "unknown",
            "regime_strength": float(
                self.smart_bus.get("regime_strength", name, default=0.5) or 0.5
            ),
        }

        # v4.2.0: Enhanced risk signal gathering from DynamicRiskController
        risk_signals = {
            "risk_data": self.smart_bus.get("risk_data", name, default={}) or {},
            "portfolio_risk": self.smart_bus.get(
                "portfolio_risk",
                name,
                default={},
            )
            or {},
            "risk_scaling": self.smart_bus.get(
                "risk_scaling",
                name,
                default={},
            )
            or {},
            "risk_assessment": self.smart_bus.get(
                "risk_assessment",
                name,
                default={},
            )
            or {},
            "risk_scale": self.smart_bus.get("risk_scale", name, default=None),
            "risk_level": self.smart_bus.get("risk_level", name, default=None),
        }

        # Memory signals (gate + danger zones)
        raw_memory_gate = self.smart_bus.get("memory_gate", name, default=None)
        raw_danger_zones = self.smart_bus.get("danger_zones", name, default=None)

        if isinstance(raw_memory_gate, dict):
            memory_gate_meta = raw_memory_gate
        else:
            try:
                memory_gate_value = (
                    float(raw_memory_gate) if raw_memory_gate is not None else 1.0
                )
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
        """
        name = "PPOAgentShell"

        # BiasAuditor outputs
        bias_adjustments = self.smart_bus.get("bias_adjustments", name, default=None)
        bias_analysis = self.smart_bus.get("bias_analysis", name, default=None)
        psychological_state = self.smart_bus.get(
            "psychological_state",
            name,
            default=None,
        )

        # CurriculumPlannerPlus outputs
        curriculum_stage = self.smart_bus.get("curriculum_stage", name, default=None)
        learning_constraints = self.smart_bus.get(
            "learning_constraints",
            name,
            default=None,
        )
        mastery_assessment = self.smart_bus.get(
            "mastery_assessment",
            name,
            default=None,
        )

        # ThesisEvolutionEngine outputs
        best_thesis = self.smart_bus.get("best_thesis", name, default=None)

        return StrategyInfo.from_bus_data(
            bias_adjustments=bias_adjustments if isinstance(bias_adjustments, dict) else None,
            bias_analysis=bias_analysis if isinstance(bias_analysis, dict) else None,
            psychological_state=psychological_state
            if isinstance(psychological_state, dict)
            else None,
            curriculum_stage=curriculum_stage
            if isinstance(curriculum_stage, dict)
            else None,
            learning_constraints=learning_constraints
            if isinstance(learning_constraints, dict)
            else None,
            mastery_assessment=mastery_assessment
            if isinstance(mastery_assessment, dict)
            else None,
            best_thesis=best_thesis if isinstance(best_thesis, dict) else None,
        )

    def _gather_trading_mode_info(self) -> TradingModeInfo:
        """
        Gather trading mode information from SmartInfoBus.

        Integrates outputs from TradingModeManager for intelligent position sizing
        and risk adjustment based on current market conditions.
        """
        name = "PPOAgentShell"

        trading_mode = self.smart_bus.get("trading_mode", name, default=None)
        mode_config = self.smart_bus.get("mode_config", name, default=None)
        mode_effectiveness = self.smart_bus.get(
            "mode_effectiveness",
            name,
            default=None,
        )
        decision_factors = self.smart_bus.get(
            "decision_factors",
            name,
            default=None,
        )

        eff_value: Optional[float] = None
        if mode_effectiveness is not None:
            if isinstance(mode_effectiveness, dict):
                eff_value = mode_effectiveness.get(
                    "value",
                    mode_effectiveness.get("effectiveness"),
                )
                if eff_value is not None:
                    try:
                        eff_value = float(eff_value)
                    except (TypeError, ValueError):
                        eff_value = None
            else:
                try:
                    eff_value = float(mode_effectiveness)
                except (TypeError, ValueError):
                    eff_value = None

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
        """
        name = "PPOAgentShell"

        market_predictions = self.smart_bus.get(
            "market_predictions",
            name,
            default=None,
        )
        prediction_confidence = self.smart_bus.get(
            "prediction_confidence",
            name,
            default=None,
        )
        scenario_generation = self.smart_bus.get(
            "scenario_generation",
            name,
            default=None,
        )
        world_model_analytics = self.smart_bus.get(
            "world_model_analytics",
            name,
            default=None,
        )

        conf_value: Optional[float] = None
        if prediction_confidence is not None:
            if isinstance(prediction_confidence, dict):
                conf_value = prediction_confidence.get(
                    "value",
                    prediction_confidence.get("confidence"),
                )
                if conf_value is not None:
                    try:
                        conf_value = float(conf_value)
                    except (TypeError, ValueError):
                        conf_value = None
            else:
                try:
                    conf_value = float(prediction_confidence)
                except (TypeError, ValueError):
                    conf_value = None

        return WorldModelInfo.from_bus_data(
            market_predictions=market_predictions if isinstance(market_predictions, dict) else None,
            prediction_confidence=conf_value,
            scenario_generation=scenario_generation
            if isinstance(scenario_generation, dict)
            else None,
            world_model_analytics=world_model_analytics
            if isinstance(world_model_analytics, dict)
            else None,
        )

    # ─────────────────────────────────────────────────────────────
    # Position Focus Mode (v4.5+)
    # ─────────────────────────────────────────────────────────────

    def _gather_position_focus_context(self) -> Optional[Dict[str, Any]]:
        """
        Gather position focus context from SmartInfoBus.

        Position manager can publish a dictionary at 'position_focus_context' with:
        - focus_mode_active: bool
        - primary_instrument: str
        - primary_side: int (1=LONG, -1=SHORT, 0=flat)
        - primary_pnl: float
        and potentially more metadata.

        When focus_mode_active is True, we:
        1. Do not open NEW positions.
        2. Treat PPO signals as guidance for managing the existing position.
        3. Use metadata to signal potential exit/hold bias instead of flipping.
        """
        name = "PPOAgentShell"
        ctx = self.smart_bus.get(
            "position_focus_context",
            name,
            default=None,
        )

        if not ctx or not isinstance(ctx, dict):
            return None

        if not ctx.get("focus_mode_active", False):
            return None

        return ctx

    def _adjust_committee_for_position_focus(
        self,
        committee_data: Dict[str, Any],
        position_focus: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Adjust committee data for position focus mode.

        We reinterpret committee output as position management signals instead
        of new-entry signals. This is mainly advisory for ArbiterLogic.
        """
        position_side = int(position_focus.get("primary_side", 0))  # 1=LONG, -1=SHORT
        primary_inst = position_focus.get("primary_instrument")

        if not position_side or not primary_inst:
            return committee_data

        adjusted = dict(committee_data)
        adjusted["position_focus_mode"] = True
        adjusted["position_side"] = position_side
        adjusted["position_instrument"] = primary_inst

        committee_action = str(committee_data.get("action", "hold")).lower()
        committee_conf = float(committee_data.get("confidence", 0.5) or 0.5)

        # Determine if committee supports or opposes the existing position
        if position_side > 0:  # LONG position
            supports_position = committee_action in ("buy", "long", "hold", "flat")
        else:  # SHORT position
            supports_position = committee_action in ("sell", "short", "hold", "flat")

        if supports_position:
            adjusted["action"] = "hold"
            adjusted["position_evaluation"] = "supports_position"
        else:
            if committee_conf > 0.7:
                adjusted["action"] = "tighten"  # strong opposition: tighten stops
                adjusted["position_evaluation"] = "strongly_opposes_position"
            elif committee_conf > 0.5:
                adjusted["action"] = "review"  # moderate opposition
                adjusted["position_evaluation"] = "opposes_position"
            else:
                adjusted["action"] = "hold"  # weak opposition: no drastic action
                adjusted["position_evaluation"] = "weakly_opposes_position"

        return adjusted

    def _apply_position_focus_to_decision(
        self,
        multi_decision: ArbiterMultiDecision,
        position_focus: Mapping[str, Any],
    ) -> ArbiterMultiDecision:
        """
        Apply position focus mode constraints to the multi-instrument decision.

        PRINCIPLES:
        - Existing position's instrument:
            * Direction is collapsed to HOLD/FLAT but we add metadata to signal
              whether PPO wants to EXIT or HOLD.
        - Other instruments:
            * If position_focus_blocks_other_instruments=True: block new entries
            * If position_focus_blocks_other_instruments=False: allow independent decisions

        We intentionally do NOT invent a new 'exit' direction here to keep
        compatibility with PositionManager / SmartPositionManager, which expects
        directions in {long, short, hold/flat}.
        """
        position_side = int(position_focus.get("primary_side", 0))
        primary_inst = position_focus.get("primary_instrument")
        position_pnl = float(position_focus.get("primary_pnl", 0.0))
        
        # Check if we should block other instruments (configurable)
        block_other_instruments = self._cfg.position_focus_blocks_other_instruments

        for inst, decision in multi_decision.instruments.items():
            # Defensive: ensure meta exists
            if decision.meta is None:
                decision.meta = {}

            if inst == primary_inst and position_side != 0:
                # Instrument with existing position: position management only
                ppo_direction = (decision.direction or "hold").lower()

                if position_side > 0:  # LONG position
                    supports = ppo_direction in ("long", "buy", "hold", "flat")
                else:  # SHORT position
                    supports = ppo_direction in ("short", "sell", "hold", "flat")

                old_reasoning = decision.reasoning

                # Map PPO view into position-management hints, not new entries
                if supports:
                    decision.reasoning = (
                        f"POSITION_FOCUS(HOLD): PPO supports existing position | "
                        f"{old_reasoning}"
                    )
                    # Do not scale by default in focus mode
                    decision.direction = "hold"
                    decision.position_size = 0.0
                    decision.gate_passed = False
                    decision.meta["position_focus_hint"] = "hold_or_scale_carefully"
                else:
                    # PPO wants opposite direction → interpret as exit pressure
                    if position_pnl > 0:
                        decision.reasoning = (
                            "POSITION_FOCUS(TAKE_PROFIT_CANDIDATE): "
                            f"PPO opposes profitable position | {old_reasoning}"
                        )
                        decision.meta["position_focus_hint"] = "take_profit_candidate"
                    else:
                        decision.reasoning = (
                            "POSITION_FOCUS(CUT_LOSS_CANDIDATE): "
                            f"PPO opposes losing position | {old_reasoning}"
                        )
                        decision.meta["position_focus_hint"] = "cut_loss_candidate"

                    # In both cases we do NOT flip direction here; we just
                    # collapse to hold and let exit engine / risk modules decide.
                    decision.direction = "hold"
                    decision.position_size = 0.0
                    decision.gate_passed = False

                decision.meta["position_focus_mode"] = True
                decision.meta["supports_position"] = supports
                decision.meta["position_pnl"] = position_pnl

            else:
                # Instruments without existing position
                if block_other_instruments:
                    # HARD BLOCK new entries for other instruments
                    if decision.direction in ("long", "short") and decision.gate_passed:
                        old_reasoning = decision.reasoning
                        decision.reasoning = (
                            f"POSITION_FOCUS(BLOCKED): Focus on {primary_inst}, "
                            f"no new entries | {old_reasoning}"
                        )
                        decision.direction = "flat"
                        decision.gate_passed = False
                        decision.position_size = 0.0
                        decision.meta["blocked_by_position_focus"] = True
                        decision.meta["position_focus_mode"] = True
                else:
                    # Allow independent decisions for other instruments
                    # Just add metadata noting we're in position focus mode
                    decision.meta["position_focus_mode"] = True
                    decision.meta["blocked_by_position_focus"] = False

        # Update global metadata
        multi_decision.global_meta["position_focus"] = {
            "active": True,
            "instrument": primary_inst,
            "side": position_side,
            "pnl": position_pnl,
            "blocks_other_instruments": block_other_instruments,
        }

        return multi_decision

    # ─────────────────────────────────────────────────────────────
    # Observation Building (v3.0+)
    # ─────────────────────────────────────────────────────────────

    def _build_observations_for_instruments(self) -> Dict[str, np.ndarray]:
        """
        Build observation vectors for each instrument.

        Uses the v3.0 observation builder which supports per-instrument
        observations. If per-instrument data is unavailable, falls back to
        zero vectors with the configured obs_size.
        """
        observations: Dict[str, np.ndarray] = {}

        for instrument in self._cfg.instruments:
            try:
                obs = self.obs_builder.build_for_instrument(
                    instrument=instrument,
                    smart_bus=self.smart_bus,
                    module_name="PPOAgentShell",
                )
                observations[instrument] = obs
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"Failed to build observation for {instrument}: {e}"
                )
                observations[instrument] = np.zeros(
                    self._cfg.core_config.obs_size,
                    dtype=np.float32,
                )

        return observations

    # ─────────────────────────────────────────────────────────────
    # Result Building
    # ─────────────────────────────────────────────────────────────

    def _build_process_result(self, multi_decision: ArbiterMultiDecision) -> Dict[str, Any]:
        """Build the result dictionary from multi-instrument decision."""
        primary = multi_decision.instruments.get(self._cfg.primary_instrument)

        agent_perf: Dict[str, Any] = {
            "decisions_made": self._performance_metrics["total_decisions"],
            "avg_processing_time_ms": self._performance_metrics["avg_processing_time_ms"],
            "health_status": self._health_status,
            "circuit_breaker": self.circuit_breaker["state"],
        }

        # Build policy_actions from primary decision (backward compatible schema)
        primary_obs = self._last_observations.get(self._cfg.primary_instrument)
        _ = primary_obs  # currently unused, but kept for potential future use

        policy_actions: Dict[str, Any] = {
            "action": [
                1.0
                if primary and primary.direction == "long"
                else -1.0
                if primary and primary.direction == "short"
                else 0.0
            ],
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
                # PPO Autonomy state (ADAPTIVE LEADERSHIP)
                "ppo_autonomy_state": self.arbiter.get_autonomy_state(),
                # Agent performance diagnostics
                "agent_performance": agent_perf,
                # Legacy voting outputs
                "PPOAgent_voting_proposal": self._build_voting_payload(multi_decision),
                "PPOAgent_confidence": primary.confidence,
                # Contract-required training outputs
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
                "ppo_autonomy_state": self.arbiter.get_autonomy_state(),
                "agent_performance": agent_perf,
                "PPOAgent_voting_proposal": {},
                "PPOAgent_confidence": 0.0,
                "policy_actions": policy_actions,
                "policy_gradients": policy_gradients,
                "rewards": rewards,
                "training_data": training_data,
                "training_metrics": training_metrics,
                "training_signals": training_signals,
            }

        return result

    def _build_voting_payload(self, multi_decision: ArbiterMultiDecision) -> Dict[str, Any]:
        """
        Build legacy voting payload from multi-decision.

        IMPORTANT:
        - We avoid importing PerInstrumentVote / InstrumentProposal directly
          to keep this module decoupled and Pylance-clean.
        - Instead we construct the same schema that PerInstrumentVote.to_dict()
          would produce.

        Schema:
            {
                "member": "PPOAgent",
                "proposals": {
                    "EURUSD": {
                        "instrument": "EURUSD",
                        "action": "long",
                        "confidence": 0.73,
                        "magnitude": 0.25,
                        "horizon": "intraday",
                        "rationale": "...",
                        "meta": {...},
                    },
                    ...
                },
                "timestamp": "...",
                "action": "<global_action>",
                "confidence": <global_confidence>,
                "proposal": {
                    "direction": "<global_action>",
                    "magnitude": <max_magnitude>,
                    "horizon": "intraday",
                },
            }
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        proposals: Dict[str, Dict[str, Any]] = {}
        global_action = "hold"
        global_confidence = 0.0
        max_magnitude = 0.0
        best_score = -1.0

        for instrument, decision in multi_decision.instruments.items():
            direction = (decision.direction or "hold").lower()
            confidence = float(decision.confidence or 0.0)
            magnitude = float(decision.position_size or 0.0)

            # Normalize confidence and magnitude into [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            magnitude = max(0.0, min(1.0, magnitude))

            proposals[instrument] = {
                "instrument": instrument,
                "action": direction,
                "confidence": round(confidence, 4),
                "magnitude": round(magnitude, 4),
                "horizon": "intraday",
                "rationale": decision.reasoning,
                "meta": decision.meta or {},
            }

            # Directional proposals get full weight; neutrals get reduced weight
            if direction in ("long", "short"):
                score = confidence * max(magnitude, 0.1)
            else:
                score = 0.5 * confidence * max(magnitude, 0.1)

            if score > best_score:
                best_score = score
                global_action = direction
                global_confidence = confidence

            if magnitude > max_magnitude:
                max_magnitude = magnitude

        return {
            "member": "PPOAgent",
            "proposals": proposals,
            "timestamp": timestamp,
            "action": global_action,
            "confidence": global_confidence,
            "proposal": {
                "direction": global_action,
                "magnitude": max_magnitude,
                "horizon": "intraday",
            },
        }

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

            # PPO Autonomy state (ADAPTIVE LEADERSHIP)
            autonomy_state = self.arbiter.get_autonomy_state()
            self.smart_bus.set(
                "ppo_autonomy_state",
                autonomy_state,
                module="PPOAgent",
                thesis=(
                    f"Autonomy: {autonomy_state['phase']} "
                    f"(level={autonomy_state['autonomy_level']:.2f}, "
                    f"WR={autonomy_state['ppo_win_rate']:.0%})"
                ),
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
                    "avg_processing_time_ms": self._performance_metrics[
                        "avg_processing_time_ms"
                    ],
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
            self.circuit_breaker["cooldown_until"] = datetime.now() + timedelta(
                seconds=60
            )

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

        # Contract-required training outputs (empty/safe fallbacks)
        policy_actions: Dict[str, Any] = {
            "action": [0.0],  # hold
            "log_prob": 0.0,
            "value_estimate": 0.0,
            "action_std": [1.0],
            "exploration_level": 0.0,
        }
        policy_gradients: Dict[str, Any] = {
            "gradient_norm": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
        }
        training_data: Dict[str, Any] = {
            "total_steps": 0,
            "episodes": 0,
            "mode": "error_fallback",
        }
        training_metrics: Dict[str, Any] = {
            "avg_reward": 0.0,
            "avg_episode_length": 0,
            "explained_variance": 0.0,
        }
        training_signals: Dict[str, Any] = {
            "gradient_norm": 0.0,
            "explained_variance": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
        }

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
            "ppo_autonomy_state": self.arbiter.get_autonomy_state(),
            "agent_performance": {
                "decisions_made": self._performance_metrics["total_decisions"],
                "avg_processing_time_ms": self._performance_metrics[
                    "avg_processing_time_ms"
                ],
                "health_status": self._health_status,
                "circuit_breaker": self.circuit_breaker["state"],
            },
            "PPOAgent_voting_proposal": {},
            "PPOAgent_confidence": 0.0,
            "policy_actions": policy_actions,
            "policy_gradients": policy_gradients,
            "rewards": [],
            "training_data": training_data,
            "training_metrics": training_metrics,
            "training_signals": training_signals,
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

    # ─────────────────────────────────────────────────────────────
    # State Persistence
    # ─────────────────────────────────────────────────────────────

    def _get_custom_state(self) -> Dict[str, Any]:
        """
        Get custom state for persistence.

        Saves:
        - Performance metrics (decision counts, avg processing time)
        - Circuit breaker state
        - Health status
        - Last decisions (for graceful recovery)
        - Current instrument config (for sanity checks)
        """
        return {
            "performance_metrics": dict(self._performance_metrics),
            "circuit_breaker": {
                "state": self.circuit_breaker.get("state", "CLOSED"),
                "failures": self.circuit_breaker.get("failures", 0),
                "last_failure": self.circuit_breaker.get("last_failure"),
            },
            "health_status": self._health_status,
            "last_decisions": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self._last_decisions.items()
            },
            "config": {
                "instruments": self._cfg.instruments,
                "primary_instrument": self._cfg.primary_instrument,
            },
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        """
        Restore custom state from persistence.
        """
        if not state:
            return

        # Restore performance metrics
        metrics = state.get("performance_metrics", {})
        if metrics:
            self._performance_metrics.update(metrics)

        # Restore circuit breaker (but keep it CLOSED on fresh start for safety)
        cb = state.get("circuit_breaker", {})
        if cb:
            # Reset circuit breaker on restart (don't persist OPEN state)
            self.circuit_breaker["failures"] = max(0, cb.get("failures", 0) - 1)
            self.circuit_breaker["state"] = "CLOSED"  # Always start fresh
            self.circuit_breaker["last_failure"] = cb.get("last_failure")

        # Restore health status (informational only)
        self._health_status = state.get("health_status", "healthy")

        self.logger.info(
            f"📂 PPOAgentShell state restored | decisions={self._performance_metrics.get('total_decisions', 0)} | "
            f"avg_time={self._performance_metrics.get('avg_processing_time_ms', 0.0):.1f}ms"
        )
