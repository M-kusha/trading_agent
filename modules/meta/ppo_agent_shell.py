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

import os
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

from modules.meta.ppo_core import PPOCore, PPOCoreConfig, DiscreteActionDecoded
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
    FEATURE_GROUPS,
    PRIMARY_TIMEFRAME,
)

from modules.meta.arbiter_logic import StrategyInfo, TradingModeInfo, WorldModelInfo

# Live action masking for MaskablePPO parity
from modules.meta.live_action_mask import LiveActionMaskBuilder, LiveMaskConfig


def _norm_symbol(sym: str) -> str:
    """Canonical symbol normalization used across shell/arbiter."""
    return "".join(ch for ch in str(sym or "").upper() if ch.isalnum())


# ═══════════════════════════════════════════════════════════════════
# PPO AGENT SHELL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PPOShellConfig:
    """Configuration for PPO Agent Shell (v3.x)."""

    # Core PPO configuration
    core_config: PPOCoreConfig = field(default_factory=PPOCoreConfig)

    # Optional model checkpoint to load on startup (SB3 .zip or PPOCore torch checkpoint)
    model_path: Optional[str] = None

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

    # Live discrete action masking
    # When True, apply timing/drawdown/trade-count rules as a hard mask (safer, reduces overtrading).
    # When False, only physical impossibilities are masked (matches training when soft rules were
    # learned via penalties).
    live_mask_enforce_hard_rules: bool = False

    # Entry cadence (training parity)
    # When True, NEW entries are only allowed once per new PRIMARY_TIMEFRAME bar (first cycle that sees a new bar timestamp).
    # This reduces mid-bar "machine-gunner" behavior and more closely matches training's step-per-bar cadence.
    entries_require_new_primary_bar: bool = False

    # ═══════════════════════════════════════════════════════════════════
    # WARMUP SETTINGS (v3.2.0)
    # ═══════════════════════════════════════════════════════════════════
    # PPO needs time to observe market conditions before making decisions.
    # During warmup, the agent observes but does NOT generate entry signals.
    # This prevents hasty trades based on incomplete market context.
    warmup_cycles: int = 30  # ~60 seconds at 3s/cycle - observe first before trading
    warmup_enabled: bool = True  # Set to False to disable warmup (not recommended)

    # Debug
    debug: bool = False

    # Observation diagnostics (helps debug "same obs for all instruments" issues)
    obs_diagnostics_enabled: bool = True
    obs_diagnostics_every_n_cycles: int = 20


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

        # Hard gate: the strict PPO observation builder supports only PRIMARY_INSTRUMENT.
        # This prevents EURUSD (or any other symbol) from being processed/log-spammed here.
        self._cfg.primary_instrument = PRIMARY_INSTRUMENT
        self._cfg.instruments = [PRIMARY_INSTRUMENT]

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
        self._trade_open_allowed: bool = True
        self._trade_open_gate: Dict[str, Any] = {}
        self._in_warmup: bool = False
        self._last_seen_primary_bar_ts: Dict[str, str] = {}

        self._health_status: str = "healthy"
        self.circuit_breaker: Dict[str, Any] = {}
        self._performance_metrics: Dict[str, Any] = {}
        self._monitoring_active: bool = False
        self._last_cooldown_state: Dict[str, bool] = {}

        # Observation diagnostics state
        self._obs_diag_prev: Dict[str, np.ndarray] = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # FRAME STACKING (DISABLED - v3.4.0)
        # ═══════════════════════════════════════════════════════════════════
        # Frame stacking is NOT needed because the 64-dim observation already
        # contains extensive temporal indicators:
        # - RSI-14 (14 bars history), MACD (26 bars), ATR-14, 20-bar slope,
        # - 10-bar momentum/ROC, 20-bar return volatility, 50-bar mean baselines
        # Frame stacking would just duplicate this temporal information.
        self._frame_stack_size: int = 1  # 1 = disabled (no stacking)
        self._obs_frame_buffer: Dict[str, List[np.ndarray]] = {}  # Per-instrument frame buffers
        
        # ═══════════════════════════════════════════════════════════════════
        # WARMUP STATE (v3.2.0)
        # ═══════════════════════════════════════════════════════════════════
        # Track session cycles to implement warmup period
        self._session_cycle_count: int = 0
        self._warmup_complete: bool = False
        self._session_start_time: datetime = datetime.now()

        # Setup components
        self._setup_logging()
        self._setup_smart_bus()
        self._setup_core_components(model_path or self._cfg.model_path)
        self._setup_health_tracking()

        # Start monitoring
        self._start_monitoring()

        warmup_info = ""
        if self._cfg.warmup_enabled:
            warmup_info = f" | warmup={self._cfg.warmup_cycles} cycles"
        
        # Observation size info
        obs_size = self._cfg.core_config.obs_size
        
        self.logger.info(
            "[PPOAgentShell] Initialized v3.4.0 (No Frame Stack) | "
            f"instruments={self._cfg.instruments} | "
            f"obs_size={obs_size}{warmup_info}"
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

        # Provide instrument ordering to PPOCore (used by SB3 action slicing)
        try:
            self.ppo_core.set_instruments(self._cfg.instruments)
        except Exception:
            pass

        # Resolve model path (explicit > config > auto-discovery)
        # FIX: Use absolute paths to handle different working directories
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        resolved_model_path = model_path
        # If a model_path is provided, resolve it robustly (cwd vs project root)
        if resolved_model_path:
            try:
                resolved_model_path = os.path.expandvars(
                    os.path.expanduser(str(resolved_model_path))
                )
                if (not os.path.isabs(resolved_model_path)) and (
                    not os.path.exists(resolved_model_path)
                ):
                    abs_try = os.path.join(project_root, resolved_model_path)
                    if os.path.exists(abs_try):
                        resolved_model_path = abs_try

                if resolved_model_path and (not os.path.exists(resolved_model_path)):
                    self.logger.warning(
                        f"[PPO] model_path not found: {resolved_model_path}; trying auto-discovery"
                    )
                    resolved_model_path = None
            except Exception:
                pass
        if not resolved_model_path:
            # Prioritize propfirm models (trained for prop firm rules)
            candidates = [
                "models/propfirm/best/best_model.zip",  # PropFirm MaskablePPO (primary)
                "models/propfirm/propfirm_ppo_final.zip",  # PropFirm final checkpoint
                "models/best/best_model.zip",  # Generic training output
                "models/ppo_trading_model.zip",
                "models/ppo_final_model.zip",
                "models/modern_ppo_final.zip",
            ]
            for cand in candidates:
                abs_cand = os.path.join(project_root, cand)
                if os.path.exists(abs_cand):
                    resolved_model_path = abs_cand
                    self.logger.info(f"[PPO] Auto-discovered model: {cand}")
                    break

        # Load model if available
        if resolved_model_path:
            try:
                self.ppo_core.load(resolved_model_path)
                self.logger.info(f"[PPO] Loaded model: {resolved_model_path}")
                
                # Log if MaskablePPO (discrete) model detected
                if self.ppo_core.is_discrete_action_space:
                    self.logger.info(
                        f"[PPO] ═══ MaskablePPO DISCRETE MODE ═══ "
                        f"Actions: {self.ppo_core.config.n_discrete_actions}, "
                        f"Size buckets: {self.ppo_core.config.size_buckets}"
                    )
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"[PPO] Failed to load model '{resolved_model_path}': {e}")
        else:
            self.logger.warning("[PPO] No model_path provided; using untrained PPOCore weights")

        # Live action mask builder (for MaskablePPO parity)
        self._live_mask_builder = LiveActionMaskBuilder(LiveMaskConfig(   
            size_buckets=self._cfg.core_config.size_buckets,
            enforce_hard_rules=bool(self._cfg.live_mask_enforce_hard_rules),
        ))
        
        # Register action mask callback with PPOCore
        self.ppo_core.set_action_mask_fn(self._get_live_action_mask)

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
            # ═══════════════════════════════════════════════════════════════════
            # WARMUP CHECK (v3.2.0)
            # ═══════════════════════════════════════════════════════════════════
            # Increment session cycle counter
            self._session_cycle_count += 1
            
            # Check if we're still in warmup period
            in_warmup = (
                self._cfg.warmup_enabled 
                and not self._warmup_complete 
                and self._session_cycle_count <= self._cfg.warmup_cycles
            )
            self._in_warmup = bool(in_warmup)
            
            if in_warmup:
                remaining = self._cfg.warmup_cycles - self._session_cycle_count
                if self._session_cycle_count == 1:
                    self.logger.info(
                        f"[PPO] ═══ WARMUP STARTED ═══ Observing market for {self._cfg.warmup_cycles} cycles (~{self._cfg.warmup_cycles * 3}s)"
                    )
                elif self._session_cycle_count % 5 == 0:  # Log every 5 cycles
                    self.logger.info(
                        f"[PPO] 🔄 WARMUP: {self._session_cycle_count}/{self._cfg.warmup_cycles} cycles │ {remaining} remaining"
                    )
            elif not self._warmup_complete and self._cfg.warmup_enabled:
                # Just completed warmup
                self._warmup_complete = True
                elapsed = (datetime.now() - self._session_start_time).total_seconds()
                self.logger.info(
                    f"[PPO] ═══ WARMUP COMPLETE ═══ Ready to trade after {elapsed:.0f}s observation"
                )

            # 0) Check for trade outcomes and update autonomy tracker
            self._check_trade_outcomes_for_autonomy()

            # 0.25) Log instrument cooldown state (from SmartPositionManager)
            self._log_instrument_cooldowns()

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
            # ONLY adjust committee globally if we're blocking other instruments
            # Otherwise, keep committee data independent and let per-instrument logic handle it
            if (
                in_position_focus_mode 
                and position_focus is not None
                and self._cfg.position_focus_blocks_other_instruments
            ):
                committee_data = self._adjust_committee_for_position_focus(
                    committee_data,
                    position_focus,
                )
                # Don't log here - will log detailed per-instrument info later

            # 2) Build observations for each instrument
            observations = self._build_observations_for_instruments()
            self._maybe_log_observation_diagnostics(observations)

            # ALWAYS read raw position context for accurate arbiter semantics
            position_ctx_raw = self._read_position_context_raw()
            positions_by_instrument = self._build_positions_by_instrument(position_ctx_raw)
            # Derive a quick "positions exist" flag for warmup/focus blocking logic
            positions_exist = any(bool(v.get("has", False)) for v in positions_by_instrument.values() if isinstance(v, dict))

            # 2.5) Compute a single global gate for NEW entries (used by action mask + final safety check)
            self._update_trade_open_gate(
                in_warmup=in_warmup,
                memory_info=memory_info,
                risk_info=risk_info,
                positions_by_instrument=positions_by_instrument,
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
                positions_by_instrument=positions_by_instrument,  # v5.6 side-aware
                defer_stats_recording=True,  # record after warmup/focus post-processing
            )

            # 3.5) Apply position focus mode adjustments to decision
            # v5.6: DO NOT destroy PPO exit semantics for instruments with positions.
            # Only block other instruments if configured.
            if positions_exist and self._cfg.position_focus_blocks_other_instruments:
                multi_decision = self._apply_position_focus_to_decision(
                    multi_decision,
                    position_ctx_raw or {},
                )

            # Final safety: never allow NEW entries when trade_open_allowed is false.
            # Position management (close/hold) remains allowed.
            multi_decision = self._apply_trade_open_gate_to_decision(
                multi_decision=multi_decision,
                positions_by_instrument=positions_by_instrument,
            )

            # ═══════════════════════════════════════════════════════════════════
            # 3.6) WARMUP GATE BLOCK (v3.2.0)
            # ═══════════════════════════════════════════════════════════════════
            # During warmup, observe but don't generate entry signals.
            # Position management (if we already have positions) is still allowed.
            all_positions_for_warmup = (position_ctx_raw or {}).get("positions", {}) if isinstance(position_ctx_raw, dict) else {}
            
            if in_warmup:
                for inst, decision in multi_decision.instruments.items():
                    inst_norm = _norm_symbol(inst)
                    pos_data = all_positions_for_warmup.get(inst) or all_positions_for_warmup.get(inst_norm) or {}
                    has_position = bool(isinstance(pos_data, dict) and int(pos_data.get("side", 0)) != 0)

                    # Only block NEW entries, not position management
                    if not has_position and decision.direction in ("long", "short"):
                        # Force a fully neutral signal for downstream safety
                        decision.direction = "flat"
                        decision.gate_passed = False
                        decision.position_size = 0.0

                        # Add metadata + reasons (best-effort; do not assume shape)
                        if decision.meta is None:
                            decision.meta = {}
                        decision.meta["warmup_blocked"] = True

                        if hasattr(decision, "gate_reasons") and isinstance(decision.gate_reasons, list):
                            decision.gate_reasons.append("WARMUP_BLOCKED")

                        original_reasoning = decision.reasoning or ""
                        decision.reasoning = f"[WARMUP_BLOCKED] {original_reasoning}"


            # Record FINAL decisions to match what we publish/execute
            self.arbiter.record_multi_decision(multi_decision)

            # Cache decisions
            self._last_multi_decision = multi_decision
            self._last_observations = observations
            self._last_decisions = multi_decision.instruments

            # Log autonomy phase (important for debugging)
            autonomy_meta = multi_decision.global_meta.get("ppo_autonomy", {})
            autonomy_phase = autonomy_meta.get("phase", "UNKNOWN")
            autonomy_level = autonomy_meta.get("autonomy_level", 0.0)
            
            # ═══════════════════════════════════════════════════════════════════
            # CLEAN LOGGING: Separate per-instrument, context-aware (v5.6)
            # ═══════════════════════════════════════════════════════════════════
            all_positions = (position_ctx_raw or {}).get("positions", {}) if isinstance(position_ctx_raw, dict) else {}
            
            for inst, decision in multi_decision.instruments.items():
                inst_norm = _norm_symbol(inst)
                pos_data = all_positions.get(inst) or all_positions.get(inst_norm) or {}
                has_position = bool(isinstance(pos_data, dict) and int(pos_data.get("side", 0)) != 0)
                
                # v5.6: Use action_intent from arbiter for clearer logging
                action_intent = decision.meta.get("action_intent", "unknown") if decision.meta else "unknown"
                thresholds = decision.meta.get("thresholds", {}) if decision.meta else {}
                
                if has_position:
                    # ─── POSITION MODE: Show position management info ───
                    side = int(pos_data.get("side", 0))
                    # Direction emoji: 📈 = LONG (bullish), 📉 = SHORT (bearish)
                    side_emoji = "📈" if side > 0 else "📉" if side < 0 else "➖"
                    side_str = "LONG" if side > 0 else "SHORT" if side < 0 else "FLAT"
                    pnl = float(pos_data.get("unrealized_pnl", 0))
                    # P&L emoji: 🟢 = profit, 🔴 = loss, ⚪ = breakeven
                    pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                    lots = float(pos_data.get("lots", 0))
                    age_h = float(pos_data.get("age_hours", 0))
                    
                    # v5.6: Use action_intent for cleaner logging
                    if action_intent == "hold":
                        action_str = "HOLD ✓"
                    elif action_intent == "exit" or action_intent == "close":
                        action_str = "⚠️ EXIT SIGNAL"
                    elif action_intent == "reverse":
                        action_str = f"⚠️ REVERSAL→{decision.direction.upper()}"
                    elif action_intent == "scale":
                        action_str = f"SCALE {decision.direction.upper()}"
                    else:
                        action_str = f"{action_intent.upper()} ({decision.direction})"
                    
                    self.logger.info(
                        f"[PPO] ═══ {inst} ═══ POSITION ACTIVE"
                    )
                    self.logger.info(
                        f"[PPO]   {side_emoji} {side_str} {lots:.2f} lots │ {pnl_emoji} P&L: €{pnl:+.2f} │ Age: {age_h:.1f}h"
                    )
                    self.logger.info(
                        f"[PPO]   Signal: {action_str} │ Conf: {decision.confidence:.0%} │ "
                        f"Trust: {decision.trust_score:.2f} │ Regime: {decision.regime}"
                    )
                else:
                    # ─── NO POSITION: Show entry signal (only if interesting) ───
                    
                    # Check if blocked by warmup
                    is_warmup_blocked = in_warmup and decision.direction != "flat"
                    
                    if is_warmup_blocked:
                        # Show warmup blocking status
                        remaining = self._cfg.warmup_cycles - self._session_cycle_count
                        self.logger.info(
                            f"[PPO] 🔄 {inst}: {decision.direction.upper()} signal observed │ "
                            f"WARMUP ({remaining} cycles left)"
                        )
                    elif decision.direction != "flat" and decision.gate_passed:
                        self.logger.info(
                            f"[PPO] ═══ {inst} ═══ ENTRY SIGNAL"
                        )
                        self.logger.info(
                            f"[PPO]   🎯 {decision.direction.upper()} │ Conf: {decision.confidence:.0%} │ "
                            f"Trust: {decision.trust_score:.2f} │ Gate: PASS"
                        )
                    elif decision.direction != "flat":
                        # ─── BLOCKED SIGNAL: Show why it was blocked ───
                        # Gate thresholds (PICKY MODE v5.1):
                        #   Direction: ±0.50, Entry: 0.55, Reversal: 0.70, Min Conf: 55%
                        trust = decision.trust_score
                        conf = decision.confidence
                        
                        # Check for position focus blocking
                        blocked_by_focus = decision.meta.get("blocked_by_position_focus", False)
                        in_focus_mode = decision.meta.get("position_focus_mode", False)
                        
                        # Determine block reason(s) - ULTRA PICKY MODE thresholds
                        block_reasons = []
                        gate_reasons = decision.gate_reasons or []

                        # Check for seasonality blocking (high priority reason)
                        if "SEASONALITY_BLOCKED" in gate_reasons:
                            block_reasons.append("SEASONALITY: Outside trading hours")
                        elif blocked_by_focus:
                            block_reasons.append("POSITION_FOCUS blocking other instruments")
                        elif gate_reasons:
                            block_reasons.append("Gate: " + ", ".join(gate_reasons[:3]))
                        if not block_reasons:
                            block_reasons.append("Gate check failed")
                        
                        self.logger.info(
                            f"[PPO] ┌─ {inst} ─ SIGNAL BLOCKED ─────────────────────┐"
                        )
                        self.logger.info(
                            f"[PPO] │  Signal: {decision.direction.upper():5} │ Trust: {trust:+.2f} │ Conf: {conf:.0%}"
                        )
                        self.logger.info(
                            f"[PPO] │  Reason: {' + '.join(block_reasons)}"
                        )
                        long_th = float(getattr(self._cfg.core_config, "direction_long_threshold", 0.35))
                        short_th = float(getattr(self._cfg.core_config, "direction_short_threshold", -0.35))
                        min_conf = 0.50
                        self.logger.info(
                            f"[PPO] │  Thresholds → Dir: {long_th:+.2f}/{short_th:+.2f} │ MinConf: {min_conf:.0%}"
                        )
                        if in_focus_mode:
                            self.logger.info(
                                f"[PPO] │  ⚠️ Position focus mode active for this instrument"
                            )
                        self.logger.info(
                            f"[PPO] └───────────────────────────────────────────────┘"
                        )
                    # else: flat with no position = nothing interesting, skip logging

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
                # Log all committee signals (not just non-flat) so user can see what modules are thinking
                if inst_action != "flat":
                    self.logger.info(
                        f"[COMMITTEE] {inst}: {inst_action.upper()} signal (conf={inst_conf:.0%})"
                    )
                else:
                    self.logger.debug(
                        f"[COMMITTEE] {inst}: FLAT/NEUTRAL (conf={inst_conf:.0%})"
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

    def _get_live_action_mask(self) -> np.ndarray:
        """
        Build action mask for MaskablePPO from live trading state.
        
        This is called by PPOCore.select_action() when using a MaskablePPO model.
        Implements the same masking logic as prop_firm_env.action_masks() for parity.
        """
        has_position = False
        try:
            # Get current position state from SmartInfoBus
            position_ctx = self._read_position_context_raw() or {}
            positions = position_ctx.get("positions", {})
            
            # Check if any position exists (any instrument)
            for inst_data in positions.values():
                if isinstance(inst_data, dict) and int(inst_data.get("side", 0)) != 0:
                    has_position = True
                    break
            
            # Get account state
            account_state = self.smart_bus.get("account_state", "PPOAgentShell", default={})
            if isinstance(account_state, dict):
                balance = float(account_state.get("balance", 100000))     
                equity = float(account_state.get("equity", balance))      
                day_start_balance = account_state.get("day_start_balance")
                peak_balance = account_state.get("peak_balance")
            else:
                balance = 100000.0
                equity = balance

                day_start_balance = None
                peak_balance = None

            # Fallback tracking for drawdown features (in case account_state doesn't include day/peak)
            try:
                today = datetime.now().date()
                if getattr(self, "_mask_day_start_date", None) != today:
                    self._mask_day_start_date = today
                    self._mask_day_start_balance = float(balance)
                    self._mask_peak_equity = float(equity)
                else:
                    # Update peak equity during the day
                    peak_eq = float(getattr(self, "_mask_peak_equity", equity) or equity)
                    self._mask_peak_equity = max(peak_eq, float(equity))

                if not isinstance(day_start_balance, (int, float)) or float(day_start_balance) <= 0:
                    day_start_balance = float(getattr(self, "_mask_day_start_balance", balance) or balance)
                if not isinstance(peak_balance, (int, float)) or float(peak_balance) <= 0:
                    peak_balance = float(getattr(self, "_mask_peak_equity", equity) or equity)
            except Exception:
                day_start_balance = balance
                peak_balance = balance

            # Compute drawdowns
            current_dd = max(0.0, (peak_balance - equity) / max(peak_balance, 1.0))
            daily_dd = max(0.0, (day_start_balance - equity) / max(day_start_balance, 1.0))

            # Get trade counts from SmartInfoBus
            trade_stats = self.smart_bus.get("trade_statistics", "PPOAgentShell", default={})
            if isinstance(trade_stats, dict):
                daily_trades = int(trade_stats.get("daily_trades", 0))    
                session_trades = int(trade_stats.get("session_trades", 0))
                consecutive_losses = int(trade_stats.get("consecutive_losses", 0))
            else:
                daily_trades = 0
                session_trades = 0
                consecutive_losses = 0

            # Fallback trade stats from Executor trade ledger (entries only)
            try:
                trade_ledger = self.smart_bus.get("trades", "PPOAgentShell", default=[]) or []
                if isinstance(trade_ledger, list) and trade_ledger:
                    now_dt = datetime.now()
                    today_start_ts = now_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

                    def _is_new_entry(t: Mapping[str, Any]) -> bool:
                        action = str(t.get("action", "")).lower()
                        comment = str(t.get("comment", "")).lower()
                        if any(x in action for x in ("scale", "close", "exit", "reverse", "reduce")):
                            return False
                        if any(x in action for x in ("open", "long", "short", "buy", "sell")):
                            return True
                        if "open" in comment:
                            return True
                        return False

                    tail = trade_ledger[-300:]
                    computed_daily_entries = 0
                    last_entry_ts = 0.0
                    last_loss_ts = 0.0

                    for t in reversed(tail):
                        if not isinstance(t, dict):
                            continue
                        try:
                            ts = float(t.get("ts", t.get("timestamp", 0.0)) or 0.0)
                        except Exception:
                            ts = 0.0
                        if ts <= 0.0:
                            continue

                        if _is_new_entry(t):
                            if ts >= today_start_ts:
                                computed_daily_entries += 1
                            if last_entry_ts <= 0.0:
                                last_entry_ts = ts

                        # Best-effort loss timestamp if realized PnL is present
                        action = str(t.get("action", "")).lower()
                        if ("close" in action) or ("exit" in action):
                            try:
                                pnl = float(
                                    t.get("realized_pnl", t.get("pnl", t.get("profit", 0.0))) or 0.0
                                )
                            except Exception:
                                pnl = 0.0
                            if pnl < 0.0 and last_loss_ts <= 0.0:
                                last_loss_ts = ts

                    if (not daily_trades) and computed_daily_entries > 0:
                        daily_trades = computed_daily_entries
                    if (not session_trades) and computed_daily_entries > 0:
                        session_trades = computed_daily_entries

                    # Store fallbacks for timing if missing
                    if not hasattr(self, "_mask_last_entry_ts"):
                        self._mask_last_entry_ts = 0.0
                    if last_entry_ts > 0.0:
                        self._mask_last_entry_ts = last_entry_ts
                    if not hasattr(self, "_mask_last_loss_ts"):
                        self._mask_last_loss_ts = 0.0
                    if last_loss_ts > 0.0:
                        self._mask_last_loss_ts = last_loss_ts
            except Exception:
                pass

            # Get timing info
            timing_state = self.smart_bus.get("timing_state", "PPOAgentShell", default={})
            last_entry_time = None
            last_loss_time = None
            if isinstance(timing_state, dict):
                last_entry_str = timing_state.get("last_entry_time")      
                last_loss_str = timing_state.get("last_loss_time")        
                if last_entry_str:
                    try:
                        last_entry_time = datetime.fromisoformat(last_entry_str)
                    except Exception:
                        pass
                if last_loss_str:
                    try:
                        last_loss_time = datetime.fromisoformat(last_loss_str)
                    except Exception:
                        pass

            # If timing_state isn't published, fall back to derived timestamps
            if last_entry_time is None:
                try:
                    ts = float(getattr(self, "_mask_last_entry_ts", 0.0) or 0.0)
                    if ts > 0.0:
                        last_entry_time = datetime.fromtimestamp(ts)
                except Exception:
                    pass
            if last_loss_time is None:
                try:
                    ts = float(getattr(self, "_mask_last_loss_ts", 0.0) or 0.0)
                    if ts > 0.0:
                        last_loss_time = datetime.fromtimestamp(ts)
                except Exception:
                    pass

            # Build mask
            mask = self._live_mask_builder.get_action_mask(
                has_position=has_position,
                trade_open_allowed=bool(getattr(self, "_trade_open_allowed", True)),
                has_pending_entry=False,  # Not tracked in live (fill is instant)
                has_pending_exit=False,
                current_dd=current_dd,
                daily_dd=daily_dd,
                daily_trades=daily_trades,
                session_trades=session_trades,
                consecutive_losses=consecutive_losses,
                last_entry_time=last_entry_time,
                last_loss_time=last_loss_time,
                current_time=datetime.now(),
            )
            
            return mask
            
        except Exception as e:
            # On error, FAIL CLOSED: allow HOLD always; allow CLOSE only if a position exists.
            self.logger.warning(f"[PPO] Action mask error: {e}, failing closed")
            n_actions = int(self._live_mask_builder.config.n_actions)
            safe_mask = np.zeros(n_actions, dtype=np.bool_)
            safe_mask[0] = True  # HOLD is always allowed
            safe_mask[-1] = bool(has_position)  # CLOSE (n_actions-1) only if in a position
            return safe_mask

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

    def _read_position_context_raw(self) -> Optional[Dict[str, Any]]:
        """
        Read raw position context from bus regardless of focus_mode_active.
        v5.6: We need real position data to build side-aware snapshots even when
        focus_mode isn't explicitly enabled.
        """
        ctx = self.smart_bus.get(
            "position_focus_context",
            "PPOAgentShell",
            default=None,
        )
        return ctx if isinstance(ctx, dict) else None

    def _build_positions_by_instrument(self, position_ctx_raw: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Build side-aware positions dict for arbiter.
        v5.6: Returns dict of instrument -> {has, side, lots, age_hours, unrealized_pnl}
        """
        result: Dict[str, Dict[str, Any]] = {}
        if not position_ctx_raw:
            return result

        all_positions = position_ctx_raw.get("positions", {})
        if not isinstance(all_positions, dict):
            return result

        for inst in self._cfg.instruments:
            inst_norm = _norm_symbol(inst)
            pos_data = all_positions.get(inst) or all_positions.get(inst_norm) or {}
            if isinstance(pos_data, dict) and int(pos_data.get("side", 0)) != 0:
                result[inst] = {
                    "has": True,
                    "side": int(pos_data.get("side", 0)),
                    "lots": float(pos_data.get("lots", 0.0)),
                    "age_hours": float(pos_data.get("age_hours", 0.0)),
                    "unrealized_pnl": float(pos_data.get("unrealized_pnl", 0.0)),
                }
            else:
                result[inst] = {"has": False, "side": 0, "lots": 0.0, "age_hours": 0.0, "unrealized_pnl": 0.0}
        return result

    def _update_trade_open_gate(
        self,
        *,
        in_warmup: bool,
        memory_info: MemoryGateInfo,
        risk_info: RiskInfo,
        positions_by_instrument: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Compute and cache a single "trade_open_allowed" gate for NEW entries.

        This gate is consumed by:
        - MaskablePPO action masking (prevents PPO loopholes)
        - A final post-decision safety pass (prevents accidental entry)
        """
        # Determine whether we should fail-closed (live/paper) vs fail-open (training/backtest).
        exec_mode = None
        try:
            exec_mode = self.smart_bus.get("execution_mode", "PPOAgentShell", default=None)
        except Exception:
            exec_mode = None
        exec_mode_str = str(exec_mode or "").lower()
        is_live_exec = exec_mode_str in ("live", "paper")

        # EntryTimingController output (per instrument)
        try:
            entry_timing_all = self.smart_bus.get("entry_timing", "PPOAgentShell", default={}) or {}
        except Exception:
            entry_timing_all = {}
        if not isinstance(entry_timing_all, dict):
            entry_timing_all = {}

        # Instrument cooldown state (per instrument)
        try:
            cooldown_state = self.smart_bus.get(
                "instrument_cooldown_state",
                "PPOAgentShell",
                default={},
            ) or {}
        except Exception:
            cooldown_state = {}
        if not isinstance(cooldown_state, dict):
            cooldown_state = {}

        # Seasonality gate (authoritative for time windows)
        seasonality_allowed = True
        seasonality_reason = "unknown"
        try:
            seasonality_allowed, seasonality_reason = self.arbiter._check_seasonality_time_gate()
        except Exception as e:  # noqa: BLE001
            seasonality_allowed = False if is_live_exec else True
            seasonality_reason = f"error:{type(e).__name__}"

        # Optional entry cadence gate: only allow NEW entries once per new PRIMARY_TIMEFRAME bar.
        # This is a live-only safety knob (training already advances per bar).
        require_new_primary_bar = bool(getattr(self._cfg, "entries_require_new_primary_bar", False)) and bool(is_live_exec)
        primary_bar_ts_by_inst: Dict[str, Optional[str]] = {}
        if require_new_primary_bar:
            for inst in self._cfg.instruments:
                alias_key = f"market_data_{inst.replace('/', '_')}_{PRIMARY_TIMEFRAME}"
                try:
                    bar_blob = self.smart_bus.get(alias_key, "PPOAgentShell", default=None)
                except Exception:
                    bar_blob = None
                ts_raw = bar_blob.get("timestamp") if isinstance(bar_blob, dict) else None
                if isinstance(ts_raw, datetime):
                    ts_key = ts_raw.isoformat()
                elif isinstance(ts_raw, str) and ts_raw:
                    ts_key = ts_raw
                else:
                    ts_key = None
                primary_bar_ts_by_inst[inst] = ts_key

        def _lookup_inst(d: Dict[str, Any], inst: str) -> Any:
            if inst in d:
                return d.get(inst)
            inst_norm = _norm_symbol(inst)
            for k, v in d.items():
                if isinstance(k, str) and _norm_symbol(k) == inst_norm:
                    return v
            return None

        per_inst: Dict[str, Dict[str, Any]] = {}
        any_allowed = False

        for inst in self._cfg.instruments:
            has_pos = bool(positions_by_instrument.get(inst, {}).get("has", False))
            reasons: List[str] = []

            # This gate controls NEW entries only; position management is allowed.
            if not has_pos:
                if in_warmup:
                    reasons.append("WARMUP_BLOCKED")

                if not seasonality_allowed:
                    reasons.append(f"SEASONALITY_BLOCKED({seasonality_reason})")

                if bool(getattr(memory_info, "veto", False)):
                    reasons.append("MEMORY_VETO")

                if bool(getattr(risk_info, "hard_block", False)):
                    reasons.append("RISK_HARD_BLOCK")
                if bool(getattr(risk_info, "emergency_mode", False)):
                    reasons.append("RISK_EMERGENCY_MODE")

                cd = _lookup_inst(cooldown_state, inst)
                if isinstance(cd, dict) and bool(cd.get("on_cooldown", False)):
                    rem = float(cd.get("cooldown_remaining", 0.0) or 0.0)
                    reasons.append(f"COOLDOWN({rem:.0f}s)")

                timing = _lookup_inst(entry_timing_all, inst)
                if not isinstance(timing, dict):
                    if is_live_exec:
                        reasons.append("ENTRY_TIMING_MISSING")
                else:
                    if not bool(timing.get("entry_allowed", True)):
                        block_reasons = timing.get("block_reasons") or timing.get("reasons") or []
                        if isinstance(block_reasons, list) and block_reasons:
                            br = ",".join(str(x) for x in block_reasons[:4])
                            reasons.append(f"ENTRY_TIMING_BLOCKED({br})")
                        else:
                            reasons.append("ENTRY_TIMING_BLOCKED")

                if require_new_primary_bar:
                    ts_key = primary_bar_ts_by_inst.get(inst)
                    if not ts_key:
                        reasons.append(f"PRIMARY_BAR_TS_MISSING({PRIMARY_TIMEFRAME})")
                    else:
                        last_seen = self._last_seen_primary_bar_ts.get(inst)
                        if last_seen == ts_key:
                            reasons.append(f"SAME_PRIMARY_BAR({PRIMARY_TIMEFRAME})")

            allowed = len(reasons) == 0
            per_inst[inst] = {
                "trade_open_allowed": bool(allowed),
                "reasons": reasons,
                "has_position": bool(has_pos),
            }
            if allowed:
                any_allowed = True

        # Update "last seen" bar timestamps after evaluation so the first cycle that sees a new bar is eligible.
        if require_new_primary_bar:
            for inst in self._cfg.instruments:
                ts_key = primary_bar_ts_by_inst.get(inst)
                if ts_key:
                    self._last_seen_primary_bar_ts[inst] = ts_key

        # Global view: single-instrument pipeline uses the primary instrument gate.
        primary_gate = per_inst.get(self._cfg.primary_instrument, {})
        trade_open_allowed = bool(primary_gate.get("trade_open_allowed", any_allowed))

        gate: Dict[str, Any] = {
            "trade_open_allowed": trade_open_allowed,
            "primary_instrument": self._cfg.primary_instrument,
            "per_instrument": per_inst,
            "seasonality_allowed": bool(seasonality_allowed),
            "seasonality_reason": seasonality_reason,
            "in_warmup": bool(in_warmup),
            "ts": datetime.utcnow().isoformat() + "Z",
        }

        self._trade_open_allowed = trade_open_allowed
        self._trade_open_gate = gate

        # Publish for transparency/debugging (best-effort; never blocks trading)
        try:
            self.smart_bus.set(
                "trade_open_allowed",
                trade_open_allowed,
                module="PPOAgent",
                thesis="Global gate for NEW entries",
            )
            self.smart_bus.set(
                "trade_open_gate",
                gate,
                module="PPOAgent",
                thesis="Trade-open gate details (NEW entries only)",
            )
        except Exception:
            pass

    def _apply_trade_open_gate_to_decision(
        self,
        *,
        multi_decision: ArbiterMultiDecision,
        positions_by_instrument: Dict[str, Dict[str, Any]],
    ) -> ArbiterMultiDecision:
        """Force-flat any NEW entry when trade_open_allowed is false (fail-safe)."""
        gate = self._trade_open_gate or {}
        per_inst = gate.get("per_instrument", {}) if isinstance(gate, dict) else {}

        # Attach to global meta for traceability
        try:
            if isinstance(multi_decision.global_meta, dict):
                multi_decision.global_meta["trade_open_gate"] = gate
        except Exception:
            pass

        for inst, decision in multi_decision.instruments.items():
            has_pos = bool(positions_by_instrument.get(inst, {}).get("has", False))
            if has_pos:
                continue
            if decision.direction not in ("long", "short"):
                continue

            inst_gate = per_inst.get(inst) if isinstance(per_inst, dict) else None
            if isinstance(inst_gate, dict) and "trade_open_allowed" in inst_gate:
                inst_allowed = bool(inst_gate.get("trade_open_allowed", True))
            else:
                inst_allowed = bool(self._trade_open_allowed)

            if inst_allowed:
                continue

            # Block NEW entries, keep position management semantics intact.
            decision.direction = "flat"
            decision.gate_passed = False
            decision.position_size = 0.0

            if decision.meta is None:
                decision.meta = {}
            decision.meta["trade_open_gate_blocked"] = True
            decision.meta["trade_open_gate"] = inst_gate if isinstance(inst_gate, dict) else {}

            if hasattr(decision, "gate_reasons") and isinstance(decision.gate_reasons, list):
                decision.gate_reasons.append("TRADE_OPEN_GATE_BLOCKED")
                gate_reasons = inst_gate.get("reasons") if isinstance(inst_gate, dict) else None
                if isinstance(gate_reasons, list):
                    for r in gate_reasons[:6]:
                        decision.gate_reasons.append(str(r))

            original_reasoning = decision.reasoning or ""
            decision.reasoning = f"[TRADE_OPEN_GATE_BLOCKED] {original_reasoning}"

        return multi_decision

    def _log_instrument_cooldowns(self) -> None:
        """
        Log per-instrument cooldown state for transparency.

        Reads the SmartPositionManager's 'instrument_cooldown_state' from the
        SmartInfoBus and logs when an instrument enters or exits cooldown,
        including the remaining cooldown time in seconds.
        """
        try:
            cooldown_state = self.smart_bus.get(
                "instrument_cooldown_state",
                "PPOAgentShell",
                default={},
            ) or {}
            if not isinstance(cooldown_state, dict):
                return
        except Exception:
            return

        for inst in self._cfg.instruments:
            inst_cd = cooldown_state.get(inst)
            if not isinstance(inst_cd, dict):
                norm = inst.replace("/", "").replace("_", "").upper()
                for key, val in cooldown_state.items():
                    if not isinstance(key, str) or not isinstance(val, dict):
                        continue
                    key_norm = key.replace("/", "").replace("_", "").upper()
                    if key_norm == norm:
                        inst_cd = val
                        break

            on_cd = bool(inst_cd.get("on_cooldown", False)) if isinstance(inst_cd, dict) else False
            remaining = float(inst_cd.get("cooldown_remaining", 0.0) or 0.0) if isinstance(inst_cd, dict) else 0.0
            prev = self._last_cooldown_state.get(inst, False)

            if on_cd and remaining > 0.0 and not prev:
                self.logger.info(f"[PPO] ═══ {inst} ═══ COOLDOWN ACTIVE")
                self.logger.info(
                    f"[PPO]   ⏸️ Trade cooldown: {remaining:.0f}s remaining"
                )
            elif not on_cd and prev:
                self.logger.info(f"[PPO] ═══ {inst} ═══ COOLDOWN CLEARED")
                self.logger.info(
                    f"[PPO]   ✅ Trade cooldown finished"
                )

            self._last_cooldown_state[inst] = on_cd

    def _log_position_active_box(
        self,
        instrument: str,
        side_emoji: str,
        side_str: str,
        lots: float,
        pnl_emoji: str,
        pnl: float,
        age_hours: float,
        action_str: str,
        confidence: float,
        trust: float,
        regime: str,
    ) -> None:
        """Log a boxed summary for an active position."""
        try:
            header = f"{instrument} – POSITION ACTIVE"
            box_width = 78
            top = f"[PPO] ┌─ {header} " + "─" * max(0, box_width - len(header) - 3)
            mid1 = (
                f"[PPO] │  {side_emoji} {side_str} {lots:.2f} lots │ "
                f"{pnl_emoji} P&L: €{pnl:+.2f} │ Age: {age_hours:.1f}h"
            )
            mid2 = (
                f"[PPO] │  Signal: {action_str} │ "
                f"Conf: {confidence:.0%} │ Trust: {trust:.2f} │ Regime: {regime}"
            )
            bot = "[PPO] └" + "─" * (box_width - 1)
            for line in (top, mid1, mid2, bot, ""):
                self.logger.info(line)
        except Exception:
            # Box logging is best-effort only; never affect trading.
            pass

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
        position_ctx: Mapping[str, Any],
    ) -> ArbiterMultiDecision:
        """
        Apply position focus mode constraints to the multi-instrument decision.

        v5.6 PRINCIPLES:
        - Instruments WITH positions: DO NOT overwrite PPO direction/intent.
          The arbiter already produced exit/reverse decisions using real position
          data. We only add metadata hints for downstream modules.
        - Instruments WITHOUT positions:
          * If position_focus_blocks_other_instruments=True: block new entries
          * Otherwise: trade independently

        This prevents destroying PPO's exit/take-profit signals which are critical
        for profitable position management.
        """
        # Get ALL positions, not just the primary
        all_positions = position_ctx.get("positions", {})
        primary_inst = position_ctx.get("primary_instrument")

        # Check if we should block other instruments (configurable)
        block_other_instruments = self._cfg.position_focus_blocks_other_instruments

        for inst, decision in multi_decision.instruments.items():
            # Defensive: ensure meta exists
            if decision.meta is None:
                decision.meta = {}

            # Normalize instrument for lookup
            inst_norm = _norm_symbol(inst)
            inst_position = all_positions.get(inst) or all_positions.get(inst_norm) or {}
            position_side = int(inst_position.get("side", 0)) if isinstance(inst_position, dict) else 0
            position_pnl = float(inst_position.get("unrealized_pnl", 0.0)) if isinstance(inst_position, dict) else 0.0

            if position_side != 0:
                # ──────────────────────────────────────────────────────────────
                # v5.6: This instrument HAS a position.
                # DO NOT overwrite direction/gate. PPO arbiter already made the
                # correct decision using PositionSnapshot. We only add hints.
                # ──────────────────────────────────────────────────────────────
                ppo_direction = (decision.direction or "hold").lower()
                action_intent = decision.meta.get("action_intent", "unknown")

                if position_side > 0:  # LONG position
                    supports = ppo_direction in ("long", "buy", "hold", "flat")
                else:  # SHORT position
                    supports = ppo_direction in ("short", "sell", "hold", "flat")

                # Add metadata hints for downstream modules (SmartPositionManager)
                decision.meta["position_focus_mode"] = True
                decision.meta["supports_position"] = supports
                decision.meta["position_pnl"] = position_pnl
                decision.meta["action_intent"] = action_intent

                if supports:
                    decision.meta["position_focus_hint"] = "hold_or_scale_carefully"
                else:
                    # PPO wants opposite direction → exit pressure
                    if position_pnl > 0:
                        decision.meta["position_focus_hint"] = "take_profit_candidate"
                    else:
                        decision.meta["position_focus_hint"] = "cut_loss_candidate"

                # DO NOT modify: decision.direction, decision.gate_passed, decision.position_size
                # The arbiter made the correct call with real position data.

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
                    decision.meta["position_focus_mode"] = False
                    decision.meta["blocked_by_position_focus"] = False

        # Update global metadata
        primary_pnl = float(position_ctx.get("primary_pnl", 0.0))
        primary_side = int(position_ctx.get("primary_side", 0))
        multi_decision.global_meta["position_focus"] = {
            "active": len(all_positions) > 0,
            "instrument": primary_inst,
            "side": primary_side,
            "pnl": primary_pnl,
            "blocks_other_instruments": block_other_instruments,
            "positions_count": len(all_positions),
        }

        return multi_decision

    # ─────────────────────────────────────────────────────────────
    # Observation Building (v3.0+ with Frame Stacking v3.3.0)
    # ─────────────────────────────────────────────────────────────

    def _build_observations_for_instruments(self) -> Dict[str, np.ndarray]:
        """
        Build observation vectors for each instrument WITH FRAME STACKING.

        Uses the v3.0 observation builder which supports per-instrument
        observations. Then applies frame stacking to match VecFrameStack
        training (n_stack=4 → 64×4=256 features).
        
        If per-instrument data is unavailable, falls back to zero vectors.
        """
        observations: Dict[str, np.ndarray] = {}
        base_obs_size = self._cfg.core_config.obs_size  # 64

        for instrument in self._cfg.instruments:
            try:
                # Build base observation (64 features)
                base_obs = self.obs_builder.build_for_instrument(
                    instrument=instrument,
                    smart_bus=self.smart_bus,
                    module_name="PPOAgentShell",
                )
                base_obs = np.asarray(base_obs, dtype=np.float32).reshape(-1)
                
                # Ensure correct size
                if base_obs.shape[0] != base_obs_size:
                    fixed = np.zeros(base_obs_size, dtype=np.float32)
                    copy_len = min(base_obs.shape[0], base_obs_size)
                    fixed[:copy_len] = base_obs[:copy_len]
                    base_obs = fixed
                
                # Apply frame stacking if enabled (disabled by default)
                if self._frame_stack_size > 1:
                    stacked_obs = self._apply_frame_stack(instrument, base_obs)
                    observations[instrument] = stacked_obs
                else:
                    observations[instrument] = base_obs
                
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"Failed to build observation for {instrument}: {e}"
                )
                # Return zeros (64 features, or stacked if enabled)
                obs_dim = base_obs_size * self._frame_stack_size if self._frame_stack_size > 1 else base_obs_size
                observations[instrument] = np.zeros(obs_dim, dtype=np.float32)

        return observations
    
    def _apply_frame_stack(self, instrument: str, new_obs: np.ndarray) -> np.ndarray:
        """
        Apply frame stacking for an instrument.
        
        Maintains a buffer of the last N observations and stacks them
        to create a larger observation vector (matching VecFrameStack).
        
        Frame order: [oldest, ..., newest] concatenated
        E.g., with n_stack=4 and obs_size=64: output is 256 features
        """
        # Initialize buffer if needed
        if instrument not in self._obs_frame_buffer:
            # Initialize with copies of first observation (avoid zeros at start)
            self._obs_frame_buffer[instrument] = [
                new_obs.copy() for _ in range(self._frame_stack_size)
            ]
        
        # Get buffer
        buffer = self._obs_frame_buffer[instrument]
        
        # Shift buffer (remove oldest, add newest)
        buffer.pop(0)
        buffer.append(new_obs.copy())
        
        # Stack all frames: [oldest, ..., newest]
        stacked = np.concatenate(buffer, axis=0)
        
        return stacked.astype(np.float32)

    def _maybe_log_observation_diagnostics(self, observations: Dict[str, np.ndarray]) -> None:
        """
        Log lightweight per-instrument observation summaries and detect when
        multiple instruments are effectively getting the same observation.

        This is intentionally throttled to avoid log spam.
        """
        try:
            if not bool(getattr(self._cfg, "obs_diagnostics_enabled", True)):
                return
            if not observations:
                return

            cycle = int(getattr(self, "_session_cycle_count", 0) or 0)
            every_n = int(getattr(self._cfg, "obs_diagnostics_every_n_cycles", 20) or 0)
            periodic = (cycle == 1) or (self.debug and every_n > 0 and cycle % every_n == 0)

            suspicious = False
            summaries: Dict[str, Dict[str, Any]] = {}

            for inst, obs in observations.items():
                arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                if arr.size <= 0:
                    continue

                # Group summaries (mean abs) – these should differ across instruments if market data is correct.
                def _g_abs(group: str) -> float:
                    start, end = FEATURE_GROUPS[group]
                    seg = arr[start:end]
                    return float(np.mean(np.abs(seg))) if seg.size else 0.0

                zero_pct = float(np.mean(arr == 0.0))
                m15_abs = _g_abs("m15_price")
                htf_abs = _g_abs("htf_context")
                acct_abs = _g_abs("account")
                mode_abs = _g_abs("trading_mode")
                gov_abs = _g_abs("governor")

                prev = self._obs_diag_prev.get(inst)
                delta_abs = None
                if isinstance(prev, np.ndarray) and prev.shape == arr.shape:
                    delta_abs = float(np.mean(np.abs(arr - prev)))
                self._obs_diag_prev[inst] = arr.copy()

                summaries[inst] = {
                    "zero_pct": zero_pct,
                    "m15_abs": m15_abs,
                    "htf_abs": htf_abs,
                    "acct_abs": acct_abs,
                    "mode_abs": mode_abs,
                    "gov_abs": gov_abs,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "delta_abs": delta_abs,
                }

                # Heuristics: missing market data usually means price groups are all zeros.
                if m15_abs < 1e-8 or htf_abs < 1e-8:
                    suspicious = True
                if gov_abs < 1e-8:
                    suspicious = True
                if zero_pct > 0.95:
                    suspicious = True
                if delta_abs is not None and delta_abs < 1e-10 and cycle > 10:  
                    suspicious = True

            # Pairwise similarity (use config order for stable logs)
            similarity: Optional[Dict[str, Any]] = None
            insts = [i for i in self._cfg.instruments if i in observations]
            if len(insts) >= 2:
                a = np.asarray(observations[insts[0]], dtype=np.float32).reshape(-1)
                b = np.asarray(observations[insts[1]], dtype=np.float32).reshape(-1)
                if a.size == b.size and a.size > 0:
                    mean_abs_diff = float(np.mean(np.abs(a - b)))
                    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                    cos = float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
                    similarity = {
                        "a": insts[0],
                        "b": insts[1],
                        "mean_abs_diff": mean_abs_diff,
                        "cosine": cos,
                    }
                    if mean_abs_diff < 1e-6 or cos > 0.99999:
                        suspicious = True

            if not (periodic or suspicious):
                return

            # Lightweight market sanity: last prices per instrument.
            price_data = None
            try:
                price_data = self.smart_bus.get_readonly_ref("price_data", "PPOAgentShell", default=None)
            except Exception:
                try:
                    price_data = self.smart_bus.get("price_data", "PPOAgentShell", default=None)
                except Exception:
                    price_data = None

            def _lookup_block(mapping: Any, instrument: str) -> Optional[Dict[str, Any]]:
                if not isinstance(mapping, dict):
                    return None
                direct = mapping.get(instrument)
                if isinstance(direct, dict):
                    return direct
                target = _norm_symbol(instrument)
                best: Optional[Dict[str, Any]] = None
                best_score: Optional[int] = None
                for k, v in mapping.items():
                    if not (isinstance(k, str) and isinstance(v, dict)):
                        continue
                    k_norm = _norm_symbol(k)
                    if not k_norm:
                        continue
                    if k_norm == target:
                        return v
                    if k_norm.startswith(target) or target.startswith(k_norm):
                        score = abs(len(k_norm) - len(target))
                        if best_score is None or score < best_score:
                            best = v
                            best_score = score
                return best

            def _last_price(instrument: str) -> Optional[float]:
                block = _lookup_block(price_data, instrument)
                if not isinstance(block, dict):
                    return None
                for k in ("last", "price", "close", "bid", "ask"):
                    v = block.get(k)
                    try:
                        if v is not None:
                            return float(v)
                    except Exception:
                        continue
                return None

            level_fn = self.logger.warning if suspicious else self.logger.info
            level_fn(
                f"[PPO][OBS] cycle={cycle} instruments={list(observations.keys())} "
                f"suspicious={suspicious}"
            )

            for inst in self._cfg.instruments:
                s = summaries.get(inst)
                if not s:
                    continue
                last = _last_price(inst)
                delta_str = f"{s['delta_abs']:.3e}" if s["delta_abs"] is not None else "n/a"
                level_fn(
                    f"[PPO][OBS] {inst}: m15={s['m15_abs']:.3f} htf={s['htf_abs']:.3f} "
                    f"acct={s['acct_abs']:.3f} mode={s['mode_abs']:.3f} gov={s['gov_abs']:.3f} "
                    f"zero={s['zero_pct']:.0%} Δ={delta_str} "
                    f"min={s['min']:.2f} max={s['max']:.2f} last={last}"
                )

            if similarity:
                level_fn(
                    f"[PPO][OBS] similarity {similarity['a']} vs {similarity['b']}: "
                    f"mean|Δ|={similarity['mean_abs_diff']:.3e} cos={similarity['cosine']:.6f}"
                )

        except Exception as e:  # noqa: BLE001
            # Diagnostics must never block trading.
            try:
                self.logger.debug(f"[PPO][OBS] diagnostics failed: {e}")
            except Exception:
                pass

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
