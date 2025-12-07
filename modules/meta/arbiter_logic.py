#!/usr/bin/env python3
"""
Arbiter Logic - Per-Instrument Decision Making
===============================================

This module contains the domain-specific arbiter logic that:
- Knows what instruments are and how to trade them
- Builds per-instrument observations
- Applies gating logic (risk/memory)
- Produces structured InstrumentDecision objects

It sits between PPOCore (pure RL) and PPOAgentShell (SmartInfoBus gateway).

Version: 3.0.0 (Multi-instrument architecture)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.meta.ppo_types import (
    InstrumentDecision,
    ArbiterMultiDecision,
    MemoryGateInfo,
    RiskInfo,
    GatingResult,
    InstrumentStatsTracker,
    DEFAULT_INSTRUMENTS,
    PRIMARY_INSTRUMENT,
)
from modules.meta.ppo_core import PPOCore


# ═══════════════════════════════════════════════════════════════════
# STRATEGY INTEGRATION TYPES
# ═══════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import Optional as OptType


@dataclass
class StrategyInfo:
    """
    Normalized strategy module information for arbiter decisions.
    
    Integrates outputs from:
    - BiasAuditor: Psychological bias adjustments
    - CurriculumPlannerPlus: Learning stage constraints
    - ThesisEvolutionEngine: Best thesis recommendations
    """
    # BiasAuditor outputs
    bias_position_multiplier: float = 1.0  # from bias_adjustments.position_size_multiplier
    bias_risk_reduction: float = 0.0       # from bias_adjustments.risk_reduction_factor
    active_biases: List[str] = field(default_factory=list)  # from bias_analysis.individual_biases
    psychological_state: str = "neutral"   # from psychological_state.state
    
    # CurriculumPlannerPlus outputs
    curriculum_stage: str = "Foundation"   # from curriculum_stage.name
    stage_difficulty: float = 1.0          # from curriculum_stage.difficulty
    max_position_size: float = 1.0         # from learning_constraints.max_position_size
    max_trades_per_day: int = 20           # from learning_constraints.max_trades_per_day
    mastery_level: float = 0.5             # from mastery_assessment.mastery_level
    
    # ThesisEvolutionEngine outputs
    best_thesis: str = ""                  # from best_thesis.thesis
    thesis_confidence: float = 0.5         # from best_thesis.confidence
    thesis_regime_alignment: bool = True   # from best_thesis.regime_aligned
    
    def __post_init__(self):
        if self.active_biases is None:
            self.active_biases = []
    
    @classmethod
    def from_bus_data(
        cls,
        bias_adjustments: OptType[Dict[str, Any]] = None,
        bias_analysis: OptType[Dict[str, Any]] = None,
        psychological_state: OptType[Dict[str, Any]] = None,
        curriculum_stage: OptType[Dict[str, Any]] = None,
        learning_constraints: OptType[Dict[str, Any]] = None,
        mastery_assessment: OptType[Dict[str, Any]] = None,
        best_thesis: OptType[Dict[str, Any]] = None,
    ) -> "StrategyInfo":
        """Create StrategyInfo from SmartInfoBus data."""
        # Bias adjustments
        bias_adj = bias_adjustments or {}
        pos_mult = float(bias_adj.get("position_size_multiplier", 1.0) or 1.0)
        risk_red = float(bias_adj.get("risk_reduction_factor", 0.0) or 0.0)
        
        # Bias analysis
        bias_anal = bias_analysis or {}
        active = []
        ind_biases = bias_anal.get("individual_biases", {})
        if isinstance(ind_biases, dict):
            active = [k for k, v in ind_biases.items() if isinstance(v, dict) and v.get("detected", False)]
        
        # Psychological state
        psych = psychological_state or {}
        psych_state = str(psych.get("state", "neutral") or "neutral")
        
        # Curriculum stage
        curr_stage = curriculum_stage or {}
        stage_name = str(curr_stage.get("name", "Foundation") or "Foundation")
        stage_diff = float(curr_stage.get("difficulty", 1.0) or 1.0)
        
        # Learning constraints
        constraints = learning_constraints or {}
        max_pos = float(constraints.get("max_position_size", 1.0) or 1.0)
        max_trades = int(constraints.get("max_trades_per_day", 20) or 20)
        
        # Mastery assessment
        mastery = mastery_assessment or {}
        mastery_lvl = float(mastery.get("mastery_level", 0.5) or 0.5)
        
        # Best thesis
        thesis = best_thesis or {}
        thesis_text = str(thesis.get("thesis", "") or "")
        thesis_conf = float(thesis.get("confidence", 0.5) or 0.5)
        thesis_aligned = bool(thesis.get("regime_aligned", True))
        
        return cls(
            bias_position_multiplier=np.clip(pos_mult, 0.1, 1.0),
            bias_risk_reduction=np.clip(risk_red, 0.0, 0.5),
            active_biases=active,
            psychological_state=psych_state,
            curriculum_stage=stage_name,
            stage_difficulty=stage_diff,
            max_position_size=np.clip(max_pos, 0.1, 2.0),
            max_trades_per_day=max(1, min(max_trades, 50)),
            mastery_level=np.clip(mastery_lvl, 0.0, 1.0),
            best_thesis=thesis_text[:200] if thesis_text else "",
            thesis_confidence=np.clip(thesis_conf, 0.0, 1.0),
            thesis_regime_alignment=thesis_aligned,
        )


@dataclass
class TradingModeInfo:
    """
    Normalized trading mode information for arbiter decisions.
    
    Integrates outputs from TradingModeManager for intelligent position sizing
    and risk adjustment based on current market conditions.
    """
    # Mode identification
    mode_name: str = "normal"               # safe, normal, aggressive, extreme
    mode_confidence: float = 0.5            # how confident the system is in this mode
    
    # Risk parameters from mode_config
    risk_multiplier: float = 1.0            # from mode_config.risk_multiplier
    max_exposure: float = 0.5               # from mode_config.max_exposure
    position_scale: float = 1.0             # from mode_config.position_scale
    stop_loss_multiplier: float = 1.0       # from mode_config.stop_loss_multiplier
    
    # Mode effectiveness metrics
    effectiveness: float = 0.5              # from mode_effectiveness
    win_rate_in_mode: float = 0.5           # historical performance in this mode
    
    # Decision factors that led to this mode
    volatility_factor: float = 0.5          # from decision_factors.volatility
    trend_factor: float = 0.5               # from decision_factors.trend
    regime_factor: float = 0.5              # from decision_factors.regime
    
    @classmethod
    def from_bus_data(
        cls,
        trading_mode: OptType[str] = None,
        mode_config: OptType[Dict[str, Any]] = None,
        mode_effectiveness: OptType[float] = None,
        decision_factors: OptType[Dict[str, Any]] = None,
    ) -> "TradingModeInfo":
        """Create TradingModeInfo from SmartInfoBus data."""
        # Mode name
        mode = str(trading_mode or "normal").lower()
        if mode not in ("safe", "normal", "aggressive", "extreme"):
            mode = "normal"
        
        # Mode config
        config = mode_config or {}
        risk_mult = float(config.get("risk_multiplier", 1.0) or 1.0)
        max_exp = float(config.get("max_exposure", 0.5) or 0.5)
        pos_scale = float(config.get("position_scale", 1.0) or 1.0)
        sl_mult = float(config.get("stop_loss_multiplier", 1.0) or 1.0)
        
        # Effectiveness
        eff = float(mode_effectiveness if mode_effectiveness is not None else 0.5)
        
        # Decision factors
        factors = decision_factors or {}
        vol_factor = float(factors.get("volatility", 0.5) or 0.5)
        trend_factor = float(factors.get("trend", 0.5) or 0.5)
        regime_factor = float(factors.get("regime", 0.5) or 0.5)
        
        # Mode confidence based on effectiveness
        mode_conf = eff if eff > 0 else 0.5
        
        return cls(
            mode_name=mode,
            mode_confidence=np.clip(mode_conf, 0.0, 1.0),
            risk_multiplier=np.clip(risk_mult, 0.25, 4.0),
            max_exposure=np.clip(max_exp, 0.1, 1.0),
            position_scale=np.clip(pos_scale, 0.25, 2.0),
            stop_loss_multiplier=np.clip(sl_mult, 0.5, 2.0),
            effectiveness=np.clip(eff, 0.0, 1.0),
            win_rate_in_mode=0.5,  # Could be fetched from analytics
            volatility_factor=np.clip(vol_factor, 0.0, 1.0),
            trend_factor=np.clip(trend_factor, 0.0, 1.0),
            regime_factor=np.clip(regime_factor, 0.0, 1.0),
        )
    
    def should_reduce_position(self) -> bool:
        """Check if current mode suggests position reduction."""
        return self.mode_name == "safe" or self.effectiveness < 0.3
    
    def should_increase_position(self) -> bool:
        """Check if current mode allows position increase."""
        return self.mode_name == "aggressive" and self.effectiveness > 0.6


@dataclass
class WorldModelInfo:
    """
    Normalized world model information for arbiter decisions.
    
    Integrates outputs from EnhancedWorldModel for predictive trading decisions
    based on LSTM price/volatility/regime forecasts.
    
    Price changes are in array format [M15, H1, H4, D1] where M15 is the primary
    decision-making timeframe.
    """
    # Model status
    is_trained: bool = False                # whether the model has been trained
    prediction_confidence: float = 0.0      # overall confidence in predictions
    
    # Price predictions (M15 is primary, others are observers)
    price_change_m15: float = 0.0           # predicted M15 price change % (PRIMARY)
    price_change_1h: float = 0.0            # predicted 1h price change %
    price_change_4h: float = 0.0            # predicted 4h price change %
    price_change_1d: float = 0.0            # predicted 1d price change %
    price_direction: str = "neutral"        # bullish, bearish, neutral
    
    # Volatility predictions
    volatility_prediction: float = 0.5      # expected volatility (0=low, 1=high)
    volatility_change: float = 0.0          # predicted change in volatility
    
    # Regime predictions
    regime_prediction: str = "ranging"      # trending_up, trending_down, ranging, volatile
    regime_confidence: float = 0.5          # confidence in regime prediction
    
    # Scenario analysis
    best_scenario_probability: float = 0.0  # probability of best-case scenario
    worst_scenario_probability: float = 0.0 # probability of worst-case scenario
    expected_move: float = 0.0              # expected price move magnitude
    
    @classmethod
    def from_bus_data(
        cls,
        market_predictions: OptType[Dict[str, Any]] = None,
        prediction_confidence: OptType[float] = None,
        scenario_generation: OptType[Dict[str, Any]] = None,
        world_model_analytics: OptType[Dict[str, Any]] = None,
    ) -> "WorldModelInfo":
        """
        Create WorldModelInfo from SmartInfoBus data.
        
        The world model provides:
        - market_predictions.latest_predictions.price_changes: [4 values] for M15, H1, H4, D1
        - market_predictions.latest_predictions.volatility_predictions: [4 values]
        - market_predictions.latest_predictions.regime_probabilities: [4 values] for regime classes
        - market_predictions.latest_predictions.confidence: float
        - market_predictions.is_trained: bool
        - market_predictions.model_confidence: float
        """
        predictions = market_predictions or {}
        
        # Check if model is trained
        is_trained = bool(predictions.get("is_trained", False))
        model_confidence = float(predictions.get("model_confidence", 0.0) or 0.0)
        
        # Get latest predictions dict
        latest = predictions.get("latest_predictions", {})
        
        # Price changes: array of [M15, H1, H4, D1] or fallback to empty
        price_changes = latest.get("price_changes", [])
        if isinstance(price_changes, (list, tuple)) and len(price_changes) >= 4:
            price_m15 = float(price_changes[0])
            price_1h = float(price_changes[1])
            price_4h = float(price_changes[2])
            price_1d = float(price_changes[3])
        elif isinstance(price_changes, (list, tuple)) and len(price_changes) >= 1:
            # Fallback: use first value as M15, others as 0
            price_m15 = float(price_changes[0])
            price_1h = float(price_changes[1]) if len(price_changes) > 1 else 0.0
            price_4h = float(price_changes[2]) if len(price_changes) > 2 else 0.0
            price_1d = float(price_changes[3]) if len(price_changes) > 3 else 0.0
        else:
            price_m15, price_1h, price_4h, price_1d = 0.0, 0.0, 0.0, 0.0
        
        # Determine direction based on M15 (primary) with weighted average from higher TFs
        # Weight: M15=0.5, H1=0.25, H4=0.15, D1=0.10
        weighted_change = price_m15 * 0.5 + price_1h * 0.25 + price_4h * 0.15 + price_1d * 0.10
        if weighted_change > 0.0005:  # 0.05% threshold
            direction = "bullish"
        elif weighted_change < -0.0005:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Volatility predictions: array of [4 values]
        vol_preds = latest.get("volatility_predictions", [])
        if isinstance(vol_preds, (list, tuple)) and len(vol_preds) >= 1:
            vol_value = float(vol_preds[0])  # Use first as primary
            vol_change = float(vol_preds[1]) - float(vol_preds[0]) if len(vol_preds) > 1 else 0.0
        else:
            vol_value, vol_change = 0.5, 0.0
        
        # Regime predictions: array of [4 probabilities] for regime classes
        regime_probs = latest.get("regime_probabilities", [])
        predicted_regime_idx = latest.get("predicted_regime", -1)
        regime_names = ["trending_up", "trending_down", "ranging", "volatile"]
        
        if predicted_regime_idx >= 0 and predicted_regime_idx < len(regime_names):
            regime = regime_names[predicted_regime_idx]
            regime_conf = float(regime_probs[predicted_regime_idx]) if isinstance(regime_probs, (list, tuple)) and len(regime_probs) > predicted_regime_idx else 0.5
        else:
            regime = "ranging"
            regime_conf = 0.5
        
        # Prediction confidence from latest predictions or model confidence
        pred_conf = float(latest.get("confidence", model_confidence) or 0.0)
        
        # Also support direct prediction_confidence parameter
        if prediction_confidence is not None:
            if isinstance(prediction_confidence, dict):
                pred_conf = float(prediction_confidence.get("current_confidence", pred_conf) or pred_conf)
            else:
                pred_conf = float(prediction_confidence)
        
        # Scenarios
        scenarios = scenario_generation or {}
        scenarios_list = scenarios.get("scenarios", [])
        best_prob = 0.0
        worst_prob = 0.0
        expected = 0.0
        
        if isinstance(scenarios_list, list) and len(scenarios_list) > 0:
            # Scenarios have probabilities and outcomes
            for s in scenarios_list:
                if isinstance(s, dict):
                    prob = float(s.get("probability", 0.0) or 0.0)
                    outcome = float(s.get("outcome", 0.0) or 0.0)
                    expected += prob * outcome
                    if outcome > 0:
                        best_prob = max(best_prob, prob)
                    elif outcome < 0:
                        worst_prob = max(worst_prob, prob)
        
        return cls(
            is_trained=is_trained,
            prediction_confidence=np.clip(pred_conf, 0.0, 1.0),
            price_change_m15=np.clip(price_m15, -0.05, 0.05),  # M15 smaller range
            price_change_1h=np.clip(price_1h, -0.1, 0.1),
            price_change_4h=np.clip(price_4h, -0.2, 0.2),
            price_change_1d=np.clip(price_1d, -0.3, 0.3),
            price_direction=direction,
            volatility_prediction=np.clip(vol_value, 0.0, 1.0),
            volatility_change=np.clip(vol_change, -0.5, 0.5),
            regime_prediction=regime,
            regime_confidence=np.clip(regime_conf, 0.0, 1.0),
            best_scenario_probability=np.clip(best_prob, 0.0, 1.0),
            worst_scenario_probability=np.clip(worst_prob, 0.0, 1.0),
            expected_move=np.clip(expected, -0.5, 0.5),
        )
    
    def should_trust_predictions(self) -> bool:
        """Check if predictions are reliable enough to use."""
        return self.is_trained and self.prediction_confidence > 0.4
    
    def get_directional_bias(self) -> float:
        """Get directional bias from predictions (-1 to 1)."""
        if not self.should_trust_predictions():
            return 0.0
        if self.price_direction == "bullish":
            return min(1.0, self.prediction_confidence)
        elif self.price_direction == "bearish":
            return -min(1.0, self.prediction_confidence)
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# ARBITER LOGIC
# ═══════════════════════════════════════════════════════════════════

class ArbiterLogic:
    """
    Domain-specific arbiter logic for multi-instrument trading.
    
    Responsibilities:
    - Build per-instrument features from market data
    - Ask PPOCore for actions
    - Apply gating pipeline (risk/memory)
    - Produce structured InstrumentDecision objects
    - Track per-instrument statistics
    
    This class knows about:
    - Instruments (XAUUSD, EURUSD)
    - Committee/expert signals
    - Risk/memory gates
    
    But does NOT know about:
    - SmartInfoBus (that's PPOAgentShell's job)
    - Module lifecycle
    - Health monitoring
    """
    
    def __init__(
        self,
        ppo_core: PPOCore,
        instruments: Optional[List[str]] = None,
        debug: bool = False,
    ) -> None:
        self.ppo_core = ppo_core
        self.instruments = instruments or DEFAULT_INSTRUMENTS
        self.primary_instrument = PRIMARY_INSTRUMENT
        self.debug = debug
        
        # Per-instrument statistics
        self.stats_tracker = InstrumentStatsTracker()
        
        # Hysteresis state per instrument
        self._last_directions: Dict[str, str] = {inst: "flat" for inst in self.instruments}
        self._direction_hold_counts: Dict[str, int] = {inst: 0 for inst in self.instruments}
        
        # Decision history for explanation
        self._decision_history: Dict[str, List[InstrumentDecision]] = {
            inst: [] for inst in self.instruments
        }
    
    # ─────────────────────────────────────────────────────────────
    # Main Decision Method
    # ─────────────────────────────────────────────────────────────
    
    def make_multi_instrument_decision(
        self,
        observations: Dict[str, np.ndarray],
        committee_data: Dict[str, Any],
        expert_signals: Dict[str, Any],
        memory_info: MemoryGateInfo,
        risk_info: RiskInfo,
        instruments: Optional[List[str]] = None,
        strategy_info: Optional[StrategyInfo] = None,
        trading_mode_info: Optional[TradingModeInfo] = None,
        world_model_info: Optional[WorldModelInfo] = None,
    ) -> ArbiterMultiDecision:
        """
        Make trading decisions for multiple instruments.
        
        Args:
            observations: Per-instrument observation vectors {instrument: obs_vec}
            committee_data: Committee consensus data (global and/or per-instrument)
            expert_signals: Expert voting signals (global and/or per-instrument)
            memory_info: Normalized global memory gate info
            risk_info: Normalized global risk info
            instruments: Optional override instrument list
            strategy_info: Optional strategy module integration info
            trading_mode_info: Optional trading mode constraints
            world_model_info: Optional world model predictions
        
        Returns:
            ArbiterMultiDecision with decisions for all instruments
        """
        instruments = instruments or self.instruments
        decisions: Dict[str, InstrumentDecision] = {}
        
        # Use provided info or create defaults
        strat = strategy_info or StrategyInfo()
        tm_info = trading_mode_info or TradingModeInfo()
        wm_info = world_model_info or WorldModelInfo()
        
        for instrument in instruments:
            # Get observation for this instrument
            obs = observations.get(instrument)
            if obs is None:
                # Use primary instrument's observation as fallback
                obs = observations.get(self.primary_instrument)
            if obs is None:
                obs = np.zeros(self.ppo_core.config.obs_size, dtype=np.float32)
            
            # Get instrument-specific data
            inst_committee = self._extract_instrument_committee(committee_data, instrument)
            inst_experts = self._extract_instrument_experts(expert_signals, instrument)
            
            # Make decision for this instrument
            decision = self._make_single_instrument_decision(
                instrument=instrument,
                observation=obs,
                committee=inst_committee,
                experts=inst_experts,
                memory_info=memory_info,
                risk_info=risk_info,
                strategy_info=strat,
                trading_mode_info=tm_info,
                world_model_info=wm_info,
            )
            
            decisions[instrument] = decision
            
            # Record statistics
            self.stats_tracker.record_decision(decision)
        
        # Global metadata with strategy, trading mode, and world model integration
        global_meta = {
            "timestamp": datetime.now().isoformat(),
            "instruments_processed": len(instruments),
            "memory_gate_value": memory_info.risk_multiplier,
            "risk_portfolio": risk_info.portfolio_risk,
            "stats": self.stats_tracker.to_dict(),
            # Strategy integration metadata
            "strategy": {
                "curriculum_stage": strat.curriculum_stage,
                "stage_difficulty": strat.stage_difficulty,
                "mastery_level": strat.mastery_level,
                "bias_position_multiplier": strat.bias_position_multiplier,
                "active_biases": strat.active_biases,
                "psychological_state": strat.psychological_state,
                "thesis_confidence": strat.thesis_confidence,
            },
            # Trading mode metadata
            "trading_mode": {
                "mode": tm_info.mode_name,
                "risk_multiplier": tm_info.risk_multiplier,
                "max_exposure": tm_info.max_exposure,
                "position_scale": tm_info.position_scale,
                "effectiveness": tm_info.effectiveness,
                "should_reduce": tm_info.should_reduce_position(),
            },
            # World model metadata
            "world_model": {
                "is_trained": wm_info.is_trained,
                "prediction_confidence": wm_info.prediction_confidence,
                "price_direction": wm_info.price_direction,
                "regime_prediction": wm_info.regime_prediction,
                "volatility": wm_info.volatility_prediction,
                "directional_bias": wm_info.get_directional_bias(),
            },
        }
        
        return ArbiterMultiDecision(
            instruments=decisions,
            global_meta=global_meta,
        )
    
    def _make_single_instrument_decision(
        self,
        instrument: str,
        observation: np.ndarray,
        committee: Dict[str, Any],
        experts: Dict[str, Any],
        memory_info: MemoryGateInfo,
        risk_info: RiskInfo,
        strategy_info: Optional[StrategyInfo] = None,
        trading_mode_info: Optional[TradingModeInfo] = None,
        world_model_info: Optional[WorldModelInfo] = None,
    ) -> InstrumentDecision:
        """
        Make a trading decision for a single instrument.
        
        Steps:
        1. Run PPO policy to get trust_score and size_score
        2. Interpret trust_score to decide follow/override/uncertain
        3. Apply hysteresis to directional intention
        4. Apply gating pipeline (risk/memory)
        5. Compute final position size and build InstrumentDecision
        """
        # 1) Run PPO policy
        action, log_prob, value = self.ppo_core.select_action(observation)
        
        trust_score = float(action[0]) if len(action) > 0 else 0.0
        size_score = float(action[1]) if len(action) > 1 else 0.0
        
        # 2) Extract committee/expert info
        committee_action = str(committee.get("action", "hold")).lower()
        committee_confidence = float(committee.get("confidence", 0.5))
        expert_consensus, expert_confidence = self._compute_expert_consensus(experts)
        regime = str(committee.get("regime", experts.get("regime", "unknown")))
        regime_strength = float(
            committee.get("regime_strength", experts.get("regime_strength", 0.5))
        )
        
        # 3) Interpret trust_score into a raw directional intention
        direction, confidence, reasoning = self._interpret_trust_score(
            trust_score=trust_score,
            committee_action=committee_action,
            committee_confidence=committee_confidence,
            expert_consensus=expert_consensus,
            expert_confidence=expert_confidence,
            instrument=instrument,
        )
        
        # 4) Apply hysteresis to stabilize direction (intention-level)
        direction = self._apply_hysteresis(instrument, direction, trust_score)
        
        # 5) Apply gating pipeline (risk + memory)
        gating_result = GatingResult.apply_gates(memory_info, risk_info, trust_score)
        
        # Scale confidence according to gates
        confidence *= gating_result.confidence_multiplier
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        if not gating_result.gate_passed:
            reasoning += f" | GATED: {', '.join(gating_result.reasons)}"
        elif gating_result.soft_scaling_applied:
            reasoning += f" | SCALED: {', '.join(gating_result.reasons)}"
        
        # 6) Calculate position size (respecting cap and gate status)
        raw_size = (size_score + 1.0) / 2.0  # Map [-1,1] to [0,1]
        position_size = float(
            np.clip(
                raw_size * confidence * gating_result.position_size_cap,
                0.0,
                1.0,
            )
        )
        
        if not gating_result.gate_passed or direction == "flat":
            position_size = 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # STRATEGY MODULE INTEGRATION (BiasAuditor + CurriculumPlanner + Thesis)
        # ═══════════════════════════════════════════════════════════════════
        strat = strategy_info or StrategyInfo()
        strategy_reasons: List[str] = []
        
        # Apply BiasAuditor adjustments
        if strat.bias_position_multiplier < 0.99 and position_size > 0:
            old_size = position_size
            position_size *= strat.bias_position_multiplier
            strategy_reasons.append(
                f"BIAS({strat.bias_position_multiplier:.2f}): {','.join(strat.active_biases[:2]) or 'psychological'}"
            )
        
        # Apply CurriculumPlannerPlus constraints
        if position_size > strat.max_position_size:
            old_size = position_size
            position_size = strat.max_position_size
            strategy_reasons.append(
                f"CURRICULUM({strat.curriculum_stage}): max_pos={strat.max_position_size:.2f}"
            )
        
        # Adjust confidence based on mastery level (Foundation students get less confidence)
        if strat.mastery_level < 0.4 and confidence > 0.5:
            confidence *= (0.6 + 0.4 * strat.mastery_level / 0.4)  # Scale down for beginners
            strategy_reasons.append(f"MASTERY({strat.mastery_level:.2f}): confidence reduced")
        
        # ThesisEvolutionEngine alignment boost
        if strat.thesis_regime_alignment and strat.thesis_confidence > 0.7:
            confidence = min(1.0, confidence * 1.05)  # Small boost for thesis alignment
        elif not strat.thesis_regime_alignment and strat.thesis_confidence > 0.5:
            confidence *= 0.95  # Small penalty for thesis misalignment
            strategy_reasons.append("THESIS: regime misaligned")
        
        # ═══════════════════════════════════════════════════════════════════
        # TRADING MODE INTEGRATION (Position scale + Risk multiplier)
        # ═══════════════════════════════════════════════════════════════════
        tm = trading_mode_info or TradingModeInfo()
        tm_reasons: List[str] = []
        
        # Apply trading mode position scaling
        if position_size > 0 and abs(tm.position_scale - 1.0) > 0.01:
            position_size *= tm.position_scale
            tm_reasons.append(f"MODE_SCALE({tm.mode_name}): x{tm.position_scale:.2f}")
        
        # Enforce max_exposure from trading mode
        if position_size > tm.max_exposure:
            position_size = tm.max_exposure
            tm_reasons.append(f"MAX_EXPOSURE({tm.mode_name}): cap={tm.max_exposure:.2f}")
        
        # Safe mode extra reduction
        if tm.should_reduce_position() and position_size > 0:
            position_size *= 0.7  # Additional 30% reduction in safe mode
            tm_reasons.append(f"SAFE_MODE: reduced 30%")
        
        # Aggressive mode confidence boost (only if effective)
        if tm.should_increase_position() and confidence > 0.5:
            confidence = min(1.0, confidence * 1.1)
            tm_reasons.append(f"AGGRESSIVE_BOOST: eff={tm.effectiveness:.2f}")
        
        # ═══════════════════════════════════════════════════════════════════
        # WORLD MODEL INTEGRATION (Predictive adjustments)
        # ═══════════════════════════════════════════════════════════════════
        wm = world_model_info or WorldModelInfo()
        wm_reasons: List[str] = []
        
        if wm.should_trust_predictions():
            directional_bias = wm.get_directional_bias()
            
            # Align with world model predictions
            if direction == "long" and directional_bias < -0.3:
                # World model says bearish but we're long - reduce confidence
                confidence *= 0.85
                wm_reasons.append(f"WM_CONTRA({wm.price_direction}): conf reduced")
            elif direction == "short" and directional_bias > 0.3:
                # World model says bullish but we're short - reduce confidence
                confidence *= 0.85
                wm_reasons.append(f"WM_CONTRA({wm.price_direction}): conf reduced")
            elif (direction == "long" and directional_bias > 0.3) or \
                 (direction == "short" and directional_bias < -0.3):
                # World model agrees - boost confidence slightly
                confidence = min(1.0, confidence * 1.08)
                wm_reasons.append(f"WM_ALIGNED({wm.price_direction}): conf boosted")
            
            # High volatility prediction - reduce position size
            if wm.volatility_prediction > 0.7 and position_size > 0:
                position_size *= 0.85
                wm_reasons.append(f"WM_HIGH_VOL({wm.volatility_prediction:.2f}): size reduced")
            
            # Regime alignment check
            regime_map = {
                "trending_up": "long", "trending_down": "short",
                "ranging": "flat", "volatile": "flat"
            }
            suggested_direction = regime_map.get(wm.regime_prediction, "flat")
            if suggested_direction != "flat" and direction != suggested_direction and position_size > 0:
                if wm.regime_confidence > 0.6:
                    confidence *= 0.9
                    wm_reasons.append(f"WM_REGIME({wm.regime_prediction}): misaligned")
        
        # Final clipping
        position_size = float(np.clip(position_size, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Append all reasons to reasoning
        all_extra_reasons = strategy_reasons + tm_reasons + wm_reasons
        if strategy_reasons:
            reasoning += f" | STRATEGY: {'; '.join(strategy_reasons)}"
        if tm_reasons:
            reasoning += f" | TRADING_MODE: {'; '.join(tm_reasons)}"
        if wm_reasons:
            reasoning += f" | WORLD_MODEL: {'; '.join(wm_reasons)}"
        
        # 7) Build decision
        decision = InstrumentDecision(
            instrument=instrument,
            direction=direction,
            confidence=confidence,
            position_size=position_size,
            trust_score=trust_score,
            committee_action=committee_action,
            committee_confidence=committee_confidence,
            expert_consensus=expert_consensus,
            expert_confidence=expert_confidence,
            regime=regime,
            regime_strength=regime_strength,
            value_estimate=value,
            raw_action=action.tolist(),
            gate_passed=gating_result.gate_passed and position_size > 0.0,
            gate_reasons=gating_result.reasons + all_extra_reasons,
            reasoning=reasoning,
            meta=self._build_decision_meta(
                trust_score=trust_score,
                size_score=size_score,
                committee=committee,
                experts=experts,
                memory_info=memory_info,
                risk_info=risk_info,
                gating_result=gating_result,
                strategy_info=strat,
                trading_mode_info=tm,
                world_model_info=wm,
            ),
        )
        
        # Store in history (lazy-init per instrument for robustness)
        history = self._decision_history.setdefault(instrument, [])
        history.append(decision)
        if len(history) > 100:
            self._decision_history[instrument] = history[-100:]
        
        return decision

    
    # ─────────────────────────────────────────────────────────────
    # Trust Score Interpretation
    # ─────────────────────────────────────────────────────────────
    
    def _interpret_trust_score(
        self,
        trust_score: float,
        committee_action: str,
        committee_confidence: float,
        expert_consensus: str,
        expert_confidence: float,
        instrument: str,
    ) -> Tuple[str, float, str]:
        """
        Interpret trust_score to determine direction and confidence.
        
        Rules:
        - trust_score > 0.3: Trust committee
        - trust_score < -0.3: Override committee (go flat)
        - Otherwise: Uncertain, follow with reduced confidence
        
        Returns:
            (direction, confidence, reasoning)
        """
        if trust_score > 0.3:
            # Trust committee
            if committee_action in ("long", "buy", "bullish"):
                direction = "long"
            elif committee_action in ("short", "sell", "bearish"):
                direction = "short"
            else:
                direction = "flat"
            
            confidence = committee_confidence * (0.7 + 0.3 * min(trust_score, 1.0))
            reasoning = f"PPO trusts committee ({trust_score:.2f}): {committee_action}"
            
            # Boost if aligned with expert consensus
            if direction == expert_consensus and expert_confidence > 0.5:
                confidence = min(1.0, confidence * 1.15)
                reasoning += " | ALIGNED with experts"
        
        elif trust_score < -0.3:
            # Override committee - go flat
            direction = "flat"
            confidence = 0.3
            reasoning = f"PPO overrides committee ({trust_score:.2f}): forcing FLAT"
        
        else:
            # Uncertain - follow committee with reduced confidence
            if committee_action in ("long", "buy", "bullish"):
                direction = "long"
            elif committee_action in ("short", "sell", "bearish"):
                direction = "short"
            else:
                direction = "flat"
            
            confidence = committee_confidence * 0.5
            reasoning = f"PPO uncertain ({trust_score:.2f}): following committee cautiously"
        
        return direction, confidence, reasoning
    
    # ─────────────────────────────────────────────────────────────
    # Expert Signal Processing
    # ─────────────────────────────────────────────────────────────
    
    def _compute_expert_consensus(self, experts: Dict[str, Any]) -> Tuple[str, float]:
        """Compute consensus direction and confidence from expert signals."""
        long_score = 0.0
        short_score = 0.0
        total_weight = 0.0
        
        for expert_name, sig in experts.items():
            if not isinstance(sig, dict):
                continue
            
            raw_prop = sig.get("proposal", sig.get("direction", "flat"))
            conf = float(sig.get("confidence", 0.0))
            
            # Parse direction
            if isinstance(raw_prop, dict):
                direction = raw_prop.get("direction", raw_prop.get("action", "flat"))
            else:
                direction = str(raw_prop)
            
            direction = direction.lower()
            
            if direction in ("long", "buy", "bullish"):
                long_score += conf
            elif direction in ("short", "sell", "bearish"):
                short_score += conf
            
            total_weight += max(conf, 0.0)
        
        if total_weight < 1e-6:
            return "flat", 0.0
        
        if long_score > short_score + 0.2:
            direction = "long"
            consensus_conf = long_score / total_weight
        elif short_score > long_score + 0.2:
            direction = "short"
            consensus_conf = short_score / total_weight
        else:
            direction = "flat"
            consensus_conf = 0.3
        
        return direction, float(np.clip(consensus_conf, 0.0, 1.0))
    
    def _extract_instrument_committee(
        self, 
        committee_data: Dict[str, Any],
        instrument: str,
    ) -> Dict[str, Any]:
        """Extract committee data for a specific instrument."""
        # Check if per-instrument data exists
        if "instruments" in committee_data:
            inst_data = committee_data["instruments"].get(instrument, {})
            if inst_data:
                return inst_data
        
        # Fall back to global committee data
        return {
            "action": committee_data.get("action", "hold"),
            "confidence": committee_data.get("confidence", 0.5),
            "consensus_score": committee_data.get("consensus_score", 0.5),
            "fragility": committee_data.get("fragility", 0.5),
            "regime": committee_data.get("regime", "unknown"),
            "regime_strength": committee_data.get("regime_strength", 0.5),
        }
    
    def _extract_instrument_experts(
        self,
        expert_signals: Dict[str, Any],
        instrument: str,
    ) -> Dict[str, Any]:
        """Extract expert signals for a specific instrument."""
        result: Dict[str, Any] = {}
        
        experts = expert_signals.get("experts", expert_signals)
        
        for expert_name, sig in experts.items():
            if not isinstance(sig, dict):
                continue
            
            # Check for per-instrument data
            if "instruments" in sig:
                inst_data = sig["instruments"].get(instrument, sig)
                result[expert_name] = inst_data
            else:
                result[expert_name] = sig
        
        # Add market context
        result["regime"] = expert_signals.get("market", {}).get("regime", "unknown")
        result["regime_strength"] = expert_signals.get("market", {}).get("regime_strength", 0.5)
        
        return result
    
    # ─────────────────────────────────────────────────────────────
    # Hysteresis
    # ─────────────────────────────────────────────────────────────
    
    def _apply_hysteresis(
        self,
        instrument: str,
        proposed_direction: str,
        trust_score: float,
    ) -> str:
        """
        Apply hysteresis to prevent flip-flopping.
        
        Uses different thresholds for entry, exit, and reversal.
        """
        last_dir = self._last_directions.get(instrument, "flat")
        
        # Thresholds
        entry_threshold = 0.10
        reversal_threshold = 0.15
        exit_threshold = 0.03
        
        if last_dir == "flat":
            # Need stronger signal to enter
            if proposed_direction != "flat" and abs(trust_score) > entry_threshold:
                direction = proposed_direction
            else:
                direction = "flat"
        
        elif last_dir == proposed_direction:
            # Already in this direction, keep it
            direction = proposed_direction
            self._direction_hold_counts[instrument] = self._direction_hold_counts.get(instrument, 0) + 1
        
        elif proposed_direction == "flat":
            # Exit signal - use exit threshold
            if abs(trust_score) < exit_threshold:
                direction = "flat"
            else:
                direction = last_dir  # Stay in position
        
        else:
            # Reversal - need strong signal
            if abs(trust_score) > reversal_threshold:
                direction = proposed_direction
                self._direction_hold_counts[instrument] = 0
            else:
                direction = "flat"  # Go flat first before reversing
        
        # Update state
        if direction != last_dir:
            self._direction_hold_counts[instrument] = 0
        self._last_directions[instrument] = direction
        
        return direction
    
    # ─────────────────────────────────────────────────────────────
    # Metadata Building
    # ─────────────────────────────────────────────────────────────
    
    def _build_decision_meta(
        self,
        trust_score: float,
        size_score: float,
        committee: Dict[str, Any],
        experts: Dict[str, Any],
        memory_info: MemoryGateInfo,
        risk_info: RiskInfo,
        gating_result: GatingResult,
        strategy_info: Optional[StrategyInfo] = None,
        trading_mode_info: Optional[TradingModeInfo] = None,
        world_model_info: Optional[WorldModelInfo] = None,
    ) -> Dict[str, Any]:
        """
        Build rich metadata for debugging and dashboard display.
        
        This is the standardized explanation payload.
        """
        strat = strategy_info or StrategyInfo()
        tm = trading_mode_info or TradingModeInfo()
        wm = world_model_info or WorldModelInfo()
        return {
            "contributors": {
                "committee": {
                    "action": committee.get("action", "hold"),
                    "confidence": committee.get("confidence", 0.5),
                    "consensus_score": committee.get("consensus_score", 0.5),
                },
                "experts": {
                    name: {
                        "proposal": sig.get("proposal", sig.get("direction", "flat")),
                        "confidence": sig.get("confidence", 0.0),
                    }
                    for name, sig in experts.items()
                    if isinstance(sig, dict) and "confidence" in sig
                },
                "ppo": {
                    "trust_score": trust_score,
                    "raw_size_score": size_score,
                    "value": self.ppo_core._last_value,
                },
                "risk": {
                    "hard_block": risk_info.hard_block,
                    "hard_cap": risk_info.hard_cap,
                    "portfolio_risk": risk_info.portfolio_risk,
                    "instrument_risk": risk_info.instrument_risk,
                    "max_dd": risk_info.max_dd,
                    "reasons": risk_info.reasons,
                },
                "memory": {
                    "veto": memory_info.veto,
                    "risk_multiplier": memory_info.risk_multiplier,
                    "danger_similarity": memory_info.danger_similarity,
                    "loss_prob": memory_info.loss_prob,
                    "reasons": memory_info.reasons,
                },
                # Strategy module integration
                "strategy": {
                    "bias_position_multiplier": strat.bias_position_multiplier,
                    "bias_risk_reduction": strat.bias_risk_reduction,
                    "active_biases": strat.active_biases,
                    "psychological_state": strat.psychological_state,
                    "curriculum_stage": strat.curriculum_stage,
                    "stage_difficulty": strat.stage_difficulty,
                    "max_position_size": strat.max_position_size,
                    "max_trades_per_day": strat.max_trades_per_day,
                    "mastery_level": strat.mastery_level,
                    "thesis_confidence": strat.thesis_confidence,
                    "thesis_regime_alignment": strat.thesis_regime_alignment,
                },
                # Trading mode integration
                "trading_mode": {
                    "mode": tm.mode_name,
                    "mode_confidence": tm.mode_confidence,
                    "risk_multiplier": tm.risk_multiplier,
                    "max_exposure": tm.max_exposure,
                    "position_scale": tm.position_scale,
                    "stop_loss_multiplier": tm.stop_loss_multiplier,
                    "effectiveness": tm.effectiveness,
                    "should_reduce": tm.should_reduce_position(),
                    "decision_factors": {
                        "volatility": tm.volatility_factor,
                        "trend": tm.trend_factor,
                        "regime": tm.regime_factor,
                    },
                },
                # World model integration
                "world_model": {
                    "is_trained": wm.is_trained,
                    "prediction_confidence": wm.prediction_confidence,
                    "price_direction": wm.price_direction,
                    "price_change_m15": wm.price_change_m15,  # PRIMARY decision timeframe
                    "price_change_1h": wm.price_change_1h,
                    "price_change_4h": wm.price_change_4h,
                    "price_change_1d": wm.price_change_1d,
                    "volatility_prediction": wm.volatility_prediction,
                    "regime_prediction": wm.regime_prediction,
                    "regime_confidence": wm.regime_confidence,
                    "directional_bias": wm.get_directional_bias(),
                    "should_trust": wm.should_trust_predictions(),
                },
            },
            "gating": gating_result.to_dict(),
        }
    
    # ─────────────────────────────────────────────────────────────
    # Explanation Generation
    # ─────────────────────────────────────────────────────────────
    
    def generate_explanation(self, decision: InstrumentDecision) -> str:
        """
        Generate a concise English explanation of the decision.
        
        Format: "PPO [trusts/overrides/uncertain] committee (X.XX) to [action]; 
                 [modifiers]; position [size]%"
        """
        parts: List[str] = []
        
        # Trust level
        if decision.trust_score > 0.3:
            parts.append(f"PPO trusts committee ({decision.trust_score:.2f})")
        elif decision.trust_score < -0.3:
            parts.append(f"PPO overrides committee ({decision.trust_score:.2f})")
        else:
            parts.append(f"PPO uncertain ({decision.trust_score:.2f})")
        
        # Direction
        parts.append(f"to {decision.direction.upper()} {decision.instrument}")
        
        # Modifiers
        modifiers: List[str] = []
        
        meta = decision.meta.get("contributors", {})
        memory = meta.get("memory", {})
        risk = meta.get("risk", {})
        
        if memory.get("risk_multiplier", 1.0) < 0.9:
            modifiers.append(f"memory caution ({memory['risk_multiplier']:.2f})")
        
        if memory.get("danger_similarity", 0.0) > 0.3:
            modifiers.append(f"danger zone ({memory['danger_similarity']:.2f})")
        
        if risk.get("portfolio_risk", 0.0) > 0.5:
            modifiers.append(f"portfolio risk {risk['portfolio_risk']:.0%}")
        
        if modifiers:
            parts.append("; " + ", ".join(modifiers))
        
        # Position size / gate
        if decision.gate_passed and decision.position_size > 0.0:
            parts.append(f"; position {decision.position_size:.1%}")
        else:
            parts.append("; NO TRADE")
        
        return "".join(parts)
    
    # ─────────────────────────────────────────────────────────────
    # Statistics Access
    # ─────────────────────────────────────────────────────────────
    
    def get_instrument_stats(self) -> Dict[str, Any]:
        """Get per-instrument statistics."""
        return self.stats_tracker.to_dict()
    
    def get_decision_history(self, instrument: str, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent decision history for an instrument."""
        history = self._decision_history.get(instrument, [])
        return [d.to_dict() for d in history[-n:]]
