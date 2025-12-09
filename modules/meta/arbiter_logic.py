#!/usr/bin/env python3
"""
Arbiter Logic - Per-Instrument Decision Making
===============================================

This module contains the domain-specific arbiter logic that:
- Knows what instruments are and how to trade them
- Builds per-instrument observations
- Applies gating logic (risk/memory)
- Produces structured InstrumentDecision objects
- **Adaptive PPO Autonomy**: Dynamically transitions leadership from experts to PPO

It sits between PPOCore (pure RL) and PPOAgentShell (SmartInfoBus gateway).

Version: 3.2.0 (Adaptive Autonomy + Hardened Parsing)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import logging
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
from modules.meta.numeric_utils import _safe_float, _clip

# Alias for typing.Optional so we can use OptType[...] as in your original code
OptType = Optional


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE PPO AUTONOMY SYSTEM
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PPOAutonomyState:
    """
    Tracks PPO's learning progress and determines appropriate autonomy level.
    
    Phases (automatic transitions based on performance):
    1. EXPERT_LED: Experts make decisions, PPO observes and learns
    2. BLENDED: PPO and experts share decision making (weighted average)
    3. PPO_LED: PPO takes lead, experts provide safety checks
    4. FULL_AUTONOMY: PPO has full control with minimal expert intervention
    """
    # Current autonomy level (0.0 = experts lead, 1.0 = PPO leads)
    autonomy_level: float = 0.0
    
    # Phase name for readability
    phase: str = "EXPERT_LED"
    
    # PPO performance metrics (rolling window)
    ppo_win_rate: float = 0.0
    ppo_profit_factor: float = 0.0
    ppo_consistency: float = 0.0  # Low variance = high consistency
    ppo_confidence_accuracy: float = 0.0  # How accurate are PPO's confidence estimates
    
    # Expert agreement tracking
    ppo_expert_agreement_rate: float = 0.0  # How often PPO agrees with experts
    ppo_override_success_rate: float = 0.0  # Success rate when PPO overrides experts
    
    # Learning velocity
    learning_velocity: float = 0.0  # Rate of improvement
    steps_at_current_phase: int = 0
    
    # Phase thresholds (configurable)
    expert_to_blended_threshold: float = 0.45  # Win rate needed to move to blended
    blended_to_ppo_threshold: float = 0.55     # Win rate + consistency for PPO lead
    ppo_to_autonomy_threshold: float = 0.65    # High performance for full autonomy
    
    # Safety bounds
    min_trades_for_evaluation: int = 20
    
    def get_ppo_weight(self) -> float:
        """Get PPO's decision weight based on current autonomy level."""
        return _clip(self.autonomy_level, 0.0, 1.0)
    
    def get_expert_weight(self) -> float:
        """Get experts' decision weight (complement of PPO weight)."""
        return 1.0 - self.get_ppo_weight()


class PPOAutonomyTracker:
    """
    Automatically tracks PPO's performance and adjusts its autonomy level.
    
    The system starts with experts leading (Phase 1) and gradually gives
    more control to PPO as it proves its competence through:
    - Consistent win rate improvement
    - Profitable trade execution
    - Alignment with (or successful override of) expert signals
    - Stable confidence calibration
    """
    
    def __init__(
        self,
        window_size: int = 50,
        adaptation_rate: float = 0.02,
        min_trades: int = 20,
    ):
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.min_trades = min_trades
        
        # State
        self.state = PPOAutonomyState()
        
        # Rolling performance tracking
        self._trade_outcomes: deque = deque(maxlen=window_size)  # (ppo_direction, expert_direction, outcome, ppo_conf, ...)
        self._override_outcomes: deque = deque(maxlen=window_size)  # When PPO disagreed with experts
        
        # Cumulative stats
        self._total_trades = 0
        self._ppo_led_wins = 0
        self._ppo_led_losses = 0
        self._expert_led_wins = 0
        self._expert_led_losses = 0
        
        self._prev_autonomy: float = 0.0
    
    def record_trade_outcome(
        self,
        ppo_direction: str,
        expert_direction: str,
        actual_outcome: float,  # PnL or +1/-1
        ppo_confidence: float,
        was_ppo_led: bool,
    ) -> None:
        """Record the outcome of a trade for autonomy evaluation."""
        self._total_trades += 1
        
        win = actual_outcome > 0
        
        self._trade_outcomes.append({
            "ppo_dir": ppo_direction,
            "expert_dir": expert_direction,
            "outcome": actual_outcome,
            "ppo_conf": ppo_confidence,
            "ppo_led": was_ppo_led,
            "win": win,
            "agreed": ppo_direction == expert_direction,
        })
        
        # Track override outcomes
        if ppo_direction != expert_direction:
            self._override_outcomes.append({
                "ppo_dir": ppo_direction,
                "expert_dir": expert_direction,
                "outcome": actual_outcome,
                "win": win,
            })
        
        # Update cumulative stats
        if was_ppo_led:
            if win:
                self._ppo_led_wins += 1
            else:
                self._ppo_led_losses += 1
        else:
            if win:
                self._expert_led_wins += 1
            else:
                self._expert_led_losses += 1
        
        # Recalculate autonomy
        self._update_autonomy_level()
    
    def _update_autonomy_level(self) -> None:
        """Recalculate autonomy level based on recent performance."""
        if len(self._trade_outcomes) < self.min_trades:
            # Not enough data - stay at current level
            self.state.steps_at_current_phase += 1
            return
        
        outcomes = list(self._trade_outcomes)
        
        # PPO-led trades
        ppo_led_trades = [t for t in outcomes if t["ppo_led"]]
        if ppo_led_trades:
            self.state.ppo_win_rate = sum(1 for t in ppo_led_trades if t["win"]) / len(ppo_led_trades)
        else:
            self.state.ppo_win_rate = 0.0
        
        # Profit factor (PPO-led only)
        wins = [t["outcome"] for t in ppo_led_trades if t["win"]]
        losses = [abs(t["outcome"]) for t in ppo_led_trades if not t["win"]]
        if losses and sum(losses) > 0:
            self.state.ppo_profit_factor = sum(wins) / sum(losses) if wins else 0.0
        else:
            self.state.ppo_profit_factor = sum(wins) if wins else 0.0
        
        # Consistency (inverse of variance) using all outcomes for stability
        if len(outcomes) > 5:
            outcome_values = [t["outcome"] for t in outcomes]
            variance = float(np.var(outcome_values))
            self.state.ppo_consistency = 1.0 / (1.0 + variance)  # 0..1, higher = more consistent
        
        # Agreement rate
        self.state.ppo_expert_agreement_rate = sum(1 for t in outcomes if t["agreed"]) / len(outcomes)
        
        # Override success rate
        overrides = list(self._override_outcomes)
        if overrides:
            self.state.ppo_override_success_rate = sum(1 for o in overrides if o["win"]) / len(overrides)
        
        # Confidence calibration (high-confidence trades only)
        high_conf_trades = [t for t in ppo_led_trades if t["ppo_conf"] > 0.6]
        if high_conf_trades:
            high_conf_accuracy = sum(1 for t in high_conf_trades if t["win"]) / len(high_conf_trades)
            self.state.ppo_confidence_accuracy = high_conf_accuracy
        
        # Determine target autonomy level
        target_autonomy = self._calculate_target_autonomy()
        
        # Smooth transition (low-pass filter)
        delta = target_autonomy - self.state.autonomy_level
        self.state.autonomy_level += delta * self.adaptation_rate
        self.state.autonomy_level = float(_clip(self.state.autonomy_level, 0.0, 1.0))
        
        # Track learning velocity and phase
        self.state.learning_velocity = self.state.autonomy_level - self._prev_autonomy
        self._prev_autonomy = self.state.autonomy_level
        
        self._update_phase_name()
    
    def _calculate_target_autonomy(self) -> float:
        """Calculate target autonomy based on composite performance metrics."""
        # Base score from win rate
        win_rate_score = self.state.ppo_win_rate
        
        # Consistency bonus
        consistency_bonus = self.state.ppo_consistency * 0.2
        
        # Successful override bonus
        override_bonus = 0.0
        if self.state.ppo_override_success_rate > 0.5:
            override_bonus = (self.state.ppo_override_success_rate - 0.5) * 0.3
        
        # Confidence calibration bonus
        confidence_bonus = 0.0
        if self.state.ppo_confidence_accuracy > 0.6:
            confidence_bonus = (self.state.ppo_confidence_accuracy - 0.6) * 0.2
        
        # Profit factor bonus
        pf_bonus = 0.0
        if self.state.ppo_profit_factor > 1.0:
            pf_bonus = min(0.2, (self.state.ppo_profit_factor - 1.0) * 0.1)
        
        composite = win_rate_score + consistency_bonus + override_bonus + confidence_bonus + pf_bonus
        
        # Map composite score to autonomy level via thresholds
        if composite < self.state.expert_to_blended_threshold:
            # Expert-led
            return composite / self.state.expert_to_blended_threshold * 0.25
        elif composite < self.state.blended_to_ppo_threshold:
            # Blended
            progress = (composite - self.state.expert_to_blended_threshold) / (
                self.state.blended_to_ppo_threshold - self.state.expert_to_blended_threshold
            )
            return 0.25 + progress * 0.35  # 0.25 → 0.60
        elif composite < self.state.ppo_to_autonomy_threshold:
            # PPO-led
            progress = (composite - self.state.blended_to_ppo_threshold) / (
                self.state.ppo_to_autonomy_threshold - self.state.blended_to_ppo_threshold
            )
            return 0.60 + progress * 0.25  # 0.60 → 0.85
        else:
            # Full autonomy
            excess = composite - self.state.ppo_to_autonomy_threshold
            return min(1.0, 0.85 + excess * 0.5)
    
    def _update_phase_name(self) -> None:
        """Update human-readable phase name."""
        level = self.state.autonomy_level
        if level < 0.25:
            new_phase = "EXPERT_LED"
        elif level < 0.60:
            new_phase = "BLENDED"
        elif level < 0.85:
            new_phase = "PPO_LED"
        else:
            new_phase = "FULL_AUTONOMY"
        
        if new_phase != self.state.phase:
            self.state.steps_at_current_phase = 0
        else:
            self.state.steps_at_current_phase += 1
        
        self.state.phase = new_phase
    
    def get_decision_weights(self) -> Tuple[float, float]:
        """
        Get (ppo_weight, expert_weight) for blending decisions.
        
        Returns:
            Tuple of (ppo_weight, expert_weight) that sum to 1.0
        """
        return self.state.get_ppo_weight(), self.state.get_expert_weight()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current autonomy state for logging/bus."""
        return {
            "phase": self.state.phase,
            "autonomy_level": round(self.state.autonomy_level, 3),
            "ppo_weight": round(self.state.get_ppo_weight(), 3),
            "expert_weight": round(self.state.get_expert_weight(), 3),
            "ppo_win_rate": round(self.state.ppo_win_rate, 3),
            "ppo_profit_factor": round(self.state.ppo_profit_factor, 3),
            "ppo_consistency": round(self.state.ppo_consistency, 3),
            "ppo_confidence_accuracy": round(self.state.ppo_confidence_accuracy, 3),
            "agreement_rate": round(self.state.ppo_expert_agreement_rate, 3),
            "override_success_rate": round(self.state.ppo_override_success_rate, 3),
            "learning_velocity": round(self.state.learning_velocity, 4),
            "total_trades_evaluated": self._total_trades,
            "steps_at_phase": self.state.steps_at_current_phase,
        }


# ═══════════════════════════════════════════════════════════════════
# STRATEGY INTEGRATION TYPES
# ═══════════════════════════════════════════════════════════════════

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
    
    def __post_init__(self) -> None:
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
        """Create StrategyInfo from SmartInfoBus data (defensive against weird shapes)."""
        # Bias adjustments
        bias_adj = bias_adjustments or {}
        pos_mult = _safe_float(bias_adj.get("position_size_multiplier"), 1.0)
        risk_red = _safe_float(bias_adj.get("risk_reduction_factor"), 0.0)
        
        # Bias analysis
        bias_anal = bias_analysis or {}
        active: List[str] = []
        ind_biases = bias_anal.get("individual_biases", {})
        if isinstance(ind_biases, dict):
            for k, v in ind_biases.items():
                if isinstance(v, dict) and v.get("detected", False):
                    active.append(str(k))
        
        # Psychological state
        psych = psychological_state or {}
        psych_state = str(psych.get("state", "neutral") or "neutral")
        
        # Curriculum stage
        curr_stage = curriculum_stage or {}
        stage_name = str(curr_stage.get("name", "Foundation") or "Foundation")
        stage_diff = _safe_float(curr_stage.get("difficulty"), 1.0)
        
        # Learning constraints
        constraints = learning_constraints or {}
        max_pos = _safe_float(constraints.get("max_position_size"), 1.0)
        max_trades_raw = constraints.get("max_trades_per_day", 20)
        try:
            max_trades = int(max_trades_raw)
        except (TypeError, ValueError):
            max_trades = 20
        
        # Mastery assessment
        mastery = mastery_assessment or {}
        mastery_lvl = _safe_float(mastery.get("mastery_level"), 0.5)
        
        # Best thesis
        thesis = best_thesis or {}
        thesis_text = str(thesis.get("thesis", "") or "")
        thesis_conf = _safe_float(thesis.get("confidence"), 0.5)
        thesis_aligned = bool(thesis.get("regime_aligned", True))
        
        return cls(
            bias_position_multiplier=_clip(pos_mult, 0.1, 1.0),
            bias_risk_reduction=_clip(risk_red, 0.0, 0.5),
            active_biases=active,
            psychological_state=psych_state,
            curriculum_stage=stage_name,
            stage_difficulty=stage_diff,
            max_position_size=_clip(max_pos, 0.1, 2.0),
            max_trades_per_day=max(1, min(max_trades, 50)),
            mastery_level=_clip(mastery_lvl, 0.0, 1.0),
            best_thesis=thesis_text[:200] if thesis_text else "",
            thesis_confidence=_clip(thesis_conf, 0.0, 1.0),
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
        mode_effectiveness: OptType[Any] = None,
        decision_factors: OptType[Dict[str, Any]] = None,
    ) -> "TradingModeInfo":
        """Create TradingModeInfo from SmartInfoBus data (robust to partial info)."""
        # Mode name
        mode = str(trading_mode or "normal").lower()
        if mode not in ("safe", "normal", "aggressive", "extreme"):
            mode = "normal"
        
        # Mode config
        config = mode_config or {}
        risk_mult = _safe_float(config.get("risk_multiplier"), 1.0)
        max_exp = _safe_float(config.get("max_exposure"), 0.5)
        pos_scale = _safe_float(config.get("position_scale"), 1.0)
        sl_mult = _safe_float(config.get("stop_loss_multiplier"), 1.0)
        
        # Effectiveness (supports either scalar or dict with score/effectiveness)
        eff_val: Any
        if isinstance(mode_effectiveness, dict):
            eff_val = mode_effectiveness.get("effectiveness", mode_effectiveness.get("score", 0.5))
        else:
            eff_val = mode_effectiveness if mode_effectiveness is not None else 0.5
        eff = _safe_float(eff_val, 0.5)
        
        # Decision factors
        factors = decision_factors or {}
        vol_factor = _safe_float(factors.get("volatility"), 0.5)
        trend_factor = _safe_float(factors.get("trend"), 0.5)
        regime_factor = _safe_float(factors.get("regime"), 0.5)
        
        # Mode confidence based on effectiveness
        mode_conf = eff if eff > 0 else 0.5
        
        return cls(
            mode_name=mode,
            mode_confidence=_clip(mode_conf, 0.0, 1.0),
            risk_multiplier=_clip(risk_mult, 0.25, 4.0),
            max_exposure=_clip(max_exp, 0.1, 1.0),
            position_scale=_clip(pos_scale, 0.25, 2.0),
            stop_loss_multiplier=_clip(sl_mult, 0.5, 2.0),
            effectiveness=_clip(eff, 0.0, 1.0),
            win_rate_in_mode=0.5,  # could be wired to analytics later
            volatility_factor=_clip(vol_factor, 0.0, 1.0),
            trend_factor=_clip(trend_factor, 0.0, 1.0),
            regime_factor=_clip(regime_factor, 0.0, 1.0),
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
        prediction_confidence: OptType[Any] = None,
        scenario_generation: OptType[Dict[str, Any]] = None,
        world_model_analytics: OptType[Dict[str, Any]] = None,  # reserved for future use
    ) -> "WorldModelInfo":
        """
        Create WorldModelInfo from SmartInfoBus data.

        The world model provides (typical schema):
        - market_predictions.latest_predictions.price_changes: [4 values]
        - market_predictions.latest_predictions.volatility_predictions: [4 values]
        - market_predictions.latest_predictions.regime_probabilities: [4 values]
        - market_predictions.latest_predictions.confidence: float
        - market_predictions.is_trained: bool
        - market_predictions.model_confidence: float
        """
        predictions = market_predictions or {}
        
        # Model status
        is_trained = bool(predictions.get("is_trained", False))
        model_confidence = _safe_float(predictions.get("model_confidence"), 0.0)
        
        latest = predictions.get("latest_predictions") or {}
        
        # Price changes: [M15, H1, H4, D1]
        price_changes = latest.get("price_changes") or []
        if isinstance(price_changes, (list, tuple)) and len(price_changes) >= 4:
            price_m15 = _safe_float(price_changes[0], 0.0)
            price_1h = _safe_float(price_changes[1], 0.0)
            price_4h = _safe_float(price_changes[2], 0.0)
            price_1d = _safe_float(price_changes[3], 0.0)
        elif isinstance(price_changes, (list, tuple)) and len(price_changes) >= 1:
            price_m15 = _safe_float(price_changes[0], 0.0)
            price_1h = _safe_float(price_changes[1], 0.0) if len(price_changes) > 1 else 0.0
            price_4h = _safe_float(price_changes[2], 0.0) if len(price_changes) > 2 else 0.0
            price_1d = _safe_float(price_changes[3], 0.0) if len(price_changes) > 3 else 0.0
        else:
            price_m15 = price_1h = price_4h = price_1d = 0.0
        
        # Weighted directional signal
        weighted_change = (
            price_m15 * 0.5
            + price_1h * 0.25
            + price_4h * 0.15
            + price_1d * 0.10
        )
        if weighted_change > 0.0005:
            direction = "bullish"
        elif weighted_change < -0.0005:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Volatility predictions
        vol_preds = latest.get("volatility_predictions") or []
        if isinstance(vol_preds, (list, tuple)) and len(vol_preds) >= 1:
            v0 = _safe_float(vol_preds[0], 0.5)
            if len(vol_preds) > 1:
                v1 = _safe_float(vol_preds[1], v0)
                vol_value = v0
                vol_change = v1 - v0
            else:
                vol_value = v0
                vol_change = 0.0
        else:
            vol_value, vol_change = 0.5, 0.0
        
        # Regime predictions
        regime_probs = latest.get("regime_probabilities") or []
        raw_idx = latest.get("predicted_regime", -1)
        try:
            predicted_regime_idx = int(raw_idx)
        except (TypeError, ValueError):
            predicted_regime_idx = -1
        
        regime_names = ["trending_up", "trending_down", "ranging", "volatile"]
        if 0 <= predicted_regime_idx < len(regime_names):
            regime = regime_names[predicted_regime_idx]
            if isinstance(regime_probs, (list, tuple)) and len(regime_probs) > predicted_regime_idx:
                regime_conf = _safe_float(regime_probs[predicted_regime_idx], 0.5)
            else:
                regime_conf = 0.5
        else:
            regime = "ranging"
            regime_conf = 0.5
        
        # Confidence from model/last prediction
        pred_conf = _safe_float(latest.get("confidence"), model_confidence)
        
        # Override with explicit prediction_confidence if provided
        if prediction_confidence is not None:
            if isinstance(prediction_confidence, dict):
                pred_conf = _safe_float(
                    prediction_confidence.get("current_confidence", pred_conf),
                    pred_conf,
                )
            else:
                pred_conf = _safe_float(prediction_confidence, pred_conf)
        
        # Scenarios
        scenarios = scenario_generation or {}
        scenarios_list = scenarios.get("scenarios") or []
        best_prob = 0.0
        worst_prob = 0.0
        expected = 0.0
        
        if isinstance(scenarios_list, list) and scenarios_list:
            for s in scenarios_list:
                if not isinstance(s, dict):
                    continue
                prob = _safe_float(s.get("probability"), 0.0)
                outcome = _safe_float(s.get("outcome"), 0.0)
                expected += prob * outcome
                if outcome > 0:
                    best_prob = max(best_prob, prob)
                elif outcome < 0:
                    worst_prob = max(worst_prob, prob)
        
        return cls(
            is_trained=is_trained,
            prediction_confidence=_clip(pred_conf, 0.0, 1.0),
            price_change_m15=_clip(price_m15, -0.05, 0.05),
            price_change_1h=_clip(price_1h, -0.1, 0.1),
            price_change_4h=_clip(price_4h, -0.2, 0.2),
            price_change_1d=_clip(price_1d, -0.3, 0.3),
            price_direction=direction,
            volatility_prediction=_clip(vol_value, 0.0, 1.0),
            volatility_change=_clip(vol_change, -0.5, 0.5),
            regime_prediction=regime,
            regime_confidence=_clip(regime_conf, 0.0, 1.0),
            best_scenario_probability=_clip(best_prob, 0.0, 1.0),
            worst_scenario_probability=_clip(worst_prob, 0.0, 1.0),
            expected_move=_clip(expected, -0.5, 0.5),
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
        if self.price_direction == "bearish":
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
    
    New in v3.2.0:
    - Hardened bus parsers (no float explosions)
    - PPOAutonomyTracker state included in decision meta
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
        
        self.logger = logging.getLogger("ArbiterLogic")
        
        # Per-instrument statistics
        self.stats_tracker = InstrumentStatsTracker()
        
        # Hysteresis state per instrument
        self._last_directions: Dict[str, str] = {inst: "flat" for inst in self.instruments}
        self._direction_hold_counts: Dict[str, int] = {inst: 0 for inst in self.instruments}
        
        # Decision history
        self._decision_history: Dict[str, List[InstrumentDecision]] = {
            inst: [] for inst in self.instruments
        }
        
        # Adaptive autonomy
        self.autonomy_tracker = PPOAutonomyTracker(
            window_size=50,
            adaptation_rate=0.02,
            min_trades=20,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Main Decision Method
    # ─────────────────────────────────────────────────────────────
    
    def make_multi_instrument_decision(
        self,
        observations: Dict[str, np.ndarray],
        committee_data: Optional[Dict[str, Any]],
        expert_signals: Optional[Dict[str, Any]],
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
        committee_data = committee_data or {}
        expert_signals = expert_signals or {}
        
        decisions: Dict[str, InstrumentDecision] = {}
        
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
            
            # Instrument-specific data
            inst_committee = self._extract_instrument_committee(committee_data, instrument)
            inst_experts = self._extract_instrument_experts(expert_signals, instrument)
            
            # Single-instrument decision
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
            self.stats_tracker.record_decision(decision)
        
        # Current autonomy state
        autonomy_state = self.autonomy_tracker.get_state_summary()
        
        # Global metadata
        global_meta = {
            "timestamp": datetime.now().isoformat(),
            "instruments_processed": len(instruments),
            "memory_gate_value": memory_info.risk_multiplier,
            "risk_portfolio": risk_info.portfolio_risk,
            "stats": self.stats_tracker.to_dict(),
            "strategy": {
                "curriculum_stage": strat.curriculum_stage,
                "stage_difficulty": strat.stage_difficulty,
                "mastery_level": strat.mastery_level,
                "bias_position_multiplier": strat.bias_position_multiplier,
                "active_biases": strat.active_biases,
                "psychological_state": strat.psychological_state,
                "thesis_confidence": strat.thesis_confidence,
            },
            "trading_mode": {
                "mode": tm_info.mode_name,
                "risk_multiplier": tm_info.risk_multiplier,
                "max_exposure": tm_info.max_exposure,
                "position_scale": tm_info.position_scale,
                "effectiveness": tm_info.effectiveness,
                "should_reduce": tm_info.should_reduce_position(),
            },
            "world_model": {
                "is_trained": wm_info.is_trained,
                "prediction_confidence": wm_info.prediction_confidence,
                "price_direction": wm_info.price_direction,
                "regime_prediction": wm_info.regime_prediction,
                "volatility": wm_info.volatility_prediction,
                "directional_bias": wm_info.get_directional_bias(),
            },
            "ppo_autonomy": autonomy_state,
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
        5. Apply strategy/mode/world-model adjustments
        6. Compute final position size and build InstrumentDecision
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
        
        # Optional debug logging of direction logic
        if self.debug:
            self.logger.info(
                f"[PPO_DIRECTION] {instrument}: committee_action={committee_action}, "
                f"expert_consensus={expert_consensus}, expert_conf={expert_confidence:.2f}, "
                f"trust_score={trust_score:.2f}"
            )
        
        # 3) Interpret trust_score into raw direction & confidence
        direction, confidence, reasoning = self._interpret_trust_score(
            trust_score=trust_score,
            committee_action=committee_action,
            committee_confidence=committee_confidence,
            expert_consensus=expert_consensus,
            expert_confidence=expert_confidence,
            instrument=instrument,
        )
        
        # 4) Hysteresis on direction
        # In EXPERT_LED mode, use committee_confidence for hysteresis threshold
        # so that PPO's random trust_score doesn't override expert decisions
        autonomy_phase = self.autonomy_tracker.state.phase
        if autonomy_phase == "EXPERT_LED":
            # Use committee confidence scaled to [-1, 1] range for hysteresis
            hysteresis_score = (committee_confidence - 0.5) * 2.0  # 0.34 -> -0.32, 0.7 -> 0.4
            # If committee has a direction, use stronger signal
            if committee_action in ("long", "short", "buy", "sell"):
                hysteresis_score = max(0.2, committee_confidence)  # Ensure we pass entry threshold
        else:
            hysteresis_score = trust_score
        direction = self._apply_hysteresis(instrument, direction, hysteresis_score)
        
        # 5) Gating pipeline
        gating_result = GatingResult.apply_gates(memory_info, risk_info, trust_score)
        
        # Apply gate confidence multiplier
        confidence *= gating_result.confidence_multiplier
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        if not gating_result.gate_passed:
            if gating_result.reasons:
                reasoning += f" | GATED: {', '.join(gating_result.reasons)}"
        elif gating_result.soft_scaling_applied and gating_result.reasons:
            reasoning += f" | SCALED: {', '.join(gating_result.reasons)}"
        
        # Base position size from size_score
        raw_size = (size_score + 1.0) / 2.0  # [-1,1] -> [0,1]
        position_size = float(
            np.clip(raw_size * confidence * gating_result.position_size_cap, 0.0, 1.0)
        )
        
        # DEBUG: Position size calculation trace
        self.logger.debug(
            f"[SIZE_DEBUG] {instrument}: size_score={size_score:.4f}, raw_size={raw_size:.4f}, "
            f"confidence={confidence:.4f}, cap={gating_result.position_size_cap:.4f}, "
            f"position_size={position_size:.4f}, gating_passed={gating_result.gate_passed}, "
            f"direction={direction}"
        )
        
        if not gating_result.gate_passed or direction == "flat":
            position_size = 0.0
        
        # Strategy / mode / world-model integrations
        strat = strategy_info or StrategyInfo()
        tm = trading_mode_info or TradingModeInfo()
        wm = world_model_info or WorldModelInfo()
        
        strategy_reasons: List[str] = []
        tm_reasons: List[str] = []
        wm_reasons: List[str] = []
        
        # ───── Strategy (BiasAuditor + Curriculum + Thesis) ─────
        if strat.bias_position_multiplier < 0.99 and position_size > 0:
            position_size *= strat.bias_position_multiplier
            bias_strs = [str(b) for b in strat.active_biases[:2]]
            strategy_reasons.append(
                f"BIAS({strat.bias_position_multiplier:.2f}): {','.join(bias_strs) or 'psychological'}"
            )
        
        if position_size > strat.max_position_size:
            position_size = strat.max_position_size
            strategy_reasons.append(
                f"CURRICULUM({strat.curriculum_stage}): max_pos={strat.max_position_size:.2f}"
            )
        
        if strat.mastery_level < 0.4 and confidence > 0.5:
            confidence *= (0.6 + 0.4 * strat.mastery_level / 0.4)  # soften confidence for beginners
            strategy_reasons.append(f"MASTERY({strat.mastery_level:.2f}): confidence reduced")
        
        if strat.thesis_regime_alignment and strat.thesis_confidence > 0.7:
            confidence = min(1.0, confidence * 1.05)
        elif not strat.thesis_regime_alignment and strat.thesis_confidence > 0.5:
            confidence *= 0.95
            strategy_reasons.append("THESIS: regime misaligned")
        
        # ───── Trading Mode (risk/scale) ─────
        if position_size > 0 and abs(tm.position_scale - 1.0) > 0.01:
            position_size *= tm.position_scale
            tm_reasons.append(f"MODE_SCALE({tm.mode_name}): x{tm.position_scale:.2f}")
        
        if position_size > tm.max_exposure:
            position_size = tm.max_exposure
            tm_reasons.append(f"MAX_EXPOSURE({tm.mode_name}): cap={tm.max_exposure:.2f}")
        
        if tm.should_reduce_position() and position_size > 0:
            position_size *= 0.7
            tm_reasons.append("SAFE_MODE: reduced 30%")
        
        if tm.should_increase_position() and confidence > 0.5:
            confidence = min(1.0, confidence * 1.1)
            tm_reasons.append(f"AGGRESSIVE_BOOST: eff={tm.effectiveness:.2f}")
        
        # ───── World Model (predictive adjustments) ─────
        if wm.should_trust_predictions():
            directional_bias = wm.get_directional_bias()
            
            if direction == "long" and directional_bias < -0.3:
                confidence *= 0.85
                wm_reasons.append(f"WM_CONTRA({wm.price_direction}): conf reduced")
            elif direction == "short" and directional_bias > 0.3:
                confidence *= 0.85
                wm_reasons.append(f"WM_CONTRA({wm.price_direction}): conf reduced")
            elif (direction == "long" and directional_bias > 0.3) or \
                 (direction == "short" and directional_bias < -0.3):
                confidence = min(1.0, confidence * 1.08)
                wm_reasons.append(f"WM_ALIGNED({wm.price_direction}): conf boosted")
            
            if wm.volatility_prediction > 0.7 and position_size > 0:
                position_size *= 0.85
                wm_reasons.append(f"WM_HIGH_VOL({wm.volatility_prediction:.2f}): size reduced")
            
            regime_map = {
                "trending_up": "long",
                "trending_down": "short",
                "ranging": "flat",
                "volatile": "flat",
            }
            suggested_direction = regime_map.get(wm.regime_prediction, "flat")
            if suggested_direction != "flat" and direction != suggested_direction and position_size > 0:
                if wm.regime_confidence > 0.6:
                    confidence *= 0.9
                    wm_reasons.append(f"WM_REGIME({wm.regime_prediction}): misaligned")
        
        # Final clipping
        position_size = float(np.clip(position_size, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Append reasons
        if strategy_reasons:
            reasoning += f" | STRATEGY: {'; '.join(strategy_reasons)}"
        if tm_reasons:
            reasoning += f" | TRADING_MODE: {'; '.join(tm_reasons)}"
        if wm_reasons:
            reasoning += f" | WORLD_MODEL: {'; '.join(wm_reasons)}"
        
        # Autonomy meta for this decision
        autonomy_meta = self.autonomy_tracker.get_state_summary()
        
        # DEBUG: Gate decision tracing
        gate_will_pass = gating_result.gate_passed and position_size > 0.0
        if not gate_will_pass:
            self.logger.warning(
                f"[ARBITER_DEBUG] {instrument}: gate_passed={gating_result.gate_passed}, "
                f"position_size={position_size:.4f}, direction={direction}, "
                f"reasons={gating_result.reasons + strategy_reasons + tm_reasons + wm_reasons}"
            )
        
        # Decision object
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
            gate_reasons=gating_result.reasons + strategy_reasons + tm_reasons + wm_reasons,
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
                autonomy_state=autonomy_meta,
            ),
        )
        
        # History
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
        
        Uses ADAPTIVE AUTONOMY system:
        - PPO autonomy mode adjusts based on performance (win rate, agreement, consistency)
        - Phases: EXPERT_LED -> BLENDED -> PPO_LED -> FULL_AUTONOMY
        - Weights shift dynamically from experts to PPO as PPO proves itself
        
        Returns:
            (direction, confidence, reasoning)
        """
        ppo_weight, expert_weight = self.autonomy_tracker.get_decision_weights()
        autonomy_summary = self.autonomy_tracker.get_state_summary()
        phase = autonomy_summary["phase"]
        autonomy_level = autonomy_summary["autonomy_level"]
        ppo_win_rate = autonomy_summary["ppo_win_rate"]
        
        # Committee direction
        if committee_action in ("long", "buy", "bullish"):
            committee_dir = "long"
        elif committee_action in ("short", "sell", "bearish"):
            committee_dir = "short"
        else:
            committee_dir = "flat"
        
        # Expert consensus direction
        if expert_consensus in ("long", "buy", "bullish"):
            expert_dir = "long"
        elif expert_consensus in ("short", "sell", "bearish"):
            expert_dir = "short"
        else:
            expert_dir = "flat"
        
        # PPO's own inclination from trust_score
        if trust_score > 0.3:
            ppo_dir = committee_dir  # PPO supports committee
            ppo_conf = abs(trust_score)
        elif trust_score < -0.3:
            ppo_dir = "flat"  # PPO wants to override by standing aside
            ppo_conf = abs(trust_score)
        else:
            ppo_dir = committee_dir  # Uncertain: lean to committee
            ppo_conf = 0.3
        
        # Phase-based blending
        if phase == "EXPERT_LED":
            direction = committee_dir if committee_confidence > 0.3 else expert_dir
            confidence = (
                max(committee_confidence, expert_confidence) * expert_weight
                + ppo_conf * ppo_weight
            )
            reasoning = f"[{phase}] Committee leads ({expert_weight:.0%}): {direction} (comm={committee_dir}, exp={expert_dir})"
            
        elif phase == "BLENDED":
            if expert_dir == ppo_dir:
                direction = expert_dir
                confidence = (expert_confidence * expert_weight + ppo_conf * ppo_weight) * 1.1
                reasoning = f"[{phase}] Agreement: {direction} (PPO+experts aligned)"
            else:
                direction = expert_dir
                confidence = expert_confidence * 0.7
                reasoning = f"[{phase}] Conflict - experts say {expert_dir}, PPO says {ppo_dir}"
                
        elif phase == "PPO_LED":
            if ppo_dir == expert_dir:
                direction = ppo_dir
                confidence = (ppo_conf * ppo_weight + expert_confidence * expert_weight) * 1.15
                reasoning = f"[{phase}] PPO-led agreement: {direction}"
            elif trust_score > 0.5:
                direction = ppo_dir
                confidence = ppo_conf * 0.85
                reasoning = f"[{phase}] PPO confident ({trust_score:.2f}): {direction}"
            else:
                direction = expert_dir
                confidence = expert_confidence * 0.6
                reasoning = f"[{phase}] PPO uncertain, defer to experts: {direction}"
                
        else:  # FULL_AUTONOMY
            if trust_score > 0.2:
                direction = ppo_dir
                confidence = ppo_conf * 0.95
                reasoning = f"[{phase}] PPO autonomous: {direction} (trust={trust_score:.2f})"
            elif trust_score < -0.2:
                direction = "flat"
                confidence = 0.4
                reasoning = f"[{phase}] PPO override: FLAT (trust={trust_score:.2f})"
            else:
                direction = expert_dir if expert_confidence > 0.4 else "flat"
                confidence = expert_confidence * 0.5
                reasoning = f"[{phase}] PPO uncertain, checking experts: {direction}"
        
        reasoning += f" | Score={autonomy_level:.2f}, WR={ppo_win_rate:.0%}"
        
        return direction, float(np.clip(confidence, 0.0, 1.0)), reasoning
    
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
            conf = _safe_float(sig.get("confidence"), 0.0)
            
            if isinstance(raw_prop, dict):
                direction = raw_prop.get("direction", raw_prop.get("action", "flat"))
            else:
                direction = str(raw_prop)
            
            d = direction.lower()
            if d in ("long", "buy", "bullish"):
                long_score += conf
            elif d in ("short", "sell", "bearish"):
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
        """Extract committee data for a specific instrument, with safe fallbacks."""
        inst_map = committee_data.get("instruments")
        if isinstance(inst_map, dict):
            inst_data = inst_map.get(instrument, {}) or {}
            if inst_data:
                if self.debug:
                    self.logger.debug(
                        f"[DIRECTION] {instrument}: per-instrument committee data: "
                        f"action={inst_data.get('action', 'N/A')}, conf={inst_data.get('confidence', 'N/A')}"
                    )
                return {
                    "action": inst_data.get("action", "hold"),
                    "confidence": inst_data.get("confidence", 0.5),
                    "consensus_score": inst_data.get("consensus_score", 0.5),
                    "fragility": committee_data.get("fragility", 0.5),
                    "regime": committee_data.get("regime", "unknown"),
                    "regime_strength": committee_data.get("regime_strength", 0.5),
                }
        
        # Fallback: global committee
        if self.debug:
            self.logger.warning(
                f"[DIRECTION] {instrument}: no per-instrument committee data; "
                f"using global action={committee_data.get('action', 'hold')}"
            )
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
        if not isinstance(experts, dict):
            return {
                "regime": expert_signals.get("market", {}).get("regime", "unknown"),
                "regime_strength": expert_signals.get("market", {}).get("regime_strength", 0.5),
            }
        
        for expert_name, sig in experts.items():
            if not isinstance(sig, dict):
                continue
            
            if "instruments" in sig and isinstance(sig["instruments"], dict):
                inst_data = sig["instruments"].get(instrument, sig)
                result[expert_name] = inst_data
            else:
                result[expert_name] = sig
        
        # Add market context
        market_info = expert_signals.get("market", {}) or {}
        result["regime"] = market_info.get("regime", "unknown")
        result["regime_strength"] = market_info.get("regime_strength", 0.5)
        
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
        
        entry_threshold = 0.10
        reversal_threshold = 0.15
        exit_threshold = 0.03
        
        if last_dir == "flat":
            if proposed_direction != "flat" and abs(trust_score) > entry_threshold:
                direction = proposed_direction
            else:
                direction = "flat"
        
        elif last_dir == proposed_direction:
            direction = proposed_direction
            self._direction_hold_counts[instrument] = self._direction_hold_counts.get(instrument, 0) + 1
        
        elif proposed_direction == "flat":
            if abs(trust_score) < exit_threshold:
                direction = "flat"
            else:
                direction = last_dir
        
        else:
            if abs(trust_score) > reversal_threshold:
                direction = proposed_direction
                self._direction_hold_counts[instrument] = 0
            else:
                direction = "flat"
        
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
        autonomy_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build rich metadata for debugging and dashboard display.
        
        This is the standardized explanation payload.
        """
        strat = strategy_info or StrategyInfo()
        tm = trading_mode_info or TradingModeInfo()
        wm = world_model_info or WorldModelInfo()
        autonomy_state = autonomy_state or self.autonomy_tracker.get_state_summary()
        
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
                "world_model": {
                    "is_trained": wm.is_trained,
                    "prediction_confidence": wm.prediction_confidence,
                    "price_direction": wm.price_direction,
                    "price_change_m15": wm.price_change_m15,
                    "price_change_1h": wm.price_change_1h,
                    "price_change_4h": wm.price_change_4h,
                    "price_change_1d": wm.price_change_1d,
                    "volatility_prediction": wm.volatility_prediction,
                    "regime_prediction": wm.regime_prediction,
                    "regime_confidence": wm.regime_confidence,
                    "directional_bias": wm.get_directional_bias(),
                    "should_trust": wm.should_trust_predictions(),
                },
                "autonomy": autonomy_state,
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
        
        if decision.trust_score > 0.3:
            parts.append(f"PPO trusts committee ({decision.trust_score:.2f})")
        elif decision.trust_score < -0.3:
            parts.append(f"PPO overrides committee ({decision.trust_score:.2f})")
        else:
            parts.append(f"PPO uncertain ({decision.trust_score:.2f})")
        
        parts.append(f"to {decision.direction.upper()} {decision.instrument}")
        
        modifiers: List[str] = []
        meta = decision.meta.get("contributors", {})
        memory = meta.get("memory", {})
        risk = meta.get("risk", {})
        
        rm = memory.get("risk_multiplier", 1.0)
        if rm < 0.9:
            modifiers.append(f"memory caution ({rm:.2f})")
        
        ds = memory.get("danger_similarity", 0.0)
        if ds > 0.3:
            modifiers.append(f"danger zone ({ds:.2f})")
        
        pr = risk.get("portfolio_risk", 0.0)
        if pr > 0.5:
            modifiers.append(f"portfolio risk {pr:.0%}")
        
        if modifiers:
            parts.append("; " + ", ".join(modifiers))
        
        if decision.gate_passed and decision.position_size > 0.0:
            parts.append(f"; position {decision.position_size:.1%}")
        else:
            parts.append("; NO TRADE")
        
        return "".join(parts)
    
    # ─────────────────────────────────────────────────────────────
    # Statistics / History / Autonomy Access
    # ─────────────────────────────────────────────────────────────
    
    def get_instrument_stats(self) -> Dict[str, Any]:
        """Get per-instrument statistics."""
        return self.stats_tracker.to_dict()
    
    def get_decision_history(self, instrument: str, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent decision history for an instrument."""
        history = self._decision_history.get(instrument, [])
        return [d.to_dict() for d in history[-n:]]
    
    def record_trade_outcome(
        self,
        instrument: str,
        ppo_direction: str,
        expert_direction: str,
        pnl: float,
        ppo_confidence: float,
        was_ppo_led: bool,
    ) -> None:
        """
        Record a trade outcome for autonomy tracking.
        
        Call this when a trade closes to update the adaptive autonomy system.
        """
        self.autonomy_tracker.record_trade_outcome(
            ppo_direction=ppo_direction,
            expert_direction=expert_direction,
            actual_outcome=pnl,
            ppo_confidence=ppo_confidence,
            was_ppo_led=was_ppo_led,
        )
        self.stats_tracker.record_trade_result(instrument, pnl)
    
    def get_autonomy_state(self) -> Dict[str, Any]:
        """Get the current PPO autonomy state for dashboard/logging."""
        return self.autonomy_tracker.get_state_summary()
    
    def get_ppo_decision_weights(self) -> Tuple[float, float]:
        """Get current PPO/expert decision weights."""
        return self.autonomy_tracker.get_decision_weights()
