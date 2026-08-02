# modules/memory/components/interventions.py
"""
Interventions Component
Anti-relapse table learned from counterfactuals over nearest neighbors.

Purpose: Learn and recommend interventions to prevent repeat losses.

State: {(pattern_label, regime) -> {intervention, strength, best_alt, exp_delta, n}}

Output:
- intervention: "avoid" | "halve_size" | "tighter_sl" | "wider_tp" | "none"
- strength: 0..1 (confidence in intervention)
- best_alt: Best alternative action description
- exp_delta: Expected PnL improvement from intervention

Integration: UnifiedMemory maps interventions to size_mult/sl_mult/tp_mult;
sets veto when intervention=='avoid' && strength > threshold.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.memory.shared.utils import safe_float

from .base import MemoryComponent


@dataclass
class InterventionRecord:
    """Record of intervention effectiveness for a (pattern, regime) key."""
    intervention: str = "none"  # avoid, halve_size, tighter_sl, wider_tp, none
    strength: float = 0.0  # Confidence in intervention [0, 1]
    best_alt: str = ""  # Best alternative description
    exp_delta: float = 0.0  # Expected PnL delta from intervention
    n: int = 0  # Sample count
    total_pnl_without: float = 0.0  # Total PnL without intervention
    total_pnl_with: float = 0.0  # Total PnL with intervention (estimated)
    last_updated: float = field(default_factory=time.time)


class InterventionsComponent(MemoryComponent):
    """
    Interventions Component - Anti-relapse table from counterfactual analysis.
    
    Learns which interventions work best for specific (pattern, regime) combinations
    by analyzing historical trades and computing counterfactual improvements.
    
    Intervention Types:
    - avoid: Don't take this trade setup
    - halve_size: Reduce position size by 50%
    - tighter_sl: Tighten stop loss
    - wider_tp: Widen take profit target
    - none: No intervention needed
    
    Counterfactual Logic:
    - For losing trades, estimate what would have happened with alternative actions
    - Track effectiveness per (pattern_label, regime) key
    - Recommend interventions with highest expected improvement
    """
    
    # Constants
    _BUFFER_FRACTION: float = 0.10
    _MIN_SAMPLES_FOR_RECOMMENDATION: int = 5
    _INTERVENTION_TYPES: Tuple[str, ...] = ("avoid", "halve_size", "tighter_sl", "wider_tp", "none")
    _STRENGTH_DECAY: float = 0.95  # Decay factor for old interventions
    _MAX_PATTERNS: int = 100  # Max patterns to track
    
    # Counterfactual estimation parameters
    _SIZE_REDUCTION_FACTOR: float = 0.5  # halve_size reduces exposure by 50%
    _SL_TIGHTENING_FACTOR: float = 0.7  # tighter_sl reduces max loss by 30%
    _TP_WIDENING_FACTOR: float = 1.2  # wider_tp increases potential gain by 20%
    
    def _initialize_component(self) -> None:
        """Initialize interventions component."""
        cfg = self.config
        
        # Configuration
        self.max_memory_size: int = int(getattr(cfg, "max_memory_size", 10_000))
        self.intervention_threshold: float = float(getattr(cfg, "intervention_threshold", 0.6))
        self.veto_threshold: float = float(getattr(cfg, "veto_threshold", 0.8))
        
        # Intervention table: (instrument, pattern_label, regime) -> InterventionRecord
        # Changed from (pattern_label, regime) to include instrument for per-instrument tracking
        self.intervention_table: Dict[Tuple[str, str, str], InterventionRecord] = {}
        
        # Historical trades buffer for learning
        self.trade_history: deque[Dict[str, Any]] = deque(
            maxlen=int(self.max_memory_size * self._BUFFER_FRACTION)
        )
        
        # Current recommendation cache
        self.current_recommendation: Optional[Dict[str, Any]] = None
        
        # Statistics
        self.patterns_tracked: int = 0
        self.interventions_applied: int = 0
        self.counterfactuals_computed: int = 0
        
        # Metrics history
        self.recommendation_history: deque[Dict[str, Any]] = deque(maxlen=100)
        self.effectiveness_history: deque[Dict[str, float]] = deque(maxlen=100)
        
        self._log_debug(
            "interventions_initialized",
            details={
                "intervention_threshold": self.intervention_threshold,
                "veto_threshold": self.veto_threshold,
                "max_patterns": self._MAX_PATTERNS,
            },
        )
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process interventions operations."""
        try:
            # 1. Learn from completed trades
            learning_result = self._process_learning_data(context)
            
            # 2. Compute counterfactuals for recent losses
            counterfactual_result = self._compute_counterfactuals()
            learning_result.update(counterfactual_result)
            
            # 3. Generate recommendation for current context
            recommendation = self._generate_recommendation(context)
            learning_result.update(recommendation)
            
            # 4. Apply decay to old interventions
            self._apply_strength_decay()
            
            return self._format_output(learning_result)
            
        except Exception as e:
            self.log_error("Interventions processing failed", e)
            return self._get_fallback_output()
    
    def _process_learning_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process trades to update intervention table."""
        trades: List[Dict[str, Any]] = context.get("trades", []) or []
        market_context: Dict[str, Any] = context.get("market_context", {}) or {}
        
        trades_processed = 0
        
        for trade in trades[-20:]:
            if not isinstance(trade, dict) or "pnl" not in trade:
                continue
            
            # Extract instrument for per-instrument tracking
            instrument = trade.get("instrument") or trade.get("symbol") or "UNKNOWN"
            
            # Extract pattern label
            pattern_label = self._extract_pattern_label(trade, context)
            regime = str(market_context.get("regime", "unknown")).lower()
            
            # Store trade with metadata
            trade_record = {
                "pnl": float(trade["pnl"]),
                "instrument": instrument,
                "pattern_label": pattern_label,
                "regime": regime,
                "size": safe_float(trade.get("size", 1.0), 1.0),
                "sl_distance": safe_float(trade.get("sl_distance", 0.0), 0.0),
                "tp_distance": safe_float(trade.get("tp_distance", 0.0), 0.0),
                "timestamp": time.time(),
                "trade": trade,
            }
            self.trade_history.append(trade_record)
            trades_processed += 1
            
            # Update intervention table with outcome (now includes instrument)
            self._update_intervention_stats(instrument, pattern_label, regime, trade_record)
        
        return {
            "trades_processed": trades_processed,
            "history_size": len(self.trade_history),
        }
    
    def _extract_pattern_label(self, trade: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Extract or generate a pattern label for the trade.
        
        P2 FIX: Enhanced pattern labels with microstructure features:
        - Spread regime (tight/normal/wide)
        - Time-of-day bucket (more granular than session)
        - Recent momentum direction
        """
        # First, check if pattern_label is already in trade metadata
        if "pattern_label" in trade:
            return str(trade["pattern_label"])
        
        # Check context for pattern detection results
        pattern_recognition = context.get("pattern_recognition", {})
        if pattern_recognition:
            # Use most recent loss pattern if available
            loss_patterns = pattern_recognition.get("loss_patterns", {})
            if loss_patterns:
                # Return most frequent pattern
                return max(loss_patterns.keys(), key=lambda k: loss_patterns[k].get("count", 0))
        
        # Generate pattern label from trade features
        market_context = context.get("market_context", {}) or {}
        features: List[str] = []
        
        # Action type
        action = trade.get("action", [0.0, 0.0])
        if isinstance(action, (list, tuple, np.ndarray)):
            a0 = float(np.asarray(action)[0]) if len(action) > 0 else 0.0
            if a0 > 0.5:
                features.append("LONG")
            elif a0 < -0.5:
                features.append("SHORT")
            else:
                features.append("FLAT")
        
        # Regime (3 chars)
        regime = str(market_context.get("regime", "unknown")).lower()[:3]
        features.append(regime.upper())
        
        # P2 FIX: Spread regime (tight/normal/wide)
        spread = safe_float(trade.get("spread"), 0.0)
        avg_spread = safe_float(market_context.get("avg_spread"), spread)
        if avg_spread > 0:
            spread_ratio = spread / avg_spread
            if spread_ratio < 0.7:
                features.append("tsprd")  # tight spread
            elif spread_ratio > 1.5:
                features.append("wsprd")  # wide spread
            # Normal spread: don't add (reduces label cardinality)
        
        # P2 FIX: Time-of-day bucket (more granular than session)
        from datetime import datetime
        hour = datetime.now().hour
        if 0 <= hour < 6:
            features.append("overnight")
        elif 6 <= hour < 9:
            features.append("premarket")
        elif 9 <= hour < 12:
            features.append("morn")  # Morning
        elif 12 <= hour < 15:
            features.append("aftn")  # Afternoon
        elif 15 <= hour < 18:
            features.append("late")  # Late session
        else:
            features.append("afterhrs")
        
        # P2 FIX: Recent momentum direction
        recent_returns = safe_float(market_context.get("returns_5m", 0.0), 0.0)
        if recent_returns == 0.0:
            # Try other momentum indicators
            recent_returns = safe_float(market_context.get("momentum", 0.0), 0.0)
        if recent_returns > 0.001:
            features.append("moup")  # momentum up
        elif recent_returns < -0.001:
            features.append("modn")  # momentum down
        
        # Confidence level (simplified)
        conf = safe_float(trade.get("confidence", 0.5), 0.5)
        if conf > 0.7:
            features.append("HC")  # high confidence
        elif conf < 0.3:
            features.append("LC")  # low confidence
        
        return "-".join(features) if features else "UNKNOWN"
    
    def _update_intervention_stats(self, instrument: str, pattern_label: str, regime: str, trade_record: Dict[str, Any]) -> None:
        """Update intervention table stats for a (instrument, pattern, regime) key."""
        key = (instrument, pattern_label, regime)
        
        if key not in self.intervention_table:
            if len(self.intervention_table) >= self._MAX_PATTERNS:
                # Evict least used pattern
                self._evict_least_used_pattern()
            self.intervention_table[key] = InterventionRecord()
            self.patterns_tracked += 1
        
        record = self.intervention_table[key]
        pnl = trade_record["pnl"]
        
        # Update stats
        record.n += 1
        record.total_pnl_without += pnl
        record.last_updated = time.time()
        
        # If this was a loss, analyze potential interventions
        if pnl < 0:
            counterfactual = self._estimate_counterfactual(trade_record)
            record.total_pnl_with += counterfactual["estimated_pnl"]
            
            # Update best intervention if this one is better
            delta = counterfactual["estimated_pnl"] - pnl
            if delta > record.exp_delta:
                record.exp_delta = delta
                record.best_alt = counterfactual["best_intervention"]
                record.intervention = counterfactual["best_intervention"]
        
        # Update strength based on accumulated evidence
        if record.n >= self._MIN_SAMPLES_FOR_RECOMMENDATION:
            avg_without = record.total_pnl_without / record.n
            avg_with = record.total_pnl_with / max(1, record.n)
            
            # Strength: how much better is intervention vs no intervention
            if avg_without < 0:  # Pattern is generally losing
                improvement_ratio = (avg_with - avg_without) / (abs(avg_without) + 1e-8)
                record.strength = float(np.clip(improvement_ratio, 0.0, 1.0))
            else:
                record.strength = 0.0  # No intervention needed for winning patterns
    
    def _estimate_counterfactual(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate what would have happened with different interventions.
        
        Returns dict with:
        - best_intervention: Name of best intervention
        - estimated_pnl: Estimated PnL with best intervention
        - all_estimates: Dict of all intervention estimates
        """
        pnl = trade_record["pnl"]
        size = trade_record.get("size", 1.0)
        
        estimates: Dict[str, float] = {}
        
        # 1. avoid: Would have avoided the loss entirely
        estimates["avoid"] = 0.0
        
        # 2. halve_size: Reduces exposure proportionally
        estimates["halve_size"] = pnl * self._SIZE_REDUCTION_FACTOR
        
        # 3. tighter_sl: Would have limited loss
        # Assumes loss could have been limited by tighter SL
        if pnl < 0:
            estimates["tighter_sl"] = pnl * self._SL_TIGHTENING_FACTOR
        else:
            estimates["tighter_sl"] = pnl
        
        # 4. wider_tp: For losing trades, doesn't help much
        # But records it for completeness
        estimates["wider_tp"] = pnl
        
        # 5. none: Keep original
        estimates["none"] = pnl
        
        # Find best intervention
        best_intervention = max(estimates.keys(), key=lambda k: estimates[k])
        
        self.counterfactuals_computed += 1
        
        return {
            "best_intervention": best_intervention,
            "estimated_pnl": estimates[best_intervention],
            "all_estimates": estimates,
        }
    
    def _compute_counterfactuals(self) -> Dict[str, Any]:
        """Batch compute counterfactuals for recent losses."""
        recent_losses = [t for t in self.trade_history if t["pnl"] < 0][-10:]
        
        counterfactuals_computed = 0
        total_potential_improvement = 0.0
        
        for trade_record in recent_losses:
            cf = self._estimate_counterfactual(trade_record)
            improvement = cf["estimated_pnl"] - trade_record["pnl"]
            total_potential_improvement += improvement
            counterfactuals_computed += 1
        
        return {
            "counterfactuals_computed": counterfactuals_computed,
            "total_potential_improvement": total_potential_improvement,
        }
    
    def _generate_recommendation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intervention recommendation for current context."""
        market_context = context.get("market_context", {}) or {}
        
        # Get current instrument (try multiple sources)
        current_instrument = context.get("instrument") or context.get("symbol") or "UNKNOWN"
        
        # Get current pattern label (from context or generate)
        current_pattern = context.get("pattern_label")
        if current_pattern is None:
            # Try to infer from similar recent trades
            current_pattern = self._infer_current_pattern(context)
        
        regime = str(market_context.get("regime", "unknown")).lower()
        # 3-tuple key: (instrument, pattern_label, regime)
        key = (str(current_instrument), str(current_pattern), regime)
        
        # Check if we have intervention data for this key
        if key in self.intervention_table:
            record = self.intervention_table[key]
            
            if record.n >= self._MIN_SAMPLES_FOR_RECOMMENDATION and record.strength > 0.1:
                recommendation = {
                    "intervention": record.intervention,
                    "strength": record.strength,
                    "best_alt": record.best_alt,
                    "exp_delta": record.exp_delta,
                    "instrument": current_instrument,
                    "pattern_label": current_pattern,
                    "regime": regime,
                    "sample_count": record.n,
                    "veto_recommended": record.intervention == "avoid" and record.strength > self.veto_threshold,
                }
                
                # Map intervention to size/sl/tp multipliers
                recommendation.update(self._intervention_to_multipliers(record))
                
                self.current_recommendation = recommendation
                self.recommendation_history.append({
                    "timestamp": time.time(),
                    **recommendation,
                })
                
                return {"recommendation": recommendation, "recommendation_available": True}
        
        # Also check for cross-instrument patterns (instrument="ALL")
        fallback_key = ("ALL", str(current_pattern), regime)
        if fallback_key in self.intervention_table:
            record = self.intervention_table[fallback_key]
            if record.n >= self._MIN_SAMPLES_FOR_RECOMMENDATION and record.strength > 0.1:
                recommendation = {
                    "intervention": record.intervention,
                    "strength": record.strength * 0.8,  # Reduce confidence for cross-instrument
                    "best_alt": record.best_alt,
                    "exp_delta": record.exp_delta * 0.8,
                    "instrument": current_instrument,
                    "pattern_label": current_pattern,
                    "regime": regime,
                    "sample_count": record.n,
                    "veto_recommended": record.intervention == "avoid" and record.strength > self.veto_threshold,
                    "cross_instrument": True,  # Flag indicating this is a cross-instrument recommendation
                }
                recommendation.update(self._intervention_to_multipliers(record))
                self.current_recommendation = recommendation
                return {"recommendation": recommendation, "recommendation_available": True}
        
        # No recommendation available
        default_rec = {
            "intervention": "none",
            "strength": 0.0,
            "best_alt": "",
            "exp_delta": 0.0,
            "size_mult": 1.0,
            "sl_mult": 1.0,
            "tp_mult": 1.0,
            "veto_recommended": False,
            "instrument": current_instrument,
        }
        self.current_recommendation = default_rec
        
        return {"recommendation": default_rec, "recommendation_available": False}
    
    def _intervention_to_multipliers(self, record: InterventionRecord) -> Dict[str, float]:
        """Convert intervention type to size/sl/tp multipliers."""
        intervention = record.intervention
        strength = record.strength
        
        multipliers = {
            "size_mult": 1.0,
            "sl_mult": 1.0,
            "tp_mult": 1.0,
        }
        
        if intervention == "avoid":
            # Strong reduction
            multipliers["size_mult"] = max(0.1, 1.0 - strength * 0.9)
        
        elif intervention == "halve_size":
            # Reduce size proportionally to strength
            multipliers["size_mult"] = max(0.2, 1.0 - strength * 0.5)
        
        elif intervention == "tighter_sl":
            # Tighten SL (multiply by < 1 to bring SL closer)
            multipliers["sl_mult"] = max(0.5, 1.0 - strength * 0.3)
        
        elif intervention == "wider_tp":
            # Widen TP (multiply by > 1 to move TP further)
            multipliers["tp_mult"] = min(1.5, 1.0 + strength * 0.3)
        
        return multipliers
    
    def _infer_current_pattern(self, context: Dict[str, Any]) -> str:
        """Infer pattern label from context features."""
        market_context = context.get("market_context", {}) or {}
        
        features: List[str] = []
        
        # Volatility state
        vol = market_context.get("volatility", "medium")
        if isinstance(vol, dict):
            vol = str(list(vol.values())[0]) if vol else "medium"
        vol_str = str(vol).lower()
        
        if vol_str in ("high", "extreme"):
            features.append("HIGH_VOL")
        elif vol_str == "low":
            features.append("LOW_VOL")
        else:
            features.append("MED_VOL")
        
        # Session
        session = str(market_context.get("session", "unknown")).lower()
        features.append(session.upper()[:3])  # First 3 chars
        
        # Trend
        trend = market_context.get("trend", "neutral")
        if isinstance(trend, dict):
            trend = str(list(trend.values())[0]) if trend else "neutral"
        features.append(str(trend).upper()[:4])
        
        return "-".join(features) if features else "UNKNOWN"
    
    def _apply_strength_decay(self) -> None:
        """Apply decay to intervention strengths over time."""
        current_time = time.time()
        
        for record in self.intervention_table.values():
            # Decay based on time since last update (hourly)
            hours_since_update = (current_time - record.last_updated) / 3600.0
            decay = self._STRENGTH_DECAY ** hours_since_update
            record.strength *= decay
    
    def _evict_least_used_pattern(self) -> None:
        """Evict the least used pattern from intervention table."""
        if not self.intervention_table:
            return
        
        # Find pattern with lowest (n * strength)
        min_key = min(
            self.intervention_table.keys(),
            key=lambda k: self.intervention_table[k].n * (self.intervention_table[k].strength + 0.01)
        )
        del self.intervention_table[min_key]
    
    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements."""
        recommendation = self.current_recommendation or {}
        
        # Get top interventions by strength (now using 3-tuple keys: instrument, pattern, regime)
        top_interventions: List[Dict[str, Any]] = []
        for key, record in sorted(
            self.intervention_table.items(),
            key=lambda x: x[1].strength,
            reverse=True
        )[:5]:
            top_interventions.append({
                "instrument": key[0],  # First element is now instrument
                "pattern": key[1],     # Second element is pattern_label
                "regime": key[2],      # Third element is regime
                "intervention": record.intervention,
                "strength": record.strength,
                "exp_delta": record.exp_delta,
                "n": record.n,
            })
        
        return {
            "intervention_recommendation": {
                "intervention": recommendation.get("intervention", "none"),
                "strength": recommendation.get("strength", 0.0),
                "best_alt": recommendation.get("best_alt", ""),
                "exp_delta": recommendation.get("exp_delta", 0.0),
                "size_mult": recommendation.get("size_mult", 1.0),
                "sl_mult": recommendation.get("sl_mult", 1.0),
                "tp_mult": recommendation.get("tp_mult", 1.0),
                "veto_recommended": recommendation.get("veto_recommended", False),
            },
            "intervention_table": {
                "patterns_tracked": self.patterns_tracked,
                "total_counterfactuals": self.counterfactuals_computed,
                "top_interventions": top_interventions,
            },
            "intervention_metrics": {
                "history_size": len(self.trade_history),
                "recommendations_made": len(self.recommendation_history),
                "avg_strength": float(np.mean([r.strength for r in self.intervention_table.values()])) if self.intervention_table else 0.0,
            },
        }
    
    def _get_fallback_output(self) -> Dict[str, Any]:
        """Return fallback output on error."""
        return {
            "intervention_recommendation": {
                "intervention": "none",
                "strength": 0.0,
                "best_alt": "",
                "exp_delta": 0.0,
                "size_mult": 1.0,
                "sl_mult": 1.0,
                "tp_mult": 1.0,
                "veto_recommended": False,
            },
            "intervention_table": {
                "patterns_tracked": 0,
                "total_counterfactuals": 0,
                "top_interventions": [],
            },
            "intervention_metrics": {
                "history_size": 0,
                "recommendations_made": 0,
                "avg_strength": 0.0,
            },
        }
