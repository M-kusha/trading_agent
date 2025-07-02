# ─────────────────────────────────────────────────────────────
# File: modules/voting/time_horizon_aligner.py
# Enhanced Time Horizon Aligner with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class TimeHorizonAligner(Module, AnalysisMixin, StateManagementMixin):
    """
    Enhanced time horizon aligner with InfoBus integration.
    Applies time-based scaling to voting weights based on member time horizons,
    market conditions, and performance patterns.
    """

    def __init__(
        self,
        horizons: List[int],
        adaptive_scaling: bool = True,
        regime_awareness: bool = True,
        performance_feedback: bool = True,
        debug: bool = False,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 100),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        
        # Core parameters
        self.horizons = np.asarray(horizons, dtype=np.float32)
        self.adaptive_scaling = bool(adaptive_scaling)
        self.regime_awareness = bool(regime_awareness)
        self.performance_feedback = bool(performance_feedback)
        
        # Time tracking
        self.clock = 0
        self.session_start = 0
        self.alignment_history = deque(maxlen=100)
        
        # Horizon analysis
        self.horizon_performance = defaultdict(lambda: {
            'total_weight': 0.0,
            'successful_weight': 0.0,
            'performance_score': 0.5,
            'recent_scores': deque(maxlen=20)
        })
        
        # Market regime adaptations
        self.regime_multipliers = {
            'trending': np.ones_like(self.horizons),
            'volatile': np.ones_like(self.horizons),
            'ranging': np.ones_like(self.horizons),
            'noise': np.ones_like(self.horizons),
            'unknown': np.ones_like(self.horizons)
        }
        
        # Dynamic scaling factors
        self.base_distances = np.ones_like(self.horizons)
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.performance_multipliers = np.ones_like(self.horizons)
        
        # Session and cyclical patterns
        self.session_patterns = defaultdict(lambda: np.ones_like(self.horizons))
        self.cyclical_adjustments = np.ones_like(self.horizons)
        
        # Statistics
        self.alignment_stats = {
            'total_alignments': 0,
            'adaptations_made': 0,
            'avg_distance_impact': 0.0,
            'regime_switches': 0,
            'performance_adjustments': 0
        }
        
        # Volatility and market state tracking
        self.current_regime = 'unknown'
        self.current_session = 'unknown'
        self.current_volatility = 0.02
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.1
        self.performance_window = 10
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "TimeHorizonAligner",
            "logs/voting/time_horizon_aligner.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("TimeHorizonAligner")
        
        self.log_operator_info(
            "⏰ Time Horizon Aligner initialized",
            horizons=horizons,
            adaptive=self.adaptive_scaling,
            regime_aware=self.regime_awareness,
            performance_feedback=self.performance_feedback
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Reset time tracking
        self.clock = 0
        self.session_start = 0
        
        # Reset multipliers to neutral
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.performance_multipliers = np.ones_like(self.horizons)
        self.cyclical_adjustments = np.ones_like(self.horizons)
        
        # Reset history
        self.alignment_history.clear()
        
        # Reset performance tracking
        self.horizon_performance.clear()
        
        # Reset market state
        self.current_regime = 'unknown'
        self.current_session = 'unknown'
        self.current_volatility = 0.02
        
        # Reset statistics
        self.alignment_stats = {
            'total_alignments': 0,
            'adaptations_made': 0,
            'avg_distance_impact': 0.0,
            'regime_switches': 0,
            'performance_adjustments': 0
        }
        
        self.log_operator_info("🔄 Time Horizon Aligner reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Increment clock
        self.clock += 1
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using basic step")
            return
        
        # Extract context and market data
        context = extract_standard_context(info_bus)
        market_data = self._extract_market_data_from_info_bus(info_bus)
        
        # Update market state tracking
        self._update_market_state(context, market_data)
        
        # Update regime-based multipliers
        self._update_regime_multipliers(context)
        
        # Update performance-based adjustments
        if self.performance_feedback:
            self._update_performance_adjustments(market_data)
        
        # Update cyclical patterns
        self._update_cyclical_patterns(context)
        
        # Update InfoBus with alignment state
        self._update_info_bus(info_bus)

    def _extract_market_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract market data for horizon alignment"""
        
        data = {}
        
        try:
            # Get trading performance
            recent_trades = info_bus.get('recent_trades', [])
            data['recent_trades'] = recent_trades
            
            # Get voting data
            module_data = info_bus.get('module_data', {})
            arbiter_data = module_data.get('strategy_arbiter', {})
            data['last_weights'] = arbiter_data.get('weights', [])
            data['last_alpha'] = arbiter_data.get('last_alpha', [])
            
            # Get market context
            market_context = info_bus.get('market_context', {})
            data['volatility'] = market_context.get('volatility', {})
            data['trend_strength'] = market_context.get('trend_strength', 0.5)
            
            # Get performance metrics
            if recent_trades:
                data['recent_pnl'] = [trade.get('pnl', 0) for trade in recent_trades[-10:]]
                data['recent_success'] = sum(1 for trade in recent_trades[-10:] if trade.get('pnl', 0) > 0)
            else:
                data['recent_pnl'] = []
                data['recent_success'] = 0
            
        except Exception as e:
            self.log_operator_warning(f"Market data extraction failed: {e}")
            data = {
                'recent_trades': [],
                'last_weights': [],
                'last_alpha': [],
                'volatility': {},
                'trend_strength': 0.5,
                'recent_pnl': [],
                'recent_success': 0
            }
        
        return data

    def _update_market_state(self, context: Dict[str, Any], market_data: Dict[str, Any]) -> None:
        """Update market state tracking"""
        
        try:
            # Update regime
            old_regime = self.current_regime
            self.current_regime = context.get('regime', 'unknown')
            
            if old_regime != self.current_regime and old_regime != 'unknown':
                self.alignment_stats['regime_switches'] += 1
                self.log_operator_info(
                    f"📊 Market regime changed: {old_regime} → {self.current_regime}",
                    clock=self.clock,
                    impact="Horizon weights will adapt"
                )
            
            # Update session
            self.current_session = context.get('session', 'unknown')
            
            # Update volatility
            volatilities = market_data.get('volatility', {})
            if volatilities:
                self.current_volatility = np.mean(list(volatilities.values()))
            else:
                self.current_volatility = 0.02
            
        except Exception as e:
            self.log_operator_warning(f"Market state update failed: {e}")

    def _update_regime_multipliers(self, context: Dict[str, Any]) -> None:
        """Update regime-based horizon multipliers"""
        
        if not self.regime_awareness:
            return
        
        try:
            regime = self.current_regime
            volatility_level = context.get('volatility_level', 'medium')
            
            # Define regime-specific horizon preferences
            if regime == 'trending':
                # Favor longer horizons in trending markets
                multipliers = np.array([
                    0.8 if h < 5 else 1.2 if h > 20 else 1.0 
                    for h in self.horizons
                ])
                
            elif regime == 'volatile':
                # Favor shorter horizons in volatile markets
                multipliers = np.array([
                    1.3 if h < 10 else 0.7 if h > 30 else 1.0 
                    for h in self.horizons
                ])
                
            elif regime == 'ranging':
                # Balanced approach in ranging markets
                multipliers = np.array([
                    1.1 if 5 <= h <= 20 else 0.9 
                    for h in self.horizons
                ])
                
            elif regime == 'noise':
                # Very short horizons in noisy markets
                multipliers = np.array([
                    1.4 if h < 5 else 0.6 if h > 15 else 1.0 
                    for h in self.horizons
                ])
                
            else:  # unknown
                multipliers = np.ones_like(self.horizons)
            
            # Adjust for volatility level
            if volatility_level == 'extreme':
                # Strongly favor short horizons
                multipliers = multipliers * np.array([
                    1.5 if h < 5 else 0.5 if h > 20 else 1.0 
                    for h in self.horizons
                ])
            elif volatility_level == 'low':
                # Allow longer horizons
                multipliers = multipliers * np.array([
                    0.9 if h < 3 else 1.2 if h > 15 else 1.0 
                    for h in self.horizons
                ])
            
            # Smooth transition
            if hasattr(self, '_last_regime_multipliers'):
                alpha = 0.3  # Smoothing factor
                multipliers = alpha * multipliers + (1 - alpha) * self._last_regime_multipliers
            
            self.regime_multipliers[regime] = multipliers
            self._last_regime_multipliers = multipliers.copy()
            
        except Exception as e:
            self.log_operator_warning(f"Regime multiplier update failed: {e}")

    def _update_performance_adjustments(self, market_data: Dict[str, Any]) -> None:
        """Update performance-based horizon adjustments"""
        
        try:
            last_weights = market_data.get('last_weights', [])
            last_alpha = market_data.get('last_alpha', [])
            recent_pnl = market_data.get('recent_pnl', [])
            
            if not last_weights or not recent_pnl:
                return
            
            # Ensure we have the right number of weights for horizons
            if len(last_weights) != len(self.horizons):
                return
            
            # Calculate recent performance
            recent_performance = np.mean(recent_pnl) if recent_pnl else 0.0
            
            # Update horizon performance tracking
            for i, (horizon, weight) in enumerate(zip(self.horizons, last_weights)):
                if i < len(self.horizons):
                    perf_data = self.horizon_performance[i]
                    
                    # Track weighted performance
                    perf_data['total_weight'] += weight
                    if recent_performance > 0:
                        perf_data['successful_weight'] += weight
                    
                    # Add recent score
                    score = 1.0 if recent_performance > 0 else 0.0
                    perf_data['recent_scores'].append(score)
                    
                    # Calculate performance score
                    if len(perf_data['recent_scores']) >= 3:
                        perf_data['performance_score'] = np.mean(list(perf_data['recent_scores']))
                    
                    # Update performance multiplier
                    if perf_data['performance_score'] > 0.7:
                        self.performance_multipliers[i] = min(1.5, self.performance_multipliers[i] + 0.1)
                    elif perf_data['performance_score'] < 0.3:
                        self.performance_multipliers[i] = max(0.5, self.performance_multipliers[i] - 0.1)
                    
                    self.alignment_stats['performance_adjustments'] += 1
            
        except Exception as e:
            self.log_operator_warning(f"Performance adjustment update failed: {e}")

    def _update_cyclical_patterns(self, context: Dict[str, Any]) -> None:
        """Update cyclical time-based patterns"""
        
        try:
            session = context.get('session', 'unknown')
            
            # Session-based patterns
            if session == 'american':
                # High activity session - favor medium-term horizons
                session_mult = np.array([
                    0.9 if h < 3 else 1.2 if 5 <= h <= 25 else 0.8 
                    for h in self.horizons
                ])
            elif session == 'european':
                # Balanced session
                session_mult = np.array([
                    1.0 if 3 <= h <= 20 else 0.9 
                    for h in self.horizons
                ])
            elif session == 'asian':
                # Lower volatility - allow longer horizons
                session_mult = np.array([
                    0.8 if h < 5 else 1.3 if h > 15 else 1.0 
                    for h in self.horizons
                ])
            elif session == 'rollover':
                # Low liquidity - favor very short horizons
                session_mult = np.array([
                    1.5 if h < 3 else 0.5 if h > 10 else 1.0 
                    for h in self.horizons
                ])
            else:
                session_mult = np.ones_like(self.horizons)
            
            # Time-of-day cyclical adjustments
            hour_cycle = (self.clock % 24) / 24.0  # Normalize to 0-1
            
            # Create sinusoidal pattern that favors different horizons at different times
            time_mult = np.array([
                1.0 + 0.2 * np.sin(2 * np.pi * hour_cycle + h / 10.0) 
                for h in self.horizons
            ])
            
            # Combine session and time patterns
            self.cyclical_adjustments = session_mult * time_mult
            
        except Exception as e:
            self.log_operator_warning(f"Cyclical pattern update failed: {e}")

    def apply(self, weights: np.ndarray) -> np.ndarray:
        """
        Enhanced time-based scaling to weights with comprehensive adjustments.
        
        Args:
            weights: Current voting weights to adjust
            
        Returns:
            Adjusted weights based on time horizons and market conditions
        """
        
        try:
            self.alignment_stats['total_alignments'] += 1
            
            # Validate inputs
            weights = np.asarray(weights, dtype=np.float32)
            
            if len(weights) != len(self.horizons):
                self.log_operator_warning(
                    f"Weight/horizon dimension mismatch: {len(weights)} vs {len(self.horizons)}"
                )
                return weights
            
            # Calculate base distance scaling
            self._calculate_base_distances()
            
            # Combine all scaling factors
            total_multipliers = self._combine_scaling_factors()
            
            # Apply distance and multiplier scaling
            distance_scaled = weights * self.base_distances
            final_scaled = distance_scaled * total_multipliers
            
            # Ensure positive and normalized
            final_scaled = np.maximum(final_scaled, 0.01)
            final_scaled = final_scaled / (final_scaled.sum() + 1e-12)
            
            # Track alignment impact
            impact = np.linalg.norm(final_scaled - weights)
            self.alignment_stats['avg_distance_impact'] = (
                self.alignment_stats['avg_distance_impact'] * 0.9 + impact * 0.1
            )
            
            # Record alignment event
            alignment_event = {
                'timestamp': datetime.datetime.now().isoformat(),
                'clock': self.clock,
                'original_weights': weights.tolist(),
                'aligned_weights': final_scaled.tolist(),
                'impact': float(impact),
                'regime': self.current_regime,
                'session': self.current_session,
                'base_distances': self.base_distances.tolist(),
                'total_multipliers': total_multipliers.tolist()
            }
            self.alignment_history.append(alignment_event)
            
            # Check for significant adaptations
            if impact > self.adaptation_threshold:
                self.alignment_stats['adaptations_made'] += 1
                self.log_operator_info(
                    f"⏰ Significant horizon alignment applied",
                    impact=f"{impact:.3f}",
                    regime=self.current_regime,
                    clock=self.clock
                )
            
            return final_scaled
            
        except Exception as e:
            self.log_operator_error(f"Horizon alignment failed: {e}")
            return weights

    def _calculate_base_distances(self) -> None:
        """Calculate base distance scaling from time horizons"""
        
        try:
            # Distance from each horizon to current time
            distances = 1.0 / (1.0 + np.abs(self.clock - self.horizons))
            
            # Normalize distances
            self.base_distances = distances / (distances.sum() + 1e-12)
            
        except Exception as e:
            self.log_operator_warning(f"Base distance calculation failed: {e}")
            self.base_distances = np.ones_like(self.horizons) / len(self.horizons)

    def _combine_scaling_factors(self) -> np.ndarray:
        """Combine all scaling factors into total multipliers"""
        
        try:
            # Start with adaptive multipliers
            total = self.adaptive_multipliers.copy()
            
            # Apply regime multipliers if available
            if self.regime_awareness and self.current_regime in self.regime_multipliers:
                total *= self.regime_multipliers[self.current_regime]
            
            # Apply performance multipliers
            if self.performance_feedback:
                total *= self.performance_multipliers
            
            # Apply cyclical adjustments
            total *= self.cyclical_adjustments
            
            # Normalize to prevent extreme scaling
            total = np.clip(total, 0.1, 3.0)
            
            return total
            
        except Exception as e:
            self.log_operator_warning(f"Scaling factor combination failed: {e}")
            return np.ones_like(self.horizons)

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with horizon alignment results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'time_horizon_aligner', {
            'clock': self.clock,
            'horizons': self.horizons.tolist(),
            'current_regime': self.current_regime,
            'current_session': self.current_session,
            'current_volatility': self.current_volatility,
            'base_distances': self.base_distances.tolist(),
            'adaptive_multipliers': self.adaptive_multipliers.tolist(),
            'performance_multipliers': self.performance_multipliers.tolist(),
            'cyclical_adjustments': self.cyclical_adjustments.tolist(),
            'alignment_stats': self.alignment_stats.copy(),
            'horizon_performance': {
                str(k): {
                    'performance_score': v['performance_score'],
                    'total_weight': v['total_weight'],
                    'successful_weight': v['successful_weight']
                } for k, v in self.horizon_performance.items()
            },
            'configuration': {
                'adaptive_scaling': self.adaptive_scaling,
                'regime_awareness': self.regime_awareness,
                'performance_feedback': self.performance_feedback
            }
        })

    def get_horizon_report(self) -> str:
        """Generate operator-friendly horizon alignment report"""
        
        # Alignment status
        recent_impact = self.alignment_stats.get('avg_distance_impact', 0.0)
        if recent_impact > 0.3:
            alignment_status = "🔄 ACTIVE"
        elif recent_impact > 0.1:
            alignment_status = "⚡ MODERATE"
        else:
            alignment_status = "→ MINIMAL"
        
        # Best performing horizons
        horizon_performance_list = []
        for i, horizon in enumerate(self.horizons):
            if i in self.horizon_performance:
                perf_data = self.horizon_performance[i]
                score = perf_data['performance_score']
                if score > 0.6:
                    emoji = "✅"
                elif score > 0.4:
                    emoji = "⚡"
                else:
                    emoji = "⚠️"
                horizon_performance_list.append(f"  {emoji} {horizon}-step: {score:.1%}")
        
        # Regime multipliers
        current_multipliers = self.regime_multipliers.get(self.current_regime, np.ones_like(self.horizons))
        multiplier_lines = []
        for i, (horizon, mult) in enumerate(zip(self.horizons, current_multipliers)):
            if mult > 1.1:
                emoji = "📈"
            elif mult < 0.9:
                emoji = "📉"
            else:
                emoji = "→"
            multiplier_lines.append(f"  {emoji} {horizon}-step: {mult:.2f}x")
        
        return f"""
⏰ TIME HORIZON ALIGNER
═══════════════════════════════════════
🎯 Alignment Status: {alignment_status}
⏱️ Current Clock: {self.clock}
📊 Recent Impact: {recent_impact:.3f}

📈 Market Context:
• Regime: {self.current_regime.title()}
• Session: {self.current_session.title()}
• Volatility: {self.current_volatility:.2%}

🎯 Horizon Configuration:
• Time Horizons: {', '.join(map(str, self.horizons))}
• Adaptive Scaling: {'✅ Enabled' if self.adaptive_scaling else '❌ Disabled'}
• Regime Awareness: {'✅ Enabled' if self.regime_awareness else '❌ Disabled'}
• Performance Feedback: {'✅ Enabled' if self.performance_feedback else '❌ Disabled'}

📊 Horizon Performance:
{chr(10).join(horizon_performance_list) if horizon_performance_list else "  📭 No performance data available"}

🎚️ Current Regime Multipliers ({self.current_regime}):
{chr(10).join(multiplier_lines[:6]) if multiplier_lines else "  📭 No multipliers available"}

📈 Alignment Statistics:
• Total Alignments: {self.alignment_stats['total_alignments']}
• Significant Adaptations: {self.alignment_stats['adaptations_made']}
• Regime Switches: {self.alignment_stats['regime_switches']}
• Performance Adjustments: {self.alignment_stats['performance_adjustments']}

🔧 Current Scaling Factors:
• Base Distance Impact: {np.mean(self.base_distances):.3f}
• Adaptive Multiplier: {np.mean(self.adaptive_multipliers):.3f}
• Performance Multiplier: {np.mean(self.performance_multipliers):.3f}
• Cyclical Adjustment: {np.mean(self.cyclical_adjustments):.3f}

📊 Recent Activity:
• Alignment History: {len(self.alignment_history)} entries
• Horizon Performance Tracked: {len(self.horizon_performance)} horizons
        """

    def get_observation_components(self) -> np.ndarray:
        """Return horizon alignment features for observation"""
        
        try:
            features = [
                float(self.clock % 100) / 100.0,  # Normalized clock
                float(self.alignment_stats.get('avg_distance_impact', 0.0)),
                float(np.mean(self.base_distances)),
                float(np.mean(self.adaptive_multipliers)),
                float(len(self.alignment_history) / 100.0)  # History fullness
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> None:
        """Legacy step interface for backward compatibility"""
        self.clock += 1

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "horizons": self.horizons.tolist(),
                "adaptive_scaling": self.adaptive_scaling,
                "regime_awareness": self.regime_awareness,
                "performance_feedback": self.performance_feedback
            },
            "time_state": {
                "clock": self.clock,
                "session_start": self.session_start,
                "current_regime": self.current_regime,
                "current_session": self.current_session,
                "current_volatility": self.current_volatility
            },
            "scaling_factors": {
                "base_distances": self.base_distances.tolist(),
                "adaptive_multipliers": self.adaptive_multipliers.tolist(),
                "performance_multipliers": self.performance_multipliers.tolist(),
                "cyclical_adjustments": self.cyclical_adjustments.tolist()
            },
            "regime_multipliers": {
                regime: multipliers.tolist() 
                for regime, multipliers in self.regime_multipliers.items()
            },
            "horizon_performance": {
                str(k): {
                    'performance_score': v['performance_score'],
                    'total_weight': v['total_weight'],
                    'successful_weight': v['successful_weight'],
                    'recent_scores': list(v['recent_scores'])
                } for k, v in self.horizon_performance.items()
            },
            "statistics": self.alignment_stats.copy(),
            "history": {
                "alignment_history": list(self.alignment_history)[-20:]
            },
            "parameters": {
                "learning_rate": self.learning_rate,
                "adaptation_threshold": self.adaptation_threshold,
                "performance_window": self.performance_window
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        horizons = config.get("horizons", self.horizons.tolist())
        self.horizons = np.array(horizons, dtype=np.float32)
        self.adaptive_scaling = bool(config.get("adaptive_scaling", self.adaptive_scaling))
        self.regime_awareness = bool(config.get("regime_awareness", self.regime_awareness))
        self.performance_feedback = bool(config.get("performance_feedback", self.performance_feedback))
        
        # Load time state
        time_state = state.get("time_state", {})
        self.clock = int(time_state.get("clock", 0))
        self.session_start = int(time_state.get("session_start", 0))
        self.current_regime = time_state.get("current_regime", "unknown")
        self.current_session = time_state.get("current_session", "unknown")
        self.current_volatility = float(time_state.get("current_volatility", 0.02))
        
        # Load scaling factors
        scaling_factors = state.get("scaling_factors", {})
        
        base_distances = scaling_factors.get("base_distances", np.ones_like(self.horizons).tolist())
        self.base_distances = np.array(base_distances, dtype=np.float32)
        
        adaptive_multipliers = scaling_factors.get("adaptive_multipliers", np.ones_like(self.horizons).tolist())
        self.adaptive_multipliers = np.array(adaptive_multipliers, dtype=np.float32)
        
        performance_multipliers = scaling_factors.get("performance_multipliers", np.ones_like(self.horizons).tolist())
        self.performance_multipliers = np.array(performance_multipliers, dtype=np.float32)
        
        cyclical_adjustments = scaling_factors.get("cyclical_adjustments", np.ones_like(self.horizons).tolist())
        self.cyclical_adjustments = np.array(cyclical_adjustments, dtype=np.float32)
        
        # Load regime multipliers
        regime_multipliers = state.get("regime_multipliers", {})
        for regime, multipliers in regime_multipliers.items():
            self.regime_multipliers[regime] = np.array(multipliers, dtype=np.float32)
        
        # Load horizon performance
        horizon_performance = state.get("horizon_performance", {})
        self.horizon_performance.clear()
        for horizon_id, perf_data in horizon_performance.items():
            horizon_idx = int(horizon_id)
            self.horizon_performance[horizon_idx] = {
                'performance_score': perf_data['performance_score'],
                'total_weight': perf_data['total_weight'],
                'successful_weight': perf_data['successful_weight'],
                'recent_scores': deque(perf_data['recent_scores'], maxlen=20)
            }
        
        # Load statistics
        self.alignment_stats.update(state.get("statistics", {}))
        
        # Load history
        history = state.get("history", {})
        alignment_history = history.get("alignment_history", [])
        self.alignment_history.clear()
        for entry in alignment_history:
            self.alignment_history.append(entry)
        
        # Load parameters
        parameters = state.get("parameters", {})
        self.learning_rate = float(parameters.get("learning_rate", self.learning_rate))
        self.adaptation_threshold = float(parameters.get("adaptation_threshold", self.adaptation_threshold))
        self.performance_window = int(parameters.get("performance_window", self.performance_window))