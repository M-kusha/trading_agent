# ─────────────────────────────────────────────────────────────
# File: modules/strategy/opponent_mode_enhancer.py
# Enhanced with InfoBus integration & intelligent mode adaptation
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class OpponentModeEnhancer(Module, AnalysisMixin, StateManagementMixin, TradingMixin):
    """
    Enhanced opponent mode enhancer with InfoBus integration.
    Analyzes market conditions and adapts strategy modes based on market behavior patterns.
    Provides intelligent mode weighting for dynamic strategy adaptation.
    """

    def __init__(
        self,
        modes: Optional[List[str]] = None,
        debug: bool = False,
        adaptation_rate: float = 0.15,
        confidence_threshold: float = 0.7,
        mode_switch_cooldown: int = 5,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 200),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.debug = bool(debug)
        self.adaptation_rate = float(adaptation_rate)
        self.confidence_threshold = float(confidence_threshold)
        self.mode_switch_cooldown = int(mode_switch_cooldown)
        
        # Initialize market modes with enhanced definitions
        self.modes = modes or ["trending", "ranging", "volatile", "breakout", "reversal"]
        self.mode_definitions = self._initialize_mode_definitions()
        
        # Mode tracking state
        self.mode_performance = defaultdict(lambda: defaultdict(list))
        self.mode_counts = defaultdict(int)
        self.mode_history = deque(maxlen=100)
        self.current_mode_weights = {mode: 1.0/len(self.modes) for mode in self.modes}
        
        # Enhanced analytics
        self.mode_analytics = {
            'mode_transitions': defaultdict(int),
            'mode_duration_stats': defaultdict(list),
            'mode_profitability': defaultdict(float),
            'mode_win_rates': defaultdict(float),
            'mode_confidence_scores': defaultdict(float),
            'last_mode_switch': None,
            'switches_since_reset': 0
        }
        
        # Market condition detection
        self.condition_detectors = {
            'trending': self._detect_trending_condition,
            'ranging': self._detect_ranging_condition,
            'volatile': self._detect_volatile_condition,
            'breakout': self._detect_breakout_condition,
            'reversal': self._detect_reversal_condition
        }
        
        # Performance thresholds for mode assessment
        self.performance_thresholds = {
            'excellent': 100.0,
            'good': 50.0,
            'neutral': 0.0,
            'poor': -25.0,
            'very_poor': -75.0
        }
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "OpponentModeEnhancer",
            "logs/strategy/opponent_mode_enhancer.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("OpponentModeEnhancer")
        
        self.log_operator_info(
            "🎯 Opponent Mode Enhancer initialized",
            modes=self.modes,
            adaptation_rate=self.adaptation_rate,
            confidence_threshold=self.confidence_threshold
        )

    def _initialize_mode_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize detailed mode definitions with characteristics"""
        
        return {
            'trending': {
                'description': 'Strong directional price movement',
                'characteristics': ['sustained_direction', 'higher_highs_lows', 'momentum'],
                'indicators': ['price_slope', 'momentum_strength', 'trend_duration'],
                'optimal_strategies': ['momentum', 'breakout', 'trend_following'],
                'risk_factors': ['trend_exhaustion', 'reversal_signals'],
                'profit_potential': 'high',
                'typical_duration': '30-120 minutes'
            },
            'ranging': {
                'description': 'Sideways price movement within bounds',
                'characteristics': ['horizontal_support_resistance', 'oscillation', 'mean_reversion'],
                'indicators': ['range_width', 'bounce_frequency', 'volume_profile'],
                'optimal_strategies': ['mean_reversion', 'support_resistance', 'scalping'],
                'risk_factors': ['range_breakdown', 'false_breakouts'],
                'profit_potential': 'medium',
                'typical_duration': '45-180 minutes'
            },
            'volatile': {
                'description': 'High price volatility and uncertainty',
                'characteristics': ['large_price_swings', 'unpredictable_moves', 'high_noise'],
                'indicators': ['atr_expansion', 'price_acceleration', 'gap_frequency'],
                'optimal_strategies': ['volatility_capture', 'wide_stops', 'reduced_size'],
                'risk_factors': ['whipsaws', 'gap_risk', 'overexposure'],
                'profit_potential': 'high_risk_high_reward',
                'typical_duration': '15-60 minutes'
            },
            'breakout': {
                'description': 'Price breaking through key levels',
                'characteristics': ['level_penetration', 'volume_surge', 'momentum_acceleration'],
                'indicators': ['breakout_strength', 'volume_confirmation', 'follow_through'],
                'optimal_strategies': ['breakout_momentum', 'continuation', 'expansion'],
                'risk_factors': ['false_breakouts', 'fade_risk', 'trap_setups'],
                'profit_potential': 'very_high',
                'typical_duration': '10-45 minutes'
            },
            'reversal': {
                'description': 'Trend change and direction reversal',
                'characteristics': ['exhaustion_signals', 'divergence', 'pattern_completion'],
                'indicators': ['reversal_patterns', 'momentum_divergence', 'volume_analysis'],
                'optimal_strategies': ['counter_trend', 'reversal_trading', 'pattern_recognition'],
                'risk_factors': ['false_reversals', 'trend_continuation', 'timing_risk'],
                'profit_potential': 'high',
                'typical_duration': '20-90 minutes'
            }
        }

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Clear mode tracking
        self.mode_performance.clear()
        self.mode_counts.clear()
        self.mode_history.clear()
        
        # Reset mode weights to equal distribution
        self.current_mode_weights = {mode: 1.0/len(self.modes) for mode in self.modes}
        
        # Reset analytics
        self.mode_analytics = {
            'mode_transitions': defaultdict(int),
            'mode_duration_stats': defaultdict(list),
            'mode_profitability': defaultdict(float),
            'mode_win_rates': defaultdict(float),
            'mode_confidence_scores': defaultdict(float),
            'last_mode_switch': None,
            'switches_since_reset': 0
        }
        
        self.log_operator_info("🔄 Opponent Mode Enhancer reset - all mode analytics cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and mode analysis"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - limited mode analysis")
            return
        
        # Extract context and market data
        context = extract_standard_context(info_bus)
        market_data = self._extract_market_data_from_info_bus(info_bus, context)
        
        # Detect current market modes
        detected_modes = self._detect_current_market_modes(market_data, context)
        
        # Update mode performance based on recent results
        self._update_mode_performance_from_info_bus(info_bus, detected_modes)
        
        # Adapt mode weights based on performance
        self._adapt_mode_weights(detected_modes, market_data)
        
        # Update InfoBus with mode analysis
        self._update_info_bus_with_mode_data(info_bus)

    def _extract_market_data_from_info_bus(self, info_bus: InfoBus, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market data needed for mode analysis"""
        
        try:
            # Get price and volatility data
            prices = info_bus.get('prices', {})
            volatility_data = info_bus.get('volatility', {})
            
            # Get recent trades for momentum analysis
            recent_trades = info_bus.get('recent_trades', [])
            
            # Get regime information
            regime = context.get('regime', 'unknown')
            volatility_level = context.get('volatility_level', 'medium')
            
            # Extract technical indicators if available
            technical_data = info_bus.get('technical_indicators', {})
            
            market_data = {
                'prices': prices,
                'volatility': volatility_data,
                'recent_trades': recent_trades,
                'regime': regime,
                'volatility_level': volatility_level,
                'technical_indicators': technical_data,
                'timestamp': datetime.datetime.now().isoformat(),
                'market_open': context.get('market_open', True),
                'session': context.get('session', 'unknown')
            }
            
            # Calculate derived metrics
            market_data.update(self._calculate_derived_market_metrics(market_data))
            
            return market_data
            
        except Exception as e:
            self.log_operator_warning(f"Market data extraction failed: {e}")
            return {'timestamp': datetime.datetime.now().isoformat()}

    def _calculate_derived_market_metrics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived metrics for mode detection"""
        
        derived_metrics = {}
        
        try:
            recent_trades = market_data.get('recent_trades', [])
            
            if recent_trades and len(recent_trades) >= 3:
                # Price momentum
                prices = [t.get('entry_price', 0) for t in recent_trades[-10:] if t.get('entry_price', 0) > 0]
                if len(prices) >= 3:
                    price_changes = np.diff(prices)
                    derived_metrics['price_momentum'] = np.mean(price_changes)
                    derived_metrics['price_volatility'] = np.std(price_changes)
                    derived_metrics['price_trend_strength'] = abs(np.mean(price_changes)) / (np.std(price_changes) + 1e-6)
                
                # Trade frequency and patterns
                trade_intervals = []
                for i in range(1, min(len(recent_trades), 10)):
                    current_time = recent_trades[i].get('timestamp', '')
                    prev_time = recent_trades[i-1].get('timestamp', '')
                    # Simplified interval calculation
                    trade_intervals.append(1.0)  # Placeholder
                
                if trade_intervals:
                    derived_metrics['trade_frequency'] = 1.0 / (np.mean(trade_intervals) + 1e-6)
                    derived_metrics['trade_regularity'] = 1.0 - (np.std(trade_intervals) / (np.mean(trade_intervals) + 1e-6))
                
                # Performance patterns
                pnls = [t.get('pnl', 0) for t in recent_trades]
                if pnls:
                    derived_metrics['recent_performance_trend'] = np.mean(pnls[-5:]) if len(pnls) >= 5 else np.mean(pnls)
                    derived_metrics['performance_consistency'] = 1.0 - (np.std(pnls) / (abs(np.mean(pnls)) + 1e-6))
            
            # Volatility assessment
            volatility_level = market_data.get('volatility_level', 'medium')
            derived_metrics['volatility_score'] = {
                'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0
            }.get(volatility_level, 0.5)
            
        except Exception as e:
            self.log_operator_warning(f"Derived metrics calculation failed: {e}")
        
        return derived_metrics

    def _detect_current_market_modes(self, market_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """Detect current market modes with confidence scores"""
        
        detected_modes = {}
        
        try:
            for mode in self.modes:
                if mode in self.condition_detectors:
                    confidence = self.condition_detectors[mode](market_data, context)
                    detected_modes[mode] = max(0.0, min(1.0, confidence))
                else:
                    detected_modes[mode] = 0.0
            
            # Normalize mode confidences
            total_confidence = sum(detected_modes.values())
            if total_confidence > 0:
                detected_modes = {k: v/total_confidence for k, v in detected_modes.items()}
            else:
                # Equal distribution if no clear mode detected
                detected_modes = {mode: 1.0/len(self.modes) for mode in self.modes}
            
            # Log significant mode detections
            dominant_mode = max(detected_modes.items(), key=lambda x: x[1])
            if dominant_mode[1] > self.confidence_threshold:
                self.log_operator_info(
                    f"🎯 Strong {dominant_mode[0]} mode detected",
                    confidence=f"{dominant_mode[1]:.1%}",
                    regime=context.get('regime', 'unknown')
                )
            
            return detected_modes
            
        except Exception as e:
            self.log_operator_warning(f"Mode detection failed: {e}")
            return {mode: 1.0/len(self.modes) for mode in self.modes}

    def _detect_trending_condition(self, market_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect trending market conditions"""
        
        confidence = 0.0
        
        try:
            # Check regime
            if context.get('regime') == 'trending':
                confidence += 0.4
            
            # Check price momentum
            price_momentum = market_data.get('price_momentum', 0)
            if abs(price_momentum) > 0.1:  # Significant price movement
                confidence += 0.3
            
            # Check trend strength
            trend_strength = market_data.get('price_trend_strength', 0)
            if trend_strength > 2.0:  # Strong trend signal
                confidence += 0.3
            
            # Check recent performance in trending conditions
            if hasattr(self, 'mode_analytics') and 'trending' in self.mode_analytics['mode_win_rates']:
                if self.mode_analytics['mode_win_rates']['trending'] > 0.6:
                    confidence += 0.2
            
        except Exception:
            pass
        
        return min(1.0, confidence)

    def _detect_ranging_condition(self, market_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect ranging market conditions"""
        
        confidence = 0.0
        
        try:
            # Check regime
            if context.get('regime') == 'ranging':
                confidence += 0.4
            
            # Check price volatility (low volatility suggests ranging)
            price_volatility = market_data.get('price_volatility', 0)
            if price_volatility < 0.05:  # Low volatility
                confidence += 0.3
            
            # Check momentum (low momentum suggests ranging)
            price_momentum = abs(market_data.get('price_momentum', 0))
            if price_momentum < 0.02:  # Low momentum
                confidence += 0.3
            
        except Exception:
            pass
        
        return min(1.0, confidence)

    def _detect_volatile_condition(self, market_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect volatile market conditions"""
        
        confidence = 0.0
        
        try:
            # Check volatility level
            volatility_score = market_data.get('volatility_score', 0.5)
            if volatility_score > 0.7:
                confidence += 0.5
            
            # Check price volatility
            price_volatility = market_data.get('price_volatility', 0)
            if price_volatility > 0.1:  # High price volatility
                confidence += 0.3
            
            # Check trade frequency (high frequency in volatile markets)
            trade_frequency = market_data.get('trade_frequency', 0)
            if trade_frequency > 2.0:  # High trade frequency
                confidence += 0.2
            
        except Exception:
            pass
        
        return min(1.0, confidence)

    def _detect_breakout_condition(self, market_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect breakout market conditions"""
        
        confidence = 0.0
        
        try:
            # Check for strong momentum with low previous volatility
            price_momentum = abs(market_data.get('price_momentum', 0))
            trend_strength = market_data.get('price_trend_strength', 0)
            
            if price_momentum > 0.08 and trend_strength > 1.5:
                confidence += 0.4
            
            # Check volatility expansion
            volatility_score = market_data.get('volatility_score', 0.5)
            if volatility_score > 0.6:
                confidence += 0.3
            
            # Check recent performance pattern
            recent_performance = market_data.get('recent_performance_trend', 0)
            if recent_performance > 20:  # Recent good performance
                confidence += 0.3
            
        except Exception:
            pass
        
        return min(1.0, confidence)

    def _detect_reversal_condition(self, market_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect reversal market conditions"""
        
        confidence = 0.0
        
        try:
            # Check for momentum divergence or exhaustion
            price_momentum = market_data.get('price_momentum', 0)
            performance_trend = market_data.get('recent_performance_trend', 0)
            
            # Negative performance despite price movement suggests reversal
            if abs(price_momentum) > 0.05 and performance_trend < -10:
                confidence += 0.4
            
            # Check consistency (low consistency might indicate reversal)
            consistency = market_data.get('performance_consistency', 0.5)
            if consistency < 0.3:
                confidence += 0.3
            
            # Check if previous trend was strong (reversal opportunity)
            trend_strength = market_data.get('price_trend_strength', 0)
            if trend_strength > 2.5:  # Strong previous trend
                confidence += 0.3
            
        except Exception:
            pass
        
        return min(1.0, confidence)

    def _update_mode_performance_from_info_bus(self, info_bus: InfoBus, detected_modes: Dict[str, float]) -> None:
        """Update mode performance based on recent trading results"""
        
        try:
            recent_trades = info_bus.get('recent_trades', [])
            
            if not recent_trades:
                return
            
            # Get the most recent trade result
            last_trade = recent_trades[-1]
            pnl = last_trade.get('pnl', 0)
            
            # Determine which mode(s) were active during this trade
            # Use the detected modes as a proxy for what was active
            for mode, confidence in detected_modes.items():
                if confidence > 0.1:  # Only update modes with significant confidence
                    self._record_mode_result(mode, pnl, confidence)
            
        except Exception as e:
            self.log_operator_warning(f"Mode performance update failed: {e}")

    def record_result(self, mode: str, pnl: float, confidence: float = 1.0) -> None:
        """Record mode result with enhanced validation and analytics"""
        
        try:
            # Validate inputs
            if not isinstance(mode, str):
                self.log_operator_warning(f"Invalid mode type: {type(mode)}")
                return
            
            if np.isnan(pnl):
                self.log_operator_warning(f"NaN PnL for mode {mode}, ignoring")
                return
            
            if mode not in self.modes:
                self.log_operator_warning(f"Unknown mode '{mode}', adding to tracking")
                self.modes.append(mode)
                self.current_mode_weights[mode] = 1.0 / len(self.modes)
                self.mode_definitions[mode] = {'description': f'Dynamic mode: {mode}'}
            
            self._record_mode_result(mode, pnl, confidence)
            
        except Exception as e:
            self.log_operator_error(f"Mode result recording failed: {e}")

    def _record_mode_result(self, mode: str, pnl: float, confidence: float) -> None:
        """Internal method to record mode result with analytics"""
        
        # Update basic tracking
        self.mode_performance[mode]['pnl'].append(pnl)
        self.mode_performance[mode]['confidence'].append(confidence)
        self.mode_counts[mode] += 1
        
        # Record in history with timestamp
        mode_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'mode': mode,
            'pnl': pnl,
            'confidence': confidence
        }
        self.mode_history.append(mode_record)
        
        # Update analytics
        self._update_mode_analytics(mode, pnl, confidence)
        
        # Log significant results
        if abs(pnl) > 25 or confidence > 0.8:
            self.log_operator_info(
                f"🎯 {mode.title()} mode result",
                pnl=f"€{pnl:+.2f}",
                confidence=f"{confidence:.1%}",
                total_count=self.mode_counts[mode]
            )

    def _update_mode_analytics(self, mode: str, pnl: float, confidence: float) -> None:
        """Update comprehensive mode analytics"""
        
        try:
            mode_pnls = self.mode_performance[mode]['pnl']
            mode_confidences = self.mode_performance[mode]['confidence']
            
            # Update profitability
            if mode_pnls:
                self.mode_analytics['mode_profitability'][mode] = sum(mode_pnls)
                
                # Update win rate
                wins = len([p for p in mode_pnls if p > 0])
                self.mode_analytics['mode_win_rates'][mode] = wins / len(mode_pnls)
                
                # Update confidence score (weighted average)
                if mode_confidences:
                    weights = np.array(mode_confidences)
                    weighted_pnls = np.array(mode_pnls) * weights
                    self.mode_analytics['mode_confidence_scores'][mode] = np.sum(weighted_pnls) / np.sum(weights)
            
        except Exception as e:
            self.log_operator_warning(f"Mode analytics update failed: {e}")

    def _adapt_mode_weights(self, detected_modes: Dict[str, float], market_data: Dict[str, Any]) -> None:
        """Adapt mode weights based on performance and market conditions"""
        
        try:
            # Calculate new weights based on recent performance
            new_weights = {}
            
            for mode in self.modes:
                base_weight = 1.0 / len(self.modes)  # Equal baseline
                
                # Performance adjustment
                performance_adj = self._calculate_performance_adjustment(mode)
                
                # Market condition adjustment
                market_adj = detected_modes.get(mode, 0.0)
                
                # Confidence adjustment
                confidence_adj = self.mode_analytics['mode_confidence_scores'].get(mode, 0.5)
                
                # Combine adjustments
                combined_weight = base_weight * (
                    0.4 * (1.0 + performance_adj) +  # Performance component
                    0.4 * market_adj +               # Market detection component
                    0.2 * confidence_adj             # Confidence component
                )
                
                new_weights[mode] = max(0.05, combined_weight)  # Minimum weight threshold
            
            # Normalize weights
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {k: v/total_weight for k, v in new_weights.items()}
            
            # Apply adaptation rate (smooth transition)
            for mode in self.modes:
                old_weight = self.current_mode_weights.get(mode, 1.0/len(self.modes))
                new_weight = new_weights.get(mode, 1.0/len(self.modes))
                
                self.current_mode_weights[mode] = (
                    old_weight * (1 - self.adaptation_rate) + 
                    new_weight * self.adaptation_rate
                )
            
            # Log significant weight changes
            self._log_weight_changes(new_weights)
            
        except Exception as e:
            self.log_operator_warning(f"Mode weight adaptation failed: {e}")

    def _calculate_performance_adjustment(self, mode: str) -> float:
        """Calculate performance-based adjustment for mode weight"""
        
        try:
            mode_pnls = self.mode_performance[mode]['pnl']
            if not mode_pnls or len(mode_pnls) < 3:
                return 0.0  # No adjustment for insufficient data
            
            # Recent performance (last 10 trades)
            recent_pnls = mode_pnls[-10:]
            recent_performance = sum(recent_pnls)
            
            # Overall performance
            total_performance = sum(mode_pnls)
            
            # Win rate component
            win_rate = self.mode_analytics['mode_win_rates'].get(mode, 0.5)
            
            # Calculate adjustment (-1.0 to +1.0 range)
            if recent_performance > 100:
                performance_adj = 0.5 + min(0.5, recent_performance / 500)
            elif recent_performance < -50:
                performance_adj = -0.5 + max(-0.5, recent_performance / 200)
            else:
                performance_adj = recent_performance / 200
            
            # Adjust based on win rate
            win_rate_adj = (win_rate - 0.5) * 0.5  # -0.25 to +0.25 range
            
            return np.clip(performance_adj + win_rate_adj, -0.8, 0.8)
            
        except Exception:
            return 0.0

    def _log_weight_changes(self, new_weights: Dict[str, float]) -> None:
        """Log significant mode weight changes"""
        
        try:
            significant_changes = []
            
            for mode, new_weight in new_weights.items():
                old_weight = self.current_mode_weights.get(mode, 1.0/len(self.modes))
                change = abs(new_weight - old_weight)
                
                if change > 0.1:  # 10% change threshold
                    direction = "↗️" if new_weight > old_weight else "↘️"
                    significant_changes.append(f"{mode}: {old_weight:.1%} → {new_weight:.1%} {direction}")
            
            if significant_changes:
                self.log_operator_info(
                    "⚖️ Significant mode weight changes",
                    changes="; ".join(significant_changes[:3])  # Show top 3 changes
                )
                
        except Exception as e:
            self.log_operator_warning(f"Weight change logging failed: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Get mode weights and performance metrics for observation"""
        
        try:
            if not self.mode_performance:
                # Return equal distribution for cold start
                num_modes = len(self.modes)
                defaults = np.full(num_modes, 1.0/num_modes, dtype=np.float32)
                self.log_operator_debug("Using default mode weights")
                return defaults
            
            # Get current mode weights
            weights = []
            for mode in self.modes:
                weight = self.current_mode_weights.get(mode, 1.0/len(self.modes))
                weights.append(weight)
            
            # Convert to numpy array
            observation = np.array(weights, dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.log_operator_error(f"Invalid mode observation: {observation}")
                observation = np.nan_to_num(observation, nan=1.0/len(self.modes))
            
            # Ensure weights sum to 1
            weight_sum = observation.sum()
            if weight_sum > 0:
                observation = observation / weight_sum
            else:
                observation = np.full(len(self.modes), 1.0/len(self.modes), dtype=np.float32)
            
            self.log_operator_debug(f"Mode weights: {dict(zip(self.modes, observation))}")
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Mode observation generation failed: {e}")
            return np.full(len(self.modes), 1.0/len(self.modes), dtype=np.float32)

    def get_mode_recommendations(self) -> Dict[str, Any]:
        """Get current mode recommendations based on analysis"""
        
        try:
            # Find best performing mode
            best_mode = None
            best_performance = float('-inf')
            
            for mode in self.modes:
                if mode in self.mode_analytics['mode_profitability']:
                    performance = self.mode_analytics['mode_profitability'][mode]
                    if performance > best_performance:
                        best_performance = performance
                        best_mode = mode
            
            # Find most confident mode
            most_confident_mode = max(self.current_mode_weights.items(), key=lambda x: x[1])
            
            # Generate recommendations
            recommendations = {
                'primary_mode': most_confident_mode[0],
                'primary_weight': most_confident_mode[1],
                'best_performing_mode': best_mode,
                'best_performance': best_performance,
                'mode_weights': self.current_mode_weights.copy(),
                'mode_analytics': {
                    'total_modes_tracked': len(self.modes),
                    'modes_with_data': len([m for m in self.modes if self.mode_counts.get(m, 0) > 0]),
                    'most_used_mode': max(self.mode_counts.items(), key=lambda x: x[1])[0] if self.mode_counts else None
                }
            }
            
            return recommendations
            
        except Exception as e:
            self.log_operator_warning(f"Mode recommendations generation failed: {e}")
            return {'primary_mode': self.modes[0], 'primary_weight': 1.0/len(self.modes)}

    def _update_info_bus_with_mode_data(self, info_bus: InfoBus) -> None:
        """Update InfoBus with mode analysis results"""
        
        try:
            # Prepare mode data
            mode_data = {
                'current_weights': self.current_mode_weights.copy(),
                'mode_analytics': self.mode_analytics.copy(),
                'mode_performance': {k: {'pnl': list(v['pnl']), 'confidence': list(v['confidence'])} 
                                   for k, v in self.mode_performance.items()},
                'mode_definitions': self.mode_definitions.copy(),
                'recommendations': self.get_mode_recommendations(),
                'adaptation_rate': self.adaptation_rate,
                'confidence_threshold': self.confidence_threshold
            }
            
            # Add to InfoBus
            InfoBusUpdater.add_module_data(info_bus, 'opponent_mode_enhancer', mode_data)
            
            # Add alerts for significant mode changes
            recommendations = mode_data['recommendations']
            if recommendations['primary_weight'] > 0.6:  # Strong mode confidence
                InfoBusUpdater.add_alert(
                    info_bus,
                    f"Strong {recommendations['primary_mode']} mode detected",
                    'opponent_mode_enhancer',
                    'info',
                    {'mode': recommendations['primary_mode'], 'confidence': recommendations['primary_weight']}
                )
            
        except Exception as e:
            self.log_operator_warning(f"InfoBus mode update failed: {e}")

    def get_mode_report(self) -> str:
        """Generate operator-friendly mode analysis report"""
        
        recommendations = self.get_mode_recommendations()
        
        # Mode performance summary
        performance_summary = ""
        for mode in self.modes:
            count = self.mode_counts.get(mode, 0)
            if count > 0:
                total_pnl = self.mode_analytics['mode_profitability'].get(mode, 0)
                win_rate = self.mode_analytics['mode_win_rates'].get(mode, 0)
                weight = self.current_mode_weights.get(mode, 0)
                
                status = "🟢" if total_pnl > 0 else "🔴" if total_pnl < -25 else "🟡"
                performance_summary += f"  • {mode.title()}: {weight:.1%} weight, €{total_pnl:+.0f} P&L, {win_rate:.1%} win rate {status}\n"
            else:
                weight = self.current_mode_weights.get(mode, 0)
                performance_summary += f"  • {mode.title()}: {weight:.1%} weight, No data yet ⚪\n"
        
        # Recent activity
        recent_activity = ""
        if self.mode_history:
            for record in list(self.mode_history)[-3:]:
                timestamp = record['timestamp'][:19].replace('T', ' ')
                mode = record['mode']
                pnl = record['pnl']
                confidence = record['confidence']
                recent_activity += f"  • {timestamp}: {mode} mode, €{pnl:+.0f}, {confidence:.1%} confidence\n"
        
        return f"""
🎯 OPPONENT MODE ENHANCER REPORT
═══════════════════════════════════════
🏆 Current Primary Mode: {recommendations['primary_mode'].title()} ({recommendations['primary_weight']:.1%})
📊 Best Performing Mode: {recommendations.get('best_performing_mode', 'N/A')} (€{recommendations.get('best_performance', 0):+.0f})
⚙️ Adaptation Rate: {self.adaptation_rate:.1%}
🎚️ Confidence Threshold: {self.confidence_threshold:.1%}

📈 Mode Performance:
{performance_summary}

🔄 Recent Activity:
{recent_activity if recent_activity else '  📭 No recent mode activity'}

📊 Analytics Summary:
• Total Modes: {len(self.modes)}
• Modes with Data: {len([m for m in self.modes if self.mode_counts.get(m, 0) > 0])}
• Total Records: {len(self.mode_history)}
• Mode Switches: {self.mode_analytics.get('switches_since_reset', 0)}

🎯 Mode Definitions:
{chr(10).join([f'  • {mode.title()}: {defn.get("description", "No description")}' for mode, defn in self.mode_definitions.items()])}
        """

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "modes": self.modes.copy(),
                "debug": self.debug,
                "adaptation_rate": self.adaptation_rate,
                "confidence_threshold": self.confidence_threshold,
                "mode_switch_cooldown": self.mode_switch_cooldown
            },
            "mode_state": {
                "current_mode_weights": self.current_mode_weights.copy(),
                "mode_performance": {k: {'pnl': list(v['pnl']), 'confidence': list(v['confidence'])} 
                                   for k, v in self.mode_performance.items()},
                "mode_counts": dict(self.mode_counts),
                "mode_history": list(self.mode_history),
                "mode_analytics": self.mode_analytics.copy()
            },
            "mode_definitions": self.mode_definitions.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.modes = config.get("modes", self.modes)
        self.debug = bool(config.get("debug", self.debug))
        self.adaptation_rate = float(config.get("adaptation_rate", self.adaptation_rate))
        self.confidence_threshold = float(config.get("confidence_threshold", self.confidence_threshold))
        self.mode_switch_cooldown = int(config.get("mode_switch_cooldown", self.mode_switch_cooldown))
        
        # Load mode state
        mode_state = state.get("mode_state", {})
        self.current_mode_weights = mode_state.get("current_mode_weights", {mode: 1.0/len(self.modes) for mode in self.modes})
        
        # Restore performance data
        performance_data = mode_state.get("mode_performance", {})
        self.mode_performance = defaultdict(lambda: defaultdict(list))
        for mode, data in performance_data.items():
            self.mode_performance[mode]['pnl'] = list(data.get('pnl', []))
            self.mode_performance[mode]['confidence'] = list(data.get('confidence', []))
        
        # Restore other state
        self.mode_counts = defaultdict(int, mode_state.get("mode_counts", {}))
        self.mode_history = deque(mode_state.get("mode_history", []), maxlen=100)
        self.mode_analytics = mode_state.get("mode_analytics", self.mode_analytics)
        
        # Load mode definitions if provided
        self.mode_definitions.update(state.get("mode_definitions", {}))