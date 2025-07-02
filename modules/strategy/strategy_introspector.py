# ─────────────────────────────────────────────────────────────
# File: modules/strategy/strategy_introspector.py
# Enhanced with InfoBus integration & intelligent strategy analysis
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class StrategyIntrospector(Module, AnalysisMixin, StateManagementMixin, TradingMixin):
    """
    Enhanced strategy introspector with InfoBus integration.
    Analyzes strategy performance patterns and provides deep insights into trading behavior.
    Tracks strategy evolution, performance metrics, and adaptation patterns.
    """

    def __init__(
        self,
        history_len: int = 10,
        debug: bool = False,
        analysis_depth: str = "comprehensive",  # "basic", "detailed", "comprehensive"
        performance_window: int = 20,
        adaptation_threshold: float = 0.1,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=history_len * 2,
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.history_len = int(history_len)
        self.debug = bool(debug)
        self.analysis_depth = analysis_depth
        self.performance_window = int(performance_window)
        self.adaptation_threshold = float(adaptation_threshold)
        
        # Strategy analysis state
        self._records = deque(maxlen=self.history_len)
        self.strategy_profiles = defaultdict(lambda: self._create_empty_profile())
        self.performance_analytics = defaultdict(list)
        self.adaptation_history = deque(maxlen=50)
        
        # Enhanced baselines for robust operation
        self._baseline_metrics = {
            'win_rate': 0.5,      # 50% baseline win rate
            'stop_loss': 1.0,     # 1% baseline stop loss
            'take_profit': 1.5,   # 1.5% baseline take profit
            'risk_reward': 1.5,   # 1.5:1 baseline risk-reward
            'avg_duration': 30,   # 30 step baseline duration
            'volatility_adj': 1.0 # No volatility adjustment baseline
        }
        
        # Strategy categorization
        self.strategy_categories = {
            'conservative': {'risk_threshold': 0.8, 'return_threshold': 1.2},
            'balanced': {'risk_threshold': 1.2, 'return_threshold': 1.8},
            'aggressive': {'risk_threshold': 2.0, 'return_threshold': 3.0},
            'scalping': {'duration_threshold': 10, 'frequency_threshold': 5},
            'swing': {'duration_threshold': 100, 'frequency_threshold': 1}
        }
        
        # Performance tracking metrics
        self.introspection_metrics = {
            'total_strategies_analyzed': 0,
            'significant_adaptations': 0,
            'performance_improvements': 0,
            'performance_degradations': 0,
            'last_major_insight': None,
            'analysis_accuracy': 0.0,
            'prediction_success_rate': 0.0
        }
        
        # Real-time analysis state
        self.current_analysis = {
            'dominant_strategy_type': 'balanced',
            'performance_trend': 'stable',
            'adaptation_needed': False,
            'recommended_adjustments': [],
            'confidence_level': 0.5,
            'analysis_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "StrategyIntrospector",
            "logs/strategy/strategy_introspector.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("StrategyIntrospector")
        
        self.log_operator_info(
            "🔍 Strategy Introspector initialized",
            history_len=self.history_len,
            analysis_depth=self.analysis_depth,
            performance_window=self.performance_window
        )

    def _create_empty_profile(self) -> Dict[str, Any]:
        """Create empty strategy profile with all required fields"""
        return {
            'win_rate': [],
            'stop_loss': [],
            'take_profit': [],
            'risk_reward': [],
            'duration': [],
            'volatility_adjustment': [],
            'pnl_history': [],
            'trade_count': 0,
            'last_updated': datetime.datetime.now().isoformat(),
            'performance_score': 0.0,
            'consistency_score': 0.0,
            'adaptation_score': 0.0
        }

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Clear strategy analysis
        self._records.clear()
        self.strategy_profiles.clear()
        self.performance_analytics.clear()
        self.adaptation_history.clear()
        
        # Reset metrics
        self.introspection_metrics = {
            'total_strategies_analyzed': 0,
            'significant_adaptations': 0,
            'performance_improvements': 0,
            'performance_degradations': 0,
            'last_major_insight': None,
            'analysis_accuracy': 0.0,
            'prediction_success_rate': 0.0
        }
        
        # Reset current analysis
        self.current_analysis = {
            'dominant_strategy_type': 'balanced',
            'performance_trend': 'stable',
            'adaptation_needed': False,
            'recommended_adjustments': [],
            'confidence_level': 0.5,
            'analysis_timestamp': datetime.datetime.now().isoformat()
        }
        
        self.log_operator_info("🔄 Strategy Introspector reset - all analysis data cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and real-time analysis"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - limited introspection capability")
            return
        
        # Extract context and strategy data
        context = extract_standard_context(info_bus)
        strategy_context = self._extract_strategy_context_from_info_bus(info_bus, context)
        
        # Perform real-time strategy analysis
        self._perform_real_time_analysis(strategy_context, context)
        
        # Update adaptation recommendations
        self._update_adaptation_recommendations(strategy_context)
        
        # Update InfoBus with introspection results
        self._update_info_bus_with_introspection_data(info_bus)

    def _extract_strategy_context_from_info_bus(self, info_bus: InfoBus, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract strategy context from InfoBus for analysis"""
        
        try:
            # Get recent trading activity
            recent_trades = info_bus.get('recent_trades', [])
            
            # Get strategy information from other modules
            module_data = info_bus.get('module_data', {})
            
            # Extract strategy arbiter data
            strategy_data = module_data.get('strategy_arbiter', {})
            active_strategies = strategy_data.get('active_strategies', [])
            strategy_weights = strategy_data.get('strategy_weights', {})
            
            # Extract genome pool data
            genome_data = module_data.get('strategy_genome_pool', {})
            active_genome = genome_data.get('active_genome', None)
            best_genome = genome_data.get('best_genome', None)
            
            # Extract risk management data
            risk_data = info_bus.get('risk', {})
            
            strategy_context = {
                'timestamp': datetime.datetime.now().isoformat(),
                'recent_trades': recent_trades,
                'active_strategies': active_strategies,
                'strategy_weights': strategy_weights,
                'active_genome': active_genome,
                'best_genome': best_genome,
                'current_balance': risk_data.get('balance', 0),
                'current_drawdown': risk_data.get('current_drawdown', 0),
                'market_regime': context.get('regime', 'unknown'),
                'volatility_level': context.get('volatility_level', 'medium'),
                'session_pnl': context.get('session_pnl', 0),
                'trade_frequency': self._calculate_trade_frequency(recent_trades)
            }
            
            return strategy_context
            
        except Exception as e:
            self.log_operator_warning(f"Strategy context extraction failed: {e}")
            return {'timestamp': datetime.datetime.now().isoformat()}

    def _calculate_trade_frequency(self, recent_trades: List[Dict]) -> float:
        """Calculate recent trade frequency"""
        
        try:
            if len(recent_trades) < 2:
                return 0.0
            
            # Simple frequency calculation (trades per hour equivalent)
            return min(10.0, len(recent_trades) / 2.0)  # Normalize to reasonable range
            
        except Exception:
            return 0.0

    def _perform_real_time_analysis(self, strategy_context: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Perform real-time strategy analysis"""
        
        try:
            # Analyze current strategy performance
            performance_metrics = self._analyze_current_performance(strategy_context)
            
            # Determine dominant strategy type
            dominant_type = self._classify_strategy_type(strategy_context, performance_metrics)
            
            # Assess performance trend
            performance_trend = self._assess_performance_trend(strategy_context)
            
            # Check adaptation needs
            adaptation_needed = self._assess_adaptation_needs(performance_metrics)
            
            # Generate recommendations
            recommendations = self._generate_strategy_recommendations(
                strategy_context, performance_metrics, dominant_type
            )
            
            # Calculate confidence level
            confidence = self._calculate_analysis_confidence(strategy_context, performance_metrics)
            
            # Update current analysis
            self.current_analysis.update({
                'dominant_strategy_type': dominant_type,
                'performance_trend': performance_trend,
                'adaptation_needed': adaptation_needed,
                'recommended_adjustments': recommendations,
                'confidence_level': confidence,
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                'performance_metrics': performance_metrics
            })
            
            # Log significant insights
            if adaptation_needed or confidence > 0.8:
                self.log_operator_info(
                    f"🔍 Strategy analysis update",
                    type=dominant_type,
                    trend=performance_trend,
                    confidence=f"{confidence:.1%}",
                    adaptation_needed="✅" if adaptation_needed else "❌"
                )
            
        except Exception as e:
            self.log_operator_warning(f"Real-time strategy analysis failed: {e}")

    def _analyze_current_performance(self, strategy_context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze current strategy performance metrics"""
        
        metrics = {}
        
        try:
            recent_trades = strategy_context.get('recent_trades', [])
            
            if recent_trades and len(recent_trades) >= 3:
                # Calculate performance metrics
                pnls = [t.get('pnl', 0) for t in recent_trades]
                durations = [t.get('duration', 30) for t in recent_trades if 'duration' in t]
                
                # Win rate
                wins = len([p for p in pnls if p > 0])
                metrics['win_rate'] = wins / len(pnls)
                
                # Average P&L
                metrics['avg_pnl'] = np.mean(pnls)
                
                # Risk-adjusted return
                if len(pnls) > 1:
                    pnl_std = np.std(pnls)
                    metrics['sharpe_ratio'] = metrics['avg_pnl'] / (pnl_std + 1e-6)
                else:
                    metrics['sharpe_ratio'] = 0.0
                
                # Trade duration analysis
                if durations:
                    metrics['avg_duration'] = np.mean(durations)
                    metrics['duration_consistency'] = 1.0 - (np.std(durations) / (np.mean(durations) + 1e-6))
                else:
                    metrics['avg_duration'] = 30.0
                    metrics['duration_consistency'] = 0.5
                
                # Profit factor
                profits = [p for p in pnls if p > 0]
                losses = [p for p in pnls if p < 0]
                if profits and losses:
                    metrics['profit_factor'] = sum(profits) / abs(sum(losses))
                else:
                    metrics['profit_factor'] = 1.0 if profits else 0.0
                
                # Drawdown analysis
                cumulative_pnl = np.cumsum(pnls)
                peak = cumulative_pnl[0]
                max_drawdown = 0.0
                for value in cumulative_pnl:
                    peak = max(peak, value)
                    drawdown = (peak - value) / abs(peak) if peak != 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                
                metrics['max_drawdown'] = max_drawdown
                
                # Consistency score
                if len(pnls) >= 5:
                    positive_streaks = self._calculate_positive_streaks(pnls)
                    metrics['consistency_score'] = min(1.0, len(positive_streaks) / (len(pnls) / 3))
                else:
                    metrics['consistency_score'] = 0.5
                    
            else:
                # Use baseline metrics when insufficient data
                metrics = {
                    'win_rate': self._baseline_metrics['win_rate'],
                    'avg_pnl': 0.0,
                    'sharpe_ratio': 0.0,
                    'avg_duration': self._baseline_metrics['avg_duration'],
                    'duration_consistency': 0.5,
                    'profit_factor': 1.0,
                    'max_drawdown': 0.0,
                    'consistency_score': 0.5
                }
            
            # Add contextual metrics
            metrics['trade_frequency'] = strategy_context.get('trade_frequency', 0.0)
            metrics['current_drawdown'] = strategy_context.get('current_drawdown', 0.0)
            metrics['session_pnl'] = strategy_context.get('session_pnl', 0.0)
            
            return metrics
            
        except Exception as e:
            self.log_operator_warning(f"Performance analysis failed: {e}")
            return {k: v for k, v in self._baseline_metrics.items()}

    def _calculate_positive_streaks(self, pnls: List[float]) -> List[int]:
        """Calculate positive P&L streaks"""
        
        streaks = []
        current_streak = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return streaks

    def _classify_strategy_type(self, strategy_context: Dict[str, Any], performance_metrics: Dict[str, float]) -> str:
        """Classify the current strategy type based on behavior patterns"""
        
        try:
            avg_duration = performance_metrics.get('avg_duration', 30)
            trade_frequency = performance_metrics.get('trade_frequency', 0)
            win_rate = performance_metrics.get('win_rate', 0.5)
            profit_factor = performance_metrics.get('profit_factor', 1.0)
            max_drawdown = performance_metrics.get('max_drawdown', 0.0)
            
            # Active genome analysis
            active_genome = strategy_context.get('active_genome', [])
            if active_genome and len(active_genome) >= 4:
                sl_ratio = active_genome[0]  # Stop loss ratio
                tp_ratio = active_genome[1]  # Take profit ratio
                risk_reward = tp_ratio / sl_ratio if sl_ratio > 0 else 1.5
            else:
                risk_reward = profit_factor
            
            # Classification logic
            if avg_duration < 15 and trade_frequency > 3:
                return 'scalping'
            elif avg_duration > 60 and trade_frequency < 2:
                return 'swing'
            elif max_drawdown < 0.02 and risk_reward < 1.2:
                return 'conservative'
            elif max_drawdown > 0.05 or risk_reward > 2.5:
                return 'aggressive'
            else:
                return 'balanced'
                
        except Exception as e:
            self.log_operator_warning(f"Strategy classification failed: {e}")
            return 'balanced'

    def _assess_performance_trend(self, strategy_context: Dict[str, Any]) -> str:
        """Assess the current performance trend"""
        
        try:
            recent_trades = strategy_context.get('recent_trades', [])
            
            if len(recent_trades) < 5:
                return 'insufficient_data'
            
            # Analyze recent vs older performance
            recent_pnls = [t.get('pnl', 0) for t in recent_trades[-5:]]
            older_pnls = [t.get('pnl', 0) for t in recent_trades[-10:-5]] if len(recent_trades) >= 10 else recent_pnls
            
            recent_avg = np.mean(recent_pnls)
            older_avg = np.mean(older_pnls)
            
            # Calculate trend strength
            trend_strength = abs(recent_avg - older_avg)
            
            if trend_strength < 5:  # Small difference
                return 'stable'
            elif recent_avg > older_avg:
                return 'improving' if trend_strength > 15 else 'slightly_improving'
            else:
                return 'declining' if trend_strength > 15 else 'slightly_declining'
                
        except Exception:
            return 'stable'

    def _assess_adaptation_needs(self, performance_metrics: Dict[str, float]) -> bool:
        """Assess if strategy adaptation is needed"""
        
        try:
            # Check for poor performance indicators
            win_rate = performance_metrics.get('win_rate', 0.5)
            max_drawdown = performance_metrics.get('max_drawdown', 0.0)
            profit_factor = performance_metrics.get('profit_factor', 1.0)
            consistency_score = performance_metrics.get('consistency_score', 0.5)
            current_drawdown = performance_metrics.get('current_drawdown', 0.0)
            
            # Adaptation criteria
            adaptation_needed = (
                win_rate < 0.4 or  # Low win rate
                max_drawdown > 0.08 or  # High max drawdown
                current_drawdown > 0.05 or  # High current drawdown
                profit_factor < 0.8 or  # Poor profit factor
                consistency_score < 0.3  # Low consistency
            )
            
            return adaptation_needed
            
        except Exception:
            return False

    def _generate_strategy_recommendations(self, strategy_context: Dict[str, Any], 
                                         performance_metrics: Dict[str, float], 
                                         strategy_type: str) -> List[str]:
        """Generate specific strategy recommendations"""
        
        recommendations = []
        
        try:
            win_rate = performance_metrics.get('win_rate', 0.5)
            max_drawdown = performance_metrics.get('max_drawdown', 0.0)
            profit_factor = performance_metrics.get('profit_factor', 1.0)
            avg_duration = performance_metrics.get('avg_duration', 30)
            trade_frequency = performance_metrics.get('trade_frequency', 0)
            
            # Win rate recommendations
            if win_rate < 0.4:
                recommendations.append("Improve entry signal quality - consider stricter filters")
            elif win_rate > 0.7:
                recommendations.append("Consider increasing position size - high win rate indicates good entries")
            
            # Drawdown recommendations
            if max_drawdown > 0.08:
                recommendations.append("Reduce position sizes - drawdown too high for sustainable trading")
            
            if performance_metrics.get('current_drawdown', 0) > 0.05:
                recommendations.append("Consider temporary trading halt - current drawdown is significant")
            
            # Profit factor recommendations
            if profit_factor < 0.8:
                recommendations.append("Review exit strategy - losses are exceeding profits")
            elif profit_factor > 3.0:
                recommendations.append("Consider taking more profits - very high profit factor suggests early exits")
            
            # Strategy-specific recommendations
            if strategy_type == 'scalping':
                if avg_duration > 20:
                    recommendations.append("Reduce holding time for scalping strategy")
                if trade_frequency < 2:
                    recommendations.append("Increase trade frequency for effective scalping")
            elif strategy_type == 'swing':
                if avg_duration < 30:
                    recommendations.append("Allow trades more time to develop for swing strategy")
            elif strategy_type == 'aggressive':
                if max_drawdown > 0.1:
                    recommendations.append("Reduce risk - aggressive strategy causing excessive drawdown")
            
            # Market regime recommendations
            market_regime = strategy_context.get('market_regime', 'unknown')
            if market_regime == 'volatile' and strategy_type != 'conservative':
                recommendations.append("Consider more conservative approach during volatile market conditions")
            
            # Default recommendation if no specific issues
            if not recommendations:
                recommendations.append("Continue current approach - performance metrics are within acceptable ranges")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            self.log_operator_warning(f"Recommendation generation failed: {e}")
            return ["Review strategy performance and consider adjustments based on recent market conditions"]

    def _calculate_analysis_confidence(self, strategy_context: Dict[str, Any], 
                                     performance_metrics: Dict[str, float]) -> float:
        """Calculate confidence level in the analysis"""
        
        try:
            recent_trades = strategy_context.get('recent_trades', [])
            
            # Base confidence on data availability
            data_confidence = min(1.0, len(recent_trades) / 10.0)
            
            # Confidence based on metric consistency
            consistency_score = performance_metrics.get('consistency_score', 0.5)
            
            # Confidence based on analysis history
            if len(self._records) >= 5:
                history_confidence = 0.8
            elif len(self._records) >= 2:
                history_confidence = 0.6
            else:
                history_confidence = 0.4
            
            # Combined confidence
            overall_confidence = (
                0.4 * data_confidence +
                0.3 * consistency_score +
                0.3 * history_confidence
            )
            
            return min(1.0, max(0.1, overall_confidence))
            
        except Exception:
            return 0.5

    def record(self, theme: np.ndarray, win_rate: float, sl: float, tp: float, **kwargs) -> None:
        """Enhanced record method with comprehensive validation and analysis"""
        
        try:
            # Validate core inputs
            if not (0 <= win_rate <= 1):
                self.log_operator_warning(f"Invalid win_rate {win_rate}, clamping to [0,1]")
                win_rate = np.clip(win_rate, 0, 1)
            
            if sl <= 0:
                self.log_operator_warning(f"Invalid sl {sl}, using baseline {self._baseline_metrics['stop_loss']}")
                sl = self._baseline_metrics['stop_loss']
                
            if tp <= 0:
                self.log_operator_warning(f"Invalid tp {tp}, using baseline {self._baseline_metrics['take_profit']}")
                tp = self._baseline_metrics['take_profit']
            
            # Extract additional metrics from kwargs
            duration = kwargs.get('duration', self._baseline_metrics['avg_duration'])
            pnl = kwargs.get('pnl', 0.0)
            market_regime = kwargs.get('market_regime', 'unknown')
            volatility_adj = kwargs.get('volatility_adjustment', 1.0)
            strategy_type = kwargs.get('strategy_type', 'balanced')
            
            # Create comprehensive record
            record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'theme': theme.tolist() if hasattr(theme, 'tolist') else theme,
                'win_rate': win_rate,
                'stop_loss': sl,
                'take_profit': tp,
                'risk_reward_ratio': tp / sl if sl > 0 else 1.5,
                'duration': duration,
                'pnl': pnl,
                'market_regime': market_regime,
                'volatility_adjustment': volatility_adj,
                'strategy_type': strategy_type
            }
            
            # Store record
            self._records.append(record)
            
            # Update strategy profiles
            self._update_strategy_profile(strategy_type, record)
            
            # Update analytics
            self._update_performance_analytics(record)
            
            # Check for significant changes
            self._check_for_adaptations(record)
            
            # Update metrics
            self.introspection_metrics['total_strategies_analyzed'] += 1
            
            self.log_operator_info(
                f"📊 Strategy recorded",
                type=strategy_type,
                win_rate=f"{win_rate:.1%}",
                risk_reward=f"{tp/sl:.2f}:1" if sl > 0 else "Invalid",
                pnl=f"€{pnl:+.2f}",
                total_records=len(self._records)
            )
            
            # Log statistics periodically
            if len(self._records) % 5 == 0:
                self._log_comprehensive_statistics()
                
        except Exception as e:
            self.log_operator_error(f"Strategy recording failed: {e}")

    def _update_strategy_profile(self, strategy_type: str, record: Dict[str, Any]) -> None:
        """Update strategy profile with new record"""
        
        try:
            profile = self.strategy_profiles[strategy_type]
            
            # Update metrics lists
            profile['win_rate'].append(record['win_rate'])
            profile['stop_loss'].append(record['stop_loss'])
            profile['take_profit'].append(record['take_profit'])
            profile['risk_reward'].append(record['risk_reward_ratio'])
            profile['duration'].append(record['duration'])
            profile['volatility_adjustment'].append(record['volatility_adjustment'])
            profile['pnl_history'].append(record['pnl'])
            profile['trade_count'] += 1
            profile['last_updated'] = record['timestamp']
            
            # Keep only recent data within window
            max_len = self.performance_window
            for key in ['win_rate', 'stop_loss', 'take_profit', 'risk_reward', 'duration', 'volatility_adjustment', 'pnl_history']:
                if len(profile[key]) > max_len:
                    profile[key] = profile[key][-max_len:]
            
            # Calculate profile scores
            self._calculate_profile_scores(profile)
            
        except Exception as e:
            self.log_operator_warning(f"Strategy profile update failed: {e}")

    def _calculate_profile_scores(self, profile: Dict[str, Any]) -> None:
        """Calculate performance scores for strategy profile"""
        
        try:
            if not profile['pnl_history']:
                return
            
            # Performance score (average P&L)
            profile['performance_score'] = np.mean(profile['pnl_history'])
            
            # Consistency score (inverse of P&L volatility)
            if len(profile['pnl_history']) > 1:
                pnl_std = np.std(profile['pnl_history'])
                pnl_mean = abs(np.mean(profile['pnl_history']))
                profile['consistency_score'] = 1.0 / (1.0 + pnl_std / (pnl_mean + 1e-6))
            else:
                profile['consistency_score'] = 0.5
            
            # Adaptation score (how much the strategy has changed)
            if len(profile['win_rate']) >= 3:
                recent_metrics = np.array([
                    profile['win_rate'][-1],
                    profile['risk_reward'][-1],
                    profile['duration'][-1] / 100.0  # Normalize duration
                ])
                older_metrics = np.array([
                    np.mean(profile['win_rate'][:-1]),
                    np.mean(profile['risk_reward'][:-1]),
                    np.mean(profile['duration'][:-1]) / 100.0
                ])
                
                adaptation_distance = np.linalg.norm(recent_metrics - older_metrics)
                profile['adaptation_score'] = min(1.0, adaptation_distance)
            else:
                profile['adaptation_score'] = 0.0
                
        except Exception as e:
            self.log_operator_warning(f"Profile score calculation failed: {e}")

    def _update_performance_analytics(self, record: Dict[str, Any]) -> None:
        """Update comprehensive performance analytics"""
        
        try:
            timestamp = record['timestamp']
            
            # Track performance by market regime
            regime = record.get('market_regime', 'unknown')
            self.performance_analytics[f'regime_{regime}'].append(record['pnl'])
            
            # Track performance by strategy type
            strategy_type = record.get('strategy_type', 'balanced')
            self.performance_analytics[f'strategy_{strategy_type}'].append(record['pnl'])
            
            # Track risk-reward evolution
            self.performance_analytics['risk_reward_evolution'].append({
                'timestamp': timestamp,
                'ratio': record['risk_reward_ratio'],
                'pnl': record['pnl']
            })
            
            # Track win rate evolution
            self.performance_analytics['win_rate_evolution'].append({
                'timestamp': timestamp,
                'win_rate': record['win_rate'],
                'pnl': record['pnl']
            })
            
            # Limit analytics history
            max_analytics_len = 100
            for key in self.performance_analytics:
                if len(self.performance_analytics[key]) > max_analytics_len:
                    self.performance_analytics[key] = self.performance_analytics[key][-max_analytics_len:]
                    
        except Exception as e:
            self.log_operator_warning(f"Performance analytics update failed: {e}")

    def _check_for_adaptations(self, record: Dict[str, Any]) -> None:
        """Check for significant strategy adaptations"""
        
        try:
            if len(self._records) < 2:
                return
            
            current = record
            previous = self._records[-2]
            
            # Check for significant changes
            adaptations_detected = []
            
            # Win rate adaptation
            wr_change = abs(current['win_rate'] - previous['win_rate'])
            if wr_change > self.adaptation_threshold:
                adaptations_detected.append(f"Win rate: {previous['win_rate']:.1%} → {current['win_rate']:.1%}")
            
            # Risk-reward adaptation
            rr_change = abs(current['risk_reward_ratio'] - previous['risk_reward_ratio'])
            if rr_change > self.adaptation_threshold:
                adaptations_detected.append(f"Risk-reward: {previous['risk_reward_ratio']:.2f} → {current['risk_reward_ratio']:.2f}")
            
            # Duration adaptation
            duration_change = abs(current['duration'] - previous['duration'])
            if duration_change > previous['duration'] * 0.3:  # 30% change
                adaptations_detected.append(f"Duration: {previous['duration']:.0f} → {current['duration']:.0f}")
            
            if adaptations_detected:
                adaptation_record = {
                    'timestamp': current['timestamp'],
                    'adaptations': adaptations_detected,
                    'strategy_type': current.get('strategy_type', 'unknown'),
                    'market_regime': current.get('market_regime', 'unknown'),
                    'performance_impact': current['pnl']
                }
                
                self.adaptation_history.append(adaptation_record)
                self.introspection_metrics['significant_adaptations'] += 1
                
                self.log_operator_info(
                    f"🔄 Strategy adaptation detected",
                    adaptations="; ".join(adaptations_detected[:2]),
                    performance_impact=f"€{current['pnl']:+.2f}"
                )
                
        except Exception as e:
            self.log_operator_warning(f"Adaptation check failed: {e}")

    def _log_comprehensive_statistics(self) -> None:
        """Log comprehensive strategy statistics"""
        
        try:
            if not self._records:
                return
            
            self.log_operator_info(f"📊 Strategy Statistics - Records: {len(self._records)}")
            
            # Overall statistics
            all_records = list(self._records)
            win_rates = [r['win_rate'] for r in all_records]
            risk_rewards = [r['risk_reward_ratio'] for r in all_records]
            pnls = [r['pnl'] for r in all_records]
            
            self.log_operator_info(
                f"  Overall: Win Rate={np.mean(win_rates):.1%}±{np.std(win_rates):.1%}, "
                f"Risk-Reward={np.mean(risk_rewards):.2f}±{np.std(risk_rewards):.2f}, "
                f"Avg PnL=€{np.mean(pnls):+.2f}"
            )
            
            # Strategy type breakdown
            strategy_types = {}
            for record in all_records:
                strategy_type = record.get('strategy_type', 'unknown')
                if strategy_type not in strategy_types:
                    strategy_types[strategy_type] = []
                strategy_types[strategy_type].append(record['pnl'])
            
            for strategy_type, pnls in strategy_types.items():
                if len(pnls) > 0:
                    profile = self.strategy_profiles.get(strategy_type, {})
                    performance_score = profile.get('performance_score', 0)
                    consistency_score = profile.get('consistency_score', 0)
                    
                    self.log_operator_info(
                        f"  {strategy_type.title()}: {len(pnls)} trades, "
                        f"Performance={performance_score:.2f}, "
                        f"Consistency={consistency_score:.2f}"
                    )
                    
        except Exception as e:
            self.log_operator_warning(f"Statistics logging failed: {e}")

    def _update_adaptation_recommendations(self, strategy_context: Dict[str, Any]) -> None:
        """Update adaptation recommendations based on current analysis"""
        
        try:
            if not self.current_analysis.get('adaptation_needed', False):
                return
            
            # Generate specific adaptation steps
            performance_metrics = self.current_analysis.get('performance_metrics', {})
            strategy_type = self.current_analysis.get('dominant_strategy_type', 'balanced')
            
            adaptation_steps = []
            
            # Based on poor performance metrics
            if performance_metrics.get('win_rate', 0.5) < 0.4:
                adaptation_steps.append({
                    'type': 'entry_criteria',
                    'action': 'tighten_filters',
                    'priority': 'high',
                    'description': 'Tighten entry criteria to improve win rate'
                })
            
            if performance_metrics.get('max_drawdown', 0) > 0.08:
                adaptation_steps.append({
                    'type': 'risk_management',
                    'action': 'reduce_position_size',
                    'priority': 'critical',
                    'description': 'Reduce position sizes to limit drawdown'
                })
            
            if performance_metrics.get('profit_factor', 1.0) < 0.8:
                adaptation_steps.append({
                    'type': 'exit_strategy',
                    'action': 'review_exits',
                    'priority': 'medium',
                    'description': 'Review and optimize exit strategy'
                })
            
            # Store adaptation recommendations
            if adaptation_steps:
                adaptation_record = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'trigger': 'performance_analysis',
                    'strategy_type': strategy_type,
                    'recommended_steps': adaptation_steps,
                    'urgency_level': max([step.get('priority', 'low') for step in adaptation_steps])
                }
                
                self.adaptation_history.append(adaptation_record)
                
                self.log_operator_warning(
                    f"🚨 Adaptation recommended",
                    steps=len(adaptation_steps),
                    urgency=adaptation_record['urgency_level'],
                    strategy=strategy_type
                )
                
        except Exception as e:
            self.log_operator_warning(f"Adaptation recommendation update failed: {e}")

    def profile(self) -> np.ndarray:
        """Get comprehensive strategy profile with enhanced validation"""
        
        try:
            if not self._records:
                # Return enhanced baseline values
                baseline = np.array([
                    self._baseline_metrics['win_rate'],
                    self._baseline_metrics['stop_loss'],
                    self._baseline_metrics['take_profit'],
                    0.0,  # Win rate variance
                    0.0,  # Risk-reward variance
                    0.5,  # Performance score
                    0.5,  # Consistency score
                    0.0   # Adaptation score
                ], dtype=np.float32)
                
                self.log_operator_debug(f"Using enhanced baseline profile: {baseline}")
                return baseline
            
            # Calculate comprehensive profile from records
            records_array = np.array([
                [r['win_rate'], r['stop_loss'], r['take_profit'], 
                 r['risk_reward_ratio'], r['duration'], r['pnl']] 
                for r in self._records
            ], dtype=np.float32)
            
            # Validate array
            if np.any(~np.isfinite(records_array)):
                self.log_operator_error(f"Invalid values in records array")
                records_array = np.nan_to_num(records_array, nan=0.0)
            
            # Calculate enhanced profile components
            mean_vals = records_array.mean(axis=0)
            var_vals = records_array.var(axis=0) if len(records_array) > 1 else np.zeros(6)
            
            # Calculate performance scores
            dominant_strategy = self.current_analysis.get('dominant_strategy_type', 'balanced')
            profile_data = self.strategy_profiles.get(dominant_strategy, self._create_empty_profile())
            
            performance_score = profile_data.get('performance_score', 0.0)
            consistency_score = profile_data.get('consistency_score', 0.5)
            adaptation_score = profile_data.get('adaptation_score', 0.0)
            
            # Combine into comprehensive profile
            profile = np.array([
                mean_vals[0],        # Mean win rate
                mean_vals[1],        # Mean stop loss
                mean_vals[2],        # Mean take profit
                var_vals[0],         # Win rate variance
                var_vals[3],         # Risk-reward variance
                performance_score,   # Performance score
                consistency_score,   # Consistency score
                adaptation_score     # Adaptation score
            ], dtype=np.float32)
            
            # Final validation
            if np.any(~np.isfinite(profile)):
                self.log_operator_error(f"Invalid profile generated: {profile}")
                profile = np.nan_to_num(profile, nan=0.5)
            
            self.log_operator_debug(f"Generated profile: WR={profile[0]:.2f}, SL={profile[1]:.2f}, TP={profile[2]:.2f}")
            return profile
            
        except Exception as e:
            self.log_operator_error(f"Profile generation failed: {e}")
            return np.array([
                self._baseline_metrics['win_rate'],
                self._baseline_metrics['stop_loss'],
                self._baseline_metrics['take_profit'],
                0.0, 0.0, 0.5, 0.5, 0.0
            ], dtype=np.float32)

    def get_observation_components(self) -> np.ndarray:
        """Return comprehensive strategy observation components"""
        return self.profile()

    def _update_info_bus_with_introspection_data(self, info_bus: InfoBus) -> None:
        """Update InfoBus with comprehensive introspection results"""
        
        try:
            # Prepare introspection data
            introspection_data = {
                'current_analysis': self.current_analysis.copy(),
                'strategy_profiles': {k: v.copy() for k, v in self.strategy_profiles.items()},
                'introspection_metrics': self.introspection_metrics.copy(),
                'recent_adaptations': list(self.adaptation_history)[-5:],  # Last 5 adaptations
                'total_records': len(self._records),
                'analysis_depth': self.analysis_depth,
                'confidence_level': self.current_analysis.get('confidence_level', 0.5)
            }
            
            # Add to InfoBus
            InfoBusUpdater.add_module_data(info_bus, 'strategy_introspector', introspection_data)
            
            # Add alerts for critical situations
            if self.current_analysis.get('adaptation_needed', False):
                InfoBusUpdater.add_alert(
                    info_bus,
                    "Strategy adaptation recommended",
                    'strategy_introspector',
                    'warning',
                    {
                        'strategy_type': self.current_analysis.get('dominant_strategy_type'),
                        'performance_trend': self.current_analysis.get('performance_trend'),
                        'recommendations_count': len(self.current_analysis.get('recommended_adjustments', []))
                    }
                )
            
            # Add alerts for poor performance
            performance_metrics = self.current_analysis.get('performance_metrics', {})
            if performance_metrics.get('max_drawdown', 0) > 0.1:
                InfoBusUpdater.add_alert(
                    info_bus,
                    f"High drawdown detected: {performance_metrics['max_drawdown']:.1%}",
                    'strategy_introspector',
                    'critical',
                    {'max_drawdown': performance_metrics['max_drawdown']}
                )
            
        except Exception as e:
            self.log_operator_warning(f"InfoBus introspection update failed: {e}")

    def get_introspection_report(self) -> str:
        """Generate comprehensive strategy introspection report"""
        
        # Current analysis summary
        current_analysis = self.current_analysis
        strategy_type = current_analysis.get('dominant_strategy_type', 'unknown')
        performance_trend = current_analysis.get('performance_trend', 'unknown')
        confidence = current_analysis.get('confidence_level', 0.0)
        adaptation_needed = current_analysis.get('adaptation_needed', False)
        
        # Performance metrics
        performance_metrics = current_analysis.get('performance_metrics', {})
        
        # Strategy profiles summary
        profiles_summary = ""
        for strategy, profile in self.strategy_profiles.items():
            if profile['trade_count'] > 0:
                performance_score = profile.get('performance_score', 0)
                consistency_score = profile.get('consistency_score', 0)
                status = "🟢" if performance_score > 10 else "🔴" if performance_score < -10 else "🟡"
                profiles_summary += f"  • {strategy.title()}: {profile['trade_count']} trades, Performance={performance_score:+.1f}, Consistency={consistency_score:.2f} {status}\n"
        
        # Recent adaptations
        recent_adaptations = ""
        if self.adaptation_history:
            for adaptation in list(self.adaptation_history)[-3:]:
                timestamp = adaptation['timestamp'][:19].replace('T', ' ')
                adaptations_list = adaptation.get('adaptations', ['Unknown adaptation'])
                recent_adaptations += f"  • {timestamp}: {'; '.join(adaptations_list[:2])}\n"
        
        # Recommendations
        recommendations = current_analysis.get('recommended_adjustments', [])
        recommendations_str = '\n'.join([f'  • {rec}' for rec in recommendations[:5]])
        
        return f"""
🔍 STRATEGY INTROSPECTOR REPORT
═══════════════════════════════════════
📊 Current Analysis:
• Dominant Strategy: {strategy_type.title()}
• Performance Trend: {performance_trend.replace('_', ' ').title()}
• Analysis Confidence: {confidence:.1%}
• Adaptation Needed: {'✅ Yes' if adaptation_needed else '❌ No'}

📈 Performance Metrics:
• Win Rate: {performance_metrics.get('win_rate', 0):.1%}
• Profit Factor: {performance_metrics.get('profit_factor', 0):.2f}
• Max Drawdown: {performance_metrics.get('max_drawdown', 0):.1%}
• Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}
• Consistency Score: {performance_metrics.get('consistency_score', 0):.2f}

🎯 Strategy Profiles:
{profiles_summary if profiles_summary else '  📭 No strategy profiles available yet'}

🔄 Recent Adaptations:
{recent_adaptations if recent_adaptations else '  📭 No recent adaptations detected'}

💡 Current Recommendations:
{recommendations_str if recommendations_str else '  ✅ No specific recommendations - continue current approach'}

📊 Analytics Summary:
• Total Strategies Analyzed: {self.introspection_metrics['total_strategies_analyzed']}
• Significant Adaptations: {self.introspection_metrics['significant_adaptations']}
• Performance Improvements: {self.introspection_metrics['performance_improvements']}
• Analysis Depth: {self.analysis_depth.title()}
• Records Maintained: {len(self._records)}/{self.history_len}
        """

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "history_len": self.history_len,
                "debug": self.debug,
                "analysis_depth": self.analysis_depth,
                "performance_window": self.performance_window,
                "adaptation_threshold": self.adaptation_threshold
            },
            "introspection_state": {
                "records": list(self._records),
                "strategy_profiles": {k: v.copy() for k, v in self.strategy_profiles.items()},
                "performance_analytics": {k: list(v) for k, v in self.performance_analytics.items()},
                "adaptation_history": list(self.adaptation_history),
                "introspection_metrics": self.introspection_metrics.copy(),
                "current_analysis": self.current_analysis.copy()
            },
            "baselines": self._baseline_metrics.copy(),
            "categories": self.strategy_categories.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.history_len = int(config.get("history_len", self.history_len))
        self.debug = bool(config.get("debug", self.debug))
        self.analysis_depth = config.get("analysis_depth", self.analysis_depth)
        self.performance_window = int(config.get("performance_window", self.performance_window))
        self.adaptation_threshold = float(config.get("adaptation_threshold", self.adaptation_threshold))
        
        # Load introspection state
        introspection_state = state.get("introspection_state", {})
        self._records = deque(introspection_state.get("records", []), maxlen=self.history_len)
        
        # Restore strategy profiles
        profiles_data = introspection_state.get("strategy_profiles", {})
        self.strategy_profiles = defaultdict(lambda: self._create_empty_profile())
        for k, v in profiles_data.items():
            self.strategy_profiles[k] = v
        
        # Restore performance analytics
        analytics_data = introspection_state.get("performance_analytics", {})
        self.performance_analytics = defaultdict(list)
        for k, v in analytics_data.items():
            self.performance_analytics[k] = list(v)
        
        # Restore other state
        self.adaptation_history = deque(introspection_state.get("adaptation_history", []), maxlen=50)
        self.introspection_metrics = introspection_state.get("introspection_metrics", self.introspection_metrics)
        self.current_analysis = introspection_state.get("current_analysis", self.current_analysis)
        
        # Load baselines and categories if provided
        self._baseline_metrics.update(state.get("baselines", {}))
        self.strategy_categories.update(state.get("categories", {}))
        
        self.log_operator_info(
            f"🔄 Strategy introspector state loaded",
            records=len(self._records),
            profiles=len(self.strategy_profiles),
            adaptations=len(self.adaptation_history)
        )