"""
🔍 Enhanced Strategy Introspector with SmartInfoBus Integration v3.0
Advanced strategy analysis system with intelligent pattern recognition and adaptation insights
"""

import asyncio
import time
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

# ═══════════════════════════════════════════════════════════════════
# MODERN SMARTINFOBUS IMPORTS
# ═══════════════════════════════════════════════════════════════════
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="StrategyIntrospector",
    version="3.0.0",
    category="strategy",
    provides=[
        "strategy_analysis", "performance_insights", "adaptation_recommendations",
        "strategy_profiles", "introspection_metrics", "behavior_patterns"
    ],
    requires=[
        "recent_trades", "module_data", "risk_data", "market_regime",
        "volatility_data", "trading_performance", "strategy_weights"
    ],
    description="Advanced strategy analysis system with intelligent pattern recognition and adaptation insights",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=180,
    priority=5,
    explainable=True,
    hot_reload=True
)
class StrategyIntrospector(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🔍 PRODUCTION-GRADE Strategy Introspector v3.0
    
    Advanced strategy analysis system with:
    - Intelligent pattern recognition and behavioral analysis
    - Performance-driven adaptation recommendations
    - Deep insights into trading strategy evolution
    - SmartInfoBus zero-wiring architecture
    - Comprehensive thesis generation for all analysis decisions
    """

    def _initialize(self):
        """Initialize advanced strategy analysis and introspection systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced configuration
        self.history_len = self.config.get('history_len', 10)
        self.analysis_depth = self.config.get('analysis_depth', 'comprehensive')
        self.performance_window = self.config.get('performance_window', 20)
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.1)
        self.debug = self.config.get('debug', False)
        
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
        
        # Advanced strategy categorization
        self.strategy_categories = {
            'conservative': {'risk_threshold': 0.8, 'return_threshold': 1.2},
            'balanced': {'risk_threshold': 1.2, 'return_threshold': 1.8},
            'aggressive': {'risk_threshold': 2.0, 'return_threshold': 3.0},
            'scalping': {'duration_threshold': 10, 'frequency_threshold': 5},
            'swing': {'duration_threshold': 100, 'frequency_threshold': 1},
            'momentum': {'trend_strength': 0.7, 'volatility_tolerance': 0.8},
            'contrarian': {'reversal_strength': 0.6, 'patience_factor': 0.9}
        }
        
        # Performance tracking metrics
        self.introspection_metrics = {
            'total_strategies_analyzed': 0,
            'significant_adaptations': 0,
            'performance_improvements': 0,
            'performance_degradations': 0,
            'last_major_insight': None,
            'analysis_accuracy': 0.0,
            'prediction_success_rate': 0.0,
            'adaptation_success_rate': 0.0
        }
        
        # Real-time analysis state
        self.current_analysis = {
            'dominant_strategy_type': 'balanced',
            'performance_trend': 'stable',
            'adaptation_needed': False,
            'recommended_adjustments': [],
            'confidence_level': 0.5,
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'behavioral_patterns': {},
            'risk_assessment': 'moderate'
        }
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Advanced analysis intelligence
        self.analysis_intelligence = {
            'pattern_sensitivity': 0.8,
            'adaptation_momentum': 0.9,
            'confidence_decay': 0.95,
            'prediction_memory': 0.85
        }
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🔍",
            message=f"Strategy Introspector v{version} initialized",
            history_len=self.history_len,
            analysis_depth=self.analysis_depth,
            performance_window=self.performance_window
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="StrategyIntrospector",
            log_path="logs/strategy/strategy_introspector.log",
            max_lines=2000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("StrategyIntrospector", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

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
            'adaptation_score': 0.0,
            'behavioral_fingerprint': {},
            'market_regime_performance': defaultdict(list)
        }

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Strategy Introspector v3.0 Initialization Complete:
        
        Advanced Strategy Analysis System:
        - Analysis depth: {self.analysis_depth} with {self.history_len} record capacity
        - Performance window: {self.performance_window} trades for trend analysis
        - Strategy categories: {len(self.strategy_categories)} distinct classification types
        - Adaptation threshold: {self.adaptation_threshold:.1%} for significant change detection
        
        Current Configuration:
        - Behavioral pattern recognition with intelligent fingerprinting
        - Multi-dimensional performance analytics across market regimes
        - Real-time adaptation recommendation engine
        - Predictive modeling for strategy evolution forecasting
        
        Analysis Intelligence Features:
        - Pattern recognition with {self.analysis_intelligence['pattern_sensitivity']:.1%} sensitivity
        - Confidence scoring with {self.analysis_intelligence['confidence_decay']:.1%} decay factor
        - Predictive memory retention of {self.analysis_intelligence['prediction_memory']:.1%}
        - Adaptation momentum tracking with {self.analysis_intelligence['adaptation_momentum']:.1%} weighting
        
        Advanced Capabilities:
        - Real-time strategy classification and behavioral analysis
        - Performance trend prediction with confidence intervals
        - Market regime-aware adaptation recommendations
        - Comprehensive strategy evolution tracking and insights
        
        Expected Outcomes:
        - Deep insights into strategy performance patterns and evolution
        - Intelligent adaptation recommendations based on behavioral analysis
        - Predictive modeling for strategy optimization opportunities
        - Transparent introspection decisions with comprehensive explanations
        """
        
        self.smart_bus.set('strategy_introspector_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'analysis_depth': self.analysis_depth,
                'strategy_categories': list(self.strategy_categories.keys()),
                'baseline_metrics': self._baseline_metrics
            }
        }, module='StrategyIntrospector', thesis=thesis)

    async def process(self) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive strategy analysis
        
        Returns:
            Dict containing strategy analysis, insights, and recommendations
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive market data from SmartInfoBus
            market_data = await self._get_comprehensive_market_data()
            
            # Core strategy analysis with error handling
            strategy_analysis = await self._analyze_strategy_patterns_comprehensive(market_data)
            
            # Update strategy profiles based on recent activity
            await self._update_strategy_profiles_comprehensive(market_data, strategy_analysis)
            
            # Generate adaptation insights with intelligent algorithms
            adaptation_insights = await self._generate_adaptation_insights_intelligent(strategy_analysis, market_data)
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_introspection_thesis(strategy_analysis, adaptation_insights)
            
            # Create comprehensive results
            results = {
                'strategy_analysis': strategy_analysis,
                'performance_insights': self._get_performance_insights(),
                'adaptation_recommendations': adaptation_insights,
                'strategy_profiles': self._get_strategy_profiles_summary(),
                'introspection_metrics': self.introspection_metrics.copy(),
                'behavior_patterns': self._get_behavioral_patterns(),
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('StrategyIntrospector', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data using modern SmartInfoBus patterns"""
        try:
            return {
                'recent_trades': self.smart_bus.get('recent_trades', 'StrategyIntrospector') or [],
                'module_data': self.smart_bus.get('module_data', 'StrategyIntrospector') or {},
                'risk_data': self.smart_bus.get('risk_data', 'StrategyIntrospector') or {},
                'market_regime': self.smart_bus.get('market_regime', 'StrategyIntrospector') or 'unknown',
                'volatility_data': self.smart_bus.get('volatility_data', 'StrategyIntrospector') or {},
                'trading_performance': self.smart_bus.get('trading_performance', 'StrategyIntrospector') or {},
                'strategy_weights': self.smart_bus.get('strategy_weights', 'StrategyIntrospector') or {},
                'market_context': self.smart_bus.get('market_context', 'StrategyIntrospector') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "StrategyIntrospector")
            self.logger.warning(f"Market data retrieval incomplete: {error_context}")
            return self._get_safe_market_defaults()

    async def _analyze_strategy_patterns_comprehensive(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive strategy pattern analysis with advanced algorithms"""
        try:
            analysis = {
                'strategy_context': {},
                'performance_analysis': {},
                'behavioral_analysis': {},
                'adaptation_analysis': {},
                'trend_analysis': {},
                'analysis_timestamp': datetime.datetime.now().isoformat()
            }
            
            # Extract strategy context from market data
            strategy_context = await self._extract_strategy_context_comprehensive(market_data)
            analysis['strategy_context'] = strategy_context
            
            # Analyze current performance patterns
            performance_analysis = await self._analyze_current_performance_comprehensive(strategy_context)
            analysis['performance_analysis'] = performance_analysis
            
            # Behavioral pattern recognition
            behavioral_analysis = await self._analyze_behavioral_patterns(strategy_context, performance_analysis)
            analysis['behavioral_analysis'] = behavioral_analysis
            
            # Adaptation needs assessment
            adaptation_analysis = await self._assess_adaptation_needs_comprehensive(performance_analysis, behavioral_analysis)
            analysis['adaptation_analysis'] = adaptation_analysis
            
            # Performance trend analysis
            trend_analysis = await self._analyze_performance_trends(strategy_context, performance_analysis)
            analysis['trend_analysis'] = trend_analysis
            
            # Update current analysis state
            await self._update_current_analysis_state(analysis)
            
            # Log significant analysis results
            await self._log_significant_analysis_results(analysis)
            
            return analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "StrategyIntrospector")
            self.logger.error(f"Strategy pattern analysis failed: {error_context}")
            return self._get_safe_analysis_defaults()

    async def _extract_strategy_context_comprehensive(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive strategy context from market data"""
        try:
            recent_trades = market_data.get('recent_trades', [])
            module_data = market_data.get('module_data', {})
            risk_data = market_data.get('risk_data', {})
            trading_performance = market_data.get('trading_performance', {})
            
            # Extract strategy information from other modules
            strategy_data = module_data.get('strategy_arbiter', {})
            genome_data = module_data.get('strategy_genome_pool', {})
            mode_data = module_data.get('opponent_mode_enhancer', {})
            
            strategy_context = {
                'timestamp': datetime.datetime.now().isoformat(),
                'recent_trades': recent_trades,
                'active_strategies': strategy_data.get('active_strategies', []),
                'strategy_weights': strategy_data.get('strategy_weights', {}),
                'active_genome': genome_data.get('active_genome', None),
                'best_genome': genome_data.get('best_genome', None),
                'mode_weights': mode_data.get('mode_weights', {}),
                'current_balance': risk_data.get('balance', 0),
                'current_drawdown': risk_data.get('current_drawdown', 0),
                'market_regime': market_data.get('market_regime', 'unknown'),
                'volatility_level': market_data.get('market_context', {}).get('volatility_level', 'medium'),
                'session_pnl': trading_performance.get('session_pnl', 0),
                'trade_frequency': self._calculate_trade_frequency_advanced(recent_trades),
                'strategy_evolution': self._analyze_strategy_evolution(strategy_data, genome_data)
            }
            
            return strategy_context
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "strategy_context")
            self.logger.warning(f"Strategy context extraction failed: {error_context}")
            return {'timestamp': datetime.datetime.now().isoformat(), 'extraction_error': str(error_context)}

    def _calculate_trade_frequency_advanced(self, recent_trades: List[Dict]) -> Dict[str, Any]:
        """Calculate advanced trade frequency metrics"""
        try:
            if len(recent_trades) < 2:
                return {'trades_per_hour': 0.0, 'activity_intensity': 0.0, 'frequency_trend': 'stable'}
            
            # Calculate basic frequency
            trades_per_hour = min(10.0, len(recent_trades) / 2.0)
            
            # Calculate activity intensity (clustering)
            if len(recent_trades) >= 5:
                # Look at trade spacing
                recent_times = [datetime.datetime.fromisoformat(t.get('timestamp', datetime.datetime.now().isoformat())) 
                              for t in recent_trades[-5:]]
                time_gaps = [(recent_times[i] - recent_times[i-1]).total_seconds() / 60 
                           for i in range(1, len(recent_times))]
                avg_gap = np.mean(time_gaps) if time_gaps else 60
                activity_intensity = max(0.0, min(1.0, 60 / (avg_gap + 1)))
            else:
                activity_intensity = 0.5
            
            # Frequency trend
            if len(recent_trades) >= 6:
                early_freq = len(recent_trades[:3])
                late_freq = len(recent_trades[-3:])
                if late_freq > early_freq * 1.2:
                    frequency_trend = 'increasing'
                elif late_freq < early_freq * 0.8:
                    frequency_trend = 'decreasing'
                else:
                    frequency_trend = 'stable'
            else:
                frequency_trend = 'stable'
            
            return {
                'trades_per_hour': trades_per_hour,
                'activity_intensity': activity_intensity,
                'frequency_trend': frequency_trend
            }
            
        except Exception:
            return {'trades_per_hour': 0.0, 'activity_intensity': 0.0, 'frequency_trend': 'stable'}

    def _analyze_strategy_evolution(self, strategy_data: Dict[str, Any], genome_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze strategy evolution patterns"""
        try:
            evolution = {
                'genome_evolution': 'stable',
                'weight_changes': 'minimal',
                'adaptation_level': 'low',
                'evolutionary_pressure': 0.5
            }
            
            # Genome evolution analysis
            if genome_data:
                current_generation = genome_data.get('current_generation', 0)
                generations_without_improvement = genome_data.get('generations_without_improvement', 0)
                
                if generations_without_improvement > 10:
                    evolution['genome_evolution'] = 'stagnant'
                elif generations_without_improvement < 3:
                    evolution['genome_evolution'] = 'active'
                else:
                    evolution['genome_evolution'] = 'moderate'
                
                evolution['evolutionary_pressure'] = min(1.0, generations_without_improvement / 20.0)
            
            # Strategy weight evolution
            strategy_weights = strategy_data.get('strategy_weights', {})
            if strategy_weights:
                weight_variance = np.var(list(strategy_weights.values())) if len(strategy_weights) > 1 else 0
                if weight_variance > 0.1:
                    evolution['weight_changes'] = 'significant'
                elif weight_variance > 0.05:
                    evolution['weight_changes'] = 'moderate'
                else:
                    evolution['weight_changes'] = 'minimal'
            
            return evolution
            
        except Exception:
            return {
                'genome_evolution': 'unknown',
                'weight_changes': 'unknown',
                'adaptation_level': 'unknown',
                'evolutionary_pressure': 0.5
            }

    async def _analyze_current_performance_comprehensive(self, strategy_context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze current strategy performance with comprehensive metrics"""
        try:
            recent_trades = strategy_context.get('recent_trades', [])
            
            if len(recent_trades) >= 3:
                # Calculate comprehensive performance metrics
                pnls = [t.get('pnl', 0) for t in recent_trades]
                durations = [t.get('duration', 30) for t in recent_trades if 'duration' in t]
                
                # Basic performance metrics
                metrics = self._calculate_basic_performance_metrics(pnls, durations)
                
                # Advanced performance metrics
                advanced_metrics = self._calculate_advanced_performance_metrics(pnls, recent_trades)
                
                # Risk metrics
                risk_metrics = self._calculate_risk_metrics(pnls, strategy_context)
                
                # Combine all metrics
                performance_metrics = {**metrics, **advanced_metrics, **risk_metrics}
                
            else:
                # Use baseline metrics when insufficient data
                performance_metrics = {k: v for k, v in self._baseline_metrics.items()}
            
            # Add contextual metrics
            performance_metrics.update({
                'trade_frequency': strategy_context.get('trade_frequency', {}).get('trades_per_hour', 0.0),
                'current_drawdown': strategy_context.get('current_drawdown', 0.0),
                'session_pnl': strategy_context.get('session_pnl', 0.0),
                'market_regime': strategy_context.get('market_regime', 'unknown'),
                'volatility_level': strategy_context.get('volatility_level', 'medium')
            })
            
            return performance_metrics
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_analysis")
            self.logger.warning(f"Performance analysis failed: {error_context}")
            return {k: v for k, v in self._baseline_metrics.items()}

    def _calculate_basic_performance_metrics(self, pnls: List[float], durations: List[float]) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        metrics = {}
        
        # Win rate
        wins = len([p for p in pnls if p > 0])
        metrics['win_rate'] = wins / len(pnls)
        
        # Average P&L
        metrics['avg_pnl'] = np.mean(pnls)
        
        # Risk-adjusted return (Sharpe-like ratio)
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
        
        return metrics

    def _calculate_advanced_performance_metrics(self, pnls: List[float], recent_trades: List[Dict]) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        metrics = {}
        
        # Profit factor
        profits = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        if profits and losses:
            metrics['profit_factor'] = sum(profits) / abs(sum(losses))
        else:
            metrics['profit_factor'] = 1.0 if profits else 0.0
        
        # Maximum drawdown
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
        
        # Performance momentum
        if len(pnls) >= 6:
            recent_avg = np.mean(pnls[-3:])
            older_avg = np.mean(pnls[-6:-3])
            metrics['performance_momentum'] = (recent_avg - older_avg) / (abs(older_avg) + 1e-6)
        else:
            metrics['performance_momentum'] = 0.0
        
        # Trade quality score
        if recent_trades:
            quality_scores = []
            for trade in recent_trades:
                pnl = trade.get('pnl', 0)
                duration = trade.get('duration', 30)
                # Quality = PnL efficiency over time
                quality = pnl / (duration + 1) if duration > 0 else 0
                quality_scores.append(quality)
            metrics['trade_quality'] = np.mean(quality_scores)
        else:
            metrics['trade_quality'] = 0.0
        
        return metrics

    def _calculate_risk_metrics(self, pnls: List[float], strategy_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        metrics = {}
        
        # Value at Risk (VaR) - 95th percentile worst case
        if len(pnls) >= 5:
            metrics['var_95'] = np.percentile(pnls, 5)  # 5th percentile (worst 5%)
        else:
            metrics['var_95'] = min(pnls) if pnls else 0.0
        
        # Risk-adjusted return based on current balance
        current_balance = strategy_context.get('current_balance', 10000)
        avg_pnl = np.mean(pnls) if pnls else 0
        metrics['risk_adjusted_return'] = (avg_pnl / current_balance) * 100 if current_balance > 0 else 0
        
        # Volatility of returns
        if len(pnls) > 1:
            metrics['return_volatility'] = np.std(pnls) / (abs(np.mean(pnls)) + 1e-6)
        else:
            metrics['return_volatility'] = 0.0
        
        # Current exposure based on drawdown
        current_drawdown = strategy_context.get('current_drawdown', 0)
        metrics['exposure_level'] = min(1.0, current_drawdown * 10)  # Scale drawdown to exposure
        
        return metrics

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

    async def _analyze_behavioral_patterns(self, strategy_context: Dict[str, Any], 
                                         performance_analysis: Dict[str, float]) -> Dict[str, Any]:
        """Analyze behavioral patterns in strategy execution"""
        try:
            behavioral_analysis = {
                'trading_style': '',
                'risk_preference': '',
                'timing_patterns': {},
                'adaptation_behavior': '',
                'market_sensitivity': {}
            }
            
            # Classify trading style
            behavioral_analysis['trading_style'] = self._classify_trading_style_advanced(
                strategy_context, performance_analysis
            )
            
            # Analyze risk preference
            behavioral_analysis['risk_preference'] = self._analyze_risk_preference(performance_analysis)
            
            # Timing pattern analysis
            behavioral_analysis['timing_patterns'] = self._analyze_timing_patterns(strategy_context)
            
            # Adaptation behavior
            behavioral_analysis['adaptation_behavior'] = self._analyze_adaptation_behavior(strategy_context)
            
            # Market sensitivity analysis
            behavioral_analysis['market_sensitivity'] = self._analyze_market_sensitivity(
                strategy_context, performance_analysis
            )
            
            return behavioral_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "behavioral_analysis")
            return {'error': str(error_context)}

    def _classify_trading_style_advanced(self, strategy_context: Dict[str, Any], 
                                       performance_analysis: Dict[str, float]) -> str:
        """Classify trading style with advanced pattern recognition"""
        try:
            avg_duration = performance_analysis.get('avg_duration', 30)
            trade_frequency = performance_analysis.get('trade_frequency', 0)
            win_rate = performance_analysis.get('win_rate', 0.5)
            profit_factor = performance_analysis.get('profit_factor', 1.0)
            max_drawdown = performance_analysis.get('max_drawdown', 0.0)
            trade_quality = performance_analysis.get('trade_quality', 0.0)
            
            # Get genome information for additional context
            active_genome = strategy_context.get('active_genome', [])
            if active_genome and len(active_genome) >= 4:
                sl_ratio = active_genome[0]
                tp_ratio = active_genome[1]
                risk_reward = tp_ratio / sl_ratio if sl_ratio > 0 else 1.5
                volatility_scale = active_genome[2]
            else:
                risk_reward = profit_factor
                volatility_scale = 1.0
            
            # Advanced classification logic
            if avg_duration < 10 and trade_frequency > 4:
                return 'high_frequency_scalping'
            elif avg_duration < 20 and trade_frequency > 2 and trade_quality > 0.5:
                return 'efficient_scalping'
            elif avg_duration > 80 and trade_frequency < 1.5:
                return 'position_trading'
            elif 30 < avg_duration < 80 and 1.5 <= trade_frequency <= 3:
                return 'swing_trading'
            elif max_drawdown < 0.015 and risk_reward < 1.2:
                return 'ultra_conservative'
            elif max_drawdown < 0.03 and win_rate > 0.6:
                return 'conservative_consistent'
            elif max_drawdown > 0.08 or risk_reward > 3.0:
                return 'high_risk_aggressive'
            elif volatility_scale > 1.5 and profit_factor > 1.5:
                return 'volatility_adaptive'
            elif win_rate > 0.65 and profit_factor > 1.3:
                return 'high_probability'
            else:
                return 'balanced_approach'
                
        except Exception:
            return 'unknown_style'

    def _analyze_risk_preference(self, performance_analysis: Dict[str, float]) -> str:
        """Analyze risk preference from performance patterns"""
        try:
            max_drawdown = performance_analysis.get('max_drawdown', 0.0)
            return_volatility = performance_analysis.get('return_volatility', 0.0)
            exposure_level = performance_analysis.get('exposure_level', 0.0)
            var_95 = performance_analysis.get('var_95', 0.0)
            
            # Risk scoring
            risk_score = 0
            
            if max_drawdown > 0.1:
                risk_score += 3
            elif max_drawdown > 0.05:
                risk_score += 2
            elif max_drawdown > 0.02:
                risk_score += 1
            
            if return_volatility > 2.0:
                risk_score += 2
            elif return_volatility > 1.0:
                risk_score += 1
            
            if abs(var_95) > 50:
                risk_score += 2
            elif abs(var_95) > 25:
                risk_score += 1
            
            # Classification
            if risk_score <= 1:
                return 'risk_averse'
            elif risk_score <= 3:
                return 'moderate_risk'
            elif risk_score <= 5:
                return 'risk_seeking'
            else:
                return 'high_risk_tolerance'
                
        except Exception:
            return 'unknown_risk_preference'

    def _analyze_timing_patterns(self, strategy_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze timing patterns in strategy execution"""
        try:
            trade_frequency_data = strategy_context.get('trade_frequency', {})
            recent_trades = strategy_context.get('recent_trades', [])
            
            patterns = {
                'frequency_trend': trade_frequency_data.get('frequency_trend', 'stable'),
                'activity_intensity': trade_frequency_data.get('activity_intensity', 0.5),
                'timing_consistency': 'unknown',
                'preferred_conditions': []
            }
            
            if len(recent_trades) >= 5:
                # Analyze trade timing consistency
                trade_gaps = []
                for i in range(1, len(recent_trades)):
                    try:
                        t1 = datetime.datetime.fromisoformat(recent_trades[i-1].get('timestamp', ''))
                        t2 = datetime.datetime.fromisoformat(recent_trades[i].get('timestamp', ''))
                        gap = (t2 - t1).total_seconds() / 60  # minutes
                        trade_gaps.append(gap)
                    except:
                        continue
                
                if trade_gaps:
                    gap_consistency = 1.0 - (np.std(trade_gaps) / (np.mean(trade_gaps) + 1e-6))
                    if gap_consistency > 0.7:
                        patterns['timing_consistency'] = 'highly_consistent'
                    elif gap_consistency > 0.5:
                        patterns['timing_consistency'] = 'moderately_consistent'
                    else:
                        patterns['timing_consistency'] = 'irregular'
                
                # Analyze preferred market conditions
                successful_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
                if successful_trades:
                    regimes = [t.get('market_regime', 'unknown') for t in successful_trades]
                    if regimes:
                        from collections import Counter
                        regime_counts = Counter(regimes)
                        preferred_regime = regime_counts.most_common(1)[0][0] if regime_counts else 'unknown'
                        if preferred_regime != 'unknown':
                            patterns['preferred_conditions'].append(f'regime_{preferred_regime}')
            
            return patterns
            
        except Exception:
            return {
                'frequency_trend': 'unknown',
                'activity_intensity': 0.5,
                'timing_consistency': 'unknown',
                'preferred_conditions': []
            }

    def _analyze_adaptation_behavior(self, strategy_context: Dict[str, Any]) -> str:
        """Analyze adaptation behavior patterns"""
        try:
            strategy_evolution = strategy_context.get('strategy_evolution', {})
            genome_evolution = strategy_evolution.get('genome_evolution', 'stable')
            weight_changes = strategy_evolution.get('weight_changes', 'minimal')
            evolutionary_pressure = strategy_evolution.get('evolutionary_pressure', 0.5)
            
            # Classify adaptation behavior
            if genome_evolution == 'stagnant' and weight_changes == 'minimal':
                return 'static_conservative'
            elif genome_evolution == 'active' and weight_changes == 'significant':
                return 'highly_adaptive'
            elif evolutionary_pressure > 0.7:
                return 'pressure_responsive'
            elif genome_evolution == 'active' or weight_changes in ['moderate', 'significant']:
                return 'moderately_adaptive'
            else:
                return 'stable_consistent'
                
        except Exception:
            return 'unknown_adaptation'

    def _analyze_market_sensitivity(self, strategy_context: Dict[str, Any], 
                                  performance_analysis: Dict[str, float]) -> Dict[str, Any]:
        """Analyze sensitivity to market conditions"""
        try:
            market_regime = strategy_context.get('market_regime', 'unknown')
            volatility_level = strategy_context.get('volatility_level', 'medium')
            recent_trades = strategy_context.get('recent_trades', [])
            
            sensitivity = {
                'regime_sensitivity': 'moderate',
                'volatility_sensitivity': 'moderate',
                'performance_stability': 'stable',
                'adaptation_speed': 'normal'
            }
            
            # Analyze regime sensitivity
            if len(recent_trades) >= 10:
                regime_performance = defaultdict(list)
                for trade in recent_trades:
                    trade_regime = trade.get('market_regime', 'unknown')
                    trade_pnl = trade.get('pnl', 0)
                    regime_performance[trade_regime].append(trade_pnl)
                
                if len(regime_performance) > 1:
                    regime_variances = {k: np.var(v) for k, v in regime_performance.items() if len(v) > 1}
                    if regime_variances:
                        avg_variance = np.mean(list(regime_variances.values()))
                        if avg_variance > 100:
                            sensitivity['regime_sensitivity'] = 'high'
                        elif avg_variance < 25:
                            sensitivity['regime_sensitivity'] = 'low'
            
            # Analyze volatility sensitivity
            return_volatility = performance_analysis.get('return_volatility', 0.0)
            if return_volatility > 1.5:
                sensitivity['volatility_sensitivity'] = 'high'
            elif return_volatility < 0.5:
                sensitivity['volatility_sensitivity'] = 'low'
            
            # Performance stability
            performance_momentum = performance_analysis.get('performance_momentum', 0.0)
            if abs(performance_momentum) > 0.5:
                sensitivity['performance_stability'] = 'unstable'
            elif abs(performance_momentum) < 0.1:
                sensitivity['performance_stability'] = 'highly_stable'
            
            return sensitivity
            
        except Exception:
            return {
                'regime_sensitivity': 'unknown',
                'volatility_sensitivity': 'unknown', 
                'performance_stability': 'unknown',
                'adaptation_speed': 'unknown'
            }

    async def _assess_adaptation_needs_comprehensive(self, performance_analysis: Dict[str, float], 
                                                   behavioral_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess adaptation needs with comprehensive analysis"""
        try:
            adaptation_assessment = {
                'adaptation_needed': False,
                'urgency_level': 'low',
                'adaptation_areas': [],
                'confidence_level': 0.5,
                'recommended_actions': []
            }
            
            # Performance-based adaptation needs
            performance_issues = self._identify_performance_issues(performance_analysis)
            
            # Behavioral-based adaptation needs
            behavioral_issues = self._identify_behavioral_issues(behavioral_analysis)
            
            # Combine assessments
            all_issues = performance_issues + behavioral_issues
            
            if all_issues:
                adaptation_assessment['adaptation_needed'] = True
                adaptation_assessment['adaptation_areas'] = [issue['area'] for issue in all_issues]
                
                # Determine urgency level
                critical_issues = [issue for issue in all_issues if issue['severity'] == 'critical']
                high_issues = [issue for issue in all_issues if issue['severity'] == 'high']
                
                if critical_issues:
                    adaptation_assessment['urgency_level'] = 'critical'
                elif high_issues:
                    adaptation_assessment['urgency_level'] = 'high'
                elif len(all_issues) >= 3:
                    adaptation_assessment['urgency_level'] = 'medium'
                else:
                    adaptation_assessment['urgency_level'] = 'low'
                
                # Generate recommended actions
                adaptation_assessment['recommended_actions'] = self._generate_adaptation_actions(all_issues)
                
                # Calculate confidence
                adaptation_assessment['confidence_level'] = min(1.0, len(all_issues) / 5.0)
            
            return adaptation_assessment
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "adaptation_assessment")
            return {'error': str(error_context)}

    def _identify_performance_issues(self, performance_analysis: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify performance-related issues"""
        issues = []
        
        # Win rate issues
        win_rate = performance_analysis.get('win_rate', 0.5)
        if win_rate < 0.3:
            issues.append({
                'area': 'entry_quality',
                'severity': 'critical',
                'description': f'Very low win rate: {win_rate:.1%}',
                'target_improvement': 'Improve entry signal quality'
            })
        elif win_rate < 0.4:
            issues.append({
                'area': 'entry_quality',
                'severity': 'high',
                'description': f'Low win rate: {win_rate:.1%}',
                'target_improvement': 'Review entry criteria'
            })
        
        # Drawdown issues
        max_drawdown = performance_analysis.get('max_drawdown', 0.0)
        if max_drawdown > 0.15:
            issues.append({
                'area': 'risk_management',
                'severity': 'critical',
                'description': f'Excessive drawdown: {max_drawdown:.1%}',
                'target_improvement': 'Implement stricter risk controls'
            })
        elif max_drawdown > 0.08:
            issues.append({
                'area': 'risk_management',
                'severity': 'high',
                'description': f'High drawdown: {max_drawdown:.1%}',
                'target_improvement': 'Reduce position sizes'
            })
        
        # Profit factor issues
        profit_factor = performance_analysis.get('profit_factor', 1.0)
        if profit_factor < 0.7:
            issues.append({
                'area': 'exit_strategy',
                'severity': 'high',
                'description': f'Poor profit factor: {profit_factor:.2f}',
                'target_improvement': 'Optimize exit strategy'
            })
        
        # Consistency issues
        consistency_score = performance_analysis.get('consistency_score', 0.5)
        if consistency_score < 0.3:
            issues.append({
                'area': 'strategy_consistency',
                'severity': 'medium',
                'description': f'Low consistency: {consistency_score:.2f}',
                'target_improvement': 'Improve strategy stability'
            })
        
        return issues

    def _identify_behavioral_issues(self, behavioral_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify behavioral-related issues"""
        issues = []
        
        trading_style = behavioral_analysis.get('trading_style', '')
        risk_preference = behavioral_analysis.get('risk_preference', '')
        timing_patterns = behavioral_analysis.get('timing_patterns', {})
        
        # High-risk behavioral issues
        if risk_preference == 'high_risk_tolerance':
            issues.append({
                'area': 'risk_behavior',
                'severity': 'medium',
                'description': 'Exhibiting high-risk behavior patterns',
                'target_improvement': 'Implement risk-limiting measures'
            })
        
        # Timing consistency issues
        timing_consistency = timing_patterns.get('timing_consistency', 'unknown')
        if timing_consistency == 'irregular':
            issues.append({
                'area': 'timing_discipline',
                'severity': 'low',
                'description': 'Irregular timing patterns detected',
                'target_improvement': 'Improve timing discipline'
            })
        
        # Adaptation behavior issues
        adaptation_behavior = behavioral_analysis.get('adaptation_behavior', '')
        if adaptation_behavior == 'static_conservative':
            issues.append({
                'area': 'adaptability',
                'severity': 'low',
                'description': 'Limited adaptation to changing conditions',
                'target_improvement': 'Increase strategy flexibility'
            })
        
        return issues

    def _generate_adaptation_actions(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate specific adaptation actions based on identified issues"""
        actions = []
        
        for issue in issues:
            area = issue['area']
            severity = issue['severity']
            
            if area == 'entry_quality':
                if severity == 'critical':
                    actions.append("Immediately implement stricter entry filters and signal validation")
                else:
                    actions.append("Review and optimize entry criteria with additional confirmation signals")
            
            elif area == 'risk_management':
                if severity == 'critical':
                    actions.append("Halt trading and implement emergency risk controls")
                else:
                    actions.append("Reduce position sizes and implement tighter stop-loss management")
            
            elif area == 'exit_strategy':
                actions.append("Analyze and optimize profit-taking and stop-loss strategies")
            
            elif area == 'strategy_consistency':
                actions.append("Implement consistency monitoring and strategy standardization")
            
            elif area == 'risk_behavior':
                actions.append("Implement behavioral risk controls and position sizing limits")
            
            elif area == 'timing_discipline':
                actions.append("Establish structured timing protocols and execution discipline")
            
            elif area == 'adaptability':
                actions.append("Increase strategy flexibility and market condition responsiveness")
        
        # Remove duplicates and limit to top 5
        return list(dict.fromkeys(actions))[:5]

    async def _analyze_performance_trends(self, strategy_context: Dict[str, Any], 
                                        performance_analysis: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance trends with predictive insights"""
        try:
            recent_trades = strategy_context.get('recent_trades', [])
            
            trend_analysis = {
                'short_term_trend': 'stable',
                'medium_term_trend': 'stable',
                'trend_strength': 0.5,
                'trend_sustainability': 0.5,
                'predicted_direction': 'neutral',
                'confidence_interval': 0.5
            }
            
            if len(recent_trades) >= 6:
                pnls = [t.get('pnl', 0) for t in recent_trades]
                
                # Short-term trend (last 3 trades)
                recent_avg = np.mean(pnls[-3:])
                older_avg = np.mean(pnls[-6:-3])
                
                trend_strength = abs(recent_avg - older_avg) / (abs(older_avg) + 1e-6)
                trend_analysis['trend_strength'] = min(1.0, trend_strength)
                
                if recent_avg > older_avg + 5:
                    trend_analysis['short_term_trend'] = 'improving'
                elif recent_avg < older_avg - 5:
                    trend_analysis['short_term_trend'] = 'declining'
                else:
                    trend_analysis['short_term_trend'] = 'stable'
                
                # Medium-term trend (if enough data)
                if len(recent_trades) >= 12:
                    very_recent = np.mean(pnls[-4:])
                    medium_term = np.mean(pnls[-12:-4])
                    
                    if very_recent > medium_term + 10:
                        trend_analysis['medium_term_trend'] = 'improving'
                    elif very_recent < medium_term - 10:
                        trend_analysis['medium_term_trend'] = 'declining'
                    else:
                        trend_analysis['medium_term_trend'] = 'stable'
                
                # Trend sustainability (consistency of direction)
                if len(pnls) >= 8:
                    # Calculate rolling averages
                    rolling_means = [np.mean(pnls[i:i+3]) for i in range(len(pnls)-2)]
                    trend_changes = sum(1 for i in range(1, len(rolling_means)) 
                                      if (rolling_means[i] > rolling_means[i-1]) != 
                                         (rolling_means[i-1] > rolling_means[i-2] if i > 1 else True))
                    
                    sustainability = 1.0 - (trend_changes / max(len(rolling_means) - 1, 1))
                    trend_analysis['trend_sustainability'] = max(0.0, sustainability)
                
                # Predicted direction
                performance_momentum = performance_analysis.get('performance_momentum', 0.0)
                if performance_momentum > 0.2:
                    trend_analysis['predicted_direction'] = 'positive'
                elif performance_momentum < -0.2:
                    trend_analysis['predicted_direction'] = 'negative'
                else:
                    trend_analysis['predicted_direction'] = 'neutral'
                
                # Confidence interval
                consistency_score = performance_analysis.get('consistency_score', 0.5)
                trend_analysis['confidence_interval'] = (trend_analysis['trend_sustainability'] + consistency_score) / 2
            
            return trend_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "trend_analysis")
            return {'error': str(error_context)}

    async def _update_current_analysis_state(self, analysis: Dict[str, Any]):
        """Update current analysis state with new insights"""
        try:
            performance_analysis = analysis.get('performance_analysis', {})
            behavioral_analysis = analysis.get('behavioral_analysis', {})
            adaptation_analysis = analysis.get('adaptation_analysis', {})
            trend_analysis = analysis.get('trend_analysis', {})
            
            # Update current analysis
            self.current_analysis.update({
                'dominant_strategy_type': behavioral_analysis.get('trading_style', 'balanced'),
                'performance_trend': trend_analysis.get('short_term_trend', 'stable'),
                'adaptation_needed': adaptation_analysis.get('adaptation_needed', False),
                'recommended_adjustments': adaptation_analysis.get('recommended_actions', []),
                'confidence_level': adaptation_analysis.get('confidence_level', 0.5),
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                'behavioral_patterns': behavioral_analysis,
                'risk_assessment': behavioral_analysis.get('risk_preference', 'moderate')
            })
            
            # Update metrics
            self.introspection_metrics['total_strategies_analyzed'] += 1
            
            if adaptation_analysis.get('adaptation_needed', False):
                self.introspection_metrics['significant_adaptations'] += 1
            
            # Update performance improvement tracking
            performance_momentum = performance_analysis.get('performance_momentum', 0.0)
            if performance_momentum > 0.1:
                self.introspection_metrics['performance_improvements'] += 1
            elif performance_momentum < -0.1:
                self.introspection_metrics['performance_degradations'] += 1
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "analysis_state_update")
            self.logger.warning(f"Analysis state update failed: {error_context}")

    async def _log_significant_analysis_results(self, analysis: Dict[str, Any]):
        """Log significant analysis results"""
        try:
            adaptation_analysis = analysis.get('adaptation_analysis', {})
            behavioral_analysis = analysis.get('behavioral_analysis', {})
            performance_analysis = analysis.get('performance_analysis', {})
            
            # Log adaptation needs
            if adaptation_analysis.get('adaptation_needed', False):
                urgency = adaptation_analysis.get('urgency_level', 'low')
                areas = adaptation_analysis.get('adaptation_areas', [])
                
                self.logger.warning(format_operator_message(
                    icon="🚨",
                    message=f"Strategy adaptation needed - {urgency} urgency",
                    areas=", ".join(areas[:3]),
                    confidence=f"{adaptation_analysis.get('confidence_level', 0.5):.1%}"
                ))
            
            # Log significant performance issues
            max_drawdown = performance_analysis.get('max_drawdown', 0.0)
            if max_drawdown > 0.1:
                self.logger.error(format_operator_message(
                    icon="📉",
                    message="High drawdown detected",
                    drawdown=f"{max_drawdown:.1%}",
                    action="immediate_review_required"
                ))
            
            # Log interesting behavioral patterns
            trading_style = behavioral_analysis.get('trading_style', '')
            if trading_style in ['high_frequency_scalping', 'high_risk_aggressive', 'ultra_conservative']:
                self.logger.info(format_operator_message(
                    icon="🎭",
                    message="Distinctive trading style detected",
                    style=trading_style,
                    risk_preference=behavioral_analysis.get('risk_preference', 'unknown')
                ))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "analysis_logging")

    async def _update_strategy_profiles_comprehensive(self, market_data: Dict[str, Any], 
                                                    strategy_analysis: Dict[str, Any]):
        """Update strategy profiles with comprehensive tracking"""
        try:
            recent_trades = market_data.get('recent_trades', [])
            behavioral_analysis = strategy_analysis.get('behavioral_analysis', {})
            performance_analysis = strategy_analysis.get('performance_analysis', {})
            
            if not recent_trades:
                return
            
            # Get the most recent trade result
            last_trade = recent_trades[-1]
            pnl = last_trade.get('pnl', 0)
            
            # Determine strategy type
            strategy_type = behavioral_analysis.get('trading_style', 'balanced')
            
            # Update profile for this strategy type
            await self._update_strategy_profile_comprehensive(
                strategy_type, last_trade, performance_analysis, behavioral_analysis
            )
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "profile_update")
            self.logger.warning(f"Strategy profiles update failed: {error_context}")

    async def _update_strategy_profile_comprehensive(self, strategy_type: str, trade_data: Dict[str, Any],
                                                   performance_analysis: Dict[str, float],
                                                   behavioral_analysis: Dict[str, Any]):
        """Update comprehensive strategy profile with new data"""
        try:
            profile = self.strategy_profiles[strategy_type]
            
            # Extract trade metrics
            win_rate = performance_analysis.get('win_rate', 0.5)
            pnl = trade_data.get('pnl', 0)
            duration = trade_data.get('duration', 30)
            market_regime = trade_data.get('market_regime', 'unknown')
            
            # Update basic metrics
            profile['win_rate'].append(win_rate)
            profile['pnl_history'].append(pnl)
            profile['duration'].append(duration)
            profile['trade_count'] += 1
            profile['last_updated'] = datetime.datetime.now().isoformat()
            
            # Update market regime performance
            profile['market_regime_performance'][market_regime].append(pnl)
            
            # Update behavioral fingerprint
            profile['behavioral_fingerprint'] = {
                'risk_preference': behavioral_analysis.get('risk_preference', 'moderate'),
                'timing_consistency': behavioral_analysis.get('timing_patterns', {}).get('timing_consistency', 'unknown'),
                'adaptation_behavior': behavioral_analysis.get('adaptation_behavior', 'stable_consistent'),
                'market_sensitivity': behavioral_analysis.get('market_sensitivity', {})
            }
            
            # Keep only recent data within performance window
            max_len = self.performance_window
            for key in ['win_rate', 'pnl_history', 'duration']:
                if len(profile[key]) > max_len:
                    profile[key] = profile[key][-max_len:]
            
            # Calculate enhanced profile scores
            await self._calculate_profile_scores_comprehensive(profile, performance_analysis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "profile_update_comprehensive")
            self.logger.warning(f"Comprehensive profile update failed: {error_context}")

    async def _calculate_profile_scores_comprehensive(self, profile: Dict[str, Any], 
                                                    performance_analysis: Dict[str, float]):
        """Calculate comprehensive performance scores for strategy profile"""
        try:
            if not profile['pnl_history']:
                return
            
            # Performance score (risk-adjusted average P&L)
            avg_pnl = np.mean(profile['pnl_history'])
            pnl_volatility = np.std(profile['pnl_history']) if len(profile['pnl_history']) > 1 else 1
            profile['performance_score'] = avg_pnl / (pnl_volatility + 1e-6)
            
            # Consistency score (based on multiple factors)
            if len(profile['pnl_history']) > 1:
                # P&L consistency
                pnl_consistency = 1.0 / (1.0 + pnl_volatility / (abs(avg_pnl) + 1e-6))
                
                # Duration consistency
                if len(profile['duration']) > 1:
                    duration_std = np.std(profile['duration'])
                    duration_mean = np.mean(profile['duration'])
                    duration_consistency = 1.0 / (1.0 + duration_std / (duration_mean + 1e-6))
                else:
                    duration_consistency = 0.5
                
                # Win rate stability
                if len(profile['win_rate']) > 3:
                    win_rate_std = np.std(profile['win_rate'])
                    win_rate_consistency = 1.0 / (1.0 + win_rate_std * 10)
                else:
                    win_rate_consistency = 0.5
                
                # Combined consistency score
                profile['consistency_score'] = (
                    0.4 * pnl_consistency +
                    0.3 * duration_consistency + 
                    0.3 * win_rate_consistency
                )
            else:
                profile['consistency_score'] = 0.5
            
            # Adaptation score (how much the strategy has evolved)
            if len(profile['pnl_history']) >= 5:
                recent_performance = np.mean(profile['pnl_history'][-3:])
                historical_performance = np.mean(profile['pnl_history'][:-3])
                
                adaptation_magnitude = abs(recent_performance - historical_performance)
                profile['adaptation_score'] = min(1.0, adaptation_magnitude / 50.0)
            else:
                profile['adaptation_score'] = 0.0
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "profile_scores_comprehensive")
            self.logger.warning(f"Comprehensive profile score calculation failed: {error_context}")

    async def _generate_adaptation_insights_intelligent(self, strategy_analysis: Dict[str, Any], 
                                                      market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent adaptation insights"""
        try:
            adaptation_analysis = strategy_analysis.get('adaptation_analysis', {})
            behavioral_analysis = strategy_analysis.get('behavioral_analysis', {})
            performance_analysis = strategy_analysis.get('performance_analysis', {})
            trend_analysis = strategy_analysis.get('trend_analysis', {})
            
            insights = {
                'immediate_actions': [],
                'strategic_recommendations': [],
                'risk_mitigation': [],
                'performance_optimization': [],
                'behavioral_adjustments': [],
                'market_adaptation': [],
                'confidence_assessment': 0.5
            }
            
            # Generate immediate actions for urgent issues
            if adaptation_analysis.get('urgency_level') in ['critical', 'high']:
                insights['immediate_actions'] = adaptation_analysis.get('recommended_actions', [])
            
            # Strategic recommendations based on trend analysis
            if trend_analysis.get('predicted_direction') == 'negative':
                insights['strategic_recommendations'].append(
                    "Consider defensive positioning due to negative trend prediction"
                )
            elif trend_analysis.get('predicted_direction') == 'positive':
                insights['strategic_recommendations'].append(
                    "Consider increasing exposure to capitalize on positive trend"
                )
            
            # Risk mitigation based on behavioral analysis
            risk_preference = behavioral_analysis.get('risk_preference', 'moderate')
            if risk_preference == 'high_risk_tolerance':
                insights['risk_mitigation'].append(
                    "Implement position sizing limits to control risk exposure"
                )
            
            # Performance optimization insights
            performance_momentum = performance_analysis.get('performance_momentum', 0.0)
            if abs(performance_momentum) > 0.3:
                insights['performance_optimization'].append(
                    f"Strong performance momentum detected - consider {'enhancing' if performance_momentum > 0 else 'reviewing'} current approach"
                )
            
            # Behavioral adjustments
            trading_style = behavioral_analysis.get('trading_style', '')
            if trading_style in ['high_frequency_scalping', 'high_risk_aggressive']:
                insights['behavioral_adjustments'].append(
                    "Monitor for overtrading and ensure adequate risk controls"
                )
            
            # Market adaptation recommendations
            market_regime = market_data.get('market_regime', 'unknown')
            market_sensitivity = behavioral_analysis.get('market_sensitivity', {})
            regime_sensitivity = market_sensitivity.get('regime_sensitivity', 'moderate')
            
            if regime_sensitivity == 'high' and market_regime != 'stable':
                insights['market_adaptation'].append(
                    f"High sensitivity to {market_regime} conditions detected - consider regime-specific adjustments"
                )
            
            # Calculate overall confidence
            confidence_factors = [
                adaptation_analysis.get('confidence_level', 0.5),
                trend_analysis.get('confidence_interval', 0.5),
                min(1.0, len(self._records) / 10.0)  # Data sufficiency
            ]
            insights['confidence_assessment'] = np.mean(confidence_factors)
            
            return insights
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "adaptation_insights")
            self.logger.error(f"Adaptation insights generation failed: {error_context}")
            return {'error': str(error_context)}

    async def _generate_comprehensive_introspection_thesis(self, strategy_analysis: Dict[str, Any], 
                                                         adaptation_insights: Dict[str, Any]) -> str:
        """Generate comprehensive thesis explaining all introspection decisions"""
        try:
            thesis_parts = []
            
            # Executive Summary
            behavioral_analysis = strategy_analysis.get('behavioral_analysis', {})
            performance_analysis = strategy_analysis.get('performance_analysis', {})
            adaptation_analysis = strategy_analysis.get('adaptation_analysis', {})
            
            trading_style = behavioral_analysis.get('trading_style', 'unknown')
            adaptation_needed = adaptation_analysis.get('adaptation_needed', False)
            thesis_parts.append(
                f"STRATEGY ANALYSIS: {trading_style.replace('_', ' ').title()} pattern with {'adaptation required' if adaptation_needed else 'stable performance'}"
            )
            
            # Performance Assessment
            win_rate = performance_analysis.get('win_rate', 0.5)
            profit_factor = performance_analysis.get('profit_factor', 1.0)
            max_drawdown = performance_analysis.get('max_drawdown', 0.0)
            thesis_parts.append(
                f"PERFORMANCE: {win_rate:.1%} win rate, {profit_factor:.2f} profit factor, {max_drawdown:.1%} max drawdown"
            )
            
            # Behavioral Insights
            risk_preference = behavioral_analysis.get('risk_preference', 'unknown')
            timing_patterns = behavioral_analysis.get('timing_patterns', {})
            timing_consistency = timing_patterns.get('timing_consistency', 'unknown')
            thesis_parts.append(
                f"BEHAVIOR: {risk_preference.replace('_', ' ')} risk profile with {timing_consistency.replace('_', ' ')} timing"
            )
            
            # Adaptation Recommendations
            immediate_actions = adaptation_insights.get('immediate_actions', [])
            strategic_recommendations = adaptation_insights.get('strategic_recommendations', [])
            if immediate_actions:
                thesis_parts.append(
                    f"IMMEDIATE ACTIONS: {len(immediate_actions)} urgent recommendations for performance improvement"
                )
            if strategic_recommendations:
                thesis_parts.append(
                    f"STRATEGIC GUIDANCE: {len(strategic_recommendations)} long-term optimization opportunities identified"
                )
            
            # Market Adaptation
            market_adaptation = adaptation_insights.get('market_adaptation', [])
            if market_adaptation:
                thesis_parts.append(
                    f"MARKET ADAPTATION: Context-specific adjustments recommended for current market conditions"
                )
            
            # Confidence Assessment
            confidence = adaptation_insights.get('confidence_assessment', 0.5)
            thesis_parts.append(
                f"ANALYSIS CONFIDENCE: {confidence:.1%} based on data sufficiency and pattern consistency"
            )
            
            # System Status
            total_analyzed = self.introspection_metrics['total_strategies_analyzed']
            significant_adaptations = self.introspection_metrics['significant_adaptations']
            thesis_parts.append(
                f"SYSTEM STATUS: {total_analyzed} strategies analyzed, {significant_adaptations} adaptations identified"
            )
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Introspection thesis generation failed: {error_context}"

    def _get_performance_insights(self) -> Dict[str, Any]:
        """Get comprehensive performance insights"""
        try:
            insights = {
                'current_analysis': self.current_analysis.copy(),
                'strategy_evolution': [],
                'performance_trends': {},
                'risk_assessment': {},
                'adaptation_history': list(self.adaptation_history)[-10:]  # Last 10
            }
            
            # Performance trends
            if self._records:
                recent_records = list(self._records)[-10:]
                pnls = [r.get('pnl', 0) for r in recent_records]
                
                if len(pnls) >= 3:
                    insights['performance_trends'] = {
                        'trend_direction': 'improving' if np.mean(pnls[-3:]) > np.mean(pnls[:-3]) else 'declining',
                        'trend_strength': abs(np.mean(pnls[-3:]) - np.mean(pnls[:-3])),
                        'volatility': np.std(pnls),
                        'consistency': len([p for p in pnls if p > 0]) / len(pnls)
                    }
            
            # Risk assessment
            current_analysis = self.current_analysis
            insights['risk_assessment'] = {
                'risk_level': current_analysis.get('risk_assessment', 'moderate'),
                'adaptation_urgency': current_analysis.get('adaptation_needed', False),
                'confidence_level': current_analysis.get('confidence_level', 0.5)
            }
            
            return insights
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_insights")
            return {'error': str(error_context)}

    def _get_strategy_profiles_summary(self) -> Dict[str, Any]:
        """Get summary of all strategy profiles"""
        try:
            summary = {}
            
            for strategy_type, profile in self.strategy_profiles.items():
                if profile['trade_count'] > 0:
                    summary[strategy_type] = {
                        'trade_count': profile['trade_count'],
                        'performance_score': profile.get('performance_score', 0.0),
                        'consistency_score': profile.get('consistency_score', 0.5),
                        'adaptation_score': profile.get('adaptation_score', 0.0),
                        'last_updated': profile.get('last_updated'),
                        'behavioral_fingerprint': profile.get('behavioral_fingerprint', {}),
                        'recent_performance': np.mean(profile['pnl_history'][-5:]) if len(profile['pnl_history']) >= 5 else 0.0
                    }
            
            return summary
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "profiles_summary")
            return {'error': str(error_context)}

    def _get_behavioral_patterns(self) -> Dict[str, Any]:
        """Get comprehensive behavioral patterns analysis"""
        try:
            patterns = {
                'dominant_patterns': {},
                'risk_behaviors': {},
                'timing_behaviors': {},
                'adaptation_behaviors': {},
                'market_responses': {}
            }
            
            # Analyze dominant patterns across all profiles
            all_fingerprints = []
            for profile in self.strategy_profiles.values():
                fingerprint = profile.get('behavioral_fingerprint', {})
                if fingerprint:
                    all_fingerprints.append(fingerprint)
            
            if all_fingerprints:
                # Aggregate patterns
                risk_preferences = [fp.get('risk_preference', 'moderate') for fp in all_fingerprints]
                timing_consistencies = [fp.get('timing_consistency', 'unknown') for fp in all_fingerprints]
                adaptation_behaviors = [fp.get('adaptation_behavior', 'stable_consistent') for fp in all_fingerprints]
                
                from collections import Counter
                patterns['dominant_patterns'] = {
                    'most_common_risk_preference': Counter(risk_preferences).most_common(1)[0][0] if risk_preferences else 'unknown',
                    'most_common_timing': Counter(timing_consistencies).most_common(1)[0][0] if timing_consistencies else 'unknown',
                    'most_common_adaptation': Counter(adaptation_behaviors).most_common(1)[0][0] if adaptation_behaviors else 'unknown'
                }
            
            return patterns
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "behavioral_patterns")
            return {'error': str(error_context)}

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive introspection results"""
        try:
            # Core strategy analysis
            self.smart_bus.set('strategy_analysis', results['strategy_analysis'],
                             module='StrategyIntrospector', thesis=thesis)
            
            # Performance insights
            insights_thesis = f"Strategy introspection: {len(self._records)} records analyzed with {self.current_analysis.get('confidence_level', 0.5):.1%} confidence"
            self.smart_bus.set('performance_insights', results['performance_insights'],
                             module='StrategyIntrospector', thesis=insights_thesis)
            
            # Adaptation recommendations
            adaptation_thesis = f"Adaptation analysis: {'immediate action required' if self.current_analysis.get('adaptation_needed', False) else 'no urgent changes needed'}"
            self.smart_bus.set('adaptation_recommendations', results['adaptation_recommendations'],
                             module='StrategyIntrospector', thesis=adaptation_thesis)
            
            # Strategy profiles
            profiles_thesis = f"Strategy profiles: {len(results['strategy_profiles'])} active strategy types tracked"
            self.smart_bus.set('strategy_profiles', results['strategy_profiles'],
                             module='StrategyIntrospector', thesis=profiles_thesis)
            
            # Introspection metrics
            metrics_thesis = f"Introspection metrics: {self.introspection_metrics['total_strategies_analyzed']} strategies analyzed"
            self.smart_bus.set('introspection_metrics', results['introspection_metrics'],
                             module='StrategyIntrospector', thesis=metrics_thesis)
            
            # Behavior patterns
            patterns_thesis = f"Behavioral patterns: {self.current_analysis.get('dominant_strategy_type', 'unknown')} trading style detected"
            self.smart_bus.set('behavior_patterns', results['behavior_patterns'],
                             module='StrategyIntrospector', thesis=patterns_thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # ERROR HANDLING AND RECOVERY
    # ═══════════════════════════════════════════════════════════════════

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "StrategyIntrospector")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="Strategy Introspector disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('StrategyIntrospector', 'process_time', processing_time, False)
        
        return {
            'strategy_analysis': {'error': str(error_context)},
            'performance_insights': {'error': str(error_context)},
            'adaptation_recommendations': {'error': str(error_context)},
            'strategy_profiles': {'error': str(error_context)},
            'introspection_metrics': {'error': str(error_context)},
            'behavior_patterns': {'error': str(error_context)},
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def _get_safe_market_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when market data retrieval fails"""
        return {
            'recent_trades': [],
            'module_data': {},
            'risk_data': {},
            'market_regime': 'unknown',
            'volatility_data': {},
            'trading_performance': {},
            'strategy_weights': {},
            'market_context': {}
        }

    def _get_safe_analysis_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when analysis fails"""
        return {
            'strategy_context': {'timestamp': datetime.datetime.now().isoformat()},
            'performance_analysis': {k: v for k, v in self._baseline_metrics.items()},
            'behavioral_analysis': {'trading_style': 'unknown', 'risk_preference': 'moderate'},
            'adaptation_analysis': {'adaptation_needed': False, 'confidence_level': 0.5},
            'trend_analysis': {'short_term_trend': 'unknown', 'predicted_direction': 'neutral'},
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'error': 'analysis_failed'
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'strategy_analysis': {'status': 'disabled'},
            'performance_insights': {'status': 'disabled'},
            'adaptation_recommendations': {'status': 'disabled'},
            'strategy_profiles': {'status': 'disabled'},
            'introspection_metrics': {'status': 'disabled'},
            'behavior_patterns': {'status': 'disabled'},
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'}
        }

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'StrategyIntrospector',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'records_count': len(self._records),
            'profiles_count': len([p for p in self.strategy_profiles.values() if p['trade_count'] > 0]),
            'adaptations_detected': self.introspection_metrics['significant_adaptations'],
            'analysis_depth': self.analysis_depth,
            'confidence_level': self.current_analysis.get('confidence_level', 0.5),
            'adaptation_needed': self.current_analysis.get('adaptation_needed', False)
        }

    def record(self, theme: np.ndarray, win_rate: float, sl: float, tp: float, **kwargs) -> None:
        """Enhanced record method with comprehensive validation and analysis"""
        try:
            # Validate core inputs
            if not (0 <= win_rate <= 1):
                self.logger.warning(f"Invalid win_rate {win_rate}, clamping to [0,1]")
                win_rate = np.clip(win_rate, 0, 1)
            
            if sl <= 0:
                self.logger.warning(f"Invalid sl {sl}, using baseline {self._baseline_metrics['stop_loss']}")
                sl = self._baseline_metrics['stop_loss']
                
            if tp <= 0:
                self.logger.warning(f"Invalid tp {tp}, using baseline {self._baseline_metrics['take_profit']}")
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
            self._update_strategy_profile_sync(strategy_type, record)
            
            # Update analytics
            self._update_performance_analytics_sync(record)
            
            # Check for significant changes
            self._check_for_adaptations_sync(record)
            
            # Update metrics
            self.introspection_metrics['total_strategies_analyzed'] += 1
            
            self.logger.info(format_operator_message(
                icon="📊",
                message="Strategy recorded",
                type=strategy_type,
                win_rate=f"{win_rate:.1%}",
                risk_reward=f"{tp/sl:.2f}:1" if sl > 0 else "Invalid",
                pnl=f"€{pnl:+.2f}",
                total_records=len(self._records)
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "strategy_recording")
            self.logger.error(f"Strategy recording failed: {error_context}")

    def _update_strategy_profile_sync(self, strategy_type: str, record: Dict[str, Any]) -> None:
        """Synchronous version of strategy profile update"""
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
            self._calculate_profile_scores_sync(profile)
            
        except Exception as e:
            self.logger.warning(f"Strategy profile update failed: {e}")

    def _calculate_profile_scores_sync(self, profile: Dict[str, Any]) -> None:
        """Synchronous version of profile score calculation"""
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
            
            # Adaptation score
            if len(profile['win_rate']) >= 3:
                recent_metrics = np.array([
                    profile['win_rate'][-1],
                    profile['risk_reward'][-1],
                    profile['duration'][-1] / 100.0
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
            self.logger.warning(f"Profile score calculation failed: {e}")

    def _update_performance_analytics_sync(self, record: Dict[str, Any]) -> None:
        """Synchronous version of performance analytics update"""
        try:
            timestamp = record['timestamp']
            
            # Track performance by market regime
            regime = record.get('market_regime', 'unknown')
            self.performance_analytics[f'regime_{regime}'].append(record['pnl'])
            
            # Track performance by strategy type
            strategy_type = record.get('strategy_type', 'balanced')
            self.performance_analytics[f'strategy_{strategy_type}'].append(record['pnl'])
            
            # Track evolution metrics
            self.performance_analytics['risk_reward_evolution'].append({
                'timestamp': timestamp,
                'ratio': record['risk_reward_ratio'],
                'pnl': record['pnl']
            })
            
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
            self.logger.warning(f"Performance analytics update failed: {e}")

    def _check_for_adaptations_sync(self, record: Dict[str, Any]) -> None:
        """Synchronous version of adaptation checking"""
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
                
                self.logger.info(format_operator_message(
                    icon="🔄",
                    message="Strategy adaptation detected",
                    adaptations="; ".join(adaptations_detected[:2]),
                    performance_impact=f"€{current['pnl']:+.2f}"
                ))
                
        except Exception as e:
            self.logger.warning(f"Adaptation check failed: {e}")

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
                
                return baseline
            
            # Calculate comprehensive profile from records
            records_array = np.array([
                [r['win_rate'], r['stop_loss'], r['take_profit'], 
                 r['risk_reward_ratio'], r['duration'], r['pnl']] 
                for r in self._records
            ], dtype=np.float32)
            
            # Validate array
            if np.any(~np.isfinite(records_array)):
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
                profile = np.nan_to_num(profile, nan=0.5)
            
            return profile
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "profile_generation")
            self.logger.error(f"Profile generation failed: {error_context}")
            return np.array([
                self._baseline_metrics['win_rate'],
                self._baseline_metrics['stop_loss'],
                self._baseline_metrics['take_profit'],
                0.0, 0.0, 0.5, 0.5, 0.0
            ], dtype=np.float32)

    def get_observation_components(self) -> np.ndarray:
        """Return comprehensive strategy observation components"""
        return self.profile()

    def get_introspection_report(self) -> str:
        """Generate comprehensive strategy introspection report"""
        try:
            # Current analysis summary
            current_analysis = self.current_analysis
            strategy_type = current_analysis.get('dominant_strategy_type', 'unknown')
            performance_trend = current_analysis.get('performance_trend', 'unknown')
            confidence = current_analysis.get('confidence_level', 0.0)
            adaptation_needed = current_analysis.get('adaptation_needed', False)
            
            # Performance metrics from current analysis
            behavioral_patterns = current_analysis.get('behavioral_patterns', {})
            performance_analysis = behavioral_patterns.get('performance_analysis', {})
            
            # Strategy profiles summary
            profiles_summary = ""
            for strategy, profile in self.strategy_profiles.items():
                if profile['trade_count'] > 0:
                    performance_score = profile.get('performance_score', 0)
                    consistency_score = profile.get('consistency_score', 0)
                    status = "🟢" if performance_score > 10 else "🔴" if performance_score < -10 else "🟡"
                    profiles_summary += f"  • {strategy.replace('_', ' ').title()}: {profile['trade_count']} trades, Performance={performance_score:+.1f}, Consistency={consistency_score:.2f} {status}\n"
            
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
            
            # Performance metrics with safe defaults
            win_rate = performance_analysis.get('win_rate', 0.5)
            profit_factor = performance_analysis.get('profit_factor', 1.0)
            max_drawdown = performance_analysis.get('max_drawdown', 0.0)
            sharpe_ratio = performance_analysis.get('sharpe_ratio', 0.0)
            consistency_score = performance_analysis.get('consistency_score', 0.5)
            
            return f"""
🔍 STRATEGY INTROSPECTOR COMPREHENSIVE REPORT
═══════════════════════════════════════════════════════════════
📊 Current Analysis:
• Dominant Strategy: {strategy_type.replace('_', ' ').title()}
• Performance Trend: {performance_trend.replace('_', ' ').title()}
• Analysis Confidence: {confidence:.1%}
• Adaptation Needed: {'✅ Yes' if adaptation_needed else '❌ No'}
• Risk Assessment: {current_analysis.get('risk_assessment', 'Moderate').title()}

📈 Performance Metrics:
• Win Rate: {win_rate:.1%}
• Profit Factor: {profit_factor:.2f}
• Max Drawdown: {max_drawdown:.1%}
• Sharpe Ratio: {sharpe_ratio:.2f}
• Consistency Score: {consistency_score:.2f}

🎯 Strategy Profiles:
{profiles_summary if profiles_summary else '  📭 No strategy profiles available yet'}

🔄 Recent Adaptations:
{recent_adaptations if recent_adaptations else '  📭 No recent adaptations detected'}

💡 Current Recommendations:
{recommendations_str if recommendations_str else '  ✅ No specific recommendations - continue current approach'}

🧠 Behavioral Insights:
• Trading Style: {behavioral_patterns.get('trading_style', 'Unknown').replace('_', ' ').title()}
• Risk Preference: {behavioral_patterns.get('risk_preference', 'Unknown').replace('_', ' ').title()}
• Timing Consistency: {behavioral_patterns.get('timing_patterns', {}).get('timing_consistency', 'Unknown').replace('_', ' ').title()}
• Adaptation Behavior: {behavioral_patterns.get('adaptation_behavior', 'Unknown').replace('_', ' ').title()}

📊 Analytics Summary:
• Total Strategies Analyzed: {self.introspection_metrics['total_strategies_analyzed']}
• Significant Adaptations: {self.introspection_metrics['significant_adaptations']}
• Performance Improvements: {self.introspection_metrics['performance_improvements']}
• Performance Degradations: {self.introspection_metrics['performance_degradations']}
• Analysis Depth: {self.analysis_depth.title()}
• Records Maintained: {len(self._records)}/{self.history_len}

🔧 System Status:
• Module Status: {'DISABLED' if self.is_disabled else 'OPERATIONAL'}
• Error Count: {self.error_count}/{self.circuit_breaker_threshold}
• Circuit Breaker: {'OPEN' if self.error_count >= self.circuit_breaker_threshold else 'CLOSED'}
• Intelligence Level: Advanced Behavioral Analysis
            """
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "introspection_report")
            return f"Introspection report generation failed: {error_context}"

    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT FOR HOT-RELOAD
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'StrategyIntrospector',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'history_len': self.history_len,
                'debug': self.debug,
                'analysis_depth': self.analysis_depth,
                'performance_window': self.performance_window,
                'adaptation_threshold': self.adaptation_threshold
            },
            'introspection_state': {
                'records': list(self._records),
                'strategy_profiles': {k: v.copy() for k, v in self.strategy_profiles.items()},
                'performance_analytics': {k: list(v) for k, v in self.performance_analytics.items()},
                'adaptation_history': list(self.adaptation_history),
                'introspection_metrics': self.introspection_metrics.copy(),
                'current_analysis': self.current_analysis.copy(),
                'analysis_intelligence': self.analysis_intelligence.copy()
            },
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'baselines': self._baseline_metrics.copy(),
            'categories': self.strategy_categories.copy(),
            'performance_metrics': self._get_health_metrics()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            # Load configuration
            config = state.get("configuration", {})
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
            self.analysis_intelligence.update(introspection_state.get("analysis_intelligence", {}))
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            # Load baselines and categories if provided
            self._baseline_metrics.update(state.get("baselines", {}))
            self.strategy_categories.update(state.get("categories", {}))
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Strategy Introspector state restored",
                records=len(self._records),
                profiles=len(self.strategy_profiles),
                adaptations=len(self.adaptation_history),
                confidence=f"{self.current_analysis.get('confidence_level', 0.5):.1%}"
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for system monitoring"""
        return {
            'module_name': 'StrategyIntrospector',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'metrics': self._get_health_metrics(),
            'alerts': self._generate_health_alerts(),
            'recommendations': self._generate_health_recommendations()
        }

    def _generate_health_alerts(self) -> List[Dict[str, Any]]:
        """Generate health-related alerts"""
        alerts = []
        
        if self.is_disabled:
            alerts.append({
                'severity': 'critical',
                'message': 'StrategyIntrospector disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })
        
        if self.error_count > 2:
            alerts.append({
                'severity': 'warning',
                'message': f'High error count: {self.error_count}',
                'action': 'Monitor for recurring introspection issues'
            })
        
        if self.current_analysis.get('adaptation_needed', False):
            urgency = 'critical' if 'critical' in str(self.current_analysis.get('recommended_adjustments', [])) else 'warning'
            alerts.append({
                'severity': urgency,
                'message': 'Strategy adaptation recommended',
                'action': 'Review and implement suggested strategy adjustments'
            })
        
        if len(self._records) < 5:
            alerts.append({
                'severity': 'info',
                'message': f'Limited analysis data: {len(self._records)} records',
                'action': 'Continue trading to build comprehensive analysis baseline'
            })
        
        # Check for poor performance indicators
        behavioral_patterns = self.current_analysis.get('behavioral_patterns', {})
        if behavioral_patterns.get('risk_preference') == 'high_risk_tolerance':
            alerts.append({
                'severity': 'warning',
                'message': 'High-risk behavioral patterns detected',
                'action': 'Consider implementing additional risk controls'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        if self.is_disabled:
            recommendations.append("Restart StrategyIntrospector module after investigating errors")
        
        if len(self._records) < 10:
            recommendations.append("Insufficient analysis history - continue operations to build comprehensive baseline")
        
        if self.current_analysis.get('confidence_level', 0.5) < 0.4:
            recommendations.append("Low analysis confidence - increase data collection for more reliable insights")
        
        adaptations_needed = self.current_analysis.get('adaptation_needed', False)
        if adaptations_needed:
            recommendations.append("Strategy adaptation recommended - review and implement suggested adjustments")
        
        significant_adaptations = self.introspection_metrics.get('significant_adaptations', 0)
        if significant_adaptations > 10:
            recommendations.append("High adaptation frequency detected - ensure strategy stability")
        
        if not recommendations:
            recommendations.append("StrategyIntrospector operating within normal parameters")
        
        return recommendations

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API METHODS (for external use)
    # ═══════════════════════════════════════════════════════════════════

    def get_strategy_insights(self) -> Dict[str, Any]:
        """Get comprehensive strategy insights for external analysis"""
        try:
            insights = {
                'current_analysis': self.current_analysis.copy(),
                'dominant_patterns': {},
                'performance_summary': {},
                'adaptation_insights': {},
                'behavioral_fingerprint': {},
                'confidence_assessment': self.current_analysis.get('confidence_level', 0.5)
            }
            
            # Dominant patterns across all strategies
            if self.strategy_profiles:
                all_styles = []
                all_risks = []
                for profile in self.strategy_profiles.values():
                    if profile['trade_count'] > 0:
                        fingerprint = profile.get('behavioral_fingerprint', {})
                        if fingerprint:
                            all_styles.append(fingerprint.get('adaptation_behavior', 'unknown'))
                            all_risks.append(fingerprint.get('risk_preference', 'moderate'))
                
                if all_styles:
                    from collections import Counter
                    insights['dominant_patterns'] = {
                        'most_common_adaptation': Counter(all_styles).most_common(1)[0][0] if all_styles else 'unknown',
                        'most_common_risk': Counter(all_risks).most_common(1)[0][0] if all_risks else 'moderate'
                    }
            
            # Performance summary
            if self._records:
                recent_records = list(self._records)[-10:]
                pnls = [r.get('pnl', 0) for r in recent_records]
                
                insights['performance_summary'] = {
                    'recent_performance': np.mean(pnls) if pnls else 0,
                    'performance_trend': 'improving' if len(pnls) >= 3 and np.mean(pnls[-3:]) > np.mean(pnls[:-3]) else 'stable',
                    'consistency': len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0.5,
                    'total_records': len(self._records)
                }
            
            # Adaptation insights
            insights['adaptation_insights'] = {
                'adaptation_needed': self.current_analysis.get('adaptation_needed', False),
                'total_adaptations': self.introspection_metrics.get('significant_adaptations', 0),
                'recent_adaptations': len(self.adaptation_history),
                'adaptation_success_rate': self.introspection_metrics.get('adaptation_success_rate', 0.5)
            }
            
            # Current behavioral fingerprint
            behavioral_patterns = self.current_analysis.get('behavioral_patterns', {})
            insights['behavioral_fingerprint'] = {
                'trading_style': behavioral_patterns.get('trading_style', 'unknown'),
                'risk_preference': behavioral_patterns.get('risk_preference', 'moderate'),
                'timing_patterns': behavioral_patterns.get('timing_patterns', {}),
                'market_sensitivity': behavioral_patterns.get('market_sensitivity', {})
            }
            
            return insights
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "strategy_insights")
            return {'error': str(error_context)}

    def get_adaptation_recommendations(self) -> List[Dict[str, Any]]:
        """Get structured adaptation recommendations"""
        try:
            recommendations = []
            
            # Current recommendations from analysis
            current_recommendations = self.current_analysis.get('recommended_adjustments', [])
            
            for i, rec in enumerate(current_recommendations):
                recommendation = {
                    'id': f"rec_{i+1}",
                    'description': rec,
                    'priority': 'high' if self.current_analysis.get('adaptation_needed', False) else 'medium',
                    'category': self._categorize_recommendation(rec),
                    'confidence': self.current_analysis.get('confidence_level', 0.5),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                recommendations.append(recommendation)
            
            # Add general recommendations based on metrics
            if self.introspection_metrics.get('performance_degradations', 0) > 3:
                recommendations.append({
                    'id': 'general_perf',
                    'description': 'Review overall strategy performance due to recent degradations',
                    'priority': 'medium',
                    'category': 'performance',
                    'confidence': 0.7,
                    'timestamp': datetime.datetime.now().isoformat()
                })
            
            return recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "adaptation_recommendations")
            return [{'error': str(error_context)}]

    def _categorize_recommendation(self, recommendation: str) -> str:
        """Categorize recommendation type"""
        rec_lower = recommendation.lower()
        
        if any(word in rec_lower for word in ['entry', 'signal', 'filter']):
            return 'entry_strategy'
        elif any(word in rec_lower for word in ['exit', 'profit', 'stop']):
            return 'exit_strategy'
        elif any(word in rec_lower for word in ['risk', 'position', 'size']):
            return 'risk_management'
        elif any(word in rec_lower for word in ['timing', 'frequency', 'discipline']):
            return 'execution'
        elif any(word in rec_lower for word in ['adapt', 'flexible', 'responsive']):
            return 'adaptability'
        else:
            return 'general'

    def reset_analysis(self) -> bool:
        """Reset analysis system to initial state"""
        try:
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
                'prediction_success_rate': 0.0,
                'adaptation_success_rate': 0.0
            }
            
            # Reset current analysis
            self.current_analysis = {
                'dominant_strategy_type': 'balanced',
                'performance_trend': 'stable',
                'adaptation_needed': False,
                'recommended_adjustments': [],
                'confidence_level': 0.5,
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                'behavioral_patterns': {},
                'risk_assessment': 'moderate'
            }
            
            # Reset error state
            self.error_count = 0
            self.is_disabled = False
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Strategy Introspector reset completed",
                analysis_depth=self.analysis_depth,
                history_capacity=self.history_len
            ))
            
            return True
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "analysis_reset")
            self.logger.error(f"Analysis reset failed: {error_context}")
            return False

    def export_analysis_data(self) -> Dict[str, Any]:
        """Export comprehensive analysis data for external use"""
        try:
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.datetime.now().isoformat(),
                    'module_version': '3.0.0',
                    'analysis_depth': self.analysis_depth,
                    'records_count': len(self._records)
                },
                'configuration': {
                    'history_len': self.history_len,
                    'performance_window': self.performance_window,
                    'adaptation_threshold': self.adaptation_threshold,
                    'baseline_metrics': self._baseline_metrics,
                    'strategy_categories': self.strategy_categories
                },
                'current_state': {
                    'records': list(self._records),
                    'current_analysis': self.current_analysis,
                    'introspection_metrics': self.introspection_metrics
                },
                'historical_data': {
                    'adaptation_history': list(self.adaptation_history),
                    'performance_analytics': {k: list(v) for k, v in self.performance_analytics.items()}
                },
                'strategy_profiles': {k: v.copy() for k, v in self.strategy_profiles.items()},
                'system_metrics': {
                    'error_count': self.error_count,
                    'is_disabled': self.is_disabled,
                    'confidence_level': self.current_analysis.get('confidence_level', 0.5)
                }
            }
            
            return export_data
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "analysis_export")
            return {'error': str(error_context)}

    def get_behavioral_analysis(self) -> Dict[str, Any]:
        """Get detailed behavioral analysis"""
        try:
            behavioral_patterns = self.current_analysis.get('behavioral_patterns', {})
            
            analysis = {
                'trading_behavior': {
                    'style': behavioral_patterns.get('trading_style', 'unknown'),
                    'style_confidence': self.current_analysis.get('confidence_level', 0.5),
                    'consistency': behavioral_patterns.get('timing_patterns', {}).get('timing_consistency', 'unknown')
                },
                'risk_behavior': {
                    'preference': behavioral_patterns.get('risk_preference', 'moderate'),
                    'tolerance_level': self._assess_risk_tolerance(),
                    'adaptation_style': behavioral_patterns.get('adaptation_behavior', 'stable_consistent')
                },
                'market_interaction': {
                    'sensitivity': behavioral_patterns.get('market_sensitivity', {}),
                    'preferred_conditions': self._identify_preferred_conditions(),
                    'adaptation_speed': self._assess_adaptation_speed()
                },
                'performance_patterns': {
                    'consistency_score': self._calculate_overall_consistency(),
                    'improvement_trend': self._assess_improvement_trend(),
                    'stability_index': self._calculate_stability_index()
                }
            }
            
            return analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "behavioral_analysis")
            return {'error': str(error_context)}

    def _assess_risk_tolerance(self) -> str:
        """Assess overall risk tolerance level"""
        try:
            if not self._records:
                return 'unknown'
            
            recent_records = list(self._records)[-10:]
            risk_rewards = [r.get('risk_reward_ratio', 1.5) for r in recent_records]
            avg_risk_reward = np.mean(risk_rewards)
            
            if avg_risk_reward > 2.5:
                return 'high'
            elif avg_risk_reward > 1.8:
                return 'moderate_high'
            elif avg_risk_reward > 1.2:
                return 'moderate'
            else:
                return 'conservative'
                
        except Exception:
            return 'unknown'

    def _identify_preferred_conditions(self) -> List[str]:
        """Identify preferred market conditions"""
        try:
            if not self._records:
                return []
            
            # Analyze performance by market regime
            regime_performance = defaultdict(list)
            for record in self._records:
                regime = record.get('market_regime', 'unknown')
                pnl = record.get('pnl', 0)
                regime_performance[regime].append(pnl)
            
            # Find regimes with positive average performance
            preferred = []
            for regime, pnls in regime_performance.items():
                if len(pnls) >= 3 and np.mean(pnls) > 5:  # At least 3 trades and positive avg
                    preferred.append(regime)
            
            return preferred
            
        except Exception:
            return []

    def _assess_adaptation_speed(self) -> str:
        """Assess how quickly the strategy adapts"""
        try:
            if len(self.adaptation_history) < 3:
                return 'unknown'
            
            # Calculate time between adaptations
            adaptation_times = []
            for i in range(1, len(self.adaptation_history)):
                try:
                    t1 = datetime.datetime.fromisoformat(self.adaptation_history[i-1]['timestamp'])
                    t2 = datetime.datetime.fromisoformat(self.adaptation_history[i]['timestamp'])
                    gap = (t2 - t1).total_seconds() / 3600  # hours
                    adaptation_times.append(gap)
                except:
                    continue
            
            if adaptation_times:
                avg_gap = np.mean(adaptation_times)
                if avg_gap < 1:
                    return 'very_fast'
                elif avg_gap < 6:
                    return 'fast'
                elif avg_gap < 24:
                    return 'moderate'
                else:
                    return 'slow'
            
            return 'unknown'
            
        except Exception:
            return 'unknown'

    def _calculate_overall_consistency(self) -> float:
        """Calculate overall performance consistency"""
        try:
            if not self._records:
                return 0.5
            
            pnls = [r.get('pnl', 0) for r in self._records]
            if len(pnls) < 3:
                return 0.5
            
            # Calculate consistency as inverse of coefficient of variation
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            
            if abs(mean_pnl) < 1e-6:
                return 0.5
            
            cv = std_pnl / abs(mean_pnl)
            consistency = 1.0 / (1.0 + cv)
            
            return float(min(1.0, max(0.0, consistency)))
            
        except Exception:
            return 0.5

    def _assess_improvement_trend(self) -> str:
        """Assess overall improvement trend"""
        try:
            if len(self._records) < 6:
                return 'insufficient_data'
            
            pnls = [r.get('pnl', 0) for r in self._records]
            
            # Compare recent vs older performance
            recent_avg = np.mean(pnls[-3:])
            older_avg = np.mean(pnls[-6:-3])
            
            improvement = (recent_avg - older_avg) / (abs(older_avg) + 1e-6)
            
            if improvement > 0.2:
                return 'strong_improvement'
            elif improvement > 0.05:
                return 'gradual_improvement'
            elif improvement > -0.05:
                return 'stable'
            elif improvement > -0.2:
                return 'gradual_decline'
            else:
                return 'strong_decline'
                
        except Exception:
            return 'unknown'

    def _calculate_stability_index(self) -> float:
        """Calculate strategy stability index"""
        try:
            if len(self._records) < 5:
                return 0.5
            
            # Calculate stability based on multiple factors
            factors = []
            
            # Win rate stability
            win_rates = []
            records_list = list(self._records)
            for i in range(len(records_list) - 2):
                recent_records = records_list[i:i+3]
                wins = sum(1 for r in recent_records if r.get('pnl', 0) > 0)
                win_rates.append(wins / 3)
            
            if win_rates:
                wr_stability = 1.0 - np.std(win_rates)
                factors.append(max(0.0, wr_stability))
            
            # Risk-reward stability
            risk_rewards = [r.get('risk_reward_ratio', 1.5) for r in self._records]
            if len(risk_rewards) > 1:
                rr_cv = np.std(risk_rewards) / (np.mean(risk_rewards) + 1e-6)
                rr_stability = 1.0 / (1.0 + rr_cv)
                factors.append(rr_stability)
            
            # Duration stability
            durations = [r.get('duration', 30) for r in self._records]
            if len(durations) > 1:
                dur_cv = np.std(durations) / (np.mean(durations) + 1e-6)
                dur_stability = 1.0 / (1.0 + dur_cv)
                factors.append(dur_stability)
            
            return float(np.mean(factors)) if factors else 0.5
            
        except Exception:
            return 0.5

    def __str__(self) -> str:
        """String representation of the introspector"""
        return f"StrategyIntrospector(records={len(self._records)}, profiles={len(self.strategy_profiles)}, confidence={self.current_analysis.get('confidence_level', 0.5):.1%})"

    def __repr__(self) -> str:
        """Detailed representation of the introspector"""
        return (f"StrategyIntrospector(records={len(self._records)}, "
                f"profiles={len([p for p in self.strategy_profiles.values() if p['trade_count'] > 0])}, "
                f"adaptations={len(self.adaptation_history)}, "
                f"confidence={self.current_analysis.get('confidence_level', 0.5):.1%}, "
                f"style='{self.current_analysis.get('dominant_strategy_type', 'unknown')}')")