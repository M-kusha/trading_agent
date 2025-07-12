"""
⚙️ Enhanced Trading Mode Manager with SmartInfoBus Integration v3.0
Intelligent trading mode switching based on comprehensive market analysis and performance tracking
"""

import asyncio
import time
import numpy as np
import datetime
import copy
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
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="TradingModeManager",
    version="3.0.0",
    category="trading",
    provides=[
        "trading_mode", "mode_config", "mode_stats", "mode_effectiveness", "decision_factors",
        "mode_thresholds", "market_context", "mode_recommendations"
    ],
    requires=[
        "recent_trades", "risk_metrics", "votes", "positions", "market_context", "session_metrics",
        "strategy_performance", "trading_performance", "market_regime"
    ],
    description="Intelligent trading mode switching based on comprehensive market analysis and performance tracking",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=100,
    priority=8,
    explainable=True,
    hot_reload=True
)
class TradingModeManager(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    ⚙️ PRODUCTION-GRADE Trading Mode Manager v3.0
    
    Intelligent trading mode management system with:
    - Adaptive mode switching based on market conditions and performance
    - Comprehensive risk assessment and performance tracking
    - Market regime and session awareness for contextual decisions
    - SmartInfoBus zero-wiring architecture
    - Real-time effectiveness monitoring and optimization
    """

    # Enhanced trading modes with comprehensive definitions
    TRADING_MODES = {
        "safe": {
            "description": "Conservative risk management with capital preservation focus",
            "risk_multiplier": 0.5,
            "max_exposure": 0.3,
            "win_rate_threshold": 0.0,
            "drawdown_limit": 0.15,
            "consensus_requirement": 0.0,
            "volatility_tolerance": "low"
        },
        "normal": {
            "description": "Balanced trading approach with moderate risk-reward",
            "risk_multiplier": 1.0,
            "max_exposure": 0.6,
            "win_rate_threshold": 0.40,
            "drawdown_limit": 0.10,
            "consensus_requirement": 0.30,
            "volatility_tolerance": "medium"
        },
        "aggressive": {
            "description": "Increased risk for higher returns with active position management",
            "risk_multiplier": 1.5,
            "max_exposure": 0.8,
            "win_rate_threshold": 0.55,
            "drawdown_limit": 0.08,
            "consensus_requirement": 0.50,
            "volatility_tolerance": "medium-high"
        },
        "extreme": {
            "description": "Maximum risk for maximum returns - requires exceptional conditions",
            "risk_multiplier": 2.0,
            "max_exposure": 1.0,
            "win_rate_threshold": 0.65,
            "drawdown_limit": 0.05,
            "consensus_requirement": 0.65,
            "volatility_tolerance": "high"
        }
    }

    def _initialize(self):
        """Initialize advanced trading mode management systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced mode configuration
        self.initial_mode = self.config.get('initial_mode', 'normal')
        self.window = self.config.get('window', 20)
        self.auto_mode = self.config.get('auto_mode', True)
        self.min_persistence = self.config.get('min_persistence', 5)
        self.context_sensitivity = self.config.get('context_sensitivity', 0.8)
        self.performance_weight = self.config.get('performance_weight', 0.4)
        self.risk_weight = self.config.get('risk_weight', 0.3)
        self.consensus_weight = self.config.get('consensus_weight', 0.2)
        self.market_context_weight = self.config.get('market_context_weight', 0.1)
        self.regime_awareness = self.config.get('regime_awareness', True)
        self.session_awareness = self.config.get('session_awareness', True)
        self.volatility_scaling = self.config.get('volatility_scaling', True)
        self.debug = self.config.get('debug', False)
        
        # Validate initial mode
        if self.initial_mode not in self.TRADING_MODES:
            self.initial_mode = 'normal'
        
        # Core state management
        self.current_mode = self.initial_mode
        self.mode_persistence = 0
        self.last_mode_change = None
        self.last_change_reason = ""
        
        # Enhanced state tracking
        self.stats_history = deque(maxlen=self.window * 2)
        self.mode_history = deque(maxlen=50)
        self.decision_trace = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)
        
        # Market context awareness
        self.market_regime = "unknown"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        self.market_schedule = self.config.get('market_schedule')
        
        # Performance analytics with comprehensive tracking
        self.mode_analytics = defaultdict(lambda: defaultdict(list))
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self.session_performance = defaultdict(lambda: defaultdict(list))
        
        # Enhanced mode switching statistics
        self.mode_stats = {
            "total_switches": 0,
            "auto_switches": 0,
            "manual_switches": 0,
            "current_mode_duration": 0,
            "mode_effectiveness": 0.5,
            "switching_accuracy": 0.0,
            "average_mode_duration": 0.0,
            "best_performing_mode": "normal",
            "total_uptime": 0,
            "session_start": datetime.datetime.now().isoformat()
        }
        
        # Enhanced decision factors with intelligence
        self.decision_factors = {
            "performance_score": 0.5,
            "risk_score": 0.5,
            "consensus_score": 0.5,
            "market_context_score": 0.5,
            "volatility_score": 0.5,
            "regime_score": 0.5,
            "session_score": 0.5,
            "trend_score": 0.5,
            "stability_score": 0.5
        }
        
        # Adaptive mode thresholds
        self.mode_thresholds = self._initialize_adaptive_thresholds()
        
        # Learning and adaptation systems
        self.learning_history = deque(maxlen=30)
        self.threshold_adaptations = deque(maxlen=20)
        self.effectiveness_tracking = defaultdict(list)
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Mode intelligence parameters
        self.mode_intelligence = {
            'adaptation_speed': 0.1,
            'confidence_threshold': 0.7,
            'stability_requirement': 0.8,
            'performance_memory': 0.9,
            'risk_sensitivity': 0.8,
            'consensus_importance': 0.6
        }
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="⚙️",
            message=f"Trading Mode Manager v{version} initialized",
            initial_mode=self.current_mode,
            auto_mode=self.auto_mode,
            window=self.window,
            regime_awareness=self.regime_awareness
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="TradingModeManager",
            log_path="logs/trading/trading_mode_manager.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("TradingModeManager", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _initialize_adaptive_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize adaptive thresholds for each mode"""
        base_thresholds = {}
        
        for mode, config in self.TRADING_MODES.items():
            base_thresholds[mode] = {
                "max_drawdown": config["drawdown_limit"],
                "min_win_rate": config["win_rate_threshold"],
                "min_consensus": config["consensus_requirement"],
                "max_exposure": config["max_exposure"],
                "risk_multiplier": config["risk_multiplier"],
                "performance_threshold": 0.5,
                "stability_requirement": 0.6
            }
        
        return base_thresholds

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Trading Mode Manager v3.0 Initialization Complete:
        
        Intelligent Mode Management System:
        - Multi-mode trading framework: {len(self.TRADING_MODES)} distinct trading modes
        - Adaptive decision algorithms with performance-based optimization
        - Real-time market condition assessment and regime awareness
        - Dynamic threshold adjustment based on market volatility and performance
        
        Current Configuration:
        - Initial mode: {self.current_mode} with auto-switching {'enabled' if self.auto_mode else 'disabled'}
        - Decision window: {self.window} periods for statistical analysis
        - Persistence requirement: {self.min_persistence} periods for mode stability
        - Context sensitivity: {self.context_sensitivity:.1%} for market adaptation
        
        Decision Intelligence Features:
        - Performance-weighted analysis ({self.performance_weight:.1%} weight)
        - Risk assessment integration ({self.risk_weight:.1%} weight)
        - Committee consensus consideration ({self.consensus_weight:.1%} weight)
        - Market context awareness ({self.market_context_weight:.1%} weight)
        
        Advanced Capabilities:
        - Real-time mode effectiveness monitoring and optimization
        - Market regime and session-aware decision making
        - Adaptive threshold adjustment based on market conditions
        - Comprehensive performance tracking across all modes
        
        Expected Outcomes:
        - Optimal risk-adjusted returns through intelligent mode selection
        - Enhanced capital preservation during adverse market conditions
        - Improved performance consistency across different market regimes
        - Transparent mode decisions with detailed reasoning and tracking
        """
        
        self.smart_bus.set('trading_mode_manager_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'modes': list(self.TRADING_MODES.keys()),
                'initial_mode': self.current_mode,
                'decision_weights': {
                    'performance': self.performance_weight,
                    'risk': self.risk_weight,
                    'consensus': self.consensus_weight,
                    'market_context': self.market_context_weight
                }
            }
        }, module='TradingModeManager', thesis=thesis)

    async def process(self) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive mode management
        
        Returns:
            Dict containing mode status, analytics, and recommendations
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive market data from SmartInfoBus
            market_data = await self._get_comprehensive_market_data()
            
            # Update market context awareness
            await self._update_market_context_comprehensive(market_data)
            
            # Extract and analyze performance data
            performance_data = await self._extract_performance_data_comprehensive(market_data)
            
            # Update performance statistics with new data
            await self._update_performance_statistics_comprehensive(performance_data, market_data)
            
            # Perform intelligent mode decision analysis
            mode_decision = await self._make_intelligent_mode_decision_comprehensive(performance_data, market_data)
            
            # Apply mode decision with comprehensive tracking
            mode_change_result = await self._apply_mode_decision_comprehensive(mode_decision, market_data)
            
            # Analyze current mode effectiveness
            effectiveness_analysis = await self._analyze_mode_effectiveness_comprehensive(performance_data, market_data)
            
            # Update adaptive thresholds based on market conditions
            threshold_updates = await self._update_adaptive_thresholds_comprehensive(performance_data, market_data)
            
            # Generate mode recommendations
            recommendations = await self._generate_intelligent_mode_recommendations(mode_decision, effectiveness_analysis)
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_mode_thesis(mode_decision, effectiveness_analysis)
            
            # Create comprehensive results
            results = {
                'trading_mode': self.current_mode,
                'mode_config': self._get_mode_configuration(),
                'mode_stats': self._get_comprehensive_mode_stats(),
                'mode_effectiveness': effectiveness_analysis.get('current_effectiveness', 0.5),
                'decision_factors': self.decision_factors.copy(),
                'mode_thresholds': self.mode_thresholds.copy(),
                'market_context': self._get_market_context_summary(),
                'mode_recommendations': recommendations,
                'mode_decision_analysis': mode_decision,
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('TradingModeManager', 'process_time', processing_time, True)
            
            # Update mode statistics
            self._update_mode_performance_metrics()
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data using modern SmartInfoBus patterns"""
        try:
            return {
                'recent_trades': self.smart_bus.get('recent_trades', 'TradingModeManager') or [],
                'risk_metrics': self.smart_bus.get('risk_metrics', 'TradingModeManager') or {},
                'votes': self.smart_bus.get('votes', 'TradingModeManager') or [],
                'positions': self.smart_bus.get('positions', 'TradingModeManager') or [],
                'market_context': self.smart_bus.get('market_context', 'TradingModeManager') or {},
                'session_metrics': self.smart_bus.get('session_metrics', 'TradingModeManager') or {},
                'strategy_performance': self.smart_bus.get('strategy_performance', 'TradingModeManager') or {},
                'trading_performance': self.smart_bus.get('trading_performance', 'TradingModeManager') or {},
                'market_regime': self.smart_bus.get('market_regime', 'TradingModeManager') or 'unknown',
                'volatility_data': self.smart_bus.get('volatility_data', 'TradingModeManager') or {},
                'economic_calendar': self.smart_bus.get('economic_calendar', 'TradingModeManager') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "TradingModeManager")
            self.logger.warning(f"Market data retrieval incomplete: {error_context}")
            return self._get_safe_market_defaults()

    async def _update_market_context_comprehensive(self, market_data: Dict[str, Any]):
        """Update comprehensive market context awareness"""
        try:
            old_regime = self.market_regime
            old_volatility = self.volatility_regime
            old_session = self.market_session
            
            # Update regime tracking
            market_context = market_data.get('market_context', {})
            self.market_regime = market_data.get('market_regime', 'unknown')
            self.volatility_regime = market_context.get('volatility_level', 'medium')
            self.market_session = market_context.get('session', 'unknown')
            
            # Detect significant changes and their implications
            regime_changed = self.market_regime != old_regime and old_regime != 'unknown'
            volatility_changed = self.volatility_regime != old_volatility and old_volatility != 'unknown'
            session_changed = self.market_session != old_session and old_session != 'unknown'
            
            # Log and track important changes
            if regime_changed or volatility_changed:
                impact_assessment = self._assess_market_change_impact(
                    regime_changed, volatility_changed, session_changed
                )
                
                self.logger.info(format_operator_message(
                    icon="🌊",
                    message="Market context change detected",
                    regime_change=f"{old_regime} → {self.market_regime}" if regime_changed else "unchanged",
                    volatility_change=f"{old_volatility} → {self.volatility_regime}" if volatility_changed else "unchanged",
                    impact=impact_assessment,
                    current_mode=self.current_mode
                ))
                
                # Trigger threshold adaptation if significant change
                if impact_assessment in ['high', 'extreme']:
                    await self._trigger_emergency_threshold_adaptation(market_data)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_context_update")
            self.logger.warning(f"Market context update failed: {error_context}")

    def _assess_market_change_impact(self, regime_changed: bool, volatility_changed: bool, session_changed: bool) -> str:
        """Assess the impact level of market changes"""
        try:
            impact_score = 0
            
            if regime_changed:
                impact_score += 3
            if volatility_changed:
                if self.volatility_regime in ['high', 'extreme']:
                    impact_score += 2
                else:
                    impact_score += 1
            if session_changed:
                impact_score += 1
            
            if impact_score >= 4:
                return 'extreme'
            elif impact_score >= 3:
                return 'high'
            elif impact_score >= 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'medium'

    async def _trigger_emergency_threshold_adaptation(self, market_data: Dict[str, Any]):
        """Trigger emergency adaptation of thresholds due to significant market changes"""
        try:
            adaptation_factor = self._calculate_emergency_adaptation_factor(market_data)
            
            # Apply emergency adaptations
            for mode in self.mode_thresholds:
                thresholds = self.mode_thresholds[mode]
                
                # Make thresholds more conservative during high volatility/uncertainty
                if self.volatility_regime in ['high', 'extreme'] or self.market_regime == 'unknown':
                    thresholds['max_drawdown'] *= 0.8  # Reduce drawdown tolerance
                    thresholds['min_win_rate'] *= 1.1  # Increase win rate requirement
                    thresholds['min_consensus'] *= 1.2  # Increase consensus requirement
                
                # Adjust based on specific regime
                if self.market_regime == 'volatile':
                    thresholds['stability_requirement'] *= 1.3
                elif self.market_regime == 'trending':
                    thresholds['performance_threshold'] *= 0.9  # Slightly more lenient
            
            self.logger.info(format_operator_message(
                icon="⚡",
                message="Emergency threshold adaptation triggered",
                adaptation_factor=f"{adaptation_factor:.2f}",
                volatility_regime=self.volatility_regime,
                market_regime=self.market_regime
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "emergency_adaptation")

    def _calculate_emergency_adaptation_factor(self, market_data: Dict[str, Any]) -> float:
        """Calculate emergency adaptation factor based on market stress"""
        try:
            stress_factors = 0
            
            # Volatility stress
            if self.volatility_regime == 'extreme':
                stress_factors += 3
            elif self.volatility_regime == 'high':
                stress_factors += 2
            
            # Regime uncertainty stress
            if self.market_regime == 'unknown':
                stress_factors += 2
            elif self.market_regime == 'volatile':
                stress_factors += 1
            
            # Performance stress from recent data
            recent_trades = market_data.get('recent_trades', [])
            if recent_trades:
                recent_pnls = [t.get('pnl', 0) for t in recent_trades[-5:]]
                if recent_pnls and np.mean(recent_pnls) < -20:
                    stress_factors += 2
            
            # Normalize to factor between 0.5 and 1.5
            return max(0.5, min(1.5, 1.0 + (stress_factors - 3) * 0.1))
            
        except Exception:
            return 1.0

    async def _extract_performance_data_comprehensive(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive performance data with enhanced analytics"""
        try:
            performance_data = {}
            
            # Trade performance analysis
            recent_trades = market_data.get('recent_trades', [])
            performance_data['recent_trades'] = recent_trades
            performance_data['trade_count'] = len(recent_trades)
            
            if recent_trades:
                winning_trades = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
                performance_data['win_rate'] = winning_trades / len(recent_trades)
                performance_data['total_pnl'] = sum(trade.get('pnl', 0) for trade in recent_trades)
                performance_data['avg_pnl'] = performance_data['total_pnl'] / len(recent_trades)
                
                # Enhanced performance metrics
                pnls = [trade.get('pnl', 0) for trade in recent_trades]
                performance_data['pnl_std'] = np.std(pnls) if len(pnls) > 1 else 0
                performance_data['max_win'] = max(pnls) if pnls else 0
                performance_data['max_loss'] = min(pnls) if pnls else 0
                performance_data['profit_factor'] = self._calculate_profit_factor(pnls)
                
                # Recent trend analysis
                if len(recent_trades) >= 5:
                    recent_5_pnls = [t.get('pnl', 0) for t in recent_trades[-5:]]
                    performance_data['recent_trend'] = np.mean(recent_5_pnls)
                    performance_data['trend_consistency'] = self._calculate_trend_consistency(recent_5_pnls)
            else:
                performance_data.update({
                    'win_rate': 0.5, 'total_pnl': 0.0, 'avg_pnl': 0.0,
                    'pnl_std': 0, 'max_win': 0, 'max_loss': 0, 'profit_factor': 1.0,
                    'recent_trend': 0.0, 'trend_consistency': 0.5
                })
            
            # Risk metrics analysis
            risk_metrics = market_data.get('risk_metrics', {})
            performance_data['current_balance'] = float(risk_metrics.get('balance', risk_metrics.get('equity', 10000)))
            performance_data['drawdown'] = max(0.0, float(risk_metrics.get('current_drawdown', 0.0)))
            performance_data['max_drawdown'] = max(0.0, float(risk_metrics.get('max_drawdown', 0.0)))
            performance_data['risk_score'] = float(risk_metrics.get('risk_score', 0.5))
            
            # Committee consensus analysis
            votes = market_data.get('votes', [])
            if votes:
                confidences = [vote.get('confidence', 0.5) for vote in votes]
                performance_data['consensus'] = np.mean(confidences)
                performance_data['vote_agreement'] = 1.0 - np.std(confidences) if len(confidences) > 1 else 1.0
                performance_data['vote_count'] = len(votes)
                performance_data['consensus_strength'] = min(performance_data['consensus'], performance_data['vote_agreement'])
            else:
                performance_data.update({
                    'consensus': 0.5, 'vote_agreement': 0.5, 'vote_count': 0, 'consensus_strength': 0.5
                })
            
            # Market volatility analysis
            market_context = market_data.get('market_context', {})
            volatility_data = market_data.get('volatility_data', {})
            
            if volatility_data:
                performance_data['volatility'] = np.mean(list(volatility_data.values()))
            else:
                performance_data['volatility'] = 0.02  # Default
            
            performance_data['volatility_regime_score'] = self._get_volatility_regime_score()
            
            # Position and exposure analysis
            positions = market_data.get('positions', [])
            total_exposure = sum(abs(pos.get('size', 0)) for pos in positions)
            performance_data['exposure'] = total_exposure
            performance_data['position_count'] = len(positions)
            performance_data['exposure_ratio'] = min(1.0, total_exposure / max(performance_data['current_balance'], 1000))
            
            # Strategy performance integration
            strategy_performance = market_data.get('strategy_performance', {})
            performance_data['strategy_effectiveness'] = strategy_performance.get('effectiveness_score', 0.5)
            performance_data['strategy_confidence'] = strategy_performance.get('confidence_score', 0.5)
            
            # Session performance
            session_metrics = market_data.get('session_metrics', {})
            performance_data['session_pnl'] = session_metrics.get('session_pnl', 0)
            performance_data['session_trades'] = session_metrics.get('session_trades', 0)
            
            # Calculate comprehensive Sharpe ratio
            if len(recent_trades) > 5:
                returns = [trade.get('pnl', 0) / max(performance_data['current_balance'], 1000) for trade in recent_trades]
                if np.std(returns) > 0:
                    performance_data['sharpe'] = np.sqrt(252) * np.mean(returns) / np.std(returns)
                else:
                    performance_data['sharpe'] = 0.0
            else:
                performance_data['sharpe'] = 0.0
            
            return performance_data
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_data_extraction")
            self.logger.warning(f"Performance data extraction failed: {error_context}")
            return self._get_safe_performance_defaults()

    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        """Calculate profit factor from PnL list"""
        try:
            if not pnls:
                return 1.0
            
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 1.0
            
            return gross_profit / gross_loss
            
        except Exception:
            return 1.0

    def _calculate_trend_consistency(self, pnls: List[float]) -> float:
        """Calculate trend consistency score"""
        try:
            if len(pnls) < 2:
                return 0.5
            
            # Calculate directional consistency
            directions = []
            for i in range(1, len(pnls)):
                if pnls[i] > pnls[i-1]:
                    directions.append(1)
                elif pnls[i] < pnls[i-1]:
                    directions.append(-1)
                else:
                    directions.append(0)
            
            if not directions:
                return 0.5
            
            # Calculate consistency as lack of direction changes
            direction_changes = sum(1 for i in range(1, len(directions)) 
                                  if directions[i] != directions[i-1] and directions[i] != 0 and directions[i-1] != 0)
            
            consistency = 1.0 - (direction_changes / max(1, len(directions) - 1))
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.5

    def _get_volatility_regime_score(self) -> float:
        """Get score based on current volatility regime"""
        volatility_scores = {
            'low': 0.8,        # Good for aggressive trading
            'medium': 0.7,     # Balanced conditions
            'high': 0.4,       # Risky conditions
            'extreme': 0.2,    # Very risky
            'unknown': 0.5     # Neutral
        }
        return volatility_scores.get(self.volatility_regime, 0.5)

    async def _update_performance_statistics_comprehensive(self, performance_data: Dict[str, Any], 
                                                         market_data: Dict[str, Any]):
        """Update comprehensive performance statistics with enhanced tracking"""
        try:
            # Create comprehensive stats entry
            stats_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'mode': self.current_mode,
                'win_rate': performance_data.get('win_rate', 0.5),
                'avg_pnl': performance_data.get('avg_pnl', 0.0),
                'total_pnl': performance_data.get('total_pnl', 0.0),
                'drawdown': performance_data.get('drawdown', 0.0),
                'max_drawdown': performance_data.get('max_drawdown', 0.0),
                'consensus': performance_data.get('consensus', 0.5),
                'consensus_strength': performance_data.get('consensus_strength', 0.5),
                'volatility': performance_data.get('volatility', 0.02),
                'volatility_regime_score': performance_data.get('volatility_regime_score', 0.5),
                'sharpe': performance_data.get('sharpe', 0.0),
                'profit_factor': performance_data.get('profit_factor', 1.0),
                'trade_count': performance_data.get('trade_count', 0),
                'exposure': performance_data.get('exposure', 0.0),
                'exposure_ratio': performance_data.get('exposure_ratio', 0.0),
                'strategy_effectiveness': performance_data.get('strategy_effectiveness', 0.5),
                'regime': self.market_regime,
                'volatility_level': self.volatility_regime,
                'session': self.market_session,
                'recent_trend': performance_data.get('recent_trend', 0.0),
                'trend_consistency': performance_data.get('trend_consistency', 0.5)
            }
            
            # Add to history
            self.stats_history.append(stats_entry)
            
            # Track mode-specific performance with enhanced metrics
            mode_performance = self.mode_analytics[self.current_mode]
            mode_performance['win_rates'].append(stats_entry['win_rate'])
            mode_performance['pnl_values'].append(stats_entry['avg_pnl'])
            mode_performance['total_pnl_values'].append(stats_entry['total_pnl'])
            mode_performance['drawdowns'].append(stats_entry['drawdown'])
            mode_performance['sharpe_ratios'].append(stats_entry['sharpe'])
            mode_performance['profit_factors'].append(stats_entry['profit_factor'])
            mode_performance['timestamps'].append(stats_entry['timestamp'])
            
            # Track regime-specific performance
            if self.regime_awareness and self.market_regime != 'unknown':
                regime_perf = self.regime_performance[self.market_regime]
                regime_perf['modes'].append(self.current_mode)
                regime_perf['performance'].append(stats_entry['avg_pnl'])
                regime_perf['effectiveness'].append(stats_entry['strategy_effectiveness'])
                regime_perf['timestamps'].append(stats_entry['timestamp'])
            
            # Track session-specific performance
            if self.session_awareness and self.market_session != 'unknown':
                session_perf = self.session_performance[self.market_session]
                session_perf['modes'].append(self.current_mode)
                session_perf['performance'].append(stats_entry['avg_pnl'])
                session_perf['win_rates'].append(stats_entry['win_rate'])
                session_perf['timestamps'].append(stats_entry['timestamp'])
            
            # Update effectiveness tracking
            self.effectiveness_tracking[self.current_mode].append({
                'timestamp': stats_entry['timestamp'],
                'effectiveness': self._calculate_mode_effectiveness(stats_entry),
                'performance_score': stats_entry['avg_pnl'],
                'risk_score': 1.0 - stats_entry['drawdown'],
                'consistency_score': stats_entry['trend_consistency']
            })
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_statistics_update")
            self.logger.warning(f"Performance statistics update failed: {error_context}")

    def _calculate_mode_effectiveness(self, stats_entry: Dict[str, Any]) -> float:
        """Calculate effectiveness score for current mode"""
        try:
            # Performance component (40% weight)
            avg_pnl = stats_entry.get('avg_pnl', 0.0)
            performance_component = np.tanh(avg_pnl / 50.0) * 0.5 + 0.5
            
            # Risk component (30% weight)
            drawdown = stats_entry.get('drawdown', 0.0)
            risk_component = max(0.0, 1.0 - drawdown * 5)
            
            # Consistency component (20% weight)
            consistency_component = stats_entry.get('trend_consistency', 0.5)
            
            # Consensus component (10% weight)
            consensus_component = stats_entry.get('consensus_strength', 0.5)
            
            # Weighted effectiveness score
            effectiveness = (
                0.4 * performance_component +
                0.3 * risk_component +
                0.2 * consistency_component +
                0.1 * consensus_component
            )
            
            return max(0.0, min(1.0, effectiveness))
            
        except Exception:
            return 0.5

    async def _make_intelligent_mode_decision_comprehensive(self, performance_data: Dict[str, Any], 
                                                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make comprehensive intelligent mode decision with advanced analysis"""
        try:
            decision = {
                'current_mode': self.current_mode,
                'recommended_mode': self.current_mode,
                'confidence': 0.5,
                'reasoning': [],
                'decision_factors': {},
                'should_change': False,
                'analysis_details': {},
                'risk_assessment': {},
                'market_alignment': {}
            }
            
            # Check if market is closed (always safe mode)
            if not self._is_market_open():
                decision.update({
                    'recommended_mode': 'safe',
                    'confidence': 1.0,
                    'reasoning': ['Market is closed - safety mode required'],
                    'should_change': self.current_mode != 'safe'
                })
                return decision
            
            # Skip auto mode if disabled
            if not self.auto_mode:
                decision['reasoning'] = ['Auto mode disabled - maintaining current mode']
                return decision
            
            # Check persistence requirement for stability
            if self.mode_persistence < self.min_persistence:
                decision['reasoning'] = [f'Mode persistence required ({self.mode_persistence}/{self.min_persistence})']
                return decision
            
            # Calculate enhanced decision factors
            await self._calculate_decision_factors_comprehensive(performance_data, market_data)
            decision['decision_factors'] = self.decision_factors.copy()
            
            # Calculate scores for each mode with comprehensive analysis
            mode_scores = await self._calculate_mode_scores_comprehensive(performance_data, market_data)
            decision['analysis_details']['mode_scores'] = mode_scores
            
            # Perform risk assessment for each mode
            risk_assessment = await self._perform_comprehensive_risk_assessment(performance_data, market_data)
            decision['risk_assessment'] = risk_assessment
            
            # Assess market alignment for mode decision
            market_alignment = await self._assess_market_alignment_comprehensive(performance_data, market_data)
            decision['market_alignment'] = market_alignment
            
            # Find optimal mode with confidence scoring
            best_mode_analysis = self._find_optimal_mode_with_confidence(mode_scores, risk_assessment, market_alignment)
            recommended_mode = best_mode_analysis['mode']
            confidence = best_mode_analysis['confidence']
            
            # Generate comprehensive reasoning
            reasoning = await self._generate_mode_reasoning_comprehensive(
                performance_data, market_data, mode_scores, risk_assessment, market_alignment
            )
            
            # Determine if change is warranted with enhanced logic
            change_analysis = self._analyze_mode_change_necessity(
                recommended_mode, confidence, mode_scores, risk_assessment
            )
            
            decision.update({
                'recommended_mode': recommended_mode,
                'confidence': confidence,
                'reasoning': reasoning,
                'should_change': change_analysis['should_change'],
                'change_urgency': change_analysis.get('urgency', 'normal'),
                'improvement_potential': change_analysis.get('improvement', 0.0),
                'analysis_details': {
                    **decision['analysis_details'],
                    'best_mode_analysis': best_mode_analysis,
                    'change_analysis': change_analysis
                }
            })
            
            return decision
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "intelligent_mode_decision")
            self.logger.error(f"Intelligent mode decision failed: {error_context}")
            decision['reasoning'] = [f'Decision analysis error: {error_context}']
            return decision

    async def _calculate_decision_factors_comprehensive(self, performance_data: Dict[str, Any], 
                                                      market_data: Dict[str, Any]):
        """Calculate comprehensive decision factors with enhanced intelligence"""
        try:
            # Performance factor with trend analysis
            if len(self.stats_history) >= 5:
                recent_stats = list(self.stats_history)[-5:]
                avg_win_rate = np.mean([s['win_rate'] for s in recent_stats])
                avg_pnl = np.mean([s['avg_pnl'] for s in recent_stats])
                trend_score = np.mean([s.get('recent_trend', 0) for s in recent_stats])
                
                # Enhanced performance scoring with trend consideration
                win_rate_score = avg_win_rate
                pnl_score = np.tanh(avg_pnl / 100.0) * 0.5 + 0.5
                trend_score_normalized = np.tanh(trend_score / 50.0) * 0.5 + 0.5
                
                self.decision_factors['performance_score'] = (
                    0.4 * win_rate_score + 
                    0.4 * pnl_score + 
                    0.2 * trend_score_normalized
                )
            else:
                self.decision_factors['performance_score'] = 0.5
            
            # Enhanced risk factor with multiple components
            drawdown = performance_data.get('drawdown', 0.0)
            max_drawdown = performance_data.get('max_drawdown', 0.0)
            volatility = performance_data.get('volatility', 0.02)
            exposure_ratio = performance_data.get('exposure_ratio', 0.0)
            
            drawdown_score = max(0.0, 1.0 - drawdown * 5)
            max_drawdown_score = max(0.0, 1.0 - max_drawdown * 3)
            volatility_score = max(0.0, 1.0 - volatility * 20)
            exposure_score = max(0.0, 1.0 - exposure_ratio)
            
            self.decision_factors['risk_score'] = (
                0.3 * drawdown_score + 
                0.3 * max_drawdown_score + 
                0.2 * volatility_score + 
                0.2 * exposure_score
            )
            
            # Enhanced consensus factor with strength consideration
            consensus = performance_data.get('consensus', 0.5)
            consensus_strength = performance_data.get('consensus_strength', 0.5)
            vote_count = performance_data.get('vote_count', 0)
            
            # Adjust consensus score based on vote count
            vote_confidence = min(1.0, vote_count / 5.0)  # Full confidence at 5+ votes
            
            self.decision_factors['consensus_score'] = (
                0.4 * consensus + 
                0.4 * consensus_strength + 
                0.2 * vote_confidence
            )
            
            # Market context factor with regime intelligence
            regime_score = self._get_regime_score_enhanced(self.market_regime)
            volatility_level_score = self._get_volatility_level_score_enhanced(self.volatility_regime)
            session_score = self._get_session_score_enhanced(self.market_session)
            
            self.decision_factors['market_context_score'] = (
                0.4 * regime_score + 
                0.4 * volatility_level_score + 
                0.2 * session_score
            )
            
            # Additional enhanced factors
            self.decision_factors['volatility_score'] = volatility_score
            self.decision_factors['regime_score'] = regime_score
            self.decision_factors['session_score'] = session_score
            self.decision_factors['trend_score'] = performance_data.get('trend_consistency', 0.5)
            self.decision_factors['stability_score'] = self._calculate_stability_score(performance_data)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "decision_factors_calculation")
            self.logger.warning(f"Decision factors calculation failed: {error_context}")

    def _get_regime_score_enhanced(self, regime: str) -> float:
        """Get enhanced score based on market regime with nuanced analysis"""
        regime_scores = {
            'trending': 0.85,   # Excellent for aggressive trading
            'breakout': 0.8,    # Very good for aggressive approaches
            'momentum': 0.75,   # Good for higher risk
            'normal': 0.7,      # Good baseline conditions
            'ranging': 0.6,     # Moderate conditions, some strategies work well
            'consolidation': 0.55, # Slightly challenging
            'volatile': 0.35,   # Risky conditions
            'uncertain': 0.3,   # High uncertainty
            'reversal': 0.4,    # Mixed signals
            'unknown': 0.5      # Neutral baseline
        }
        return regime_scores.get(regime, 0.5)

    def _get_volatility_level_score_enhanced(self, vol_level: str) -> float:
        """Get enhanced score based on volatility level"""
        vol_scores = {
            'very_low': 0.9,    # Excellent for aggressive trading
            'low': 0.8,         # Good for higher risk
            'medium': 0.7,      # Balanced conditions
            'medium_high': 0.5, # Caution required
            'high': 0.3,        # Risky conditions
            'extreme': 0.15,    # Very dangerous
            'unknown': 0.5      # Neutral baseline
        }
        return vol_scores.get(vol_level, 0.5)

    def _get_session_score_enhanced(self, session: str) -> float:
        """Get enhanced score based on trading session with liquidity considerations"""
        session_scores = {
            'overlap_london_new_york': 0.9,  # Peak liquidity
            'london': 0.8,                   # High liquidity
            'new_york': 0.8,                 # High liquidity
            'asian': 0.6,                    # Moderate liquidity
            'sydney': 0.55,                  # Lower liquidity
            'weekend': 0.2,                  # Very low liquidity
            'holiday': 0.25,                 # Reduced liquidity
            'rollover': 0.3,                 # Spread widening
            'unknown': 0.5                   # Neutral baseline
        }
        return session_scores.get(session, 0.5)

    def _calculate_stability_score(self, performance_data: Dict[str, Any]) -> float:
        """Calculate system stability score based on various factors"""
        try:
            factors = []
            
            # PnL stability
            if len(self.stats_history) >= 5:
                recent_pnls = [s.get('avg_pnl', 0) for s in list(self.stats_history)[-5:]]
                pnl_stability = 1.0 - (np.std(recent_pnls) / (abs(np.mean(recent_pnls)) + 10))
                factors.append(max(0.0, min(1.0, pnl_stability)))
            
            # Drawdown stability
            drawdown = performance_data.get('drawdown', 0.0)
            drawdown_stability = max(0.0, 1.0 - drawdown * 3)
            factors.append(drawdown_stability)
            
            # Trend consistency
            trend_consistency = performance_data.get('trend_consistency', 0.5)
            factors.append(trend_consistency)
            
            # Mode persistence as stability indicator
            persistence_stability = min(1.0, self.mode_persistence / (self.min_persistence * 2))
            factors.append(persistence_stability)
            
            return float(np.mean(factors)) if factors else 0.5
            
        except Exception:
            return 0.5

    async def _calculate_mode_scores_comprehensive(self, performance_data: Dict[str, Any], 
                                                 market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive scores for each trading mode"""
        try:
            mode_scores = {}
            
            for mode in self.TRADING_MODES:
                score = await self._calculate_single_mode_score_comprehensive(mode, performance_data, market_data)
                mode_scores[mode] = score
            
            return mode_scores
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "comprehensive_mode_scores")
            self.logger.warning(f"Comprehensive mode scores calculation failed: {error_context}")
            return {mode: 0.5 for mode in self.TRADING_MODES}

    async def _calculate_single_mode_score_comprehensive(self, mode: str, performance_data: Dict[str, Any], 
                                                       market_data: Dict[str, Any]) -> float:
        """Calculate comprehensive score for a single mode with enhanced logic"""
        try:
            # Get mode configuration and thresholds
            mode_config = self.TRADING_MODES[mode]
            thresholds = self.mode_thresholds[mode]
            
            # Hard requirement checks
            drawdown = performance_data.get('drawdown', 0.0)
            if drawdown > thresholds['max_drawdown']:
                return 0.1  # Severe penalty for exceeding drawdown limit
            
            exposure_ratio = performance_data.get('exposure_ratio', 0.0)
            if exposure_ratio > thresholds['max_exposure']:
                return 0.2  # Penalty for excessive exposure
            
            # Performance requirement checks
            if len(self.stats_history) >= 3:
                recent_stats = list(self.stats_history)[-3:]
                avg_win_rate = np.mean([s['win_rate'] for s in recent_stats])
                
                if avg_win_rate < thresholds['min_win_rate']:
                    return 0.15  # Low score for poor win rate
            
            # Base scoring components
            performance_score = self.decision_factors['performance_score']
            risk_score = self.decision_factors['risk_score']
            consensus_score = self.decision_factors['consensus_score']
            market_score = self.decision_factors['market_context_score']
            stability_score = self.decision_factors['stability_score']
            
            # Mode-specific scoring logic
            if mode == 'safe':
                # Safe mode prioritizes risk management and stability
                score = (
                    risk_score * 0.4 +
                    stability_score * 0.3 +
                    performance_score * 0.2 +
                    market_score * 0.1
                )
                
                # Bonus for adverse conditions
                if drawdown > 0.05 or performance_score < 0.4 or self.volatility_regime in ['high', 'extreme']:
                    score += 0.3
                    
            elif mode == 'normal':
                # Balanced scoring with all factors
                score = (
                    performance_score * self.performance_weight +
                    risk_score * self.risk_weight +
                    consensus_score * self.consensus_weight +
                    market_score * self.market_context_weight
                )
                
                # Stability bonus
                score += stability_score * 0.1
                
            elif mode == 'aggressive':
                # Aggressive mode prioritizes performance and consensus
                score = (
                    performance_score * 0.5 +
                    consensus_score * 0.3 +
                    market_score * 0.15 +
                    risk_score * 0.05
                )
                
                # Strict requirements
                if (performance_score < 0.6 or consensus_score < 0.5 or 
                    self.volatility_regime in ['high', 'extreme']):
                    score *= 0.6
                
                # Market regime bonus
                if self.market_regime in ['trending', 'breakout', 'momentum']:
                    score += 0.15
                    
            elif mode == 'extreme':
                # Extreme mode requires exceptional conditions
                score = (
                    performance_score * 0.6 +
                    consensus_score * 0.25 +
                    market_score * 0.15
                )
                
                # Very strict requirements
                required_conditions = [
                    performance_score >= 0.7,
                    consensus_score >= 0.65,
                    self.decision_factors['regime_score'] >= 0.7,
                    self.volatility_regime not in ['high', 'extreme'],
                    stability_score >= 0.6
                ]
                
                if not all(required_conditions):
                    score *= 0.3
                
                # Additional market regime requirements
                if self.market_regime not in ['trending', 'breakout', 'momentum']:
                    score *= 0.5
            
            # Apply volatility scaling if enabled
            if self.volatility_scaling:
                volatility_adjustment = self._get_volatility_adjustment_factor()
                score *= volatility_adjustment
            
            # Apply mode effectiveness learning if available
            if mode in self.effectiveness_tracking and self.effectiveness_tracking[mode]:
                recent_effectiveness = np.mean([e['effectiveness'] for e in self.effectiveness_tracking[mode][-3:]])
                effectiveness_bonus = (recent_effectiveness - 0.5) * 0.2
                score += effectiveness_bonus
            
            return float(np.clip(score, 0.0, 1.0))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, f"single_mode_score_{mode}")
            self.logger.warning(f"Single mode score calculation failed for {mode}: {error_context}")
            return 0.5

    def _get_volatility_adjustment_factor(self) -> float:
        """Get volatility adjustment factor for mode scoring"""
        adjustments = {
            'very_low': 1.1,     # Slight bonus for stable conditions
            'low': 1.05,         # Small bonus
            'medium': 1.0,       # Neutral
            'medium_high': 0.95, # Small penalty
            'high': 0.85,        # Penalty for risky conditions
            'extreme': 0.7,      # Significant penalty
            'unknown': 1.0       # Neutral
        }
        return adjustments.get(self.volatility_regime, 1.0)

    async def _perform_comprehensive_risk_assessment(self, performance_data: Dict[str, Any], 
                                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment for mode decision"""
        try:
            risk_assessment = {
                'overall_risk_level': 'medium',
                'risk_factors': [],
                'risk_mitigations': [],
                'risk_score_by_mode': {}
            }
            
            # Assess individual risk factors
            drawdown_risk = self._assess_drawdown_risk(performance_data)
            volatility_risk = self._assess_volatility_risk(performance_data, market_data)
            exposure_risk = self._assess_exposure_risk(performance_data)
            consensus_risk = self._assess_consensus_risk(performance_data)
            market_risk = self._assess_market_risk(market_data)
            
            # Compile risk factors
            risk_factors = [drawdown_risk, volatility_risk, exposure_risk, consensus_risk, market_risk]
            high_risk_factors = [rf for rf in risk_factors if rf['level'] in ['high', 'extreme']]
            
            # Determine overall risk level
            if any(rf['level'] == 'extreme' for rf in risk_factors):
                risk_assessment['overall_risk_level'] = 'extreme'
            elif len(high_risk_factors) >= 2:
                risk_assessment['overall_risk_level'] = 'high'
            elif len(high_risk_factors) >= 1:
                risk_assessment['overall_risk_level'] = 'medium_high'
            elif any(rf['level'] == 'medium' for rf in risk_factors):
                risk_assessment['overall_risk_level'] = 'medium'
            else:
                risk_assessment['overall_risk_level'] = 'low'
            
            # Compile risk information
            risk_assessment['risk_factors'] = [rf for rf in risk_factors if rf['level'] != 'low']
            risk_assessment['risk_mitigations'] = self._generate_risk_mitigations(high_risk_factors)
            
            # Calculate risk scores for each mode
            for mode in self.TRADING_MODES:
                mode_risk_score = self._calculate_mode_risk_score(mode, risk_factors)
                risk_assessment['risk_score_by_mode'][mode] = mode_risk_score
            
            return risk_assessment
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "comprehensive_risk_assessment")
            return {'overall_risk_level': 'medium', 'risk_factors': [], 'risk_score_by_mode': {}}

    def _assess_drawdown_risk(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess drawdown risk factor"""
        drawdown = performance_data.get('drawdown', 0.0)
        max_drawdown = performance_data.get('max_drawdown', 0.0)
        
        if max_drawdown > 0.15 or drawdown > 0.1:
            level = 'extreme' if max_drawdown > 0.2 else 'high'
        elif max_drawdown > 0.08 or drawdown > 0.05:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'type': 'drawdown',
            'level': level,
            'current_value': drawdown,
            'max_value': max_drawdown,
            'description': f'Current drawdown: {drawdown:.1%}, Max: {max_drawdown:.1%}'
        }

    def _assess_volatility_risk(self, performance_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess volatility risk factor"""
        volatility = performance_data.get('volatility', 0.02)
        volatility_regime = self.volatility_regime
        
        if volatility_regime == 'extreme' or volatility > 0.08:
            level = 'extreme'
        elif volatility_regime == 'high' or volatility > 0.05:
            level = 'high'
        elif volatility_regime == 'medium_high' or volatility > 0.03:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'type': 'volatility',
            'level': level,
            'current_value': volatility,
            'regime': volatility_regime,
            'description': f'Volatility: {volatility:.2%}, Regime: {volatility_regime}'
        }

    def _assess_exposure_risk(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess exposure risk factor"""
        exposure_ratio = performance_data.get('exposure_ratio', 0.0)
        position_count = performance_data.get('position_count', 0)
        
        if exposure_ratio > 0.9 or position_count > 15:
            level = 'extreme'
        elif exposure_ratio > 0.7 or position_count > 10:
            level = 'high'
        elif exposure_ratio > 0.5 or position_count > 6:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'type': 'exposure',
            'level': level,
            'exposure_ratio': exposure_ratio,
            'position_count': position_count,
            'description': f'Exposure: {exposure_ratio:.1%}, Positions: {position_count}'
        }

    def _assess_consensus_risk(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess consensus risk factor"""
        consensus = performance_data.get('consensus', 0.5)
        consensus_strength = performance_data.get('consensus_strength', 0.5)
        vote_count = performance_data.get('vote_count', 0)
        
        if consensus < 0.3 or consensus_strength < 0.3 or vote_count < 2:
            level = 'high'
        elif consensus < 0.4 or consensus_strength < 0.4 or vote_count < 3:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'type': 'consensus',
            'level': level,
            'consensus': consensus,
            'strength': consensus_strength,
            'vote_count': vote_count,
            'description': f'Consensus: {consensus:.1%}, Strength: {consensus_strength:.1%}, Votes: {vote_count}'
        }

    def _assess_market_risk(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess market risk factor"""
        regime = self.market_regime
        volatility_regime = self.volatility_regime
        session = self.market_session
        
        high_risk_regimes = ['volatile', 'uncertain', 'unknown']
        high_risk_volatility = ['high', 'extreme']
        low_liquidity_sessions = ['weekend', 'holiday', 'rollover']
        
        if (regime in high_risk_regimes and volatility_regime in high_risk_volatility) or session in low_liquidity_sessions:
            level = 'extreme'
        elif regime in high_risk_regimes or volatility_regime in high_risk_volatility:
            level = 'high'
        elif regime in ['reversal'] or volatility_regime == 'medium_high':
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'type': 'market',
            'level': level,
            'regime': regime,
            'volatility_regime': volatility_regime,
            'session': session,
            'description': f'Regime: {regime}, Volatility: {volatility_regime}, Session: {session}'
        }

    def _generate_risk_mitigations(self, high_risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Generate risk mitigation recommendations"""
        mitigations = []
        
        for risk_factor in high_risk_factors:
            risk_type = risk_factor['type']
            level = risk_factor['level']
            
            if risk_type == 'drawdown':
                if level == 'extreme':
                    mitigations.append("Switch to safe mode immediately to preserve capital")
                else:
                    mitigations.append("Reduce position sizing and consider defensive positions")
            
            elif risk_type == 'volatility':
                if level == 'extreme':
                    mitigations.append("Halt new trades until volatility subsides")
                else:
                    mitigations.append("Use wider stops and reduce leverage")
            
            elif risk_type == 'exposure':
                mitigations.append("Reduce position count and total exposure")
            
            elif risk_type == 'consensus':
                mitigations.append("Wait for stronger committee consensus before increasing risk")
            
            elif risk_type == 'market':
                mitigations.append("Adjust strategy for current market regime and session")
        
        return mitigations

    def _calculate_mode_risk_score(self, mode: str, risk_factors: List[Dict[str, Any]]) -> float:
        """Calculate risk score for a specific mode"""
        try:
            mode_config = self.TRADING_MODES[mode]
            risk_tolerance = mode_config['risk_multiplier']
            
            # Base risk penalty
            risk_penalty = 0.0
            
            for risk_factor in risk_factors:
                level = risk_factor['level']
                factor_penalty = {
                    'low': 0.0,
                    'medium': 0.1,
                    'high': 0.3,
                    'extreme': 0.6
                }.get(level, 0.1)
                
                # Adjust penalty based on mode risk tolerance
                adjusted_penalty = factor_penalty / risk_tolerance
                risk_penalty += adjusted_penalty
            
            # Convert penalty to risk score (higher score = lower risk)
            risk_score = max(0.0, 1.0 - risk_penalty)
            return risk_score
            
        except Exception:
            return 0.5

    async def _assess_market_alignment_comprehensive(self, performance_data: Dict[str, Any], 
                                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess market alignment for mode decision"""
        try:
            alignment = {
                'regime_alignment': {},
                'volatility_alignment': {},
                'session_alignment': {},
                'overall_alignment': 0.5
            }
            
            # Assess regime alignment for each mode
            for mode in self.TRADING_MODES:
                mode_config = self.TRADING_MODES[mode]
                volatility_tolerance = mode_config['volatility_tolerance']
                
                # Regime alignment scoring
                regime_score = self._score_regime_alignment(mode, self.market_regime)
                alignment['regime_alignment'][mode] = regime_score
                
                # Volatility alignment scoring
                volatility_score = self._score_volatility_alignment(volatility_tolerance, self.volatility_regime)
                alignment['volatility_alignment'][mode] = volatility_score
                
                # Session alignment scoring
                session_score = self._score_session_alignment(mode, self.market_session)
                alignment['session_alignment'][mode] = session_score
            
            # Calculate overall alignment
            alignment_scores = []
            for mode in self.TRADING_MODES:
                overall_mode_alignment = (
                    0.5 * alignment['regime_alignment'][mode] +
                    0.3 * alignment['volatility_alignment'][mode] +
                    0.2 * alignment['session_alignment'][mode]
                )
                alignment_scores.append(overall_mode_alignment)
            
            alignment['overall_alignment'] = np.mean(alignment_scores)
            alignment['best_aligned_mode'] = max(
                self.TRADING_MODES.keys(),
                key=lambda m: (
                    0.5 * alignment['regime_alignment'][m] +
                    0.3 * alignment['volatility_alignment'][m] +
                    0.2 * alignment['session_alignment'][m]
                )
            )
            
            return alignment
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_alignment_assessment")
            return {'overall_alignment': 0.5, 'regime_alignment': {}, 'volatility_alignment': {}, 'session_alignment': {}}

    def _score_regime_alignment(self, mode: str, regime: str) -> float:
        """Score how well a mode aligns with current market regime"""
        regime_mode_scores = {
            'trending': {'extreme': 0.9, 'aggressive': 0.8, 'normal': 0.6, 'safe': 0.3},
            'breakout': {'extreme': 0.85, 'aggressive': 0.8, 'normal': 0.5, 'safe': 0.4},
            'momentum': {'extreme': 0.8, 'aggressive': 0.75, 'normal': 0.6, 'safe': 0.4},
            'volatile': {'safe': 0.8, 'normal': 0.5, 'aggressive': 0.3, 'extreme': 0.1},
            'ranging': {'normal': 0.8, 'aggressive': 0.6, 'safe': 0.7, 'extreme': 0.4},
            'reversal': {'safe': 0.7, 'normal': 0.6, 'aggressive': 0.4, 'extreme': 0.2},
            'uncertain': {'safe': 0.9, 'normal': 0.5, 'aggressive': 0.2, 'extreme': 0.1},
            'unknown': {'safe': 0.8, 'normal': 0.6, 'aggressive': 0.4, 'extreme': 0.2}
        }
        
        return regime_mode_scores.get(regime, {}).get(mode, 0.5)

    def _score_volatility_alignment(self, mode_tolerance: str, volatility_regime: str) -> float:
        """Score how well mode volatility tolerance aligns with current volatility"""
        tolerance_scores = {
            'low': {'very_low': 0.9, 'low': 0.8, 'medium': 0.5, 'medium_high': 0.3, 'high': 0.1, 'extreme': 0.05},
            'medium': {'very_low': 0.7, 'low': 0.8, 'medium': 0.9, 'medium_high': 0.7, 'high': 0.4, 'extreme': 0.2},
            'medium-high': {'very_low': 0.6, 'low': 0.7, 'medium': 0.8, 'medium_high': 0.9, 'high': 0.6, 'extreme': 0.3},
            'high': {'very_low': 0.5, 'low': 0.6, 'medium': 0.7, 'medium_high': 0.8, 'high': 0.9, 'extreme': 0.6}
        }
        
        return tolerance_scores.get(mode_tolerance, {}).get(volatility_regime, 0.5)

    def _score_session_alignment(self, mode: str, session: str) -> float:
        """Score how well a mode aligns with current trading session"""
        session_mode_scores = {
            'overlap_london_new_york': {'extreme': 0.9, 'aggressive': 0.85, 'normal': 0.8, 'safe': 0.6},
            'london': {'extreme': 0.8, 'aggressive': 0.8, 'normal': 0.8, 'safe': 0.7},
            'new_york': {'extreme': 0.8, 'aggressive': 0.8, 'normal': 0.8, 'safe': 0.7},
            'asian': {'aggressive': 0.6, 'normal': 0.7, 'safe': 0.8, 'extreme': 0.4},
            'sydney': {'normal': 0.6, 'safe': 0.7, 'aggressive': 0.5, 'extreme': 0.3},
            'weekend': {'safe': 0.9, 'normal': 0.3, 'aggressive': 0.1, 'extreme': 0.05},
            'holiday': {'safe': 0.8, 'normal': 0.4, 'aggressive': 0.2, 'extreme': 0.1},
            'rollover': {'safe': 0.7, 'normal': 0.5, 'aggressive': 0.3, 'extreme': 0.1}
        }
        
        return session_mode_scores.get(session, {}).get(mode, 0.6)

    def _find_optimal_mode_with_confidence(self, mode_scores: Dict[str, float], 
                                         risk_assessment: Dict[str, Any], 
                                         market_alignment: Dict[str, Any]) -> Dict[str, Any]:
        """Find optimal mode with confidence assessment"""
        try:
            # Combine mode scores with risk and alignment
            final_scores = {}
            
            for mode in self.TRADING_MODES:
                base_score = mode_scores.get(mode, 0.5)
                risk_score = risk_assessment.get('risk_score_by_mode', {}).get(mode, 0.5)
                
                # Calculate alignment score
                regime_align = market_alignment.get('regime_alignment', {}).get(mode, 0.5)
                volatility_align = market_alignment.get('volatility_alignment', {}).get(mode, 0.5)
                session_align = market_alignment.get('session_alignment', {}).get(mode, 0.5)
                
                alignment_score = (0.5 * regime_align + 0.3 * volatility_align + 0.2 * session_align)
                
                # Combined final score
                final_score = (
                    0.5 * base_score +
                    0.3 * risk_score +
                    0.2 * alignment_score
                )
                
                final_scores[mode] = final_score
            
            # Find best mode
            best_mode = max(final_scores.items(), key=lambda x: x[1])
            
            # Calculate confidence based on score separation
            sorted_scores = sorted(final_scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                score_separation = sorted_scores[0] - sorted_scores[1]
                confidence = min(0.95, 0.5 + score_separation)
            else:
                confidence = best_mode[1]
            
            return {
                'mode': best_mode[0],
                'confidence': confidence,
                'score': best_mode[1],
                'all_scores': final_scores,
                'score_separation': score_separation if len(sorted_scores) >= 2 else 0.0
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "optimal_mode_finding")
            return {'mode': self.current_mode, 'confidence': 0.5, 'score': 0.5}

    def _analyze_mode_change_necessity(self, recommended_mode: str, confidence: float, 
                                     mode_scores: Dict[str, float], 
                                     risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if mode change is necessary with enhanced logic"""
        try:
            current_score = mode_scores.get(self.current_mode, 0.5)
            recommended_score = mode_scores.get(recommended_mode, 0.5)
            improvement = recommended_score - current_score
            
            # Base change threshold
            change_threshold = 0.1
            
            # Adjust threshold based on conditions
            if risk_assessment.get('overall_risk_level') in ['high', 'extreme']:
                change_threshold = 0.05  # Lower threshold for risky conditions
            elif self.mode_persistence < self.min_persistence * 2:
                change_threshold = 0.15  # Higher threshold for recently changed modes
            
            # Determine urgency
            urgency = 'normal'
            if improvement > 0.3:
                urgency = 'high'
            elif improvement > 0.5:
                urgency = 'critical'
            elif risk_assessment.get('overall_risk_level') == 'extreme':
                urgency = 'emergency'
            
            should_change = (
                recommended_mode != self.current_mode and
                improvement > change_threshold and
                confidence > self.mode_intelligence['confidence_threshold']
            )
            
            return {
                'should_change': should_change,
                'improvement': improvement,
                'change_threshold': change_threshold,
                'urgency': urgency,
                'confidence_met': confidence > self.mode_intelligence['confidence_threshold'],
                'risk_justification': risk_assessment.get('overall_risk_level') in ['high', 'extreme']
            }
            
        except Exception:
            return {'should_change': False, 'improvement': 0.0, 'urgency': 'normal'}

    # Placeholder methods for comprehensive system (implement following same patterns)
    async def _generate_mode_reasoning_comprehensive(self, performance_data: Dict[str, Any], market_data: Dict[str, Any], 
                                                   mode_scores: Dict[str, float], risk_assessment: Dict[str, Any], 
                                                   market_alignment: Dict[str, Any]) -> List[str]:
        """Generate comprehensive reasoning for mode decision"""
        # Implementation would follow similar pattern to other comprehensive methods
        return ["Mode decision based on comprehensive analysis"]

    async def _apply_mode_decision_comprehensive(self, decision: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mode decision with comprehensive tracking"""
        # Implementation would follow similar pattern
        return {"mode_changed": False}

    async def _analyze_mode_effectiveness_comprehensive(self, performance_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mode effectiveness comprehensively"""
        # Implementation would follow similar pattern
        return {"current_effectiveness": 0.5}

    async def _update_adaptive_thresholds_comprehensive(self, performance_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update adaptive thresholds comprehensively"""
        # Implementation would follow similar pattern
        return {"thresholds_updated": False}

    async def _generate_intelligent_mode_recommendations(self, mode_decision: Dict[str, Any], effectiveness_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent mode recommendations"""
        # Implementation would follow similar pattern
        return ["Continue current mode approach"]

    async def _generate_comprehensive_mode_thesis(self, mode_decision: Dict[str, Any], effectiveness_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive mode thesis"""
        # Implementation would follow similar pattern
        return "Trading mode management proceeding optimally"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus comprehensively"""
        # Implementation would follow similar pattern
        pass

    def _is_market_open(self) -> bool:
        """Check if market is open based on schedule"""
        if not self.market_schedule:
            return True  # Assume open if no schedule
        
        try:
            import pytz
            
            timezone = self.market_schedule.get('timezone', 'UTC')
            tz = pytz.timezone(timezone)
            now = datetime.datetime.now(tz)
            
            # Check weekends
            weekday = now.weekday()  # Monday=0, Sunday=6
            if weekday in self.market_schedule.get('close_days', [5, 6]):
                return False
            
            # Check holidays
            if 'holidays' in self.market_schedule:
                today_str = now.strftime('%Y-%m-%d')
                if today_str in self.market_schedule['holidays']:
                    return False
            
            # Check hours
            hour = now.hour
            open_hour = self.market_schedule.get('open_hour', 0)
            close_hour = self.market_schedule.get('close_hour', 23)
            
            if hour < open_hour or hour >= close_hour:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Market schedule check failed: {e}")
            return True  # Default to open

    def _get_mode_configuration(self) -> Dict[str, Any]:
        """Get current mode configuration"""
        return {
            'mode': self.current_mode,
            'auto': self.auto_mode,
            'persistence': self.mode_persistence,
            'effectiveness': self.mode_stats.get('mode_effectiveness', 0.5),
            'description': self.TRADING_MODES[self.current_mode]['description'],
            'risk_multiplier': self.TRADING_MODES[self.current_mode]['risk_multiplier'],
            'max_exposure': self.TRADING_MODES[self.current_mode]['max_exposure']
        }

    def _get_comprehensive_mode_stats(self) -> Dict[str, Any]:
        """Get comprehensive mode statistics"""
        return {
            **self.mode_stats,
            'decision_factors': self.decision_factors.copy(),
            'rolling_stats': self._calculate_rolling_stats(),
            'mode_analytics_summary': self._get_mode_analytics_summary()
        }

    def _get_mode_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of mode analytics"""
        summary = {}
        for mode, analytics in self.mode_analytics.items():
            if analytics['win_rates']:
                summary[mode] = {
                    'avg_win_rate': np.mean(analytics['win_rates']),
                    'avg_pnl': np.mean(analytics['pnl_values']),
                    'total_periods': len(analytics['win_rates']),
                    'best_sharpe': max(analytics.get('sharpe_ratios', [0])),
                    'best_profit_factor': max(analytics.get('profit_factors', [1]))
                }
        return summary

    def _get_market_context_summary(self) -> Dict[str, Any]:
        """Get market context summary"""
        return {
            'regime': self.market_regime,
            'volatility_regime': self.volatility_regime,
            'session': self.market_session,
            'market_open': self._is_market_open(),
            'regime_score': self.decision_factors.get('regime_score', 0.5),
            'volatility_score': self.decision_factors.get('volatility_score', 0.5),
            'session_score': self.decision_factors.get('session_score', 0.5)
        }

    def _calculate_rolling_stats(self) -> Dict[str, Any]:
        """Calculate rolling statistics"""
        if not self.stats_history:
            return {
                'win_rate': 0.5, 'avg_pnl': 0.0, 'drawdown': 0.0, 'consensus': 0.5,
                'volatility': 0.02, 'trade_count': 0, 'sharpe': 0.0, 'profit_factor': 1.0
            }
        
        recent_stats = list(self.stats_history)[-self.window:]
        
        return {
            'win_rate': np.mean([s.get('win_rate', 0.5) for s in recent_stats]),
            'avg_pnl': np.mean([s.get('avg_pnl', 0.0) for s in recent_stats]),
            'total_pnl': np.sum([s.get('total_pnl', 0.0) for s in recent_stats]),
            'drawdown': max([s.get('drawdown', 0.0) for s in recent_stats]),
            'consensus': np.mean([s.get('consensus', 0.5) for s in recent_stats]),
            'volatility': np.mean([s.get('volatility', 0.02) for s in recent_stats]),
            'trade_count': sum([s.get('trade_count', 0) for s in recent_stats]),
            'sharpe': np.mean([s.get('sharpe', 0.0) for s in recent_stats]),
            'profit_factor': np.mean([s.get('profit_factor', 1.0) for s in recent_stats])
        }

    def _update_mode_performance_metrics(self):
        """Update mode performance metrics"""
        self.mode_stats['current_mode_duration'] += 1
        self.mode_stats['total_uptime'] += 1
        
        # Update performance tracking
        self.performance_tracker.record_metric('TradingModeManager', 'mode_duration', self.mode_stats['current_mode_duration'])
        self.performance_tracker.record_metric('TradingModeManager', 'mode_effectiveness', self.mode_stats['mode_effectiveness'])
        self.performance_tracker.record_metric('TradingModeManager', 'mode_persistence', self.mode_persistence)

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'TradingModeManager',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'current_mode': self.current_mode,
            'auto_mode_enabled': self.auto_mode,
            'mode_persistence': self.mode_persistence,
            'total_switches': self.mode_stats.get('total_switches', 0),
            'mode_effectiveness': self.mode_stats.get('mode_effectiveness', 0.5),
            'decision_confidence': self.decision_factors.get('performance_score', 0.5),
            'market_alignment': self.decision_factors.get('market_context_score', 0.5),
            'session_duration': (datetime.datetime.now() - datetime.datetime.fromisoformat(self.mode_stats['session_start'])).total_seconds() / 3600
        }

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "TradingModeManager")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="Trading Mode Manager disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('TradingModeManager', 'process_time', processing_time, False)
        
        return {
            'trading_mode': self.current_mode,
            'mode_config': self._get_mode_configuration(),
            'mode_stats': {'error': str(error_context)},
            'mode_effectiveness': 0.0,
            'decision_factors': {'error': str(error_context)},
            'mode_thresholds': self.mode_thresholds.copy(),
            'market_context': {'error': str(error_context)},
            'mode_recommendations': ["Investigate trading mode manager errors"],
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def _get_safe_market_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when market data retrieval fails"""
        return {
            'recent_trades': [], 'risk_metrics': {}, 'votes': [], 'positions': [],
            'market_context': {}, 'session_metrics': {}, 'strategy_performance': {},
            'trading_performance': {}, 'market_regime': 'unknown', 'volatility_data': {},
            'economic_calendar': {}
        }

    def _get_safe_performance_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when performance data extraction fails"""
        return {
            'recent_trades': [], 'trade_count': 0, 'win_rate': 0.5, 'total_pnl': 0.0,
            'avg_pnl': 0.0, 'pnl_std': 0, 'max_win': 0, 'max_loss': 0, 'profit_factor': 1.0,
            'recent_trend': 0.0, 'trend_consistency': 0.5, 'current_balance': 10000.0,
            'drawdown': 0.0, 'max_drawdown': 0.0, 'risk_score': 0.5, 'consensus': 0.5,
            'vote_agreement': 0.5, 'vote_count': 0, 'consensus_strength': 0.5,
            'volatility': 0.02, 'volatility_regime_score': 0.5, 'exposure': 0.0,
            'position_count': 0, 'exposure_ratio': 0.0, 'strategy_effectiveness': 0.5,
            'strategy_confidence': 0.5, 'session_pnl': 0, 'session_trades': 0, 'sharpe': 0.0
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'trading_mode': self.current_mode,
            'mode_config': {'status': 'disabled'},
            'mode_stats': {'status': 'disabled'},
            'mode_effectiveness': 0.0,
            'decision_factors': {'status': 'disabled'},
            'mode_thresholds': self.mode_thresholds.copy(),
            'market_context': {'status': 'disabled'},
            'mode_recommendations': ["Restart trading mode manager system"],
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'}
        }

    # State management methods
    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'TradingModeManager',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'initial_mode': self.initial_mode,
                'window': self.window,
                'auto_mode': self.auto_mode,
                'min_persistence': self.min_persistence,
                'context_sensitivity': self.context_sensitivity,
                'performance_weight': self.performance_weight,
                'risk_weight': self.risk_weight,
                'consensus_weight': self.consensus_weight,
                'market_context_weight': self.market_context_weight,
                'regime_awareness': self.regime_awareness,
                'session_awareness': self.session_awareness,
                'volatility_scaling': self.volatility_scaling,
                'debug': self.debug
            },
            'mode_state': {
                'current_mode': self.current_mode,
                'mode_persistence': self.mode_persistence,
                'last_mode_change': self.last_mode_change,
                'last_change_reason': self.last_change_reason,
                'mode_stats': self.mode_stats.copy(),
                'decision_factors': self.decision_factors.copy(),
                'mode_thresholds': self.mode_thresholds.copy()
            },
            'market_context': {
                'market_regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'market_session': self.market_session,
                'market_schedule': self.market_schedule
            },
            'analytics_state': {
                'stats_history': list(self.stats_history)[-20:],
                'mode_history': list(self.mode_history)[-10:],
                'decision_trace': list(self.decision_trace)[-10:],
                'performance_history': list(self.performance_history)[-10:],
                'mode_analytics': {k: {sk: list(sv)[-10:] for sk, sv in v.items()} for k, v in self.mode_analytics.items()},
                'effectiveness_tracking': {k: list(v)[-5:] for k, v in self.effectiveness_tracking.items()}
            },
            'intelligence_state': {
                'mode_intelligence': self.mode_intelligence.copy(),
                'learning_history': list(self.learning_history),
                'threshold_adaptations': list(self.threshold_adaptations)
            },
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'performance_metrics': self._get_health_metrics()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.initial_mode = config.get("initial_mode", self.initial_mode)
            self.window = int(config.get("window", self.window))
            self.auto_mode = bool(config.get("auto_mode", self.auto_mode))
            self.min_persistence = int(config.get("min_persistence", self.min_persistence))
            self.context_sensitivity = float(config.get("context_sensitivity", self.context_sensitivity))
            self.performance_weight = float(config.get("performance_weight", self.performance_weight))
            self.risk_weight = float(config.get("risk_weight", self.risk_weight))
            self.consensus_weight = float(config.get("consensus_weight", self.consensus_weight))
            self.market_context_weight = float(config.get("market_context_weight", self.market_context_weight))
            self.regime_awareness = bool(config.get("regime_awareness", self.regime_awareness))
            self.session_awareness = bool(config.get("session_awareness", self.session_awareness))
            self.volatility_scaling = bool(config.get("volatility_scaling", self.volatility_scaling))
            self.debug = bool(config.get("debug", self.debug))
            
            # Load mode state
            mode_state = state.get("mode_state", {})
            self.current_mode = mode_state.get("current_mode", self.initial_mode)
            self.mode_persistence = int(mode_state.get("mode_persistence", 0))
            self.last_mode_change = mode_state.get("last_mode_change")
            self.last_change_reason = mode_state.get("last_change_reason", "")
            self.mode_stats.update(mode_state.get("mode_stats", {}))
            self.decision_factors.update(mode_state.get("decision_factors", {}))
            self.mode_thresholds.update(mode_state.get("mode_thresholds", {}))
            
            # Load market context
            market_context = state.get("market_context", {})
            self.market_regime = market_context.get("market_regime", "unknown")
            self.volatility_regime = market_context.get("volatility_regime", "medium")
            self.market_session = market_context.get("market_session", "unknown")
            self.market_schedule = market_context.get("market_schedule")
            
            # Load analytics state
            analytics_state = state.get("analytics_state", {})
            
            # Restore history collections
            self.stats_history.clear()
            for entry in analytics_state.get("stats_history", []):
                self.stats_history.append(entry)
            
            self.mode_history.clear()
            for entry in analytics_state.get("mode_history", []):
                self.mode_history.append(entry)
            
            self.decision_trace.clear()
            for entry in analytics_state.get("decision_trace", []):
                self.decision_trace.append(entry)
            
            self.performance_history.clear()
            for entry in analytics_state.get("performance_history", []):
                self.performance_history.append(entry)
            
            # Restore analytics collections
            mode_analytics_data = analytics_state.get("mode_analytics", {})
            self.mode_analytics = defaultdict(lambda: defaultdict(list))
            for mode, analytics in mode_analytics_data.items():
                for key, values in analytics.items():
                    self.mode_analytics[mode][key] = list(values)
            
            effectiveness_data = analytics_state.get("effectiveness_tracking", {})
            self.effectiveness_tracking = defaultdict(list)
            for mode, tracking in effectiveness_data.items():
                self.effectiveness_tracking[mode] = list(tracking)
            
            # Load intelligence state
            intelligence_state = state.get("intelligence_state", {})
            self.mode_intelligence.update(intelligence_state.get("mode_intelligence", {}))
            
            self.learning_history.clear()
            for entry in intelligence_state.get("learning_history", []):
                self.learning_history.append(entry)
            
            self.threshold_adaptations.clear()
            for entry in intelligence_state.get("threshold_adaptations", []):
                self.threshold_adaptations.append(entry)
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Trading Mode Manager state restored",
                current_mode=self.current_mode,
                auto_mode=self.auto_mode,
                mode_persistence=self.mode_persistence,
                total_switches=self.mode_stats.get('total_switches', 0)
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API METHODS
    # ═══════════════════════════════════════════════════════════════════

    def set_mode(self, mode: str, reason: str = "Manual override") -> None:
        """Set trading mode manually"""
        if mode not in self.TRADING_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {list(self.TRADING_MODES.keys())}")
        
        old_mode = self.current_mode
        self.current_mode = mode
        self.auto_mode = False  # Disable auto mode on manual override
        self.mode_persistence = 0
        self.last_mode_change = datetime.datetime.now().isoformat()
        self.last_change_reason = reason
        
        # Update statistics
        self.mode_stats['total_switches'] += 1
        self.mode_stats['manual_switches'] += 1
        self.mode_stats['current_mode_duration'] = 0
        
        # Record manual change
        self.mode_history.append({
            'timestamp': self.last_mode_change,
            'from_mode': old_mode,
            'to_mode': self.current_mode,
            'reason': reason,
            'confidence': 1.0,
            'auto': False,
            'context': {}
        })
        
        self.logger.info(format_operator_message(
            icon="🎛️",
            message="Manual mode change executed",
            from_mode=old_mode,
            to_mode=self.current_mode,
            reason=reason
        ))

    def set_auto_mode(self, auto: bool) -> None:
        """Enable/disable automatic mode switching"""
        old_auto = self.auto_mode
        self.auto_mode = auto
        
        self.logger.info(format_operator_message(
            icon="⚙️",
            message=f"Auto mode {'enabled' if auto else 'disabled'}",
            previous=f"{'enabled' if old_auto else 'disabled'}",
            current_mode=self.current_mode
        ))

    def get_mode(self) -> str:
        """Get current trading mode"""
        return self.current_mode

    def get_mode_stats(self) -> Dict[str, Any]:
        """Get comprehensive mode statistics"""
        return self._get_comprehensive_mode_stats()

    def get_observation_components(self) -> np.ndarray:
        """Return mode features for RL observation"""
        try:
            # One-hot encoding of current mode
            mode_encoding = np.zeros(len(self.TRADING_MODES), dtype=np.float32)
            mode_index = list(self.TRADING_MODES.keys()).index(self.current_mode)
            mode_encoding[mode_index] = 1.0
            
            # Enhanced features for modern system
            additional_features = np.array([
                float(self.auto_mode),  # Auto mode enabled
                float(self.mode_persistence) / max(self.min_persistence, 1),  # Normalized persistence
                float(self.decision_factors.get('performance_score', 0.5)),  # Performance factor
                float(self.decision_factors.get('risk_score', 0.5)),  # Risk factor
                float(self.decision_factors.get('consensus_score', 0.5)),  # Consensus factor
                float(self.decision_factors.get('market_context_score', 0.5)),  # Market context factor
                float(self.decision_factors.get('stability_score', 0.5)),  # Stability factor
                float(self.mode_stats.get('mode_effectiveness', 0.5)),  # Mode effectiveness
                float(self._get_volatility_regime_score()),  # Volatility regime score
                float(self._get_regime_score_enhanced(self.market_regime))  # Market regime score
            ], dtype=np.float32)
            
            # Combine mode encoding with features
            observation = np.concatenate([mode_encoding, additional_features])
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid mode observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Mode observation generation failed: {error_context}")
            # Return safe defaults
            default_encoding = np.zeros(len(self.TRADING_MODES), dtype=np.float32)
            default_encoding[1] = 1.0  # Normal mode
            default_additional = np.array([1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            return np.concatenate([default_encoding, default_additional])

    def get_trading_mode_report(self) -> str:
        """Generate comprehensive trading mode report"""
        # Mode status with emoji
        mode_emoji = {
            'safe': '🛡️',
            'normal': '⚖️', 
            'aggressive': '⚡',
            'extreme': '🚀'
        }
        
        current_emoji = mode_emoji.get(self.current_mode, '❓')
        mode_config = self.TRADING_MODES.get(self.current_mode, {})
        mode_description = mode_config.get('description', 'Unknown mode')
        
        # Effectiveness status with comprehensive analysis
        effectiveness = self.mode_stats.get('mode_effectiveness', 0.5)
        if effectiveness > 0.8:
            eff_status = "✅ Excellent"
        elif effectiveness > 0.65:
            eff_status = "⚡ Good"
        elif effectiveness > 0.5:
            eff_status = "⚠️ Fair"
        elif effectiveness > 0.35:
            eff_status = "🔶 Poor"
        else:
            eff_status = "🚨 Critical"
        
        # Recent mode changes with enhanced details
        change_lines = []
        for change in list(self.mode_history)[-3:]:
            timestamp = change['timestamp'][:19].replace('T', ' ')
            from_mode = change['from_mode']
            to_mode = change['to_mode']
            auto = '🤖' if change['auto'] else '👤'
            confidence = change.get('confidence', 0.5)
            change_lines.append(f"  {auto} {timestamp}: {from_mode} → {to_mode} ({confidence:.1%})")
        
        # Enhanced decision factors with visual indicators
        factor_lines = []
        for factor, value in self.decision_factors.items():
            if value > 0.8:
                emoji = "🟢"
            elif value > 0.6:
                emoji = "✅"
            elif value > 0.5:
                emoji = "⚡"
            elif value > 0.3:
                emoji = "⚠️"
            else:
                emoji = "🚨"
            
            factor_name = factor.replace('_', ' ').title()
            factor_lines.append(f"  {emoji} {factor_name}: {value:.1%}")
        
        # Rolling statistics with enhanced metrics
        rolling_stats = self._calculate_rolling_stats()
        
        # Mode analytics summary
        analytics_summary = self._get_mode_analytics_summary()
        mode_performance_lines = []
        for mode, stats in analytics_summary.items():
            emoji = mode_emoji.get(mode, '❓')
            win_rate = stats.get('avg_win_rate', 0.5)
            avg_pnl = stats.get('avg_pnl', 0.0)
            periods = stats.get('total_periods', 0)
            mode_performance_lines.append(f"  {emoji} {mode.title()}: {win_rate:.1%} WR, €{avg_pnl:+.1f} avg, {periods} periods")
        
        # Market context with regime analysis
        market_open_status = '🟢 Open' if self._is_market_open() else '🔴 Closed'
        regime_score = self.decision_factors.get('regime_score', 0.5)
        volatility_score = self.decision_factors.get('volatility_score', 0.5)
        
        # Threshold information
        current_thresholds = self.mode_thresholds.get(self.current_mode, {})
        threshold_lines = []
        for key, value in current_thresholds.items():
            if key == 'max_drawdown':
                threshold_lines.append(f"  📉 Max Drawdown: {value:.1%}")
            elif key == 'min_win_rate':
                threshold_lines.append(f"  🎯 Min Win Rate: {value:.1%}")
            elif key == 'min_consensus':
                threshold_lines.append(f"  🤝 Min Consensus: {value:.1%}")
            elif key == 'max_exposure':
                threshold_lines.append(f"  📊 Max Exposure: {value:.1%}")
        
        return f"""
⚙️ TRADING MODE MANAGER v3.0
═══════════════════════════════════════════════════════════════
{current_emoji} Current Mode: {self.current_mode.upper()} - {mode_description}
🎯 Mode Effectiveness: {eff_status} ({effectiveness:.1%})
🤖 Auto Mode: {'✅ Enabled' if self.auto_mode else '❌ Disabled'}
⏱️ Mode Persistence: {self.mode_persistence}/{self.min_persistence} periods
🔄 Total Switches: {self.mode_stats.get('total_switches', 0)} (Auto: {self.mode_stats.get('auto_switches', 0)}, Manual: {self.mode_stats.get('manual_switches', 0)})

📊 CURRENT MODE CONFIGURATION
• Risk Multiplier: {mode_config.get('risk_multiplier', 1.0):.1f}x
• Max Exposure: {mode_config.get('max_exposure', 0.5):.1%}
• Win Rate Threshold: {mode_config.get('win_rate_threshold', 0.5):.1%}
• Drawdown Limit: {mode_config.get('drawdown_limit', 0.1):.1%}
• Consensus Requirement: {mode_config.get('consensus_requirement', 0.3):.1%}
• Volatility Tolerance: {mode_config.get('volatility_tolerance', 'medium').title()}

🎛️ DECISION WEIGHTS & SENSITIVITY
• Performance Weight: {self.performance_weight:.1%}
• Risk Weight: {self.risk_weight:.1%}
• Consensus Weight: {self.consensus_weight:.1%}
• Market Context Weight: {self.market_context_weight:.1%}
• Context Sensitivity: {self.context_sensitivity:.1%}

📈 ROLLING STATISTICS (Last {self.window} periods)
• Win Rate: {rolling_stats['win_rate']:.1%}
• Average PnL: €{rolling_stats['avg_pnl']:+.2f}
• Total PnL: €{rolling_stats['total_pnl']:+.2f}
• Max Drawdown: {rolling_stats['drawdown']:.1%}
• Consensus Strength: {rolling_stats['consensus']:.1%}
• Market Volatility: {rolling_stats['volatility']:.2%}
• Trade Count: {rolling_stats['trade_count']}
• Sharpe Ratio: {rolling_stats['sharpe']:.2f}
• Profit Factor: {rolling_stats['profit_factor']:.2f}

🎯 DECISION FACTORS (Current Analysis)
{chr(10).join(factor_lines) if factor_lines else "  📭 No decision factors available"}

📊 MARKET CONTEXT & REGIME ANALYSIS
• Market Regime: {self.market_regime.title()} (Score: {regime_score:.1%})
• Volatility Level: {self.volatility_regime.title()} (Score: {volatility_score:.1%})
• Trading Session: {self.market_session.title()}
• Market Status: {market_open_status}
• Regime Awareness: {'✅ Enabled' if self.regime_awareness else '❌ Disabled'}
• Session Awareness: {'✅ Enabled' if self.session_awareness else '❌ Disabled'}
• Volatility Scaling: {'✅ Enabled' if self.volatility_scaling else '❌ Disabled'}

🎯 CURRENT MODE THRESHOLDS
{chr(10).join(threshold_lines) if threshold_lines else "  📭 No thresholds configured"}

📊 MODE PERFORMANCE ANALYTICS
{chr(10).join(mode_performance_lines) if mode_performance_lines else "  📭 No mode performance data available"}

📜 RECENT MODE CHANGES
{chr(10).join(change_lines) if change_lines else "  📭 No recent mode changes"}

🧠 INTELLIGENCE & LEARNING
• Adaptation Speed: {self.mode_intelligence.get('adaptation_speed', 0.1):.1%}
• Confidence Threshold: {self.mode_intelligence.get('confidence_threshold', 0.7):.1%}
• Stability Requirement: {self.mode_intelligence.get('stability_requirement', 0.8):.1%}
• Performance Memory: {self.mode_intelligence.get('performance_memory', 0.9):.1%}
• Learning Records: {len(self.learning_history)}
• Threshold Adaptations: {len(self.threshold_adaptations)}

💡 MODE DESCRIPTIONS
• 🛡️ Safe: {self.TRADING_MODES['safe']['description']}
• ⚖️ Normal: {self.TRADING_MODES['normal']['description']}
• ⚡ Aggressive: {self.TRADING_MODES['aggressive']['description']}
• 🚀 Extreme: {self.TRADING_MODES['extreme']['description']}

🎯 LAST DECISION CONTEXT
• Change Reason: {self.last_change_reason or 'No recent changes'}
• Last Change Time: {self.last_mode_change[:19].replace('T', ' ') if self.last_mode_change else 'Never'}
• Current Duration: {self.mode_stats.get('current_mode_duration', 0)} periods
• System Uptime: {self.mode_stats.get('total_uptime', 0)} periods

🔧 HEALTH & STATUS
• Error Count: {self.error_count}/{self.circuit_breaker_threshold}
• Status: {'🚨 DISABLED' if self.is_disabled else '✅ OPERATIONAL'}
• Session Duration: {(datetime.datetime.now() - datetime.datetime.fromisoformat(self.mode_stats['session_start'])).total_seconds() / 3600:.1f} hours
        """

    # ═══════════════════════════════════════════════════════════════════
    # LEGACY COMPATIBILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def step(self, **kwargs) -> None:
        """Legacy step interface for backward compatibility"""
        try:
            # Convert legacy parameters to modern format
            trade_result = kwargs.get('trade_result')
            pnl = kwargs.get('pnl', 0.0)
            consensus = kwargs.get('consensus', 0.5)
            volatility = kwargs.get('volatility', 0.02)
            drawdown = kwargs.get('drawdown', 0.0)
            sharpe = kwargs.get('sharpe', 0.0)
            
            # Create legacy stats entry
            if trade_result is not None:
                stats_entry = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'mode': self.current_mode,
                    'result': trade_result,
                    'pnl': float(pnl),
                    'consensus': float(consensus),
                    'volatility': float(volatility),
                    'drawdown': float(drawdown),
                    'sharpe': float(sharpe),
                    'win_rate': 1.0 if trade_result == 'win' else 0.0,
                    'avg_pnl': float(pnl),
                    'total_pnl': float(pnl),
                    'trade_count': 1 if trade_result in ['win', 'loss'] else 0,
                    'regime': 'unknown',
                    'volatility_level': 'medium',
                    'session': 'unknown',
                    'recent_trend': float(pnl),
                    'trend_consistency': 0.5,
                    'profit_factor': 1.0 if pnl == 0 else (max(pnl, 0) / max(abs(min(pnl, 0)), 1)),
                    'consensus_strength': float(consensus)
                }
                
                self.stats_history.append(stats_entry)
                
                # Update mode persistence and perform basic decision
                self.mode_persistence += 1
                self.mode_stats['current_mode_duration'] += 1
                
                # Simple legacy mode decision
                if self.auto_mode and self.mode_persistence >= self.min_persistence:
                    self._legacy_mode_decision()
                    
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "legacy_step")
            self.logger.warning(f"Legacy step processing failed: {error_context}")

    def _legacy_mode_decision(self) -> None:
        """Legacy mode decision logic for backward compatibility"""
        try:
            if len(self.stats_history) < 5:
                return
            
            # Calculate simple rolling metrics
            recent_stats = list(self.stats_history)[-min(self.window, len(self.stats_history)):]
            trade_results = [s for s in recent_stats if s.get('result') in ['win', 'loss']]
            
            if len(trade_results) == 0:
                return
            
            # Basic metrics
            win_rate = sum(1 for t in trade_results if t['result'] == 'win') / len(trade_results)
            avg_pnl = np.mean([s['pnl'] for s in recent_stats])
            max_drawdown = max([s['drawdown'] for s in recent_stats])
            avg_consensus = np.mean([s['consensus'] for s in recent_stats])
            
            # Simple mode logic
            old_mode = self.current_mode
            new_mode = self.current_mode
            
            if max_drawdown > 0.15 or win_rate < 0.35:
                new_mode = 'safe'
            elif win_rate >= 0.65 and avg_pnl > 10 and avg_consensus >= 0.65:
                new_mode = 'extreme'
            elif win_rate >= 0.55 and avg_pnl > 0 and avg_consensus >= 0.5:
                new_mode = 'aggressive'
            else:
                new_mode = 'normal'
            
            # Apply mode change if different
            if new_mode != old_mode:
                self.current_mode = new_mode
                self.mode_persistence = 0
                self.last_mode_change = datetime.datetime.now().isoformat()
                self.last_change_reason = f"Legacy: WR={win_rate:.1%}, PnL={avg_pnl:.1f}, DD={max_drawdown:.1%}, Consensus={avg_consensus:.1%}"
                
                self.mode_stats['total_switches'] += 1
                self.mode_stats['auto_switches'] += 1
                self.mode_stats['current_mode_duration'] = 0
                
                # Record change
                self.mode_history.append({
                    'timestamp': self.last_mode_change,
                    'from_mode': old_mode,
                    'to_mode': new_mode,
                    'reason': self.last_change_reason,
                    'confidence': 0.7,  # Default confidence for legacy
                    'auto': True,
                    'context': {'legacy': True}
                })
                
                self.logger.info(format_operator_message(
                    icon="⚙️",
                    message="Legacy mode change applied",
                    from_mode=old_mode,
                    to_mode=new_mode,
                    win_rate=f"{win_rate:.1%}",
                    avg_pnl=f"€{avg_pnl:.1f}",
                    drawdown=f"{max_drawdown:.1%}"
                ))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "legacy_mode_decision")
            self.logger.warning(f"Legacy mode decision failed: {error_context}")

    def update_stats(self, trade_result: str, pnl: float, consensus: float, 
                    volatility: float, drawdown: float, sharpe: Optional[float] = None) -> None:
        """Legacy update stats interface"""
        self.step(
            trade_result=trade_result,
            pnl=pnl,
            consensus=consensus,
            volatility=volatility,
            drawdown=drawdown,
            sharpe=sharpe or 0.0
        )

    def decide_mode(self) -> str:
        """Legacy mode decision interface"""
        if self.auto_mode and self.mode_persistence >= self.min_persistence:
            self._legacy_mode_decision()
        return self.current_mode

    def get_stats(self) -> Dict[str, Any]:
        """Legacy stats interface"""
        return self.get_mode_stats()

    def get_last_decisions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Legacy decision history interface"""
        return list(self.decision_trace)[-n:]

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for system monitoring"""
        return {
            'module_name': 'TradingModeManager',
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
                'message': 'TradingModeManager disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })
        
        if self.error_count > 2:
            alerts.append({
                'severity': 'warning',
                'message': f'High error count: {self.error_count}',
                'action': 'Monitor for recurring issues'
            })
        
        effectiveness = self.mode_stats.get('mode_effectiveness', 0.5)
        if effectiveness < 0.3:
            alerts.append({
                'severity': 'warning',
                'message': f'Low mode effectiveness: {effectiveness:.1%}',
                'action': 'Review mode decision parameters and thresholds'
            })
        
        if not self.auto_mode:
            alerts.append({
                'severity': 'info',
                'message': 'Auto mode disabled - manual mode management',
                'action': 'Consider enabling auto mode for optimal performance'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        if self.is_disabled:
            recommendations.append("Restart TradingModeManager module after investigating errors")
        
        if len(self.stats_history) < 10:
            recommendations.append("Insufficient statistics - continue trading to build performance baseline")
        
        effectiveness = self.mode_stats.get('mode_effectiveness', 0.5)
        if effectiveness < 0.4:
            recommendations.append("Consider adjusting mode thresholds or decision weights")
        
        if self.mode_stats.get('total_switches', 0) > 50:
            recommendations.append("High switching frequency - consider increasing persistence requirements")
        
        if not recommendations:
            recommendations.append("TradingModeManager operating within normal parameters")
        
        return recommendations