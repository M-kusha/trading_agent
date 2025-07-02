# ─────────────────────────────────────────────────────────────
# File: modules/trading_modes/trading_mode.py
# Enhanced Trading Mode Manager with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import copy
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class TradingModeManager(Module, AnalysisMixin, StateManagementMixin):
    """
    Enhanced trading mode manager with InfoBus integration.
    Provides intelligent mode switching based on comprehensive market analysis,
    performance tracking, and context-aware decision making.
    """

    # Trading modes with descriptions
    TRADING_MODES = {
        "safe": "Conservative risk management",
        "normal": "Balanced trading approach", 
        "aggressive": "Increased risk for higher returns",
        "extreme": "Maximum risk for maximum returns"
    }

    # Enhanced default configuration
    ENHANCED_DEFAULTS = {
        "initial_mode": "normal",
        "window": 20,
        "auto_mode": True,
        "min_persistence": 5,
        "context_sensitivity": 0.8,
        "performance_weight": 0.4,
        "risk_weight": 0.3,
        "consensus_weight": 0.2,
        "market_context_weight": 0.1,
        "regime_awareness": True,
        "session_awareness": True,
        "volatility_scaling": True
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        initial_mode: str = "normal",
        window: int = 20,
        auto_mode: bool = True,
        debug: bool = False,
        market_schedule: Optional[Dict[str, Any]] = None,
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
        
        # Merge configuration with enhanced defaults
        self.mode_config = copy.deepcopy(self.ENHANCED_DEFAULTS)
        if config:
            self.mode_config.update(config)
        
        # Validate initial mode
        if initial_mode not in self.TRADING_MODES:
            raise ValueError(f"Invalid initial_mode '{initial_mode}'. Must be one of {list(self.TRADING_MODES.keys())}")
        
        # Core parameters
        self.current_mode = initial_mode
        self.window = int(window)
        self.auto_mode = auto_mode
        self.min_persistence = int(self.mode_config["min_persistence"])
        self.context_sensitivity = float(self.mode_config["context_sensitivity"])
        self.performance_weight = float(self.mode_config["performance_weight"])
        self.risk_weight = float(self.mode_config["risk_weight"])
        self.consensus_weight = float(self.mode_config["consensus_weight"])
        self.market_context_weight = float(self.mode_config["market_context_weight"])
        self.regime_awareness = bool(self.mode_config["regime_awareness"])
        self.session_awareness = bool(self.mode_config["session_awareness"])
        self.volatility_scaling = bool(self.mode_config["volatility_scaling"])
        
        # Enhanced state tracking
        self.stats_history = deque(maxlen=self.window * 2)
        self.mode_history = deque(maxlen=50)
        self.decision_trace = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)
        
        # Mode persistence tracking
        self.mode_persistence = 0
        self.last_mode_change = None
        self.last_change_reason = ""
        
        # Market context awareness
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        self.market_schedule = market_schedule
        
        # Performance analytics
        self.mode_analytics = defaultdict(lambda: defaultdict(list))
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self.session_performance = defaultdict(lambda: defaultdict(list))
        
        # Mode switching statistics
        self.mode_stats = {
            "total_switches": 0,
            "auto_switches": 0,
            "manual_switches": 0,
            "current_mode_duration": 0,
            "mode_effectiveness": 0.0,
            "switching_accuracy": 0.0
        }
        
        # Enhanced decision factors
        self.decision_factors = {
            "performance_score": 0.5,
            "risk_score": 0.5,
            "consensus_score": 0.5,
            "market_context_score": 0.5,
            "volatility_score": 0.5,
            "regime_score": 0.5
        }
        
        # Mode thresholds (adaptive)
        self.mode_thresholds = {
            "safe": {"max_drawdown": 0.15, "min_win_rate": 0.0, "min_consensus": 0.0},
            "normal": {"max_drawdown": 0.10, "min_win_rate": 0.40, "min_consensus": 0.30},
            "aggressive": {"max_drawdown": 0.08, "min_win_rate": 0.55, "min_consensus": 0.50},
            "extreme": {"max_drawdown": 0.05, "min_win_rate": 0.65, "min_consensus": 0.65}
        }
        
        # Learning and adaptation
        self.learning_history = deque(maxlen=30)
        self.threshold_adaptations = deque(maxlen=20)
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "TradingModeManager",
            "logs/trading_modes/trading_mode_manager.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("TradingModeManager")
        
        self.log_operator_info(
            "⚙️ Enhanced Trading Mode Manager initialized",
            initial_mode=self.current_mode,
            window=self.window,
            auto_mode=self.auto_mode,
            regime_awareness=self.regime_awareness,
            session_awareness=self.session_awareness
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Reset mode state
        self.current_mode = "normal"
        self.mode_persistence = 0
        self.last_mode_change = None
        self.last_change_reason = ""
        
        # Reset history
        self.stats_history.clear()
        self.mode_history.clear()
        self.decision_trace.clear()
        self.performance_history.clear()
        
        # Reset market context
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        
        # Reset statistics
        self.mode_stats = {
            "total_switches": 0,
            "auto_switches": 0,
            "manual_switches": 0,
            "current_mode_duration": 0,
            "mode_effectiveness": 0.0,
            "switching_accuracy": 0.0
        }
        
        # Reset decision factors
        self.decision_factors = {factor: 0.5 for factor in self.decision_factors}
        
        # Reset analytics
        self.mode_analytics.clear()
        self.regime_performance.clear()
        self.session_performance.clear()
        
        # Reset learning
        self.learning_history.clear()
        self.threshold_adaptations.clear()
        
        self.log_operator_info("🔄 Trading Mode Manager reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using fallback mode")
            self._process_legacy_step(**kwargs)
            return
        
        # Extract comprehensive context
        context = extract_standard_context(info_bus)
        
        # Update market context awareness
        self._update_market_context(context, info_bus)
        
        # Extract performance data from InfoBus
        performance_data = self._extract_performance_data_from_info_bus(info_bus)
        
        # Update statistics with new data
        self._update_performance_statistics(performance_data, context)
        
        # Perform intelligent mode decision
        mode_decision = self._make_intelligent_mode_decision(performance_data, context)
        
        # Apply mode change if needed
        self._apply_mode_decision(mode_decision, context)
        
        # Analyze mode effectiveness
        self._analyze_mode_effectiveness(performance_data, context)
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds(performance_data, context)
        
        # Update InfoBus with results
        self._update_info_bus(info_bus)
        
        # Record audit for mode decisions
        self._record_mode_audit(info_bus, context, mode_decision)
        
        # Update performance metrics
        self._update_mode_performance_metrics()

    def _extract_performance_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract comprehensive performance data from InfoBus"""
        
        data = {}
        
        try:
            # Get recent trades for performance calculation
            recent_trades = info_bus.get('recent_trades', [])
            data['recent_trades'] = recent_trades
            data['trade_count'] = len(recent_trades)
            
            # Calculate trade performance
            if recent_trades:
                winning_trades = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
                data['win_rate'] = winning_trades / len(recent_trades)
                data['total_pnl'] = sum(trade.get('pnl', 0) for trade in recent_trades)
                data['avg_pnl'] = data['total_pnl'] / len(recent_trades)
            else:
                data['win_rate'] = 0.5  # Neutral
                data['total_pnl'] = 0.0
                data['avg_pnl'] = 0.0
            
            # Get risk metrics
            risk_data = info_bus.get('risk', {})
            data['current_balance'] = float(risk_data.get('balance', risk_data.get('equity', 10000)))
            data['drawdown'] = max(0.0, float(risk_data.get('current_drawdown', 0.0)))
            data['max_drawdown'] = max(0.0, float(risk_data.get('max_drawdown', 0.0)))
            
            # Get committee consensus
            votes = info_bus.get('votes', [])
            if votes:
                confidences = [vote.get('confidence', 0.5) for vote in votes]
                data['consensus'] = np.mean(confidences)
                data['vote_agreement'] = 1.0 - np.std(confidences)  # Higher std = lower agreement
            else:
                data['consensus'] = 0.5
                data['vote_agreement'] = 0.5
            
            # Get volatility from market context
            market_context = info_bus.get('market_context', {})
            volatilities = market_context.get('volatility', {})
            if volatilities:
                data['volatility'] = np.mean(list(volatilities.values()))
            else:
                data['volatility'] = 0.02  # Default
            
            # Get positions for exposure calculation
            positions = InfoBusExtractor.get_positions(info_bus)
            total_exposure = sum(abs(pos.get('size', 0)) for pos in positions)
            data['exposure'] = total_exposure
            data['position_count'] = len(positions)
            
            # Get module performance scores
            module_data = info_bus.get('module_data', {})
            
            # Risk module scores
            risk_scores = []
            for module_name in ['dynamic_risk_controller', 'portfolio_risk_system', 'execution_quality_monitor']:
                module_info = module_data.get(module_name, {})
                if 'quality_score' in module_info:
                    risk_scores.append(module_info['quality_score'])
                elif 'effectiveness_score' in module_info:
                    risk_scores.append(module_info['effectiveness_score'])
            
            data['avg_risk_score'] = np.mean(risk_scores) if risk_scores else 0.5
            
            # Simulation scores
            simulation_data = module_data.get('shadow_simulator', {})
            data['simulation_confidence'] = simulation_data.get('confidence_score', 0.5)
            
            # Calculate Sharpe ratio if possible
            if len(recent_trades) > 5:
                returns = [trade.get('pnl', 0) / max(data['current_balance'], 1000) for trade in recent_trades]
                if np.std(returns) > 0:
                    data['sharpe'] = np.sqrt(252) * np.mean(returns) / np.std(returns)
                else:
                    data['sharpe'] = 0.0
            else:
                data['sharpe'] = 0.0
            
        except Exception as e:
            self.log_operator_warning(f"Performance data extraction failed: {e}")
            # Provide safe defaults
            data = {
                'recent_trades': [],
                'trade_count': 0,
                'win_rate': 0.5,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'current_balance': 10000.0,
                'drawdown': 0.0,
                'max_drawdown': 0.0,
                'consensus': 0.5,
                'vote_agreement': 0.5,
                'volatility': 0.02,
                'exposure': 0.0,
                'position_count': 0,
                'avg_risk_score': 0.5,
                'simulation_confidence': 0.5,
                'sharpe': 0.0
            }
        
        return data

    def _update_market_context(self, context: Dict[str, Any], info_bus: InfoBus) -> None:
        """Update market context awareness"""
        
        try:
            # Update regime tracking
            old_regime = self.market_regime
            self.market_regime = context.get('regime', 'unknown')
            self.volatility_regime = context.get('volatility_level', 'medium')
            self.market_session = context.get('session', 'unknown')
            
            # Log regime changes
            if self.market_regime != old_regime and self.regime_awareness:
                self.log_operator_info(
                    f"📊 Market regime change: {old_regime} → {self.market_regime}",
                    volatility=self.volatility_regime,
                    session=self.market_session,
                    mode_impact="Mode thresholds may adjust"
                )
            
        except Exception as e:
            self.log_operator_warning(f"Market context update failed: {e}")

    def _update_performance_statistics(self, performance_data: Dict[str, Any], 
                                      context: Dict[str, Any]) -> None:
        """Update performance statistics with new data"""
        
        try:
            # Create comprehensive stats entry
            stats_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'mode': self.current_mode,
                'win_rate': performance_data.get('win_rate', 0.5),
                'avg_pnl': performance_data.get('avg_pnl', 0.0),
                'drawdown': performance_data.get('drawdown', 0.0),
                'consensus': performance_data.get('consensus', 0.5),
                'volatility': performance_data.get('volatility', 0.02),
                'sharpe': performance_data.get('sharpe', 0.0),
                'trade_count': performance_data.get('trade_count', 0),
                'exposure': performance_data.get('exposure', 0.0),
                'regime': context.get('regime', 'unknown'),
                'volatility_level': context.get('volatility_level', 'medium'),
                'session': context.get('session', 'unknown')
            }
            
            # Add to history
            self.stats_history.append(stats_entry)
            
            # Track mode-specific performance
            mode_performance = self.mode_analytics[self.current_mode]
            mode_performance['win_rates'].append(stats_entry['win_rate'])
            mode_performance['pnl_values'].append(stats_entry['avg_pnl'])
            mode_performance['drawdowns'].append(stats_entry['drawdown'])
            mode_performance['timestamps'].append(stats_entry['timestamp'])
            
            # Track regime-specific performance
            if self.regime_awareness and self.market_regime != 'unknown':
                regime_perf = self.regime_performance[self.market_regime]
                regime_perf['modes'].append(self.current_mode)
                regime_perf['performance'].append(stats_entry['avg_pnl'])
                regime_perf['timestamps'].append(stats_entry['timestamp'])
            
            # Track session-specific performance
            if self.session_awareness and self.market_session != 'unknown':
                session_perf = self.session_performance[self.market_session]
                session_perf['modes'].append(self.current_mode)
                session_perf['performance'].append(stats_entry['avg_pnl'])
                session_perf['timestamps'].append(stats_entry['timestamp'])
            
        except Exception as e:
            self.log_operator_warning(f"Performance statistics update failed: {e}")

    def _make_intelligent_mode_decision(self, performance_data: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent mode decision based on comprehensive analysis"""
        
        decision = {
            'current_mode': self.current_mode,
            'recommended_mode': self.current_mode,
            'confidence': 0.5,
            'reasoning': [],
            'decision_factors': {},
            'should_change': False
        }
        
        try:
            # Check if market is closed (always safe)
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
            
            # Check persistence requirement
            if self.mode_persistence < self.min_persistence:
                decision['reasoning'] = [f'Mode persistence required ({self.mode_persistence}/{self.min_persistence})']
                return decision
            
            # Calculate decision factors
            self._calculate_decision_factors(performance_data, context)
            decision['decision_factors'] = self.decision_factors.copy()
            
            # Calculate overall score for each mode
            mode_scores = self._calculate_mode_scores(performance_data, context)
            
            # Find best mode
            best_mode = max(mode_scores.items(), key=lambda x: x[1])
            recommended_mode = best_mode[0]
            confidence = best_mode[1]
            
            # Generate reasoning
            reasoning = self._generate_mode_reasoning(performance_data, context, mode_scores)
            
            # Check if change is warranted
            current_score = mode_scores.get(self.current_mode, 0.0)
            improvement = confidence - current_score
            should_change = improvement > 0.1  # 10% improvement threshold
            
            decision.update({
                'recommended_mode': recommended_mode,
                'confidence': confidence,
                'reasoning': reasoning,
                'should_change': should_change,
                'mode_scores': mode_scores,
                'improvement': improvement
            })
            
        except Exception as e:
            self.log_operator_error(f"Mode decision failed: {e}")
            decision['reasoning'] = [f'Decision error: {e}']
        
        return decision

    def _calculate_decision_factors(self, performance_data: Dict[str, Any], 
                                   context: Dict[str, Any]) -> None:
        """Calculate individual decision factors"""
        
        try:
            # Performance factor
            if len(self.stats_history) >= 5:
                recent_stats = list(self.stats_history)[-5:]
                avg_win_rate = np.mean([s['win_rate'] for s in recent_stats])
                avg_pnl = np.mean([s['avg_pnl'] for s in recent_stats])
                
                # Normalize performance (0-1 scale)
                win_rate_score = avg_win_rate  # Already 0-1
                pnl_score = np.tanh(avg_pnl / 100.0) * 0.5 + 0.5  # Normalize around 0
                self.decision_factors['performance_score'] = (win_rate_score + pnl_score) / 2
            else:
                self.decision_factors['performance_score'] = 0.5
            
            # Risk factor
            drawdown = performance_data.get('drawdown', 0.0)
            volatility = performance_data.get('volatility', 0.02)
            
            # Lower risk = higher score
            drawdown_score = max(0.0, 1.0 - drawdown * 5)  # Penalize drawdown
            volatility_score = max(0.0, 1.0 - volatility * 20)  # Penalize high volatility
            self.decision_factors['risk_score'] = (drawdown_score + volatility_score) / 2
            
            # Consensus factor
            consensus = performance_data.get('consensus', 0.5)
            vote_agreement = performance_data.get('vote_agreement', 0.5)
            self.decision_factors['consensus_score'] = (consensus + vote_agreement) / 2
            
            # Market context factor
            regime_score = self._get_regime_score(context.get('regime', 'unknown'))
            volatility_level_score = self._get_volatility_level_score(context.get('volatility_level', 'medium'))
            session_score = self._get_session_score(context.get('session', 'unknown'))
            self.decision_factors['market_context_score'] = (regime_score + volatility_level_score + session_score) / 3
            
            # Additional factors
            self.decision_factors['volatility_score'] = volatility_score
            self.decision_factors['regime_score'] = regime_score
            
        except Exception as e:
            self.log_operator_warning(f"Decision factors calculation failed: {e}")

    def _get_regime_score(self, regime: str) -> float:
        """Get score based on market regime"""
        
        regime_scores = {
            'trending': 0.8,    # Good for aggressive trading
            'volatile': 0.3,    # Risky conditions
            'ranging': 0.6,     # Moderate conditions
            'normal': 0.7,      # Good general conditions
            'unknown': 0.5      # Neutral
        }
        
        return regime_scores.get(regime, 0.5)

    def _get_volatility_level_score(self, vol_level: str) -> float:
        """Get score based on volatility level"""
        
        vol_scores = {
            'low': 0.8,        # Good for aggressive trading
            'medium': 0.7,     # Good general conditions
            'high': 0.4,       # Risky conditions
            'extreme': 0.2,    # Very risky
            'unknown': 0.5     # Neutral
        }
        
        return vol_scores.get(vol_level, 0.5)

    def _get_session_score(self, session: str) -> float:
        """Get score based on trading session"""
        
        session_scores = {
            'american': 0.8,    # High liquidity
            'european': 0.7,    # Good liquidity
            'asian': 0.6,       # Moderate liquidity
            'rollover': 0.3,    # Low liquidity, high spreads
            'unknown': 0.5      # Neutral
        }
        
        return session_scores.get(session, 0.5)

    def _calculate_mode_scores(self, performance_data: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scores for each trading mode"""
        
        mode_scores = {}
        
        try:
            for mode in self.TRADING_MODES:
                score = self._calculate_single_mode_score(mode, performance_data, context)
                mode_scores[mode] = score
                
        except Exception as e:
            self.log_operator_warning(f"Mode scores calculation failed: {e}")
            # Fallback scores
            mode_scores = {mode: 0.5 for mode in self.TRADING_MODES}
        
        return mode_scores

    def _calculate_single_mode_score(self, mode: str, performance_data: Dict[str, Any], 
                                    context: Dict[str, Any]) -> float:
        """Calculate score for a single mode"""
        
        try:
            # Get mode thresholds
            thresholds = self.mode_thresholds[mode]
            
            # Check hard requirements
            drawdown = performance_data.get('drawdown', 0.0)
            if drawdown > thresholds['max_drawdown']:
                return 0.0  # Hard fail
            
            # Get recent performance if available
            if len(self.stats_history) >= 3:
                recent_stats = list(self.stats_history)[-3:]
                avg_win_rate = np.mean([s['win_rate'] for s in recent_stats])
                
                if avg_win_rate < thresholds['min_win_rate']:
                    return 0.1  # Very low score for poor performance
            
            # Calculate weighted score
            performance_score = self.decision_factors['performance_score']
            risk_score = self.decision_factors['risk_score']
            consensus_score = self.decision_factors['consensus_score']
            market_score = self.decision_factors['market_context_score']
            
            # Mode-specific scoring
            if mode == 'safe':
                # Safe mode prioritizes risk management
                score = (
                    risk_score * 0.5 +
                    performance_score * 0.2 +
                    consensus_score * 0.2 +
                    market_score * 0.1
                )
                # Bonus for poor conditions
                if drawdown > 0.05 or performance_score < 0.4:
                    score += 0.3
                    
            elif mode == 'normal':
                # Balanced scoring
                score = (
                    performance_score * self.performance_weight +
                    risk_score * self.risk_weight +
                    consensus_score * self.consensus_weight +
                    market_score * self.market_context_weight
                )
                
            elif mode == 'aggressive':
                # Aggressive mode prioritizes performance
                score = (
                    performance_score * 0.5 +
                    consensus_score * 0.3 +
                    risk_score * 0.1 +
                    market_score * 0.1
                )
                # Requires good conditions
                if performance_score < 0.6 or consensus_score < 0.5:
                    score *= 0.5
                    
            elif mode == 'extreme':
                # Extreme mode requires excellent conditions
                score = (
                    performance_score * 0.6 +
                    consensus_score * 0.3 +
                    market_score * 0.1
                )
                # Very strict requirements
                if (performance_score < 0.7 or consensus_score < 0.6 or 
                    self.decision_factors['regime_score'] < 0.6):
                    score *= 0.3
            
            return float(np.clip(score, 0.0, 1.0))
            
        except Exception as e:
            self.log_operator_warning(f"Single mode score calculation failed for {mode}: {e}")
            return 0.5

    def _generate_mode_reasoning(self, performance_data: Dict[str, Any], 
                                context: Dict[str, Any], mode_scores: Dict[str, float]) -> List[str]:
        """Generate human-readable reasoning for mode decision"""
        
        reasoning = []
        
        try:
            # Performance analysis
            performance_score = self.decision_factors['performance_score']
            if performance_score > 0.7:
                reasoning.append(f"✅ Strong performance ({performance_score:.1%})")
            elif performance_score < 0.4:
                reasoning.append(f"📉 Poor performance ({performance_score:.1%})")
            else:
                reasoning.append(f"⚡ Moderate performance ({performance_score:.1%})")
            
            # Risk analysis
            risk_score = self.decision_factors['risk_score']
            drawdown = performance_data.get('drawdown', 0.0)
            if risk_score > 0.7:
                reasoning.append(f"🛡️ Low risk conditions ({risk_score:.1%})")
            elif drawdown > 0.1:
                reasoning.append(f"⚠️ High drawdown ({drawdown:.1%})")
            elif risk_score < 0.4:
                reasoning.append(f"🚨 High risk conditions ({risk_score:.1%})")
            
            # Consensus analysis
            consensus_score = self.decision_factors['consensus_score']
            if consensus_score > 0.7:
                reasoning.append(f"🤝 Strong consensus ({consensus_score:.1%})")
            elif consensus_score < 0.4:
                reasoning.append(f"🤔 Low consensus ({consensus_score:.1%})")
            
            # Market context analysis
            regime = context.get('regime', 'unknown')
            vol_level = context.get('volatility_level', 'medium')
            
            if regime == 'trending':
                reasoning.append("📈 Trending market favors aggressive modes")
            elif regime == 'volatile':
                reasoning.append("💥 Volatile market requires caution")
            elif regime == 'ranging':
                reasoning.append("↔️ Ranging market - moderate approach")
            
            if vol_level == 'extreme':
                reasoning.append("🌪️ Extreme volatility - safety priority")
            elif vol_level == 'high':
                reasoning.append("⚡ High volatility detected")
            
            # Mode comparison
            sorted_modes = sorted(mode_scores.items(), key=lambda x: x[1], reverse=True)
            top_mode = sorted_modes[0]
            second_mode = sorted_modes[1] if len(sorted_modes) > 1 else ('none', 0)
            
            if top_mode[1] > 0.8:
                reasoning.append(f"🎯 {top_mode[0].title()} mode strongly recommended")
            elif top_mode[1] - second_mode[1] > 0.2:
                reasoning.append(f"⚡ {top_mode[0].title()} mode preferred over {second_mode[0]}")
            else:
                reasoning.append(f"🤔 Close decision between {top_mode[0]} and {second_mode[0]}")
            
        except Exception as e:
            self.log_operator_warning(f"Reasoning generation failed: {e}")
            reasoning = ["⚠️ Unable to generate detailed reasoning"]
        
        return reasoning[:5]  # Limit to top 5 reasons

    def _apply_mode_decision(self, decision: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Apply mode decision with comprehensive tracking"""
        
        try:
            recommended_mode = decision['recommended_mode']
            should_change = decision['should_change']
            
            if should_change and recommended_mode != self.current_mode:
                # Record the change
                old_mode = self.current_mode
                self.current_mode = recommended_mode
                self.mode_persistence = 0
                self.last_mode_change = datetime.datetime.now().isoformat()
                self.last_change_reason = "; ".join(decision['reasoning'])
                
                # Update statistics
                self.mode_stats['total_switches'] += 1
                self.mode_stats['auto_switches'] += 1
                self.mode_stats['current_mode_duration'] = 0
                
                # Record mode change
                self.mode_history.append({
                    'timestamp': self.last_mode_change,
                    'from_mode': old_mode,
                    'to_mode': self.current_mode,
                    'reason': self.last_change_reason,
                    'confidence': decision['confidence'],
                    'auto': True,
                    'context': context.copy()
                })
                
                # Log the change
                self.log_operator_info(
                    f"⚙️ Mode changed: {old_mode} → {self.current_mode}",
                    confidence=f"{decision['confidence']:.1%}",
                    reason=decision['reasoning'][0] if decision['reasoning'] else "Analysis complete",
                    regime=context.get('regime', 'unknown')
                )
                
            else:
                # Increment persistence
                self.mode_persistence += 1
                self.mode_stats['current_mode_duration'] += 1
            
            # Record decision trace
            self.decision_trace.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'current_mode': self.current_mode,
                'recommended_mode': recommended_mode,
                'decision': decision.copy(),
                'context': context.copy(),
                'changed': should_change
            })
            
        except Exception as e:
            self.log_operator_error(f"Mode decision application failed: {e}")

    def _analyze_mode_effectiveness(self, performance_data: Dict[str, Any], 
                                   context: Dict[str, Any]) -> None:
        """Analyze effectiveness of current mode"""
        
        try:
            # Calculate current mode effectiveness
            if len(self.stats_history) >= 5:
                mode_stats = [s for s in self.stats_history if s['mode'] == self.current_mode]
                
                if len(mode_stats) >= 3:
                    avg_performance = np.mean([s['avg_pnl'] for s in mode_stats[-5:]])
                    avg_win_rate = np.mean([s['win_rate'] for s in mode_stats[-5:]])
                    avg_drawdown = np.mean([s['drawdown'] for s in mode_stats[-5:]])
                    
                    # Calculate effectiveness score
                    performance_component = np.tanh(avg_performance / 50.0) * 0.5 + 0.5
                    win_rate_component = avg_win_rate
                    risk_component = max(0.0, 1.0 - avg_drawdown * 5)
                    
                    effectiveness = (performance_component + win_rate_component + risk_component) / 3
                    self.mode_stats['mode_effectiveness'] = float(effectiveness)
            
            # Store effectiveness history
            self.performance_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'mode': self.current_mode,
                'effectiveness': self.mode_stats.get('mode_effectiveness', 0.5),
                'performance_score': self.decision_factors.get('performance_score', 0.5),
                'risk_score': self.decision_factors.get('risk_score', 0.5)
            })
            
        except Exception as e:
            self.log_operator_warning(f"Mode effectiveness analysis failed: {e}")

    def _update_adaptive_thresholds(self, performance_data: Dict[str, Any], 
                                   context: Dict[str, Any]) -> None:
        """Update adaptive thresholds based on market conditions and performance"""
        
        try:
            # Only adapt if we have enough data
            if len(self.stats_history) < 20:
                return
            
            # Analyze recent performance patterns
            recent_stats = list(self.stats_history)[-20:]
            
            # Calculate baseline performance
            baseline_win_rate = np.mean([s['win_rate'] for s in recent_stats])
            baseline_drawdown = np.mean([s['drawdown'] for s in recent_stats])
            baseline_volatility = np.mean([s['volatility'] for s in recent_stats])
            
            # Adjust thresholds based on market conditions
            regime = context.get('regime', 'unknown')
            vol_level = context.get('volatility_level', 'medium')
            
            adaptation_factor = 1.0
            
            # Regime-based adaptations
            if regime == 'volatile':
                adaptation_factor = 0.9  # More conservative thresholds
            elif regime == 'trending':
                adaptation_factor = 1.1  # More aggressive thresholds
            
            # Volatility-based adaptations
            if vol_level in ['high', 'extreme']:
                adaptation_factor *= 0.9
            elif vol_level == 'low':
                adaptation_factor *= 1.1
            
            # Apply adaptations
            for mode in self.mode_thresholds:
                thresholds = self.mode_thresholds[mode]
                
                # Adapt win rate thresholds
                base_win_rate = thresholds['min_win_rate']
                adapted_win_rate = max(0.0, base_win_rate * adaptation_factor)
                thresholds['min_win_rate'] = adapted_win_rate
                
                # Adapt drawdown thresholds
                base_drawdown = thresholds['max_drawdown']
                adapted_drawdown = base_drawdown / adaptation_factor
                thresholds['max_drawdown'] = adapted_drawdown
            
            # Record adaptation
            self.threshold_adaptations.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'adaptation_factor': adaptation_factor,
                'regime': regime,
                'volatility_level': vol_level,
                'thresholds': copy.deepcopy(self.mode_thresholds)
            })
            
        except Exception as e:
            self.log_operator_warning(f"Adaptive threshold update failed: {e}")

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
            self.log_operator_warning(f"Market schedule check failed: {e}")
            return True  # Default to open

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with trading mode results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'trading_mode_manager', {
            'current_mode': self.current_mode,
            'auto_mode': self.auto_mode,
            'mode_persistence': self.mode_persistence,
            'last_change_reason': self.last_change_reason,
            'mode_stats': self.mode_stats.copy(),
            'decision_factors': self.decision_factors.copy(),
            'mode_thresholds': self.mode_thresholds.copy(),
            'market_context': {
                'regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'session': self.market_session
            },
            'effectiveness': self.mode_stats.get('mode_effectiveness', 0.5)
        })
        
        # Update trading mode in main InfoBus
        info_bus['trading_mode'] = self.current_mode
        info_bus['mode_config'] = {
            'mode': self.current_mode,
            'auto': self.auto_mode,
            'persistence': self.mode_persistence,
            'effectiveness': self.mode_stats.get('mode_effectiveness', 0.5)
        }
        
        # Add alerts for mode changes
        if hasattr(self, '_last_alert_mode') and self._last_alert_mode != self.current_mode:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Trading mode changed to {self.current_mode.upper()}",
                severity="info",
                module="TradingModeManager"
            )
            self._last_alert_mode = self.current_mode
        elif not hasattr(self, '_last_alert_mode'):
            self._last_alert_mode = self.current_mode

    def _record_mode_audit(self, info_bus: InfoBus, context: Dict[str, Any], 
                          decision: Dict[str, Any]) -> None:
        """Record comprehensive audit trail"""
        
        # Only audit mode changes or periodically
        should_audit = (
            decision.get('should_change', False) or
            info_bus.get('step_idx', 0) % 25 == 0
        )
        
        if should_audit:
            audit_data = {
                'mode_status': {
                    'current_mode': self.current_mode,
                    'auto_mode': self.auto_mode,
                    'persistence': self.mode_persistence,
                    'last_change': self.last_mode_change
                },
                'decision_analysis': decision.copy(),
                'decision_factors': self.decision_factors.copy(),
                'mode_thresholds': self.mode_thresholds.copy(),
                'context': context.copy(),
                'statistics': self.mode_stats.copy(),
                'effectiveness': {
                    'current_effectiveness': self.mode_stats.get('mode_effectiveness', 0.5),
                    'recent_performance': len(self.performance_history),
                    'mode_analytics': {k: len(v['win_rates']) for k, v in self.mode_analytics.items()}
                }
            }
            
            severity = "warning" if decision.get('should_change', False) else "info"
            
            self.audit_tracker.record_event(
                event_type="trading_mode_decision",
                module="TradingModeManager",
                details=audit_data,
                severity=severity
            )

    def _update_mode_performance_metrics(self) -> None:
        """Update performance metrics"""
        
        # Update performance metrics
        self._update_performance_metric('current_mode_duration', self.mode_stats['current_mode_duration'])
        self._update_performance_metric('total_switches', self.mode_stats['total_switches'])
        self._update_performance_metric('mode_effectiveness', self.mode_stats['mode_effectiveness'])
        self._update_performance_metric('mode_persistence', self.mode_persistence)
        
        # Update decision factors
        for factor_name, factor_value in self.decision_factors.items():
            self._update_performance_metric(f'decision_{factor_name}', factor_value)

    def _process_legacy_step(self, **kwargs) -> None:
        """Process legacy step parameters for backward compatibility"""
        
        try:
            # Extract legacy parameters
            trade_result = kwargs.get('trade_result')
            pnl = kwargs.get('pnl', 0.0)
            consensus = kwargs.get('consensus', 0.5)
            volatility = kwargs.get('volatility', 0.02)
            drawdown = kwargs.get('drawdown', 0.0)
            sharpe = kwargs.get('sharpe')
            
            # Update stats if data provided
            if trade_result is not None:
                if trade_result not in ['win', 'loss', 'hold']:
                    trade_result = 'hold'
                
                # Create legacy stats entry
                stats_entry = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'mode': self.current_mode,
                    'result': trade_result,
                    'pnl': float(pnl),
                    'consensus': float(consensus),
                    'volatility': float(volatility),
                    'drawdown': float(drawdown),
                    'sharpe': float(sharpe) if sharpe is not None else 0.0,
                    'win_rate': 1.0 if trade_result == 'win' else 0.0,
                    'avg_pnl': float(pnl),
                    'trade_count': 1 if trade_result in ['win', 'loss'] else 0,
                    'regime': 'unknown',
                    'volatility_level': 'medium',
                    'session': 'unknown'
                }
                
                self.stats_history.append(stats_entry)
            
            # Perform basic mode decision
            self._legacy_mode_decision()
            
        except Exception as e:
            self.log_operator_error(f"Legacy step processing failed: {e}")

    def _legacy_mode_decision(self) -> None:
        """Legacy mode decision logic"""
        
        try:
            if not self.auto_mode or len(self.stats_history) == 0:
                return
            
            # Increment persistence
            self.mode_persistence += 1
            
            if self.mode_persistence < self.min_persistence:
                return
            
            # Calculate simple rolling stats
            recent_stats = list(self.stats_history)[-min(self.window, len(self.stats_history)):]
            
            # Calculate basic metrics
            trade_results = [s for s in recent_stats if s.get('result') in ['win', 'loss']]
            
            if len(trade_results) == 0:
                return  # Not enough trading data
            
            win_rate = sum(1 for t in trade_results if t['result'] == 'win') / len(trade_results)
            avg_pnl = np.mean([s['pnl'] for s in recent_stats])
            max_drawdown = max([s['drawdown'] for s in recent_stats])
            avg_consensus = np.mean([s['consensus'] for s in recent_stats])
            
            # Simple mode logic
            old_mode = self.current_mode
            new_mode = self.current_mode
            
            if max_drawdown > 0.15 or win_rate < 0.35:
                new_mode = 'safe'
            elif win_rate >= 0.65 and avg_pnl > 0.5 and avg_consensus >= 0.6:
                new_mode = 'extreme'
            elif win_rate >= 0.55 and avg_pnl > 0.0 and avg_consensus >= 0.5:
                new_mode = 'aggressive'
            else:
                new_mode = 'normal'
            
            if new_mode != old_mode:
                self.current_mode = new_mode
                self.mode_persistence = 0
                self.last_mode_change = datetime.datetime.now().isoformat()
                self.last_change_reason = f"Legacy decision: WR={win_rate:.1%}, PnL={avg_pnl:.2f}, DD={max_drawdown:.1%}"
                
                self.mode_stats['total_switches'] += 1
                self.mode_stats['auto_switches'] += 1
                
                self.log_operator_info(
                    f"⚙️ Legacy mode change: {old_mode} → {new_mode}",
                    win_rate=f"{win_rate:.1%}",
                    pnl=f"{avg_pnl:.2f}",
                    drawdown=f"{max_drawdown:.1%}"
                )
                
        except Exception as e:
            self.log_operator_warning(f"Legacy mode decision failed: {e}")

    # ================== PUBLIC INTERFACE METHODS ==================

    def set_mode(self, mode: str, reason: str = "Manual override") -> None:
        """Set trading mode manually"""
        
        if mode not in self.TRADING_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {list(self.TRADING_MODES.keys())}")
        
        old_mode = self.current_mode
        self.current_mode = mode
        self.auto_mode = False
        self.mode_persistence = 0
        self.last_mode_change = datetime.datetime.now().isoformat()
        self.last_change_reason = reason
        
        # Update statistics
        self.mode_stats['total_switches'] += 1
        self.mode_stats['manual_switches'] += 1
        
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
        
        self.log_operator_info(
            f"🎛️ Manual mode change: {old_mode} → {self.current_mode}",
            reason=reason
        )

    def set_auto_mode(self, auto: bool) -> None:
        """Enable/disable automatic mode switching"""
        
        old_auto = self.auto_mode
        self.auto_mode = auto
        
        self.log_operator_info(
            f"⚙️ Auto mode {'enabled' if auto else 'disabled'}",
            previous=f"{'enabled' if old_auto else 'disabled'}"
        )

    def get_mode(self) -> str:
        """Get current trading mode"""
        return self.current_mode

    def get_mode_stats(self) -> Dict[str, Any]:
        """Get comprehensive mode statistics"""
        
        # Calculate rolling stats
        rolling_stats = self._calculate_rolling_stats()
        
        return {
            'current_mode': self.current_mode,
            'auto_mode': self.auto_mode,
            'mode_persistence': self.mode_persistence,
            'last_change': self.last_mode_change,
            'last_reason': self.last_change_reason,
            'rolling_stats': rolling_stats,
            'mode_stats': self.mode_stats.copy(),
            'decision_factors': self.decision_factors.copy(),
            'market_context': {
                'regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'session': self.market_session
            },
            'thresholds': self.mode_thresholds.copy()
        }

    def _calculate_rolling_stats(self) -> Dict[str, Any]:
        """Calculate rolling statistics"""
        
        if not self.stats_history:
            return {
                'win_rate': 0.5,
                'avg_pnl': 0.0,
                'drawdown': 0.0,
                'consensus': 0.5,
                'volatility': 0.02,
                'trade_count': 0
            }
        
        recent_stats = list(self.stats_history)[-self.window:]
        
        return {
            'win_rate': np.mean([s.get('win_rate', 0.5) for s in recent_stats]),
            'avg_pnl': np.mean([s.get('avg_pnl', 0.0) for s in recent_stats]),
            'drawdown': max([s.get('drawdown', 0.0) for s in recent_stats]),
            'consensus': np.mean([s.get('consensus', 0.5) for s in recent_stats]),
            'volatility': np.mean([s.get('volatility', 0.02) for s in recent_stats]),
            'trade_count': len([s for s in recent_stats if s.get('trade_count', 0) > 0])
        }

    def get_observation_components(self) -> np.ndarray:
        """Return mode features for observation"""
        
        try:
            # One-hot encoding of current mode
            mode_encoding = np.zeros(len(self.TRADING_MODES), dtype=np.float32)
            mode_index = list(self.TRADING_MODES.keys()).index(self.current_mode)
            mode_encoding[mode_index] = 1.0
            
            # Additional features
            additional_features = np.array([
                float(self.auto_mode),
                float(self.mode_persistence) / self.min_persistence,
                float(self.decision_factors.get('performance_score', 0.5)),
                float(self.decision_factors.get('risk_score', 0.5)),
                float(self.mode_stats.get('mode_effectiveness', 0.5))
            ], dtype=np.float32)
            
            return np.concatenate([mode_encoding, additional_features])
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            # Return safe defaults
            default_encoding = np.zeros(len(self.TRADING_MODES), dtype=np.float32)
            default_encoding[1] = 1.0  # Normal mode
            default_additional = np.array([1.0, 0.0, 0.5, 0.5, 0.5], dtype=np.float32)
            return np.concatenate([default_encoding, default_additional])

    def get_trading_mode_report(self) -> str:
        """Generate operator-friendly trading mode report"""
        
        # Mode status
        mode_emoji = {
            'safe': '🛡️',
            'normal': '⚖️', 
            'aggressive': '⚡',
            'extreme': '🚀'
        }
        
        current_emoji = mode_emoji.get(self.current_mode, '❓')
        mode_description = self.TRADING_MODES.get(self.current_mode, 'Unknown mode')
        
        # Effectiveness status
        effectiveness = self.mode_stats.get('mode_effectiveness', 0.5)
        if effectiveness > 0.8:
            eff_status = "✅ Excellent"
        elif effectiveness > 0.6:
            eff_status = "⚡ Good"
        elif effectiveness > 0.4:
            eff_status = "⚠️ Fair"
        else:
            eff_status = "🚨 Poor"
        
        # Recent mode changes
        change_lines = []
        for change in list(self.mode_history)[-3:]:
            timestamp = change['timestamp'][:19]
            from_mode = change['from_mode']
            to_mode = change['to_mode']
            auto = '🤖' if change['auto'] else '👤'
            change_lines.append(f"  {auto} {timestamp}: {from_mode} → {to_mode}")
        
        # Decision factors
        factor_lines = []
        for factor, value in self.decision_factors.items():
            if value > 0.7:
                emoji = "✅"
            elif value > 0.5:
                emoji = "⚡"
            elif value > 0.3:
                emoji = "⚠️"
            else:
                emoji = "🚨"
            
            factor_name = factor.replace('_', ' ').title()
            factor_lines.append(f"  {emoji} {factor_name}: {value:.1%}")
        
        # Rolling statistics
        rolling_stats = self._calculate_rolling_stats()
        
        return f"""
⚙️ TRADING MODE MANAGER
═══════════════════════════════════════
{current_emoji} Current Mode: {self.current_mode.upper()} - {mode_description}
🎯 Effectiveness: {eff_status} ({effectiveness:.1%})
🤖 Auto Mode: {'✅ Enabled' if self.auto_mode else '❌ Disabled'}
⏱️ Persistence: {self.mode_persistence}/{self.min_persistence} steps

📊 MODE CONFIGURATION
• Context Sensitivity: {self.context_sensitivity:.1%}
• Performance Weight: {self.performance_weight:.1%}
• Risk Weight: {self.risk_weight:.1%}
• Consensus Weight: {self.consensus_weight:.1%}
• Market Context Weight: {self.market_context_weight:.1%}

📈 ROLLING STATISTICS (Last {self.window} periods)
• Win Rate: {rolling_stats['win_rate']:.1%}
• Avg PnL: {rolling_stats['avg_pnl']:+.2f}
• Max Drawdown: {rolling_stats['drawdown']:.1%}
• Consensus: {rolling_stats['consensus']:.1%}
• Volatility: {rolling_stats['volatility']:.2%}
• Trade Count: {rolling_stats['trade_count']}

🎯 DECISION FACTORS
{chr(10).join(factor_lines) if factor_lines else "  📭 No decision factors available"}

🔧 MODE PERFORMANCE
• Total Switches: {self.mode_stats['total_switches']}
• Auto Switches: {self.mode_stats['auto_switches']}
• Manual Switches: {self.mode_stats['manual_switches']}
• Current Duration: {self.mode_stats['current_mode_duration']} steps
• Mode Effectiveness: {self.mode_stats.get('mode_effectiveness', 0.5):.1%}

📊 MARKET CONTEXT
• Regime: {self.market_regime.title()}
• Volatility Level: {self.volatility_regime.title()}
• Trading Session: {self.market_session.title()}
• Market Status: {'🟢 Open' if self._is_market_open() else '🔴 Closed'}

📜 RECENT MODE CHANGES
{chr(10).join(change_lines) if change_lines else "  📭 No recent mode changes"}

💡 MODE DESCRIPTIONS
• 🛡️ Safe: Conservative risk management
• ⚖️ Normal: Balanced trading approach
• ⚡ Aggressive: Increased risk for higher returns
• 🚀 Extreme: Maximum risk for maximum returns

🎯 LAST DECISION
• Change Reason: {self.last_change_reason or 'No recent changes'}
• Last Change: {self.last_mode_change[:19] if self.last_mode_change else 'Never'}
        """

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "current_mode": self.current_mode,
                "auto_mode": self.auto_mode,
                "window": self.window,
                "min_persistence": self.min_persistence,
                "context_sensitivity": self.context_sensitivity
            },
            "tracking": {
                "mode_persistence": self.mode_persistence,
                "last_mode_change": self.last_mode_change,
                "last_change_reason": self.last_change_reason
            },
            "market_context": {
                "regime": self.market_regime,
                "volatility_regime": self.volatility_regime,
                "session": self.market_session,
                "market_schedule": self.market_schedule
            },
            "statistics": self.mode_stats.copy(),
            "decision_factors": self.decision_factors.copy(),
            "thresholds": self.mode_thresholds.copy(),
            "history": {
                "stats_history": list(self.stats_history)[-20:],
                "mode_history": list(self.mode_history)[-10:],
                "performance_history": list(self.performance_history)[-10:]
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.current_mode = config.get("current_mode", "normal")
        self.auto_mode = bool(config.get("auto_mode", True))
        self.window = int(config.get("window", self.window))
        self.min_persistence = int(config.get("min_persistence", self.min_persistence))
        self.context_sensitivity = float(config.get("context_sensitivity", self.context_sensitivity))
        
        # Load tracking
        tracking = state.get("tracking", {})
        self.mode_persistence = int(tracking.get("mode_persistence", 0))
        self.last_mode_change = tracking.get("last_mode_change")
        self.last_change_reason = tracking.get("last_change_reason", "")
        
        # Load market context
        context = state.get("market_context", {})
        self.market_regime = context.get("regime", "normal")
        self.volatility_regime = context.get("volatility_regime", "medium")
        self.market_session = context.get("session", "unknown")
        self.market_schedule = context.get("market_schedule")
        
        # Load statistics
        self.mode_stats.update(state.get("statistics", {}))
        self.decision_factors.update(state.get("decision_factors", {}))
        self.mode_thresholds.update(state.get("thresholds", {}))
        
        # Load history
        history = state.get("history", {})
        
        stats_history = history.get("stats_history", [])
        self.stats_history.clear()
        for entry in stats_history:
            self.stats_history.append(entry)
            
        mode_history = history.get("mode_history", [])
        self.mode_history.clear()
        for entry in mode_history:
            self.mode_history.append(entry)
            
        performance_history = history.get("performance_history", [])
        self.performance_history.clear()
        for entry in performance_history:
            self.performance_history.append(entry)

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> None:
        """Legacy step interface for backward compatibility"""
        self._process_legacy_step(**kwargs)

    def update_stats(self, trade_result: str, pnl: float, consensus: float, 
                    volatility: float, drawdown: float, sharpe: Optional[float] = None) -> None:
        """Legacy update stats interface"""
        
        stats_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'mode': self.current_mode,
            'result': trade_result,
            'pnl': float(pnl),
            'consensus': float(consensus),
            'volatility': float(volatility),
            'drawdown': float(drawdown),
            'sharpe': float(sharpe) if sharpe is not None else 0.0,
            'win_rate': 1.0 if trade_result == 'win' else 0.0,
            'avg_pnl': float(pnl),
            'trade_count': 1 if trade_result in ['win', 'loss'] else 0,
            'regime': 'unknown',
            'volatility_level': 'medium',
            'session': 'unknown'
        }
        
        self.stats_history.append(stats_entry)

    def decide_mode(self) -> str:
        """Legacy mode decision interface"""
        
        self._legacy_mode_decision()
        return self.current_mode

    def get_stats(self) -> Dict[str, Any]:
        """Legacy stats interface"""
        return self.get_mode_stats()

    def get_last_decisions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Legacy decision history interface"""
        return list(self.decision_trace)[-n:]