# ─────────────────────────────────────────────────────────────
# File: modules/simulation/role_coach.py
# Enhanced Role Coach with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import copy
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class RoleCoach(Module, AnalysisMixin, StateManagementMixin):
    """
    Enhanced role coach with InfoBus integration.
    Provides intelligent trade discipline coaching with context-aware
    penalties and performance tracking across market regimes.
    """

    # Coaching modes
    COACHING_MODES = {
        "strict": "Strict discipline enforcement",
        "adaptive": "Context-aware discipline",
        "lenient": "Flexible trade management",
        "performance_based": "Performance-driven limits",
        "regime_aware": "Market regime specific rules"
    }

    # Enhanced default configuration
    ENHANCED_DEFAULTS = {
        "max_trades": 2,
        "penalty_multiplier": 1.0,
        "coaching_mode": "adaptive",
        "regime_sensitivity": 0.8,
        "performance_adjustment": True,
        "session_aware": True,
        "volatility_scaling": True,
        "learning_rate": 0.1,
        "penalty_decay": 0.95
    }

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], ModuleConfig]] = None,
        max_trades: int = 2,
        penalty_multiplier: float = 1.0,
        coaching_mode: str = "adaptive",
        debug: bool = False,
        **kwargs
    ):
        # --- split the ModuleConfig (from env) vs. override dict ---
        if isinstance(config, ModuleConfig):
            module_cfg   = config
            override_cfg = None
        else:
            module_cfg   = ModuleConfig(
                debug=debug,
                max_history=kwargs.get("max_history", 100),
                audit_enabled=kwargs.get("audit_enabled", True),
                **kwargs
            )
            override_cfg = config

        # initialize base Module
        super().__init__(module_cfg)
        self._initialize_analysis_state()

        # build your coach_config from defaults, then only update if override_cfg is a dict
        self.coach_config = copy.deepcopy(self.ENHANCED_DEFAULTS)
        if isinstance(override_cfg, dict):
            self.coach_config.update(override_cfg)

        # now pull everything out of coach_config as before
        self.max_trades          = int(max_trades)
        self.penalty_multiplier  = float(penalty_multiplier)
        self.coaching_mode       = (
            coaching_mode
            if coaching_mode in self.COACHING_MODES
            else "adaptive"
        )
        self.regime_sensitivity      = float(self.coach_config["regime_sensitivity"])
        self.performance_adjustment  = bool(self.coach_config["performance_adjustment"])
        self.session_aware           = bool(self.coach_config["session_aware"])
        self.volatility_scaling      = bool(self.coach_config["volatility_scaling"])
        self.learning_rate           = float(self.coach_config["learning_rate"])
        self.penalty_decay           = float(self.coach_config["penalty_decay"])
        
        # Enhanced state tracking
        self.discipline_history = deque(maxlen=100)
        self.penalty_history = deque(maxlen=50)
        self.performance_history = deque(maxlen=50)
        self.coaching_sessions = deque(maxlen=20)
        
        # Market context awareness
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        
        # Adaptive parameters
        self.adaptive_max_trades = self.max_trades
        self.adaptive_penalty = self.penalty_multiplier
        self.current_performance_score = 0.5
        
        # Session-specific limits
        self.session_limits = {
            "asian": int(self.max_trades * 0.8),
            "european": self.max_trades,
            "american": int(self.max_trades * 1.2),
            "rollover": int(self.max_trades * 0.6)
        }
        
        # Regime-specific multipliers
        self.regime_multipliers = {
            "trending": 1.2,
            "volatile": 0.8,
            "ranging": 1.0,
            "unknown": 1.0
        }
        
        # Coaching statistics
        self.coaching_stats = {
            "total_sessions": 0,
            "penalties_applied": 0,
            "total_penalty_amount": 0.0,
            "discipline_score": 1.0,
            "improvement_rate": 0.0,
            "violations": 0,
            "compliance_rate": 1.0
        }
        
        # Performance analytics
        self.coaching_analytics = defaultdict(list)
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self.session_performance = defaultdict(lambda: defaultdict(list))
        
        # Learning and adaptation
        self.learning_history = deque(maxlen=30)
        self.effectiveness_scores = deque(maxlen=20)
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "RoleCoach",
            "logs/simulation/role_coach.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("RoleCoach")
        
        self.log_operator_info(
            "🎯 Enhanced Role Coach initialized",
            max_trades=self.max_trades,
            penalty_multiplier=f"{self.penalty_multiplier:.2f}",
            coaching_mode=self.coaching_mode,
            regime_sensitivity=f"{self.regime_sensitivity:.1%}",
            performance_adjustment=self.performance_adjustment
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Reset discipline tracking
        self.discipline_history.clear()
        self.penalty_history.clear()
        self.performance_history.clear()
        self.coaching_sessions.clear()
        
        # Reset market context
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        
        # Reset adaptive parameters
        self.adaptive_max_trades = self.max_trades
        self.adaptive_penalty = self.penalty_multiplier
        self.current_performance_score = 0.5
        
        # Reset statistics
        self.coaching_stats = {
            "total_sessions": 0,
            "penalties_applied": 0,
            "total_penalty_amount": 0.0,
            "discipline_score": 1.0,
            "improvement_rate": 0.0,
            "violations": 0,
            "compliance_rate": 1.0
        }
        
        # Reset analytics
        self.coaching_analytics.clear()
        self.regime_performance.clear()
        self.session_performance.clear()
        
        # Reset learning
        self.learning_history.clear()
        self.effectiveness_scores.clear()
        
        self.log_operator_info("🔄 Role Coach reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using fallback mode")
            penalty = self._process_legacy_step(**kwargs)
            return
        
        # Extract comprehensive context
        context = extract_standard_context(info_bus)
        
        # Update market context awareness
        self._update_market_context(context, info_bus)
        
        # Extract trading activity from InfoBus
        trading_activity = self._extract_trading_activity_from_info_bus(info_bus)
        
        # Conduct coaching session
        coaching_result = self._conduct_coaching_session(trading_activity, context)
        
        # Update adaptive parameters
        self._update_adaptive_parameters(coaching_result, context)
        
        # Analyze coaching effectiveness
        self._analyze_coaching_effectiveness(coaching_result, context)
        
        # Update InfoBus with results
        self._update_info_bus(info_bus, coaching_result)
        
        # Record audit for coaching sessions
        self._record_coaching_audit(info_bus, context, coaching_result)
        
        # Update performance metrics
        self._update_coaching_performance_metrics()

    def _extract_trading_activity_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract trading activity data from InfoBus"""
        
        activity = {}
        
        try:
            # Get recent trades
            recent_trades = info_bus.get('recent_trades', [])
            activity['recent_trades'] = recent_trades
            activity['trade_count'] = len(recent_trades)
            
            # Get pending orders
            pending_orders = info_bus.get('pending_orders', [])
            activity['pending_orders'] = pending_orders
            activity['order_count'] = len(pending_orders)
            
            # Get current positions
            positions = InfoBusExtractor.get_positions(info_bus)
            activity['positions'] = positions
            activity['position_count'] = len(positions)
            
            # Calculate trading intensity
            total_activity = activity['trade_count'] + activity['order_count']
            activity['trading_intensity'] = total_activity
            
            # Get performance data
            risk_data = info_bus.get('risk', {})
            current_balance = risk_data.get('balance', risk_data.get('equity', 0))
            activity['current_balance'] = current_balance
            
            # Calculate recent PnL
            recent_pnl = sum(trade.get('pnl', 0) for trade in recent_trades)
            activity['recent_pnl'] = recent_pnl
            
            # Get trade timing information
            activity['trade_timing'] = self._analyze_trade_timing(recent_trades)
            
        except Exception as e:
            self.log_operator_warning(f"Trading activity extraction failed: {e}")
            # Provide safe defaults
            activity = {
                'recent_trades': [],
                'trade_count': 0,
                'pending_orders': [],
                'order_count': 0,
                'positions': [],
                'position_count': 0,
                'trading_intensity': 0,
                'current_balance': 10000,
                'recent_pnl': 0.0,
                'trade_timing': {'frequency': 'normal', 'clustering': False}
            }
        
        return activity

    def _analyze_trade_timing(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade timing patterns"""
        
        try:
            if len(trades) < 2:
                return {'frequency': 'low', 'clustering': False, 'intervals': []}
            
            # Calculate intervals between trades
            timestamps = []
            for trade in trades:
                timestamp_str = trade.get('timestamp', '')
                if timestamp_str:
                    try:
                        timestamp = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        timestamps.append(timestamp)
                    except:
                        continue
            
            if len(timestamps) < 2:
                return {'frequency': 'low', 'clustering': False, 'intervals': []}
            
            timestamps.sort()
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            
            # Analyze frequency
            avg_interval = np.mean(intervals) if intervals else 3600
            if avg_interval < 300:  # 5 minutes
                frequency = 'high'
            elif avg_interval < 1800:  # 30 minutes
                frequency = 'medium'
            else:
                frequency = 'low'
            
            # Detect clustering (rapid consecutive trades)
            clustering = any(interval < 60 for interval in intervals)  # Trades within 1 minute
            
            return {
                'frequency': frequency,
                'clustering': clustering,
                'intervals': intervals,
                'avg_interval': avg_interval,
                'min_interval': min(intervals) if intervals else 0,
                'rapid_trades': sum(1 for interval in intervals if interval < 300)
            }
            
        except Exception as e:
            self.log_operator_warning(f"Trade timing analysis failed: {e}")
            return {'frequency': 'normal', 'clustering': False, 'intervals': []}

    def _update_market_context(self, context: Dict[str, Any], info_bus: InfoBus) -> None:
        """Update market context awareness"""
        
        try:
            # Update regime tracking
            old_regime = self.market_regime
            self.market_regime = context.get('regime', 'unknown')
            self.volatility_regime = context.get('volatility_level', 'medium')
            self.market_session = context.get('session', 'unknown')
            
            # Log regime changes and adapt coaching
            if self.market_regime != old_regime:
                self._adapt_coaching_for_regime_change(old_regime, self.market_regime)
                
                self.log_operator_info(
                    f"📊 Regime change detected: {old_regime} → {self.market_regime}",
                    coaching_adaptation="Trade limits adjusted",
                    session=self.market_session
                )
            
        except Exception as e:
            self.log_operator_warning(f"Market context update failed: {e}")

    def _adapt_coaching_for_regime_change(self, old_regime: str, new_regime: str) -> None:
        """Adapt coaching parameters for regime change"""
        
        try:
            # Adjust adaptive max trades based on regime
            regime_multiplier = self.regime_multipliers.get(new_regime, 1.0)
            self.adaptive_max_trades = int(self.max_trades * regime_multiplier)
            
            # Apply regime sensitivity
            adjustment_factor = 1.0 + (regime_multiplier - 1.0) * self.regime_sensitivity
            self.adaptive_max_trades = int(self.max_trades * adjustment_factor)
            
            # Ensure minimum of 1 trade
            self.adaptive_max_trades = max(1, self.adaptive_max_trades)
            
        except Exception as e:
            self.log_operator_warning(f"Regime adaptation failed: {e}")

    def _conduct_coaching_session(self, trading_activity: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive coaching session"""
        
        session_result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'coaching_mode': self.coaching_mode,
            'trade_analysis': {},
            'discipline_assessment': {},
            'penalties': {},
            'recommendations': [],
            'context': context.copy()
        }
        
        try:
            # Get effective trade limits
            effective_limits = self._calculate_effective_trade_limits(context)
            
            # Analyze trading activity
            trade_analysis = self._analyze_trading_activity(trading_activity, effective_limits, context)
            session_result['trade_analysis'] = trade_analysis
            
            # Assess discipline
            discipline_assessment = self._assess_trading_discipline(trade_analysis, context)
            session_result['discipline_assessment'] = discipline_assessment
            
            # Calculate penalties
            penalties = self._calculate_context_aware_penalties(discipline_assessment, context)
            session_result['penalties'] = penalties
            
            # Generate recommendations
            recommendations = self._generate_coaching_recommendations(trade_analysis, discipline_assessment, context)
            session_result['recommendations'] = recommendations
            
            # Update coaching session history
            self.coaching_sessions.append(session_result)
            
            # Update statistics
            self.coaching_stats['total_sessions'] += 1
            if penalties.get('total_penalty', 0) > 0:
                self.coaching_stats['penalties_applied'] += 1
                self.coaching_stats['total_penalty_amount'] += penalties['total_penalty']
                
            if discipline_assessment.get('violations', 0) > 0:
                self.coaching_stats['violations'] += discipline_assessment['violations']
            
            # Update compliance rate
            total_sessions = self.coaching_stats['total_sessions']
            violations = self.coaching_stats['violations']
            self.coaching_stats['compliance_rate'] = 1.0 - (violations / max(total_sessions, 1))
            
            # Update discipline score
            self.coaching_stats['discipline_score'] = discipline_assessment.get('overall_score', 1.0)
            
        except Exception as e:
            self.log_operator_error(f"Coaching session failed: {e}")
            session_result['error'] = str(e)
        
        return session_result

    def _calculate_effective_trade_limits(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate effective trade limits based on context"""
        
        try:
            # Start with adaptive limits
            base_limit = self.adaptive_max_trades
            
            # Apply session-specific adjustments
            if self.session_aware:
                session = context.get('session', 'unknown')
                session_limit = self.session_limits.get(session, base_limit)
                base_limit = session_limit
            
            # Apply volatility adjustments
            if self.volatility_scaling:
                vol_level = context.get('volatility_level', 'medium')
                vol_adjustments = {
                    'low': 1.2,
                    'medium': 1.0,
                    'high': 0.8,
                    'extreme': 0.6
                }
                base_limit = int(base_limit * vol_adjustments.get(vol_level, 1.0))
            
            # Apply performance adjustments
            if self.performance_adjustment:
                if self.current_performance_score > 0.8:
                    base_limit = int(base_limit * 1.2)  # Reward good performance
                elif self.current_performance_score < 0.3:
                    base_limit = int(base_limit * 0.7)  # Restrict poor performance
            
            # Ensure minimum
            base_limit = max(1, base_limit)
            
            return {
                'max_trades': base_limit,
                'base_trades': self.max_trades,
                'adaptive_trades': self.adaptive_max_trades,
                'session_adjustment': self.session_limits.get(context.get('session', 'unknown'), 1.0),
                'volatility_adjustment': vol_adjustments.get(context.get('volatility_level', 'medium'), 1.0) if self.volatility_scaling else 1.0,
                'performance_adjustment': 1.2 if self.current_performance_score > 0.8 else (0.7 if self.current_performance_score < 0.3 else 1.0) if self.performance_adjustment else 1.0
            }
            
        except Exception as e:
            self.log_operator_warning(f"Effective limits calculation failed: {e}")
            return {'max_trades': self.max_trades, 'base_trades': self.max_trades}

    def _analyze_trading_activity(self, trading_activity: Dict[str, Any], 
                                 effective_limits: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading activity against limits and context"""
        
        try:
            trade_count = trading_activity.get('trade_count', 0)
            max_trades = effective_limits.get('max_trades', self.max_trades)
            trading_intensity = trading_activity.get('trading_intensity', 0)
            trade_timing = trading_activity.get('trade_timing', {})
            
            # Basic compliance check
            over_limit = max(0, trade_count - max_trades)
            compliance = trade_count <= max_trades
            
            # Intensity analysis
            intensity_level = 'low'
            if trading_intensity > max_trades * 2:
                intensity_level = 'excessive'
            elif trading_intensity > max_trades * 1.5:
                intensity_level = 'high'
            elif trading_intensity > max_trades:
                intensity_level = 'moderate'
            
            # Timing analysis
            timing_issues = []
            if trade_timing.get('clustering', False):
                timing_issues.append('rapid_clustering')
            if trade_timing.get('frequency') == 'high':
                timing_issues.append('high_frequency')
            if trade_timing.get('rapid_trades', 0) > 3:
                timing_issues.append('excessive_rapid_trades')
            
            # Context-specific analysis
            regime = context.get('regime', 'unknown')
            context_appropriateness = 'appropriate'
            
            if regime == 'volatile' and trade_count > max_trades * 0.8:
                context_appropriateness = 'questionable'
            elif regime == 'ranging' and trade_count > max_trades * 1.2:
                context_appropriateness = 'excessive'
            
            return {
                'trade_count': trade_count,
                'max_trades': max_trades,
                'over_limit': over_limit,
                'compliance': compliance,
                'trading_intensity': trading_intensity,
                'intensity_level': intensity_level,
                'timing_issues': timing_issues,
                'context_appropriateness': context_appropriateness,
                'trade_timing': trade_timing,
                'recent_pnl': trading_activity.get('recent_pnl', 0.0)
            }
            
        except Exception as e:
            self.log_operator_warning(f"Trading activity analysis failed: {e}")
            return {'trade_count': 0, 'compliance': True, 'over_limit': 0}

    def _assess_trading_discipline(self, trade_analysis: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall trading discipline"""
        
        try:
            # Compliance score
            compliance_score = 1.0 if trade_analysis.get('compliance', True) else 0.0
            
            # Intensity score
            intensity_level = trade_analysis.get('intensity_level', 'low')
            intensity_scores = {
                'low': 1.0,
                'moderate': 0.8,
                'high': 0.5,
                'excessive': 0.2
            }
            intensity_score = intensity_scores.get(intensity_level, 0.5)
            
            # Timing score
            timing_issues = trade_analysis.get('timing_issues', [])
            timing_score = max(0.0, 1.0 - len(timing_issues) * 0.2)
            
            # Context appropriateness score
            context_appropriateness = trade_analysis.get('context_appropriateness', 'appropriate')
            context_scores = {
                'appropriate': 1.0,
                'questionable': 0.6,
                'excessive': 0.2
            }
            context_score = context_scores.get(context_appropriateness, 0.5)
            
            # Performance impact score
            recent_pnl = trade_analysis.get('recent_pnl', 0.0)
            if recent_pnl > 0:
                performance_score = min(1.0, 0.5 + recent_pnl / 1000.0)  # Normalize PnL
            else:
                performance_score = max(0.0, 0.5 + recent_pnl / 1000.0)
            
            # Overall discipline score (weighted average)
            weights = {'compliance': 0.3, 'intensity': 0.2, 'timing': 0.2, 'context': 0.2, 'performance': 0.1}
            overall_score = (
                compliance_score * weights['compliance'] +
                intensity_score * weights['intensity'] +
                timing_score * weights['timing'] +
                context_score * weights['context'] +
                performance_score * weights['performance']
            )
            
            # Count violations
            violations = 0
            if not trade_analysis.get('compliance', True):
                violations += 1
            if intensity_level in ['high', 'excessive']:
                violations += 1
            if len(timing_issues) > 1:
                violations += 1
            if context_appropriateness in ['questionable', 'excessive']:
                violations += 1
            
            return {
                'compliance_score': compliance_score,
                'intensity_score': intensity_score,
                'timing_score': timing_score,
                'context_score': context_score,
                'performance_score': performance_score,
                'overall_score': overall_score,
                'violations': violations,
                'discipline_grade': self._get_discipline_grade(overall_score)
            }
            
        except Exception as e:
            self.log_operator_warning(f"Discipline assessment failed: {e}")
            return {'overall_score': 0.5, 'violations': 0, 'discipline_grade': 'C'}

    def _get_discipline_grade(self, score: float) -> str:
        """Convert discipline score to letter grade"""
        
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        elif score >= 0.5:
            return 'D'
        else:
            return 'F'

    def _calculate_context_aware_penalties(self, discipline_assessment: Dict[str, Any], 
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate context-aware penalties"""
        
        try:
            # Base penalty calculation
            violations = discipline_assessment.get('violations', 0)
            overall_score = discipline_assessment.get('overall_score', 1.0)
            
            # Calculate base penalty
            base_penalty = violations * self.adaptive_penalty
            
            # Apply score-based penalty
            if overall_score < 0.5:
                score_penalty = (0.5 - overall_score) * 2.0 * self.adaptive_penalty
            else:
                score_penalty = 0.0
            
            # Apply coaching mode adjustments
            mode_multipliers = {
                'strict': 1.5,
                'adaptive': 1.0,
                'lenient': 0.7,
                'performance_based': 1.2,
                'regime_aware': 1.0
            }
            mode_multiplier = mode_multipliers.get(self.coaching_mode, 1.0)
            
            # Apply regime-specific adjustments
            regime = context.get('regime', 'unknown')
            regime_penalty_multipliers = {
                'volatile': 1.3,  # Higher penalties in volatile markets
                'trending': 0.9,  # Lower penalties in trending markets
                'ranging': 1.0,
                'unknown': 1.0
            }
            regime_multiplier = regime_penalty_multipliers.get(regime, 1.0)
            
            # Apply session adjustments
            session = context.get('session', 'unknown')
            session_multipliers = {
                'asian': 1.2,  # Higher penalties during low liquidity
                'rollover': 1.4,  # Highest penalties during rollover
                'european': 1.0,
                'american': 1.0
            }
            session_multiplier = session_multipliers.get(session, 1.0)
            
            # Calculate total penalty
            total_penalty = (base_penalty + score_penalty) * mode_multiplier * regime_multiplier * session_multiplier
            
            # Apply penalty decay for learning
            if len(self.penalty_history) > 0:
                recent_penalties = list(self.penalty_history)[-5:]
                if all(p > 0 for p in recent_penalties):  # Consecutive penalties
                    total_penalty *= self.penalty_decay  # Reduce for learning
            
            # Store penalty
            self.penalty_history.append(total_penalty)
            
            return {
                'base_penalty': base_penalty,
                'score_penalty': score_penalty,
                'mode_multiplier': mode_multiplier,
                'regime_multiplier': regime_multiplier,
                'session_multiplier': session_multiplier,
                'total_penalty': float(total_penalty),
                'penalty_rationale': f"Violations: {violations}, Score: {overall_score:.1%}, Mode: {self.coaching_mode}"
            }
            
        except Exception as e:
            self.log_operator_warning(f"Penalty calculation failed: {e}")
            return {'total_penalty': 0.0, 'penalty_rationale': 'Calculation failed'}

    def _generate_coaching_recommendations(self, trade_analysis: Dict[str, Any], 
                                          discipline_assessment: Dict[str, Any], 
                                          context: Dict[str, Any]) -> List[str]:
        """Generate intelligent coaching recommendations"""
        
        recommendations = []
        
        try:
            overall_score = discipline_assessment.get('overall_score', 1.0)
            violations = discipline_assessment.get('violations', 0)
            
            # Basic compliance recommendations
            if not trade_analysis.get('compliance', True):
                over_limit = trade_analysis.get('over_limit', 0)
                recommendations.append(f"🚫 Trade limit exceeded by {over_limit}. Consider reducing position sizes or waiting for better setups.")
            
            # Intensity recommendations
            intensity_level = trade_analysis.get('intensity_level', 'low')
            if intensity_level == 'excessive':
                recommendations.append("⚡ Excessive trading intensity detected. Take a break and reassess your strategy.")
            elif intensity_level == 'high':
                recommendations.append("⚠️ High trading intensity. Consider reducing frequency to improve decision quality.")
            
            # Timing recommendations
            timing_issues = trade_analysis.get('timing_issues', [])
            if 'rapid_clustering' in timing_issues:
                recommendations.append("⏰ Rapid trade clustering detected. Allow more time between trades for better analysis.")
            if 'high_frequency' in timing_issues:
                recommendations.append("📊 High-frequency trading pattern. Consider longer-term setups for better risk-reward.")
            
            # Context-specific recommendations
            regime = context.get('regime', 'unknown')
            vol_level = context.get('volatility_level', 'medium')
            
            if regime == 'volatile' and trade_analysis.get('trade_count', 0) > 1:
                recommendations.append("💥 Volatile market conditions. Consider reducing trade frequency and increasing caution.")
            elif regime == 'ranging' and trade_analysis.get('trade_count', 0) > 2:
                recommendations.append("↔️ Ranging market. Focus on breakout trades rather than frequent small moves.")
            elif regime == 'trending' and trade_analysis.get('trade_count', 0) == 0:
                recommendations.append("📈 Trending market opportunity. Consider capturing trend momentum with disciplined entries.")
            
            if vol_level == 'extreme':
                recommendations.append("🌪️ Extreme volatility. Reduce position sizes and consider wider stops.")
            
            # Performance-based recommendations
            recent_pnl = trade_analysis.get('recent_pnl', 0.0)
            if recent_pnl < -100:
                recommendations.append("📉 Recent losses detected. Consider taking a break to reassess strategy.")
            elif recent_pnl > 100:
                recommendations.append("📈 Good performance! Maintain discipline to protect gains.")
            
            # Discipline improvement recommendations
            if overall_score < 0.6:
                recommendations.append("🎯 Discipline needs improvement. Focus on quality over quantity in trade selection.")
            elif overall_score > 0.8:
                recommendations.append("✅ Excellent discipline! Continue following your trading plan.")
            
            # Session-specific recommendations
            session = context.get('session', 'unknown')
            if session == 'asian' and trade_analysis.get('trade_count', 0) > 1:
                recommendations.append("🌏 Asian session - lower liquidity. Be extra selective with trades.")
            elif session == 'rollover':
                recommendations.append("🔄 Rollover period. Avoid trading due to potential spread widening.")
            
            # Learning recommendations
            if len(self.penalty_history) > 3 and all(p > 0 for p in list(self.penalty_history)[-3:]):
                recommendations.append("📚 Consistent discipline issues. Consider reviewing your trading rules and psychology.")
            
        except Exception as e:
            self.log_operator_warning(f"Recommendation generation failed: {e}")
            recommendations.append("⚠️ Unable to generate specific recommendations at this time.")
        
        return recommendations[:5]  # Limit to top 5 recommendations

    def _update_adaptive_parameters(self, coaching_result: Dict[str, Any], 
                                   context: Dict[str, Any]) -> None:
        """Update adaptive coaching parameters"""
        
        try:
            discipline_assessment = coaching_result.get('discipline_assessment', {})
            overall_score = discipline_assessment.get('overall_score', 0.5)
            
            # Update performance score with exponential moving average
            self.current_performance_score = (
                self.current_performance_score * (1 - self.learning_rate) +
                overall_score * self.learning_rate
            )
            
            # Adapt penalty based on effectiveness
            if len(self.effectiveness_scores) > 5:
                avg_effectiveness = np.mean(list(self.effectiveness_scores)[-5:])
                if avg_effectiveness < 0.4:  # Poor effectiveness
                    self.adaptive_penalty *= 1.05  # Increase penalty
                elif avg_effectiveness > 0.8:  # High effectiveness
                    self.adaptive_penalty *= 0.98  # Slightly decrease penalty
            
            # Store learning data
            self.learning_history.append({
                'timestamp': coaching_result.get('timestamp'),
                'performance_score': self.current_performance_score,
                'overall_score': overall_score,
                'context': context.copy()
            })
            
        except Exception as e:
            self.log_operator_warning(f"Adaptive parameter update failed: {e}")

    def _analyze_coaching_effectiveness(self, coaching_result: Dict[str, Any], 
                                      context: Dict[str, Any]) -> None:
        """Analyze coaching effectiveness"""
        
        try:
            discipline_assessment = coaching_result.get('discipline_assessment', {})
            penalties = coaching_result.get('penalties', {})
            
            # Calculate effectiveness score
            overall_score = discipline_assessment.get('overall_score', 0.5)
            total_penalty = penalties.get('total_penalty', 0.0)
            
            # Effectiveness based on improvement and appropriate penalties
            if overall_score > 0.8:
                effectiveness = 0.9  # High discipline = high effectiveness
            elif overall_score > 0.6:
                effectiveness = 0.7
            elif total_penalty > 0 and overall_score < 0.5:
                effectiveness = 0.6  # Appropriate penalty for poor discipline
            else:
                effectiveness = 0.4  # Low effectiveness
            
            # Adjust for context appropriateness
            trade_analysis = coaching_result.get('trade_analysis', {})
            context_appropriateness = trade_analysis.get('context_appropriateness', 'appropriate')
            if context_appropriateness == 'appropriate':
                effectiveness += 0.1
            elif context_appropriateness == 'excessive':
                effectiveness -= 0.1
            
            # Store effectiveness
            self.effectiveness_scores.append(effectiveness)
            
            # Update improvement rate
            if len(list(self.effectiveness_scores)) >= 5:
                recent_scores = list(self.effectiveness_scores)[-5:]
                older_scores = list(self.effectiveness_scores)[-10:-5] if len(self.effectiveness_scores) >= 10 else []
                
                if older_scores:
                    recent_avg = np.mean(recent_scores)
                    older_avg = np.mean(older_scores)
                    self.coaching_stats['improvement_rate'] = recent_avg - older_avg
                else:
                    self.coaching_stats['improvement_rate'] = 0.0
            
        except Exception as e:
            self.log_operator_warning(f"Effectiveness analysis failed: {e}")

    def _update_info_bus(self, info_bus: InfoBus, coaching_result: Dict[str, Any]) -> None:
        """Update InfoBus with coaching results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'role_coach', {
            'coaching_mode': self.coaching_mode,
            'max_trades': self.max_trades,
            'adaptive_max_trades': self.adaptive_max_trades,
            'penalty_multiplier': self.penalty_multiplier,
            'adaptive_penalty': self.adaptive_penalty,
            'coaching_stats': self.coaching_stats.copy(),
            'current_performance_score': self.current_performance_score,
            'discipline_assessment': coaching_result.get('discipline_assessment', {}),
            'penalties': coaching_result.get('penalties', {}),
            'recommendations': coaching_result.get('recommendations', []),
            'market_context': {
                'regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'session': self.market_session
            }
        })
        
        # Add penalty to risk snapshot if applied
        penalties = coaching_result.get('penalties', {})
        total_penalty = penalties.get('total_penalty', 0.0)
        
        if total_penalty > 0:
            InfoBusUpdater.update_risk_snapshot(info_bus, {
                'discipline_penalty': total_penalty,
                'discipline_score': self.coaching_stats.get('discipline_score', 1.0),
                'compliance_rate': self.coaching_stats.get('compliance_rate', 1.0)
            })
        
        # Add alerts for significant discipline issues
        discipline_assessment = coaching_result.get('discipline_assessment', {})
        violations = discipline_assessment.get('violations', 0)
        overall_score = discipline_assessment.get('overall_score', 1.0)
        
        if violations > 2:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Multiple discipline violations: {violations} issues detected",
                severity="warning",
                module="RoleCoach"
            )
        elif overall_score < 0.4:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Poor trading discipline: {overall_score:.1%} score",
                severity="critical",
                module="RoleCoach"
            )

    def _record_coaching_audit(self, info_bus: InfoBus, context: Dict[str, Any], 
                              coaching_result: Dict[str, Any]) -> None:
        """Record comprehensive audit trail"""
        
        # Only audit coaching sessions with penalties or violations
        should_audit = (
            coaching_result.get('penalties', {}).get('total_penalty', 0) > 0 or
            coaching_result.get('discipline_assessment', {}).get('violations', 0) > 0 or
            info_bus.get('step_idx', 0) % 25 == 0
        )
        
        if should_audit:
            audit_data = {
                'coaching_session': {
                    'mode': self.coaching_mode,
                    'max_trades': self.max_trades,
                    'adaptive_max_trades': self.adaptive_max_trades,
                    'penalty_multiplier': self.adaptive_penalty
                },
                'trade_analysis': coaching_result.get('trade_analysis', {}),
                'discipline_assessment': coaching_result.get('discipline_assessment', {}),
                'penalties': coaching_result.get('penalties', {}),
                'recommendations': coaching_result.get('recommendations', []),
                'context': context.copy(),
                'statistics': self.coaching_stats.copy(),
                'performance': {
                    'current_score': self.current_performance_score,
                    'improvement_rate': self.coaching_stats.get('improvement_rate', 0.0),
                    'effectiveness_score': list(self.effectiveness_scores)[-1] if self.effectiveness_scores else 0.5
                }
            }
            
            violations = coaching_result.get('discipline_assessment', {}).get('violations', 0)
            severity = "critical" if violations > 2 else "warning" if violations > 0 else "info"
            
            self.audit_tracker.record_event(
                event_type="coaching_session",
                module="RoleCoach",
                details=audit_data,
                severity=severity
            )

    def _update_coaching_performance_metrics(self) -> None:
        """Update performance metrics"""
        
        # Update performance metrics
        self._update_performance_metric('total_sessions', self.coaching_stats['total_sessions'])
        self._update_performance_metric('penalties_applied', self.coaching_stats['penalties_applied'])
        self._update_performance_metric('discipline_score', self.coaching_stats['discipline_score'])
        self._update_performance_metric('compliance_rate', self.coaching_stats['compliance_rate'])
        self._update_performance_metric('improvement_rate', self.coaching_stats['improvement_rate'])
        self._update_performance_metric('current_performance_score', self.current_performance_score)

    def _process_legacy_step(self, **kwargs) -> float:
        """Process legacy step parameters for backward compatibility"""
        
        try:
            # Extract legacy trades data
            trades = kwargs.get('trades', [])
            
            # Basic legacy coaching
            trade_count = len(trades)
            over_limit = max(0, trade_count - self.max_trades)
            penalty = over_limit * self.penalty_multiplier
            
            # Update basic statistics
            self.coaching_stats['total_sessions'] += 1
            if penalty > 0:
                self.coaching_stats['penalties_applied'] += 1
                self.coaching_stats['total_penalty_amount'] += penalty
            
            # Log coaching result
            if over_limit > 0:
                self.log_operator_warning(
                    f"🎯 Trade discipline violation",
                    trades=f"{trade_count}/{self.max_trades}",
                    penalty=f"{penalty:.2f}",
                    over_limit=over_limit
                )
            else:
                self.log_operator_info(
                    f"✅ Trade discipline maintained",
                    trades=f"{trade_count}/{self.max_trades}",
                    penalty="none"
                )
            
            return penalty
            
        except Exception as e:
            self.log_operator_error(f"Legacy step processing failed: {e}")
            return 0.0

    # ================== PUBLIC INTERFACE METHODS ==================

    def get_coaching_penalty(self, trade_count: int, context: Optional[Dict[str, Any]] = None) -> float:
        """Get coaching penalty for given trade count (legacy interface)"""
        
        if context is None:
            context = {'regime': 'unknown', 'volatility_level': 'medium', 'session': 'unknown'}
        
        # Calculate effective limits
        effective_limits = self._calculate_effective_trade_limits(context)
        max_trades = effective_limits.get('max_trades', self.max_trades)
        
        # Calculate penalty
        over_limit = max(0, trade_count - max_trades)
        penalty = over_limit * self.adaptive_penalty
        
        return penalty

    def get_observation_components(self) -> np.ndarray:
        """Return coaching features for observation"""
        
        try:
            mode_idx = float(list(self.COACHING_MODES.keys()).index(self.coaching_mode))
            recent_penalty = list(self.penalty_history)[-1] if self.penalty_history else 0.0
            
            return np.array([
                float(self.max_trades),
                float(self.adaptive_max_trades),
                float(self.penalty_multiplier),
                float(self.adaptive_penalty),
                mode_idx / len(self.COACHING_MODES),
                float(self.current_performance_score),
                float(self.coaching_stats.get('discipline_score', 1.0)),
                float(min(1.0, recent_penalty / 10.0))  # Normalize penalty
            ], dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.array([2.0, 2.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32)

    def get_role_coaching_report(self) -> str:
        """Generate operator-friendly coaching report"""
        
        # Performance status
        discipline_score = self.coaching_stats.get('discipline_score', 1.0)
        if discipline_score > 0.8:
            discipline_status = "✅ Excellent"
        elif discipline_score > 0.6:
            discipline_status = "⚡ Good"
        elif discipline_score > 0.4:
            discipline_status = "⚠️ Needs Improvement"
        else:
            discipline_status = "🚨 Poor"
        
        # Recent coaching sessions
        session_lines = []
        for session in list(self.coaching_sessions)[-3:]:
            timestamp = session['timestamp'][:19]
            penalties = session.get('penalties', {})
            penalty = penalties.get('total_penalty', 0.0)
            violations = session.get('discipline_assessment', {}).get('violations', 0)
            
            if penalty > 0:
                emoji = "🚨" if violations > 2 else "⚠️"
                session_lines.append(f"  {emoji} {timestamp}: {violations} violations, penalty {penalty:.2f}")
            else:
                session_lines.append(f"  ✅ {timestamp}: No violations, good discipline")
        
        # Recent recommendations
        latest_session = list(self.coaching_sessions)[-1] if self.coaching_sessions else {}
        recommendations = latest_session.get('recommendations', [])
        rec_lines = [f"  • {rec}" for rec in recommendations[:3]]
        
        return f"""
🎯 ROLE COACH
═══════════════════════════════════════
🏆 Discipline Status: {discipline_status} ({discipline_score:.1%})
📊 Coaching Mode: {self.coaching_mode.title().replace('_', ' ')}
🎯 Trade Limits: Base {self.max_trades} | Adaptive {self.adaptive_max_trades}
⚖️ Penalty Scale: Base {self.penalty_multiplier:.1f} | Adaptive {self.adaptive_penalty:.1f}

📈 COACHING CONFIGURATION
• Regime Sensitivity: {self.regime_sensitivity:.1%}
• Performance Adjustment: {'✅ Enabled' if self.performance_adjustment else '❌ Disabled'}
• Session Awareness: {'✅ Enabled' if self.session_aware else '❌ Disabled'}
• Volatility Scaling: {'✅ Enabled' if self.volatility_scaling else '❌ Disabled'}
• Learning Rate: {self.learning_rate:.1%}
• Penalty Decay: {self.penalty_decay:.1%}

📊 PERFORMANCE STATISTICS
• Total Sessions: {self.coaching_stats['total_sessions']:,}
• Penalties Applied: {self.coaching_stats['penalties_applied']:,}
• Total Penalty Amount: {self.coaching_stats['total_penalty_amount']:.2f}
• Compliance Rate: {self.coaching_stats['compliance_rate']:.1%}
• Violations: {self.coaching_stats['violations']}
• Improvement Rate: {self.coaching_stats['improvement_rate']:+.1%}

🔧 ADAPTIVE PARAMETERS
• Current Performance Score: {self.current_performance_score:.1%}
• Market Regime: {self.market_regime.title()}
• Volatility Level: {self.volatility_regime.title()}
• Market Session: {self.market_session.title()}

📋 SESSION LIMITS BY CONTEXT
• Asian Session: {self.session_limits['asian']} trades
• European Session: {self.session_limits['european']} trades
• American Session: {self.session_limits['american']} trades
• Rollover Period: {self.session_limits['rollover']} trades

📜 RECENT COACHING SESSIONS
{chr(10).join(session_lines) if session_lines else "  📭 No recent coaching sessions"}

💡 LATEST RECOMMENDATIONS
{chr(10).join(rec_lines) if rec_lines else "  📭 No current recommendations"}

🎓 COACHING MODES AVAILABLE
• Strict: Strict discipline enforcement
• Adaptive: Context-aware discipline (Current)
• Lenient: Flexible trade management
• Performance Based: Performance-driven limits
• Regime Aware: Market regime specific rules

🎯 DISCIPLINE METRICS
• Current Score: {discipline_score:.1%}
• Compliance Rate: {self.coaching_stats['compliance_rate']:.1%}
• Effectiveness: {list(self.effectiveness_scores)[-1]:.1%} if self.effectiveness_scores else 0.5
• Learning Progress: {len(self.learning_history)} sessions tracked
        """

    # ================== EVOLUTIONARY METHODS ==================

    def mutate(self, std: float = 1.0) -> None:
        """Mutate coaching parameters"""
        
        old_max_trades = self.max_trades
        old_penalty = self.penalty_multiplier
        
        # Mutate max trades
        self.max_trades = max(1, self.max_trades + np.random.randint(-1, 2))
        
        # Mutate penalty
        self.penalty_multiplier = float(np.clip(
            self.penalty_multiplier + np.random.normal(0, std),
            0.1, 10.0
        ))
        
        # Update adaptive parameters
        self.adaptive_max_trades = self.max_trades
        self.adaptive_penalty = self.penalty_multiplier
        
        self.log_operator_info(
            f"🧬 Coaching mutation applied",
            max_trades=f"{old_max_trades} → {self.max_trades}",
            penalty=f"{old_penalty:.1f} → {self.penalty_multiplier:.1f}"
        )

    def crossover(self, other: "RoleCoach") -> "RoleCoach":
        """Create offspring through crossover"""
        
        # Select parameters from parents
        max_trades = self.max_trades if np.random.random() < 0.5 else other.max_trades
        penalty = self.penalty_multiplier if np.random.random() < 0.5 else other.penalty_multiplier
        mode = self.coaching_mode if np.random.random() < 0.5 else other.coaching_mode
        
        # Create offspring
        offspring = RoleCoach(
            max_trades=max_trades,
            penalty_multiplier=penalty,
            coaching_mode=mode,
            debug=self.config.debug
        )
        
        # Mix other parameters
        offspring.regime_sensitivity = (self.regime_sensitivity + other.regime_sensitivity) / 2
        offspring.performance_adjustment = self.performance_adjustment if np.random.random() < 0.5 else other.performance_adjustment
        offspring.session_aware = self.session_aware if np.random.random() < 0.5 else other.session_aware
        
        self.log_operator_info(
            f"🔬 Coaching crossover created offspring",
            max_trades=max_trades,
            penalty=f"{penalty:.1f}",
            mode=mode
        )
        
        return offspring

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "max_trades": self.max_trades,
                "penalty_multiplier": self.penalty_multiplier,
                "coaching_mode": self.coaching_mode,
                "regime_sensitivity": self.regime_sensitivity,
                "performance_adjustment": self.performance_adjustment,
                "session_aware": self.session_aware,
                "volatility_scaling": self.volatility_scaling
            },
            "adaptive_parameters": {
                "adaptive_max_trades": self.adaptive_max_trades,
                "adaptive_penalty": self.adaptive_penalty,
                "current_performance_score": self.current_performance_score
            },
            "market_context": {
                "regime": self.market_regime,
                "volatility_regime": self.volatility_regime,
                "session": self.market_session
            },
            "statistics": self.coaching_stats.copy(),
            "history": {
                "penalty_history": list(self.penalty_history)[-20:],
                "effectiveness_scores": list(self.effectiveness_scores)[-10:],
                "learning_history": list(self.learning_history)[-10:]
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.max_trades = int(config.get("max_trades", self.max_trades))
        self.penalty_multiplier = float(config.get("penalty_multiplier", self.penalty_multiplier))
        self.coaching_mode = config.get("coaching_mode", self.coaching_mode)
        self.regime_sensitivity = float(config.get("regime_sensitivity", self.regime_sensitivity))
        self.performance_adjustment = bool(config.get("performance_adjustment", self.performance_adjustment))
        self.session_aware = bool(config.get("session_aware", self.session_aware))
        self.volatility_scaling = bool(config.get("volatility_scaling", self.volatility_scaling))
        
        # Load adaptive parameters
        adaptive = state.get("adaptive_parameters", {})
        self.adaptive_max_trades = int(adaptive.get("adaptive_max_trades", self.max_trades))
        self.adaptive_penalty = float(adaptive.get("adaptive_penalty", self.penalty_multiplier))
        self.current_performance_score = float(adaptive.get("current_performance_score", 0.5))
        
        # Load market context
        context = state.get("market_context", {})
        self.market_regime = context.get("regime", "normal")
        self.volatility_regime = context.get("volatility_regime", "medium")
        self.market_session = context.get("session", "unknown")
        
        # Load statistics
        self.coaching_stats.update(state.get("statistics", {}))
        
        # Load history
        history = state.get("history", {})
        
        penalty_history = history.get("penalty_history", [])
        self.penalty_history.clear()
        for penalty in penalty_history:
            self.penalty_history.append(penalty)
            
        effectiveness_scores = history.get("effectiveness_scores", [])
        self.effectiveness_scores.clear()
        for score in effectiveness_scores:
            self.effectiveness_scores.append(score)
            
        learning_history = history.get("learning_history", [])
        self.learning_history.clear()
        for entry in learning_history:
            self.learning_history.append(entry)

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> float:
        """Legacy step interface for backward compatibility"""
        return self._process_legacy_step(**kwargs)