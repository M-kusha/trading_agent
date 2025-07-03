# ─────────────────────────────────────────────────────────────
# File: modules/meta/metacognitive_planner.py
# Enhanced with InfoBus integration & intelligent automation
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from enum import Enum
import json

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class PlanningPhase(Enum):
    """Metacognitive planning phases"""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    ADAPTATION = "adaptation"


class MetaCognitivePlanner(Module, AnalysisMixin, TradingMixin):
    """
    Enhanced metacognitive planner with InfoBus integration and intelligent automation.
    Provides high-level strategic planning and adaptation for the trading system.
    Monitors performance patterns and adapts strategies dynamically.
    """
    
    def __init__(self, window: int = 20, debug: bool = True,
                 planning_horizon: int = 100,
                 adaptation_threshold: float = 0.15,
                 **kwargs):
        
        # Enhanced configuration
        config = ModuleConfig(
            debug=debug,
            max_history=1000,
            health_check_interval=300,  # 5 minutes
            performance_window=50,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.window = window
        self.planning_horizon = planning_horizon
        self.adaptation_threshold = adaptation_threshold
        
        # Planning state
        self.current_phase = PlanningPhase.ANALYSIS
        self.phase_start_time = datetime.datetime.now()
        self.planning_cycle = 0
        
        # Episode and session tracking
        self.episode_history = deque(maxlen=window)
        self.session_plans = deque(maxlen=10)
        self.adaptation_history = deque(maxlen=50)
        
        # Strategic planning components
        self.strategic_objectives = {
            'profit_target': 150.0,
            'max_drawdown': 15.0,
            'min_win_rate': 0.55,
            'risk_budget': 0.10,
            'diversification_target': 0.7,
            'adaptive_learning_rate': 0.02
        }
        
        # Planning analytics
        self.planning_effectiveness = defaultdict(lambda: {'successful': 0, 'total': 0, 'avg_outcome': 0.0})
        self.strategy_performance = defaultdict(list)
        self.market_adaptation_patterns = defaultdict(list)
        
        # Cognitive metrics
        self.cognitive_load = 0.5
        self.planning_confidence = 0.6
        self.strategy_coherence = 0.7
        self.adaptation_speed = 0.5
        
        # Planning recommendations and insights
        self.current_recommendations = []
        self.strategic_insights = deque(maxlen=20)
        self.performance_forecasts = {}
        
        # Learning and improvement
        self.learning_history = deque(maxlen=100)
        self.meta_learning_rate = 0.01
        self.strategy_evolution_trace = []
        
        # Enhanced logging with rotation
        self.logger = RotatingLogger(
            "MetaCognitivePlanner",
            "logs/strategy/meta/metacognitive_planner.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("MetaCognitivePlanner")
        
        self.log_operator_info(
            "🧠 Enhanced MetaCognitive Planner initialized",
            window=window,
            planning_horizon=planning_horizon,
            adaptation_threshold=f"{adaptation_threshold:.3f}",
            current_phase=self.current_phase.value,
            strategic_objectives=len(self.strategic_objectives)
        )
    
    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_trading_state()
        
        # Reset planning state
        self.current_phase = PlanningPhase.ANALYSIS
        self.phase_start_time = datetime.datetime.now()
        self.planning_cycle = 0
        
        # Reset tracking
        self.episode_history.clear()
        self.session_plans.clear()
        self.adaptation_history.clear()
        
        # Reset analytics
        self.planning_effectiveness.clear()
        self.strategy_performance.clear()
        self.market_adaptation_patterns.clear()
        
        # Reset cognitive metrics
        self.cognitive_load = 0.5
        self.planning_confidence = 0.6
        self.strategy_coherence = 0.7
        self.adaptation_speed = 0.5
        
        # Reset recommendations and insights
        self.current_recommendations.clear()
        self.strategic_insights.clear()
        self.performance_forecasts.clear()
        
        # Reset learning
        self.learning_history.clear()
        self.strategy_evolution_trace.clear()
        
        self.log_operator_info("🔄 MetaCognitive Planner reset - all state cleared")
    
    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and cognitive planning"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using fallback mode")
            self._process_legacy_step(**kwargs)
            return
        
        # Extract comprehensive context
        context = extract_standard_context(info_bus)
        self._update_cognitive_metrics(info_bus, context)
        
        # Execute current planning phase
        self._execute_planning_phase(info_bus, context)
        
        # Check for phase transitions
        self._evaluate_phase_transition(info_bus, context)
        
        # Update strategic insights
        self._generate_strategic_insights(info_bus, context)
        
        # Adapt objectives if needed
        self._adapt_strategic_objectives(info_bus, context)
        
        # Publish planning status
        self._publish_planning_status(info_bus)
    
    def _process_legacy_step(self, **kwargs):
        """Fallback processing for backward compatibility"""
        self.planning_cycle += 1
        
        # Basic planning cycle without InfoBus
        phase_duration = (datetime.datetime.now() - self.phase_start_time).total_seconds()
        if phase_duration > 300:  # 5 minutes per phase
            self._advance_planning_phase("Time-based transition")
    
    def _update_cognitive_metrics(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Update cognitive planning metrics from InfoBus"""
        
        # Extract system complexity indicators
        portfolio_complexity = len(info_bus.get('active_positions', []))
        decision_complexity = len(info_bus.get('pending_decisions', []))
        market_complexity = self._assess_market_complexity(context)
        
        # Calculate cognitive load
        base_load = 0.3
        complexity_load = min(0.5, (portfolio_complexity + decision_complexity) / 20.0)
        market_load = market_complexity * 0.2
        
        self.cognitive_load = base_load + complexity_load + market_load
        
        # Update planning confidence based on recent performance
        recent_performance = self._extract_recent_performance(info_bus)
        if recent_performance['win_rate'] > 0.6:
            confidence_boost = 0.1
        elif recent_performance['win_rate'] < 0.4:
            confidence_boost = -0.1
        else:
            confidence_boost = 0.0
        
        self.planning_confidence = np.clip(
            self.planning_confidence + confidence_boost * 0.1, 0.1, 1.0
        )
        
        # Update strategy coherence
        strategy_alignment = self._assess_strategy_alignment(info_bus)
        self.strategy_coherence = 0.9 * self.strategy_coherence + 0.1 * strategy_alignment
        
        # Update metrics
        self._update_performance_metric('cognitive_load', self.cognitive_load)
        self._update_performance_metric('planning_confidence', self.planning_confidence)
        self._update_performance_metric('strategy_coherence', self.strategy_coherence)
    
    def _assess_market_complexity(self, context: Dict[str, Any]) -> float:
        """Assess current market complexity"""
        
        complexity = 0.5  # Base complexity
        
        # Regime complexity
        regime = context.get('regime', 'unknown')
        if regime == 'volatile':
            complexity += 0.3
        elif regime == 'ranging':
            complexity += 0.2
        elif regime == 'trending':
            complexity += 0.1
        
        # Volatility complexity
        vol_level = context.get('volatility_level', 'medium')
        vol_multiplier = {'low': 0.1, 'medium': 0.2, 'high': 0.3, 'extreme': 0.4}
        complexity += vol_multiplier.get(vol_level, 0.2)
        
        # Session complexity
        session = context.get('session', 'unknown')
        if session in ['asian', 'overlap']:
            complexity += 0.1  # Generally more complex
        
        return min(1.0, complexity)
    
    def _extract_recent_performance(self, info_bus: InfoBus) -> Dict[str, float]:
        """Extract recent performance metrics"""
        
        recent_trades = info_bus.get('recent_trades', [])
        
        if not recent_trades:
            return {'win_rate': 0.5, 'avg_pnl': 0.0, 'total_pnl': 0.0}
        
        # Analyze recent trades
        wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
        total_trades = len(recent_trades)
        total_pnl = sum(trade.get('pnl', 0) for trade in recent_trades)
        
        return {
            'win_rate': wins / max(total_trades, 1),
            'avg_pnl': total_pnl / max(total_trades, 1),
            'total_pnl': total_pnl
        }
    
    def _assess_strategy_alignment(self, info_bus: InfoBus) -> float:
        """Assess alignment between different strategy components"""
        
        # Extract strategy signals and decisions
        committee_votes = info_bus.get('committee_votes', {})
        risk_signals = info_bus.get('risk_metrics', {})
        meta_signals = info_bus.get('meta_status', {})
        
        alignment_score = 0.7  # Base alignment
        
        # Check committee consensus
        if committee_votes:
            consensus = committee_votes.get('consensus', 0.5)
            if consensus > 0.7:
                alignment_score += 0.2
            elif consensus < 0.3:
                alignment_score -= 0.2
        
        # Check risk-strategy alignment
        risk_level = risk_signals.get('risk_score', 0.5)
        if 0.3 <= risk_level <= 0.7:  # Moderate risk preferred
            alignment_score += 0.1
        else:
            alignment_score -= 0.1
        
        return np.clip(alignment_score, 0.0, 1.0)
    
    def _execute_planning_phase(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Execute current planning phase"""
        
        if self.current_phase == PlanningPhase.ANALYSIS:
            self._execute_analysis_phase(info_bus, context)
        elif self.current_phase == PlanningPhase.PLANNING:
            self._execute_planning_phase_impl(info_bus, context)
        elif self.current_phase == PlanningPhase.EXECUTION:
            self._execute_execution_phase(info_bus, context)
        elif self.current_phase == PlanningPhase.REFLECTION:
            self._execute_reflection_phase(info_bus, context)
        elif self.current_phase == PlanningPhase.ADAPTATION:
            self._execute_adaptation_phase(info_bus, context)
    
    def _execute_analysis_phase(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Analysis phase: Gather and analyze current situation"""
        
        # Analyze market conditions
        market_analysis = {
            'regime': context.get('regime', 'unknown'),
            'volatility': context.get('volatility_level', 'medium'),
            'session': context.get('session', 'unknown'),
            'complexity': self._assess_market_complexity(context),
            'opportunities': self._identify_market_opportunities(info_bus, context)
        }
        
        # Analyze system performance
        performance_analysis = {
            'recent_performance': self._extract_recent_performance(info_bus),
            'risk_metrics': info_bus.get('risk_metrics', {}),
            'system_health': self._assess_system_health(info_bus),
            'cognitive_state': {
                'load': self.cognitive_load,
                'confidence': self.planning_confidence,
                'coherence': self.strategy_coherence
            }
        }
        
        # Store analysis results
        analysis_result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'phase': 'analysis',
            'market_analysis': market_analysis,
            'performance_analysis': performance_analysis,
            'insights': self._generate_analysis_insights(market_analysis, performance_analysis)
        }
        
        self.strategic_insights.append(analysis_result)
        
        self.log_operator_info(
            "🔍 Analysis phase completed",
            market_regime=market_analysis['regime'],
            complexity=f"{market_analysis['complexity']:.3f}",
            system_health=f"{performance_analysis['system_health']:.3f}",
            insights=len(analysis_result['insights'])
        )
    
    def _execute_planning_phase_impl(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Planning phase: Generate strategic plans"""
        
        # Get latest analysis
        latest_analysis = self.strategic_insights[-1] if self.strategic_insights else {}
        
        # Generate strategic plan
        strategic_plan = self._generate_strategic_plan(info_bus, context, latest_analysis)
        
        # Generate tactical recommendations
        tactical_recommendations = self._generate_tactical_recommendations(info_bus, context)
        
        # Generate risk management plan
        risk_plan = self._generate_risk_management_plan(info_bus, context)
        
        # Combine into comprehensive plan
        comprehensive_plan = {
            'timestamp': datetime.datetime.now().isoformat(),
            'phase': 'planning',
            'planning_cycle': self.planning_cycle,
            'strategic_plan': strategic_plan,
            'tactical_recommendations': tactical_recommendations,
            'risk_plan': risk_plan,
            'success_criteria': self._define_success_criteria(),
            'fallback_plans': self._generate_fallback_plans(context)
        }
        
        self.session_plans.append(comprehensive_plan)
        self.current_recommendations = tactical_recommendations
        
        self.log_operator_info(
            "📋 Planning phase completed",
            cycle=self.planning_cycle,
            strategic_objectives=len(strategic_plan),
            tactical_recommendations=len(tactical_recommendations),
            risk_controls=len(risk_plan),
            planning_confidence=f"{self.planning_confidence:.3f}"
        )
    
    def _execute_execution_phase(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Execution phase: Monitor plan execution"""
        
        if not self.session_plans:
            self.log_operator_warning("No active plan to execute")
            return
        
        current_plan = self.session_plans[-1]
        
        # Monitor plan execution
        execution_status = self._monitor_plan_execution(info_bus, context, current_plan)
        
        # Check for deviations
        deviations = self._detect_plan_deviations(execution_status, current_plan)
        
        # Adjust execution if needed
        if deviations:
            adjustments = self._generate_execution_adjustments(deviations, current_plan)
            self._apply_execution_adjustments(adjustments, info_bus)
            
            self.log_operator_warning(
                "⚠️ Plan deviations detected",
                deviations=len(deviations),
                adjustments=len(adjustments),
                execution_score=f"{execution_status.get('score', 0.5):.3f}"
            )
        else:
            self.log_operator_info(
                "✅ Plan execution on track",
                execution_score=f"{execution_status.get('score', 0.5):.3f}",
                completion=f"{execution_status.get('completion', 0.0):.1%}"
            )
    
    def _execute_reflection_phase(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Reflection phase: Evaluate outcomes and learn"""
        
        if not self.session_plans:
            return
        
        recent_plan = self.session_plans[-1]
        
        # Evaluate plan outcomes
        outcomes = self._evaluate_plan_outcomes(info_bus, context, recent_plan)
        
        # Generate lessons learned
        lessons = self._extract_lessons_learned(outcomes, recent_plan)
        
        # Update planning effectiveness
        self._update_planning_effectiveness(outcomes, recent_plan)
        
        # Store reflection results
        reflection_result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'phase': 'reflection',
            'plan_id': recent_plan.get('timestamp'),
            'outcomes': outcomes,
            'lessons_learned': lessons,
            'effectiveness_score': outcomes.get('effectiveness_score', 0.5),
            'improvement_areas': self._identify_improvement_areas(outcomes)
        }
        
        self.learning_history.append(reflection_result)
        
        self.log_operator_info(
            "🤔 Reflection phase completed",
            effectiveness=f"{outcomes.get('effectiveness_score', 0.5):.3f}",
            lessons_learned=len(lessons),
            improvement_areas=len(reflection_result['improvement_areas']),
            total_learning_entries=len(self.learning_history)
        )
    
    def _execute_adaptation_phase(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Adaptation phase: Adapt strategies and objectives"""
        
        # Analyze learning history for adaptation opportunities
        adaptation_opportunities = self._identify_adaptation_opportunities()
        
        if not adaptation_opportunities:
            self.log_operator_info("No significant adaptation opportunities found")
            return
        
        # Generate adaptations
        adaptations = self._generate_adaptations(adaptation_opportunities, context)
        
        # Apply adaptations
        applied_adaptations = self._apply_adaptations(adaptations, info_bus)
        
        # Track adaptation history
        adaptation_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'phase': 'adaptation',
            'opportunities': adaptation_opportunities,
            'adaptations_applied': applied_adaptations,
            'adaptation_score': self._calculate_adaptation_score(applied_adaptations)
        }
        
        self.adaptation_history.append(adaptation_record)
        
        self.log_operator_info(
            "🔄 Adaptation phase completed",
            opportunities=len(adaptation_opportunities),
            adaptations=len(applied_adaptations),
            adaptation_score=f"{adaptation_record['adaptation_score']:.3f}",
            strategic_evolution=True
        )
    
    def _evaluate_phase_transition(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Evaluate whether to transition to next planning phase"""
        
        phase_duration = (datetime.datetime.now() - self.phase_start_time).total_seconds()
        min_phase_duration = 120  # 2 minutes minimum
        max_phase_duration = 600  # 10 minutes maximum
        
        should_transition = False
        transition_reason = ""
        
        # Time-based transitions
        if phase_duration > max_phase_duration:
            should_transition = True
            transition_reason = f"Maximum phase duration reached ({max_phase_duration}s)"
        
        # Phase-specific completion criteria
        elif phase_duration > min_phase_duration:
            if self.current_phase == PlanningPhase.ANALYSIS:
                if len(self.strategic_insights) > 0:
                    should_transition = True
                    transition_reason = "Analysis completed"
            
            elif self.current_phase == PlanningPhase.PLANNING:
                if len(self.session_plans) > 0 and self.current_recommendations:
                    should_transition = True
                    transition_reason = "Planning completed"
            
            elif self.current_phase == PlanningPhase.EXECUTION:
                # Check if current plan has been executed sufficiently
                if self.session_plans:
                    execution_completion = self._assess_execution_completion()
                    if execution_completion > 0.7:
                        should_transition = True
                        transition_reason = f"Execution sufficient ({execution_completion:.1%})"
            
            elif self.current_phase == PlanningPhase.REFLECTION:
                if len(self.learning_history) > 0:
                    should_transition = True
                    transition_reason = "Reflection completed"
            
            elif self.current_phase == PlanningPhase.ADAPTATION:
                should_transition = True
                transition_reason = "Adaptation cycle completed"
        
        if should_transition:
            self._advance_planning_phase(transition_reason)
    
    def _advance_planning_phase(self, reason: str):
        """Advance to next planning phase"""
        
        # Define phase sequence
        phase_sequence = [
            PlanningPhase.ANALYSIS,
            PlanningPhase.PLANNING,
            PlanningPhase.EXECUTION,
            PlanningPhase.REFLECTION,
            PlanningPhase.ADAPTATION
        ]
        
        current_index = phase_sequence.index(self.current_phase)
        next_index = (current_index + 1) % len(phase_sequence)
        next_phase = phase_sequence[next_index]
        
        # Complete planning cycle if returning to analysis
        if next_phase == PlanningPhase.ANALYSIS:
            self.planning_cycle += 1
        
        old_phase = self.current_phase
        phase_duration = (datetime.datetime.now() - self.phase_start_time).total_seconds()
        
        self.current_phase = next_phase
        self.phase_start_time = datetime.datetime.now()
        
        self.log_operator_info(
            f"🔄 Planning phase transition: {old_phase.value} → {next_phase.value}",
            reason=reason,
            duration=f"{phase_duration:.0f}s",
            cycle=self.planning_cycle,
            cognitive_load=f"{self.cognitive_load:.3f}"
        )
    
    def _generate_strategic_plan(self, info_bus: InfoBus, context: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategic plan based on analysis"""
        
        strategic_objectives = []
        
        # Market-based objectives
        market_regime = context.get('regime', 'unknown')
        if market_regime == 'trending':
            strategic_objectives.append({
                'type': 'trend_following',
                'priority': 'high',
                'target': 'Maximize trend capture',
                'metrics': ['trend_alignment', 'momentum_score'],
                'timeline': 'short_term'
            })
        elif market_regime == 'ranging':
            strategic_objectives.append({
                'type': 'mean_reversion',
                'priority': 'medium',
                'target': 'Exploit range boundaries',
                'metrics': ['range_efficiency', 'reversal_accuracy'],
                'timeline': 'medium_term'
            })
        
        # Performance-based objectives
        recent_perf = analysis.get('performance_analysis', {}).get('recent_performance', {})
        if recent_perf.get('win_rate', 0.5) < 0.5:
            strategic_objectives.append({
                'type': 'performance_improvement',
                'priority': 'critical',
                'target': 'Improve win rate above 55%',
                'metrics': ['win_rate', 'profit_factor'],
                'timeline': 'immediate'
            })
        
        # Risk management objectives
        risk_metrics = analysis.get('performance_analysis', {}).get('risk_metrics', {})
        if risk_metrics.get('max_drawdown', 0) > 0.1:
            strategic_objectives.append({
                'type': 'risk_reduction',
                'priority': 'high',
                'target': 'Reduce maximum drawdown below 10%',
                'metrics': ['max_drawdown', 'risk_adjusted_return'],
                'timeline': 'short_term'
            })
        
        return strategic_objectives
    
    def _generate_tactical_recommendations(self, info_bus: InfoBus, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tactical recommendations"""
        
        recommendations = []
        
        # Market condition based recommendations
        vol_level = context.get('volatility_level', 'medium')
        if vol_level == 'extreme':
            recommendations.append({
                'type': 'position_sizing',
                'action': 'reduce_size',
                'rationale': 'Extreme volatility detected',
                'confidence': 0.8,
                'urgency': 'high'
            })
        
        regime = context.get('regime', 'unknown')
        if regime == 'volatile':
            recommendations.append({
                'type': 'strategy_adjustment',
                'action': 'increase_risk_management',
                'rationale': 'Volatile market regime',
                'confidence': 0.7,
                'urgency': 'medium'
            })
        
        # Performance-based recommendations
        recent_perf = self._extract_recent_performance(info_bus)
        if recent_perf['win_rate'] < 0.4:
            recommendations.append({
                'type': 'strategy_modification',
                'action': 'review_entry_criteria',
                'rationale': f"Low win rate: {recent_perf['win_rate']:.1%}",
                'confidence': 0.9,
                'urgency': 'high'
            })
        
        # System health recommendations
        if self.cognitive_load > 0.8:
            recommendations.append({
                'type': 'system_optimization',
                'action': 'simplify_decision_process',
                'rationale': f"High cognitive load: {self.cognitive_load:.3f}",
                'confidence': 0.6,
                'urgency': 'medium'
            })
        
        return recommendations
    
    def _generate_risk_management_plan(self, info_bus: InfoBus, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk management plan"""
        
        risk_controls = []
        
        # Dynamic position sizing
        risk_controls.append({
            'type': 'position_sizing',
            'rule': 'volatility_adjusted',
            'parameters': {
                'base_size': 0.02,
                'volatility_multiplier': 0.8,
                'max_size': 0.05
            },
            'conditions': ['vol_level != extreme'],
            'priority': 'critical'
        })
        
        # Drawdown protection
        risk_controls.append({
            'type': 'drawdown_control',
            'rule': 'progressive_reduction',
            'parameters': {
                'trigger_level': 0.05,
                'reduction_factor': 0.5,
                'recovery_threshold': 0.03
            },
            'conditions': ['drawdown > trigger_level'],
            'priority': 'high'
        })
        
        # Correlation limits
        risk_controls.append({
            'type': 'correlation_control',
            'rule': 'maximum_correlation',
            'parameters': {
                'max_correlation': 0.7,
                'lookback_period': 20,
                'rebalance_threshold': 0.8
            },
            'conditions': ['portfolio_correlation > max_correlation'],
            'priority': 'medium'
        })
        
        return risk_controls
    
    def record_episode(self, result: Dict[str, Any]):
        """Enhanced episode recording with comprehensive analysis"""
        
        try:
            # Validate and sanitize input
            if not isinstance(result, dict):
                self.log_operator_error(f"Invalid episode result type: {type(result)}")
                return
            
            pnl = result.get("pnl", 0)
            if np.isnan(pnl):
                self.log_operator_error("NaN PnL in episode result, setting to 0")
                result = result.copy()
                result["pnl"] = 0
                pnl = 0
            
            # Enhance result with metadata
            enhanced_result = {
                **result,
                'timestamp': datetime.datetime.now().isoformat(),
                'planning_cycle': self.planning_cycle,
                'planning_phase': self.current_phase.value,
                'cognitive_load': self.cognitive_load,
                'planning_confidence': self.planning_confidence,
                'strategy_coherence': self.strategy_coherence
            }
            
            # Add to history
            self.episode_history.append(enhanced_result)
            
            # Update trading metrics
            self._update_trading_metrics({'pnl': pnl})
            
            # Analyze episode patterns
            self._analyze_episode_patterns(enhanced_result)
            
            # Update cognitive metrics based on episode outcome
            self._update_cognitive_metrics_from_episode(enhanced_result)
            
            self.log_operator_info(
                f"📊 Episode recorded",
                pnl=f"€{pnl:.2f}",
                phase=self.current_phase.value,
                cycle=self.planning_cycle,
                cognitive_load=f"{self.cognitive_load:.3f}",
                total_episodes=len(self.episode_history)
            )
            
        except Exception as e:
            self.log_operator_error(f"Episode recording failed: {e}")
    
    def _analyze_episode_patterns(self, episode_result: Dict[str, Any]):
        """Analyze patterns in episode results"""
        
        if len(self.episode_history) < 5:
            return
        
        # Analyze recent episodes
        recent_episodes = list(self.episode_history)[-5:]
        
        # PnL patterns
        pnls = [ep.get('pnl', 0) for ep in recent_episodes]
        avg_pnl = np.mean(pnls)
        pnl_trend = np.polyfit(range(len(pnls)), pnls, 1)[0]  # Linear trend
        
        # Planning phase effectiveness
        phase_performance = defaultdict(list)
        for ep in recent_episodes:
            phase = ep.get('planning_phase', 'unknown')
            pnl = ep.get('pnl', 0)
            phase_performance[phase].append(pnl)
        
        # Update planning effectiveness
        for phase, pnls_list in phase_performance.items():
            if phase in self.planning_effectiveness:
                self.planning_effectiveness[phase]['total'] += len(pnls_list)
                self.planning_effectiveness[phase]['successful'] += sum(1 for p in pnls_list if p > 0)
                self.planning_effectiveness[phase]['avg_outcome'] = (
                    0.9 * self.planning_effectiveness[phase]['avg_outcome'] + 
                    0.1 * np.mean(pnls_list)
                )
        
        # Store pattern insights
        pattern_insight = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'episode_pattern_analysis',
            'avg_pnl': avg_pnl,
            'pnl_trend': pnl_trend,
            'phase_effectiveness': dict(self.planning_effectiveness),
            'pattern_strength': abs(pnl_trend) if not np.isnan(pnl_trend) else 0.0
        }
        
        self.strategic_insights.append(pattern_insight)
    
    def _publish_planning_status(self, info_bus: InfoBus):
        """Publish planning status to InfoBus"""
        
        planning_status = {
            'current_phase': self.current_phase.value,
            'planning_cycle': self.planning_cycle,
            'phase_duration': (datetime.datetime.now() - self.phase_start_time).total_seconds(),
            'cognitive_metrics': {
                'load': self.cognitive_load,
                'confidence': self.planning_confidence,
                'coherence': self.strategy_coherence,
                'adaptation_speed': self.adaptation_speed
            },
            'current_recommendations': self.current_recommendations,
            'strategic_insights': list(self.strategic_insights)[-3:],  # Recent insights
            'planning_effectiveness': dict(self.planning_effectiveness),
            'episode_count': len(self.episode_history),
            'adaptation_count': len(self.adaptation_history)
        }
        
        InfoBusUpdater.update_planning_status(info_bus, planning_status)
    
    # ═══════════════════════════════════════════════════════════════════
    # OBSERVATION AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation with cognitive planning metrics"""
        
        try:
            # Base cognitive metrics
            cognitive_components = [
                self.cognitive_load,
                self.planning_confidence,
                self.strategy_coherence,
                self.adaptation_speed
            ]
            
            # Planning phase encoding
            phase_encoding = {
                PlanningPhase.ANALYSIS: [1, 0, 0, 0, 0],
                PlanningPhase.PLANNING: [0, 1, 0, 0, 0],
                PlanningPhase.EXECUTION: [0, 0, 1, 0, 0],
                PlanningPhase.REFLECTION: [0, 0, 0, 1, 0],
                PlanningPhase.ADAPTATION: [0, 0, 0, 0, 1]
            }
            phase_components = phase_encoding.get(self.current_phase, [0.2, 0.2, 0.2, 0.2, 0.2])
            
            # Performance metrics
            if self.episode_history:
                recent_episodes = list(self.episode_history)[-10:]
                pnls = [ep.get('pnl', 0) for ep in recent_episodes]
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                avg_pnl = np.mean(pnls)
                pnl_volatility = np.std(pnls) if len(pnls) > 1 else 0.0
            else:
                win_rate = 0.5
                avg_pnl = 0.0
                pnl_volatility = 0.0
            
            performance_components = [
                win_rate,
                avg_pnl / 100.0,  # Normalize
                min(pnl_volatility / 50.0, 1.0)  # Normalize and cap
            ]
            
            # Planning effectiveness
            if self.planning_effectiveness:
                effectiveness_scores = []
                for phase_data in self.planning_effectiveness.values():
                    if phase_data['total'] > 0:
                        effectiveness = phase_data['successful'] / phase_data['total']
                        effectiveness_scores.append(effectiveness)
                
                planning_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 0.5
            else:
                planning_effectiveness = 0.5
            
            # Combine all components
            observation = np.array(
                cognitive_components + 
                phase_components + 
                performance_components + 
                [planning_effectiveness], 
                dtype=np.float32
            )
            
            # Validate observation
            if np.any(np.isnan(observation)):
                self.log_operator_error(f"NaN in observation: {observation}")
                observation = np.nan_to_num(observation)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(13, dtype=np.float32)  # 4 + 5 + 3 + 1 components
    
    # ═══════════════════════════════════════════════════════════════════
    # UTILITY AND HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_planning_report(self) -> str:
        """Generate operator-friendly planning report"""
        
        # Phase status with emoji
        phase_emoji = {
            PlanningPhase.ANALYSIS: "🔍",
            PlanningPhase.PLANNING: "📋",
            PlanningPhase.EXECUTION: "⚡",
            PlanningPhase.REFLECTION: "🤔",
            PlanningPhase.ADAPTATION: "🔄"
        }
        
        phase_duration = (datetime.datetime.now() - self.phase_start_time).total_seconds()
        
        # Calculate effectiveness
        if self.planning_effectiveness:
            avg_effectiveness = np.mean([
                data['successful'] / max(data['total'], 1) 
                for data in self.planning_effectiveness.values()
            ])
        else:
            avg_effectiveness = 0.5
        
        # Recent performance
        if self.episode_history:
            recent_pnls = [ep.get('pnl', 0) for ep in list(self.episode_history)[-10:]]
            recent_win_rate = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
            recent_avg = np.mean(recent_pnls)
        else:
            recent_win_rate = 0.5
            recent_avg = 0.0
        
        return f"""
🧠 METACOGNITIVE PLANNER
═══════════════════════════════════════
{phase_emoji.get(self.current_phase, '❓')} Phase: {self.current_phase.value.upper()}
⏱️ Duration: {phase_duration/60:.1f} minutes
🔄 Cycle: {self.planning_cycle}

🎯 COGNITIVE METRICS
• Load: {self.cognitive_load:.3f}
• Confidence: {self.planning_confidence:.3f}
• Coherence: {self.strategy_coherence:.3f}
• Adaptation Speed: {self.adaptation_speed:.3f}

📊 PLANNING EFFECTIVENESS
• Overall: {avg_effectiveness:.3f}
• Active Recommendations: {len(self.current_recommendations)}
• Strategic Insights: {len(self.strategic_insights)}
• Learning Entries: {len(self.learning_history)}

📈 RECENT PERFORMANCE
• Episodes: {len(self.episode_history)}
• Win Rate: {recent_win_rate:.1%}
• Avg PnL: €{recent_avg:.2f}
• Adaptations: {len(self.adaptation_history)}

🎯 STRATEGIC OBJECTIVES
{chr(10).join([f"• {obj}: {val}" for obj, val in list(self.strategic_objectives.items())[:5]])}

💡 CURRENT RECOMMENDATIONS
{chr(10).join([f"• {rec.get('type', 'unknown')}: {rec.get('action', 'no action')}" for rec in self.current_recommendations[:3]])}

🔮 RECENT INSIGHTS
{chr(10).join([f"• {insight.get('type', 'unknown')}" for insight in list(self.strategic_insights)[-3:]])}
        """
    
    # Legacy compatibility methods
    def step(self, **kwargs):
        """Legacy step method for backward compatibility"""
        self._process_legacy_step(**kwargs)
    
    def get_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        base_state = super().get_state()
        
        planner_state = {
            'current_phase': self.current_phase.value,
            'planning_cycle': self.planning_cycle,
            'phase_start_time': self.phase_start_time.isoformat(),
            'episode_history': list(self.episode_history),
            'session_plans': list(self.session_plans),
            'adaptation_history': list(self.adaptation_history),
            'strategic_objectives': self.strategic_objectives.copy(),
            'planning_effectiveness': dict(self.planning_effectiveness),
            'cognitive_metrics': {
                'load': self.cognitive_load,
                'confidence': self.planning_confidence,
                'coherence': self.strategy_coherence,
                'adaptation_speed': self.adaptation_speed
            },
            'current_recommendations': self.current_recommendations.copy(),
            'strategic_insights': list(self.strategic_insights),
            'learning_history': list(self.learning_history)
        }
        
        if base_state:
            base_state.update(planner_state)
            return base_state
        
        return planner_state
    
    def set_state(self, state: Dict[str, Any]):
        """Enhanced state restoration"""
        super().set_state(state)
        
        self.current_phase = PlanningPhase(state.get('current_phase', 'analysis'))
        self.planning_cycle = state.get('planning_cycle', 0)
        self.phase_start_time = datetime.datetime.fromisoformat(
            state.get('phase_start_time', datetime.datetime.now().isoformat())
        )
        
        self.episode_history = deque(state.get('episode_history', []), maxlen=self.window)
        self.session_plans = deque(state.get('session_plans', []), maxlen=10)
        self.adaptation_history = deque(state.get('adaptation_history', []), maxlen=50)
        
        self.strategic_objectives.update(state.get('strategic_objectives', {}))
        self.planning_effectiveness.update(state.get('planning_effectiveness', {}))
        
        cognitive_metrics = state.get('cognitive_metrics', {})
        self.cognitive_load = cognitive_metrics.get('load', 0.5)
        self.planning_confidence = cognitive_metrics.get('confidence', 0.6)
        self.strategy_coherence = cognitive_metrics.get('coherence', 0.7)
        self.adaptation_speed = cognitive_metrics.get('adaptation_speed', 0.5)
        
        self.current_recommendations = state.get('current_recommendations', [])
        self.strategic_insights = deque(state.get('strategic_insights', []), maxlen=20)
        self.learning_history = deque(state.get('learning_history', []), maxlen=100)