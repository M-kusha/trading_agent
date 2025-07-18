# ─────────────────────────────────────────────────────────────
# File: modules/meta/metacognitive_planner.py
# [ROCKET] PRODUCTION-READY Metacognitive Planning System
# Enhanced with SmartInfoBus integration & intelligent automation
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


class PlanningPhase(Enum):
    """Metacognitive planning phases"""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    ADAPTATION = "adaptation"


@dataclass
class PlanningConfig:
    """Configuration for Metacognitive Planner"""
    window: int = 20
    planning_horizon: int = 100
    adaptation_threshold: float = 0.15
    phase_min_duration: int = 120
    phase_max_duration: int = 600
    
    # Performance thresholds
    max_processing_time_ms: float = 200
    circuit_breaker_threshold: int = 3
    min_confidence_threshold: float = 0.6
    
    # Strategic parameters
    profit_target: float = 150.0
    max_drawdown: float = 15.0
    min_win_rate: float = 0.55
    risk_budget: float = 0.10


@module(
    name="MetaCognitivePlanner",
    version="3.0.0",
    category="meta",
    provides=["planning_status", "strategic_insights", "tactical_recommendations", "adaptation_metrics"],
    requires=["trades", "actions", "market_data", "performance_metrics"],
    description="Advanced metacognitive planner with strategic planning and market adaptation",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class MetaCognitivePlanner(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced metacognitive planner with SmartInfoBus integration.
    Provides strategic planning, tactical recommendations, and adaptive optimization.
    """

    def __init__(self, 
                 config: Optional[PlanningConfig] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        # Store config first (preservation pattern)
        self.planning_config = config or PlanningConfig()
        self.config = self.planning_config  # Set config early for init methods
        
        # Initialize advanced systems before super().__init__()
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize planning state
        self._initialize_planning_state()
        
        super().__init__()
        
        # Restore our config after BaseModule initialization (prevents dict conversion)
        self.config = self.planning_config
        
        # Start monitoring after all initialization is complete
        
        self._start_monitoring()
        
        
        
        self.logger.info(
            format_operator_message(
                "🧠", "METACOGNITIVE_PLANNER_INITIALIZED",
                details=f"Planning horizon: {self.config.planning_horizon}, Window: {self.config.window}",
                result="Strategic planning system ready",
                context="metacognitive_planning"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for metacognitive planning"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="MetaCognitivePlanner", 
            log_path="logs/metacognitive_planner.log", 
            max_lines=3000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("MetaCognitivePlanner", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for planning operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        # Note: _start_monitoring() moved to end of initialization

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
        """Initialize genome-based parameters"""
        if genome:
            self.genome = {
                "window": int(genome.get("window", self.config.window)),
                "planning_horizon": int(genome.get("planning_horizon", self.config.planning_horizon)),
                "adaptation_threshold": float(genome.get("adaptation_threshold", self.config.adaptation_threshold)),
                "profit_target": float(genome.get("profit_target", self.config.profit_target)),
                "max_drawdown": float(genome.get("max_drawdown", self.config.max_drawdown)),
                "min_win_rate": float(genome.get("min_win_rate", self.config.min_win_rate))
            }
        else:
            self.genome = {
                "window": self.config.window,
                "planning_horizon": self.config.planning_horizon,
                "adaptation_threshold": self.config.adaptation_threshold,
                "profit_target": self.config.profit_target,
                "max_drawdown": self.config.max_drawdown,
                "min_win_rate": self.config.min_win_rate
            }

    def _initialize_planning_state(self):
        """Initialize metacognitive planning state"""
        # Core planning state
        self.current_phase = PlanningPhase.ANALYSIS
        self.phase_start_time = datetime.now()
        self.planning_cycle = 0
        
        # Episode and session tracking
        self.episode_history = deque(maxlen=self.genome["window"])
        self.session_plans = deque(maxlen=10)
        self.adaptation_history = deque(maxlen=50)
        
        # Strategic planning components
        self.strategic_objectives = {
            'profit_target': self.genome["profit_target"],
            'max_drawdown': self.genome["max_drawdown"],
            'min_win_rate': self.genome["min_win_rate"],
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

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_planning_health()
                    self._analyze_planning_performance()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    def _initialize(self):
        """Initialize module"""
        try:
            # Set initial planning status in SmartInfoBus
            initial_status = {
                "current_phase": self.current_phase.value,
                "planning_cycle": 0,
                "cognitive_load": self.cognitive_load,
                "planning_confidence": self.planning_confidence
            }
            
            self.smart_bus.set(
                'planning_status',
                initial_status,
                module='MetaCognitivePlanner',
                thesis="Initial metacognitive planning status"
            )
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process metacognitive planning operations"""
        start_time = time.time()
        
        try:
            # Extract planning data
            planning_data = await self._extract_planning_data(**inputs)
            
            if not planning_data:
                return await self._handle_no_data_fallback()
            
            # Execute current planning phase
            phase_result = await self._execute_planning_phase(planning_data)
            
            # Check for phase transitions
            transition_result = await self._evaluate_phase_transition(planning_data)
            phase_result.update(transition_result)
            
            # Generate strategic insights
            insights_result = await self._generate_strategic_insights(planning_data)
            phase_result.update(insights_result)
            
            # Update cognitive metrics
            cognitive_result = await self._update_cognitive_metrics(planning_data)
            phase_result.update(cognitive_result)
            
            # Generate thesis
            thesis = await self._generate_planning_thesis(planning_data, phase_result)
            
            # Update SmartInfoBus
            await self._update_planning_smart_bus(phase_result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return phase_result
            
        except Exception as e:
            return await self._handle_planning_error(e, start_time)

    async def _extract_planning_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract planning data from SmartInfoBus"""
        try:
            # Get recent trades
            trades = self.smart_bus.get('trades', 'MetaCognitivePlanner') or []
            
            # Get actions
            actions = self.smart_bus.get('actions', 'MetaCognitivePlanner') or []
            
            # Get market data
            market_data = self.smart_bus.get('market_data', 'MetaCognitivePlanner') or {}
            
            # Get performance metrics
            performance_metrics = self.smart_bus.get('performance_metrics', 'MetaCognitivePlanner') or {}
            
            # Extract context from market data
            context = self._extract_standard_context(market_data)
            
            return {
                'trades': trades,
                'actions': actions,
                'market_data': market_data,
                'performance_metrics': performance_metrics,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'episode_data': inputs.get('episode_data', {})
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract planning data: {e}")
            return None

    def _extract_standard_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standard market context"""
        return {
            'regime': market_data.get('regime', 'unknown'),
            'volatility_level': market_data.get('volatility_level', 'medium'),
            'session': market_data.get('session', 'unknown'),
            'drawdown_pct': market_data.get('drawdown_pct', 0.0),
            'exposure_pct': market_data.get('exposure_pct', 0.0),
            'position_count': market_data.get('position_count', 0),
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_planning_phase(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute current planning phase"""
        try:
            if self.current_phase == PlanningPhase.ANALYSIS:
                return await self._execute_analysis_phase(planning_data)
            elif self.current_phase == PlanningPhase.PLANNING:
                return await self._execute_planning_phase_impl(planning_data)
            elif self.current_phase == PlanningPhase.EXECUTION:
                return await self._execute_execution_phase(planning_data)
            elif self.current_phase == PlanningPhase.REFLECTION:
                return await self._execute_reflection_phase(planning_data)
            elif self.current_phase == PlanningPhase.ADAPTATION:
                return await self._execute_adaptation_phase(planning_data)
            else:
                return {'phase_executed': False, 'reason': 'unknown_phase'}
                
        except Exception as e:
            self.logger.error(f"Phase execution failed: {e}")
            return self._create_fallback_response("phase execution failed")

    async def _execute_analysis_phase(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analysis phase: Gather and analyze current situation"""
        try:
            context = planning_data['context']
            
            # Analyze market conditions
            market_analysis = {
                'regime': context.get('regime', 'unknown'),
                'volatility': context.get('volatility_level', 'medium'),
                'session': context.get('session', 'unknown'),
                'complexity': self._assess_market_complexity(context),
                'opportunities': await self._identify_market_opportunities(planning_data)
            }
            
            # Analyze system performance
            performance_analysis = {
                'recent_performance': self._extract_recent_performance(planning_data),
                'risk_metrics': planning_data.get('performance_metrics', {}),
                'system_health': self._assess_system_health(planning_data),
                'cognitive_state': {
                    'load': self.cognitive_load,
                    'confidence': self.planning_confidence,
                    'coherence': self.strategy_coherence
                }
            }
            
            # Store analysis results
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'phase': 'analysis',
                'market_analysis': market_analysis,
                'performance_analysis': performance_analysis,
                'insights': self._generate_analysis_insights(market_analysis, performance_analysis)
            }
            
            self.strategic_insights.append(analysis_result)
            
            return {
                'phase_executed': True,
                'analysis_completed': True,
                'market_complexity': market_analysis['complexity'],
                'system_health': performance_analysis['system_health'],
                'insights_generated': len(analysis_result['insights'])
            }
            
        except Exception as e:
            self.logger.error(f"Analysis phase failed: {e}")
            return self._create_fallback_response("analysis phase failed")

    async def _execute_planning_phase_impl(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Planning phase: Generate strategic plans"""
        try:
            # Get latest analysis
            latest_analysis = self.strategic_insights[-1] if self.strategic_insights else {}
            
            # Generate strategic plan
            strategic_plan = self._generate_strategic_plan(planning_data, latest_analysis)
            
            # Generate tactical recommendations
            tactical_recommendations = self._generate_tactical_recommendations(planning_data)
            
            # Generate risk management plan
            risk_plan = self._generate_risk_management_plan(planning_data)
            
            # Combine into comprehensive plan
            comprehensive_plan = {
                'timestamp': datetime.now().isoformat(),
                'phase': 'planning',
                'planning_cycle': self.planning_cycle,
                'strategic_plan': strategic_plan,
                'tactical_recommendations': tactical_recommendations,
                'risk_plan': risk_plan,
                'success_criteria': self._define_success_criteria(),
                'fallback_plans': self._generate_fallback_plans(planning_data['context'])
            }
            
            self.session_plans.append(comprehensive_plan)
            self.current_recommendations = tactical_recommendations
            
            return {
                'phase_executed': True,
                'planning_completed': True,
                'strategic_objectives': len(strategic_plan),
                'tactical_recommendations': len(tactical_recommendations),
                'risk_controls': len(risk_plan),
                'planning_confidence': self.planning_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Planning phase failed: {e}")
            return self._create_fallback_response("planning phase failed")

    async def _execute_execution_phase(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execution phase: Monitor plan execution"""
        try:
            if not self.session_plans:
                return {'phase_executed': False, 'reason': 'no_active_plan'}
            
            current_plan = self.session_plans[-1]
            
            # Monitor plan execution
            execution_status = self._monitor_plan_execution(planning_data, current_plan)
            
            # Check for deviations
            deviations = self._detect_plan_deviations(execution_status, current_plan)
            
            # Adjust execution if needed
            adjustments = []
            if deviations:
                adjustments = self._generate_execution_adjustments(deviations, current_plan)
            
            return {
                'phase_executed': True,
                'execution_monitored': True,
                'execution_score': execution_status.get('score', 0.5),
                'deviations_detected': len(deviations),
                'adjustments_made': len(adjustments),
                'plan_on_track': len(deviations) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Execution phase failed: {e}")
            return self._create_fallback_response("execution phase failed")

    async def _execute_reflection_phase(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reflection phase: Evaluate outcomes and learn"""
        try:
            if not self.session_plans:
                return {'phase_executed': False, 'reason': 'no_plan_to_reflect'}
            
            recent_plan = self.session_plans[-1]
            
            # Evaluate plan outcomes
            outcomes = self._evaluate_plan_outcomes(planning_data, recent_plan)
            
            # Generate lessons learned
            lessons = self._extract_lessons_learned(outcomes, recent_plan)
            
            # Update planning effectiveness
            self._update_planning_effectiveness(outcomes, recent_plan)
            
            # Store reflection results
            reflection_result = {
                'timestamp': datetime.now().isoformat(),
                'phase': 'reflection',
                'plan_id': recent_plan.get('timestamp'),
                'outcomes': outcomes,
                'lessons_learned': lessons,
                'effectiveness_score': outcomes.get('effectiveness_score', 0.5),
                'improvement_areas': self._identify_improvement_areas(outcomes)
            }
            
            self.learning_history.append(reflection_result)
            
            return {
                'phase_executed': True,
                'reflection_completed': True,
                'effectiveness_score': outcomes.get('effectiveness_score', 0.5),
                'lessons_learned': len(lessons),
                'improvement_areas': len(reflection_result['improvement_areas'])
            }
            
        except Exception as e:
            self.logger.error(f"Reflection phase failed: {e}")
            return self._create_fallback_response("reflection phase failed")

    async def _execute_adaptation_phase(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptation phase: Adapt strategies and objectives"""
        try:
            # Analyze learning history for adaptation opportunities
            adaptation_opportunities = self._identify_adaptation_opportunities()
            
            if not adaptation_opportunities:
                return {
                    'phase_executed': True,
                    'adaptation_completed': True,
                    'opportunities_found': 0,
                    'adaptations_applied': 0
                }
            
            # Generate adaptations
            adaptations = self._generate_adaptations(adaptation_opportunities, planning_data['context'])
            
            # Apply adaptations
            applied_adaptations = self._apply_adaptations(adaptations)
            
            # Track adaptation history
            adaptation_record = {
                'timestamp': datetime.now().isoformat(),
                'phase': 'adaptation',
                'opportunities': adaptation_opportunities,
                'adaptations_applied': applied_adaptations,
                'adaptation_score': self._calculate_adaptation_score(applied_adaptations)
            }
            
            self.adaptation_history.append(adaptation_record)
            
            return {
                'phase_executed': True,
                'adaptation_completed': True,
                'opportunities_found': len(adaptation_opportunities),
                'adaptations_applied': len(applied_adaptations),
                'adaptation_score': adaptation_record['adaptation_score']
            }
            
        except Exception as e:
            self.logger.error(f"Adaptation phase failed: {e}")
            return self._create_fallback_response("adaptation phase failed")

    async def _evaluate_phase_transition(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate whether to transition to next planning phase"""
        try:
            phase_duration = (datetime.now() - self.phase_start_time).total_seconds()
            should_transition = False
            transition_reason = ""
            
            # Time-based transitions
            if phase_duration > self.config.phase_max_duration:
                should_transition = True
                transition_reason = f"Maximum phase duration reached ({self.config.phase_max_duration}s)"
            
            # Phase-specific completion criteria
            elif phase_duration > self.config.phase_min_duration:
                if self.current_phase == PlanningPhase.ANALYSIS:
                    if len(self.strategic_insights) > 0:
                        should_transition = True
                        transition_reason = "Analysis completed"
                
                elif self.current_phase == PlanningPhase.PLANNING:
                    if len(self.session_plans) > 0 and self.current_recommendations:
                        should_transition = True
                        transition_reason = "Planning completed"
                
                elif self.current_phase == PlanningPhase.EXECUTION:
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
                return {
                    'phase_transitioned': True,
                    'new_phase': self.current_phase.value,
                    'transition_reason': transition_reason
                }
            
            return {
                'phase_transitioned': False,
                'current_phase': self.current_phase.value,
                'phase_duration': phase_duration
            }
            
        except Exception as e:
            self.logger.error(f"Phase transition evaluation failed: {e}")
            return {'phase_transitioned': False, 'error': str(e)}

    def _advance_planning_phase(self, reason: str):
        """Advance to next planning phase"""
        try:
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
            phase_duration = (datetime.now() - self.phase_start_time).total_seconds()
            
            self.current_phase = next_phase
            self.phase_start_time = datetime.now()
            
            # Start monitoring after all initialization is complete
            
            self._start_monitoring()
            
            
            
            self.logger.info(
                format_operator_message(
                    "[RELOAD]", "PLANNING_PHASE_TRANSITION",
                    from_phase=old_phase.value,
                    to_phase=next_phase.value,
                    reason=reason,
                    duration=f"{phase_duration:.0f}s",
                    context="planning_transition"
                )
            )
            
        except Exception as e:
            self.logger.error(f"Phase advancement failed: {e}")

    async def _generate_strategic_insights(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic insights and recommendations"""
        try:
            insights = []
            
            # Market-based insights
            context = planning_data['context']
            regime = context.get('regime', 'unknown')
            vol_level = context.get('volatility_level', 'medium')
            
            if regime == 'volatile' and vol_level == 'extreme':
                insights.append({
                    'type': 'risk_management',
                    'insight': 'Extreme volatility detected - reduce position sizes',
                    'confidence': 0.9,
                    'urgency': 'high'
                })
            
            # Performance-based insights
            recent_perf = self._extract_recent_performance(planning_data)
            if recent_perf.get('win_rate', 0.5) < self.genome["min_win_rate"]:
                insights.append({
                    'type': 'strategy_adjustment',
                    'insight': f'Win rate below target: {recent_perf.get("win_rate", 0.5):.1%}',
                    'confidence': 0.8,
                    'urgency': 'medium'
                })
            
            return {
                'insights_generated': len(insights),
                'strategic_insights': insights,
                'insight_confidence': np.mean([i['confidence'] for i in insights]) if insights else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Strategic insights generation failed: {e}")
            return {'insights_generated': 0, 'strategic_insights': []}

    async def _update_cognitive_metrics(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update cognitive planning metrics"""
        try:
            # Extract system complexity indicators
            trades_count = len(planning_data.get('trades', []))
            actions_count = len(planning_data.get('actions', []))
            
            # Calculate cognitive load
            base_load = 0.3
            complexity_load = min(0.5, (trades_count + actions_count) / 20.0)
            
            self.cognitive_load = base_load + complexity_load
            
            # Update planning confidence based on recent performance
            recent_performance = self._extract_recent_performance(planning_data)
            if recent_performance.get('win_rate', 0.5) > 0.6:
                confidence_boost = 0.1
            elif recent_performance.get('win_rate', 0.5) < 0.4:
                confidence_boost = -0.1
            else:
                confidence_boost = 0.0
            
            self.planning_confidence = np.clip(
                self.planning_confidence + confidence_boost * 0.1, 0.1, 1.0
            )
            
            # Update strategy coherence
            strategy_alignment = self._assess_strategy_alignment(planning_data)
            self.strategy_coherence = 0.9 * self.strategy_coherence + 0.1 * strategy_alignment
            
            return {
                'cognitive_load': self.cognitive_load,
                'planning_confidence': self.planning_confidence,
                'strategy_coherence': self.strategy_coherence,
                'metrics_updated': True
            }
            
        except Exception as e:
            self.logger.error(f"Cognitive metrics update failed: {e}")
            return {'metrics_updated': False, 'error': str(e)}

    async def _generate_planning_thesis(self, planning_data: Dict[str, Any], 
                                       planning_result: Dict[str, Any]) -> str:
        """Generate comprehensive planning thesis"""
        try:
            # Planning metrics
            current_phase = self.current_phase.value
            planning_cycle = self.planning_cycle
            cognitive_load = self.cognitive_load
            
            # Performance metrics
            phase_executed = planning_result.get('phase_executed', False)
            insights_generated = planning_result.get('insights_generated', 0)
            
            thesis_parts = [
                f"Metacognitive Planning: Phase {current_phase} (cycle {planning_cycle}) with {cognitive_load:.2f} cognitive load",
                f"Planning execution: {'successful' if phase_executed else 'failed'} with {insights_generated} strategic insights"
            ]
            
            # Phase-specific details
            if current_phase == 'analysis':
                market_complexity = planning_result.get('market_complexity', 0.5)
                system_health = planning_result.get('system_health', 0.5)
                thesis_parts.append(f"Analysis: {market_complexity:.2f} market complexity, {system_health:.2f} system health")
            
            elif current_phase == 'planning':
                strategic_objectives = planning_result.get('strategic_objectives', 0)
                tactical_recommendations = planning_result.get('tactical_recommendations', 0)
                thesis_parts.append(f"Planning: {strategic_objectives} strategic objectives, {tactical_recommendations} tactical recommendations")
            
            elif current_phase == 'execution':
                execution_score = planning_result.get('execution_score', 0.5)
                plan_on_track = planning_result.get('plan_on_track', False)
                thesis_parts.append(f"Execution: {execution_score:.2f} score, {'on track' if plan_on_track else 'deviations detected'}")
            
            # Cognitive state
            thesis_parts.append(f"Cognitive state: {self.planning_confidence:.2f} confidence, {self.strategy_coherence:.2f} coherence")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Planning thesis generation failed: {str(e)} - Metacognitive processing continuing"

    async def _update_planning_smart_bus(self, planning_result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with planning results"""
        try:
            # Planning status
            planning_status = {
                'current_phase': self.current_phase.value,
                'planning_cycle': self.planning_cycle,
                'phase_duration': (datetime.now() - self.phase_start_time).total_seconds(),
                'cognitive_load': self.cognitive_load,
                'planning_confidence': self.planning_confidence,
                'strategy_coherence': self.strategy_coherence
            }
            
            self.smart_bus.set(
                'planning_status',
                planning_status,
                module='MetaCognitivePlanner',
                thesis=thesis
            )
            
            # Strategic insights
            insights_data = {
                'total_insights': len(self.strategic_insights),
                'current_recommendations': self.current_recommendations,
                'recent_insights': list(self.strategic_insights)[-3:] if self.strategic_insights else [],
                'planning_effectiveness': dict(self.planning_effectiveness)
            }
            
            self.smart_bus.set(
                'strategic_insights',
                insights_data,
                module='MetaCognitivePlanner',
                thesis=f"Strategic insights: {len(self.strategic_insights)} total insights generated"
            )
            
            # Tactical recommendations
            tactical_data = {
                'active_recommendations': self.current_recommendations,
                'session_plans': len(self.session_plans),
                'strategic_objectives': self.strategic_objectives
            }
            
            self.smart_bus.set(
                'tactical_recommendations',
                tactical_data,
                module='MetaCognitivePlanner',
                thesis="Tactical recommendations and strategic objectives"
            )
            
            # Adaptation metrics
            adaptation_data = {
                'total_adaptations': len(self.adaptation_history),
                'adaptation_speed': self.adaptation_speed,
                'learning_entries': len(self.learning_history),
                'strategy_evolution': len(self.strategy_evolution_trace)
            }
            
            self.smart_bus.set(
                'adaptation_metrics',
                adaptation_data,
                module='MetaCognitivePlanner',
                thesis="Adaptation metrics and learning progress tracking"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

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
        
        return min(1.0, complexity)

    def _extract_recent_performance(self, planning_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract recent performance metrics"""
        trades = planning_data.get('trades', [])
        
        if not trades:
            return {'win_rate': 0.5, 'avg_pnl': 0.0, 'total_pnl': 0.0}
        
        wins = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        total_trades = len(trades)
        total_pnl = sum(trade.get('pnl', 0) for trade in trades)
        
        return {
            'win_rate': wins / max(total_trades, 1),
            'avg_pnl': total_pnl / max(total_trades, 1),
            'total_pnl': total_pnl
        }

    def _assess_system_health(self, planning_data: Dict[str, Any]) -> float:
        """Assess overall system health"""
        health_score = 0.7  # Base health
        
        # Check performance metrics
        performance_metrics = planning_data.get('performance_metrics', {})
        if performance_metrics:
            health_score += 0.2
        
        # Check data availability
        if planning_data.get('trades') and planning_data.get('market_data'):
            health_score += 0.1
        
        return min(1.0, health_score)

    def _assess_strategy_alignment(self, planning_data: Dict[str, Any]) -> float:
        """Assess alignment between different strategy components"""
        alignment_score = 0.7  # Base alignment
        
        # Check if we have consistent data
        trades = planning_data.get('trades', [])
        actions = planning_data.get('actions', [])
        
        if len(trades) > 0 and len(actions) > 0:
            alignment_score += 0.2
        
        return np.clip(alignment_score, 0.0, 1.0)

    # Helper methods (simplified versions)
    async def _identify_market_opportunities(self, planning_data: Dict[str, Any]) -> List[str]:
        """Identify market opportunities"""
        opportunities = []
        context = planning_data['context']
        
        if context.get('regime') == 'trending':
            opportunities.append('trend_following')
        if context.get('volatility_level') == 'high':
            opportunities.append('volatility_trading')
        
        return opportunities

    def _generate_strategic_plan(self, planning_data: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategic plan"""
        return [
            {
                'type': 'performance_improvement',
                'priority': 'high',
                'target': f'Achieve {self.genome["min_win_rate"]:.1%} win rate',
                'timeline': 'short_term'
            }
        ]

    def _generate_tactical_recommendations(self, planning_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tactical recommendations"""
        recommendations = []
        context = planning_data['context']
        
        if context.get('volatility_level') == 'extreme':
            recommendations.append({
                'type': 'position_sizing',
                'action': 'reduce_size',
                'rationale': 'Extreme volatility detected',
                'confidence': 0.8
            })
        
        return recommendations

    def _generate_risk_management_plan(self, planning_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk management plan"""
        return [
            {
                'type': 'position_sizing',
                'rule': 'volatility_adjusted',
                'priority': 'critical'
            }
        ]

    def _define_success_criteria(self) -> Dict[str, Any]:
        """Define success criteria for current plan"""
        return {
            'profit_target': self.genome["profit_target"],
            'max_drawdown': self.genome["max_drawdown"],
            'min_win_rate': self.genome["min_win_rate"]
        }

    def _generate_fallback_plans(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fallback plans"""
        return [
            {
                'trigger': 'high_drawdown',
                'action': 'reduce_risk',
                'threshold': self.genome["max_drawdown"]
            }
        ]

    def _monitor_plan_execution(self, planning_data: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor plan execution"""
        return {
            'score': 0.7,
            'completion': 0.5,
            'on_track': True
        }

    def _detect_plan_deviations(self, execution_status: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect plan deviations"""
        deviations = []
        if execution_status.get('score', 0.5) < 0.4:
            deviations.append({
                'type': 'performance_deviation',
                'severity': 'medium'
            })
        return deviations

    def _generate_execution_adjustments(self, deviations: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate execution adjustments"""
        return [
            {
                'type': 'parameter_adjustment',
                'adjustment': 'reduce_risk'
            } for _ in deviations
        ]

    def _evaluate_plan_outcomes(self, planning_data: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate plan outcomes"""
        return {
            'effectiveness_score': 0.7,
            'success': True
        }

    def _extract_lessons_learned(self, outcomes: Dict[str, Any], plan: Dict[str, Any]) -> List[str]:
        """Extract lessons learned"""
        return ['Market adaptation is important', 'Risk management is crucial']

    def _update_planning_effectiveness(self, outcomes: Dict[str, Any], plan: Dict[str, Any]):
        """Update planning effectiveness metrics"""
        effectiveness_score = outcomes.get('effectiveness_score', 0.5)
        phase = plan.get('phase', 'unknown')
        
        self.planning_effectiveness[phase]['total'] += 1
        if effectiveness_score > 0.6:
            self.planning_effectiveness[phase]['successful'] += 1

    def _identify_improvement_areas(self, outcomes: Dict[str, Any]) -> List[str]:
        """Identify improvement areas"""
        areas = []
        if outcomes.get('effectiveness_score', 0.5) < 0.6:
            areas.append('strategic_planning')
        return areas

    def _identify_adaptation_opportunities(self) -> List[Dict[str, Any]]:
        """Identify adaptation opportunities"""
        opportunities = []
        if len(self.learning_history) > 5:
            opportunities.append({
                'type': 'parameter_optimization',
                'confidence': 0.7
            })
        return opportunities

    def _generate_adaptations(self, opportunities: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adaptations"""
        return [
            {
                'type': 'threshold_adjustment',
                'parameter': 'adaptation_threshold',
                'adjustment': 0.05
            } for opportunity in opportunities
        ]

    def _apply_adaptations(self, adaptations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply adaptations"""
        applied = []
        for adaptation in adaptations:
            if adaptation['type'] == 'threshold_adjustment':
                parameter = adaptation['parameter']
                adjustment = adaptation['adjustment']
                if parameter in self.genome:
                    self.genome[parameter] += adjustment
                    applied.append(adaptation)
        return applied

    def _calculate_adaptation_score(self, adaptations: List[Dict[str, Any]]) -> float:
        """Calculate adaptation score"""
        return min(1.0, len(adaptations) / 5.0)

    def _assess_execution_completion(self) -> float:
        """Assess execution completion"""
        return 0.8  # Default completion score

    def _generate_analysis_insights(self, market_analysis: Dict[str, Any], performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate analysis insights"""
        insights = []
        if market_analysis.get('complexity', 0.5) > 0.7:
            insights.append('High market complexity detected')
        if performance_analysis.get('system_health', 0.5) < 0.6:
            insights.append('System health requires attention')
        return insights

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no planning data is available"""
        self.logger.warning("No planning data available - using cached state")
        
        return {
            'current_phase': self.current_phase.value,
            'planning_cycle': self.planning_cycle,
            'cognitive_load': self.cognitive_load,
            'fallback_reason': 'no_planning_data'
        }

    async def _handle_planning_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle planning operation errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "MetaCognitivePlanner")
        explanation = self.english_explainer.explain_error(
            "MetaCognitivePlanner", str(error), "planning operations"
        )
        
        self.logger.error(
            format_operator_message(
                "[CRASH]", "PLANNING_OPERATION_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="metacognitive_planning"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'current_phase': self.current_phase.value,
            'planning_cycle': self.planning_cycle,
            'cognitive_load': self.cognitive_load,
            'fallback_reason': reason,
            'circuit_breaker_state': self.circuit_breaker['state']
        }

    def _update_planning_health(self):
        """Update planning health metrics"""
        try:
            # Check planning effectiveness
            if self.planning_effectiveness:
                avg_effectiveness = np.mean([
                    data['successful'] / max(data['total'], 1) 
                    for data in self.planning_effectiveness.values()
                ])
                if avg_effectiveness < 0.5:
                    self._health_status = 'warning'
                else:
                    self._health_status = 'healthy'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_planning_performance(self):
        """Analyze planning performance metrics"""
        try:
            if len(self.episode_history) > 10:
                recent_performance = [ep.get('pnl', 0) for ep in list(self.episode_history)[-10:]]
                avg_performance = np.mean(recent_performance)
                
                if avg_performance > 20:
                    self.logger.info(
                        format_operator_message(
                            "[TARGET]", "HIGH_PERFORMANCE_DETECTED",
                            avg_performance=f"{avg_performance:.2f}",
                            episodes_analyzed=len(recent_performance),
                            context="planning_performance"
                        )
                    )
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'MetaCognitivePlanner', 'planning_cycle', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'MetaCognitivePlanner', 'planning_cycle', 0, False
        )

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in planning recommendations"""
        try:
            base_confidence = self.planning_confidence
            
            # Adjust based on cognitive load
            load_adjustment = (1.0 - self.cognitive_load) * 0.2
            
            # Adjust based on strategy coherence
            coherence_adjustment = self.strategy_coherence * 0.2
            
            # Adjust based on phase
            phase_adjustment = 0.1 if self.current_phase == PlanningPhase.EXECUTION else 0.0
            
            # Action-specific adjustments
            if isinstance(action, dict):
                action_type = action.get('action_type', 'unknown')
                if action_type in ['strategic_plan', 'tactical_recommendation']:
                    action_adjustment = 0.1
                else:
                    action_adjustment = 0.0
            else:
                action_adjustment = 0.0
            
            confidence = base_confidence + load_adjustment + coherence_adjustment + phase_adjustment + action_adjustment
            
            return float(np.clip(confidence, 0.1, 1.0))
            
        except Exception as e:
            self.logger.error(f"Planning confidence calculation failed: {e}")
            return 0.5

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'config': self.config.__dict__,
            'genome': self.genome.copy(),
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
            'learning_history': list(self.learning_history),
            'health_status': self._health_status,
            'circuit_breaker': self.circuit_breaker.copy()
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'current_phase' in state:
            self.current_phase = PlanningPhase(state['current_phase'])
        
        if 'planning_cycle' in state:
            self.planning_cycle = state['planning_cycle']
        
        if 'phase_start_time' in state:
            self.phase_start_time = datetime.fromisoformat(state['phase_start_time'])
        
        if 'episode_history' in state:
            self.episode_history = deque(state['episode_history'], maxlen=self.genome["window"])
        
        if 'session_plans' in state:
            self.session_plans = deque(state['session_plans'], maxlen=10)
        
        if 'adaptation_history' in state:
            self.adaptation_history = deque(state['adaptation_history'], maxlen=50)
        
        if 'strategic_objectives' in state:
            self.strategic_objectives.update(state['strategic_objectives'])
        
        if 'planning_effectiveness' in state:
            self.planning_effectiveness.update(state['planning_effectiveness'])
        
        if 'cognitive_metrics' in state:
            metrics = state['cognitive_metrics']
            self.cognitive_load = metrics.get('load', 0.5)
            self.planning_confidence = metrics.get('confidence', 0.6)
            self.strategy_coherence = metrics.get('coherence', 0.7)
            self.adaptation_speed = metrics.get('adaptation_speed', 0.5)
        
        if 'current_recommendations' in state:
            self.current_recommendations = state['current_recommendations']
        
        if 'strategic_insights' in state:
            self.strategic_insights = deque(state['strategic_insights'], maxlen=20)
        
        if 'learning_history' in state:
            self.learning_history = deque(state['learning_history'], maxlen=100)
        
        if 'health_status' in state:
            self._health_status = state['health_status']
        
        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'current_phase': self.current_phase.value,
            'planning_confidence': self.planning_confidence,
            'cognitive_load': self.cognitive_load
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    # Legacy compatibility methods
    def record_episode(self, result: Dict[str, Any]):
        """Legacy compatibility for episode recording"""
        if isinstance(result, dict) and 'pnl' in result:
            enhanced_result = {
                **result,
                'timestamp': datetime.now().isoformat(),
                'planning_cycle': self.planning_cycle,
                'planning_phase': self.current_phase.value
            }
            self.episode_history.append(enhanced_result)

    def step(self, **kwargs):
        """Legacy compatibility step method"""
        pass

    def get_observation_components(self) -> np.ndarray:
        """Legacy compatibility for observation components"""
        try:
            # Cognitive metrics
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
            else:
                win_rate = 0.5
                avg_pnl = 0.0
            
            performance_components = [
                win_rate,
                avg_pnl / 100.0,  # Normalize
                float(len(self.episode_history)) / self.genome["window"]
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
                observation = np.nan_to_num(observation)
            
            return observation
            
        except Exception as e:
            return np.zeros(13, dtype=np.float32)

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose planning-based action"""
        try:
            # Planning actions focus on strategic and tactical recommendations
            current_phase = self.current_phase.value
            cognitive_load = self.cognitive_load
            planning_confidence = self.planning_confidence
            
            # Propose action based on current planning phase
            if self.current_phase == PlanningPhase.PLANNING:
                action_type = "strategic_plan"
                magnitude = min(planning_confidence * 1.5, 1.0)
            elif self.current_phase == PlanningPhase.EXECUTION:
                action_type = "execution_adjustment"
                magnitude = max(0.5, planning_confidence)
            elif self.current_phase == PlanningPhase.ANALYSIS:
                action_type = "tactical_recommendation" 
                magnitude = planning_confidence
            elif self.current_phase == PlanningPhase.REFLECTION:
                action_type = "planning_analysis"
                magnitude = planning_confidence * 0.8
            else:  # ADAPTATION
                action_type = "adaptation_plan"
                magnitude = 0.6
            
            return {
                'action_type': action_type,
                'magnitude': magnitude,
                'confidence': planning_confidence,
                'reasoning': f"Planning phase: {current_phase}, cognitive load: {cognitive_load:.2f}",
                'current_phase': current_phase,
                'cognitive_load': cognitive_load,
                'planning_cycle': self.planning_cycle
            }
            
        except Exception as e:
            self.logger.error(f"Planning action proposal failed: {e}")
            return {
                'action_type': 'no_action',
                'confidence': 0.0,
                'reasoning': f'Planning action proposal error: {str(e)}',
                'error': str(e)
            }

    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        # Since calculate_confidence is now async, we need to return a basic confidence
        return float(self.planning_confidence)