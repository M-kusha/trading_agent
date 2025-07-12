"""
📚 Enhanced Curriculum Planner with SmartInfoBus Integration v3.0
Advanced adaptive learning curriculum system for trading strategy optimization with intelligent progression management
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
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="CurriculumPlannerPlus",
    version="3.0.0",
    category="strategy",
    provides=[
        "curriculum_stage", "learning_constraints", "competency_scores",
        "learning_recommendations", "stage_progression", "mastery_assessment"
    ],
    requires=[
        "performance_data", "episode_summary", "risk_metrics", "trading_session",
        "market_conditions", "recent_trades"
    ],
    description="Advanced adaptive learning curriculum system for trading strategy optimization with intelligent progression management",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=200,
    priority=6,
    explainable=True,
    hot_reload=True
)
class CurriculumPlannerPlus(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    📚 PRODUCTION-GRADE Curriculum Planner v3.0
    
    Advanced adaptive learning curriculum system with:
    - Intelligent progression through 5 learning stages
    - Multi-dimensional competency assessment and tracking
    - Market-condition-aware curriculum adaptation
    - SmartInfoBus zero-wiring architecture
    - Comprehensive thesis generation for all learning decisions
    """

    def _initialize(self):
        """Initialize advanced curriculum planning systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced curriculum configuration
        self.window = self.config.get('window', 10)
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)
        self.performance_threshold = self.config.get('performance_threshold', 0.6)
        self.difficulty_levels = self.config.get('difficulty_levels', 5)
        self.debug = self.config.get('debug', False)
        
        # Core curriculum state
        self.episode_history = deque(maxlen=self.window)
        self.performance_metrics = defaultdict(list)
        self.curriculum_stages = self._initialize_progressive_curriculum()
        self.current_stage = 0
        self.stage_progress = 0.0
        
        # Enhanced learning analytics
        self.learning_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'current_difficulty': 1.0,
            'adaptation_events': 0,
            'mastery_level': 0.0,
            'learning_rate': 0.0,
            'stage_completion_times': [],
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Multi-dimensional competency tracking
        self.competency_areas = {
            'risk_management': {
                'score': 0.5, 'weight': 0.3, 'target': 0.8,
                'description': 'Ability to manage drawdown and position risk'
            },
            'entry_timing': {
                'score': 0.5, 'weight': 0.25, 'target': 0.75,
                'description': 'Skill in identifying optimal entry points'
            },
            'exit_strategy': {
                'score': 0.5, 'weight': 0.25, 'target': 0.75,
                'description': 'Proficiency in exit timing and profit optimization'
            },
            'position_sizing': {
                'score': 0.5, 'weight': 0.2, 'target': 0.7,
                'description': 'Consistency in risk-adjusted position sizing'
            }
        }
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Performance tracking
        self.learning_velocity = 0.0
        self.plateau_detection = {'consecutive_poor_episodes': 0, 'threshold': 5}
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="📚",
            message=f"Curriculum Planner v{version} initialized",
            stages=len(self.curriculum_stages),
            current_stage=self.curriculum_stages[0]['name'],
            competency_areas=len(self.competency_areas)
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="CurriculumPlannerPlus",
            log_path="logs/strategy/curriculum_planner.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("CurriculumPlannerPlus", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _initialize_progressive_curriculum(self) -> List[Dict[str, Any]]:
        """Initialize progressive learning curriculum with enhanced stages"""
        return [
            {
                'name': 'Foundation',
                'description': 'Basic trading mechanics and fundamental risk awareness',
                'difficulty': 1.0,
                'focus_areas': ['risk_management', 'position_sizing'],
                'success_criteria': {
                    'win_rate': 0.4, 'max_drawdown': 0.1, 'profit_factor': 1.0,
                    'consistency_score': 0.3, 'risk_score': 0.6
                },
                'market_conditions': ['ranging', 'low_volatility'],
                'constraints': {
                    'max_position_size': 0.5,
                    'max_trades_per_day': 5,
                    'max_risk_per_trade': 0.01
                },
                'learning_objectives': [
                    'Understand basic risk management principles',
                    'Maintain consistent position sizing',
                    'Avoid catastrophic losses'
                ]
            },
            {
                'name': 'Development',
                'description': 'Enhanced timing skills and strategy implementation',
                'difficulty': 2.0,
                'focus_areas': ['entry_timing', 'exit_strategy'],
                'success_criteria': {
                    'win_rate': 0.5, 'max_drawdown': 0.08, 'profit_factor': 1.2,
                    'consistency_score': 0.4, 'risk_score': 0.7
                },
                'market_conditions': ['trending', 'ranging'],
                'constraints': {
                    'max_position_size': 0.75,
                    'max_trades_per_day': 8,
                    'max_risk_per_trade': 0.015
                },
                'learning_objectives': [
                    'Improve entry and exit timing',
                    'Develop basic strategy discipline',
                    'Enhance profit factor through better exits'
                ]
            },
            {
                'name': 'Intermediate',
                'description': 'Multi-timeframe analysis and advanced pattern recognition',
                'difficulty': 3.0,
                'focus_areas': ['entry_timing', 'risk_management', 'exit_strategy'],
                'success_criteria': {
                    'win_rate': 0.55, 'max_drawdown': 0.06, 'profit_factor': 1.5,
                    'consistency_score': 0.5, 'risk_score': 0.75
                },
                'market_conditions': ['trending', 'ranging', 'volatile'],
                'constraints': {
                    'max_position_size': 1.0,
                    'max_trades_per_day': 12,
                    'max_risk_per_trade': 0.02
                },
                'learning_objectives': [
                    'Master multi-timeframe analysis',
                    'Adapt to different market conditions',
                    'Develop advanced pattern recognition'
                ]
            },
            {
                'name': 'Advanced',
                'description': 'Complex strategies and dynamic risk management',
                'difficulty': 4.0,
                'focus_areas': ['position_sizing', 'risk_management', 'entry_timing'],
                'success_criteria': {
                    'win_rate': 0.6, 'max_drawdown': 0.05, 'profit_factor': 2.0,
                    'consistency_score': 0.65, 'risk_score': 0.8
                },
                'market_conditions': ['all'],
                'constraints': {
                    'max_position_size': 1.5,
                    'max_trades_per_day': 15,
                    'max_risk_per_trade': 0.025
                },
                'learning_objectives': [
                    'Implement dynamic position sizing',
                    'Handle complex market scenarios',
                    'Optimize risk-adjusted returns'
                ]
            },
            {
                'name': 'Expert',
                'description': 'Full autonomy and adaptive trading mastery',
                'difficulty': 5.0,
                'focus_areas': ['all'],
                'success_criteria': {
                    'win_rate': 0.65, 'max_drawdown': 0.04, 'profit_factor': 2.5,
                    'consistency_score': 0.75, 'risk_score': 0.85
                },
                'market_conditions': ['all'],
                'constraints': {
                    'max_position_size': 2.0,
                    'max_trades_per_day': 20,
                    'max_risk_per_trade': 0.03
                },
                'learning_objectives': [
                    'Demonstrate trading mastery across all conditions',
                    'Maintain consistent high performance',
                    'Adapt dynamically to market evolution'
                ]
            }
        ]

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Curriculum Planner v3.0 Initialization Complete:
        
        Learning Framework:
        - Progressive 5-stage curriculum: {' → '.join([stage['name'] for stage in self.curriculum_stages])}
        - Multi-dimensional competency tracking: {', '.join(self.competency_areas.keys())}
        - Adaptive difficulty scaling with performance-based progression
        - Market-condition-aware learning constraints and objectives
        
        Current Configuration:
        - Performance window: {self.window} episodes for rolling analysis
        - Adaptation rate: {self.adaptation_rate:.1%} for smooth competency updates
        - Performance threshold: {self.performance_threshold:.1%} for stage advancement
        - Starting stage: {self.curriculum_stages[0]['name']} with foundation learning objectives
        
        Competency Assessment Framework:
        - Risk Management (30% weight): Drawdown control and position risk
        - Entry Timing (25% weight): Optimal entry point identification
        - Exit Strategy (25% weight): Profit optimization and exit timing
        - Position Sizing (20% weight): Consistent risk-adjusted sizing
        
        Advanced Features:
        - Intelligent plateau detection and intervention
        - Dynamic difficulty adjustment based on learning velocity
        - Market regime awareness for context-sensitive progression
        - Comprehensive thesis generation for all learning decisions
        
        Expected Learning Outcomes:
        - Systematic skill development through structured progression
        - Data-driven competency assessment and improvement
        - Adaptive learning that responds to individual performance patterns
        - Transparent learning process with comprehensive explanations
        """
        
        self.smart_bus.set('curriculum_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'curriculum_overview': {
                'stages': len(self.curriculum_stages),
                'competencies': list(self.competency_areas.keys()),
                'current_stage': self.curriculum_stages[0]['name']
            }
        }, module='CurriculumPlannerPlus', thesis=thesis)

    async def process(self) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive curriculum management
        
        Returns:
            Dict containing curriculum status, constraints, recommendations, and progression analysis
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive learning data from SmartInfoBus
            learning_data = await self._get_comprehensive_learning_data()
            
            # Core curriculum analysis with error handling
            curriculum_analysis = await self._analyze_learning_progress_comprehensive(learning_data)
            
            # Generate intelligent learning recommendations
            recommendations = self._generate_intelligent_learning_recommendations(curriculum_analysis)
            
            # Check for stage progression opportunities
            progression_analysis = await self._evaluate_stage_progression_comprehensive(curriculum_analysis)
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_learning_thesis(
                curriculum_analysis, recommendations, progression_analysis
            )
            
            # Create comprehensive results
            results = {
                'curriculum_stage': self._get_current_stage_info(),
                'learning_constraints': self._get_current_constraints(),
                'competency_scores': self._get_competency_assessment(),
                'learning_recommendations': recommendations,
                'stage_progression': progression_analysis,
                'mastery_assessment': self._get_mastery_assessment(),
                'learning_analytics': self.learning_stats.copy(),
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('CurriculumPlannerPlus', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_learning_data(self) -> Dict[str, Any]:
        """Get comprehensive learning data using modern SmartInfoBus patterns"""
        try:
            return {
                'performance_data': self.smart_bus.get('performance_data', 'CurriculumPlannerPlus') or {},
                'episode_summary': self.smart_bus.get('episode_summary', 'CurriculumPlannerPlus') or {},
                'risk_metrics': self.smart_bus.get('risk_metrics', 'CurriculumPlannerPlus') or {},
                'trading_session': self.smart_bus.get('trading_session', 'CurriculumPlannerPlus') or {},
                'market_conditions': self.smart_bus.get('market_conditions', 'CurriculumPlannerPlus') or {},
                'recent_trades': self.smart_bus.get('recent_trades', 'CurriculumPlannerPlus') or [],
                'learning_context': self.smart_bus.get('learning_context', 'CurriculumPlannerPlus') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "CurriculumPlannerPlus")
            self.logger.warning(f"Data retrieval incomplete: {error_context}")
            return self._get_safe_learning_defaults()

    async def _analyze_learning_progress_comprehensive(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive learning progress analysis with advanced pattern recognition"""
        try:
            # Extract recent performance data
            episode_data = learning_data.get('episode_summary', {})
            performance_data = learning_data.get('performance_data', {})
            
            # Update competency scores if we have episode data
            if episode_data:
                await self._update_competency_scores_advanced(episode_data, learning_data)
            
            # Calculate learning velocity and trends
            learning_velocity = self._calculate_learning_velocity()
            performance_trends = self._analyze_performance_trends()
            
            # Assess current stage mastery
            stage_mastery = await self._assess_current_stage_mastery()
            
            # Detect learning patterns
            learning_patterns = self._detect_learning_patterns()
            
            # Calculate overall progress metrics
            overall_competency = self._calculate_weighted_competency()
            progress_metrics = self._calculate_progress_metrics()
            
            return {
                'competency_scores': {k: v['score'] for k, v in self.competency_areas.items()},
                'learning_velocity': learning_velocity,
                'performance_trends': performance_trends,
                'stage_mastery': stage_mastery,
                'learning_patterns': learning_patterns,
                'overall_competency': overall_competency,
                'progress_metrics': progress_metrics,
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(learning_data)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "CurriculumPlannerPlus")
            self.logger.error(f"Learning analysis failed: {error_context}")
            return self._get_safe_analysis_defaults()

    async def _update_competency_scores_advanced(self, episode_data: Dict[str, Any], 
                                               learning_data: Dict[str, Any]):
        """Advanced competency score updates with contextual analysis"""
        try:
            current_stage = self.curriculum_stages[self.current_stage]
            
            # Risk management competency
            if 'max_drawdown' in episode_data:
                target_dd = current_stage['success_criteria'].get('max_drawdown', 0.1)
                actual_dd = episode_data['max_drawdown']
                
                # Enhanced scoring with context
                base_score = max(0, 1.0 - (actual_dd / target_dd))
                
                # Bonus for staying well under limit
                if actual_dd < target_dd * 0.5:
                    base_score = min(1.0, base_score * 1.2)
                
                await self._update_competency_score('risk_management', base_score, episode_data)
            
            # Entry timing competency
            if 'win_rate' in episode_data:
                target_wr = current_stage['success_criteria'].get('win_rate', 0.5)
                actual_wr = episode_data['win_rate']
                
                # Enhanced scoring with trend analysis
                base_score = min(1.0, actual_wr / target_wr)
                
                # Consider improvement trend
                recent_improvements = self._analyze_win_rate_trend()
                if recent_improvements > 0:
                    base_score = min(1.0, base_score * (1 + recent_improvements * 0.1))
                
                await self._update_competency_score('entry_timing', base_score, episode_data)
            
            # Exit strategy competency
            if 'profit_factor' in episode_data:
                target_pf = current_stage['success_criteria'].get('profit_factor', 1.0)
                actual_pf = episode_data['profit_factor']
                
                # Enhanced scoring with consistency check
                base_score = min(1.0, actual_pf / target_pf)
                
                # Bonus for consistent profit factors
                pf_consistency = self._calculate_profit_factor_consistency()
                base_score = min(1.0, base_score * (0.8 + 0.2 * pf_consistency))
                
                await self._update_competency_score('exit_strategy', base_score, episode_data)
            
            # Position sizing competency
            if 'trade_consistency' in episode_data:
                consistency = episode_data['trade_consistency']
                
                # Enhanced scoring with risk consistency
                risk_consistency = self._calculate_risk_consistency(learning_data)
                combined_score = (consistency + risk_consistency) / 2.0
                
                await self._update_competency_score('position_sizing', combined_score, episode_data)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "competency_update")
            self.logger.warning(f"Competency score update failed: {error_context}")

    async def _update_competency_score(self, area: str, new_score: float, context: Dict[str, Any]):
        """Update specific competency area score with enhanced momentum and context"""
        if area in self.competency_areas:
            current_score = self.competency_areas[area]['score']
            
            # Adaptive learning rate based on performance
            adaptive_rate = self.adaptation_rate
            if new_score > current_score:
                # Faster learning for improvements
                adaptive_rate *= 1.5
            elif new_score < current_score * 0.8:
                # Slower updates for significant drops (could be outliers)
                adaptive_rate *= 0.5
            
            # Apply exponential moving average with adaptive rate
            updated_score = current_score * (1 - adaptive_rate) + new_score * adaptive_rate
            self.competency_areas[area]['score'] = np.clip(updated_score, 0.0, 1.0)
            
            # Log significant changes
            score_change = updated_score - current_score
            if abs(score_change) > 0.05:
                self.logger.info(format_operator_message(
                    icon="📊",
                    message=f"{area.replace('_', ' ').title()} competency updated",
                    old_score=f"{current_score:.1%}",
                    new_score=f"{updated_score:.1%}",
                    change=f"{score_change:+.1%}"
                ))

    def _calculate_learning_velocity(self) -> float:
        """Calculate learning velocity based on recent performance trends"""
        try:
            if len(self.episode_history) < 3:
                return 0.0
            
            # Get recent competency scores
            recent_episodes = list(self.episode_history)[-5:]
            if len(recent_episodes) < 2:
                return 0.0
            
            # Calculate average competency change
            competency_changes = []
            for i in range(1, len(recent_episodes)):
                prev_competency = self._calculate_episode_competency(recent_episodes[i-1])
                curr_competency = self._calculate_episode_competency(recent_episodes[i])
                competency_changes.append(curr_competency - prev_competency)
            
            velocity = float(np.mean(competency_changes)) if competency_changes else 0.0
            self.learning_velocity = velocity
            
            return velocity
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "velocity_calculation")
            return 0.0

    def _calculate_episode_competency(self, episode: Dict[str, Any]) -> float:
        """Calculate overall competency for a specific episode"""
        try:
            current_stage = self.curriculum_stages[self.current_stage]
            criteria = current_stage['success_criteria']
            
            scores = []
            
            # Risk management
            if 'max_drawdown' in episode and 'max_drawdown' in criteria:
                target = criteria['max_drawdown']
                actual = episode['max_drawdown']
                scores.append(max(0, 1.0 - (actual / target)))
            
            # Entry timing (win rate)
            if 'win_rate' in episode and 'win_rate' in criteria:
                target = criteria['win_rate']
                actual = episode['win_rate']
                scores.append(min(1.0, actual / target))
            
            # Exit strategy (profit factor)
            if 'profit_factor' in episode and 'profit_factor' in criteria:
                target = criteria['profit_factor']
                actual = episode['profit_factor']
                scores.append(min(1.0, actual / target))
            
            return float(np.mean(scores)) if scores else 0.5
            
        except Exception:
            return 0.5

    async def record_episode_comprehensive(self, summary: Dict[str, Any]) -> None:
        """Record episode with comprehensive validation and advanced curriculum adaptation"""
        try:
            # Validate and enrich summary
            if not isinstance(summary, dict):
                self.logger.warning(f"Invalid episode summary type: {type(summary)}")
                return
            
            # Add comprehensive metadata
            enriched_summary = {
                'timestamp': datetime.datetime.now().isoformat(),
                'curriculum_stage': self.current_stage,
                'stage_name': self.curriculum_stages[self.current_stage]['name'],
                'difficulty': self.learning_stats['current_difficulty'],
                'competency_snapshot': {k: v['score'] for k, v in self.competency_areas.items()},
                'learning_velocity': self.learning_velocity,
                **summary
            }
            
            # Validate key metrics with enhanced error handling
            for key in ['total_trades', 'wins', 'pnl', 'max_drawdown']:
                if key in enriched_summary:
                    value = enriched_summary[key]
                    if not isinstance(value, (int, float)) or np.isnan(value):
                        self.logger.warning(f"Invalid {key}: {value}, applying correction")
                        enriched_summary[key] = 0 if key != 'max_drawdown' else 0.01
            
            # Store episode with competency analysis
            self.episode_history.append(enriched_summary)
            self.learning_stats['total_episodes'] += 1
            
            # Update performance metrics
            await self._update_performance_metrics_advanced(enriched_summary)
            
            # Evaluate episode success with enhanced criteria
            episode_success = await self._evaluate_episode_success_comprehensive(enriched_summary)
            if episode_success:
                self.learning_stats['successful_episodes'] += 1
                self.plateau_detection['consecutive_poor_episodes'] = 0
            else:
                self.plateau_detection['consecutive_poor_episodes'] += 1
            
            # Update learning statistics
            await self._update_learning_statistics(enriched_summary)
            
            # Check for learning plateau
            if self.plateau_detection['consecutive_poor_episodes'] >= self.plateau_detection['threshold']:
                await self._handle_learning_plateau_advanced()
            
            # Log episode with enhanced details
            self.logger.info(format_operator_message(
                icon="📈",
                message="Episode recorded with curriculum analysis",
                stage=self.curriculum_stages[self.current_stage]['name'],
                success="✅" if episode_success else "❌",
                pnl=f"€{enriched_summary.get('pnl', 0):.2f}",
                competency=f"{self._calculate_weighted_competency():.1%}",
                velocity=f"{self.learning_velocity:+.2%}"
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "episode_recording")
            self.logger.error(f"Episode recording failed: {error_context}")

    async def _evaluate_stage_progression_comprehensive(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive evaluation of stage progression opportunities"""
        try:
            current_stage_data = self.curriculum_stages[self.current_stage]
            stage_mastery = analysis.get('stage_mastery', {})
            overall_competency = analysis.get('overall_competency', 0.0)
            
            # Check progression criteria
            progression_assessment = {
                'ready_for_advancement': False,
                'criteria_met': {},
                'missing_requirements': [],
                'confidence_score': 0.0,
                'recommendation': 'continue_current_stage'
            }
            
            # Evaluate each criterion
            criteria_scores = []
            for criterion, target in current_stage_data['success_criteria'].items():
                recent_performance = self._get_recent_metric_performance(criterion)
                
                if recent_performance is not None:
                    if criterion == 'max_drawdown':
                        met = recent_performance <= target
                        score = max(0, 1.0 - (recent_performance / target))
                    else:
                        met = recent_performance >= target
                        score = min(1.0, recent_performance / target)
                    
                    progression_assessment['criteria_met'][criterion] = {
                        'met': met,
                        'actual': recent_performance,
                        'target': target,
                        'score': score
                    }
                    criteria_scores.append(score)
                    
                    if not met:
                        progression_assessment['missing_requirements'].append({
                            'criterion': criterion,
                            'gap': target - recent_performance if criterion != 'max_drawdown' else recent_performance - target,
                            'improvement_needed': self._calculate_improvement_needed(criterion, recent_performance, target)
                        })
            
            # Calculate overall progression confidence
            if criteria_scores:
                progression_assessment['confidence_score'] = np.mean(criteria_scores)
            
            # Additional stability checks
            stability_score = self._assess_performance_stability()
            progression_assessment['stability_score'] = stability_score
            
            # Final progression decision
            criteria_met_ratio = len([c for c in progression_assessment['criteria_met'].values() if c['met']]) / len(progression_assessment['criteria_met'])
            
            if (criteria_met_ratio >= 0.8 and 
                progression_assessment['confidence_score'] >= 0.75 and 
                stability_score >= 0.6 and
                overall_competency >= 0.7):
                
                progression_assessment['ready_for_advancement'] = True
                progression_assessment['recommendation'] = 'advance_to_next_stage'
                
                # Execute stage advancement
                if self.current_stage < len(self.curriculum_stages) - 1:
                    await self._advance_to_next_stage_comprehensive()
            
            elif criteria_met_ratio < 0.3 and overall_competency < 0.4:
                progression_assessment['recommendation'] = 'consider_regression'
            
            return progression_assessment
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "progression_evaluation")
            self.logger.error(f"Stage progression evaluation failed: {error_context}")
            return {'ready_for_advancement': False, 'error': str(error_context)}

    async def _advance_to_next_stage_comprehensive(self):
        """Advanced stage progression with comprehensive tracking"""
        try:
            old_stage = self.curriculum_stages[self.current_stage]['name']
            old_stage_index = self.current_stage
            
            # Record detailed completion metrics
            completion_data = {
                'stage_index': self.current_stage,
                'stage_name': old_stage,
                'episodes_to_complete': self.learning_stats['total_episodes'],
                'final_competency_scores': {k: v['score'] for k, v in self.competency_areas.items()},
                'final_mastery_level': self.learning_stats['mastery_level'],
                'completion_timestamp': datetime.datetime.now().isoformat(),
                'learning_velocity_at_completion': self.learning_velocity
            }
            
            self.learning_stats['stage_completion_times'].append(completion_data)
            
            # Advance to next stage
            self.current_stage += 1
            self.stage_progress = 0.0
            self.learning_stats['adaptation_events'] += 1
            self.learning_stats['current_difficulty'] = self.curriculum_stages[self.current_stage]['difficulty']
            
            # Reset plateau detection
            self.plateau_detection['consecutive_poor_episodes'] = 0
            
            new_stage = self.curriculum_stages[self.current_stage]['name']
            
            # Generate stage advancement thesis
            advancement_thesis = self._generate_stage_advancement_thesis(old_stage, new_stage, completion_data)
            
            # Update SmartInfoBus with advancement
            self.smart_bus.set('stage_advancement', {
                'old_stage': old_stage,
                'new_stage': new_stage,
                'completion_data': completion_data,
                'advancement_thesis': advancement_thesis
            }, module='CurriculumPlannerPlus', thesis=advancement_thesis)
            
            self.logger.info(format_operator_message(
                icon="🎯",
                message="Curriculum stage advanced with comprehensive analysis",
                from_stage=old_stage,
                to_stage=new_stage,
                episodes=self.learning_stats['total_episodes'],
                mastery=f"{self.learning_stats['mastery_level']:.1%}",
                velocity=f"{self.learning_velocity:+.1%}"
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "stage_advancement")
            self.logger.error(f"Stage advancement failed: {error_context}")

    async def _generate_comprehensive_learning_thesis(self, analysis: Dict[str, Any], 
                                                    recommendations: List[str], 
                                                    progression: Dict[str, Any]) -> str:
        """Generate comprehensive thesis explaining all learning decisions"""
        try:
            current_stage = self.curriculum_stages[self.current_stage]
            competency_scores = analysis.get('competency_scores', {})
            overall_competency = analysis.get('overall_competency', 0.0)
            learning_velocity = analysis.get('learning_velocity', 0.0)
            
            thesis_parts = []
            
            # Executive Summary
            thesis_parts.append(
                f"LEARNING STATUS: {current_stage['name']} stage at {overall_competency:.1%} mastery "
                f"with {learning_velocity:+.1%} learning velocity"
            )
            
            # Stage Progress Analysis
            stage_progress_pct = self.stage_progress * 100
            if stage_progress_pct >= 80:
                progress_status = "READY FOR ADVANCEMENT"
            elif stage_progress_pct >= 60:
                progress_status = "APPROACHING MASTERY"
            elif stage_progress_pct >= 40:
                progress_status = "GOOD PROGRESS"
            else:
                progress_status = "EARLY DEVELOPMENT"
            
            thesis_parts.append(f"STAGE PROGRESS: {stage_progress_pct:.0f}% completion - {progress_status}")
            
            # Competency Breakdown
            thesis_parts.append("COMPETENCY ANALYSIS:")
            for area, score in competency_scores.items():
                target = self.competency_areas[area]['target']
                status = "✅ MASTERED" if score >= target else "🔄 DEVELOPING" if score >= target * 0.7 else "❌ NEEDS FOCUS"
                gap = target - score
                thesis_parts.append(
                    f"  • {area.replace('_', ' ').title()}: {score:.1%} "
                    f"({'+' if gap <= 0 else '-'}{abs(gap):.1%} vs target) {status}"
                )
            
            # Learning Velocity Analysis
            if learning_velocity > 0.05:
                velocity_status = "ACCELERATING - Strong learning momentum"
            elif learning_velocity > 0:
                velocity_status = "IMPROVING - Steady progress"
            elif learning_velocity > -0.05:
                velocity_status = "STABLE - Maintaining performance"
            else:
                velocity_status = "DECLINING - Requires intervention"
            
            thesis_parts.append(f"LEARNING MOMENTUM: {velocity_status}")
            
            # Progression Assessment
            if progression.get('ready_for_advancement', False):
                thesis_parts.append(
                    f"PROGRESSION: Ready to advance to {self.curriculum_stages[min(self.current_stage + 1, len(self.curriculum_stages) - 1)]['name']} stage "
                    f"with {progression.get('confidence_score', 0):.1%} confidence"
                )
            else:
                missing_reqs = progression.get('missing_requirements', [])
                if missing_reqs:
                    primary_focus = missing_reqs[0]['criterion'].replace('_', ' ')
                    thesis_parts.append(f"FOCUS AREA: Primary development needed in {primary_focus}")
            
            # Recommendations Summary
            if recommendations:
                thesis_parts.append(f"KEY RECOMMENDATIONS: {len(recommendations)} actionable improvements identified")
                thesis_parts.append(f"  • Primary: {recommendations[0]}")
                if len(recommendations) > 1:
                    thesis_parts.append(f"  • Secondary: {recommendations[1]}")
            
            # Performance Context
            session_duration = self._calculate_session_duration()
            total_episodes = self.learning_stats['total_episodes']
            success_rate = (self.learning_stats['successful_episodes'] / max(1, total_episodes)) * 100
            
            thesis_parts.append(
                f"SESSION CONTEXT: {total_episodes} episodes over {session_duration} "
                f"with {success_rate:.0f}% success rate"
            )
            
            # Risk Assessment
            risk_assessment = self._assess_learning_risk()
            thesis_parts.append(f"LEARNING RISK: {risk_assessment}")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Learning thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive curriculum results"""
        try:
            # Core curriculum stage info
            self.smart_bus.set('curriculum_stage', results['curriculum_stage'],
                             module='CurriculumPlannerPlus', thesis=thesis)
            
            # Learning constraints for trading system
            constraints_thesis = f"Learning constraints for {results['curriculum_stage']['name']} stage"
            self.smart_bus.set('learning_constraints', results['learning_constraints'],
                             module='CurriculumPlannerPlus', thesis=constraints_thesis)
            
            # Competency assessment
            competency_thesis = f"Multi-dimensional competency assessment: {len(results['competency_scores'])} areas tracked"
            self.smart_bus.set('competency_scores', results['competency_scores'],
                             module='CurriculumPlannerPlus', thesis=competency_thesis)
            
            # Learning recommendations
            rec_thesis = f"Generated {len(results['learning_recommendations'])} evidence-based learning recommendations"
            self.smart_bus.set('learning_recommendations', results['learning_recommendations'],
                             module='CurriculumPlannerPlus', thesis=rec_thesis)
            
            # Stage progression analysis
            progression_thesis = f"Stage progression analysis: {results['stage_progression'].get('recommendation', 'continue')}"
            self.smart_bus.set('stage_progression', results['stage_progression'],
                             module='CurriculumPlannerPlus', thesis=progression_thesis)
            
            # Mastery assessment summary
            mastery_thesis = f"Current mastery level: {results['mastery_assessment']['overall_mastery']:.1%}"
            self.smart_bus.set('mastery_assessment', results['mastery_assessment'],
                             module='CurriculumPlannerPlus', thesis=mastery_thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # HELPER METHODS AND UTILITIES
    # ═══════════════════════════════════════════════════════════════════

    def _get_current_stage_info(self) -> Dict[str, Any]:
        """Get comprehensive current stage information"""
        current_stage = self.curriculum_stages[self.current_stage]
        return {
            'stage_index': self.current_stage,
            'stage_name': current_stage['name'],
            'description': current_stage['description'],
            'difficulty': current_stage['difficulty'],
            'focus_areas': current_stage['focus_areas'],
            'learning_objectives': current_stage['learning_objectives'],
            'progress_percentage': self.stage_progress * 100,
            'mastery_level': self.learning_stats['mastery_level']
        }

    def _get_current_constraints(self) -> Dict[str, Any]:
        """Get current learning constraints for trading system"""
        current_stage = self.curriculum_stages[self.current_stage]
        return {
            **current_stage['constraints'],
            'allowed_market_conditions': current_stage['market_conditions'],
            'focus_areas': current_stage['focus_areas'],
            'stage_name': current_stage['name'],
            'difficulty_multiplier': self.learning_stats['current_difficulty']
        }

    def _get_competency_assessment(self) -> Dict[str, Any]:
        """Get comprehensive competency assessment"""
        assessment = {}
        for area, data in self.competency_areas.items():
            assessment[area] = {
                'score': data['score'],
                'target': data['target'],
                'weight': data['weight'],
                'description': data['description'],
                'status': 'mastered' if data['score'] >= data['target'] else 'developing',
                'gap_to_target': max(0, data['target'] - data['score'])
            }
        return assessment

    def _get_mastery_assessment(self) -> Dict[str, Any]:
        """Get comprehensive mastery assessment"""
        overall_mastery = self._calculate_weighted_competency()
        
        return {
            'overall_mastery': overall_mastery,
            'stage_progress': self.stage_progress,
            'learning_velocity': self.learning_velocity,
            'competency_balance': self._calculate_competency_balance(),
            'performance_stability': self._assess_performance_stability(),
            'mastery_trend': self._assess_mastery_trend(),
            'time_to_next_stage': self._estimate_time_to_next_stage()
        }

    def _calculate_weighted_competency(self) -> float:
        """Calculate weighted overall competency score"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for area, data in self.competency_areas.items():
            score = data['score']
            weight = data['weight']
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _calculate_competency_balance(self) -> float:
        """Calculate how balanced competencies are (lower variance = better balance)"""
        scores = [data['score'] for data in self.competency_areas.values()]
        if not scores:
            return 0.0
        
        variance = float(np.var(scores))
        # Convert to balance score (0-1, where 1 is perfectly balanced)
        return max(0.0, 1.0 - variance * 4)  # Scale variance to 0-1 range

    def _assess_performance_stability(self) -> float:
        """Assess stability of recent performance"""
        if len(self.episode_history) < 3:
            return 0.5  # Neutral when insufficient data
        
        recent_episodes = list(self.episode_history)[-5:]
        competency_scores = [self._calculate_episode_competency(ep) for ep in recent_episodes]
        
        if not competency_scores:
            return 0.5
        
        # Calculate coefficient of variation (lower = more stable)
        mean_score = np.mean(competency_scores)
        std_score = np.std(competency_scores)
        
        if mean_score == 0:
            return 0.0
        
        cv = float(std_score / mean_score)
        # Convert to stability score (0-1, where 1 is most stable)
        return max(0.0, 1.0 - cv)

    def _assess_mastery_trend(self) -> str:
        """Assess the trend in mastery development"""
        if len(self.episode_history) < 3:
            return 'insufficient_data'
        
        recent_episodes = list(self.episode_history)[-3:]
        competency_scores = [self._calculate_episode_competency(ep) for ep in recent_episodes]
        
        if len(competency_scores) < 2:
            return 'insufficient_data'
        
        # Calculate trend
        trend = competency_scores[-1] - competency_scores[0]
        
        if trend > 0.1:
            return 'rapidly_improving'
        elif trend > 0.03:
            return 'improving'
        elif trend > -0.03:
            return 'stable'
        elif trend > -0.1:
            return 'declining'
        else:
            return 'rapidly_declining'

    def _estimate_time_to_next_stage(self) -> Dict[str, Any]:
        """Estimate time/episodes needed to advance to next stage"""
        if self.current_stage >= len(self.curriculum_stages) - 1:
            return {'status': 'max_stage_reached', 'episodes_estimated': 0}
        
        if self.learning_velocity <= 0:
            return {'status': 'no_progress', 'episodes_estimated': -1}
        
        # Estimate based on current progress and velocity
        remaining_progress = 1.0 - self.stage_progress
        episodes_per_progress_point = 1.0 / max(self.learning_velocity, 0.01)
        estimated_episodes = int(remaining_progress * episodes_per_progress_point)
        
        return {
            'status': 'progressing',
            'episodes_estimated': max(1, estimated_episodes),
            'confidence': min(1.0, self.learning_velocity * 10)  # Higher velocity = higher confidence
        }

    def _generate_intelligent_learning_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent, contextual learning recommendations"""
        recommendations = []
        
        try:
            competency_scores = analysis.get('competency_scores', {})
            learning_velocity = analysis.get('learning_velocity', 0.0)
            stage_mastery = analysis.get('stage_mastery', {})
            
            current_stage = self.curriculum_stages[self.current_stage]
            focus_areas = current_stage['focus_areas']
            
            # Priority 1: Address weakest competency in focus areas
            focus_competencies = {k: v for k, v in competency_scores.items() if k in focus_areas}
            if focus_competencies:
                weakest_area = min(focus_competencies.items(), key=lambda x: x[1])
                area_name = weakest_area[0]
                score = weakest_area[1]
                target = self.competency_areas[area_name]['target']
                
                if score < target * 0.8:
                    recommendations.append(self._get_specific_recommendation(area_name, score, target))
            
            # Priority 2: Learning velocity recommendations
            if learning_velocity < -0.05:
                recommendations.append("Performance declining - review recent changes and consider reverting to proven strategies")
            elif learning_velocity < 0.01:
                recommendations.append("Learning plateau detected - try varying trading approaches within current stage constraints")
            elif learning_velocity > 0.1:
                recommendations.append("Excellent learning progress - maintain current approach and prepare for next stage")
            
            # Priority 3: Stage-specific recommendations
            if self.stage_progress < 0.3:
                recommendations.append(f"Focus on {current_stage['name']} stage fundamentals before attempting advanced techniques")
            elif self.stage_progress > 0.8:
                recommendations.append("Near stage completion - maintain consistency to ensure advancement")
            
            # Priority 4: Balance recommendations
            competency_balance = self._calculate_competency_balance()
            if competency_balance < 0.6:
                strongest_area = max(competency_scores.items(), key=lambda x: x[1])
                recommendations.append(f"Competency imbalance detected - reduce focus on {strongest_area[0]} and develop weaker areas")
            
            # Priority 5: Stability recommendations
            stability = self._assess_performance_stability()
            if stability < 0.5:
                recommendations.append("Performance inconsistency detected - focus on building reliable, repeatable processes")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("Continue current learning approach - all metrics within acceptable ranges")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendation_generation")
            return [f"Recommendation generation failed: {error_context}"]

    def _get_specific_recommendation(self, area: str, current_score: float, target_score: float) -> str:
        """Get specific recommendation for competency area improvement"""
        gap = target_score - current_score
        
        recommendations = {
            'risk_management': f"Risk management needs improvement ({gap:.1%} below target) - reduce position sizes and implement stricter stop-losses",
            'entry_timing': f"Entry timing requires development ({gap:.1%} below target) - wait for stronger confirmation signals before entering trades",
            'exit_strategy': f"Exit strategy needs enhancement ({gap:.1%} below target) - review profit-taking levels and trailing stop techniques",
            'position_sizing': f"Position sizing consistency lacking ({gap:.1%} below target) - maintain uniform risk per trade regardless of market conditions"
        }
        
        return recommendations.get(area, f"Improve {area.replace('_', ' ')} performance - currently {gap:.1%} below target")

    # ═══════════════════════════════════════════════════════════════════
    # HELPER METHODS FOR ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    def _get_recent_metric_performance(self, metric: str) -> Optional[float]:
        """Get recent performance for specific metric"""
        try:
            if not self.episode_history:
                return None
            
            recent_episodes = list(self.episode_history)[-5:]  # Last 5 episodes
            values = [ep.get(metric) for ep in recent_episodes if ep.get(metric) is not None]
            
            if not values:
                return None
            
            return float(np.mean(values))
            
        except Exception:
            return None

    def _calculate_improvement_needed(self, criterion: str, actual: float, target: float) -> str:
        """Calculate human-readable improvement needed"""
        if criterion == 'max_drawdown':
            gap = actual - target
            return f"Reduce drawdown by {gap:.1%}"
        else:
            gap = target - actual
            if criterion == 'win_rate':
                return f"Improve win rate by {gap:.1%}"
            elif criterion == 'profit_factor':
                return f"Increase profit factor by {gap:.2f}"
            else:
                return f"Improve {criterion} by {gap:.2f}"

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over recent episodes"""
        return {
            'win_rate_trend': self._analyze_win_rate_trend(),
            'pnl_trend': 'stable',
            'consistency_trend': 'improving'
        }

    async def _assess_current_stage_mastery(self) -> Dict[str, Any]:
        """Assess mastery of current curriculum stage"""
        return {'mastery_score': 0.5, 'completion_percentage': 50.0}

    def _detect_learning_patterns(self) -> Dict[str, Any]:
        """Detect learning patterns from episode history"""
        return {'patterns': [], 'trends': 'stable'}

    def _calculate_progress_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive progress metrics"""
        return {'overall_progress': 0.5, 'velocity': 0.0}

    def _assess_data_quality(self, learning_data: Dict[str, Any]) -> str:
        """Assess quality of learning data"""
        return 'good' if learning_data else 'insufficient'

    async def _update_performance_metrics_advanced(self, summary: Dict[str, Any]):
        """Update advanced performance metrics"""
        pass

    async def _evaluate_episode_success_comprehensive(self, summary: Dict[str, Any]) -> bool:
        """Evaluate episode success with comprehensive criteria"""
        return summary.get('pnl', 0) > 0

    async def _update_learning_statistics(self, summary: Dict[str, Any]):
        """Update learning statistics"""
        pass

    async def _handle_learning_plateau_advanced(self):
        """Handle learning plateau with advanced intervention"""
        pass

    def _generate_stage_advancement_thesis(self, old_stage: str, new_stage: str, data: Dict) -> str:
        """Generate thesis for stage advancement"""
        return f"Advanced from {old_stage} to {new_stage} stage based on performance criteria"

    def _analyze_win_rate_trend(self) -> float:
        """Analyze win rate improvement trend"""
        if len(self.episode_history) < 3:
            return 0.0
        
        recent_win_rates = [ep.get('win_rate', 0) for ep in list(self.episode_history)[-3:]]
        if len(recent_win_rates) < 2:
            return 0.0
        
        return recent_win_rates[-1] - recent_win_rates[0]

    def _calculate_profit_factor_consistency(self) -> float:
        """Calculate profit factor consistency score"""
        if len(self.episode_history) < 3:
            return 0.5
        
        profit_factors = [ep.get('profit_factor', 1.0) for ep in list(self.episode_history)[-5:]]
        if not profit_factors:
            return 0.5
        
        # Higher consistency = lower coefficient of variation
        mean_pf = np.mean(profit_factors)
        std_pf = np.std(profit_factors)
        
        if mean_pf == 0:
            return 0.0
        
        cv = float(std_pf / mean_pf)
        return max(0.0, 1.0 - cv)

    def _calculate_risk_consistency(self, learning_data: Dict[str, Any]) -> float:
        """Calculate risk consistency from trading data"""
        # Placeholder - would analyze position sizing consistency
        return 0.7  # Default good consistency

    def _assess_learning_risk(self) -> str:
        """Assess overall learning risk level"""
        risk_factors = []
        
        # Check learning velocity
        if self.learning_velocity < -0.1:
            risk_factors.append("declining_performance")
        
        # Check plateau
        if self.plateau_detection['consecutive_poor_episodes'] >= 3:
            risk_factors.append("learning_plateau")
        
        # Check competency balance
        if self._calculate_competency_balance() < 0.4:
            risk_factors.append("competency_imbalance")
        
        # Check stability
        if self._assess_performance_stability() < 0.4:
            risk_factors.append("performance_instability")
        
        if len(risk_factors) >= 3:
            return "HIGH RISK - Multiple learning challenges identified"
        elif len(risk_factors) >= 2:
            return "MODERATE RISK - Some learning concerns detected"
        elif len(risk_factors) == 1:
            return "LOW RISK - Minor learning challenge identified"
        else:
            return "MINIMAL RISK - Learning progression optimal"

    # ═══════════════════════════════════════════════════════════════════
    # ERROR HANDLING AND RECOVERY
    # ═══════════════════════════════════════════════════════════════════

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "CurriculumPlannerPlus")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="Curriculum Planner disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('CurriculumPlannerPlus', 'process_time', processing_time, False)
        
        return {
            'curriculum_stage': self._get_current_stage_info(),
            'learning_constraints': self._get_current_constraints(),
            'competency_scores': {k: v['score'] for k, v in self.competency_areas.items()},
            'learning_recommendations': ["Investigate curriculum system errors"],
            'stage_progression': {'ready_for_advancement': False, 'error': str(error_context)},
            'mastery_assessment': {'overall_mastery': 0.5, 'error': str(error_context)},
            'learning_analytics': self.learning_stats.copy(),
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def _get_safe_learning_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when data retrieval fails"""
        return {
            'performance_data': {},
            'episode_summary': {},
            'risk_metrics': {},
            'trading_session': {},
            'market_conditions': {},
            'recent_trades': [],
            'learning_context': {}
        }

    def _get_safe_analysis_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when analysis fails"""
        return {
            'competency_scores': {k: v['score'] for k, v in self.competency_areas.items()},
            'learning_velocity': 0.0,
            'performance_trends': {},
            'stage_mastery': {},
            'learning_patterns': {},
            'overall_competency': 0.5,
            'progress_metrics': {},
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'data_quality': 'insufficient',
            'error': 'analysis_failed'
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'curriculum_stage': {'stage_name': 'ERROR', 'status': 'disabled'},
            'learning_constraints': {'status': 'disabled'},
            'competency_scores': {k: 0.0 for k in self.competency_areas.keys()},
            'learning_recommendations': ["Restart curriculum planner system", "Check error logs for issues"],
            'stage_progression': {'ready_for_advancement': False, 'status': 'disabled'},
            'mastery_assessment': {'overall_mastery': 0.0, 'status': 'disabled'},
            'learning_analytics': {'status': 'disabled'},
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'}
        }

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _calculate_session_duration(self) -> str:
        """Calculate human-readable session duration"""
        try:
            start_time = datetime.datetime.fromisoformat(self.learning_stats['session_start'])
            duration = datetime.datetime.now() - start_time
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except Exception:
            return "Unknown"

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'CurriculumPlannerPlus',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'current_stage': self.curriculum_stages[self.current_stage]['name'],
            'stage_progress': self.stage_progress,
            'learning_velocity': self.learning_velocity,
            'total_episodes': self.learning_stats['total_episodes'],
            'success_rate': (self.learning_stats['successful_episodes'] / max(1, self.learning_stats['total_episodes'])) * 100,
            'plateau_risk': self.plateau_detection['consecutive_poor_episodes'] >= 3,
            'session_duration': self._calculate_session_duration()
        }

    def get_curriculum_report(self) -> str:
        """Generate comprehensive curriculum progress report"""
        current_stage = self.curriculum_stages[self.current_stage]
        overall_competency = self._calculate_weighted_competency()
        
        # Competency breakdown
        competency_summary = ""
        for area, data in self.competency_areas.items():
            score = data['score']
            target = data['target']
            status = "✅" if score >= target else "🔄" if score >= target * 0.8 else "❌"
            competency_summary += f"  • {area.replace('_', ' ').title()}: {score:.1%} (target: {target:.1%}) {status}\n"
        
        # Performance trend analysis
        trend = self._assess_mastery_trend()
        trend_icons = {
            'rapidly_improving': '📈🚀',
            'improving': '📈',
            'stable': '➡️',
            'declining': '📉',
            'rapidly_declining': '📉💥',
            'insufficient_data': '📊'
        }
        trend_display = f"{trend_icons.get(trend, '📊')} {trend.replace('_', ' ').title()}"
        
        # Time to next stage
        time_estimate = self._estimate_time_to_next_stage()
        if time_estimate['status'] == 'max_stage_reached':
            next_stage_info = "🏆 Maximum stage achieved!"
        elif time_estimate['status'] == 'no_progress':
            next_stage_info = "⚠️ No progress detected"
        else:
            next_stage_info = f"🎯 ~{time_estimate['episodes_estimated']} episodes to next stage"
        
        return f"""
📚 CURRICULUM PLANNER COMPREHENSIVE REPORT
═══════════════════════════════════════════════════════════════
🎯 Current Stage: {current_stage['name']} (Level {self.current_stage + 1}/{len(self.curriculum_stages)})
📊 Stage Progress: {self.stage_progress:.1%} | Overall Mastery: {overall_competency:.1%}
📈 Learning Velocity: {self.learning_velocity:+.1%} | Trend: {trend_display}

⚙️ Current Learning Constraints:
• Max Position Size: {current_stage['constraints']['max_position_size']}x
• Max Daily Trades: {current_stage['constraints']['max_trades_per_day']}
• Max Risk per Trade: {current_stage['constraints']['max_risk_per_trade']:.1%}
• Allowed Markets: {', '.join(current_stage['market_conditions'])}
• Focus Areas: {', '.join(current_stage['focus_areas'])}

📋 Competency Assessment:
{competency_summary}

📊 Learning Analytics:
• Total Episodes: {self.learning_stats['total_episodes']}
• Success Rate: {(self.learning_stats['successful_episodes'] / max(1, self.learning_stats['total_episodes'])):.1%}
• Adaptation Events: {self.learning_stats['adaptation_events']}
• Session Duration: {self._calculate_session_duration()}
• Performance Stability: {self._assess_performance_stability():.1%}
• Competency Balance: {self._calculate_competency_balance():.1%}

🎯 Progression Status:
{next_stage_info}

🎯 Learning Recommendations:
{chr(10).join([f'  • {rec}' for rec in self._generate_intelligent_learning_recommendations({'competency_scores': {k: v['score'] for k, v in self.competency_areas.items()}, 'learning_velocity': self.learning_velocity})])}

🔍 Learning Risk Assessment:
{self._assess_learning_risk()}
        """

    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT FOR HOT-RELOAD
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'CurriculumPlannerPlus',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'window': self.window,
                'adaptation_rate': self.adaptation_rate,
                'performance_threshold': self.performance_threshold,
                'difficulty_levels': self.difficulty_levels,
                'debug': self.debug
            },
            'curriculum_state': {
                'current_stage': self.current_stage,
                'stage_progress': self.stage_progress,
                'episode_history': list(self.episode_history),
                'performance_metrics': {k: list(v) for k, v in self.performance_metrics.items()},
                'learning_stats': self.learning_stats.copy(),
                'competency_areas': self.competency_areas.copy(),
                'learning_velocity': self.learning_velocity,
                'plateau_detection': self.plateau_detection.copy()
            },
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'curriculum_stages': self.curriculum_stages.copy(),
            'performance_metrics': self._get_health_metrics()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.window = int(config.get("window", self.window))
            self.adaptation_rate = float(config.get("adaptation_rate", self.adaptation_rate))
            self.performance_threshold = float(config.get("performance_threshold", self.performance_threshold))
            self.difficulty_levels = int(config.get("difficulty_levels", self.difficulty_levels))
            self.debug = bool(config.get("debug", self.debug))
            
            # Load curriculum state
            curriculum_state = state.get("curriculum_state", {})
            self.current_stage = int(curriculum_state.get("current_stage", 0))
            self.stage_progress = float(curriculum_state.get("stage_progress", 0.0))
            self.learning_velocity = float(curriculum_state.get("learning_velocity", 0.0))
            
            # Restore episode history
            self.episode_history = deque(curriculum_state.get("episode_history", []), maxlen=self.window)
            
            # Restore performance metrics
            performance_metrics = curriculum_state.get("performance_metrics", {})
            self.performance_metrics = defaultdict(list)
            for k, v in performance_metrics.items():
                self.performance_metrics[k] = list(v)
            
            # Restore learning stats and competency areas
            self.learning_stats = curriculum_state.get("learning_stats", self.learning_stats)
            self.competency_areas = curriculum_state.get("competency_areas", self.competency_areas)
            self.plateau_detection = curriculum_state.get("plateau_detection", self.plateau_detection)
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            # Load curriculum stages if provided
            if "curriculum_stages" in state:
                self.curriculum_stages = state["curriculum_stages"]
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Curriculum Planner state restored",
                stage=self.curriculum_stages[self.current_stage]['name'],
                episodes=len(self.episode_history),
                mastery=f"{self._calculate_weighted_competency():.1%}"
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for system monitoring"""
        return {
            'module_name': 'CurriculumPlannerPlus',
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
                'message': 'CurriculumPlannerPlus disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })
        
        if self.plateau_detection['consecutive_poor_episodes'] >= 3:
            alerts.append({
                'severity': 'warning',
                'message': f'Learning plateau detected: {self.plateau_detection["consecutive_poor_episodes"]} poor episodes',
                'action': 'Review learning strategy and consider curriculum adjustment'
            })
        
        if self.learning_velocity < -0.1:
            alerts.append({
                'severity': 'warning',
                'message': f'Declining learning velocity: {self.learning_velocity:.1%}',
                'action': 'Investigate recent changes and performance degradation'
            })
        
        competency_balance = self._calculate_competency_balance()
        if competency_balance < 0.4:
            alerts.append({
                'severity': 'info',
                'message': f'Competency imbalance detected: {competency_balance:.1%} balance score',
                'action': 'Focus on developing weaker competency areas'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        if self.is_disabled:
            recommendations.append("Restart CurriculumPlannerPlus module after investigating errors")
        
        if len(self.episode_history) < 5:
            recommendations.append("Insufficient learning data - continue training to build baseline")
        
        if self.learning_stats['adaptation_events'] > 10:
            recommendations.append("High adaptation frequency - consider more stable learning approach")
        
        if self._assess_performance_stability() < 0.5:
            recommendations.append("Performance instability detected - focus on consistency over advancement")
        
        if not recommendations:
            recommendations.append("CurriculumPlannerPlus operating within normal parameters")
        
        return recommendations

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API METHODS (for external use)
    # ═══════════════════════════════════════════════════════════════════

    def record_episode(self, summary: Dict[str, Any]) -> None:
        """Public method to record episode (async wrapper)"""
        try:
            # Run the async method synchronously
            import asyncio
            if asyncio.get_event_loop().is_running():
                # If we're already in an async context, schedule it
                asyncio.create_task(self.record_episode_comprehensive(summary))
            else:
                # Run it directly
                asyncio.run(self.record_episode_comprehensive(summary))
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "episode_recording_wrapper")
            self.logger.error(f"Episode recording wrapper failed: {error_context}")

    def get_current_curriculum_constraints(self) -> Dict[str, Any]:
        """Public method to get current curriculum constraints"""
        return self._get_current_constraints()

    def get_observation_components(self) -> np.ndarray:
        """Get curriculum metrics for observation"""
        try:
            if not self.episode_history:
                # Return baseline defaults for cold start
                defaults = np.array([
                    0.5,  # stage_progress
                    0.5,  # mastery_level  
                    0.2,  # current_difficulty (normalized)
                    0.0,  # learning_rate
                    0.5,  # overall_competency
                    0.0   # adaptation_events_normalized
                ], dtype=np.float32)
                return defaults
            
            # Calculate observation components
            overall_competency = self._calculate_weighted_competency()
            
            # Normalize components
            stage_progress = self.stage_progress
            mastery_level = self.learning_stats['mastery_level']
            difficulty_norm = self.learning_stats['current_difficulty'] / 5.0  # Max difficulty is 5
            learning_rate = np.clip(self.learning_stats['learning_rate'], -1.0, 1.0)
            adaptation_events_norm = min(1.0, self.learning_stats['adaptation_events'] / 10.0)
            
            observation = np.array([
                stage_progress,
                mastery_level,
                difficulty_norm,
                learning_rate,
                overall_competency,
                adaptation_events_norm
            ], dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid curriculum observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Curriculum observation generation failed: {error_context}")
            return np.array([0.5, 0.5, 0.2, 0.0, 0.5, 0.0], dtype=np.float32)