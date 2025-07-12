"""
🎤 Enhanced Explanation Generator with SmartInfoBus Integration v3.0
Advanced intelligent explanation system for trading decisions and system state with contextual adaptation
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
    name="ExplanationGenerator",
    version="3.0.0",
    category="interface",
    provides=[
        "trading_explanations", "system_explanations", "performance_insights",
        "contextual_narratives", "operator_updates", "decision_rationales"
    ],
    requires=[
        "recent_trades", "positions", "risk_data", "system_alerts",
        "market_context", "session_metrics", "module_insights"
    ],
    description="Advanced intelligent explanation system for trading decisions and system state with contextual adaptation",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=100,
    priority=9,
    explainable=True,
    hot_reload=True
)
class ExplanationGenerator(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🎤 PRODUCTION-GRADE Explanation Generator v3.0
    
    Advanced intelligent explanation system with:
    - Real-time contextual explanation generation
    - Multi-depth explanation capabilities (brief, detailed, comprehensive)
    - Adaptive narrative generation based on market conditions and performance
    - SmartInfoBus zero-wiring architecture
    - Comprehensive thesis generation for all explanations
    """

    def _initialize(self):
        """Initialize advanced explanation generation systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced explanation configuration
        self.explanation_depth = self.config.get('explanation_depth', 'detailed')
        self.update_frequency = self.config.get('update_frequency', 1)
        self.target_daily_profit = self.config.get('target_daily_profit', 150.0)
        self.debug = self.config.get('debug', False)
        
        # Core explanation state
        self.last_explanation = ""
        self.explanation_history = deque(maxlen=100)
        self.explanation_templates = self._initialize_advanced_explanation_templates()
        self.context_priorities = self._initialize_intelligent_context_priorities()
        
        # Enhanced session metrics
        self.session_metrics = {
            'trade_count': 0,
            'profit_today': 0.0,
            'session_start': datetime.datetime.now(),
            'last_update': datetime.datetime.now(),
            'explanations_generated': 0,
            'avg_explanation_relevance': 0.0,
            'explanation_quality_score': 0.0,
            'context_accuracy': 0.0
        }
        
        # Advanced decision context tracking
        self.decision_context = {
            'current_strategy': 'unknown',
            'market_regime': 'unknown',
            'risk_level': 'medium',
            'confidence_level': 0.5,
            'recent_performance': 'neutral',
            'system_alerts': [],
            'learning_stage': 'unknown',
            'bias_warnings': [],
            'market_sentiment': 'neutral'
        }
        
        # Enhanced explanation categories with intelligence
        self.explanation_categories = {
            'trade_decision': {
                'description': 'Trade entry/exit explanations with rationale',
                'priority': 9,
                'contexts': ['entry', 'exit', 'modification', 'cancel']
            },
            'risk_management': {
                'description': 'Risk and position sizing explanations',
                'priority': 8,
                'contexts': ['alert', 'adjustment', 'protection', 'recovery']
            },
            'market_analysis': {
                'description': 'Market condition and regime explanations',
                'priority': 6,
                'contexts': ['regime_change', 'volatility_shift', 'sentiment_change']
            },
            'performance_update': {
                'description': 'Performance and progress explanations',
                'priority': 5,
                'contexts': ['target_progress', 'milestone', 'benchmark']
            },
            'system_status': {
                'description': 'System health and alert explanations',
                'priority': 7,
                'contexts': ['health_check', 'error_recovery', 'optimization']
            },
            'learning_insights': {
                'description': 'Learning and adaptation explanations',
                'priority': 4,
                'contexts': ['stage_progression', 'competency_update', 'plateau']
            }
        }
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Advanced narrative intelligence
        self.narrative_intelligence = {
            'emotional_tone': 'neutral',
            'urgency_level': 'normal',
            'technical_depth': 'moderate',
            'operator_experience_level': 'intermediate'
        }
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🎤",
            message=f"Explanation Generator v{version} initialized",
            depth=self.explanation_depth,
            categories=len(self.explanation_categories),
            target_profit=f"€{self.target_daily_profit}"
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="ExplanationGenerator",
            log_path="logs/interface/explanation_generator.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("ExplanationGenerator", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

    def _initialize_advanced_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize advanced explanation templates with enhanced intelligence"""
        return {
            'trade_entry': {
                'brief': "📈 {instrument} {direction} entry - {reason} (Size: {size})",
                'detailed': "🎯 Trade Entry: {instrument} {direction} | Rationale: {reason} | Size: {size} | Risk: {risk_level} | Confidence: {confidence}",
                'comprehensive': """🎯 COMPREHENSIVE TRADE ENTRY ANALYSIS
• Instrument: {instrument} ({market_cap} market)
• Direction: {direction} ({strategy_type} strategy)
• Entry Rationale: {reason}
• Position Size: {size} (Risk: {risk_level})
• Market Context: {market_context}
• Technical Setup: {technical_setup}
• Risk/Reward: {risk_reward_ratio}
• Exit Strategy: {exit_strategy}
• Confidence Level: {confidence}
• Expected Outcome: {expected_outcome}
• Timeline: {expected_duration}"""
            },
            'trade_exit': {
                'brief': "📊 {instrument} closed - {exit_reason} | P&L: {pnl}",
                'detailed': "🏁 Trade Exit: {instrument} | Reason: {exit_reason} | P&L: {pnl} | Duration: {duration} | Performance: {performance_rating}",
                'comprehensive': """🏁 COMPREHENSIVE TRADE EXIT ANALYSIS
• Instrument: {instrument}
• Exit Trigger: {exit_reason}
• Final P&L: {pnl} ({pnl_percentage})
• Trade Duration: {duration}
• Performance vs Expectation: {vs_expectation}
• Target Achievement: {target_achievement}
• Risk Management: {risk_management_effectiveness}
• Market Conditions During Trade: {market_evolution}
• Key Learning Points: {lessons_learned}
• Strategy Validation: {strategy_validation}
• Impact on Session: {session_impact}"""
            },
            'performance_update': {
                'brief': "💰 Session: {profit_pct}% to target | {trade_count} trades | {performance_trend}",
                'detailed': "📊 Progress Update: €{profit_today}/€{target} ({profit_pct}%) | Trades: {trade_count} | Win Rate: {win_rate}% | Trend: {performance_trend}",
                'comprehensive': """📈 COMPREHENSIVE SESSION PERFORMANCE
• Current Profit: €{profit_today} ({profit_pct}% of €{target} target)
• Trade Statistics: {trade_count} trades, {win_rate}% win rate
• Average Trade: €{avg_trade} | Best: €{best_trade} | Worst: €{worst_trade}
• Performance Trend: {performance_trend} ({velocity} velocity)
• Risk Metrics: {max_drawdown}% max drawdown, {current_exposure} exposure
• Time Analysis: {session_duration} elapsed, {time_remaining} remaining
• Efficiency Rating: {efficiency_rating}/10
• Target Probability: {target_probability}% chance by EOD
• Strategy Performance: {strategy_breakdown}
• Market Adaptation: {market_adaptation_score}
• Next Actions: {recommended_actions}"""
            },
            'risk_alert': {
                'brief': "⚠️ Risk Alert: {alert_type} - {severity}",
                'detailed': "🚨 Risk Management: {alert_type} | Severity: {severity} | Exposure: {exposure} | Action: {immediate_action}",
                'comprehensive': """🚨 COMPREHENSIVE RISK MANAGEMENT ALERT
• Alert Classification: {alert_type} (Severity: {severity})
• Risk Metrics: {risk_metrics}
• Current Exposure: {exposure} ({exposure_percentage}% of capital)
• Triggering Factors: {trigger_factors}
• Immediate Actions Required: {immediate_actions}
• Recovery Timeline: {recovery_timeline}
• Preventive Measures: {preventive_measures}
• Historical Context: {historical_context}
• System Responses: {automated_responses}
• Manual Interventions: {manual_interventions}
• Monitoring Plan: {monitoring_plan}"""
            },
            'market_insight': {
                'brief': "🌊 Market: {regime} regime | Vol: {volatility} | Opportunity: {opportunity_level}",
                'detailed': "🌍 Market Analysis: {regime} regime | Volatility: {volatility} | Drivers: {key_drivers} | Opportunities: {opportunities}",
                'comprehensive': """🌍 COMPREHENSIVE MARKET ANALYSIS
• Current Regime: {regime} ({regime_confidence}% confidence)
• Volatility Profile: {volatility} ({volatility_percentile}th percentile)
• Primary Market Drivers: {primary_drivers}
• Sentiment Indicators: {sentiment_analysis}
• Technical Environment: {technical_environment}
• Trading Opportunities: {opportunity_analysis}
• Risk Factors: {risk_factors}
• Correlation Analysis: {correlation_insights}
• Strategy Recommendations: {strategy_recommendations}
• Position Sizing Guidance: {sizing_guidance}
• Time Horizon Considerations: {time_horizon_analysis}"""
            },
            'learning_update': {
                'brief': "📚 Learning: {stage} stage | Progress: {progress}% | Focus: {focus_area}",
                'detailed': "🎓 Learning Progress: {stage} stage ({progress}%) | Competency: {competency_level} | Focus: {focus_areas}",
                'comprehensive': """🎓 COMPREHENSIVE LEARNING ASSESSMENT
• Current Stage: {stage} ({progress}% completion)
• Overall Competency: {overall_competency}/10
• Competency Breakdown: {competency_breakdown}
• Learning Velocity: {learning_velocity}
• Mastery Areas: {mastery_areas}
• Development Areas: {development_areas}
• Recent Improvements: {recent_improvements}
• Challenge Areas: {challenge_areas}
• Recommended Focus: {focus_recommendations}
• Next Milestones: {upcoming_milestones}
• Estimated Timeline: {progression_timeline}"""
            },
            'system_status': {
                'brief': "🔧 System: {status} | Health: {health_score}/10 | Alerts: {alert_count}",
                'detailed': "⚙️ System Health: {status} | Score: {health_score}/10 | Performance: {performance_metrics} | Alerts: {alert_summary}",
                'comprehensive': """⚙️ COMPREHENSIVE SYSTEM STATUS
• Overall Health: {status} (Score: {health_score}/10)
• Component Status: {component_status}
• Performance Metrics: {performance_metrics}
• Resource Utilization: {resource_utilization}
• Active Alerts: {active_alerts}
• Recent Optimizations: {recent_optimizations}
• Stability Indicators: {stability_indicators}
• Error Analysis: {error_analysis}
• Maintenance Status: {maintenance_status}
• Upgrade Recommendations: {upgrade_recommendations}
• Monitoring Summary: {monitoring_summary}"""
            }
        }

    def _initialize_intelligent_context_priorities(self) -> Dict[str, int]:
        """Initialize intelligent context priority levels with dynamic weighting"""
        return {
            'system_critical_error': 10,
            'high_profit_trade': 9,
            'high_loss_trade': 9,
            'risk_limit_breach': 8,
            'target_achieved': 8,
            'drawdown_alert': 7,
            'regime_change': 6,
            'learning_milestone': 5,
            'bias_warning': 5,
            'performance_milestone': 4,
            'routine_update': 3,
            'market_update': 2,
            'background_info': 1
        }

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Explanation Generator v3.0 Initialization Complete:
        
        Advanced Explanation System:
        - Multi-depth explanations: {', '.join(['brief', 'detailed', 'comprehensive'])}
        - Contextual adaptation based on priority levels and market conditions
        - Intelligent narrative generation with emotional tone adjustment
        - Real-time explanation quality assessment and optimization
        
        Current Configuration:
        - Explanation depth: {self.explanation_depth}
        - Update frequency: every {self.update_frequency} minute(s)
        - Target profit context: €{self.target_daily_profit} daily
        - Categories tracked: {len(self.explanation_categories)} explanation types
        
        Narrative Intelligence Features:
        - Adaptive emotional tone based on performance and market conditions
        - Urgency level adjustment for critical situations
        - Technical depth scaling based on operator experience
        - Context-aware priority system with {len(self.context_priorities)} priority levels
        
        Advanced Capabilities:
        - Real-time context extraction from all system modules
        - Intelligent explanation enhancement with module insights
        - Comprehensive thesis generation for all explanations
        - Performance tracking and explanation quality optimization
        
        Expected Outcomes:
        - Clear, actionable explanations of all trading decisions
        - Enhanced operator understanding of system behavior
        - Improved decision-making through contextual insights
        - Transparent system operation with comprehensive explanations
        """
        
        self.smart_bus.set('explanation_generator_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'capabilities': {
                'depths': ['brief', 'detailed', 'comprehensive'],
                'categories': list(self.explanation_categories.keys()),
                'priority_levels': len(self.context_priorities)
            }
        }, module='ExplanationGenerator', thesis=thesis)

    async def process(self) -> Dict[str, Any]:
        """
        Modern async processing with intelligent explanation generation
        
        Returns:
            Dict containing explanations, insights, and system narratives
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive context from SmartInfoBus
            explanation_context = await self._get_comprehensive_explanation_context()
            
            # Core explanation analysis with error handling
            explanation_analysis = await self._analyze_explanation_requirements(explanation_context)
            
            # Generate intelligent explanations
            explanations = await self._generate_intelligent_explanations(explanation_analysis)
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_explanation_thesis(explanations, explanation_analysis)
            
            # Create comprehensive results
            results = {
                'trading_explanations': explanations.get('trading', []),
                'system_explanations': explanations.get('system', []),
                'performance_insights': explanations.get('performance', {}),
                'contextual_narratives': explanations.get('narratives', []),
                'operator_updates': explanations.get('updates', []),
                'decision_rationales': explanations.get('rationales', []),
                'explanation_metrics': self.session_metrics.copy(),
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('ExplanationGenerator', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_explanation_context(self) -> Dict[str, Any]:
        """Get comprehensive context for explanation generation using modern SmartInfoBus"""
        try:
            return {
                'recent_trades': self.smart_bus.get('recent_trades', 'ExplanationGenerator') or [],
                'positions': self.smart_bus.get('positions', 'ExplanationGenerator') or [],
                'risk_data': self.smart_bus.get('risk_data', 'ExplanationGenerator') or {},
                'system_alerts': self.smart_bus.get('system_alerts', 'ExplanationGenerator') or [],
                'market_context': self.smart_bus.get('market_context', 'ExplanationGenerator') or {},
                'session_metrics': self.smart_bus.get('session_metrics', 'ExplanationGenerator') or {},
                'module_insights': self.smart_bus.get('module_insights', 'ExplanationGenerator') or {},
                'performance_data': self.smart_bus.get('performance_data', 'ExplanationGenerator') or {},
                'learning_status': self.smart_bus.get('learning_status', 'ExplanationGenerator') or {},
                'bias_analysis': self.smart_bus.get('bias_analysis', 'ExplanationGenerator') or {},
                'strategy_status': self.smart_bus.get('strategy_status', 'ExplanationGenerator') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "ExplanationGenerator")
            self.logger.warning(f"Context retrieval incomplete: {error_context}")
            return self._get_safe_context_defaults()

    async def _analyze_explanation_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what explanations are needed based on current context"""
        try:
            analysis = {
                'priority_contexts': [],
                'explanation_triggers': [],
                'context_changes': [],
                'narrative_requirements': {},
                'urgency_level': 'normal',
                'emotional_tone': 'neutral',
                'technical_depth': 'moderate'
            }
            
            # Analyze priority contexts
            priority_contexts = await self._identify_priority_contexts(context)
            analysis['priority_contexts'] = priority_contexts
            
            # Determine explanation triggers
            triggers = await self._identify_explanation_triggers(context)
            analysis['explanation_triggers'] = triggers
            
            # Detect context changes
            changes = self._detect_context_changes(context)
            analysis['context_changes'] = changes
            
            # Assess narrative requirements
            narrative_reqs = await self._assess_narrative_requirements(context, priority_contexts)
            analysis['narrative_requirements'] = narrative_reqs
            
            # Update decision context
            await self._update_decision_context_advanced(context)
            
            return analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "ExplanationGenerator")
            self.logger.error(f"Explanation analysis failed: {error_context}")
            return self._get_safe_analysis_defaults()

    async def _identify_priority_contexts(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify high-priority contexts requiring explanation"""
        priority_contexts = []
        
        try:
            # Check for critical system alerts
            alerts = context.get('system_alerts', [])
            critical_alerts = [a for a in alerts if a.get('severity') in ['critical', 'error']]
            if critical_alerts:
                priority_contexts.append({
                    'type': 'system_critical_error',
                    'priority': 10,
                    'data': critical_alerts,
                    'explanation_required': True
                })
            
            # Check for significant trades
            recent_trades = context.get('recent_trades', [])
            if recent_trades:
                last_trade = recent_trades[-1]
                pnl = last_trade.get('pnl', 0)
                
                if pnl > 100:  # High profit trade
                    priority_contexts.append({
                        'type': 'high_profit_trade',
                        'priority': 9,
                        'data': last_trade,
                        'explanation_required': True
                    })
                elif pnl < -50:  # High loss trade
                    priority_contexts.append({
                        'type': 'high_loss_trade',
                        'priority': 9,
                        'data': last_trade,
                        'explanation_required': True
                    })
            
            # Check for risk situations
            risk_data = context.get('risk_data', {})
            drawdown = risk_data.get('current_drawdown', 0)
            if drawdown > 0.1:  # 10% drawdown
                priority_contexts.append({
                    'type': 'drawdown_alert',
                    'priority': 7,
                    'data': {'drawdown': drawdown, 'risk_metrics': risk_data},
                    'explanation_required': True
                })
            
            # Check for target achievement
            session_pnl = sum(t.get('pnl', 0) for t in recent_trades)
            if session_pnl >= self.target_daily_profit:
                priority_contexts.append({
                    'type': 'target_achieved',
                    'priority': 8,
                    'data': {'profit': session_pnl, 'target': self.target_daily_profit},
                    'explanation_required': True
                })
            
            # Check for learning milestones
            learning_status = context.get('learning_status', {})
            if learning_status.get('stage_progression', {}).get('ready_for_advancement', False):
                priority_contexts.append({
                    'type': 'learning_milestone',
                    'priority': 5,
                    'data': learning_status,
                    'explanation_required': True
                })
            
            # Sort by priority
            priority_contexts.sort(key=lambda x: x['priority'], reverse=True)
            
            return priority_contexts
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "priority_identification")
            return []

    async def _identify_explanation_triggers(self, context: Dict[str, Any]) -> List[str]:
        """Identify triggers that require explanation generation"""
        triggers = []
        
        try:
            # Time-based trigger
            time_since_last = datetime.datetime.now() - self.session_metrics['last_update']
            if time_since_last.total_seconds() >= (self.update_frequency * 60):
                triggers.append('time_based_update')
            
            # Trade count change
            current_trade_count = len(context.get('recent_trades', []))
            if current_trade_count != self.session_metrics['trade_count']:
                triggers.append('trade_count_change')
            
            # Alert generation
            if context.get('system_alerts'):
                triggers.append('new_alerts')
            
            # Market regime change
            current_regime = context.get('market_context', {}).get('regime', 'unknown')
            if current_regime != self.decision_context.get('market_regime', 'unknown'):
                triggers.append('regime_change')
            
            # Performance milestone
            session_pnl = sum(t.get('pnl', 0) for t in context.get('recent_trades', []))
            profit_percentage = (session_pnl / self.target_daily_profit) * 100
            
            # Check for percentage milestones (25%, 50%, 75%, 100%, 125%)
            milestones = [25, 50, 75, 100, 125]
            for milestone in milestones:
                if (profit_percentage >= milestone and 
                    not hasattr(self, f'milestone_{milestone}_reached')):
                    triggers.append(f'milestone_{milestone}')
                    setattr(self, f'milestone_{milestone}_reached', True)
            
            return triggers
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "trigger_identification")
            return []

    async def _generate_intelligent_explanations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent explanations based on analysis"""
        try:
            explanations = {
                'trading': [],
                'system': [],
                'performance': {},
                'narratives': [],
                'updates': [],
                'rationales': []
            }
            
            priority_contexts = analysis.get('priority_contexts', [])
            triggers = analysis.get('explanation_triggers', [])
            narrative_reqs = analysis.get('narrative_requirements', {})
            
            # Generate explanations for priority contexts
            for context in priority_contexts:
                explanation = await self._generate_context_specific_explanation(context)
                if explanation:
                    category = self._categorize_explanation(context['type'])
                    explanations[category].append(explanation)
            
            # Generate trigger-based explanations
            for trigger in triggers:
                explanation = await self._generate_trigger_explanation(trigger)
                if explanation:
                    explanations['updates'].append(explanation)
            
            # Generate performance insights
            if 'performance_analysis' in narrative_reqs:
                performance_insights = await self._generate_performance_insights()
                explanations['performance'] = performance_insights
            
            # Generate contextual narratives
            if narrative_reqs.get('narrative_depth', 'none') != 'none':
                narratives = await self._generate_contextual_narratives(narrative_reqs)
                explanations['narratives'] = narratives
            
            # Generate decision rationales
            rationales = await self._generate_decision_rationales(analysis)
            explanations['rationales'] = rationales
            
            # Update explanation metrics
            self._update_explanation_metrics(explanations)
            
            return explanations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "explanation_generation")
            self.logger.error(f"Explanation generation failed: {error_context}")
            return self._get_safe_explanation_defaults()

    async def _generate_context_specific_explanation(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate explanation for specific context"""
        try:
            context_type = context['type']
            context_data = context['data']
            
            if context_type == 'high_profit_trade':
                return await self._generate_profitable_trade_explanation(context_data)
            elif context_type == 'high_loss_trade':
                return await self._generate_loss_trade_explanation(context_data)
            elif context_type == 'target_achieved':
                return await self._generate_target_achievement_explanation(context_data)
            elif context_type == 'system_critical_error':
                return await self._generate_critical_error_explanation(context_data)
            elif context_type == 'drawdown_alert':
                return await self._generate_drawdown_explanation(context_data)
            elif context_type == 'learning_milestone':
                return await self._generate_learning_milestone_explanation(context_data)
            else:
                return await self._generate_generic_explanation(context_type, context_data)
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "context_explanation")
            return f"Explanation generation error for {context.get('type', 'unknown')}: {error_context}"

    async def _assess_narrative_requirements(self, context: Dict[str, Any], 
                                           priority_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess narrative requirements based on context and priorities"""
        try:
            requirements = {
                'narrative_depth': 'none',
                'emotional_tone': 'neutral',
                'urgency_level': 'normal',
                'technical_detail': 'moderate'
            }
            
            # Assess based on priority contexts
            if priority_contexts:
                highest_priority = priority_contexts[0]['priority']
                
                if highest_priority >= 8:  # Critical situations
                    requirements.update({
                        'narrative_depth': 'comprehensive',
                        'emotional_tone': 'urgent',
                        'urgency_level': 'high',
                        'technical_detail': 'detailed'
                    })
                elif highest_priority >= 6:  # Important situations
                    requirements.update({
                        'narrative_depth': 'detailed',
                        'emotional_tone': 'focused',
                        'urgency_level': 'elevated',
                        'technical_detail': 'moderate'
                    })
                elif highest_priority >= 4:  # Routine situations
                    requirements.update({
                        'narrative_depth': 'brief',
                        'emotional_tone': 'informative',
                        'urgency_level': 'normal',
                        'technical_detail': 'basic'
                    })
            
            # Adjust based on performance
            performance = self.decision_context.get('recent_performance', 'neutral')
            if performance in ['excellent', 'good']:
                requirements['emotional_tone'] = 'positive'
            elif performance == 'poor':
                requirements['emotional_tone'] = 'concerned'
            
            return requirements
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "narrative_assessment")
            return {'narrative_depth': 'basic', 'emotional_tone': 'neutral'}

    async def _generate_trigger_explanation(self, trigger: str) -> Optional[str]:
        """Generate explanation for specific trigger"""
        try:
            trigger_explanations = {
                'time_based_update': f"⏰ Scheduled update: Current session progress and system status",
                'trade_count_change': f"📊 Trade activity update: New trading activity detected",
                'new_alerts': f"🚨 System alerts: New notifications require attention",
                'regime_change': f"🌊 Market regime change: Trading conditions have shifted",
                'milestone_25': f"🎯 25% milestone reached: Quarter way to daily target",
                'milestone_50': f"🎯 50% milestone reached: Halfway to daily target",
                'milestone_75': f"🎯 75% milestone reached: Three-quarters to daily target",
                'milestone_100': f"🎯 100% milestone reached: Daily target achieved!",
                'milestone_125': f"🎯 125% milestone reached: Exceeded daily target!",
                'error_recovery': f"🔧 Error recovery: System recovering from previous issues"
            }
            
            return trigger_explanations.get(trigger, f"System update: {trigger.replace('_', ' ')}")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "trigger_explanation")
            return f"Update triggered: {trigger}"

    async def _generate_contextual_narratives(self, narrative_reqs: Dict[str, Any]) -> List[str]:
        """Generate contextual narratives based on requirements"""
        try:
            narratives = []
            depth = narrative_reqs.get('narrative_depth', 'none')
            
            if depth == 'none':
                return narratives
            
            # Market narrative
            market_regime = self.decision_context.get('market_regime', 'unknown')
            if market_regime != 'unknown':
                if depth == 'comprehensive':
                    narratives.append(f"🌍 Market Environment: Currently operating in {market_regime} regime with adaptive strategy selection and risk management protocols active.")
                else:
                    narratives.append(f"🌊 Market: {market_regime.title()} conditions")
            
            # Performance narrative
            performance = self.decision_context.get('recent_performance', 'neutral')
            if depth == 'comprehensive':
                narratives.append(f"📈 Performance Status: Recent trading performance is {performance}, with system continuously adapting to market conditions and optimizing execution quality.")
            elif depth == 'detailed':
                narratives.append(f"📊 Performance: {performance.title()} recent results")
            
            # Risk narrative
            risk_level = self.decision_context.get('risk_level', 'medium')
            if risk_level in ['high', 'critical']:
                if depth == 'comprehensive':
                    narratives.append(f"⚠️ Risk Management: Current risk level is {risk_level}, with enhanced monitoring and protective measures actively engaged.")
                else:
                    narratives.append(f"⚠️ Risk: {risk_level.title()} level - monitoring closely")
            
            return narratives
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "narrative_generation")
            return [f"Narrative generation error: {error_context}"]

    async def _generate_decision_rationales(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate decision rationales based on analysis"""
        try:
            rationales = []
            
            # Priority-based rationales
            priority_contexts = analysis.get('priority_contexts', [])
            for context in priority_contexts[:3]:  # Top 3 priorities
                context_type = context['type']
                priority = context['priority']
                
                if context_type == 'high_profit_trade':
                    rationales.append(f"✅ Trade Success Rationale: High-profit trade validates current strategy effectiveness and market timing accuracy.")
                elif context_type == 'high_loss_trade':
                    rationales.append(f"⚠️ Loss Analysis Rationale: Significant loss triggers risk management review and strategy adjustment protocols.")
                elif context_type == 'target_achieved':
                    rationales.append(f"🎯 Target Achievement Rationale: Daily target reached through systematic execution and risk-controlled trading approach.")
                elif context_type == 'system_critical_error':
                    rationales.append(f"🚨 Critical Response Rationale: System error requires immediate attention to maintain trading operation integrity.")
                elif context_type == 'drawdown_alert':
                    rationales.append(f"🛡️ Risk Protection Rationale: Drawdown threshold breach activates protective measures to preserve capital.")
            
            # Context change rationales
            context_changes = analysis.get('context_changes', [])
            for change in context_changes:
                if change == 'market_regime_change':
                    rationales.append(f"🔄 Adaptation Rationale: Market regime shift requires strategy recalibration for optimal performance.")
                elif change == 'risk_level_change':
                    rationales.append(f"⚖️ Risk Adjustment Rationale: Risk level change triggers position sizing and exposure adjustments.")
            
            # Default rationale if none specific
            if not rationales:
                rationales.append(f"📊 Routine Rationale: Standard monitoring and status updates maintain operational transparency.")
            
            return rationales
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "rationale_generation")
            return [f"Rationale generation error: {error_context}"]

    async def _generate_loss_trade_explanation(self, trade_data: Dict[str, Any]) -> str:
        """Generate explanation for loss trade"""
        try:
            template = self.explanation_templates['trade_exit'][self.explanation_depth]
            
            # Extract trade details
            instrument = trade_data.get('symbol', 'Unknown')
            pnl = trade_data.get('pnl', 0)
            duration = trade_data.get('duration', 0)
            size = trade_data.get('size', 0)
            
            # Calculate impact metrics
            target_impact = abs(pnl / self.target_daily_profit) * 100
            
            if self.explanation_depth == 'comprehensive':
                return template.format(
                    instrument=instrument,
                    exit_reason="Stop loss activated for capital protection",
                    pnl=f"€{pnl:+.2f}",
                    pnl_percentage=f"{target_impact:.1f}% of daily target",
                    duration=f"{duration} steps",
                    vs_expectation="Risk management protocol executed as designed",
                    target_achievement=f"Protected remaining capital ({100-target_impact:.1f}% preserved)",
                    risk_management_effectiveness="Effective - prevented larger losses through systematic stop loss",
                    market_evolution=f"Market conditions became unfavorable during {duration} step position",
                    lessons_learned="Validate entry criteria and market timing for future improvements",
                    strategy_validation="Risk management working correctly - review entry signals",
                    session_impact=f"Controlled loss - capital preserved for future opportunities"
                )
            elif self.explanation_depth == 'detailed':
                return template.format(
                    instrument=instrument,
                    exit_reason="Stop loss activated",
                    pnl=f"€{pnl:+.2f}",
                    duration=f"{duration} steps",
                    performance_rating="Risk managed"
                )
            else:  # brief
                return template.format(
                    instrument=instrument,
                    exit_reason="Stop loss",
                    pnl=f"€{pnl:+.2f}"
                )
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "loss_trade_explanation")
            return f"Loss trade explanation error: {error_context}"

    async def _generate_target_achievement_explanation(self, target_data: Dict[str, Any]) -> str:
        """Generate explanation for target achievement"""
        try:
            profit = target_data.get('profit', 0)
            target = target_data.get('target', self.target_daily_profit)
            
            session_duration = datetime.datetime.now() - self.session_metrics['session_start']
            hours_taken = session_duration.total_seconds() / 3600
            over_target = ((profit / target) - 1) * 100
            
            if self.explanation_depth == 'comprehensive':
                return f"""🎯 COMPREHENSIVE TARGET ACHIEVEMENT ANALYSIS
• Achievement Status: Daily profit target of €{target} successfully reached
• Final Profit: €{profit:.2f} ({over_target:+.1f}% over target)
• Time to Target: {hours_taken:.1f} hours of {8.0} hour session
• Efficiency Rating: {self._calculate_efficiency_rating(profit, hours_taken):.1f}/10
• Trade Execution: {self.session_metrics['trade_count']} trades with systematic approach
• Risk Management: Capital preserved while achieving profitable outcome
• Strategy Validation: Current approach proven effective in market conditions
• Session Continuation: Consider reducing position sizes to protect gains
• Performance Impact: Excellent execution demonstrates system effectiveness
• Next Phase: Focus on capital preservation for remainder of session"""
            elif self.explanation_depth == 'detailed':
                return f"🎯 Target Achieved: €{profit:.2f} ({over_target:+.1f}% over €{target} target) in {hours_taken:.1f}h with {self.session_metrics['trade_count']} trades"
            else:  # brief
                return f"🎯 Target Achieved: €{profit:.2f} ({over_target:+.1f}% over target)"
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "target_achievement_explanation")
            return f"Target achievement explanation error: {error_context}"

    async def _generate_critical_error_explanation(self, error_data: List[Dict[str, Any]]) -> str:
        """Generate explanation for critical system errors"""
        try:
            if not error_data:
                return "🚨 Critical system alert - no details available"
            
            error_count = len(error_data)
            latest_error = error_data[-1] if error_data else {}
            error_message = latest_error.get('message', 'Unknown error')
            error_source = latest_error.get('source', 'System')
            
            if self.explanation_depth == 'comprehensive':
                return f"""🚨 COMPREHENSIVE CRITICAL ERROR ANALYSIS
• Error Classification: Critical system error requiring immediate attention
• Error Count: {error_count} critical alerts active
• Primary Error: {error_message}
• Error Source: {error_source}
• System Impact: Trading operations may be compromised
• Immediate Actions: 
  - Review system logs for detailed error information
  - Check module health status and connectivity
  - Consider temporary trading halt if errors persist
• Recovery Procedures:
  - Restart affected modules if safe to do so
  - Verify data integrity and system connections
  - Monitor for error resolution and system stability
• Risk Assessment: High - system reliability compromised
• Monitoring: Continuous error tracking until resolution
• Escalation: Technical support may be required for complex issues"""
            elif self.explanation_depth == 'detailed':
                return f"🚨 Critical Error: {error_count} alerts - {error_message} from {error_source}. Immediate attention required."
            else:  # brief
                return f"🚨 Critical Error: {error_count} alerts - {error_message}"
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "critical_error_explanation")
            return f"Critical error explanation failed: {error_context}"

    async def _generate_drawdown_explanation(self, drawdown_data: Dict[str, Any]) -> str:
        """Generate explanation for drawdown situation"""
        try:
            drawdown = drawdown_data.get('drawdown', 0)
            risk_metrics = drawdown_data.get('risk_metrics', {})
            
            drawdown_pct = drawdown * 100
            severity = "CRITICAL" if drawdown > 0.15 else "HIGH" if drawdown > 0.1 else "MODERATE"
            
            if self.explanation_depth == 'comprehensive':
                return f"""⚠️ COMPREHENSIVE DRAWDOWN RISK ANALYSIS
• Drawdown Level: {drawdown_pct:.1f}% - {severity} RISK
• Risk Classification: Capital preservation measures required
• Current Exposure: {risk_metrics.get('total_exposure', 'Unknown')}
• Position Count: {len(self.smart_bus.get('positions', 'ExplanationGenerator') or [])}
• Impact Assessment: Significant capital at risk requiring immediate action
• Protective Measures:
  - Reduce position sizes immediately
  - Review and tighten stop-loss levels
  - Consider closing underperforming positions
  - Halt new position entries temporarily
• Recovery Strategy:
  - Focus on capital preservation over profit generation
  - Wait for favorable market conditions before re-engaging
  - Implement smaller position sizes for future trades
• Monitoring Protocol: Continuous drawdown tracking until recovery
• Risk Tolerance: Reassess risk parameters and position sizing rules
• Session Adjustment: Prioritize damage control over profit targets"""
            elif self.explanation_depth == 'detailed':
                return f"⚠️ Drawdown Alert: {drawdown_pct:.1f}% ({severity}) - Reduce positions and implement protective measures"
            else:  # brief
                return f"⚠️ Drawdown: {drawdown_pct:.1f}% - Risk management required"
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "drawdown_explanation")
            return f"Drawdown explanation error: {error_context}"

    async def _generate_learning_milestone_explanation(self, learning_data: Dict[str, Any]) -> str:
        """Generate explanation for learning milestone"""
        try:
            stage_info = learning_data.get('stage_progression', {})
            current_stage = learning_data.get('current_stage', 'Unknown')
            progress = learning_data.get('stage_progress', 0) * 100
            
            if self.explanation_depth == 'comprehensive':
                return f"""🎓 COMPREHENSIVE LEARNING MILESTONE ANALYSIS
• Learning Achievement: Significant progress milestone reached
• Current Stage: {current_stage} ({progress:.0f}% completion)
• Milestone Type: {stage_info.get('recommendation', 'Stage progression')}
• Competency Development: Multiple skill areas showing improvement
• Performance Validation: Recent trading results support advancement
• Skill Assessment:
  - Risk Management: Enhanced through practical application
  - Entry Timing: Improved through market experience
  - Exit Strategy: Refined through outcome analysis
  - Position Sizing: Optimized through performance feedback
• Learning Velocity: Consistent improvement trend maintained
• Next Phase: Prepare for advanced skill development
• Strategy Evolution: Incorporate learned techniques into systematic approach
• Confidence Building: Successful milestone completion builds trading confidence
• Systematic Progress: Learning framework proving effective for skill development"""
            elif self.explanation_depth == 'detailed':
                return f"🎓 Learning Milestone: {current_stage} stage progress ({progress:.0f}%) - {stage_info.get('recommendation', 'Continue development')}"
            else:  # brief
                return f"🎓 Learning: {current_stage} milestone reached"
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "learning_milestone_explanation")
            return f"Learning milestone explanation error: {error_context}"

    async def _generate_generic_explanation(self, context_type: str, context_data: Any) -> str:
        """Generate generic explanation for unknown context types"""
        try:
            # Format context type for display
            formatted_type = context_type.replace('_', ' ').title()
            
            if self.explanation_depth == 'comprehensive':
                return f"""ℹ️ SYSTEM EVENT ANALYSIS
• Event Type: {formatted_type}
• Event Classification: System monitoring and status update
• Context Information: {str(context_data)[:200]}{'...' if len(str(context_data)) > 200 else ''}
• System Response: Event logged and processed by explanation system
• Impact Assessment: Routine system operation - no immediate action required
• Monitoring Status: Event tracked for pattern analysis and system optimization
• Operational Context: Part of normal system monitoring and status reporting
• Next Steps: Continue monitoring for trends and patterns
• Information Value: Contributes to overall system awareness and transparency
• Integration: Event data integrated into system decision-making processes"""
            elif self.explanation_depth == 'detailed':
                return f"ℹ️ System Event: {formatted_type} - {str(context_data)[:100]}{'...' if len(str(context_data)) > 100 else ''}"
            else:  # brief
                return f"ℹ️ {formatted_type} event"
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "generic_explanation")
            return f"System event: {context_type}"

    async def _generate_profitable_trade_explanation(self, trade_data: Dict[str, Any]) -> str:
        """Generate explanation for profitable trade"""
        try:
            template = self.explanation_templates['trade_exit'][self.explanation_depth]
            
            # Extract trade details
            instrument = trade_data.get('symbol', 'Unknown')
            pnl = trade_data.get('pnl', 0)
            duration = trade_data.get('duration', 0)
            size = trade_data.get('size', 0)
            
            # Calculate additional metrics
            target_percentage = (pnl / self.target_daily_profit) * 100
            performance_rating = self._calculate_trade_performance_rating(trade_data)
            
            if self.explanation_depth == 'comprehensive':
                return template.format(
                    instrument=instrument,
                    exit_reason="Profit target achieved",
                    pnl=f"€{pnl:+.2f}",
                    pnl_percentage=f"{target_percentage:.1f}% of daily target",
                    duration=f"{duration} steps",
                    vs_expectation="Exceeded expectations",
                    target_achievement=f"{target_percentage:.1f}% of daily target",
                    risk_management_effectiveness="Excellent - protected capital while maximizing gains",
                    market_evolution=f"Market conditions favorable during {duration} step holding period",
                    lessons_learned="Strategy validation - current approach working well",
                    strategy_validation="Confirmed effectiveness in current market regime",
                    session_impact=f"Positive momentum - contributes {target_percentage:.1f}% to daily goal"
                )
            elif self.explanation_depth == 'detailed':
                return template.format(
                    instrument=instrument,
                    exit_reason="Profit target achieved",
                    pnl=f"€{pnl:+.2f}",
                    duration=f"{duration} steps",
                    performance_rating=performance_rating
                )
            else:  # brief
                return template.format(
                    instrument=instrument,
                    exit_reason="Profit target achieved",
                    pnl=f"€{pnl:+.2f}"
                )
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "profitable_trade_explanation")
            return f"Profitable trade explanation error: {error_context}"

    async def _generate_performance_insights(self) -> Dict[str, Any]:
        """Generate comprehensive performance insights"""
        try:
            insights = {}
            
            # Calculate session metrics
            recent_trades = self.smart_bus.get('recent_trades', 'ExplanationGenerator') or []
            session_pnl = sum(t.get('pnl', 0) for t in recent_trades)
            
            if recent_trades:
                pnls = [t.get('pnl', 0) for t in recent_trades]
                win_rate = (len([p for p in pnls if p > 0]) / len(pnls)) * 100
                avg_trade = np.mean(pnls)
                best_trade = max(pnls)
                worst_trade = min(pnls)
            else:
                win_rate = 0
                avg_trade = 0
                best_trade = 0
                worst_trade = 0
            
            # Performance trend analysis
            performance_trend = self._analyze_performance_trend(recent_trades)
            
            # Target progress
            target_progress = (session_pnl / self.target_daily_profit) * 100
            
            # Session duration
            session_duration = datetime.datetime.now() - self.session_metrics['session_start']
            hours_elapsed = session_duration.total_seconds() / 3600
            
            insights = {
                'session_pnl': session_pnl,
                'target_progress': target_progress,
                'trade_count': len(recent_trades),
                'win_rate': win_rate,
                'avg_trade': avg_trade,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'performance_trend': performance_trend,
                'session_duration': hours_elapsed,
                'efficiency_rating': self._calculate_efficiency_rating(session_pnl, hours_elapsed),
                'target_probability': self._estimate_target_probability(session_pnl, hours_elapsed)
            }
            
            return insights
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_insights")
            return {'error': str(error_context)}

    async def _generate_comprehensive_explanation_thesis(self, explanations: Dict[str, Any], 
                                                        analysis: Dict[str, Any]) -> str:
        """Generate comprehensive thesis explaining all explanation decisions"""
        try:
            thesis_parts = []
            
            # Executive Summary
            total_explanations = sum(len(v) if isinstance(v, list) else 1 for v in explanations.values())
            priority_contexts = analysis.get('priority_contexts', [])
            
            if priority_contexts:
                highest_priority = priority_contexts[0]
                thesis_parts.append(
                    f"EXPLANATION FOCUS: {highest_priority['type'].replace('_', ' ').title()} "
                    f"(Priority {highest_priority['priority']}) requiring immediate attention"
                )
            else:
                thesis_parts.append("ROUTINE EXPLANATION CYCLE: Standard system status and performance updates")
            
            # Explanation Generation Summary
            thesis_parts.append(f"EXPLANATION OUTPUT: Generated {total_explanations} explanations across categories")
            
            # Category Breakdown
            for category, content in explanations.items():
                if isinstance(content, list) and content:
                    thesis_parts.append(f"  • {category.replace('_', ' ').title()}: {len(content)} explanations")
                elif isinstance(content, dict) and content:
                    thesis_parts.append(f"  • {category.replace('_', ' ').title()}: Comprehensive analysis provided")
            
            # Context Analysis
            triggers = analysis.get('explanation_triggers', [])
            if triggers:
                thesis_parts.append(f"TRIGGER ANALYSIS: {len(triggers)} explanation triggers identified")
                primary_trigger = triggers[0].replace('_', ' ').title()
                thesis_parts.append(f"  • Primary: {primary_trigger}")
            
            # Narrative Intelligence
            narrative_reqs = analysis.get('narrative_requirements', {})
            urgency = analysis.get('urgency_level', 'normal')
            tone = analysis.get('emotional_tone', 'neutral')
            depth = analysis.get('technical_depth', 'moderate')
            
            thesis_parts.append(
                f"NARRATIVE INTELLIGENCE: {urgency.title()} urgency, {tone} tone, {depth} technical depth"
            )
            
            # Performance Context
            performance_data = explanations.get('performance', {})
            if performance_data:
                session_pnl = performance_data.get('session_pnl', 0)
                target_progress = performance_data.get('target_progress', 0)
                thesis_parts.append(
                    f"PERFORMANCE CONTEXT: €{session_pnl:.2f} session P&L ({target_progress:.1f}% of target)"
                )
            
            # System Health Context
            health_metrics = self._get_health_metrics()
            health_status = health_metrics.get('status', 'unknown')
            thesis_parts.append(f"SYSTEM CONTEXT: Explanation system {health_status}")
            
            # Quality Assessment
            explanation_quality = self.session_metrics.get('explanation_quality_score', 0)
            context_accuracy = self.session_metrics.get('context_accuracy', 0)
            thesis_parts.append(
                f"QUALITY METRICS: {explanation_quality:.1%} explanation quality, "
                f"{context_accuracy:.1%} context accuracy"
            )
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Explanation thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive explanation results"""
        try:
            # Core explanations
            self.smart_bus.set('trading_explanations', results['trading_explanations'],
                             module='ExplanationGenerator', thesis=thesis)
            
            # System explanations
            system_thesis = f"Generated {len(results['system_explanations'])} system explanations"
            self.smart_bus.set('system_explanations', results['system_explanations'],
                             module='ExplanationGenerator', thesis=system_thesis)
            
            # Performance insights
            performance_thesis = f"Performance insights: {results['performance_insights'].get('target_progress', 0):.1f}% target progress"
            self.smart_bus.set('performance_insights', results['performance_insights'],
                             module='ExplanationGenerator', thesis=performance_thesis)
            
            # Contextual narratives
            narrative_thesis = f"Generated {len(results['contextual_narratives'])} contextual narratives"
            self.smart_bus.set('contextual_narratives', results['contextual_narratives'],
                             module='ExplanationGenerator', thesis=narrative_thesis)
            
            # Operator updates
            update_thesis = f"Operator updates: {len(results['operator_updates'])} status updates"
            self.smart_bus.set('operator_updates', results['operator_updates'],
                             module='ExplanationGenerator', thesis=update_thesis)
            
            # Decision rationales
            rationale_thesis = f"Decision rationales: {len(results['decision_rationales'])} explanations"
            self.smart_bus.set('decision_rationales', results['decision_rationales'],
                             module='ExplanationGenerator', thesis=rationale_thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # HELPER METHODS AND UTILITIES
    # ═══════════════════════════════════════════════════════════════════

    def _categorize_explanation(self, context_type: str) -> str:
        """Categorize explanation into appropriate result category"""
        trading_contexts = ['high_profit_trade', 'high_loss_trade', 'trade_entry', 'trade_exit']
        system_contexts = ['system_critical_error', 'system_status', 'health_check']
        
        if context_type in trading_contexts:
            return 'trading'
        elif context_type in system_contexts:
            return 'system'
        elif 'performance' in context_type or 'target' in context_type:
            return 'updates'
        else:
            return 'narratives'

    def _calculate_trade_performance_rating(self, trade_data: Dict[str, Any]) -> str:
        """Calculate performance rating for a trade"""
        pnl = trade_data.get('pnl', 0)
        
        if pnl > 100:
            return "Excellent"
        elif pnl > 50:
            return "Good"
        elif pnl > 0:
            return "Positive"
        elif pnl > -25:
            return "Acceptable"
        else:
            return "Poor"

    def _analyze_performance_trend(self, trades: List[Dict[str, Any]]) -> str:
        """Analyze performance trend from recent trades"""
        if not trades or len(trades) < 3:
            return "Insufficient data"
        
        recent_pnls = [t.get('pnl', 0) for t in trades[-5:]]
        
        if len(recent_pnls) < 2:
            return "Neutral"
        
        # Calculate trend
        trend = recent_pnls[-1] - recent_pnls[0]
        avg_pnl = np.mean(recent_pnls)
        
        if trend > 20 and avg_pnl > 0:
            return "Strongly Improving"
        elif trend > 0 and avg_pnl > 0:
            return "Improving"
        elif abs(trend) <= 10:
            return "Stable"
        elif trend < 0 and avg_pnl < 0:
            return "Declining"
        else:
            return "Mixed"

    def _calculate_efficiency_rating(self, session_pnl: float, hours_elapsed: float) -> float:
        """Calculate efficiency rating (0-10)"""
        if hours_elapsed <= 0:
            return 5.0
        
        hourly_rate = session_pnl / hours_elapsed
        target_hourly = self.target_daily_profit / 8  # 8-hour day
        
        efficiency = (hourly_rate / target_hourly) * 5  # Scale to 0-10
        return min(10.0, max(0.0, efficiency))

    def _estimate_target_probability(self, session_pnl: float, hours_elapsed: float) -> float:
        """Estimate probability of reaching daily target"""
        if hours_elapsed <= 0:
            return 50.0
        
        current_progress = session_pnl / self.target_daily_profit
        time_progress = hours_elapsed / 8  # 8-hour day
        
        if current_progress >= 1.0:
            return 100.0
        
        # Simple linear projection with some optimism
        remaining_time = max(0, 8 - hours_elapsed)
        if remaining_time <= 0:
            return 0.0 if current_progress < 1.0 else 100.0
        
        required_rate = (self.target_daily_profit - session_pnl) / remaining_time
        current_rate = session_pnl / hours_elapsed if hours_elapsed > 0 else 0
        
        if current_rate <= 0:
            return 10.0  # Low but not zero
        
        probability = min(90.0, (current_rate / required_rate) * 50)
        return max(10.0, probability)

    def _detect_context_changes(self, context: Dict[str, Any]) -> List[str]:
        """Detect significant context changes"""
        changes = []
        
        try:
            # Market regime change
            current_regime = context.get('market_context', {}).get('regime', 'unknown')
            if current_regime != self.decision_context.get('market_regime', 'unknown'):
                changes.append('market_regime_change')
            
            # Risk level change
            risk_data = context.get('risk_data', {})
            current_risk = self._assess_current_risk_level(risk_data)
            if current_risk != self.decision_context.get('risk_level', 'medium'):
                changes.append('risk_level_change')
            
            # Performance trend change
            recent_trades = context.get('recent_trades', [])
            current_performance = self._assess_recent_performance(recent_trades)
            if current_performance != self.decision_context.get('recent_performance', 'neutral'):
                changes.append('performance_trend_change')
            
            return changes
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "context_change_detection")
            return []

    def _assess_current_risk_level(self, risk_data: Dict[str, Any]) -> str:
        """Assess current risk level from risk data"""
        try:
            drawdown = risk_data.get('current_drawdown', 0)
            exposure = risk_data.get('total_exposure', 0)
            
            if drawdown > 0.15 or exposure > 3.0:
                return 'critical'
            elif drawdown > 0.08 or exposure > 2.0:
                return 'high'
            elif drawdown > 0.03 or exposure > 1.0:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'medium'

    def _assess_recent_performance(self, recent_trades: List[Dict]) -> str:
        """Assess recent performance trend"""
        try:
            if not recent_trades or len(recent_trades) < 3:
                return 'neutral'
            
            last_trades = recent_trades[-5:]
            pnls = [t.get('pnl', 0) for t in last_trades]
            
            total_pnl = sum(pnls)
            win_rate = len([p for p in pnls if p > 0]) / len(pnls)
            
            if total_pnl > 50 and win_rate > 0.6:
                return 'excellent'
            elif total_pnl > 0 and win_rate > 0.5:
                return 'good'
            elif total_pnl > -50 and win_rate > 0.4:
                return 'neutral'
            else:
                return 'poor'
                
        except Exception:
            return 'neutral'

    async def _update_decision_context_advanced(self, context: Dict[str, Any]):
        """Update decision context with advanced analysis"""
        try:
            # Update basic context
            self.decision_context.update({
                'market_regime': context.get('market_context', {}).get('regime', 'unknown'),
                'risk_level': self._assess_current_risk_level(context.get('risk_data', {})),
                'recent_performance': self._assess_recent_performance(context.get('recent_trades', [])),
                'system_alerts': context.get('system_alerts', [])
            })
            
            # Update advanced context from module insights
            bias_analysis = context.get('bias_analysis', {})
            if bias_analysis:
                active_biases = bias_analysis.get('individual_biases', {})
                significant_biases = [k for k, v in active_biases.items() if v > 0.3]
                self.decision_context['bias_warnings'] = significant_biases
            
            learning_status = context.get('learning_status', {})
            if learning_status:
                self.decision_context['learning_stage'] = learning_status.get('stage_name', 'unknown')
            
            strategy_status = context.get('strategy_status', {})
            if strategy_status:
                self.decision_context['current_strategy'] = strategy_status.get('active_strategy', 'unknown')
                self.decision_context['confidence_level'] = strategy_status.get('confidence', 0.5)
            
            # Update session metrics
            recent_trades = context.get('recent_trades', [])
            self.session_metrics.update({
                'trade_count': len(recent_trades),
                'profit_today': sum(t.get('pnl', 0) for t in recent_trades),
                'last_update': datetime.datetime.now()
            })
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "context_update")
            self.logger.warning(f"Context update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # ERROR HANDLING AND RECOVERY
    # ═══════════════════════════════════════════════════════════════════

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "ExplanationGenerator")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="Explanation Generator disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('ExplanationGenerator', 'process_time', processing_time, False)
        
        return {
            'trading_explanations': [f"Explanation system error: {error_context}"],
            'system_explanations': ["Explanation generation temporarily degraded"],
            'performance_insights': {'error': str(error_context)},
            'contextual_narratives': ["Narrative generation failed"],
            'operator_updates': ["System operating in degraded mode"],
            'decision_rationales': ["Rationale generation unavailable"],
            'explanation_metrics': self.session_metrics.copy(),
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def _get_safe_context_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when context retrieval fails"""
        return {
            'recent_trades': [],
            'positions': [],
            'risk_data': {},
            'system_alerts': [],
            'market_context': {},
            'session_metrics': {},
            'module_insights': {},
            'performance_data': {},
            'learning_status': {},
            'bias_analysis': {},
            'strategy_status': {}
        }

    def _get_safe_analysis_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when analysis fails"""
        return {
            'priority_contexts': [],
            'explanation_triggers': ['error_recovery'],
            'context_changes': [],
            'narrative_requirements': {},
            'urgency_level': 'normal',
            'emotional_tone': 'neutral',
            'technical_depth': 'moderate'
        }

    def _get_safe_explanation_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when explanation generation fails"""
        return {
            'trading': ["Explanation generation temporarily unavailable"],
            'system': ["System operating with limited explanations"],
            'performance': {'error': 'Performance insights unavailable'},
            'narratives': ["Narrative generation failed"],
            'updates': ["Limited system updates available"],
            'rationales': ["Decision rationales temporarily unavailable"]
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'trading_explanations': ["Explanation Generator disabled"],
            'system_explanations': ["System explanations unavailable"],
            'performance_insights': {'status': 'disabled'},
            'contextual_narratives': ["Narrative generation disabled"],
            'operator_updates': ["Restart explanation system"],
            'decision_rationales': ["Rationale generation disabled"],
            'explanation_metrics': {'status': 'disabled'},
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'}
        }

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _update_explanation_metrics(self, explanations: Dict[str, Any]):
        """Update explanation quality and performance metrics"""
        try:
            total_explanations = sum(len(v) if isinstance(v, list) else 1 for v in explanations.values())
            self.session_metrics['explanations_generated'] += total_explanations
            
            # Calculate quality score (placeholder - would use more sophisticated metrics)
            quality_score = 0.8  # Base quality
            if explanations.get('trading'):
                quality_score += 0.1
            if explanations.get('performance', {}).get('target_progress'):
                quality_score += 0.1
            
            self.session_metrics['explanation_quality_score'] = min(1.0, quality_score)
            self.session_metrics['context_accuracy'] = 0.9  # Placeholder
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "metrics_update")
            self.logger.warning(f"Metrics update failed: {error_context}")

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'ExplanationGenerator',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'explanations_generated': self.session_metrics['explanations_generated'],
            'explanation_quality': self.session_metrics.get('explanation_quality_score', 0),
            'context_accuracy': self.session_metrics.get('context_accuracy', 0),
            'current_depth': self.explanation_depth,
            'categories_active': len(self.explanation_categories),
            'session_duration': (datetime.datetime.now() - self.session_metrics['session_start']).total_seconds() / 3600
        }

    def get_explanation_report(self) -> str:
        """Generate comprehensive explanation activity report"""
        session_duration = datetime.datetime.now() - self.session_metrics['session_start']
        hours_active = session_duration.total_seconds() / 3600
        
        # Recent explanations summary
        recent_explanations = ""
        if self.explanation_history:
            for exp in list(self.explanation_history)[-3:]:
                timestamp = exp['timestamp'][:19].replace('T', ' ')
                context = exp['context']
                recent_explanations += f"  • {timestamp}: {context}\n"
        
        # Performance metrics
        quality_score = self.session_metrics.get('explanation_quality_score', 0)
        context_accuracy = self.session_metrics.get('context_accuracy', 0)
        
        return f"""
🎤 EXPLANATION GENERATOR COMPREHENSIVE REPORT
═══════════════════════════════════════════════════════════════
📊 Session Overview:
• Session Duration: {hours_active:.1f} hours
• Explanations Generated: {self.session_metrics['explanations_generated']}
• Current Depth: {self.explanation_depth.title()}
• Update Frequency: Every {self.update_frequency} minute(s)
• Quality Score: {quality_score:.1%}
• Context Accuracy: {context_accuracy:.1%}

💰 Performance Context:
• Current P&L: €{self.session_metrics['profit_today']:.2f}
• Target Progress: {(self.session_metrics['profit_today']/self.target_daily_profit)*100:.1f}%
• Trade Count: {self.session_metrics['trade_count']}
• Recent Performance: {self.decision_context['recent_performance'].title()}

🌊 Market Context:
• Market Regime: {self.decision_context['market_regime'].title()}
• Risk Level: {self.decision_context['risk_level'].title()}
• System Confidence: {self.decision_context['confidence_level']:.1%}
• Active Strategy: {self.decision_context['current_strategy'].title()}
• Learning Stage: {self.decision_context['learning_stage'].title()}

🧠 Intelligence Context:
• Bias Warnings: {len(self.decision_context['bias_warnings'])} active
• System Alerts: {len(self.decision_context['system_alerts'])} pending
• Narrative Tone: {self.narrative_intelligence['emotional_tone'].title()}
• Technical Depth: {self.narrative_intelligence['technical_depth'].title()}

📝 Recent Explanations:
{recent_explanations if recent_explanations else '  📭 No recent explanations'}

🎯 Current Focus:
• Explanation Categories: {len(self.explanation_categories)} available
• Priority Levels: {len(self.context_priorities)} context priorities
• Circuit Breaker: {'ACTIVE' if self.is_disabled else 'INACTIVE'}

📊 Health Status:
• Error Count: {self.error_count}/{self.circuit_breaker_threshold}
• System Status: {'DISABLED' if self.is_disabled else 'OPERATIONAL'}
• Performance Rating: {self._calculate_performance_rating()}/10

📈 Last Generated Content:
{self.last_explanation[:300]}{'...' if len(self.last_explanation) > 300 else ''}
        """

    def _calculate_performance_rating(self) -> float:
        """Calculate overall performance rating (0-10)"""
        try:
            base_score = 8.0  # Good baseline
            
            # Deduct for errors
            error_penalty = self.error_count * 0.5
            base_score -= error_penalty
            
            # Bonus for quality
            quality_bonus = self.session_metrics.get('explanation_quality_score', 0.8) * 2
            base_score += quality_bonus
            
            # Bonus for accuracy
            accuracy_bonus = self.session_metrics.get('context_accuracy', 0.8) * 1
            base_score += accuracy_bonus
            
            return min(10.0, max(0.0, base_score))
            
        except Exception:
            return 5.0

    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT FOR HOT-RELOAD
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'ExplanationGenerator',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'explanation_depth': self.explanation_depth,
                'update_frequency': self.update_frequency,
                'target_daily_profit': self.target_daily_profit,
                'debug': self.debug
            },
            'session_state': {
                'last_explanation': self.last_explanation,
                'session_metrics': self.session_metrics.copy(),
                'decision_context': self.decision_context.copy(),
                'explanation_history': list(self.explanation_history)[-20:],  # Keep recent only
                'narrative_intelligence': self.narrative_intelligence.copy()
            },
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'templates': self.explanation_templates.copy(),
            'priorities': self.context_priorities.copy(),
            'performance_metrics': self._get_health_metrics()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.explanation_depth = config.get("explanation_depth", self.explanation_depth)
            self.update_frequency = int(config.get("update_frequency", self.update_frequency))
            self.target_daily_profit = float(config.get("target_daily_profit", self.target_daily_profit))
            self.debug = bool(config.get("debug", self.debug))
            
            # Load session state
            session_state = state.get("session_state", {})
            self.last_explanation = session_state.get("last_explanation", "")
            self.session_metrics.update(session_state.get("session_metrics", {}))
            self.decision_context.update(session_state.get("decision_context", {}))
            self.narrative_intelligence.update(session_state.get("narrative_intelligence", {}))
            
            # Restore explanation history
            history_data = session_state.get("explanation_history", [])
            self.explanation_history = deque(history_data, maxlen=100)
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            # Load templates and priorities if provided
            self.explanation_templates.update(state.get("templates", {}))
            self.context_priorities.update(state.get("priorities", {}))
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Explanation Generator state restored",
                depth=self.explanation_depth,
                explanations=len(self.explanation_history),
                quality=f"{self.session_metrics.get('explanation_quality_score', 0):.1%}"
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for system monitoring"""
        return {
            'module_name': 'ExplanationGenerator',
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
                'message': 'ExplanationGenerator disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })
        
        if self.error_count > 2:
            alerts.append({
                'severity': 'warning',
                'message': f'High error count: {self.error_count}',
                'action': 'Monitor for recurring issues'
            })
        
        quality_score = self.session_metrics.get('explanation_quality_score', 1.0)
        if quality_score < 0.6:
            alerts.append({
                'severity': 'warning',
                'message': f'Low explanation quality: {quality_score:.1%}',
                'action': 'Review explanation templates and context accuracy'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        if self.is_disabled:
            recommendations.append("Restart ExplanationGenerator module after investigating errors")
        
        if len(self.explanation_history) < 10:
            recommendations.append("Insufficient explanation history - continue operations to build baseline")
        
        if self.session_metrics.get('explanation_quality_score', 1.0) < 0.7:
            recommendations.append("Consider improving explanation templates and context analysis")
        
        if not recommendations:
            recommendations.append("ExplanationGenerator operating within normal parameters")
        
        return recommendations

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API METHODS (for external use)
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Return explanation metrics for observation"""
        try:
            # Current session metrics
            progress_ratio = self.session_metrics['profit_today'] / max(1, self.target_daily_profit)
            trade_frequency = self.session_metrics['trade_count'] / max(1, 
                (datetime.datetime.now() - self.session_metrics['session_start']).total_seconds() / 3600)
            
            # Risk and performance assessment
            risk_score = {
                'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0
            }.get(self.decision_context['risk_level'], 0.5)
            
            performance_score = {
                'poor': 0.1, 'neutral': 0.5, 'good': 0.7, 'excellent': 0.9
            }.get(self.decision_context['recent_performance'], 0.5)
            
            # Explanation metrics
            explanation_frequency = float(self.session_metrics['explanations_generated']) / 100
            quality_score = self.session_metrics.get('explanation_quality_score', 0.8)
            
            observation = np.array([
                np.clip(progress_ratio, -2.0, 3.0),  # Progress toward target
                float(self.session_metrics['trade_count']),  # Trade count
                np.clip(trade_frequency, 0, 10),  # Trade frequency per hour
                risk_score,  # Current risk level
                performance_score,  # Recent performance
                self.decision_context['confidence_level'],  # System confidence
                float(len(self.decision_context['system_alerts'])),  # Alert count
                explanation_frequency,  # Explanation frequency
                quality_score,  # Explanation quality
                float(len(self.decision_context['bias_warnings']))  # Bias warning count
            ], dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid explanation observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Explanation observation generation failed: {error_context}")
            return np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.8, 0.0], dtype=np.float32)