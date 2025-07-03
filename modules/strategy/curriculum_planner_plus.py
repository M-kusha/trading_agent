# ─────────────────────────────────────────────────────────────
# File: modules/strategy/curriculum_planner_plus.py
# Enhanced with InfoBus integration & intelligent curriculum adaptation
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class CurriculumPlannerPlus(Module, AnalysisMixin, StateManagementMixin, TradingMixin):
    """
    Enhanced curriculum planner with InfoBus integration.
    Dynamically adapts training curriculum based on performance patterns and market conditions.
    Provides intelligent learning progression for strategy optimization.
    """

    def __init__(
        self,
        window: int = 10,
        debug: bool = False,
        adaptation_rate: float = 0.1,
        performance_threshold: float = 0.6,
        difficulty_levels: int = 5,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=window * 2,
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.window = int(window)
        self.debug = bool(debug)
        self.adaptation_rate = float(adaptation_rate)
        self.performance_threshold = float(performance_threshold)
        self.difficulty_levels = int(difficulty_levels)
        
        # Curriculum state
        self.episode_history = deque(maxlen=self.window)
        self.performance_metrics = defaultdict(list)
        self.curriculum_stages = self._initialize_curriculum_stages()
        self.current_stage = 0
        self.stage_progress = 0.0
        
        # Learning analytics
        self.learning_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'current_difficulty': 1.0,
            'adaptation_events': 0,
            'mastery_level': 0.0,
            'learning_rate': 0.0,
            'stage_completion_times': []
        }
        
        # Performance tracking
        self.competency_areas = {
            'risk_management': {'score': 0.5, 'weight': 0.3, 'target': 0.8},
            'entry_timing': {'score': 0.5, 'weight': 0.25, 'target': 0.75},
            'exit_strategy': {'score': 0.5, 'weight': 0.25, 'target': 0.75},
            'position_sizing': {'score': 0.5, 'weight': 0.2, 'target': 0.7}
        }
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "CurriculumPlannerPlus",
            "logs/strategy/curriculum_planner.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("CurriculumPlannerPlus")
        
        self.log_operator_info(
            "📚 Curriculum Planner initialized",
            window=self.window,
            difficulty_levels=self.difficulty_levels,
            current_stage=self.current_stage
        )

    def _initialize_curriculum_stages(self) -> List[Dict[str, Any]]:
        """Initialize progressive curriculum stages"""
        
        return [
            {
                'name': 'Foundation',
                'description': 'Basic trading mechanics and risk awareness',
                'difficulty': 1.0,
                'focus_areas': ['risk_management', 'position_sizing'],
                'success_criteria': {'win_rate': 0.4, 'max_drawdown': 0.1, 'profit_factor': 1.0},
                'market_conditions': ['ranging', 'low_volatility'],
                'max_position_size': 0.5,
                'max_trades_per_day': 5
            },
            {
                'name': 'Development',
                'description': 'Improved timing and basic strategy implementation',
                'difficulty': 2.0,
                'focus_areas': ['entry_timing', 'exit_strategy'],
                'success_criteria': {'win_rate': 0.5, 'max_drawdown': 0.08, 'profit_factor': 1.2},
                'market_conditions': ['trending', 'ranging'],
                'max_position_size': 0.75,
                'max_trades_per_day': 8
            },
            {
                'name': 'Intermediate',
                'description': 'Multi-timeframe analysis and advanced entries',
                'difficulty': 3.0,
                'focus_areas': ['entry_timing', 'risk_management', 'exit_strategy'],
                'success_criteria': {'win_rate': 0.55, 'max_drawdown': 0.06, 'profit_factor': 1.5},
                'market_conditions': ['trending', 'ranging', 'volatile'],
                'max_position_size': 1.0,
                'max_trades_per_day': 12
            },
            {
                'name': 'Advanced',
                'description': 'Complex strategies and adaptive position sizing',
                'difficulty': 4.0,
                'focus_areas': ['position_sizing', 'risk_management', 'entry_timing'],
                'success_criteria': {'win_rate': 0.6, 'max_drawdown': 0.05, 'profit_factor': 2.0},
                'market_conditions': ['all'],
                'max_position_size': 1.5,
                'max_trades_per_day': 15
            },
            {
                'name': 'Expert',
                'description': 'Full autonomy and adaptive trading mastery',
                'difficulty': 5.0,
                'focus_areas': ['all'],
                'success_criteria': {'win_rate': 0.65, 'max_drawdown': 0.04, 'profit_factor': 2.5},
                'market_conditions': ['all'],
                'max_position_size': 2.0,
                'max_trades_per_day': 20
            }
        ]

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Clear curriculum history
        self.episode_history.clear()
        self.performance_metrics.clear()
        
        # Reset to initial stage
        self.current_stage = 0
        self.stage_progress = 0.0
        
        # Reset learning stats
        self.learning_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'current_difficulty': 1.0,
            'adaptation_events': 0,
            'mastery_level': 0.0,
            'learning_rate': 0.0,
            'stage_completion_times': []
        }
        
        # Reset competency scores
        for area in self.competency_areas:
            self.competency_areas[area]['score'] = 0.5
        
        self.log_operator_info("🔄 Curriculum Planner reset - returning to Foundation stage")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - limited curriculum analysis")
            return
        
        # Extract context and performance data
        context = extract_standard_context(info_bus)
        performance_data = self._extract_performance_data_from_info_bus(info_bus, context)
        
        # Update curriculum based on recent performance
        if performance_data:
            self._update_curriculum_progress(performance_data, context)
        
        # Check for stage progression
        self._evaluate_stage_progression(info_bus, context)
        
        # Update InfoBus with curriculum recommendations
        self._update_info_bus_with_curriculum_data(info_bus)

    def _extract_performance_data_from_info_bus(self, info_bus: InfoBus, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance data needed for curriculum adaptation"""
        
        try:
            # Get trading performance
            risk_data = info_bus.get('risk', {})
            recent_trades = info_bus.get('recent_trades', [])
            
            performance_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'balance': risk_data.get('balance', 0),
                'equity': risk_data.get('equity', 0),
                'drawdown': risk_data.get('current_drawdown', 0),
                'trades_count': len(recent_trades),
                'session_pnl': context.get('session_pnl', 0),
                'market_regime': context.get('regime', 'unknown')
            }
            
            # Calculate episode-specific metrics if we have trades
            if recent_trades:
                pnls = [t.get('pnl', 0) for t in recent_trades]
                win_rate = len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
                profit_factor = sum([p for p in pnls if p > 0]) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else float('inf')
                
                performance_data.update({
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_trade_pnl': np.mean(pnls) if pnls else 0,
                    'trade_consistency': 1.0 - (np.std(pnls) / abs(np.mean(pnls))) if pnls and np.mean(pnls) != 0 else 0
                })
            
            return performance_data
            
        except Exception as e:
            self.log_operator_warning(f"Performance data extraction failed: {e}")
            return {}

    def record_episode(self, summary: Dict[str, Any]) -> None:
        """Record episode with enhanced validation and curriculum adaptation"""
        
        try:
            # Validate and enrich summary
            if not isinstance(summary, dict):
                self.log_operator_warning(f"Invalid episode summary type: {type(summary)}")
                return
            
            # Add timestamp and stage info
            enriched_summary = {
                'timestamp': datetime.datetime.now().isoformat(),
                'curriculum_stage': self.current_stage,
                'stage_name': self.curriculum_stages[self.current_stage]['name'],
                'difficulty': self.learning_stats['current_difficulty'],
                **summary
            }
            
            # Validate key metrics
            for key in ['total_trades', 'wins', 'pnl']:
                if key in enriched_summary:
                    value = enriched_summary[key]
                    if not isinstance(value, (int, float)) or np.isnan(value):
                        self.log_operator_warning(f"Invalid {key}: {value}, setting to 0")
                        enriched_summary[key] = 0
            
            # Store episode
            self.episode_history.append(enriched_summary)
            self.learning_stats['total_episodes'] += 1
            
            # Update performance metrics
            self._update_performance_metrics(enriched_summary)
            
            # Evaluate episode success
            episode_success = self._evaluate_episode_success(enriched_summary)
            if episode_success:
                self.learning_stats['successful_episodes'] += 1
            
            # Update competency areas
            self._update_competency_scores(enriched_summary)
            
            # Log episode
            self.log_operator_info(
                f"📈 Episode recorded",
                stage=self.curriculum_stages[self.current_stage]['name'],
                success="✅" if episode_success else "❌",
                pnl=f"€{enriched_summary.get('pnl', 0):.2f}",
                total_episodes=self.learning_stats['total_episodes']
            )
            
        except Exception as e:
            self.log_operator_error(f"Episode recording failed: {e}")

    def _update_performance_metrics(self, episode: Dict[str, Any]) -> None:
        """Update rolling performance metrics"""
        
        for metric in ['win_rate', 'profit_factor', 'avg_duration', 'avg_drawdown', 'pnl']:
            if metric in episode:
                self.performance_metrics[metric].append(episode[metric])
                
                # Keep only recent values
                if len(self.performance_metrics[metric]) > self.window * 2:
                    self.performance_metrics[metric] = self.performance_metrics[metric][-self.window:]

    def _evaluate_episode_success(self, episode: Dict[str, Any]) -> bool:
        """Evaluate if episode meets current stage success criteria"""
        
        try:
            current_stage_criteria = self.curriculum_stages[self.current_stage]['success_criteria']
            
            # Check each criterion
            for criterion, target in current_stage_criteria.items():
                actual_value = episode.get(criterion, 0)
                
                if criterion == 'max_drawdown':
                    # For drawdown, actual should be less than target
                    if actual_value > target:
                        return False
                else:
                    # For other metrics, actual should be greater than target
                    if actual_value < target:
                        return False
            
            return True
            
        except Exception as e:
            self.log_operator_warning(f"Episode success evaluation failed: {e}")
            return False

    def _update_competency_scores(self, episode: Dict[str, Any]) -> None:
        """Update competency area scores based on episode performance"""
        
        try:
            # Risk management competency
            if 'max_drawdown' in episode:
                target_dd = self.curriculum_stages[self.current_stage]['success_criteria'].get('max_drawdown', 0.1)
                actual_dd = episode['max_drawdown']
                risk_score = max(0, 1.0 - (actual_dd / target_dd))
                self._update_competency_score('risk_management', risk_score)
            
            # Entry timing competency
            if 'win_rate' in episode:
                target_wr = self.curriculum_stages[self.current_stage]['success_criteria'].get('win_rate', 0.5)
                actual_wr = episode['win_rate']
                entry_score = min(1.0, actual_wr / target_wr)
                self._update_competency_score('entry_timing', entry_score)
            
            # Exit strategy competency
            if 'profit_factor' in episode:
                target_pf = self.curriculum_stages[self.current_stage]['success_criteria'].get('profit_factor', 1.0)
                actual_pf = episode['profit_factor']
                exit_score = min(1.0, actual_pf / target_pf)
                self._update_competency_score('exit_strategy', exit_score)
            
            # Position sizing competency (based on consistency)
            if 'trade_consistency' in episode:
                consistency = episode['trade_consistency']
                self._update_competency_score('position_sizing', consistency)
            
        except Exception as e:
            self.log_operator_warning(f"Competency score update failed: {e}")

    def _update_competency_score(self, area: str, new_score: float) -> None:
        """Update specific competency area score with momentum"""
        
        if area in self.competency_areas:
            current_score = self.competency_areas[area]['score']
            # Apply exponential moving average for smooth updates
            updated_score = current_score * (1 - self.adaptation_rate) + new_score * self.adaptation_rate
            self.competency_areas[area]['score'] = np.clip(updated_score, 0.0, 1.0)

    def _update_curriculum_progress(self, performance_data: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Update curriculum progress based on performance"""
        
        try:
            # Calculate overall competency level
            overall_competency = self._calculate_overall_competency()
            
            # Update mastery level
            self.learning_stats['mastery_level'] = overall_competency
            
            # Calculate learning rate (improvement rate)
            if len(self.episode_history) >= 2:
                recent_episodes = list(self.episode_history)[-5:]
                if len(recent_episodes) >= 2:
                    recent_performance = np.mean([e.get('pnl', 0) for e in recent_episodes])
                    older_performance = np.mean([e.get('pnl', 0) for e in list(self.episode_history)[:-5]]) if len(self.episode_history) > 5 else 0
                    
                    self.learning_stats['learning_rate'] = (recent_performance - older_performance) / max(1, abs(older_performance))
            
            # Update stage progress
            stage_criteria = self.curriculum_stages[self.current_stage]['success_criteria']
            criteria_met = 0
            
            for criterion, target in stage_criteria.items():
                recent_values = self.performance_metrics.get(criterion, [])
                if recent_values:
                    recent_avg = np.mean(recent_values[-5:])  # Last 5 episodes
                    if criterion == 'max_drawdown':
                        if recent_avg <= target:
                            criteria_met += 1
                    else:
                        if recent_avg >= target:
                            criteria_met += 1
            
            self.stage_progress = criteria_met / len(stage_criteria)
            
        except Exception as e:
            self.log_operator_warning(f"Curriculum progress update failed: {e}")

    def _calculate_overall_competency(self) -> float:
        """Calculate weighted overall competency score"""
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for area, data in self.competency_areas.items():
            score = data['score']
            weight = data['weight']
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _evaluate_stage_progression(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Evaluate if ready to progress to next curriculum stage"""
        
        try:
            # Check if current stage is mastered
            if self.stage_progress >= 0.8 and self.learning_stats['mastery_level'] >= 0.75:
                
                # Additional checks for stability
                recent_episodes = list(self.episode_history)[-5:] if len(self.episode_history) >= 5 else []
                if recent_episodes:
                    recent_success_rate = len([e for e in recent_episodes if self._evaluate_episode_success(e)]) / len(recent_episodes)
                    
                    if recent_success_rate >= 0.6:  # 60% recent success rate
                        self._advance_to_next_stage()
                        return
            
            # Check for regression (automatic stage adjustment)
            if self.stage_progress < 0.3 and self.learning_stats['mastery_level'] < 0.4:
                recent_failures = len([e for e in list(self.episode_history)[-10:] if not self._evaluate_episode_success(e)])
                if recent_failures >= 7:  # 70% failure rate in last 10 episodes
                    self._handle_learning_plateau()
            
        except Exception as e:
            self.log_operator_warning(f"Stage progression evaluation failed: {e}")

    def _advance_to_next_stage(self) -> None:
        """Advance to the next curriculum stage"""
        
        if self.current_stage < len(self.curriculum_stages) - 1:
            old_stage = self.curriculum_stages[self.current_stage]['name']
            
            # Record completion time
            self.learning_stats['stage_completion_times'].append({
                'stage': self.current_stage,
                'episodes': self.learning_stats['total_episodes'],
                'completion_time': datetime.datetime.now().isoformat()
            })
            
            # Advance stage
            self.current_stage += 1
            self.stage_progress = 0.0
            self.learning_stats['adaptation_events'] += 1
            self.learning_stats['current_difficulty'] = self.curriculum_stages[self.current_stage]['difficulty']
            
            new_stage = self.curriculum_stages[self.current_stage]['name']
            
            self.log_operator_info(
                f"🎯 Curriculum stage advanced!",
                from_stage=old_stage,
                to_stage=new_stage,
                total_episodes=self.learning_stats['total_episodes'],
                mastery_level=f"{self.learning_stats['mastery_level']:.1%}"
            )
        else:
            self.log_operator_info("🏆 Maximum curriculum stage reached - Expert level achieved!")

    def _handle_learning_plateau(self) -> None:
        """Handle learning plateau by adjusting curriculum"""
        
        # Option 1: Reduce difficulty temporarily
        if self.learning_stats['current_difficulty'] > 1.0:
            self.learning_stats['current_difficulty'] *= 0.9
            self.learning_stats['adaptation_events'] += 1
            
            self.log_operator_warning(
                "📉 Learning plateau detected - reducing difficulty",
                new_difficulty=f"{self.learning_stats['current_difficulty']:.1f}",
                adaptation_events=self.learning_stats['adaptation_events']
            )
        
        # Option 2: Focus on weakest competency areas
        weakest_area = min(self.competency_areas.items(), key=lambda x: x[1]['score'])
        self.log_operator_info(
            f"🎯 Focusing on weakest area: {weakest_area[0]}",
            current_score=f"{weakest_area[1]['score']:.1%}"
        )

    def get_current_curriculum_constraints(self) -> Dict[str, Any]:
        """Get current curriculum stage constraints for trading system"""
        
        current_stage_data = self.curriculum_stages[self.current_stage]
        
        constraints = {
            'stage_name': current_stage_data['name'],
            'difficulty': self.learning_stats['current_difficulty'],
            'max_position_size': current_stage_data['max_position_size'],
            'max_trades_per_day': current_stage_data['max_trades_per_day'],
            'allowed_market_conditions': current_stage_data['market_conditions'],
            'focus_areas': current_stage_data['focus_areas'],
            'stage_progress': self.stage_progress,
            'mastery_level': self.learning_stats['mastery_level']
        }
        
        return constraints

    def get_observation_components(self) -> np.ndarray:
        """Get curriculum metrics for observation"""
        
        try:
            if not self.episode_history:
                # Return baseline defaults for cold start
                defaults = np.array([
                    0.5,  # stage_progress
                    0.5,  # mastery_level  
                    1.0,  # current_difficulty
                    0.0,  # learning_rate
                    0.5,  # overall_competency
                    0.0   # adaptation_events_normalized
                ], dtype=np.float32)
                self.log_operator_debug("Using default curriculum observations")
                return defaults
            
            # Calculate observation components
            overall_competency = self._calculate_overall_competency()
            
            # Normalize adaptation events
            adaptation_events_norm = min(1.0, self.learning_stats['adaptation_events'] / 10.0)
            
            observation = np.array([
                self.stage_progress,
                self.learning_stats['mastery_level'],
                self.learning_stats['current_difficulty'] / 5.0,  # Normalize by max difficulty
                np.clip(self.learning_stats['learning_rate'], -1.0, 1.0),  # Clip learning rate
                overall_competency,
                adaptation_events_norm
            ], dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.log_operator_error(f"Invalid curriculum observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            self.log_operator_debug(f"Curriculum observation: progress={self.stage_progress:.2f}, mastery={self.learning_stats['mastery_level']:.2f}")
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Curriculum observation generation failed: {e}")
            return np.array([0.5, 0.5, 1.0, 0.0, 0.5, 0.0], dtype=np.float32)

    def _update_info_bus_with_curriculum_data(self, info_bus: InfoBus) -> None:
        """Update InfoBus with curriculum status and recommendations"""
        
        try:
            # Prepare curriculum data
            curriculum_data = {
                'current_stage': self.current_stage,
                'stage_name': self.curriculum_stages[self.current_stage]['name'],
                'stage_progress': self.stage_progress,
                'constraints': self.get_current_curriculum_constraints(),
                'competency_scores': {k: v['score'] for k, v in self.competency_areas.items()},
                'learning_stats': self.learning_stats.copy(),
                'recommendations': self._generate_learning_recommendations()
            }
            
            # Add to InfoBus
            InfoBusUpdater.add_module_data(info_bus, 'curriculum_planner', curriculum_data)
            
            # Add performance alerts
            if self.learning_stats['mastery_level'] < 0.3:
                InfoBusUpdater.add_alert(
                    info_bus,
                    "Low learning performance detected",
                    'curriculum_planner',
                    'warning',
                    {'mastery_level': self.learning_stats['mastery_level']}
                )
            
            if self.stage_progress >= 0.8:
                InfoBusUpdater.add_alert(
                    info_bus,
                    "Ready for curriculum stage advancement",
                    'curriculum_planner',
                    'info',
                    {'stage': self.curriculum_stages[self.current_stage]['name']}
                )
            
        except Exception as e:
            self.log_operator_warning(f"InfoBus curriculum update failed: {e}")

    def _generate_learning_recommendations(self) -> List[str]:
        """Generate actionable learning recommendations"""
        
        recommendations = []
        
        # Stage-specific recommendations
        current_stage = self.curriculum_stages[self.current_stage]
        focus_areas = current_stage['focus_areas']
        
        if 'risk_management' in focus_areas and self.competency_areas['risk_management']['score'] < 0.6:
            recommendations.append("Focus on reducing drawdown - consider smaller position sizes")
        
        if 'entry_timing' in focus_areas and self.competency_areas['entry_timing']['score'] < 0.6:
            recommendations.append("Improve entry timing - wait for better confirmation signals")
        
        if 'exit_strategy' in focus_areas and self.competency_areas['exit_strategy']['score'] < 0.6:
            recommendations.append("Enhance exit strategy - review take profit and stop loss levels")
        
        if 'position_sizing' in focus_areas and self.competency_areas['position_sizing']['score'] < 0.6:
            recommendations.append("Optimize position sizing - ensure consistent risk per trade")
        
        # Learning rate recommendations
        if self.learning_stats['learning_rate'] < -0.1:
            recommendations.append("Performance declining - consider reverting to previous successful strategies")
        elif self.learning_stats['learning_rate'] > 0.1:
            recommendations.append("Good learning progress - continue current approach")
        
        # General recommendations
        if self.stage_progress < 0.3:
            recommendations.append("Focus on mastering current stage requirements before advancing")
        
        if not recommendations:
            recommendations.append("Continue current learning approach - performance is stable")
        
        return recommendations
    
    def log_operator_debug(self, message: str, **kwargs):
            """Log debug message with proper formatting"""
            if hasattr(self, 'debug') and self.debug and hasattr(self, 'logger'):
                # Convert kwargs to details string
                details = ""
                if kwargs:
                    detail_parts = []
                    for key, value in kwargs.items():
                        if isinstance(value, float):
                            if 'pct' in key or 'rate' in key or '%' in str(value):
                                detail_parts.append(f"{key}={value:.1%}")
                            else:
                                detail_parts.append(f"{key}={value:.3f}")
                        else:
                            detail_parts.append(f"{key}={value}")
                    details = " | ".join(detail_parts)
                
                # Only log if we have a logger configured for debug
                if hasattr(self.logger, 'debug'):
                    from modules.utils.audit_utils import format_operator_message
                    formatted_message = format_operator_message(
                        emoji="🔧",
                        action=message,
                        details=details
                    )
                    self.logger.debug(formatted_message)

    def get_curriculum_report(self) -> str:
        """Generate operator-friendly curriculum progress report"""
        
        current_stage = self.curriculum_stages[self.current_stage]
        overall_competency = self._calculate_overall_competency()
        
        # Competency breakdown
        competency_summary = ""
        for area, data in self.competency_areas.items():
            score = data['score']
            target = data['target']
            status = "✅" if score >= target else "🔄" if score >= target * 0.8 else "❌"
            competency_summary += f"  • {area.replace('_', ' ').title()}: {score:.1%} {status}\n"
        
        # Recent performance trend
        if len(self.episode_history) >= 3:
            recent_pnls = [e.get('pnl', 0) for e in list(self.episode_history)[-3:]]
            trend = "📈 Improving" if recent_pnls[-1] > recent_pnls[0] else "📉 Declining" if recent_pnls[-1] < recent_pnls[0] else "➡️ Stable"
        else:
            trend = "📊 Insufficient data"
        
        return f"""
📚 CURRICULUM PLANNER REPORT
═══════════════════════════════════════
🎯 Current Stage: {current_stage['name']} (Level {self.current_stage + 1}/{len(self.curriculum_stages)})
📊 Stage Progress: {self.stage_progress:.1%}
🧠 Overall Mastery: {overall_competency:.1%}
📈 Learning Rate: {self.learning_stats['learning_rate']:+.1%}

⚙️ Current Constraints:
• Max Position Size: {current_stage['max_position_size']}x
• Max Daily Trades: {current_stage['max_trades_per_day']}
• Allowed Markets: {', '.join(current_stage['market_conditions'])}
• Focus Areas: {', '.join(current_stage['focus_areas'])}

📋 Competency Scores:
{competency_summary}

📊 Learning Statistics:
• Total Episodes: {self.learning_stats['total_episodes']}
• Success Rate: {(self.learning_stats['successful_episodes'] / max(1, self.learning_stats['total_episodes'])):.1%}
• Adaptation Events: {self.learning_stats['adaptation_events']}
• Performance Trend: {trend}

🎯 Next Steps:
{chr(10).join([f'  • {rec}' for rec in self._generate_learning_recommendations()])}
        """

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "window": self.window,
                "debug": self.debug,
                "adaptation_rate": self.adaptation_rate,
                "performance_threshold": self.performance_threshold,
                "difficulty_levels": self.difficulty_levels
            },
            "curriculum_state": {
                "current_stage": self.current_stage,
                "stage_progress": self.stage_progress,
                "episode_history": list(self.episode_history),
                "performance_metrics": {k: list(v) for k, v in self.performance_metrics.items()},
                "learning_stats": self.learning_stats.copy(),
                "competency_areas": self.competency_areas.copy()
            },
            "curriculum_stages": self.curriculum_stages.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.window = int(config.get("window", self.window))
        self.debug = bool(config.get("debug", self.debug))
        self.adaptation_rate = float(config.get("adaptation_rate", self.adaptation_rate))
        self.performance_threshold = float(config.get("performance_threshold", self.performance_threshold))
        self.difficulty_levels = int(config.get("difficulty_levels", self.difficulty_levels))
        
        # Load curriculum state
        curriculum_state = state.get("curriculum_state", {})
        self.current_stage = int(curriculum_state.get("current_stage", 0))
        self.stage_progress = float(curriculum_state.get("stage_progress", 0.0))
        
        # Restore history
        self.episode_history = deque(curriculum_state.get("episode_history", []), maxlen=self.window)
        
        # Restore performance metrics
        performance_metrics = curriculum_state.get("performance_metrics", {})
        self.performance_metrics = defaultdict(list)
        for k, v in performance_metrics.items():
            self.performance_metrics[k] = list(v)
        
        # Restore other state
        self.learning_stats = curriculum_state.get("learning_stats", self.learning_stats)
        self.competency_areas = curriculum_state.get("competency_areas", self.competency_areas)
        
        # Load curriculum stages if provided
        if "curriculum_stages" in state:
            self.curriculum_stages = state["curriculum_stages"]