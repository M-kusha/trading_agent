# ─────────────────────────────────────────────────────────────
# File: modules/strategy/explanation_generator.py
# Enhanced with InfoBus integration & intelligent explanation generation
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class ExplanationGenerator(Module, AnalysisMixin, StateManagementMixin, TradingMixin):
    """
    Enhanced explanation generator with InfoBus integration.
    Generates comprehensive, human-readable explanations of trading decisions and system state.
    Provides actionable insights for operators and adaptive explanations based on context.
    """

    def __init__(
        self,
        debug: bool = False,
        explanation_depth: str = "detailed",  # "brief", "detailed", "comprehensive"
        update_frequency: int = 1,  # How often to generate explanations
        target_daily_profit: float = 150.0,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 50),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.debug = bool(debug)
        self.explanation_depth = explanation_depth
        self.update_frequency = int(update_frequency)
        self.target_daily_profit = float(target_daily_profit)
        
        # Explanation state
        self.last_explanation = ""
        self.explanation_history = deque(maxlen=100)
        self.explanation_templates = self._initialize_explanation_templates()
        self.context_priorities = self._initialize_context_priorities()
        
        # Performance tracking
        self.session_metrics = {
            'trade_count': 0,
            'profit_today': 0.0,
            'session_start': datetime.datetime.now(),
            'last_update': datetime.datetime.now(),
            'explanations_generated': 0,
            'avg_explanation_relevance': 0.0
        }
        
        # Decision context tracking
        self.decision_context = {
            'current_strategy': 'unknown',
            'market_regime': 'unknown',
            'risk_level': 'medium',
            'confidence_level': 0.5,
            'recent_performance': 'neutral',
            'system_alerts': []
        }
        
        # Explanation categories
        self.explanation_categories = {
            'trade_decision': 'Trade entry/exit explanations',
            'risk_management': 'Risk and position sizing explanations',
            'market_analysis': 'Market condition and regime explanations',
            'performance_update': 'Performance and progress explanations',
            'system_status': 'System health and alert explanations',
            'learning_insights': 'Learning and adaptation explanations'
        }
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "ExplanationGenerator",
            "logs/strategy/explanation_generator.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("ExplanationGenerator")
        
        self.log_operator_info(
            "🎤 Explanation Generator initialized",
            depth=self.explanation_depth,
            target_profit=f"€{self.target_daily_profit}",
            update_frequency=self.update_frequency
        )

    def _initialize_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize explanation templates for different contexts"""
        
        return {
            'trade_entry': {
                'brief': "Entering {instrument} {direction} - {reason}",
                'detailed': "📈 Trade Entry: {instrument} {direction} | Reason: {reason} | Size: {size} | Risk: {risk_level}",
                'comprehensive': "🎯 TRADE ENTRY ANALYSIS\n• Instrument: {instrument}\n• Direction: {direction}\n• Entry Reason: {reason}\n• Position Size: {size}\n• Risk Assessment: {risk_level}\n• Market Context: {market_context}\n• Expected Outcome: {expected_outcome}"
            },
            'trade_exit': {
                'brief': "Closed {instrument} - {exit_reason} | P&L: {pnl}",
                'detailed': "📊 Trade Closed: {instrument} | Reason: {exit_reason} | P&L: {pnl} | Duration: {duration}",
                'comprehensive': "🏁 TRADE EXIT SUMMARY\n• Instrument: {instrument}\n• Exit Reason: {exit_reason}\n• P&L: {pnl}\n• Duration: {duration}\n• Performance vs Target: {vs_target}\n• Lessons Learned: {lessons}"
            },
            'performance_update': {
                'brief': "Session: {profit_pct}% to target | {trade_count} trades",
                'detailed': "📊 Progress: €{profit_today}/€{target} ({profit_pct}%) | Trades: {trade_count} | {performance_trend}",
                'comprehensive': "📈 SESSION PERFORMANCE UPDATE\n• Current Profit: €{profit_today}\n• Target Progress: {profit_pct}%\n• Trade Count: {trade_count}\n• Win Rate: {win_rate}%\n• Avg Trade: €{avg_trade}\n• Performance Trend: {performance_trend}\n• Time Remaining: {time_remaining}"
            },
            'risk_alert': {
                'brief': "⚠️ Risk Alert: {alert_type}",
                'detailed': "⚠️ Risk Management: {alert_type} | Current Exposure: {exposure} | Action: {action}",
                'comprehensive': "🚨 RISK MANAGEMENT ALERT\n• Alert Type: {alert_type}\n• Current Exposure: {exposure}\n• Risk Metrics: {risk_metrics}\n• Recommended Action: {action}\n• Timeline: {timeline}\n• Recovery Plan: {recovery_plan}"
            },
            'market_insight': {
                'brief': "Market: {regime} regime | Volatility: {volatility}",
                'detailed': "🌊 Market State: {regime} regime | Vol: {volatility} | Opportunities: {opportunities}",
                'comprehensive': "🌍 MARKET ANALYSIS UPDATE\n• Regime: {regime}\n• Volatility Level: {volatility}\n• Key Drivers: {drivers}\n• Trading Opportunities: {opportunities}\n• Risk Factors: {risk_factors}\n• Strategy Recommendations: {recommendations}"
            }
        }

    def _initialize_context_priorities(self) -> Dict[str, int]:
        """Initialize context priority levels for explanation focus"""
        
        return {
            'high_profit_trade': 10,
            'high_loss_trade': 9,
            'risk_alert': 8,
            'target_achieved': 8,
            'system_error': 7,
            'regime_change': 6,
            'performance_milestone': 5,
            'routine_update': 3,
            'background_info': 1
        }

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Clear explanation state
        self.last_explanation = ""
        self.explanation_history.clear()
        
        # Reset session metrics
        self.session_metrics = {
            'trade_count': 0,
            'profit_today': 0.0,
            'session_start': datetime.datetime.now(),
            'last_update': datetime.datetime.now(),
            'explanations_generated': 0,
            'avg_explanation_relevance': 0.0
        }
        
        # Reset decision context
        self.decision_context = {
            'current_strategy': 'unknown',
            'market_regime': 'unknown',
            'risk_level': 'medium',
            'confidence_level': 0.5,
            'recent_performance': 'neutral',
            'system_alerts': []
        }
        
        self.log_operator_info("🔄 Explanation Generator reset - new session started")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and adaptive explanation generation"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - limited explanation capability")
            return
        
        # Extract comprehensive context
        context = extract_standard_context(info_bus)
        explanation_context = self._extract_explanation_context_from_info_bus(info_bus, context)
        
        # Update internal state
        self._update_internal_state(explanation_context, context)
        
        # Generate explanation if needed
        if self._should_generate_explanation(explanation_context):
            explanation = self._generate_adaptive_explanation(explanation_context, context)
            if explanation:
                self._record_explanation(explanation, explanation_context)
        
        # Update InfoBus with explanation data
        self._update_info_bus_with_explanation_data(info_bus)

    def _extract_explanation_context_from_info_bus(self, info_bus: InfoBus, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive context for explanation generation"""
        
        try:
            # Get trading data
            recent_trades = info_bus.get('recent_trades', [])
            positions = InfoBusExtractor.get_positions(info_bus)
            risk_data = info_bus.get('risk', {})
            alerts = info_bus.get('alerts', [])
            
            # Get module data for insights
            module_data = info_bus.get('module_data', {})
            
            explanation_context = {
                'timestamp': datetime.datetime.now().isoformat(),
                'session_pnl': context.get('session_pnl', 0),
                'balance': risk_data.get('balance', 0),
                'equity': risk_data.get('equity', 0),
                'drawdown': risk_data.get('current_drawdown', 0),
                'positions': positions,
                'recent_trades': recent_trades,
                'alerts': alerts,
                'market_regime': context.get('regime', 'unknown'),
                'volatility_level': context.get('volatility_level', 'medium'),
                'market_open': context.get('market_open', True),
                'module_insights': self._extract_module_insights(module_data),
                'priority_context': self._determine_priority_context(recent_trades, alerts, risk_data)
            }
            
            return explanation_context
            
        except Exception as e:
            self.log_operator_warning(f"Explanation context extraction failed: {e}")
            return {'timestamp': datetime.datetime.now().isoformat()}

    def _extract_module_insights(self, module_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from other system modules"""
        
        insights = {}
        
        try:
            # Bias auditor insights
            if 'bias_auditor' in module_data:
                bias_data = module_data['bias_auditor']
                insights['bias_status'] = {
                    'active_biases': bias_data.get('active_biases', {}),
                    'total_biases': bias_data.get('total_biases_tracked', 0)
                }
            
            # Risk controller insights
            if 'risk_controller' in module_data:
                risk_data = module_data['risk_controller']
                insights['risk_status'] = {
                    'risk_level': risk_data.get('current_risk_level', 'medium'),
                    'position_limits': risk_data.get('position_limits', {})
                }
            
            # Strategy insights
            if 'strategy_arbiter' in module_data:
                strategy_data = module_data['strategy_arbiter']
                insights['strategy_status'] = {
                    'active_strategy': strategy_data.get('top_strategy', 'unknown'),
                    'confidence': strategy_data.get('confidence', 0.5)
                }
            
            # Curriculum insights
            if 'curriculum_planner' in module_data:
                curriculum_data = module_data['curriculum_planner']
                insights['learning_status'] = {
                    'stage': curriculum_data.get('stage_name', 'unknown'),
                    'progress': curriculum_data.get('stage_progress', 0)
                }
            
        except Exception as e:
            self.log_operator_warning(f"Module insights extraction failed: {e}")
        
        return insights

    def _determine_priority_context(self, recent_trades: List[Dict], alerts: List[Dict], risk_data: Dict) -> str:
        """Determine the highest priority context for explanation focus"""
        
        # Check for critical situations first
        if risk_data.get('current_drawdown', 0) > 0.1:  # 10% drawdown
            return 'high_drawdown'
        
        if alerts:
            critical_alerts = [a for a in alerts if a.get('level') in ['critical', 'error']]
            if critical_alerts:
                return 'critical_alert'
        
        # Check recent trade performance
        if recent_trades:
            last_trade = recent_trades[-1]
            pnl = last_trade.get('pnl', 0)
            
            if pnl > 100:  # High profit trade
                return 'high_profit_trade'
            elif pnl < -50:  # High loss trade
                return 'high_loss_trade'
        
        # Check for target achievement
        session_pnl = sum(t.get('pnl', 0) for t in recent_trades)
        if session_pnl >= self.target_daily_profit:
            return 'target_achieved'
        
        # Default to routine update
        return 'routine_update'

    def _should_generate_explanation(self, explanation_context: Dict[str, Any]) -> bool:
        """Determine if an explanation should be generated"""
        
        try:
            # Always generate for high priority contexts
            priority_context = explanation_context.get('priority_context', 'routine_update')
            if priority_context in ['critical_alert', 'high_profit_trade', 'high_loss_trade', 'target_achieved']:
                return True
            
            # Generate based on frequency for routine updates
            time_since_last = datetime.datetime.now() - self.session_metrics['last_update']
            if time_since_last.total_seconds() >= (self.update_frequency * 60):  # Convert to seconds
                return True
            
            # Generate for significant trade count changes
            current_trade_count = len(explanation_context.get('recent_trades', []))
            if current_trade_count != self.session_metrics['trade_count']:
                return True
            
            return False
            
        except Exception as e:
            self.log_operator_warning(f"Explanation generation check failed: {e}")
            return False

    def _update_internal_state(self, explanation_context: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Update internal state based on new context"""
        
        try:
            # Update session metrics
            recent_trades = explanation_context.get('recent_trades', [])
            self.session_metrics['trade_count'] = len(recent_trades)
            self.session_metrics['profit_today'] = explanation_context.get('session_pnl', 0)
            self.session_metrics['last_update'] = datetime.datetime.now()
            
            # Update decision context
            self.decision_context.update({
                'market_regime': explanation_context.get('market_regime', 'unknown'),
                'risk_level': self._assess_current_risk_level(explanation_context),
                'recent_performance': self._assess_recent_performance(recent_trades),
                'system_alerts': explanation_context.get('alerts', [])
            })
            
            # Update from module insights
            module_insights = explanation_context.get('module_insights', {})
            if 'strategy_status' in module_insights:
                self.decision_context['current_strategy'] = module_insights['strategy_status'].get('active_strategy', 'unknown')
                self.decision_context['confidence_level'] = module_insights['strategy_status'].get('confidence', 0.5)
            
        except Exception as e:
            self.log_operator_warning(f"Internal state update failed: {e}")

    def _assess_current_risk_level(self, explanation_context: Dict[str, Any]) -> str:
        """Assess current risk level based on context"""
        
        try:
            drawdown = explanation_context.get('drawdown', 0)
            position_count = len(explanation_context.get('positions', []))
            alerts = explanation_context.get('alerts', [])
            
            # Critical risk factors
            if drawdown > 0.15 or any(a.get('level') == 'critical' for a in alerts):
                return 'critical'
            
            # High risk factors
            if drawdown > 0.08 or position_count > 5:
                return 'high'
            
            # Medium risk factors
            if drawdown > 0.03 or position_count > 2:
                return 'medium'
            
            return 'low'
            
        except Exception:
            return 'medium'

    def _assess_recent_performance(self, recent_trades: List[Dict]) -> str:
        """Assess recent performance trend"""
        
        try:
            if not recent_trades or len(recent_trades) < 3:
                return 'neutral'
            
            # Look at last 5 trades
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

    def _generate_adaptive_explanation(self, explanation_context: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate adaptive explanation based on context and priority"""
        
        try:
            priority_context = explanation_context.get('priority_context', 'routine_update')
            
            # Generate explanation based on priority context
            if priority_context == 'high_profit_trade':
                explanation = self._generate_trade_explanation(explanation_context, 'profit')
            elif priority_context == 'high_loss_trade':
                explanation = self._generate_trade_explanation(explanation_context, 'loss')
            elif priority_context == 'target_achieved':
                explanation = self._generate_target_achievement_explanation(explanation_context)
            elif priority_context == 'critical_alert':
                explanation = self._generate_alert_explanation(explanation_context)
            elif priority_context == 'high_drawdown':
                explanation = self._generate_risk_explanation(explanation_context)
            else:
                explanation = self._generate_performance_update_explanation(explanation_context)
            
            # Add contextual insights
            enhanced_explanation = self._enhance_explanation_with_insights(explanation, explanation_context)
            
            return enhanced_explanation
            
        except Exception as e:
            self.log_operator_error(f"Explanation generation failed: {e}")
            return self._generate_fallback_explanation(explanation_context)

    def _generate_trade_explanation(self, explanation_context: Dict[str, Any], trade_type: str) -> str:
        """Generate explanation for significant trades"""
        
        try:
            recent_trades = explanation_context.get('recent_trades', [])
            if not recent_trades:
                return "No recent trade data available"
            
            last_trade = recent_trades[-1]
            pnl = last_trade.get('pnl', 0)
            instrument = last_trade.get('symbol', 'Unknown')
            size = last_trade.get('size', 0)
            duration = last_trade.get('duration', 0)
            
            if trade_type == 'profit':
                template = self.explanation_templates['trade_exit'][self.explanation_depth]
                explanation = template.format(
                    instrument=instrument,
                    exit_reason="Target achieved",
                    pnl=f"€{pnl:+.2f}",
                    duration=f"{duration} steps",
                    vs_target=f"{(pnl/self.target_daily_profit)*100:.1f}% of daily target",
                    lessons="Strategy working well in current conditions"
                )
            else:  # loss
                template = self.explanation_templates['trade_exit'][self.explanation_depth]
                explanation = template.format(
                    instrument=instrument,
                    exit_reason="Stop loss triggered",
                    pnl=f"€{pnl:+.2f}",
                    duration=f"{duration} steps",
                    vs_target="Risk management protected capital",
                    lessons="Review entry criteria and market conditions"
                )
            
            return explanation
            
        except Exception as e:
            return f"Trade explanation error: {e}"

    def _generate_performance_update_explanation(self, explanation_context: Dict[str, Any]) -> str:
        """Generate performance update explanation"""
        
        try:
            profit_today = explanation_context.get('session_pnl', 0)
            trade_count = len(explanation_context.get('recent_trades', []))
            
            profit_pct = (profit_today / self.target_daily_profit) * 100
            
            # Calculate additional metrics
            recent_trades = explanation_context.get('recent_trades', [])
            if recent_trades:
                pnls = [t.get('pnl', 0) for t in recent_trades]
                win_rate = (len([p for p in pnls if p > 0]) / len(pnls)) * 100
                avg_trade = np.mean(pnls)
            else:
                win_rate = 0
                avg_trade = 0
            
            performance_trend = self.decision_context.get('recent_performance', 'neutral').title()
            
            # Calculate time remaining
            session_duration = datetime.datetime.now() - self.session_metrics['session_start']
            hours_elapsed = session_duration.total_seconds() / 3600
            estimated_session_length = 8  # 8-hour trading session
            time_remaining = max(0, estimated_session_length - hours_elapsed)
            
            template = self.explanation_templates['performance_update'][self.explanation_depth]
            explanation = template.format(
                profit_today=profit_today,
                target=self.target_daily_profit,
                profit_pct=profit_pct,
                trade_count=trade_count,
                win_rate=win_rate,
                avg_trade=avg_trade,
                performance_trend=performance_trend,
                time_remaining=f"{time_remaining:.1f}h"
            )
            
            return explanation
            
        except Exception as e:
            return f"Performance update error: {e}"

    def _generate_target_achievement_explanation(self, explanation_context: Dict[str, Any]) -> str:
        """Generate explanation for target achievement"""
        
        profit_today = explanation_context.get('session_pnl', 0)
        trade_count = len(explanation_context.get('recent_trades', []))
        
        session_duration = datetime.datetime.now() - self.session_metrics['session_start']
        hours_taken = session_duration.total_seconds() / 3600
        
        return f"""
🎯 TARGET ACHIEVED! 🎯
Daily profit target of €{self.target_daily_profit} reached!
• Final Profit: €{profit_today:.2f} ({((profit_today/self.target_daily_profit)-1)*100:+.1f}% over target)
• Trades Executed: {trade_count}
• Time Taken: {hours_taken:.1f} hours
• Performance: Excellent execution!

RECOMMENDATION: Consider reducing position sizes for remainder of session to preserve gains.
        """.strip()

    def _generate_alert_explanation(self, explanation_context: Dict[str, Any]) -> str:
        """Generate explanation for system alerts"""
        
        alerts = explanation_context.get('alerts', [])
        if not alerts:
            return "No active alerts"
        
        critical_alerts = [a for a in alerts if a.get('level') in ['critical', 'error']]
        warning_alerts = [a for a in alerts if a.get('level') == 'warning']
        
        explanation = "🚨 SYSTEM ALERTS:\n"
        
        if critical_alerts:
            explanation += f"• CRITICAL: {len(critical_alerts)} alerts requiring immediate attention\n"
        
        if warning_alerts:
            explanation += f"• WARNINGS: {len(warning_alerts)} advisory alerts\n"
        
        # Add most recent alert details
        latest_alert = alerts[-1]
        explanation += f"\nLatest Alert: {latest_alert.get('message', 'Unknown alert')}"
        
        return explanation

    def _generate_risk_explanation(self, explanation_context: Dict[str, Any]) -> str:
        """Generate explanation for risk situations"""
        
        drawdown = explanation_context.get('drawdown', 0)
        positions = explanation_context.get('positions', [])
        
        return f"""
⚠️ RISK MANAGEMENT ALERT
Current drawdown: {drawdown:.1%}
Active positions: {len(positions)}
Risk level: {self.decision_context['risk_level'].upper()}

RECOMMENDED ACTIONS:
• Reduce position sizes
• Review open positions
• Consider temporary trading halt
• Wait for market stability
        """.strip()

    def _enhance_explanation_with_insights(self, base_explanation: str, explanation_context: Dict[str, Any]) -> str:
        """Enhance explanation with module insights and contextual information"""
        
        try:
            enhanced = base_explanation
            module_insights = explanation_context.get('module_insights', {})
            
            # Add bias insights
            if 'bias_status' in module_insights:
                bias_data = module_insights['bias_status']
                active_biases = bias_data.get('active_biases', {})
                significant_biases = {k: v for k, v in active_biases.items() if v > 0.3}
                
                if significant_biases:
                    bias_warning = f"\n🧠 Bias Alert: {', '.join(significant_biases.keys())} detected"
                    enhanced += bias_warning
            
            # Add learning insights
            if 'learning_status' in module_insights:
                learning_data = module_insights['learning_status']
                stage = learning_data.get('stage', 'unknown')
                progress = learning_data.get('progress', 0)
                
                enhanced += f"\n📚 Learning: {stage} stage ({progress:.0%} complete)"
            
            # Add market context
            regime = explanation_context.get('market_regime', 'unknown')
            volatility = explanation_context.get('volatility_level', 'medium')
            enhanced += f"\n🌊 Market: {regime.title()} regime, {volatility} volatility"
            
            return enhanced
            
        except Exception as e:
            self.log_operator_warning(f"Explanation enhancement failed: {e}")
            return base_explanation

    def _generate_fallback_explanation(self, explanation_context: Dict[str, Any]) -> str:
        """Generate simple fallback explanation when main generation fails"""
        
        profit = explanation_context.get('session_pnl', 0)
        trades = len(explanation_context.get('recent_trades', []))
        
        return f"Session Update: €{profit:.2f} P&L, {trades} trades executed"

    def _record_explanation(self, explanation: str, context: Dict[str, Any]) -> None:
        """Record generated explanation with context"""
        
        try:
            explanation_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'explanation': explanation,
                'context': context.get('priority_context', 'unknown'),
                'depth': self.explanation_depth,
                'trade_count': self.session_metrics['trade_count'],
                'session_pnl': self.session_metrics['profit_today']
            }
            
            self.explanation_history.append(explanation_record)
            self.last_explanation = explanation
            self.session_metrics['explanations_generated'] += 1
            
            # Log explanation generation
            self.log_operator_info(
                "🎤 Explanation generated",
                context=context.get('priority_context', 'unknown'),
                depth=self.explanation_depth,
                length=len(explanation)
            )
            
            # Debug output if enabled
            if self.debug:
                print(f"[ExplanationGenerator] {explanation}")
            
        except Exception as e:
            self.log_operator_error(f"Explanation recording failed: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Return explanation metrics for observation"""
        
        try:
            # Current session metrics
            progress_ratio = self.session_metrics['profit_today'] / max(1, self.target_daily_profit)
            trade_frequency = self.session_metrics['trade_count'] / max(1, 
                (datetime.datetime.now() - self.session_metrics['session_start']).total_seconds() / 3600)
            
            # Risk assessment
            risk_score = {
                'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0
            }.get(self.decision_context['risk_level'], 0.5)
            
            # Performance assessment
            performance_score = {
                'poor': 0.1, 'neutral': 0.5, 'good': 0.7, 'excellent': 0.9
            }.get(self.decision_context['recent_performance'], 0.5)
            
            observation = np.array([
                np.clip(progress_ratio, -2.0, 3.0),  # Progress toward target
                float(self.session_metrics['trade_count']),  # Trade count
                np.clip(trade_frequency, 0, 10),  # Trade frequency per hour
                risk_score,  # Current risk level
                performance_score,  # Recent performance
                self.decision_context['confidence_level'],  # System confidence
                float(len(self.decision_context['system_alerts'])),  # Alert count
                float(self.session_metrics['explanations_generated']) / 100  # Explanation frequency
            ], dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.log_operator_error(f"Invalid explanation observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Explanation observation generation failed: {e}")
            return np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)

    def _update_info_bus_with_explanation_data(self, info_bus: InfoBus) -> None:
        """Update InfoBus with explanation generator status"""
        
        try:
            # Prepare explanation data
            explanation_data = {
                'last_explanation': self.last_explanation,
                'session_metrics': self.session_metrics.copy(),
                'decision_context': self.decision_context.copy(),
                'explanation_depth': self.explanation_depth,
                'explanations_generated': self.session_metrics['explanations_generated'],
                'available_depths': ['brief', 'detailed', 'comprehensive'],
                'categories': list(self.explanation_categories.keys())
            }
            
            # Add to InfoBus
            InfoBusUpdater.add_module_data(info_bus, 'explanation_generator', explanation_data)
            
        except Exception as e:
            self.log_operator_warning(f"InfoBus explanation update failed: {e}")

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
        
        return f"""
🎤 EXPLANATION GENERATOR REPORT
═══════════════════════════════════════
📊 Session Overview:
• Session Duration: {hours_active:.1f} hours
• Explanations Generated: {self.session_metrics['explanations_generated']}
• Current Depth: {self.explanation_depth.title()}
• Update Frequency: Every {self.update_frequency} minute(s)

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

📝 Recent Explanations:
{recent_explanations if recent_explanations else '  📭 No recent explanations'}

🎯 Current Focus:
• Explanation Categories: {len(self.explanation_categories)} available
• Context Priorities: Adaptive based on market conditions
• System Alerts: {len(self.decision_context['system_alerts'])} active

📈 Last Explanation:
{self.last_explanation[:200]}{'...' if len(self.last_explanation) > 200 else ''}
        """

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "debug": self.debug,
                "explanation_depth": self.explanation_depth,
                "update_frequency": self.update_frequency,
                "target_daily_profit": self.target_daily_profit
            },
            "session_state": {
                "last_explanation": self.last_explanation,
                "session_metrics": self.session_metrics.copy(),
                "decision_context": self.decision_context.copy(),
                "explanation_history": list(self.explanation_history)[-20:]  # Keep recent only
            },
            "templates": self.explanation_templates.copy(),
            "priorities": self.context_priorities.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.debug = bool(config.get("debug", self.debug))
        self.explanation_depth = config.get("explanation_depth", self.explanation_depth)
        self.update_frequency = int(config.get("update_frequency", self.update_frequency))
        self.target_daily_profit = float(config.get("target_daily_profit", self.target_daily_profit))
        
        # Load session state
        session_state = state.get("session_state", {})
        self.last_explanation = session_state.get("last_explanation", "")
        self.session_metrics.update(session_state.get("session_metrics", {}))
        self.decision_context.update(session_state.get("decision_context", {}))
        
        # Restore explanation history
        history_data = session_state.get("explanation_history", [])
        self.explanation_history = deque(history_data, maxlen=100)
        
        # Load templates and priorities if provided
        self.explanation_templates.update(state.get("templates", {}))
        self.context_priorities.update(state.get("priorities", {}))